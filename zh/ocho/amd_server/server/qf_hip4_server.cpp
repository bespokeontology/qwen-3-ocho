// qf_hip4_server.cpp - persistent HTTP/SSE inference server for the 4x AMD
// gfx906 (MI50) QfHip4 pipeline engine running
// RadixArk-Qwen3.8-Flash-Next-NVFP4.
//
// Zero external dependencies: hand-rolled HTTP/1.1 + SSE over POSIX sockets,
// C++17, pthreads. JSON via the in-tree minimal parser (json.h). The engine
// is serial (one request in flight across the whole 4-GPU pipeline); a
// global timed mutex serializes generation and a small worker pool handles
// connections. There is no CPU inference fallback: if qf_hip4_init fails the
// process exits nonzero.
//
// Engine flow per request (frozen QfHip4 API):
//   qf_hip4_reset
//   submit_prefill(tok[i], pos=i) for i in 0..n-2
//   decode_step(tok[n-1], pos=n-1, logits)  -> logits for first generated token
//   loop: sample; then submit(t, pos) + wait(pos, logits) for the next
//
#include "qf_hip4.h"
#include "json.h"
#include "sampler.h"
#include "tokenizer_json.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

// ------------------------------------------------------------------ config

constexpr int    kEosId         = 248044;   // RadixArk-Qwen3.8-Flash-Next EOS
constexpr long   kModelContext  = 262144;   // hard clamp from the model card
constexpr size_t kMaxRequest    = 1u << 20; // 1 MiB request cap
constexpr int    kWorkers       = 8;
constexpr int    kEngineWaitS   = 120;      // 503 "engine busy" beyond this
constexpr int    kReadTimeoutMs = 30000;

struct Config {
    std::string model_dir;
    std::string tokenizer = "auto";
    std::string host = "127.0.0.1";
    int  port = 8088;
    long max_context = 131072;
};

Config       g_cfg;
QfHip4      *g_engine = nullptr;
qf::QfTokenizer *g_tok = nullptr;
std::timed_mutex g_engine_mu;                       // serializes generation
std::atomic<bool> g_stop{false};
int           g_listen_fd = -1;
std::chrono::steady_clock::time_point g_start;
std::atomic<uint64_t> g_req_id{0};

// ------------------------------------------------------------------ signals

void on_signal(int) {
    g_stop.store(true, std::memory_order_relaxed);
    if (g_listen_fd >= 0) ::close(g_listen_fd);  // close() is async-signal-safe
    g_listen_fd = -1;
}

// ------------------------------------------------------------------ sockets

bool write_all(int fd, const char *data, size_t len, std::atomic<bool> &dead) {
    if (dead.load(std::memory_order_relaxed)) return false;
    size_t off = 0;
    while (off < len) {
        ssize_t w = ::send(fd, data + off, len - off, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            dead.store(true);
            return false;
        }
        if (w == 0) { dead.store(true); return false; }
        off += (size_t)w;
    }
    return true;
}

const char *status_text(int s) {
    switch (s) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

struct Conn {
    int fd = -1;
    std::atomic<bool> dead{false};
    bool headers_sent = false;

    void send_json(int status, const std::string &body) {
        char head[256];
        int n = snprintf(head, sizeof(head),
                         "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
                         "Connection: close\r\nContent-Length: %zu\r\n\r\n",
                         status, status_text(status), body.size());
        write_all(fd, head, (size_t)n, dead);
        write_all(fd, body.data(), body.size(), dead);
        dead.store(true);
        headers_sent = true;
    }
    void send_error(int status, const char *msg) {
        qf::Json o = qf::Json::make_obj();
        o.set("error", qf::Json::make_str(msg));
        send_json(status, qf::json_dump(o));
    }
    void begin_sse() {
        const char *head =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        write_all(fd, head, strlen(head), dead);
        headers_sent = true;
    }
    bool write_sse(const std::string &payload) {
        std::string frame = "data: " + payload + "\n\n";
        return write_all(fd, frame.data(), frame.size(), dead);
    }
};

// ------------------------------------------------------------------ HTTP in

struct HttpRequest {
    std::string method, path, body;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string header(const char *name) const {
        for (auto &h : headers)
            if (h.first == name) return h.second;
        return "";
    }
};

// Read one request (headers + Content-Length body). Returns false on
// close/timeout/oversize; on oversize `too_large` is set so the caller can
// answer 413.
bool read_request(int fd, HttpRequest &req, bool &too_large) {
    too_large = false;
    std::string buf;
    char tmp[16384];
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kReadTimeoutMs);
    while (buf.find("\r\n\r\n") == std::string::npos) {
        if (buf.size() > kMaxRequest) { too_large = true; return false; }
        auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now()).count();
        if (remain <= 0) return false;
        struct pollfd pfd{fd, POLLIN, 0};
        if (::poll(&pfd, 1, (int)remain) <= 0) return false;
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n == 0) return false;
        if (n < 0) { if (errno == EINTR) continue; return false; }
        buf.append(tmp, (size_t)n);
    }
    size_t head_end = buf.find("\r\n\r\n");
    std::string head = buf.substr(0, head_end);
    req.body = buf.substr(head_end + 4);

    size_t eol = head.find("\r\n");
    std::string rline = head.substr(0, eol);
    size_t sp1 = rline.find(' ');
    size_t sp2 = sp1 == std::string::npos ? sp1 : rline.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = rline.substr(0, sp1);
    std::string target = rline.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t qm = target.find('?');
    req.path = qm == std::string::npos ? target : target.substr(0, qm);

    size_t hpos = eol == std::string::npos ? head.size() : eol + 2;
    while (hpos < head.size()) {
        size_t he = head.find("\r\n", hpos);
        std::string line = head.substr(hpos, he == std::string::npos ? he : he - hpos);
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string name = line.substr(0, colon);
            for (auto &c : name) c = (char)tolower((unsigned char)c);
            std::string val = line.substr(colon + 1);
            size_t vs = val.find_first_not_of(" \t");
            if (vs != std::string::npos) val = val.substr(vs);
            req.headers.emplace_back(name, val);
        }
        if (he == std::string::npos) break;
        hpos = he + 2;
    }

    size_t content_length = (size_t)strtoull(req.header("content-length").c_str(), nullptr, 10);
    if (content_length > kMaxRequest) { too_large = true; return false; }
    while (req.body.size() < content_length) {
        auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now()).count();
        if (remain <= 0) return false;
        struct pollfd pfd{fd, POLLIN, 0};
        if (::poll(&pfd, 1, (int)remain) <= 0) return false;
        char b2[65536];
        size_t want = content_length - req.body.size();
        if (want > sizeof(b2)) want = sizeof(b2);
        ssize_t n = ::recv(fd, b2, want, 0);
        if (n <= 0) { if (n < 0 && errno == EINTR) continue; return false; }
        req.body.append(b2, (size_t)n);
        if (req.body.size() > kMaxRequest) { too_large = true; return false; }
    }
    if (req.body.size() > content_length) req.body.resize(content_length);
    return true;
}

// ------------------------------------------------------------ JSON response

std::string json_escape(const std::string &s) {
    return qf::json_dump(qf::Json::make_str(s));  // quoted + escaped
}

// ----------------------------------------------------------- UTF-8 assembly

// Length of the longest prefix of s that ends on a complete UTF-8 codepoint.
// Invalid lead bytes pass through as single bytes so a malformed piece can
// never stall the stream.
size_t utf8_complete_prefix(const std::string &s) {
    size_t n = s.size(), done = 0;
    while (done < n) {
        unsigned char c = (unsigned char)s[done];
        int len = (c < 0x80) ? 1 : (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        if (len == 1) { done++; continue; }
        if (done + (size_t)len > n) break;  // incomplete tail: wait for more bytes
        done += (size_t)len;
    }
    return done;
}

// ------------------------------------------------------------- request body

struct GenParams {
    std::string prompt;                 // resolved (chat template applied)
    int   max_tokens = 128;
    float temperature = 1.0f;
    int   top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    bool  has_seed = false;
    bool  stream = false;
    bool  is_chat = false;
    std::vector<std::string> stop;      // max 4
};

// Minimal Qwen-style chat template:
//   <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
bool apply_chat_template(const qf::Json &messages, std::string &out, std::string &err) {
    if (!messages.is_arr() || messages.arr.empty()) {
        err = "messages must be a non-empty array";
        return false;
    }
    out.clear();
    for (const auto &m : messages.arr) {
        if (!m.is_obj()) { err = "message must be an object"; return false; }
        std::string role = m.get_str("role");
        std::string content = m.get_str("content");
        if (role.empty()) { err = "message missing role"; return false; }
        out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    return true;
}

// Extract the handful of supported fields; unknown fields are ignored.
bool parse_gen_params(const qf::Json &root, bool is_chat, GenParams &gp, std::string &err) {
    if (!root.is_obj()) { err = "request body must be a JSON object"; return false; }
    gp.is_chat = is_chat;
    if (is_chat) {
        const qf::Json *msgs = root.get("messages");
        if (!msgs || !apply_chat_template(*msgs, gp.prompt, err)) {
            if (err.empty()) err = "missing messages";
            return false;
        }
    } else {
        gp.prompt = root.get_str("prompt");
        if (root.get("prompt") && !root.get("prompt")->is_str()) {
            err = "prompt must be a string";
            return false;
        }
    }
    gp.max_tokens = (int)root.get_int("max_tokens", 128);
    if (gp.max_tokens < 1) gp.max_tokens = 1;
    if (gp.max_tokens > 4096) gp.max_tokens = 4096;
    gp.temperature = (float)root.get_num("temperature", 1.0);
    gp.top_k = (int)root.get_int("top_k", 0);
    if (gp.top_k < 0) gp.top_k = 0;
    gp.top_p = (float)root.get_num("top_p", 1.0);
    if (gp.top_p <= 0.0f) gp.top_p = 1.0f;
    if (gp.top_p > 1.0f) gp.top_p = 1.0f;
    gp.repetition_penalty = (float)root.get_num("repetition_penalty", 1.0);
    if (gp.repetition_penalty <= 0.0f) gp.repetition_penalty = 1.0f;
    const qf::Json *seed = root.get("seed");
    if (seed && seed->is_num()) { gp.seed = (uint64_t)seed->num; gp.has_seed = true; }
    gp.stream = root.get_bool("stream", false);
    const qf::Json *stop = root.get("stop");
    if (stop) {
        if (stop->is_str()) {
            if (!stop->str.empty()) gp.stop.push_back(stop->str);
        } else if (stop->is_arr()) {
            for (const auto &s : stop->arr) {
                if (gp.stop.size() >= 4) break;
                if (s.is_str() && !s.str.empty()) gp.stop.push_back(s.str);
            }
        }
    }
    return true;
}

// --------------------------------------------------------------- generation

struct GenResult {
    std::string text;            // full assembled output (non-stream use)
    std::string finish_reason = "stop";
    int prompt_tokens = 0;
    int gen_tokens = 0;
    double prefill_s = 0.0;
    double decode_tok_s = 0.0;
    bool ok = false;             // false => engine error / client abort
};

// emit(text) is called with UTF-8-complete output pieces; returning false
// aborts generation (client gone).
GenResult run_generation(const std::vector<int32_t> &ids, const GenParams &gp,
                         const std::function<bool(const std::string &)> &emit) {
    GenResult res;
    res.prompt_tokens = (int)ids.size();

    qf::SamplerParams sp;
    sp.temperature = gp.temperature;
    sp.top_k = gp.top_k;
    sp.top_p = gp.top_p;
    sp.repetition_penalty = gp.repetition_penalty;
    sp.seed = gp.seed;
    sp.has_seed = gp.has_seed;
    qf::Sampler sampler(sp);

    std::vector<float> logits(QF4_NVOCAB);
    std::vector<int32_t> seen(ids.begin(), ids.end());

    // Stop-string holdback: with stop strings configured, hold back up to
    // max_stop_len-1 chars so a partial match at the tail is never emitted.
    size_t max_stop = 0;
    for (auto &s : gp.stop) max_stop = std::max(max_stop, s.size());
    std::string pending;  // undelivered, UTF-8-complete text
    bool stopped = false;

    auto flush = [&](bool final) -> bool {
        if (pending.empty()) return true;
        // Earliest stop match wins.
        size_t hit = std::string::npos;
        for (auto &s : gp.stop) {
            size_t p = pending.find(s);
            if (p < hit) hit = p;
        }
        if (hit != std::string::npos) {
            std::string out = pending.substr(0, hit);
            pending.clear();
            stopped = true;
            if (!out.empty()) { res.text += out; return emit(out); }
            return true;
        }
        size_t safe = final ? pending.size()
                            : (pending.size() > max_stop ? pending.size() - max_stop : 0);
        if (safe == 0) return true;
        std::string out = pending.substr(0, safe);
        pending.erase(0, safe);
        res.text += out;
        return emit(out);
    };

    // --- prefill ---
    auto tp0 = std::chrono::steady_clock::now();
    long n = (long)ids.size();
    for (long i = 0; i + 1 < n; i++) {
        if (qf_hip4_submit_prefill(g_engine, ids[(size_t)i], i) != 0) {
            fprintf(stderr, "engine: submit_prefill failed at pos %ld\n", i);
            return res;
        }
    }
    if (qf_hip4_decode_step(g_engine, ids[(size_t)n - 1], n - 1, logits.data()) != 0) {
        fprintf(stderr, "engine: decode_step failed at pos %ld\n", n - 1);
        return res;
    }
    auto tp1 = std::chrono::steady_clock::now();
    res.prefill_s = std::chrono::duration<double>(tp1 - tp0).count();

    // --- decode ---
    std::string bytebuf;  // raw detokenized bytes awaiting UTF-8 assembly
    long pos = n;
    int token = -1;
    for (int g = 0; g < gp.max_tokens; g++) {
        if (g > 0) {
            if (qf_hip4_submit(g_engine, token, pos - 1) != 0 ||
                qf_hip4_wait(g_engine, pos - 1, logits.data()) != 0) {
                fprintf(stderr, "engine: submit/wait failed at pos %ld\n", pos - 1);
                return res;
            }
        }
        token = sampler.sample(logits.data(), logits.size(), seen);
        if (token == kEosId) { res.finish_reason = "stop"; break; }
        res.gen_tokens++;
        seen.push_back(token);
        bytebuf += g_tok->decode_token(token);
        size_t complete = utf8_complete_prefix(bytebuf);
        if (complete > 0) {
            pending += bytebuf.substr(0, complete);
            bytebuf.erase(0, complete);
        }
        if (!flush(false)) { res.finish_reason = "cancelled"; return res; }
        if (stopped) { res.finish_reason = "stop"; break; }
        pos++;
        if (g + 1 == gp.max_tokens) res.finish_reason = "length";
    }
    // Flush the tail: any bytes held by UTF-8 assembly, then the holdback.
    if (!bytebuf.empty()) { pending += bytebuf; bytebuf.clear(); }
    if (!flush(true)) { res.finish_reason = "cancelled"; return res; }

    auto tp2 = std::chrono::steady_clock::now();
    double dec_s = std::chrono::duration<double>(tp2 - tp1).count();
    res.decode_tok_s = dec_s > 0.0 && res.gen_tokens > 0 ? res.gen_tokens / dec_s : 0.0;
    res.ok = true;
    return res;
}

// ------------------------------------------------------------------ handlers

std::string timings_json(const GenResult &r) {
    char buf[160];
    snprintf(buf, sizeof(buf),
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d},"
             "\"timings\":{\"prefill_s\":%.4f,\"decode_tok_s\":%.2f}",
             r.prompt_tokens, r.gen_tokens, r.prefill_s, r.decode_tok_s);
    return buf;
}

void handle_health(Conn &c) {
    QfHip4Stats st;
    memset(&st, 0, sizeof(st));
    qf_hip4_stats(g_engine, &st);  // read-only counters; no engine lock needed
    double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
    char mib[128];
    int off = snprintf(mib, sizeof(mib), "[");
    for (int i = 0; i < QF4_NGPU; i++)
        off += snprintf(mib + off, sizeof(mib) - (size_t)off, "%s%.0f", i ? "," : "",
                        (double)st.hbm_bytes[i] / 1048576.0);
    snprintf(mib + off, sizeof(mib) - (size_t)off, "]");
    char body[512];
    snprintf(body, sizeof(body),
             "{\"status\":\"ok\",\"engine\":\"qf_hip4\",\"handoff\":%d,"
             "\"tokens_decoded\":%llu,\"hbm_mib\":%s,\"max_context\":%ld,"
             "\"uptime_s\":%.1f}",
             st.handoff_mode, (unsigned long long)st.tokens_decoded, mib,
             g_cfg.max_context, up);
    c.send_json(200, body);
}

void handle_models(Conn &c) {
    c.send_json(200,
        "{\"object\":\"list\",\"data\":[{\"id\":\"RadixArk-Qwen3.8-Flash-Next-NVFP4\","
        "\"object\":\"model\",\"owned_by\":\"radixark\"}]}");
}

// Raw token-id entry point for the correctness acceptance test.
//
// POST /v1/raw  {"tokens":[760,6511,314,9338,369], "max_tokens":48}
//   -> {"tokens":[...48 greedy ids...], ...}
//
// Prompt ids go in raw and generated ids come out raw: neither the chat
// template nor the BPE encode/decode round trip can pollute the comparison.
// Pure greedy argmax (ties -> lowest id): no sampler, no repetition penalty,
// no stop strings, and no EOS break, so the caller always gets exactly
// max_tokens ids and the FIRST DIVERGENT INDEX localizes the bug.
void handle_raw(Conn &c, const HttpRequest &req) {
    qf::Json root;
    std::string err;
    if (!json_parse(req.body, root, err)) {
        c.send_error(400, "malformed JSON body");
        return;
    }
    const qf::Json *t = root.get("tokens");
    if (!t || !t->is_arr() || t->arr.empty()) {
        c.send_error(400, "tokens must be a non-empty array of token ids");
        return;
    }
    std::vector<int32_t> ids;
    for (const auto &e : t->arr) {
        if (!e.is_num()) { c.send_error(400, "tokens must be integers"); return; }
        long v = (long)e.num;
        if (v < 0 || v >= QF4_NVOCAB) { c.send_error(400, "token id out of range"); return; }
        ids.push_back((int32_t)v);
    }
    long max_tokens = root.get_int("max_tokens", 48);
    if (max_tokens < 1) max_tokens = 1;
    if ((long)ids.size() + max_tokens > g_cfg.max_context) {
        c.send_error(400, "prompt_tokens + max_tokens exceeds max_context");
        return;
    }

    if (!g_engine_mu.try_lock_for(std::chrono::seconds(kEngineWaitS))) {
        c.send_error(503, "engine busy");
        return;
    }
    std::unique_lock<std::timed_mutex> lk(g_engine_mu, std::adopt_lock);
    qf_hip4_reset(g_engine);

    std::vector<float> logits(QF4_NVOCAB);
    const long n = (long)ids.size();
    auto tp0 = std::chrono::steady_clock::now();
    for (long i = 0; i + 1 < n; i++) {
        if (qf_hip4_submit_prefill(g_engine, ids[(size_t)i], i) != 0) {
            c.send_error(500, "prefill failed");
            return;
        }
    }
    if (qf_hip4_decode_step(g_engine, ids[(size_t)n - 1], n - 1, logits.data()) != 0) {
        c.send_error(500, "decode_step failed");
        return;
    }
    auto tp1 = std::chrono::steady_clock::now();

    std::vector<int32_t> out;
    std::string topk;
    long pos = n;
    int token = -1;
    for (long g = 0; g < max_tokens; g++) {
        if (g > 0) {
            if (qf_hip4_submit(g_engine, token, pos - 1) != 0 ||
                qf_hip4_wait(g_engine, pos - 1, logits.data()) != 0) {
                c.send_error(500, "submit/wait failed");
                return;
            }
        }
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < QF4_NVOCAB; i++)
            if (logits[(size_t)i] > bv) { bv = logits[(size_t)i]; best = i; }

        // Top-5 per step: a greedy stream can diverge from a reference either
        // because a stage is WRONG or because small numeric drift flipped a
        // near-tie. The margin between rank 1 and rank 2 tells which, and a
        // token list alone cannot.
        {
            int id5[5];
            float v5[5];
            for (int r = 0; r < 5; r++) { id5[r] = -1; v5[r] = -3e38f; }
            for (int i = 0; i < QF4_NVOCAB; i++) {
                float v = logits[(size_t)i];
                for (int r = 0; r < 5; r++) {
                    if (v > v5[r]) {
                        for (int q = 4; q > r; q--) { v5[q] = v5[q - 1]; id5[q] = id5[q - 1]; }
                        v5[r] = v; id5[r] = i;
                        break;
                    }
                }
            }
            if (!topk.empty()) topk += ",";
            topk += "[";
            for (int r = 0; r < 5; r++) {
                char e[64];
                snprintf(e, sizeof(e), "%s[%d,%.4f]", r ? "," : "", id5[r], v5[r]);
                topk += e;
            }
            topk += "]";
        }

        token = best;
        out.push_back(token);
        pos++;
    }
    auto tp2 = std::chrono::steady_clock::now();

    double prefill_s = std::chrono::duration<double>(tp1 - tp0).count();
    double dec_s = std::chrono::duration<double>(tp2 - tp1).count();
    std::string body = "{\"tokens\":[";
    for (size_t i = 0; i < out.size(); i++) {
        if (i) body += ",";
        body += std::to_string(out[i]);
    }
    qf_hip4_prof_dump(stderr);
    body += "],\"top5\":[" + topk + "]";
    body += ",\"prompt_tokens\":" + std::to_string(n);
    char tb[160];
    snprintf(tb, sizeof(tb), ",\"timings\":{\"prefill_s\":%.4f,\"decode_tok_s\":%.2f}}",
             prefill_s, dec_s > 0.0 ? (double)out.size() / dec_s : 0.0);
    body += tb;
    c.send_json(200, body);
}

void handle_generate(Conn &c, const HttpRequest &req, bool is_chat) {
    uint64_t id = g_req_id.fetch_add(1, std::memory_order_relaxed) + 1;

    qf::Json root;
    std::string err;
    if (!json_parse(req.body, root, err)) {
        c.send_error(400, "malformed JSON body");
        return;
    }
    GenParams gp;
    if (!parse_gen_params(root, is_chat, gp, err)) {
        c.send_error(400, err.c_str());
        return;
    }

    // Tokenize before touching the engine (tokenizer is read-only here).
    std::vector<int32_t> ids;
    g_tok->encode(gp.prompt, ids);
    if (ids.empty()) {
        c.send_error(400, "prompt is empty after tokenization");
        return;
    }
    if ((long)ids.size() + gp.max_tokens > g_cfg.max_context) {
        c.send_error(400, "prompt_tokens + max_tokens exceeds max_context");
        return;
    }

    // The engine is serial: one request in flight across the 4-GPU pipeline.
    if (!g_engine_mu.try_lock_for(std::chrono::seconds(kEngineWaitS))) {
        c.send_error(503, "engine busy");
        return;
    }
    std::unique_lock<std::timed_mutex> lk(g_engine_mu, std::adopt_lock);
    qf_hip4_reset(g_engine);

    std::string req_tag = "qfc-" + std::to_string(id);
    long created = (long)time(nullptr);
    GenResult res;

    if (!gp.stream) {
        res = run_generation(ids, gp, [&](const std::string &) { return !g_stop.load(); });
    } else {
        c.begin_sse();
        bool first = true;
        res = run_generation(ids, gp, [&](const std::string &piece) {
            std::string chunk;
            if (is_chat) {
                chunk = "{\"id\":\"" + req_tag + "\",\"object\":\"chat.completion.chunk\","
                        "\"created\":" + std::to_string(created) + ","
                        "\"model\":\"RadixArk-Qwen3.8-Flash-Next-NVFP4\",\"choices\":[{\"index\":0,"
                        "\"delta\":{" + (first ? "\"role\":\"assistant\"," : "") +
                        "\"content\":" + json_escape(piece) + "},\"finish_reason\":null}]}";
            } else {
                chunk = "{\"id\":\"" + req_tag + "\",\"object\":\"text_completion\","
                        "\"created\":" + std::to_string(created) + ","
                        "\"choices\":[{\"index\":0,\"text\":" + json_escape(piece) +
                        ",\"finish_reason\":null}]}";
            }
            first = false;
            return c.write_sse(chunk) && !g_stop.load();
        });
        // Final chunk: finish_reason + usage + timings, then [DONE].
        std::string fin;
        if (is_chat) {
            fin = "{\"id\":\"" + req_tag + "\",\"object\":\"chat.completion.chunk\","
                  "\"created\":" + std::to_string(created) + ","
                  "\"model\":\"RadixArk-Qwen3.8-Flash-Next-NVFP4\",\"choices\":[{\"index\":0,"
                  "\"delta\":{},\"finish_reason\":\"" + res.finish_reason + "\"}]," +
                  timings_json(res) + "}";
        } else {
            fin = "{\"id\":\"" + req_tag + "\",\"object\":\"text_completion\","
                  "\"created\":" + std::to_string(created) + ","
                  "\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"" +
                  res.finish_reason + "\"}]," + timings_json(res) + "}";
        }
        c.write_sse(fin);
        c.write_sse("[DONE]");
        c.dead.store(true);
    }

    if (!gp.stream) {
        if (!res.ok && res.finish_reason != "cancelled") {
            c.send_error(500, "engine error during generation");
        } else {
            std::string body;
            if (is_chat) {
                body = "{\"id\":\"" + req_tag + "\",\"object\":\"chat.completion\","
                       "\"created\":" + std::to_string(created) + ","
                       "\"model\":\"RadixArk-Qwen3.8-Flash-Next-NVFP4\",\"choices\":[{\"index\":0,"
                       "\"message\":{\"role\":\"assistant\",\"content\":" + json_escape(res.text) +
                       "},\"finish_reason\":\"" + res.finish_reason + "\"}]," +
                       timings_json(res) + "}";
            } else {
                body = "{\"id\":\"" + req_tag + "\",\"object\":\"text_completion\","
                       "\"created\":" + std::to_string(created) + ","
                       "\"choices\":[{\"index\":0,\"text\":" + json_escape(res.text) +
                       ",\"finish_reason\":\"" + res.finish_reason + "\"}]," +
                       timings_json(res) + "}";
            }
            c.send_json(200, body);
        }
    }

    fprintf(stderr, "req id=%llu prompt_tokens=%d gen_tokens=%d prefill_s=%.4f decode_toks_per_s=%.2f\n",
            (unsigned long long)id, res.prompt_tokens, res.gen_tokens,
            res.prefill_s, res.decode_tok_s);
}

// ------------------------------------------------------------- worker loop

void handle_conn(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    Conn c;
    c.fd = fd;
    HttpRequest req;
    bool too_large = false;
    if (!read_request(fd, req, too_large)) {
        if (too_large) c.send_error(413, "request too large");
        return;
    }

    if (req.method == "GET" && req.path == "/health") { handle_health(c); return; }
    if (req.method == "GET" && req.path == "/v1/models") { handle_models(c); return; }
    if (req.method == "POST" && req.path == "/v1/completions") { handle_generate(c, req, false); return; }
    if (req.method == "POST" && req.path == "/v1/chat/completions") { handle_generate(c, req, true); return; }
    if (req.method == "POST" && req.path == "/v1/raw") { handle_raw(c, req); return; }

    if ((req.path == "/v1/completions" || req.path == "/v1/chat/completions" ||
         req.path == "/v1/raw" || req.path == "/health" || req.path == "/v1/models"))
        c.send_error(405, "method not allowed");
    else
        c.send_error(404, "not found");
}

void worker_loop() {
    while (!g_stop.load(std::memory_order_relaxed)) {
        sockaddr_in cli{};
        socklen_t len = sizeof(cli);
        int fd = ::accept(g_listen_fd, (sockaddr *)&cli, &len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break;  // listen socket closed (shutdown) or fatal
        }
        handle_conn(fd);
        ::close(fd);
    }
}

// --------------------------------------------------------------------- main

void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s --model DIR [--port N] [--host ADDR] [--max-context N] [--tokenizer auto|PATH]\n"
        "  --model DIR        checkpoint directory (required)\n"
        "  --port N           listen port (default 8088)\n"
        "  --host ADDR        bind address (default 127.0.0.1)\n"
        "  --max-context N    admission cap (default 131072, clamp 262144)\n"
        "  --tokenizer SPEC   'auto' = <model>/tokenizer.json, or explicit path\n",
        argv0);
}

bool parse_args(int argc, char **argv, Config &cfg) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *v) { return v; };
        if (a == "--model" && i + 1 < argc) cfg.model_dir = need(argv[++i]);
        else if (a == "--port" && i + 1 < argc) cfg.port = atoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) cfg.host = argv[++i];
        else if (a == "--max-context" && i + 1 < argc) cfg.max_context = atol(argv[++i]);
        else if (a == "--tokenizer" && i + 1 < argc) cfg.tokenizer = argv[++i];
        else if (a == "--help" || a == "-h") return false;
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return false; }
    }
    if (cfg.model_dir.empty()) {
        fprintf(stderr, "error: --model DIR is required\n");
        return false;
    }
    if (cfg.max_context < 1) cfg.max_context = 1;
    if (cfg.max_context > kModelContext) cfg.max_context = kModelContext;
    if (cfg.port <= 0 || cfg.port > 65535) cfg.port = 8088;
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (!parse_args(argc, argv, g_cfg)) {
        usage(argv[0]);
        return 2;
    }

    fprintf(stderr, "qf_hip4_server: model=%s host=%s port=%d max_context=%ld tokenizer=%s\n",
            g_cfg.model_dir.c_str(), g_cfg.host.c_str(), g_cfg.port,
            g_cfg.max_context, g_cfg.tokenizer.c_str());

    // Tokenizer: "auto" resolves to <model>/tokenizer.json.
    std::string tok_path = g_cfg.tokenizer == "auto"
        ? g_cfg.model_dir + "/tokenizer.json" : g_cfg.tokenizer;
    auto tok = qf::qf_load_tokenizer_json(tok_path);
    if (!tok) {
        fprintf(stderr, "error: cannot load tokenizer %s\n", tok_path.c_str());
        return 2;
    }
    g_tok = tok.get();

    // Engine init loads all 4 GPUs. No CPU fallback: any failure is fatal.
    if (qf_hip4_init(&g_engine, g_cfg.model_dir.c_str()) != 0 || !g_engine) {
        fprintf(stderr, "error: qf_hip4_init failed for %s (no CPU fallback; exiting)\n",
                g_cfg.model_dir.c_str());
        return 2;
    }
    {
        QfHip4Stats st;
        memset(&st, 0, sizeof(st));
        if (qf_hip4_stats(g_engine, &st) == 0) {
            fprintf(stderr, "engine: loaded in %.2fs, handoff=%d, max_context=%ld\n",
                    st.load_seconds, st.handoff_mode, st.max_context);
            for (int i = 0; i < QF4_NGPU; i++)
                fprintf(stderr, "engine: gpu%d hbm %.0f MiB\n",
                        i, (double)st.hbm_bytes[i] / 1048576.0);
        }
    }

    ::signal(SIGPIPE, SIG_IGN);
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    ::sigaction(SIGINT, &sa, nullptr);
    ::sigaction(SIGTERM, &sa, nullptr);

    g_listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_fd < 0) { perror("socket"); return 2; }
    int one = 1;
    setsockopt(g_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)g_cfg.port);
    if (::inet_pton(AF_INET, g_cfg.host.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "error: bad bind address %s\n", g_cfg.host.c_str());
        return 2;
    }
    if (::bind(g_listen_fd, (sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 2; }
    if (::listen(g_listen_fd, 64) < 0) { perror("listen"); return 2; }
    fprintf(stderr, "http: listening on %s:%d (%d workers, engine serial)\n",
            g_cfg.host.c_str(), g_cfg.port, kWorkers);

    g_start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int i = 0; i < kWorkers; i++) workers.emplace_back(worker_loop);
    for (auto &t : workers) t.join();

    // Graceful shutdown: give an in-flight request up to 5 s to finish, then
    // free the engine regardless (it stays resident for the process lifetime
    // by design; teardown happens only here).
    if (!g_engine_mu.try_lock_for(std::chrono::seconds(5)))
        fprintf(stderr, "shutdown: aborting in-flight request\n");
    else
        g_engine_mu.unlock();
    qf_hip4_free(g_engine);
    g_engine = nullptr;
    fprintf(stderr, "shutdown: clean\n");
    return 0;
}
