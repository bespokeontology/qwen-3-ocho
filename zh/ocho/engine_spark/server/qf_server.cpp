// qf_server.cpp - OpenAI-compatible streaming server for the QF engine.
//
// Endpoints:
//   GET  /healthz               liveness + queue + engine error state
//   GET  /v1/stats              timing/throughput counters (ttft, tok/s)
//   GET  /v1/models
//   POST /v1/chat/completions   (stream + non-stream)
//   POST /v1/completions        (stream + non-stream)
//
// Generation runs through qf_model_load/qf_generate (qwenflash.h). The model
// is loaded once at startup and stays resident for the life of the process;
// after tokenization every request runs only GPU-native decode APIs. There
// is no per-token debug or device readback path here: tokens flow from the
// qf_generate callback straight into the SSE buffer.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "http_server.h"
#include "json.h"
#include "scheduler.h"
#include "tokenizer.h"
#include "qwenflash.h"

// Added to qwenflash.h by qf_server.patch; repeated here so the host-only
// stub check can also run against an unpatched live header.
extern int qf_generate_init(QfModel *m);

namespace qf {
namespace {

struct ServerConfig {
    std::string model_dir;
    std::string model_id = "qwen3.8-flash-next-nvfp4";
    std::string tokenizer_spec = "auto";
    std::string bind_addr = "0.0.0.0";
    int port = 8000;
    int http_threads = 8;
    int decode_workers = 1;      // engine decode is serial per GPU
    size_t max_queue = 64;       // bounded pending requests
    long max_context = 262144;   // clamped to the model limit after config load
    long max_gen_default = 4096; // per-request default max_tokens ceiling
};

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }

// ---- helpers ---------------------------------------------------------------

long now_unix() {
    return (long)std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string json_error(const std::string &msg, const std::string &type, const char *code = nullptr) {
    Json e = Json::make_obj();
    e.set("message", Json::make_str(msg));
    e.set("type", Json::make_str(type));
    if (code) e.set("code", Json::make_str(code));
    Json root = Json::make_obj();
    root.set("error", std::move(e));
    return json_dump(root);
}

// Pull the model's true context limit from <model_dir>/config.json.
long model_context_limit(const std::string &dir, long fallback) {
    std::string path = dir + "/config.json";
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return fallback;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string text((size_t)n, '\0');
    size_t rd = fread(&text[0], 1, (size_t)n, f);
    fclose(f);
    text.resize(rd);
    Json j;
    std::string err;
    if (!json_parse(text, j, err)) return fallback;
    long v = j.get_int("max_position_embeddings", 0);
    return v > 0 ? v : fallback;
}

// Qwen chat template. Special-token strings are literal; the tokenizer hook
// must map them (the native tokenizer.json loader and plugin tokenizers do;
// "ids" mode clients pre-template).
std::string apply_chat_template(const Json &messages) {
    std::string out;
    bool has_system = false;
    for (const auto &m : messages.arr)
        if (m.get_str("role") == "system") has_system = true;
    if (!has_system)
        out += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    for (const auto &m : messages.arr) {
        std::string role = m.get_str("role");
        std::string content = m.get_str("content");
        if (content.empty() && m.get("content") && m.get("content")->is_arr()) {
            // OpenAI multipart content: concat text parts.
            for (const auto &part : m.get("content")->arr)
                if (part.get_str("type") == "text") content += part.get_str("text");
        }
        out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    return out;
}

struct CompletionRequest {
    std::string prompt_text;
    bool is_chat = false;
    bool stream = false;
    bool include_usage = false;
    GenParams gen;
};

// Parse shared OpenAI request fields. On error returns false and fills err.
bool parse_completion_request(const Json &body, const ServerConfig &cfg, bool is_chat,
                              CompletionRequest &out, std::string &err) {
    out.is_chat = is_chat;
    if (is_chat) {
        const Json *msgs = body.get("messages");
        if (!msgs || !msgs->is_arr() || msgs->arr.empty()) {
            err = "messages must be a non-empty array";
            return false;
        }
        out.prompt_text = apply_chat_template(*msgs);
    } else {
        const Json *p = body.get("prompt");
        if (!p) {
            err = "prompt is required";
            return false;
        }
        if (p->is_str()) {
            out.prompt_text = p->str;
        } else if (p->is_arr()) {
            // array of token ids
            out.prompt_text.clear();
            for (const auto &v : p->arr) {
                if (!v.is_num()) {
                    err = "prompt array must contain token ids";
                    return false;
                }
                if (!out.prompt_text.empty()) out.prompt_text += ',';
                out.prompt_text += std::to_string((long long)v.num);
            }
        } else {
            err = "prompt must be a string or an array of token ids";
            return false;
        }
    }
    out.stream = body.get_bool("stream", false);
    if (out.stream) {
        const Json *so = body.get("stream_options");
        if (so && so->is_obj()) out.include_usage = so->get_bool("include_usage", false);
    }
    long mt = body.get_int("max_tokens", -1);
    if (mt < 0) mt = body.get_int("max_completion_tokens", -1);
    out.gen.max_tokens = mt > 0 ? (int)mt : (int)cfg.max_gen_default;
    out.gen.temperature = (float)body.get_num("temperature", 1.0);
    out.gen.top_p = (float)body.get_num("top_p", 1.0);
    out.gen.top_k = (int)body.get_int("top_k", 0);
    if (out.gen.temperature < 0) out.gen.temperature = 0;
    if (out.gen.top_p <= 0 || out.gen.top_p > 1) out.gen.top_p = 1.0f;
    if (out.gen.top_k < 0) out.gen.top_k = 0;
    return true;
}

// ---- server ----------------------------------------------------------------

class QfServer {
public:
    QfServer(ServerConfig cfg, QfModel *model, QfTokenizer *tok)
        : cfg_(std::move(cfg)), tok_(tok),
          sched_(model, tok, cfg_.max_queue, cfg_.decode_workers) {}

    void routes(HttpServer &http) {
        http.route("GET", "/healthz", [this](const HttpRequest &, HttpResponse &r) {
            SchedStats st = sched_.stats();
            int route_err = qf_route_error();
            Json root = Json::make_obj();
            root.set("status", Json::make_str(route_err ? "degraded" : "ok"));
            root.set("model", Json::make_str(cfg_.model_id));
            root.set("uptime_s", Json::make_num(st.uptime_s));
            root.set("queued", Json::make_num((double)st.queued));
            root.set("queue_capacity", Json::make_num((double)sched_.queue_capacity()));
            root.set("max_context", Json::make_num((double)cfg_.max_context));
            root.set("route_error", Json::make_num(route_err));
            r.send(route_err ? 503 : 200, "application/json", json_dump(root));
        });
        http.route("GET", "/v1/stats", [this](const HttpRequest &, HttpResponse &r) {
            SchedStats st = sched_.stats();
            double avg_tok_s = st.decode_seconds > 0.0
                                   ? (double)st.completion_tokens / st.decode_seconds
                                   : 0.0;
            Json root = Json::make_obj();
            root.set("model", Json::make_str(cfg_.model_id));
            root.set("uptime_s", Json::make_num(st.uptime_s));
            root.set("requests_done", Json::make_num((double)st.requests_done));
            root.set("requests_error", Json::make_num((double)st.requests_error));
            root.set("requests_cancelled", Json::make_num((double)st.requests_cancelled));
            root.set("prompt_tokens", Json::make_num((double)st.prompt_tokens));
            root.set("completion_tokens", Json::make_num((double)st.completion_tokens));
            root.set("decode_seconds", Json::make_num(st.decode_seconds));
            root.set("avg_completion_tok_s", Json::make_num(avg_tok_s));
            root.set("last_ttft_ms", Json::make_num(st.last_ttft_ms));
            root.set("last_request_tok_s", Json::make_num(st.last_tok_s));
            root.set("queued", Json::make_num((double)st.queued));
            r.send(200, "application/json", json_dump(root));
        });
        http.route("GET", "/v1/models", [this](const HttpRequest &, HttpResponse &r) {
            Json m = Json::make_obj();
            m.set("id", Json::make_str(cfg_.model_id));
            m.set("object", Json::make_str("model"));
            m.set("created", Json::make_num((double)now_unix()));
            m.set("owned_by", Json::make_str("qf"));
            Json data = Json::make_arr();
            data.push(std::move(m));
            Json root = Json::make_obj();
            root.set("object", Json::make_str("list"));
            root.set("data", std::move(data));
            r.send(200, "application/json", json_dump(root));
        });
        http.route("POST", "/v1/chat/completions",
                   [this](const HttpRequest &q, HttpResponse &r) { handle(q, r, true); });
        http.route("POST", "/v1/completions",
                   [this](const HttpRequest &q, HttpResponse &r) { handle(q, r, false); });
    }

    void start() { sched_.start(); }
    void stop() { sched_.stop(); }

private:
    ServerConfig cfg_;
    QfTokenizer *tok_;
    Scheduler sched_;
    std::atomic<uint64_t> req_seq_{1};

    void handle(const HttpRequest &http_req, HttpResponse &resp, bool is_chat) {
        Json body;
        std::string err;
        if (!json_parse(http_req.body, body, err) || !body.is_obj()) {
            resp.send(400, "application/json",
                      json_error("invalid JSON body: " + err, "invalid_request_error"));
            return;
        }

        CompletionRequest cr;
        if (!parse_completion_request(body, cfg_, is_chat, cr, err)) {
            resp.send(400, "application/json", json_error(err, "invalid_request_error"));
            return;
        }

        std::vector<int32_t> prompt;
        if (!tok_->encode(cr.prompt_text, prompt) || prompt.empty()) {
            resp.send(400, "application/json",
                      json_error("failed to tokenize prompt", "invalid_request_error"));
            return;
        }
        long n_prompt = (long)prompt.size();
        if (n_prompt >= cfg_.max_context) {
            char msg[160];
            snprintf(msg, sizeof(msg),
                     "prompt is %ld tokens; model context limit is %ld",
                     n_prompt, cfg_.max_context);
            resp.send(400, "application/json",
                      json_error(msg, "invalid_request_error", "context_length_exceeded"));
            return;
        }
        long budget = cfg_.max_context - n_prompt;
        if ((long)cr.gen.max_tokens > budget) cr.gen.max_tokens = (int)budget;
        if (cr.gen.max_tokens <= 0) {
            resp.send(400, "application/json",
                      json_error("no context budget left for generation",
                                 "invalid_request_error", "context_length_exceeded"));
            return;
        }

        auto req = std::make_shared<GenRequest>();
        req->prompt = std::move(prompt);
        req->params = cr.gen;

        std::string cmpl_id = "chatcmpl-qf" + std::to_string(req_seq_.fetch_add(1));
        long created = now_unix();

        if (cr.stream) {
            serve_stream(resp, req, cr, cmpl_id, created, n_prompt);
        } else {
            serve_blocking(resp, req, cr, cmpl_id, created, n_prompt);
        }
    }

    // Admits the request to the bounded queue; answers 503 when full.
    bool admit(HttpResponse &resp, const std::shared_ptr<GenRequest> &req) {
        if (sched_.submit(req)) return true;
        resp.send(503, "application/json",
                  json_error("request queue is full; retry later", "server_error", "queue_full"));
        return false;
    }

    std::string sse_chunk(const CompletionRequest &cr, const std::string &id, long created,
                          const char *delta_field, const std::string &content,
                          const char *finish_reason) {
        Json choice = Json::make_obj();
        choice.set("index", Json::make_num(0));
        if (cr.is_chat) {
            Json delta = Json::make_obj();
            if (delta_field) {
                delta.set("role", Json::make_str("assistant"));
                if (!content.empty()) delta.set("content", Json::make_str(content));
            }
            choice.set("delta", std::move(delta));
        } else {
            // text_completion chunks carry the piece directly on the choice
            choice.set("text", Json::make_str(content));
        }
        if (finish_reason)
            choice.set("finish_reason", Json::make_str(finish_reason));
        else
            choice.set("finish_reason", Json::make_null());
        Json root = Json::make_obj();
        root.set("id", Json::make_str(id));
        root.set("object", Json::make_str(cr.is_chat ? "chat.completion.chunk" : "text_completion"));
        root.set("created", Json::make_num((double)created));
        root.set("model", Json::make_str(cfg_.model_id));
        Json choices = Json::make_arr();
        choices.push(std::move(choice));
        root.set("choices", std::move(choices));
        return "data: " + json_dump(root) + "\n\n";
    }

    void serve_stream(HttpResponse &resp, const std::shared_ptr<GenRequest> &req,
                      const CompletionRequest &cr, const std::string &id, long created,
                      long n_prompt) {
        resp.begin_stream(200, "text/event-stream",
                          {{"Cache-Control", "no-cache"}, {"X-Accel-Buffering", "no"}});

        // Per-token sink: straight from the decode callback into the SSE
        // stream. A failed write means the client left -> cancel decode.
        auto on_token = [this, &resp, &req, &cr, &id, created](int32_t, const std::string &text) -> bool {
            if (req->cancel.load(std::memory_order_relaxed)) return false;
            return resp.write_chunk(sse_chunk(cr, id, created, "role", text, nullptr));
        };
        req->on_token = on_token;

        // Queue admission happens after headers are sent, so a full queue is
        // reported as an SSE error event (OpenAI-compatible behavior).
        if (!sched_.submit(req)) {
            resp.write_chunk("data: " + json_error("request queue is full; retry later",
                                                   "server_error", "queue_full") + "\n\n");
            resp.write_chunk("data: [DONE]\n\n");
            return;
        }

        // Wait for completion while watching for client disconnect.
        while (true) {
            {
                std::unique_lock<std::mutex> lk(req->mu);
                if (req->cv.wait_for(lk, std::chrono::milliseconds(200),
                                     [&] { return req->finished; }))
                    break;
            }
            if (!resp.client_alive()) req->cancel.store(true);
        }

        std::string finish = req->finish_reason == "cancelled" ? "stop" : req->finish_reason;
        if (finish == "error") {
            resp.write_chunk("data: " + json_error(req->error, "server_error") + "\n\n");
        } else {
            resp.write_chunk(sse_chunk(cr, id, created, nullptr, "", finish.c_str()));
            if (cr.include_usage) {
                Json u = Json::make_obj();
                u.set("prompt_tokens", Json::make_num((double)n_prompt));
                u.set("completion_tokens", Json::make_num(req->completion_tokens));
                u.set("total_tokens", Json::make_num((double)n_prompt + req->completion_tokens));
                Json root = Json::make_obj();
                root.set("id", Json::make_str(id));
                root.set("object", Json::make_str(cr.is_chat ? "chat.completion.chunk" : "text_completion"));
                root.set("created", Json::make_num((double)created));
                root.set("model", Json::make_str(cfg_.model_id));
                root.set("choices", Json::make_arr());
                root.set("usage", std::move(u));
                resp.write_chunk("data: " + json_dump(root) + "\n\n");
            }
        }
        resp.write_chunk("data: [DONE]\n\n");
    }

    void serve_blocking(HttpResponse &resp, const std::shared_ptr<GenRequest> &req,
                        const CompletionRequest &cr, const std::string &id, long created,
                        long n_prompt) {
        if (!admit(resp, req)) return;

        auto text = std::make_shared<std::string>();
        req->on_token = [text, &req](int32_t, const std::string &piece) {
            text->append(piece);
            return !req->cancel.load(std::memory_order_relaxed);
        };

        while (true) {
            {
                std::unique_lock<std::mutex> lk(req->mu);
                if (req->cv.wait_for(lk, std::chrono::milliseconds(200),
                                     [&] { return req->finished; }))
                    break;
            }
            if (!resp.client_alive()) req->cancel.store(true);
        }

        if (req->finish_reason == "error") {
            resp.send(500, "application/json", json_error(req->error, "server_error"));
            return;
        }
        if (req->finish_reason == "cancelled") return;  // client is gone anyway

        Json choice = Json::make_obj();
        choice.set("index", Json::make_num(0));
        choice.set("finish_reason", Json::make_str(req->finish_reason));
        if (cr.is_chat) {
            Json msg = Json::make_obj();
            msg.set("role", Json::make_str("assistant"));
            msg.set("content", Json::make_str(*text));
            choice.set("message", std::move(msg));
        } else {
            choice.set("text", Json::make_str(*text));
        }
        Json u = Json::make_obj();
        u.set("prompt_tokens", Json::make_num((double)n_prompt));
        u.set("completion_tokens", Json::make_num(req->completion_tokens));
        u.set("total_tokens", Json::make_num((double)n_prompt + req->completion_tokens));
        Json root = Json::make_obj();
        root.set("id", Json::make_str(id));
        root.set("object", Json::make_str(cr.is_chat ? "chat.completion" : "text_completion"));
        root.set("created", Json::make_num((double)created));
        root.set("model", Json::make_str(cfg_.model_id));
        Json choices = Json::make_arr();
        choices.push(std::move(choice));
        root.set("choices", std::move(choices));
        root.set("usage", std::move(u));
        resp.send(200, "application/json", json_dump(root));
    }
};

void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model DIR [options]\n"
            "  --model DIR          checkpoint dir (required; or QF_MODEL_DIR)\n"
            "  --model-id NAME      served model name (default: qwen3.8-flash-next-nvfp4)\n"
            "  --tokenizer SPEC     \"auto\" | \"json\" | \"ids\" | tokenizer.json path | plugin .so\n"
            "  --host ADDR          bind address (default 0.0.0.0)\n"
            "  --port N             listen port (default 8000, or QF_PORT)\n"
            "  --max-context N      server context cap, clamped to model limit (default 262144)\n"
            "  --max-queue N        bounded pending-request queue (default 64)\n"
            "  --max-gen N          default max_tokens ceiling (default 4096)\n"
            "  --http-threads N     connection workers (default 8)\n"
            "  --decode-workers N   decode workers (default 1; engine is serial per GPU)\n",
            argv0);
}

} // namespace
} // namespace qf

int main(int argc, char **argv) {
    using namespace qf;
    ServerConfig cfg;
    if (const char *e = getenv("QF_MODEL_DIR")) cfg.model_dir = e;
    if (const char *e = getenv("QF_PORT")) cfg.port = atoi(e);
    if (const char *e = getenv("QF_MAX_CONTEXT")) cfg.max_context = atol(e);
    if (const char *e = getenv("QF_TOKENIZER")) cfg.tokenizer_spec = e;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") cfg.model_dir = need("--model");
        else if (a == "--model-id") cfg.model_id = need("--model-id");
        else if (a == "--tokenizer") cfg.tokenizer_spec = need("--tokenizer");
        else if (a == "--host") cfg.bind_addr = need("--host");
        else if (a == "--port") cfg.port = atoi(need("--port"));
        else if (a == "--max-context") cfg.max_context = atol(need("--max-context"));
        else if (a == "--max-queue") cfg.max_queue = (size_t)atol(need("--max-queue"));
        else if (a == "--max-gen") cfg.max_gen_default = atol(need("--max-gen"));
        else if (a == "--http-threads") cfg.http_threads = atoi(need("--http-threads"));
        else if (a == "--decode-workers") cfg.decode_workers = atoi(need("--decode-workers"));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    if (cfg.model_dir.empty()) {
        usage(argv[0]);
        return 2;
    }

    // Configurable high context, hard-capped at the model's true limit.
    long model_limit = model_context_limit(cfg.model_dir, cfg.max_context);
    if (cfg.max_context > model_limit) {
        fprintf(stderr, "server: --max-context %ld clamped to model limit %ld\n",
                cfg.max_context, model_limit);
        cfg.max_context = model_limit;
    }
    // Keep the engine's KV-cache sizing and the loader's budget guard in
    // lockstep with the server cap: qf.cu and planner.cpp resolve
    // QF_MAX_CONTEXT once (at qf_forward_init / qf_model_load), so setting it
    // here is always in time.
    if (!getenv("QF_MAX_CONTEXT")) {
        char buf[24];
        snprintf(buf, sizeof(buf), "%ld", cfg.max_context);
        setenv("QF_MAX_CONTEXT", buf, 0);
    }

    auto tok = qf_load_tokenizer(cfg.tokenizer_spec, cfg.model_dir);
    if (!tok) return 1;

    static QfModel model;  // static: engine owns device pointers, freed at exit
    fprintf(stderr, "server: loading model from %s ...\n", cfg.model_dir.c_str());
    if (qf_model_load(&model, cfg.model_dir.c_str()) != 0) {
        fprintf(stderr, "server: qf_model_load failed\n");
        return 1;
    }
    // Allocate persistent decode state and build the dense/FP4 plans before
    // accepting traffic. qf_generate_init is idempotent, so requests reuse
    // this state rather than paying first-request initialization or reloading.
    if (qf_generate_init(&model) != 0) {
        fprintf(stderr, "server: qf_generate_init failed\n");
        qf_model_free(&model);
        return 1;
    }
    fprintf(stderr, "server: model and decode state resident (context cap %ld, queue %zu)\n",
            cfg.max_context, cfg.max_queue);

    QfServer app(cfg, &model, tok.get());
    app.start();

    HttpServer::Options ho;
    ho.port = cfg.port;
    ho.bind_addr = cfg.bind_addr;
    ho.threads = cfg.http_threads;
    HttpServer http(ho);
    app.routes(http);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    std::thread watch([&] {
        while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        http.stop();
    });

    bool ok = http.run();
    watch.join();
    app.stop();
    qf_model_free(&model);
    return ok ? 0 : 1;
}
