// scheduler.cpp - bounded queue + decode worker over qf_generate().
//
// Timing hooks: run_request brackets qf_generate with steady_clock and the
// token callback stamps time-to-first-token. One mutex-protected aggregate
// update per finished request; nothing here touches the decode hot path.
#include "scheduler.h"

#include <cstdio>

namespace qf {

Scheduler::Scheduler(QfModel *model, QfTokenizer *tok, size_t max_queue, int workers)
    : model_(model), tok_(tok), max_queue_(max_queue), workers_(workers > 0 ? workers : 1),
      started_(std::chrono::steady_clock::now()) {}

Scheduler::~Scheduler() { stop(); }

bool Scheduler::submit(const std::shared_ptr<GenRequest> &req) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopping_) return false;
        if (queue_.size() >= max_queue_) return false;  // bounded: caller answers 503
        req->id = next_id_.fetch_add(1, std::memory_order_relaxed);
        queue_.push_back(req);
    }
    cv_.notify_one();
    return true;
}

size_t Scheduler::queued() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex &>(mu_));
    return queue_.size();
}

SchedStats Scheduler::stats() const {
    SchedStats s;
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        s = stats_;
    }
    s.uptime_s = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - started_)
                     .count();
    s.queued = queued();
    return s;
}

void Scheduler::start() {
    for (int i = 0; i < workers_; i++)
        threads_.emplace_back([this] { worker_loop(); });
}

void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopping_) return;
        stopping_ = true;
        for (auto &r : queue_) r->cancel.store(true);
        for (auto &r : running_) r->cancel.store(true);
    }
    cv_.notify_all();
    for (auto &t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
}

void Scheduler::worker_loop() {
    while (true) {
        std::shared_ptr<GenRequest> req;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_) {
                // Drain: finish every queued request so no waiter hangs.
                while (!queue_.empty()) {
                    auto r = queue_.front();
                    queue_.pop_front();
                    std::lock_guard<std::mutex> rlk(r->mu);
                    r->finish_reason = "cancelled";
                    r->finished = true;
                    r->cv.notify_all();
                }
                return;
            }
            req = queue_.front();
            queue_.pop_front();
            running_.push_back(req);
        }
        run_request(req);
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t i = 0; i < running_.size(); i++)
                if (running_[i] == req) {
                    running_.erase(running_.begin() + (long)i);
                    break;
                }
        }
    }
}

namespace {
struct CbCtx {
    GenRequest *req;
    QfTokenizer *tok;
    std::string pending;  // undelivered tail of a split UTF-8 codepoint
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point first_token;
    std::chrono::steady_clock::time_point last_token;
    double ttft_ms = -1.0;  // <0 until the first generated token lands
    int timed_tokens = 0;
};

// Length of the longest prefix of s that ends on a complete UTF-8 codepoint.
// Invalid lead bytes are passed through as single bytes so a malformed piece
// can never stall the stream.
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

// Called by qf_generate once per sampled token, on the decode thread.
// Nonzero return aborts generation (cancellation).
int token_cb(int token, void *ud) {
    CbCtx *ctx = (CbCtx *)ud;
    if (ctx->req->cancel.load(std::memory_order_relaxed)) return 1;
    if (token < 0) return 0;  // prefill progress poll: cancellation check only
    const auto now = std::chrono::steady_clock::now();
    if (ctx->ttft_ms < 0.0) {
        ctx->ttft_ms = std::chrono::duration<double, std::milli>(
                           now - ctx->t_start)
                           .count();
        ctx->first_token = now;
    }
    ctx->last_token = now;
    ctx->timed_tokens++;
    ctx->pending += ctx->tok->decode_token(token);
    ctx->req->completion_tokens++;
    // Emit only whole codepoints: byte-level BPE tokens can split a
    // multi-byte UTF-8 character across tokens, and SSE chunks carrying a
    // partial sequence corrupt strict clients.
    size_t complete = utf8_complete_prefix(ctx->pending);
    if (complete == 0) return 0;
    std::string text = ctx->pending.substr(0, complete);
    ctx->pending.erase(0, complete);
    bool cont = true;
    if (ctx->req->on_token) cont = ctx->req->on_token(token, text);
    if (!cont) {
        ctx->req->cancel.store(true);
        return 1;
    }
    return 0;
}
} // namespace

void Scheduler::run_request(const std::shared_ptr<GenRequest> &req) {
    const auto started = std::chrono::steady_clock::now();
    CbCtx ctx{req.get(), tok_, {}, started, started, started, -1.0, 0};

    int rc = qf_generate(model_,
                         req->prompt.data(), (int)req->prompt.size(),
                         req->params.max_tokens,
                         req->params.temperature, req->params.top_k, req->params.top_p,
                         token_cb, &ctx);

    const double wall_s = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - ctx.t_start)
                              .count();

    // Flush any bytes still held back by UTF-8 assembly (tail of the final
    // codepoint) before signalling completion, unless the client is gone.
    if (!ctx.pending.empty() && req->on_token && !req->cancel.load(std::memory_order_relaxed)) {
        req->on_token(-1, ctx.pending);
        ctx.pending.clear();
    }

    {
        std::lock_guard<std::mutex> lk(req->mu);
        if (req->cancel.load(std::memory_order_relaxed) && rc != 0)
            req->finish_reason = "cancelled";
        else if (rc != 0) {
            req->finish_reason = "error";
            req->error = "qf_generate failed";
        } else if (req->completion_tokens >= req->params.max_tokens)
            req->finish_reason = "length";
        else
            req->finish_reason = "stop";
        req->finished = true;
    }

    // Aggregate timing: one update per request, off the decode hot path.
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.prompt_tokens += (uint64_t)req->prompt.size();
        stats_.completion_tokens += (uint64_t)req->completion_tokens;
        stats_.decode_seconds += wall_s;
        if (req->finish_reason == "error") stats_.requests_error++;
        else if (req->finish_reason == "cancelled") stats_.requests_cancelled++;
        else stats_.requests_done++;
        if (ctx.ttft_ms >= 0.0) stats_.last_ttft_ms = ctx.ttft_ms;
        if (ctx.timed_tokens > 1) {
            const double decode_s = std::chrono::duration<double>(
                                        ctx.last_token - ctx.first_token)
                                        .count();
            if (decode_s > 0.0)
                stats_.last_tok_s = (ctx.timed_tokens - 1) / decode_s;
        } else if (req->completion_tokens > 0 && wall_s > 0.0) {
            stats_.last_tok_s = req->completion_tokens / wall_s;
        }
    }
    req->cv.notify_all();
}

} // namespace qf
