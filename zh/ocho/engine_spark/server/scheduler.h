// scheduler.h - bounded request queue in front of the serial decode engine.
//
// The engine's qf_generate() is a single-GPU serial decode loop, so the
// scheduler runs a small fixed number of decode workers (default 1) fed by a
// bounded FIFO queue. submit() never blocks: a full queue is rejected so the
// HTTP layer can answer 503 instead of growing memory unboundedly.
// Cancellation is a per-request atomic flag observed inside the qf_generate
// token callback, so a disconnected client stops decode at the next token.
//
// Timing: every completed request records wall time, prompt/completion token
// counts and time-to-first-token into an aggregate Stats block exposed to the
// /healthz and /v1/stats endpoints. Timing is host-side only (steady_clock
// around qf_generate and its callback) and adds no device synchronization.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <deque>

#include "qwenflash.h"
#include "tokenizer.h"

namespace qf {

struct GenParams {
    int max_tokens = 256;
    float temperature = 1.0f;
    int top_k = 0;        // 0 = disabled
    float top_p = 1.0f;
};

struct GenRequest {
    uint64_t id = 0;
    std::vector<int32_t> prompt;
    GenParams params;

    // Called once per generated token from the decode worker thread.
    // `text` is the token's byte piece. Return false to cancel generation.
    std::function<bool(int32_t token, const std::string &text)> on_token;

    std::atomic<bool> cancel{false};

    // Completion state (set once, then `done` is signalled).
    std::mutex mu;
    std::condition_variable cv;
    bool finished = false;
    std::string finish_reason;  // "stop" | "length" | "cancelled" | "error"
    std::string error;
    int completion_tokens = 0;
};

// Aggregate server-side timing/throughput snapshot (all requests so far).
struct SchedStats {
    uint64_t requests_done = 0;      // finished with stop/length
    uint64_t requests_error = 0;     // finished with error
    uint64_t requests_cancelled = 0;
    uint64_t prompt_tokens = 0;
    uint64_t completion_tokens = 0;
    double decode_seconds = 0.0;     // sum of in-engine wall time per request
    double last_ttft_ms = 0.0;       // time to first token of the last request
    double last_tok_s = 0.0;         // decode tok/s of the last request
    double uptime_s = 0.0;
    size_t queued = 0;               // requests waiting for a worker
};

class Scheduler {
public:
    // max_queue: maximum queued (not yet running) requests; 0 rejects nothing
    // until running slots fill (i.e. queue capacity 0 = admit only if idle).
    Scheduler(QfModel *model, QfTokenizer *tok, size_t max_queue, int workers = 1);
    ~Scheduler();

    // Non-blocking admission. Returns false when the queue is full.
    bool submit(const std::shared_ptr<GenRequest> &req);

    size_t queued() const;
    size_t queue_capacity() const { return max_queue_; }

    // Snapshot of aggregate timing/throughput counters (thread-safe).
    SchedStats stats() const;

    void start();
    void stop();  // cancels in-flight work and joins workers

private:
    QfModel *model_;
    QfTokenizer *tok_;
    size_t max_queue_;
    int workers_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<GenRequest>> queue_;
    std::vector<std::shared_ptr<GenRequest>> running_;  // in-flight, for stop() cancellation
    bool stopping_ = false;
    std::atomic<uint64_t> next_id_{1};
    std::vector<std::thread> threads_;

    // stats_ is guarded by its own mutex: the decode worker updates it once
    // per finished request, readers are the HTTP threads.
    mutable std::mutex stats_mu_;
    SchedStats stats_;
    std::chrono::steady_clock::time_point started_;

    void worker_loop();
    void run_request(const std::shared_ptr<GenRequest> &req);
};

} // namespace qf
