// http_server.h - small HTTP/1.1 server for the QF engine.
// Thread-pooled, bounded body size, streaming responses with liveness checks.
// POSIX only (Linux/macOS). No TLS here; put a terminator in front if needed.
#pragma once
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace qf {

struct HttpRequest {
    std::string method;                            // uppercase
    std::string path;                              // decoded path, no query
    std::map<std::string, std::string> query;      // decoded
    std::map<std::string, std::string> headers;    // lower-cased keys
    std::string body;
};

// Streaming-capable response writer bound to one connection.
// All methods are safe to call until the handler returns.
class HttpResponse {
public:
    // Send a complete response in one shot (no further calls allowed).
    void send(int status, const std::string &content_type, const std::string &body);

    // Begin a streamed response, then push raw bytes with write_chunk().
    // write_chunk returns false once the client is gone; handlers must stop.
    void begin_stream(int status, const std::string &content_type,
                      const std::vector<std::pair<std::string, std::string>> &extra_headers = {});
    bool write_chunk(const std::string &data);  // false on disconnect
    bool write_chunk(const char *data, size_t len);

    // Cheap client-liveness probe (0-timeout poll + peek). Use from long-running
    // handlers to abort work early when the client has gone away.
    bool client_alive() const;

    int status_sent() const { return status_; }

    // ---- internals (constructed by HttpServer) ----
    struct Impl;
    explicit HttpResponse(Impl *impl) : impl_(impl) {}

private:
    Impl *impl_;
    int status_ = 0;
};

using HttpHandler = std::function<void(const HttpRequest &, HttpResponse &)>;

class HttpServer {
public:
    struct Options {
        int port = 8000;
        std::string bind_addr = "0.0.0.0";
        int threads = 8;                 // connection workers
        int backlog = 128;               // listen backlog (bounded pending conns)
        size_t max_body = 512ull << 20;  // 512 MiB request cap
        int read_timeout_ms = 120000;    // header+body read deadline
    };

    explicit HttpServer(Options opt);
    ~HttpServer();

    void route(const std::string &method, const std::string &path, HttpHandler fn);

    // Blocks until stop() is called (e.g. from a signal handler).
    bool run();
    void stop();

private:
    Options opt_;
    int listen_fd_ = -1;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> workers_;
    std::map<std::string, HttpHandler> routes_;  // key: "METHOD /path"

    void worker_loop();
    void handle_conn(int fd);
};

const char *http_status_text(int status);

} // namespace qf
