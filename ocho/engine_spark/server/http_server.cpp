// http_server.cpp - HTTP/1.1 server implementation.
#include "http_server.h"

#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdio>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace qf {

const char *http_status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

// ---------------------------------------------------------------- HttpResponse

struct HttpResponse::Impl {
    int fd = -1;
    std::atomic<bool> dead{false};
    bool headers_sent = false;

    bool write_all(const char *data, size_t len) {
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

    bool probe_alive() const {
        if (dead.load(std::memory_order_relaxed)) return false;
        struct pollfd pfd{fd, (short)(POLLIN | POLLERR | POLLHUP), 0};
        int r = ::poll(&pfd, 1, 0);
        if (r <= 0) return r == 0;  // no events => still there
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;
        if (pfd.revents & POLLIN) {
            // Client should not be sending mid-response; readable means EOF/RST.
            char c;
            ssize_t n = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
            if (n == 0) return false;                    // clean close
            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) return false;
        }
        return true;
    }

    void send_head(int status, const std::string &content_type,
                   const std::vector<std::pair<std::string, std::string>> &extra,
                   long content_length) {
        std::string head;
        char line[128];
        snprintf(line, sizeof(line), "HTTP/1.1 %d %s\r\n", status, http_status_text(status));
        head = line;
        head += "Content-Type: " + content_type + "\r\n";
        head += "Connection: close\r\n";
        if (content_length >= 0) {
            snprintf(line, sizeof(line), "Content-Length: %ld\r\n", content_length);
            head += line;
        }
        for (auto &h : extra) head += h.first + ": " + h.second + "\r\n";
        head += "\r\n";
        write_all(head.data(), head.size());
        headers_sent = true;
    }
};

void HttpResponse::send(int status, const std::string &content_type, const std::string &body) {
    status_ = status;
    impl_->send_head(status, content_type, {}, (long)body.size());
    impl_->write_all(body.data(), body.size());
    impl_->dead.store(true);
}

void HttpResponse::begin_stream(int status, const std::string &content_type,
                                const std::vector<std::pair<std::string, std::string>> &extra) {
    status_ = status;
    impl_->send_head(status, content_type, extra, -1);  // no length; close-delimited
}

bool HttpResponse::write_chunk(const std::string &data) {
    return write_chunk(data.data(), data.size());
}

bool HttpResponse::write_chunk(const char *data, size_t len) {
    return impl_->write_all(data, len);
}

bool HttpResponse::client_alive() const { return impl_->probe_alive(); }

// ---------------------------------------------------------------- HttpServer

namespace {

std::string lower(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

std::string url_decode(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hex(s[i + 1]), lo = hex(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out += (char)((hi << 4) | lo);
                i += 2;
                continue;
            }
        }
        out += s[i] == '+' ? ' ' : s[i];
    }
    return out;
}

// Read from fd until `buf` contains "\r\n\r\n" or the deadline/size cap hits.
// Returns: 1 headers complete, 0 closed, -1 error/timeout.
int read_headers(int fd, std::string &buf, size_t cap, int timeout_ms) {
    char tmp[8192];
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (buf.find("\r\n\r\n") == std::string::npos) {
        if (buf.size() > cap) return -1;
        auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now())
                          .count();
        if (remain <= 0) return -1;
        struct pollfd pfd{fd, POLLIN, 0};
        int r = ::poll(&pfd, 1, (int)remain);
        if (r <= 0) return -1;
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n == 0) return 0;
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        buf.append(tmp, (size_t)n);
    }
    return 1;
}

} // namespace

HttpServer::HttpServer(Options opt) : opt_(std::move(opt)) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::route(const std::string &method, const std::string &path, HttpHandler fn) {
    routes_[method + " " + path] = std::move(fn);
}

void HttpServer::stop() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) return;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    for (auto &t : workers_)
        if (t.joinable()) t.join();
}

bool HttpServer::run() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("socket");
        return false;
    }
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)opt_.port);
    if (::inet_pton(AF_INET, opt_.bind_addr.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "http: bad bind address %s\n", opt_.bind_addr.c_str());
        return false;
    }
    if (::bind(listen_fd_, (sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return false;
    }
    if (::listen(listen_fd_, opt_.backlog) < 0) {
        perror("listen");
        return false;
    }
    fprintf(stderr, "http: listening on %s:%d (%d workers)\n",
            opt_.bind_addr.c_str(), opt_.port, opt_.threads);

    for (int i = 0; i < opt_.threads; i++)
        workers_.emplace_back([this] { worker_loop(); });
    for (auto &t : workers_)
        if (t.joinable()) t.join();
    workers_.clear();
    return true;
}

void HttpServer::worker_loop() {
    while (!stop_.load(std::memory_order_relaxed)) {
        sockaddr_in cli{};
        socklen_t len = sizeof(cli);
        int fd = ::accept(listen_fd_, (sockaddr *)&cli, &len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (stop_.load(std::memory_order_relaxed)) break;
            if (errno == EMFILE || errno == ENFILE) { ::usleep(10000); continue; }
            break;  // listen socket closed
        }
        handle_conn(fd);
        ::close(fd);
    }
}

void HttpServer::handle_conn(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    HttpRequest req;
    HttpResponse::Impl impl;
    impl.fd = fd;
    HttpResponse resp(&impl);

    std::string buf;
    int hr = read_headers(fd, buf, 1u << 20, opt_.read_timeout_ms);
    if (hr != 1) {
        if (hr < 0) resp.send(408, "text/plain", "header read timeout\n");
        return;
    }

    size_t head_end = buf.find("\r\n\r\n");
    std::string head = buf.substr(0, head_end);
    std::string rest = buf.substr(head_end + 4);

    // Request line.
    size_t eol = head.find("\r\n");
    std::string rline = head.substr(0, eol);
    size_t sp1 = rline.find(' ');
    size_t sp2 = rline.find(' ', sp1 == std::string::npos ? sp1 : sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        resp.send(400, "text/plain", "malformed request line\n");
        return;
    }
    req.method = rline.substr(0, sp1);
    std::string target = rline.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t qm = target.find('?');
    req.path = url_decode(qm == std::string::npos ? target : target.substr(0, qm));
    if (qm != std::string::npos) {
        std::string qs = target.substr(qm + 1);
        size_t pos = 0;
        while (pos <= qs.size()) {
            size_t amp = qs.find('&', pos);
            std::string kv = qs.substr(pos, amp == std::string::npos ? amp : amp - pos);
            size_t eq = kv.find('=');
            if (eq != std::string::npos)
                req.query[url_decode(kv.substr(0, eq))] = url_decode(kv.substr(eq + 1));
            else if (!kv.empty())
                req.query[url_decode(kv)] = "";
            if (amp == std::string::npos) break;
            pos = amp + 1;
        }
    }

    // Headers.
    size_t hpos = eol == std::string::npos ? head.size() : eol + 2;
    while (hpos < head.size()) {
        size_t he = head.find("\r\n", hpos);
        std::string line = head.substr(hpos, he == std::string::npos ? he : he - hpos);
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string val = line.substr(colon + 1);
            size_t vs = val.find_first_not_of(" \t");
            if (vs != std::string::npos) val = val.substr(vs);
            req.headers[lower(line.substr(0, colon))] = val;
        }
        if (he == std::string::npos) break;
        hpos = he + 2;
    }

    // Body (Content-Length only; no chunked request support).
    size_t content_length = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        content_length = (size_t)strtoull(it->second.c_str(), nullptr, 10);
        if (content_length > opt_.max_body) {
            resp.send(413, "text/plain", "request body too large\n");
            return;
        }
    }
    req.body = std::move(rest);
    while (req.body.size() < content_length) {
        char tmp[65536];
        size_t want = content_length - req.body.size();
        if (want > sizeof(tmp)) want = sizeof(tmp);
        struct pollfd pfd{fd, POLLIN, 0};
        int r = ::poll(&pfd, 1, opt_.read_timeout_ms);
        if (r <= 0) {
            resp.send(408, "text/plain", "body read timeout\n");
            return;
        }
        ssize_t n = ::recv(fd, tmp, want, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            resp.send(400, "text/plain", "incomplete body\n");
            return;
        }
        req.body.append(tmp, (size_t)n);
    }
    if (req.body.size() > content_length) req.body.resize(content_length);

    auto ri = routes_.find(req.method + " " + req.path);
    if (ri == routes_.end()) {
        // Method mismatch on a known path gets a 405, otherwise 404.
        for (auto &kv : routes_) {
            if (kv.first.size() > req.path.size() + 1 &&
                kv.first.compare(kv.first.size() - req.path.size(), req.path.size(), req.path) == 0) {
                resp.send(405, "application/json",
                          "{\"error\":{\"message\":\"method not allowed\",\"type\":\"invalid_request_error\"}}");
                return;
            }
        }
        resp.send(404, "application/json",
                  "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}");
        return;
    }

    try {
        ri->second(req, resp);
    } catch (const std::exception &e) {
        fprintf(stderr, "http: handler exception: %s\n", e.what());
        if (!impl.headers_sent)
            resp.send(500, "application/json",
                      "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}");
    } catch (...) {
        if (!impl.headers_sent)
            resp.send(500, "application/json",
                      "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}");
    }
}

} // namespace qf
