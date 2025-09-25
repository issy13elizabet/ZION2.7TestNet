#include "network/pool.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>

namespace zion {

// Forward declare send_all implemented below
static bool send_all(int fd, const std::string& s);

// --- Minimal JSON helpers for line-delimited JSON-RPC ---
static void send_json(int fd, const std::string& s) {
    std::string line = s; if (line.empty() || line.back()!='\n') line.push_back('\n');
    send_all(fd, line);
}

static void send_json_result(int fd, const std::string& id, const std::string& result_json) {
    std::string id_part = id.empty() ? "null" : id;
    std::string msg = std::string("{\"jsonrpc\":\"2.0\",\"id\":") + id_part + ",\"result\":" + result_json + "}";
    send_json(fd, msg);
}

static void send_json_error(int fd, const std::string& id, const std::string& message) {
    std::string id_part = id.empty() ? "null" : id;
    std::string msg = std::string("{\"jsonrpc\":\"2.0\",\"id\":") + id_part + ",\"error\":{\"message\":\"" + message + "\"}}";
    send_json(fd, msg);
}

static bool extract_json_string_field(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = std::string("\"") + key + "\":\"";
    auto p = json.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    auto e = json.find('"', p);
    if (e == std::string::npos) return false;
    out = json.substr(p, e - p);
    return true;
}

static bool extract_json_number_field(const std::string& json, const std::string& key, uint32_t& out) {
    std::string pat = std::string("\"") + key + "\":";
    auto p = json.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    // skip spaces
    while (p < json.size() && isspace((unsigned char)json[p])) ++p;
    size_t q = p;
    while (q < json.size() && (isdigit((unsigned char)json[q]) || json[q]=='-')) ++q;
    if (q == p) return false;
    try { out = static_cast<uint32_t>(std::stoul(json.substr(p, q - p))); } catch (...) { return false; }
    return true;
}

static bool extract_json_object(const std::string& json, const std::string& key, std::string& out) {
    std::string pat = std::string("\"") + key + "\":";
    auto p = json.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    while (p < json.size() && json[p] != '{' && json[p] != 'n' && json[p] != '"') ++p;
    if (p >= json.size()) return false;
    if (json[p] != '{') { out = "null"; return true; }
    int depth = 0; size_t start = p; size_t i = p;
    for (; i < json.size(); ++i) {
        if (json[i] == '{') depth++;
        else if (json[i] == '}') { depth--; if (depth == 0) { ++i; break; } }
    }
    if (depth != 0) return false;
    out = json.substr(start, i - start);
    return true;
}

static bool parse_json_rpc(const std::string& json, std::string& method, std::string& id, std::string& params_obj) {
    method.clear(); id = "1"; params_obj = "{}";
    extract_json_string_field(json, "method", method);
    // id can be number or string; capture raw token
    auto id_pos = json.find("\"id\":");
    if (id_pos != std::string::npos) {
        size_t p = id_pos + 5;
        while (p < json.size() && isspace((unsigned char)json[p])) ++p;
        size_t q = p;
        if (p < json.size() && (json[p] == '"')) {
            ++q; auto e = json.find('"', q); if (e != std::string::npos) id = json.substr(q, e - q); else id = "1";
        } else {
            while (q < json.size() && (isdigit((unsigned char)json[q]) || json[q]=='-')) ++q;
            if (q > p) id = json.substr(p, q - p); else id = "1";
        }
    }
    extract_json_object(json, "params", params_obj);
    return !method.empty();
}

static std::tuple<std::string,std::string,std::string> parse_login_params(const std::string& params) {
    std::string user, pass, agent;
    extract_json_string_field(params, "login", user);
    extract_json_string_field(params, "pass", pass);
    extract_json_string_field(params, "agent", agent);
    return {user, pass, agent};
}

static std::tuple<std::string,std::string,std::string> parse_submit_params(const std::string& params) {
    std::string job_id, nonce, result;
    extract_json_string_field(params, "job_id", job_id);
    extract_json_string_field(params, "nonce", nonce);
    extract_json_string_field(params, "result", result);
    return {job_id, nonce, result};
}

PoolServer::PoolServer(const PoolConfig& cfg, Callbacks cbs) : cfg_(cfg), cbs_(cbs) {}
PoolServer::~PoolServer() { stop(); }

bool PoolServer::start() {
    if (running_) return true;
    running_ = true;

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) { running_ = false; return false; }

    int opt=1; setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(cfg_.port); addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(listen_fd_); running_ = false; return false; }
    if (listen(listen_fd_, 16) < 0) { ::close(listen_fd_); running_ = false; return false; }

    accept_thread_ = std::thread([this]{ accept_loop(); });
    std::cout << "[POOL] Listening on port " << cfg_.port << std::endl;
    return true;
}

void PoolServer::stop() {
    if (!running_) return;
    running_ = false;
    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1; }
    if (accept_thread_.joinable()) accept_thread_.join();
}

void PoolServer::accept_loop() {
    while (running_) {
        sockaddr_in cli{}; socklen_t clilen=sizeof(cli);
        int cfd = ::accept(listen_fd_, (sockaddr*)&cli, &clilen);
        if (cfd < 0) { if (!running_) break; continue; }
        std::thread(&PoolServer::handle_client, this, cfd).detach();
    }
}

static bool send_all(int fd, const std::string& s) {
    const char* p = s.data(); size_t n = s.size(); size_t off=0;
    while (off < n) { ssize_t w = ::send(fd, p+off, n-off, 0); if (w <= 0) return false; off += (size_t)w; }
    return true;
}

static bool recv_line(int fd, std::string& out) {
    out.clear(); char ch;
    while (true) {
        ssize_t r = ::recv(fd, &ch, 1, 0);
        if (r <= 0) return false;
        if (ch == '\n') break;
        out.push_back(ch);
        if (out.size() > 4096) return false;
    }
    return true;
}

void PoolServer::handle_client(int fd) {
    // Stratum-like JSON-RPC protocol
    std::string worker_id;
    bool logged_in = false;
    
    while (running_) {
        std::string line;
        if (!recv_line(fd, line)) break;
        
        // Parse JSON-RPC request
        std::string method, id = "1", params;
        if (!parse_json_rpc(line, method, id, params)) {
            send_json_error(fd, id, "Parse error");
            continue;
        }
        
        if (method == "login") {
            auto [user, pass, agent] = parse_login_params(params);
            (void)agent;
            if (cbs_.login) {
                std::string result = cbs_.login(user, pass);
                if (!result.empty() && result[0] == '{') {
                    worker_id = user;
                    logged_in = true;
                    send_json_result(fd, id, result);
                } else {
                    send_json_error(fd, id, "Login failed");
                }
            } else {
                send_json_error(fd, id, "Login not supported");
            }
        } else if (method == "getjob") {
            // Return a job even if not logged in (stateless mode allowed)
            std::string job = cbs_.get_job ? cbs_.get_job() : "{}";
            send_json_result(fd, id, job);
        } else if (method == "submit") {
            auto [job_id, nonce, result] = parse_submit_params(params);
            bool ok = cbs_.submit_share ? cbs_.submit_share(job_id, nonce, result, worker_id) : false;
            send_json_result(fd, id, ok ? "\"OK\"" : "\"REJECTED\"");
        } else {
            send_json_error(fd, id, "Unknown method");
        }
    }
    ::close(fd);
}

} // namespace zion
