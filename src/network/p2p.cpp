#include "network/p2p.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include "randomx_wrapper.h"

namespace zion {

static bool split_host_port(const std::string& hp, std::string& host, std::string& port) {
    auto pos = hp.rfind(':');
    if (pos == std::string::npos) return false;
    host = hp.substr(0, pos);
    port = hp.substr(pos+1);
    return !host.empty() && !port.empty();
}

P2PNode::P2PNode(const P2PConfig& cfg) : cfg_(cfg) {}
P2PNode::~P2PNode() { stop(); }

bool P2PNode::start() {
    if (running_) return true;
    running_ = true;

    // Create listening socket
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) { log("socket() failed"); running_ = false; return false; }

    int opt = 1; setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(cfg_.bind_port);
    if (cfg_.bind_host == "0.0.0.0") addr.sin_addr.s_addr = INADDR_ANY; else inet_pton(AF_INET, cfg_.bind_host.c_str(), &addr.sin_addr);

    if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) { log("bind() failed"); ::close(listen_fd_); running_ = false; return false; }
    if (listen(listen_fd_, 16) < 0) { log("listen() failed"); ::close(listen_fd_); running_ = false; return false; }

    // Start accept loop
    accept_thread_ = std::thread([this]{ accept_loop(); });

    // Connect to seeds in background
    std::thread([this]{ connect_seeds(); }).detach();

    log("P2P started on port " + std::to_string(cfg_.bind_port));
    return true;
}

void P2PNode::stop() {
    if (!running_) return;
    running_ = false;

    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1; }
    if (accept_thread_.joinable()) accept_thread_.join();

    std::lock_guard<std::mutex> lk(peers_mutex_);
    for (int fd : peer_fds_) { ::shutdown(fd, SHUT_RDWR); ::close(fd); }
    peer_fds_.clear();
    for (auto& t : peer_threads_) if (t.joinable()) t.join();
    peer_threads_.clear();
}

void P2PNode::accept_loop() {
    while (running_) {
        sockaddr_in cli{}; socklen_t clilen = sizeof(cli);
        int cfd = ::accept(listen_fd_, (sockaddr*)&cli, &clilen);
        if (cfd < 0) {
            if (!running_) break;
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(peers_mutex_);
            peer_fds_.insert(cfd);
        }
        peer_threads_.emplace_back([this, cfd]{ handle_peer(cfd); });
    }
}

static bool send_all(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::send(fd, p + off, len - off, 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

static bool recv_line(int fd, std::string& out) {
    out.clear();
    char ch;
    while (true) {
        ssize_t n = ::recv(fd, &ch, 1, 0);
        if (n <= 0) return false;
        if (ch == '\n') break;
        out.push_back(ch);
        if (out.size() > 4096) return false;
    }
    return true;
}

void P2PNode::broadcast_inv(const std::string& inv_type, const std::string& hash_hex, int exclude_fd) {
    std::lock_guard<std::mutex> lk(peers_mutex_);
    for (int fd : peer_fds_) {
        if (fd == exclude_fd) continue;
        std::string msg = "INV " + inv_type + " " + hash_hex + "\n";
        send_all(fd, msg.data(), msg.size());
    }
}

void P2PNode::handle_peer(int fd) {
    log("Peer connected");
    // Simple handshake: send VERSION and expect VERSION
    uint32_t height = callbacks_.get_height ? callbacks_.get_height() : 0;
    std::string ver = "VERSION " + cfg_.network_id + " " + std::to_string(cfg_.chain_id) + " " + std::to_string(height) + " ZION/1\n";
    if (!send_all(fd, ver.data(), ver.size())) { ::close(fd); return; }

    std::string line;
    if (!recv_line(fd, line)) { ::close(fd); return; }
    if (line.rfind("VERSION", 0) != 0) { ::close(fd); return; }
    // VERSION <network_id> <chain_id> <height>
    char nid[256]={0}; int rchain=0; unsigned int rheight=0;
    if (sscanf(line.c_str(), "VERSION %255s %d %u", nid, &rchain, &rheight) != 3) { ::close(fd); return; }
    if ((!cfg_.network_id.empty() && cfg_.network_id != std::string(nid)) || (cfg_.chain_id && cfg_.chain_id != rchain)) {
        log("Peer network mismatch, closing");
        ::close(fd); return;
    }
    const std::string verack = "VERACK\n";
    if (!send_all(fd, verack.data(), verack.size())) { ::close(fd); return; }

    // Initiate sync if remote height > local
    if (rheight > height) {
        std::string req = "GETHEADERS " + std::to_string(height) + " 2000\n";
        if (!send_all(fd, req.data(), req.size())) { ::close(fd); return; }
    }

    auto last_ping = std::chrono::steady_clock::now();
    while (running_) {
        // Send ping every 10s
        auto now = std::chrono::steady_clock::now();
        if (now - last_ping > std::chrono::seconds(10)) {
            const std::string ping = "PING\n";
            if (!send_all(fd, ping.data(), ping.size())) break;
            last_ping = now;
        }
        // Non-blocking read with small timeout
        fd_set rfds; FD_ZERO(&rfds); FD_SET(fd, &rfds);
        timeval tv{0, 200 * 1000}; // 200ms
        int rv = select(fd+1, &rfds, nullptr, nullptr, &tv);
        if (rv > 0 && FD_ISSET(fd, &rfds)) {
            std::string msg;
            if (!recv_line(fd, msg)) break;

            if (msg == "PING") {
                const std::string pong = "PONG\n";
                if (!send_all(fd, pong.data(), pong.size())) break;
                continue;
            }

            if (msg.rfind("HEADERS", 0) == 0) {
                // HEADERS N followed by N header hex lines and END
                unsigned int n = 0; sscanf(msg.c_str(), "HEADERS %u", &n);
                for (unsigned int i = 0; i < n; ++i) {
                    std::string hdr_hex; if (!recv_line(fd, hdr_hex)) { n = i; break; }
                    // Convert hex to bytes and compute hash = sha256(header_bytes)
                    if (hdr_hex.size() % 2 != 0) continue;
                    std::vector<uint8_t> bytes(hdr_hex.size()/2);
                    auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
                    for(size_t j=0;j<bytes.size();++j){ bytes[j]=(uint8_t)((hexval(hdr_hex[2*j])<<4)|hexval(hdr_hex[2*j+1])); }
                    Hash h = sha256(bytes.data(), bytes.size());
                    static const char* x = "0123456789abcdef"; std::string hash_hex; hash_hex.reserve(64);
                    for (auto b : h) { hash_hex.push_back(x[b>>4]); hash_hex.push_back(x[b&0xF]); }
                    // Request block data for each header
                    std::string req = std::string("GETDATA BLOCK ") + hash_hex + "\n";
                    if (!send_all(fd, req.data(), req.size())) break;
                }
                // Read END line
                std::string endline; recv_line(fd, endline);
                continue;
            }

            if (msg.rfind("GETHEADERS", 0) == 0) {
                // GETHEADERS <start_height> <max_count>
                uint32_t start = 0, count = 2000;
                sscanf(msg.c_str(), "GETHEADERS %u %u", &start, &count);
                std::vector<std::string> headers;
                if (callbacks_.get_headers && callbacks_.get_headers(start, count, headers)) {
                    // Send HEADERS N then each header hex on its own line, end with END
                    std::string hdr = "HEADERS " + std::to_string(headers.size()) + "\n";
                    if (!send_all(fd, hdr.data(), hdr.size())) break;
                    for (auto& h : headers) {
                        std::string ln = h + "\n";
                        if (!send_all(fd, ln.data(), ln.size())) { rv = -1; break; }
                    }
                    const std::string end = "END\n";
                    if (!send_all(fd, end.data(), end.size())) break;
                }
                continue;
            }

            if (msg.rfind("GETDATA", 0) == 0) {
                // GETDATA <type> <hash_hex>
                char type[16]={0}; char hashbuf[129] = {0};
                if (sscanf(msg.c_str(), "GETDATA %15s %128s", type, hashbuf) >= 2) {
                    std::string hash_hex(hashbuf);
                    std::string payload_hex;
                    if (std::string(type)=="BLOCK") {
                        if (callbacks_.get_block && callbacks_.get_block(hash_hex, payload_hex)) {
                            std::string ln = "BLOCK " + std::to_string(payload_hex.size()) + "\n";
                            if (!send_all(fd, ln.data(), ln.size())) break;
                            payload_hex.push_back('\n');
                            if (!send_all(fd, payload_hex.data(), payload_hex.size())) break;
                        }
                    } else if (std::string(type)=="TX") {
                        if (callbacks_.get_tx && callbacks_.get_tx(hash_hex, payload_hex)) {
                            std::string ln = "TX " + std::to_string(payload_hex.size()) + "\n";
                            if (!send_all(fd, ln.data(), ln.size())) break;
                            payload_hex.push_back('\n');
                            if (!send_all(fd, payload_hex.data(), payload_hex.size())) break;
                        }
                    }
                }
                continue;
            }

            if (msg.rfind("INV", 0) == 0) {
                // INV <type> <hash>
                char type[16]={0}; char hashbuf[129] = {0};
                if (sscanf(msg.c_str(), "INV %15s %128s", type, hashbuf) >= 2) {
                    std::string hash_hex(hashbuf);
                    // Request data
                    std::string req = std::string("GETDATA ") + type + " " + hash_hex + "\n";
                    if (!send_all(fd, req.data(), req.size())) break;
                }
                continue;
            }

            if (msg.rfind("BLOCK", 0) == 0) {
                // BLOCK <len>, next line contains hex payload
                size_t len = 0;
                sscanf(msg.c_str(), "BLOCK %zu", &len);
                std::string payload;
                if (!recv_line(fd, payload)) break;
                std::string hash_hex;
                if (callbacks_.submit_block && callbacks_.submit_block(payload, hash_hex)) {
                    // Broadcast INV to others
                    broadcast_inv("BLOCK", hash_hex, fd);
                }
                continue;
            }

            if (msg.rfind("TX", 0) == 0) {
                // TX <len>, next line contains hex payload
                size_t len = 0;
                sscanf(msg.c_str(), "TX %zu", &len);
                std::string payload;
                if (!recv_line(fd, payload)) break;
                std::string hash_hex;
                if (callbacks_.submit_tx && callbacks_.submit_tx(payload, hash_hex)) {
                    broadcast_inv("TX", hash_hex, fd);
                }
                continue;
            }
        }
    }

    ::close(fd);
    std::lock_guard<std::mutex> lk(peers_mutex_);
    peer_fds_.erase(fd);
    log("Peer disconnected");
}

void P2PNode::connect_seeds() {
    for (const auto& hp : cfg_.seed_nodes) {
        if (!running_) break;
        std::string host, port;
        if (!split_host_port(hp, host, port)) continue;

        addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) continue;

        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { freeaddrinfo(res); continue; }
        if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            log("Connected to seed: " + hp);
            {
                std::lock_guard<std::mutex> lk(peers_mutex_);
                peer_fds_.insert(fd);
            }
            peer_threads_.emplace_back([this, fd]{ handle_peer(fd); });
        } else {
            ::close(fd);
        }
        freeaddrinfo(res);
    }
}

} // namespace zion
