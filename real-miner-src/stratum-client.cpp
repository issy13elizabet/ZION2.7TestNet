/*
 * ZION Stratum Client Implementation - MIT Licensed
 * Cross-platform support: Windows, Linux, macOS, Android, iOS
 * Copyright (c) 2025 Maitreya-ZionNet
 */

#include "zion-miner-mit.h"
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

// Platform-specific networking
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    #define CLOSE_SOCKET closesocket
    #define SOCKET_ERROR_CHECK(x) ((x) == SOCKET_ERROR)
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define CLOSE_SOCKET close
    #define SOCKET_ERROR_CHECK(x) ((x) < 0)
#endif

// Mobile platform detection
#ifdef __ANDROID__
    #include <android/log.h>
    #define MOBILE_PLATFORM "Android"
#elif defined(__IPHONE_OS_VERSION_MIN_REQUIRED)
    #include <os/log.h>
    #define MOBILE_PLATFORM "iOS"
#else
    #define MOBILE_PLATFORM "Desktop"
#endif

// JSON parsing (simple implementation)
#include <regex>
#include <map>

namespace zion::mining {

/**
 * Simple JSON Parser for Stratum Protocol
 */
class SimpleJSON {
public:
    std::map<std::string, std::string> values;
    
    static SimpleJSON parse(const std::string& json_str) {
        SimpleJSON result;
        
        // Simple regex-based JSON parsing
        std::regex field_regex(R"("([^"]+)":\s*"([^"]*)")");
        std::regex number_regex(R"("([^"]+)":\s*([0-9]+))");
        std::regex null_regex(R"("([^"]+)":\s*null)");
        
        std::sregex_iterator iter(json_str.begin(), json_str.end(), field_regex);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            result.values[(*iter)[1]] = (*iter)[2];
        }
        
        // Parse numbers
        iter = std::sregex_iterator(json_str.begin(), json_str.end(), number_regex);
        for (; iter != end; ++iter) {
            result.values[(*iter)[1]] = (*iter)[2];
        }
        
        return result;
    }
    
    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second : default_value;
    }
    
    int get_int(const std::string& key, int default_value = 0) const {
        auto it = values.find(key);
        return (it != values.end()) ? std::stoi(it->second) : default_value;
    }
};

/**
 * Cross-Platform Stratum Client Implementation
 */
class StratumClient::Impl {
private:
    SOCKET socket_;
    std::string host_;
    int port_;
    bool connected_;
    std::atomic<bool> shutdown_requested_{false};
    
    std::thread receive_thread_;
    JobCallback job_callback_;
    DifficultyCallback difficulty_callback_;
    
    mutable std::mutex socket_mutex_;
    std::string partial_message_;
    
    // Platform-specific initialization
    bool network_initialized_;

public:
    Impl() : socket_(INVALID_SOCKET), port_(0), connected_(false), network_initialized_(false) {
        #ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0) {
            network_initialized_ = true;
        }
        #else
        network_initialized_ = true;
        #endif
    }
    
    ~Impl() {
        disconnect();
        
        #ifdef _WIN32
        if (network_initialized_) {
            WSACleanup();
        }
        #endif
    }
    
    bool connect(const std::string& host, int port) {
        if (!network_initialized_) {
            log_error("Network not initialized");
            return false;
        }
        
        disconnect(); // Ensure clean state
        
        host_ = host;
        port_ = port;
        
        log_info("Connecting to " + host + ":" + std::to_string(port));
        
        // Create socket
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ == INVALID_SOCKET) {
            log_error("Failed to create socket");
            return false;
        }
        
        // Resolve hostname
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(static_cast<uint16_t>(port));
        
        struct hostent* host_entry = gethostbyname(host.c_str());
        if (!host_entry) {
            // Try direct IP address
            if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
                log_error("Invalid host address: " + host);
                CLOSE_SOCKET(socket_);
                socket_ = INVALID_SOCKET;
                return false;
            }
        } else {
            memcpy(&server_addr.sin_addr, host_entry->h_addr_list[0], host_entry->h_length);
        }
        
        // Connect
        if (SOCKET_ERROR_CHECK(::connect(socket_, 
                                       reinterpret_cast<struct sockaddr*>(&server_addr), 
                                       sizeof(server_addr)))) {
            log_error("Failed to connect to " + host + ":" + std::to_string(port));
            CLOSE_SOCKET(socket_);
            socket_ = INVALID_SOCKET;
            return false;
        }
        
        connected_ = true;
        shutdown_requested_ = false;
        
        // Start receive thread
        receive_thread_ = std::thread(&Impl::receive_thread_worker, this);
        
        log_info("Connected to pool successfully");
        return true;
    }
    
    bool authenticate(const std::string& username, const std::string& password) {
        if (!connected_) return false;
        
        // Send mining.authorize
        std::string auth_msg = "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"" + 
                              username + "\",\"" + password + "\"]}\n";
        
        return send_message(auth_msg);
    }
    
    bool subscribe() {
        if (!connected_) return false;
        
        // Send mining.subscribe
        std::string client_info = std::string(MOBILE_PLATFORM) + "/ZION-Miner-MIT/1.0";
        std::string subscribe_msg = "{\"id\":2,\"method\":\"mining.subscribe\",\"params\":[\"" + 
                                   client_info + "\"]}\n";
        
        return send_message(subscribe_msg);
    }
    
    void disconnect() {
        if (connected_) {
            log_info("Disconnecting from pool");
            shutdown_requested_ = true;
            connected_ = false;
            
            if (socket_ != INVALID_SOCKET) {
                CLOSE_SOCKET(socket_);
                socket_ = INVALID_SOCKET;
            }
            
            if (receive_thread_.joinable()) {
                receive_thread_.join();
            }
        }
    }
    
    bool submit_share(const std::string& job_id, uint64_t nonce, const std::vector<uint8_t>& hash) {
        if (!connected_) return false;
        
        // Convert hash to hex string
        std::stringstream ss;
        for (const auto& byte : hash) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
        }
        std::string hash_hex = ss.str();
        
        // Convert nonce to hex string
        std::stringstream nonce_ss;
        nonce_ss << std::hex << std::setfill('0') << std::setw(16) << nonce;
        std::string nonce_hex = nonce_ss.str();
        
        // Send mining.submit
        std::string submit_msg = "{\"id\":3,\"method\":\"mining.submit\",\"params\":[\"" + 
                                job_id + "\",\"" + nonce_hex + "\",\"" + hash_hex + "\"]}\n";
        
        return send_message(submit_msg);
    }
    
    void set_job_callback(JobCallback callback) {
        job_callback_ = callback;
    }
    
    void set_difficulty_callback(DifficultyCallback callback) {
        difficulty_callback_ = callback;
    }
    
    bool is_connected() const {
        return connected_;
    }
    
    std::string get_connection_info() const {
        if (connected_) {
            return host_ + ":" + std::to_string(port_) + " (Connected)";
        } else {
            return "Not connected";
        }
    }

private:
    bool send_message(const std::string& message) {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        
        if (!connected_ || socket_ == INVALID_SOCKET) {
            return false;
        }
        
        int bytes_sent = send(socket_, message.c_str(), static_cast<int>(message.length()), 0);
        if (SOCKET_ERROR_CHECK(bytes_sent)) {
            log_error("Failed to send message");
            return false;
        }
        
        log_debug("Sent: " + message.substr(0, message.length() - 1)); // Remove \n for logging
        return true;
    }
    
    void receive_thread_worker() {
        log_info("Stratum receive thread started");
        
        char buffer[4096];
        
        while (!shutdown_requested_ && connected_) {
            int bytes_received = recv(socket_, buffer, sizeof(buffer) - 1, 0);
            
            if (bytes_received <= 0) {
                if (!shutdown_requested_) {
                    log_error("Connection lost");
                    connected_ = false;
                }
                break;
            }
            
            buffer[bytes_received] = '\0';
            
            // Handle partial messages
            std::string received_data = partial_message_ + std::string(buffer);
            partial_message_.clear();
            
            // Split by newlines
            size_t pos = 0;
            while ((pos = received_data.find('\n')) != std::string::npos) {
                std::string message = received_data.substr(0, pos);
                received_data.erase(0, pos + 1);
                
                if (!message.empty()) {
                    handle_stratum_message(message);
                }
            }
            
            // Store remaining partial message
            if (!received_data.empty()) {
                partial_message_ = received_data;
            }
        }
        
        log_info("Stratum receive thread stopped");
    }
    
    void handle_stratum_message(const std::string& message) {
        log_debug("Received: " + message);
        
        try {
            SimpleJSON json = SimpleJSON::parse(message);
            
            std::string method = json.get("method");
            
            if (method == "mining.notify") {
                handle_mining_notify(json);
            } else if (method == "mining.set_difficulty") {
                handle_set_difficulty(json);
            } else if (json.values.count("result") || json.values.count("error")) {
                handle_response(json);
            }
        } catch (const std::exception& e) {
            log_error("Failed to parse stratum message: " + std::string(e.what()));
        }
    }
    
    void handle_mining_notify(const SimpleJSON& json) {
        // Parse mining.notify parameters
        // This is a simplified version - real implementation would parse the full params array
        
        JobInfo job;
        job.job_id = json.get("job_id", "default_job");
        
        // Create mock block header for now
        std::string prev_hash = json.get("prevhash", "0000000000000000000000000000000000000000000000000000000000000000");
        job.block_header.resize(80); // Standard block header size
        
        // Fill with mock data (in real implementation, this would be properly parsed)
        for (size_t i = 0; i < job.block_header.size(); ++i) {
            job.block_header[i] = static_cast<uint8_t>(i ^ 0x42);
        }
        
        job.target_difficulty = 1000; // Default difficulty
        job.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        job.extra_nonce = "00000000";
        
        if (job_callback_) {
            job_callback_(job);
        }
    }
    
    void handle_set_difficulty(const SimpleJSON& json) {
        uint64_t difficulty = static_cast<uint64_t>(json.get_int("difficulty", 1000));
        
        if (difficulty_callback_) {
            difficulty_callback_(difficulty);
        }
    }
    
    void handle_response(const SimpleJSON& json) {
        std::string result = json.get("result");
        std::string error = json.get("error");
        
        if (!error.empty() && error != "null") {
            log_error("Pool error: " + error);
        } else if (!result.empty()) {
            log_info("Pool response: " + result);
        }
    }
    
    void log_info(const std::string& message) {
        log_message("INFO", message);
    }
    
    void log_error(const std::string& message) {
        log_message("ERROR", message);
    }
    
    void log_debug(const std::string& message) {
        log_message("DEBUG", message);
    }
    
    void log_message(const std::string& level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        #ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "ZionMiner", "[%s] %s", level.c_str(), message.c_str());
        #elif defined(__IPHONE_OS_VERSION_MIN_REQUIRED)
        os_log(OS_LOG_DEFAULT, "[%s] %s", level.c_str(), message.c_str());
        #else
        std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] [" 
                  << level << "] " << message << std::endl;
        #endif
    }
};

// StratumClient wrapper implementation
StratumClient::StratumClient() : pImpl(std::make_unique<Impl>()) {}
StratumClient::~StratumClient() = default;

bool StratumClient::connect(const std::string& host, int port) {
    return pImpl->connect(host, port);
}

bool StratumClient::authenticate(const std::string& username, const std::string& password) {
    return pImpl->authenticate(username, password);
}

bool StratumClient::subscribe() {
    return pImpl->subscribe();
}

void StratumClient::disconnect() {
    pImpl->disconnect();
}

bool StratumClient::submit_share(const std::string& job_id, uint64_t nonce, const std::vector<uint8_t>& hash) {
    return pImpl->submit_share(job_id, nonce, hash);
}

void StratumClient::set_job_callback(JobCallback callback) {
    pImpl->set_job_callback(callback);
}

void StratumClient::set_difficulty_callback(DifficultyCallback callback) {
    pImpl->set_difficulty_callback(callback);
}

bool StratumClient::is_connected() const {
    return pImpl->is_connected();
}

std::string StratumClient::get_connection_info() const {
    return pImpl->get_connection_info();
}

} // namespace zion::mining