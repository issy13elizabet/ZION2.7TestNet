#include "pool-connection.h"
#include <iostream>
#include <chrono>
#include <thread>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netdb.h>
#endif

PoolConnection::PoolConnection(const std::string& address, int port, const std::string& wallet)
    : pool_address_(address), pool_port_(port), wallet_address_(wallet) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

PoolConnection::~PoolConnection() {
    disconnect();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool PoolConnection::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connected_.load()) {
        return true;
    }
    
    // For simulation purposes, always return true
    // In real implementation, this would establish TCP connection
    std::cout << "🔗 Připojuji k poolu " << pool_address_ << ":" << pool_port_ << std::endl;
    std::cout << "💰 Wallet: " << wallet_address_ << std::endl;
    
    connected_.store(true);
    
    // Start connection maintenance thread
    connection_thread_ = std::thread(&PoolConnection::maintain_connection, this);
    
    return true;
}

void PoolConnection::disconnect() {
    connected_.store(false);
    
    if (connection_thread_.joinable()) {
        connection_thread_.join();
    }
    
    std::cout << "🔌 Odpojeno od poolu" << std::endl;
}

bool PoolConnection::submit_share(uint32_t nonce, uint64_t hash) {
    if (!connected_.load()) {
        return false;
    }
    
    // Simulate share submission
    std::cout << "📤 Odesílám share - Nonce: " << nonce 
              << ", Hash: 0x" << std::hex << hash << std::dec << std::endl;
    
    // Simulate network delay
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    return true;
}

void PoolConnection::send_donation(const std::string& address, uint64_t amount) {
    if (!connected_.load()) {
        return;
    }
    
    std::cout << "🎁 Donation: " << amount << " ZION → " << address << std::endl;
}

void PoolConnection::maintain_connection() {
    while (connected_.load()) {
        // Simulate keepalive
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        if (connected_.load()) {
            // Send ping to pool
            send_stratum_message("{\"id\":1,\"method\":\"mining.ping\",\"params\":[]}");
        }
    }
}

bool PoolConnection::send_stratum_message(const std::string& message) {
    // Simulate stratum protocol message
    return true;
}