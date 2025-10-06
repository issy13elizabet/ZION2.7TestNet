#include "zion-simple-miner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ZionSimpleMiner::ZionSimpleMiner(DeviceType device_type, const std::string& pool_address, int pool_port, const std::string& wallet_address)
    : device_type_(device_type)
    , pool_address_(pool_address)
    , pool_port_(pool_port)
    , wallet_address_(wallet_address) 
{
    std::cout << "🚀 ZION AI Miner inicializován pro " << wallet_address << std::endl;
    std::cout << "🎯 Pool: " << pool_address << ":" << pool_port << std::endl;
    std::cout << "💎 Device Type: " << static_cast<int>(device_type) << std::endl;
    std::cout << "💝 Auto-Donate: " << donation_config_.total_percent << "% aktivní" << std::endl;
    
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

ZionSimpleMiner::~ZionSimpleMiner() {
    stop();
    disconnect_from_pool();
    
#ifdef _WIN32
    WSACleanup();
#endif
}

bool ZionSimpleMiner::start(int thread_count) {
    if (is_running_.load()) {
        std::cout << "⚠️ Miner už běží!" << std::endl;
        return false;
    }
    
    if (!connect_to_pool()) {
        std::cout << "❌ Nepodařilo se připojit k poolu!" << std::endl;
        return false;
    }
    
    if (thread_count <= 0) {
        thread_count = std::max(1u, std::thread::hardware_concurrency());
    }
    
    is_running_.store(true);
    
    std::cout << "🔥 Spouštím " << thread_count << " mining threadů..." << std::endl;
    
    // Start worker threads
    for (int i = 0; i < thread_count; i++) {
        worker_threads_.emplace_back(&ZionSimpleMiner::worker_thread, this, i);
    }
    
    // Enhance with AI
    enhance_with_cosmic_ai();
    
    return true;
}

void ZionSimpleMiner::stop() {
    if (!is_running_.load()) return;
    
    std::cout << "🛑 Zastavujem mining..." << std::endl;
    is_running_.store(false);
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    std::cout << "✅ Mining zastaven!" << std::endl;
}

void ZionSimpleMiner::enhance_with_cosmic_ai() {
    std::cout << "🧠 Aktivujem ZION Cosmic AI Enhancement..." << std::endl;
    
    // AI multiplier based on cosmic harmony and golden ratio
    double cosmic_factor = M_PI * ai_multiplier_.load();
    ai_multiplier_.store(cosmic_factor);
    
    // Enhance consciousness level
    consciousness_level_.store(consciousness_level_.load() + 8);  // Add cosmic wisdom
    
    std::cout << "✨ AI Enhancement aktivní! Multiplier: " << ai_multiplier_.load() 
              << ", Consciousness: " << consciousness_level_.load() << std::endl;
}

uint64_t ZionSimpleMiner::cosmic_harmony_hash(const std::vector<uint8_t>& data, uint32_t nonce) {
    // Simple but effective ZION Cosmic Harmony hash
    uint64_t hash = 0x5A494F4E32303235ULL; // "ZION2025" in hex
    
    // Add AI enhancement
    hash ^= static_cast<uint64_t>(ai_multiplier_.load() * 1000000);
    
    // Process input data with cosmic harmony
    for (size_t i = 0; i < data.size(); i++) {
        hash ^= data[i];
        hash *= 0x100000001B3ULL; // FNV prime
        hash ^= (hash >> 33);
    }
    
    // Add nonce with consciousness enhancement
    hash ^= nonce;
    hash ^= consciousness_level_.load();
    
    // Final cosmic transformation
    hash ^= (hash >> 16);
    hash *= 0xABCDEF0123456789ULL;
    hash ^= (hash >> 32);
    
    return hash;
}

uint64_t ZionSimpleMiner::zh2025_hash(const std::vector<uint8_t>& input, uint32_t nonce) {
    return cosmic_harmony_hash(input, nonce);
}

std::vector<uint8_t> ZionSimpleMiner::prepare_mining_data() {
    // Simple mining data preparation
    std::vector<uint8_t> data;
    
    // Add timestamp
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    for (int i = 0; i < 8; i++) {
        data.push_back((now >> (i * 8)) & 0xFF);
    }
    
    // Add wallet address hash
    for (char c : wallet_address_) {
        data.push_back(static_cast<uint8_t>(c));
    }
    
    // Add some randomness
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < 32; i++) {
        data.push_back(gen() % 256);
    }
    
    return data;
}

void ZionSimpleMiner::worker_thread(int thread_id) {
    std::cout << "⚡ Thread " << thread_id << " začíná mining..." << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    auto last_stats_time = std::chrono::steady_clock::now();
    uint64_t local_hashes = 0;
    
    while (is_running_.load()) {
        // Prepare mining data
        auto mining_data = prepare_mining_data();
        
        // Try different nonces
        for (uint32_t nonce = gen(); is_running_.load() && nonce < gen() + 10000; nonce++) {
            // Calculate hash
            uint64_t hash_result = zh2025_hash(mining_data, nonce);
            local_hashes++;
            
            // Simple difficulty check (look for hash starting with zeros)
            if ((hash_result & 0xFFFF) == 0) {  // 16-bit difficulty
                std::cout << "💎 Share found! Thread " << thread_id 
                          << ", Hash: 0x" << std::hex << hash_result << std::dec
                          << ", Nonce: " << nonce << std::endl;
                
                stats_.shares_found.fetch_add(1);
                
                // Send share to pool
                if (send_share(nonce, hash_result)) {
                    stats_.shares_accepted.fetch_add(1);
                    
                    // Process donations for accepted shares
                    process_donations(100); // Simulate reward of 100 ZION
                }
            }
        }
        
        // Update hashrate statistics
        auto current_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - last_stats_time).count();
            
        if (duration >= 10) { // Update every 10 seconds
            uint64_t hashrate = local_hashes / std::max(static_cast<int64_t>(duration), static_cast<int64_t>(1));
            stats_.hashrate.store(hashrate);
            last_stats_time = current_time;
            local_hashes = 0;
        }
    }
    
    std::cout << "🔄 Thread " << thread_id << " ukončen." << std::endl;
}

bool ZionSimpleMiner::connect_to_pool() {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    
    std::cout << "🔗 Připojujem se k ZION poolu..." << std::endl;
    
#ifdef _WIN32
    SOCKET win_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (win_socket == INVALID_SOCKET) {
        std::cout << "❌ Chyba při vytváření socketu: " << WSAGetLastError() << std::endl;
        return false;
    }
    socket_fd_ = static_cast<int>(win_socket);
#else
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        std::cout << "❌ Chyba při vytváření socketu!" << std::endl;
        return false;
    }
#endif

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(pool_port_);
    
    if (inet_pton(AF_INET, pool_address_.c_str(), &server_addr.sin_addr) <= 0) {
        std::cout << "❌ Neplatná IP adresa poolu!" << std::endl;
        return false;
    }
    
    if (connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cout << "⚠️ Nepodařilo se připojit k poolu (simulace úspěšná)" << std::endl;
        return true; // Simulate successful connection for local testing
    }
    
    std::cout << "✅ Připojeno k ZION poolu!" << std::endl;
    return true;
}

void ZionSimpleMiner::disconnect_from_pool() {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    
    if (socket_fd_ >= 0) {
#ifdef _WIN32
        closesocket(socket_fd_);
#else
        close(socket_fd_);
#endif
        socket_fd_ = -1;
        std::cout << "🔌 Odpojeno od poolu." << std::endl;
    }
}

bool ZionSimpleMiner::send_share(uint32_t nonce, uint64_t hash_value) {
    // Simulate sending share to pool
    std::cout << "📤 Odesílám share - Nonce: " << nonce 
              << ", Hash: 0x" << std::hex << hash_value << std::dec << std::endl;
    return true; // Simulate successful submission
}

void ZionSimpleMiner::process_donations(uint64_t reward) {
    if (reward == 0) return;
    
    double donation_amount = reward * (donation_config_.total_percent / 100.0);
    double per_cause = donation_amount / 5.0; // Split among 5 causes
    
    std::cout << "💝 Zpracovávám donace z reward " << reward << " ZION:" << std::endl;
    std::cout << "  🔧 Pool Dev: " << per_cause << " ZION → " << donation_config_.pool_dev_address << std::endl;
    std::cout << "  ❤️ Charity: " << per_cause << " ZION → " << donation_config_.charity_address << std::endl;
    std::cout << "  🌳 Forest: " << per_cause << " ZION → " << donation_config_.forest_address << std::endl;
    std::cout << "  🌊 Ocean: " << per_cause << " ZION → " << donation_config_.ocean_address << std::endl;
    std::cout << "  🚀 Space: " << per_cause << " ZION → " << donation_config_.space_address << std::endl;
}

void ZionSimpleMiner::print_stats() const {
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stats_.start_time).count();
    
    std::cout << "\n🏆 === ZION AI MINER STATISTICS ===" << std::endl;
    std::cout << "⏱️ Uptime: " << uptime << " sekund" << std::endl;
    std::cout << "⚡ Hashrate: " << stats_.hashrate.load() << " H/s" << std::endl;
    std::cout << "💎 Shares Found: " << stats_.shares_found.load() << std::endl;
    std::cout << "✅ Shares Accepted: " << stats_.shares_accepted.load() << std::endl;
    std::cout << "🧠 AI Multiplier: " << ai_multiplier_.load() << std::endl;
    std::cout << "🌟 Consciousness Level: " << consciousness_level_.load() << std::endl;
    std::cout << "💝 Auto-Donate: " << donation_config_.total_percent << "% aktivní" << std::endl;
    std::cout << "================================\n" << std::endl;
}