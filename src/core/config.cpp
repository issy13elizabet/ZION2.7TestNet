#include "config.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace zion {

static bool parse_hex_64(const std::string& hex, PublicKey& out) {
    // Expect 64 hex chars
    if (hex.size() != 64) return false;
    auto hexval = [](char c)->int{
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        int hi = hexval(hex[2*i]);
        int lo = hexval(hex[2*i+1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

bool load_node_config(const std::string& path, NodeConfig& out, std::string& error) {
    std::ifstream f(path);
    if (!f) { error = "Cannot open config: " + path; return false; }

    std::string line; std::string section;
    while (std::getline(f, line)) {
        // trim
        auto ltrim = [](std::string& s){ s.erase(0, s.find_first_not_of(" \t")); };
        auto rtrim = [](std::string& s){ s.erase(s.find_last_not_of(" \t") + 1); };
        ltrim(line); rtrim(line);
        if (line.empty() || line[0] == '#') continue;
        if (line.front() == '[' && line.back() == ']') { section = line.substr(1, line.size()-2); continue; }
        auto eq = line.find('='); if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq); std::string val = line.substr(eq+1);
        ltrim(key); rtrim(key); ltrim(val); rtrim(val);
        // strip quotes
        if (!val.empty() && (val.front()=='"' || val.front()=='\'')) {
            if (val.back()==val.front()) { val = val.substr(1, val.size()-2); }
        }
        // Interested in [blockchain] and [network]
        if (section == "blockchain") {
            if (key == "genesis_timestamp") {
                try { out.genesis_timestamp = std::stoull(val); } catch (...) {}
            } else if (key == "genesis_difficulty") {
                try { out.genesis_difficulty = static_cast<uint32_t>(std::stoul(val)); } catch (...) {}
            } else if (key == "genesis_coinbase_address") {
                PublicKey pk{};
                if (parse_hex_64(val, pk)) out.genesis_coinbase_address = pk; // else leave default
            } else if (key == "genesis_hash") {
                out.expected_genesis_hash = val;
            }
        } else if (section == "network") {
            if (key == "network_id") out.network_id = val;
            else if (key == "chain_id") { try { out.chain_id = std::stoi(val); } catch (...) {} }
            else if (key == "p2p_port") { try { out.p2p_port = static_cast<uint16_t>(std::stoul(val)); } catch (...) {} }
            else if (key == "rpc_port") { try { out.rpc_port = static_cast<uint16_t>(std::stoul(val)); } catch (...) {} }
            else if (key == "seed_nodes") {
                // seed_nodes = ["host:port", "host:port", ...]
                // jednoduchý parser: vytáhnout položky mezi uvozovkami
                out.seed_nodes.clear();
                std::string s = val;
                // odstranění hranatých závorek
                if (!s.empty() && s.front()=='[' && s.back()==']') s = s.substr(1, s.size()-2);
                std::string item;
                std::istringstream iss(s);
                while (std::getline(iss, item, ',')) {
                    // trim
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    if (!item.empty() && (item.front()=='"' || item.front()=='\'')) {
                        if (item.back()==item.front()) item = item.substr(1, item.size()-2);
                    }
                    if (!item.empty()) out.seed_nodes.push_back(item);
                }
            }
        } else if (section == "pool") {
            if (key == "enable_pool") out.pool_enable = (val == "true" || val == "1");
            else if (key == "pool_port") { try { out.pool_port = static_cast<uint16_t>(std::stoul(val)); } catch (...) {} }
            else if (key == "pool_require_auth") out.pool_require_auth = (val == "true" || val == "1");
            else if (key == "pool_password") out.pool_password = val;
        } else if (section == "mempool") {
            if (key == "mempool_min_fee") { try { out.mempool_min_fee = std::stoull(val); } catch (...) {} }
        }
    }
    return true;
}

bool load_json_config(const std::string& path, NodeConfig& out, std::string& error) {
    std::ifstream f(path);
    if (!f) { error = "Cannot open config: " + path; return false; }
    
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string content = buffer.str();
    
    // Jednoduchý JSON parser pro naše klíče
    auto find_value = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return "";
        
        pos = content.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++;
        
        // Přeskočit whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n')) pos++;
        
        if (pos >= content.size()) return "";
        
        if (content[pos] == '"') {
            // String hodnota
            pos++;
            size_t end = content.find('"', pos);
            if (end == std::string::npos) return "";
            return content.substr(pos, end - pos);
        } else if (content[pos] == 't' || content[pos] == 'f') {
            // Boolean hodnota
            if (content.substr(pos, 4) == "true") return "true";
            if (content.substr(pos, 5) == "false") return "false";
            return "";
        } else {
            // Číselná hodnota
            size_t end = pos;
            while (end < content.size() && (isdigit(content[end]) || content[end] == '.')) end++;
            return content.substr(pos, end - pos);
        }
    };
    
    // Načíst hodnoty
    std::string val;
    
    val = find_value("p2p_port");
    if (!val.empty()) { try { out.p2p_port = static_cast<uint16_t>(std::stoul(val)); } catch (...) {} }
    
    val = find_value("rpc_port");
    if (!val.empty()) { try { out.rpc_port = static_cast<uint16_t>(std::stoul(val)); } catch (...) {} }
    
    val = find_value("pool_enable");
    if (!val.empty()) { out.pool_enable = (val == "true"); }
    
    val = find_value("pool_port");
    if (!val.empty()) { try { out.pool_port = static_cast<uint16_t>(std::stoul(val)); } catch (...) {} }
    
    val = find_value("pool_difficulty");
    if (!val.empty()) { try { out.genesis_difficulty = static_cast<uint32_t>(std::stoul(val)); } catch (...) {} }
    
    return true;
}

} // namespace zion
