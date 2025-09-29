// Stratum client implementation
#include "stratum_client.h"
#include "job_manager.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread>
#include <array>
#include <algorithm>

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

using namespace std::chrono_literals;

static std::vector<uint8_t> hex_to_bytes(const std::string& hex){
    std::vector<uint8_t> out; out.reserve(hex.size()/2);
    auto hv=[](char c){ if(c>='0'&&c<='9') return c-'0'; if(c>='a'&&c<='f') return 10+(c-'a'); if(c>='A'&&c<='F') return 10+(c-'A'); return 0; };
    for(size_t i=0;i+1<hex.size(); i+=2){ out.push_back((hv(hex[i])<<4)|hv(hex[i+1])); }
    return out;
}

StratumClient::StratumClient(ZionJobManager* jm, std::string host, int port, std::string wallet, std::string worker)
    : job_manager_(jm), host_(std::move(host)), port_(port), wallet_(std::move(wallet)), worker_(std::move(worker))
{
#ifdef _WIN32
    WSADATA wsaData; WSAStartup(MAKEWORD(2,2), &wsaData);
#endif
}

StratumClient::~StratumClient(){ stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool StratumClient::start(){ if(running_.exchange(true)) return true; if(!connect_socket()){ running_=false; return false;} io_thread_=std::thread(&StratumClient::io_loop,this); return true; }
void StratumClient::stop(){ running_.store(false); if(io_thread_.joinable()) io_thread_.join(); if(sock_fd_!=-1){
#ifdef _WIN32
    closesocket(sock_fd_);
#else
    ::close(sock_fd_);
#endif
    sock_fd_=-1; }
}

bool StratumClient::connect_socket(){
    struct addrinfo hints{}; hints.ai_family=AF_UNSPEC; hints.ai_socktype=SOCK_STREAM; struct addrinfo* res=nullptr; std::string port_str=std::to_string(port_);
    if(getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &res)!=0){ std::cerr<<"Stratum DNS fail"<<std::endl; return false; }
    int s=-1; for(auto p=res; p; p=p->ai_next){ s=::socket(p->ai_family,p->ai_socktype,p->ai_protocol); if(s==-1) continue; if(::connect(s,p->ai_addr,p->ai_addrlen)==0) { sock_fd_=s; break;} 
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
    s=-1; }
    freeaddrinfo(res);
    if(sock_fd_<0){ std::cerr<<"Stratum connect fail"<<std::endl; return false; }
    std::cout<<"[Stratum] Connected to "<<host_<<":"<<port_<<"\n";
    // Send mining.subscribe & mining.authorize
    send_json("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\""+worker_+"\"]}");
    send_json("{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\""+wallet_+"\",\"x\"]}");
    return true;
}

void StratumClient::io_loop(){
    std::string buffer; buffer.reserve(8192);
    while(running_.load()){
        char temp[1024];
        int n=0;
        if(sock_fd_>=0){
#ifdef _WIN32
            n=recv(sock_fd_, temp, sizeof(temp), 0);
#else
            n=::recv(sock_fd_, temp, sizeof(temp), 0);
#endif
        }
        if(n<=0){
            std::cerr<<"[Stratum] Disconnected, retry in 5s"<<std::endl;
            std::this_thread::sleep_for(5s);
            connect_socket();
            continue;
        }
        buffer.append(temp, n);
        size_t pos;
        while((pos=buffer.find('\n'))!=std::string::npos){
            std::string line=buffer.substr(0,pos); buffer.erase(0,pos+1);
            if(!line.empty()) handle_line(line);
        }
    }
}

// --- Lightweight JSON scanning helpers (sufficient for Stratum basic usage) ---
struct JsonKV { std::string key; std::string value; bool is_string{false}; };
static void scan_json_flat(const std::string& s, std::vector<JsonKV>& out){
    enum State{Idle,InString,Esc,AfterKey,InValueString,InValueRaw};
    State st=Idle; std::string cur; std::string key; bool collecting=false; bool is_str=false; std::string val;
    auto push=[&]{ if(!key.empty()){ out.push_back({key,val,is_str}); } key.clear(); val.clear(); };
    for(size_t i=0;i<s.size();++i){ char c=s[i];
        switch(st){
            case Idle:
                if(c=='"'){ st=InString; cur.clear(); }
                break;
            case InString:
                if(c=='\\'){ st=Esc; }
                else if(c=='"'){ // string ended
                    if(key.empty()) { key=cur; st=AfterKey; }
                    else { val=cur; is_str=true; push(); st=Idle; }
                } else cur.push_back(c);
                break;
            case Esc: cur.push_back(c); st=InString; break;
            case AfterKey:
                if(c==':'){ // decide value type next
                    // lookahead first non-space
                    size_t j=i+1; while(j<s.size() && isspace((unsigned char)s[j])) j++; if(j<s.size()){
                        if(s[j]=='"'){ i=j; st=InString; cur.clear(); }
                        else { // raw value until , or }
                            size_t k=j; while(k<s.size() && s[k]!=',' && s[k] != '}' && s[k] != '\n') k++; val = s.substr(j,k-j); // trim
                            // trim whitespace
                            while(!val.empty() && isspace((unsigned char)val.back())) val.pop_back();
                            while(!val.empty() && isspace((unsigned char)val.front())) val.erase(val.begin());
                            is_str=false; push(); i=k; st=Idle; }
                    }
                }
                break;
            default: break;
        }
    }
}

void StratumClient::handle_line(const std::string& line){
    // Detect share result ("result":true/false and "id":X without method)
    if(line.find("method")==std::string::npos && line.find("result")!=std::string::npos){
        std::vector<JsonKV> kv; kv.reserve(16); scan_json_flat(line, kv);
        for(auto& e: kv){
            if(e.key=="result"){
                std::string v=e.value; for(auto& ch: v) ch= (char)tolower((unsigned char)ch);
                if(v.find("true")!=std::string::npos || v=="1") accepted_++; else if(v.find("false")!=std::string::npos || v=="0") rejected_++;
            }
        }
        return;
    }
    // Job notification: must contain method:"mining.notify" or params array with blob
    if(line.find("notify")!=std::string::npos || (line.find("job")!=std::string::npos && line.find("blob")!=std::string::npos)){
        parse_job_notification(line);
    }
}

void StratumClient::parse_job_notification(const std::string& json_text){
    // First try to find params array; if present extract ordered fields typical for CryptoNote pools
    // Common pattern: {"method":"mining.notify","params":["job_id","blob","target","seed","..."],"id":null}
    auto ppos = json_text.find("[", json_text.find("params"));
    std::string job_id, blob_hex, target_hex, seed_hash;
    if(ppos!=std::string::npos){
        // naive split of first params array until closing ]
        size_t end = json_text.find(']', ppos);
        if(end!=std::string::npos){
            std::string arr = json_text.substr(ppos+1, end-ppos-1);
            std::vector<std::string> fields; fields.reserve(8);
            std::string cur; bool in_str=false; bool esc=false;
            for(char c: arr){
                if(in_str){
                    if(esc){ cur.push_back(c); esc=false; }
                    else if(c=='\\'){ esc=true; }
                    else if(c=='"'){ in_str=false; fields.push_back(cur); cur.clear(); }
                    else cur.push_back(c);
                } else {
                    if(c=='"'){ in_str=true; }
                }
            }
            if(fields.size()>=4){ job_id=fields[0]; blob_hex=fields[1]; target_hex=fields[2]; seed_hash=fields[3]; }
        }
    }
    // Fallback: scan key-value for named entries (some pools send object form)
    if(blob_hex.empty()){
        std::vector<JsonKV> kv; kv.reserve(32); scan_json_flat(json_text, kv);
        auto get=[&](const char* k){ for(auto& e: kv) if(e.key==k) return e.value; return std::string(); };
        if(job_id.empty()) job_id = get("job_id");
        if(blob_hex.empty()) blob_hex = get("blob");
        if(seed_hash.empty()) seed_hash = get("seed_hash");
        if(target_hex.empty()) target_hex = get("target");
    }
    if(blob_hex.size()<80) return; // invalid
    StratumJobData jd; jd.job_id=job_id; jd.blob_hex=blob_hex; jd.seed_hash=seed_hash; jd.target_difficulty=0; // numeric only fallback
    std::array<uint8_t,32> target_mask{}; target_mask.fill(0);
    if(!target_hex.empty()){
        // Normalize hex (remove 0x)
        std::string hex = target_hex;
        if(hex.rfind("0x",0)==0 || hex.rfind("0X",0)==0) hex = hex.substr(2);
        // CryptoNote target is little-endian 256-bit value represented as hex LE (pool-dependent). We accept either 64 hex chars (big-endian) or 64 LE.
        // Strategy: if length <=16 treat as numeric difficulty; if length >= 64 treat as mask.
        if(hex.size() <= 16){
            try { jd.target_difficulty = std::stoull(hex, nullptr, 16); } catch(...) { jd.target_difficulty=0; }
        } else if(hex.size() >= 64){
            // pad to 64 even chars
            if(hex.size()>64) hex = hex.substr(0,64); else if(hex.size()<64) hex = std::string(64-hex.size(),'0') + hex;
            // interpret hex as little-endian mask: pairs -> bytes
            auto hv=[&](char c){ if(c>='0'&&c<='9') return c-'0'; if(c>='a'&&c<='f') return 10+(c-'a'); if(c>='A'&&c<='F') return 10+(c-'A'); return 0; };
            // Some pools send big-endian; heuristic: if most significant nibble is 0, assume big-endian and reverse at end.
            std::array<uint8_t,32> tmp{}; tmp.fill(0);
            for(size_t i=0;i<64;i+=2){ tmp[i/2] = (hv(hex[i])<<4) | hv(hex[i+1]); }
            // Heuristic: if last 16 bytes are zero heavy and first 16 non-zero, assume little-endian already. Else reverse.
            int leading_nonzero=0; for(int i=0;i<16;i++) if(tmp[31-i]!=0) { leading_nonzero++; }
            int trailing_nonzero=0; for(int i=0;i<16;i++) if(tmp[i]!=0) { trailing_nonzero++; }
            bool looks_big_endian = leading_nonzero > trailing_nonzero; // crude
            if(looks_big_endian){ for(int i=0;i<16;i++){ std::swap(tmp[i], tmp[31-i]); } }
            target_mask = tmp;
        }
    }
    auto blob_bytes = hex_to_bytes(jd.blob_hex);
    ZionMiningJob job; job.job_id=jd.job_id; job.blob=blob_bytes; job.seed_hash=jd.seed_hash; job.target_difficulty= jd.target_difficulty? jd.target_difficulty:0; job.nonce_offset=39; job.epoch_id=0; job.target_mask = target_mask;
    job_manager_->update_job(job);
    std::ostringstream extra;
    bool has_mask=false; for(auto b: target_mask){ if(b){ has_mask=true; break; } }
    if(has_mask){ extra << " mask-set"; }
    std::cout << "[Stratum] New job: "<< job.job_id << (job.target_difficulty? (std::string(" diff=")+std::to_string(job.target_difficulty)) : "") << extra.str() << " seed=" << (job.seed_hash.size()>=8? job.seed_hash.substr(0,8): job.seed_hash) << ".." << std::endl;
}

void StratumClient::submit_share(const std::string& job_id, uint32_t nonce, const std::string& result_hex, uint64_t difficulty_value){
    // For now send simplified share submit (needs alignment with actual pool protocol)
    std::ostringstream oss; int id=id_counter_++;
    oss << "{\"id\":"<<id<<",\"method\":\"mining.submit\",\"params\":[\""<<wallet_<<"\",\""<<job_id<<"\",\""<<std::hex<<std::setw(8)<<std::setfill('0')<<nonce<<"\",\""<<result_hex<<"\"]}"; 
    send_json(oss.str());
}

void StratumClient::send_json(const std::string& json_line){
    std::string line = json_line + "\n";
    if(sock_fd_<0) return;
#ifdef _WIN32
    send(sock_fd_, line.c_str(), (int)line.size(), 0);
#else
    ::send(sock_fd_, line.c_str(), line.size(), 0);
#endif
}
