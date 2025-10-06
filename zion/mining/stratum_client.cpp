#include "stratum_client.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>
#include <array>
#include <algorithm>
#include <fstream>

#include "include/zion-big256.h"

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

struct JsonKV { std::string key; std::string value; bool is_string{false}; };
static void scan_json_flat(const std::string& s, std::vector<JsonKV>& out){
    enum State{Idle,InString,Esc,AfterKey};
    State st=Idle; std::string cur; std::string key; std::string val; bool is_str=false;
    auto push=[&]{ if(!key.empty()){ out.push_back({key,val,is_str}); } key.clear(); val.clear(); is_str=false; };
    for(size_t i=0;i<s.size();++i){ char c=s[i];
        switch(st){
            case Idle: if(c=='"'){ st=InString; cur.clear(); } break;
            case InString:
                if(c=='\\'){ st=Esc; }
                else if(c=='"'){
                    if(key.empty()){ key=cur; st=AfterKey; }
                    else { val=cur; is_str=true; push(); st=Idle; }
                } else cur.push_back(c); break;
            case Esc: cur.push_back(c); st=InString; break;
            case AfterKey:
                if(c==':'){
                    size_t j=i+1; while(j<s.size() && isspace((unsigned char)s[j])) j++; if(j<s.size()){
                        if(s[j]=='"'){ i=j; st=InString; cur.clear(); }
                        else {
                            size_t k=j; while(k<s.size() && s[k]!=',' && s[k] != '}' && s[k] != '\n') k++; val = s.substr(j,k-j);
                            while(!val.empty() && isspace((unsigned char)val.back())) val.pop_back();
                            while(!val.empty() && isspace((unsigned char)val.front())) val.erase(val.begin());
                            is_str=false; push(); i=k; st=Idle;
                        }
                    }
                }
                break;
        }
    }
}

static std::vector<uint8_t> hex_to_bytes(const std::string& hex){
    std::vector<uint8_t> out; out.reserve(hex.size()/2);
    auto hv=[](char c){ if(c>='0'&&c<='9') return c-'0'; if(c>='a'&&c<='f') return 10+(c-'a'); if(c>='A'&&c<='F') return 10+(c-'A'); return 0; };
    for(size_t i=0;i+1<hex.size(); i+=2){ out.push_back((uint8_t)((hv(hex[i])<<4)|hv(hex[i+1]))); }
    return out;
}

StratumClient::StratumClient(ZionJobManager* jm, std::string host, int port, std::string wallet, std::string worker, StratumProtocol proto)
    : job_manager_(jm), host_(std::move(host)), port_(port), wallet_(std::move(wallet)), worker_(std::move(worker)), protocol_(proto)
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
    if(getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &res)!=0){ std::cerr<<"[Stratum] DNS fail"<<std::endl; return false; }
    int s=-1; for(auto p=res; p; p=p->ai_next){ s=(int)::socket(p->ai_family,p->ai_socktype,p->ai_protocol); if(s==-1) continue; if(::connect(s,p->ai_addr,p->ai_addrlen)==0) { sock_fd_=s; break;}
#ifdef _WIN32
        closesocket(s);
#else
        ::close(s);
#endif
        s=-1; }
    freeaddrinfo(res);
    if(sock_fd_<0){ std::cerr<<"[Stratum] Connect fail"<<std::endl; return false; }
    std::cout<<"[Stratum] Connected to "<<host_<<":"<<port_<<" (protocol="<<(protocol_==StratumProtocol::CryptoNote?"CryptoNote":"Stratum")<<")\n";
    if(protocol_==StratumProtocol::CryptoNote){
        send_login();
    } else {
        send_json("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\""+worker_+"\"]}");
        send_json("{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\""+wallet_+"\",\"x\"]}");
    }
    return true;
}

void StratumClient::io_loop(){
    std::string buffer; buffer.reserve(8192);
    while(running_.load()){
        char temp[1024]; int n=0; if(sock_fd_>=0){
#ifdef _WIN32
            n=recv(sock_fd_, temp, sizeof(temp), 0);
#else
            n=(int)::recv(sock_fd_, temp, sizeof(temp), 0);
#endif
        }
        if(n<=0){ std::cerr<<"[Stratum] Disconnected, retry 5s"<<std::endl; std::this_thread::sleep_for(5s); connect_socket(); continue; }
        buffer.append(temp, n);
        size_t pos; while((pos=buffer.find('\n'))!=std::string::npos){ std::string line=buffer.substr(0,pos); buffer.erase(0,pos+1); if(!line.empty()) handle_line(line); }
    }
}

void StratumClient::handle_line(const std::string& line){
    if(verbose_.load()){
        std::cout << "[Stratum RAW] " << line << std::endl;
        std::ofstream f("stratum-raw.log", std::ios::app); if(f) f<<line<<"\n";
    }
    // Extract numeric id (pro mapování shares)
    int msg_id=-1; {
        size_t pid=line.find("\"id\""); if(pid!=std::string::npos){ pid=line.find(':',pid); if(pid!=std::string::npos){ ++pid; while(pid<line.size() && isspace((unsigned char)line[pid])) ++pid; size_t s=pid; while(pid<line.size() && isdigit((unsigned char)line[pid])) ++pid; if(pid>s){ msg_id=std::stoi(line.substr(s,pid-s)); } } }
    }
    bool has_job_marker = (line.find("job")!=std::string::npos && line.find("blob")!=std::string::npos);
    bool has_result = (line.find("result")!=std::string::npos);
    // Share submit ack (CryptoNote): result object with status OK/INVALID and known pending id, no job
    if(has_result && !has_job_marker && msg_id>0){
        std::lock_guard<std::mutex> lk(pending_mutex_);
        auto it = pending_.find(msg_id);
        if(it!=pending_.end()){
            std::string lower=line; for(char &c: lower) c=(char)tolower((unsigned char)c);
            if(lower.find("ok")!=std::string::npos) accepted_++; else if(lower.find("invalid")!=std::string::npos || lower.find("false")!=std::string::npos) rejected_++; else {
                // fallback heuristika
                if(lower.find("true")!=std::string::npos) accepted_++; else if(lower.find("false")!=std::string::npos) rejected_++; }
            pending_.erase(it);
            return;
        }
    }
    // Login / getjob styl (result.job)
    if(has_result && has_job_marker){
        parse_job_notification(line);
        return;
    }
    // Notify fallback
    if(line.find("notify")!=std::string::npos){
        parse_job_notification(line);
    }
}

void StratumClient::parse_job_notification(const std::string& json_text){
    std::string job_id, blob_hex, target_hex, seed_hash;
    std::string extranonce_hex;
    // Pokus o extrakci z result.job (login/getjob)
    if(json_text.find("result")!=std::string::npos && json_text.find("job")!=std::string::npos){
        auto grab=[&](const char* key)->std::string{
            std::string k="\""; k+=key; k+="\""; size_t p=json_text.find(k); if(p==std::string::npos) return {}; p=json_text.find(':',p); if(p==std::string::npos) return {}; while(p<json_text.size() && json_text[p] != '"') p++; if(p>=json_text.size()) return {}; size_t s=++p; while(p<json_text.size() && json_text[p] != '"') p++; if(p>=json_text.size()) return {}; return json_text.substr(s,p-s); };
        job_id = grab("job_id");
        blob_hex = grab("blob");
        target_hex = grab("target");
        seed_hash = grab("seed_hash");
        auto sid = grab("id"); if(!sid.empty()) session_id_ = sid;
    auto extran = grab("extranonce"); if(!extran.empty()){ extranonce_ = extran; extranonce_hex = extran; }
    }
    // Fallback params array styl
    if(blob_hex.empty()){
        auto ppos = json_text.find("[", json_text.find("params"));
        if(ppos!=std::string::npos){
            size_t end = json_text.find(']', ppos); if(end!=std::string::npos){ std::string arr=json_text.substr(ppos+1,end-ppos-1); std::vector<std::string> fields; fields.reserve(8); std::string cur; bool in_str=false; bool esc=false; for(char c: arr){ if(in_str){ if(esc){ cur.push_back(c); esc=false; } else if(c=='\\'){ esc=true; } else if(c=='"'){ in_str=false; fields.push_back(cur); cur.clear(); } else cur.push_back(c);} else { if(c=='"') in_str=true; } } if(fields.size()>=4){ job_id=fields[0]; blob_hex=fields[1]; target_hex=fields[2]; seed_hash=fields[3]; } }
        }
    }
    if(blob_hex.size()<80) return;
    // Parsování target (pokud 64 hex = 256-bit big endian)
    uint64_t target_val = 0; std::array<uint8_t,32> target_mask{}; target_mask.fill(0);
    if(!target_hex.empty()){
        std::string t = target_hex; if(t.rfind("0x",0)==0 || t.rfind("0X",0)==0) t = t.substr(2);
        // pokud délka >= 64, zkusíme naplnit 32B big-endian
        if(t.size() >= 64){
            if(t.size()>64) t = t.substr(t.size()-64); // vezmi posledních 64 (nižší bity)
            for(size_t i=0;i<32;i++){
                auto hexbyte = t.substr(i*2,2);
                uint8_t val=0; for(char c: hexbyte){ val <<=4; if(c>='0'&&c<='9') val|=c-'0'; else if(c>='a'&&c<='f') val|=10+(c-'a'); else if(c>='A'&&c<='F') val|=10+(c-'A'); }
                target_mask[i]=val;
            }
            // pro 64-bit fallback vezmeme posledních 16 hex (nižších 64 bitů)
            std::string low64 = t.substr(48); // posledních 16
            for(char c: low64){ target_val <<=4; if(c>='0'&&c<='9') target_val|=(c-'0'); else if(c>='a'&&c<='f') target_val|=(10+(c-'a')); else if(c>='A'&&c<='F') target_val|=(10+(c-'A')); }
            if(target_val==0) target_val=0xFFFFFFFFFFFFULL;
        } else {
            // kratší než 64 => použij původní 64-bit metodu
            if(t.size()>16) t = t.substr(t.size()-16);
            for(char c: t){ target_val <<=4; if(c>='0'&&c<='9') target_val|=(c-'0'); else if(c>='a'&&c<='f') target_val|=(10+(c-'a')); else if(c>='A'&&c<='F') target_val|=(10+(c-'A')); }
            if(target_val==0) target_val=0xFFFFFFFFFFFFULL;
        }
    } else { target_val=0xFFFFFFFFFFFFULL; }
    ZionMiningJob job; job.job_id=job_id; job.blob=hex_to_bytes(blob_hex); job.seed_hash=seed_hash; job.target_difficulty=target_val; job.nonce_offset=39; job.epoch_id=0; job.target_mask = target_mask;
    // Compute difficulty_display + difficulty_24
    if(target_mask[0] || target_mask[1]){
        // 256-bit path
        ZionBig256 t = ZionBig256::from_be_bytes(target_mask.data());
        uint64_t d = zion_difficulty_from_target(t);
        job.difficulty_display = (double)d;
    } else if(target_val){
        // Treat target_val as lower bits (e.g. 24-bit in 00ffffff). Diff24 baseline = 0x00ffffff / target_val
        double base24 = (double)0x00FFFFFFu;
        if(target_val <= 0x00FFFFFFull){ job.difficulty_24 = base24 / (double)target_val; job.difficulty_display = job.difficulty_24; }
        else { job.difficulty_display = (double) ( (double)0xFFFFFFFFFFFFULL / (double)target_val ); }
    }
    // Insert extranonce if provided and blob is long enough. Heuristic: place at nonce_offset-8 if space.
    if(!extranonce_hex.empty()){
        auto exb = hex_to_bytes(extranonce_hex);
        // Avoid overrunning header size (80B assumption). If blob shorter just skip.
        if(job.blob.size() >= job.nonce_offset){
            size_t ex_space = (job.nonce_offset>8)? 8 : job.nonce_offset; // reserve up to 8 bytes before nonce for extranonce
            size_t insert_pos = job.nonce_offset - ex_space;
            for(size_t i=0;i<ex_space && i<exb.size(); ++i){ job.blob[insert_pos + i] = exb[i]; }
        }
    }
    job_manager_->update_job(job);
    std::cout<<"[Stratum] New job: "<<job.job_id<<" blob="<<blob_hex.substr(0,16)<<".. t64=0x"<<std::hex<<target_val<<std::dec
             <<" t256="<<(target_mask[0]||target_mask[1]?"yes":"no")
             <<" diff="<< (job.difficulty_display>0? job.difficulty_display:0.0)
             << (job.difficulty_24>0? std::string(" diff24=")+std::to_string(job.difficulty_24):"")
             << std::endl;
}

void StratumClient::send_login(){
    int id = id_counter_++;
    std::ostringstream oss;
    oss << "{\"id\":"<<id<<",\"method\":\"login\",\"params\":{\"login\":\""<<wallet_<<"\",\"pass\":\""<<worker_<<"\",\"agent\":\"zion-miner/1.4\"}}";
    send_json(oss.str());
}

void StratumClient::submit_share(const std::string& job_id, uint32_t nonce, const std::string& result_hex, uint64_t difficulty_value){
    int id=id_counter_++;
    std::ostringstream oss;
    if(protocol_==StratumProtocol::CryptoNote){
        // nonce jako 8 hex znaků (lowercase)
        oss<<"{\"id\":"<<id<<",\"method\":\"submit\",\"params\":{\"id\":\""<< (session_id_.empty()?"sess":session_id_) <<"\",\"job_id\":\""<<job_id<<"\",\"nonce\":\""<<std::hex<<std::setw(8)<<std::setfill('0')<<nonce<<"\",\"result\":\""<<result_hex<<"\"}}";
    } else {
        oss<<"{\"id\":"<<id<<",\"method\":\"mining.submit\",\"params\":[\""<<wallet_<<"\",\""<<job_id<<"\",\""<<std::hex<<std::setw(8)<<std::setfill('0')<<nonce<<"\",\""<<result_hex<<"\"]}";
    }
    {
        std::lock_guard<std::mutex> lk(pending_mutex_);
        pending_[id] = PendingShare{job_id, nonce, std::chrono::steady_clock::now()};
    }
    send_json(oss.str());
}

void StratumClient::send_json(const std::string& json_line){
    if(sock_fd_<0) return; std::string line=json_line+"\n";
#ifdef _WIN32
    send(sock_fd_, line.c_str(), (int)line.size(), 0);
#else
    ::send(sock_fd_, line.c_str(), line.size(), 0);
#endif
}
