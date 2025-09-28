#include "cpu_worker.h"
#include "share_submitter.h"
#include "mining_stats.h"
#ifdef ZION_HAVE_RANDOMX
#include "../../../include/randomx_wrapper.h"
#endif
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <sstream>

using namespace std::chrono_literals;

// Bottleneck notes (pre-optimization):
// 1. Per-hash std::vector<uint8_t> blob_copy allocation + full copy.
// 2. Per-submission std::ostringstream (hex formatting) even when share not accepted yet.
// 3. stats_.add_hashes(1) called every hash (atomic contention).
ZionCpuWorker::ZionCpuWorker(ZionJobManager* jm, ZionShareSubmitter* submitter, const ZionCpuWorkerConfig& cfg, ZionMiningStatsAggregator* stats)
    : job_manager_(jm), submitter_(submitter), cfg_(cfg), agg_(stats) {}

ZionCpuWorker::~ZionCpuWorker(){ stop(); }

void ZionCpuWorker::start(){ if(running_.exchange(true)) return; thread_ = std::thread(&ZionCpuWorker::run, this);} 
void ZionCpuWorker::stop(){ running_.store(false); if(thread_.joinable()) thread_.join(); }

uint32_t ZionCpuWorker::compute_next_nonce(uint64_t base_nonce, unsigned thread_index, unsigned total) const {
    return static_cast<uint32_t>(base_nonce + thread_index);
}

void ZionCpuWorker::apply_affinity(unsigned logical_index){
#ifdef _WIN32
    // Windows: use SetThreadAffinityMask
    HANDLE h = (HANDLE)thread_.native_handle();
    DWORD_PTR mask = (1ull << (logical_index % (sizeof(DWORD_PTR)*8)));
    SetThreadAffinityMask(h, mask);
#else
    cpu_set_t cpuset; CPU_ZERO(&cpuset);
    CPU_SET(logical_index % CPU_SETSIZE, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

void ZionCpuWorker::run(){
    // Pin after thread starts (thread_ already created before run executes)
    if(cfg_.pin_threads){
        apply_affinity(cfg_.index);
    }
    uint64_t local_epoch = 0;
    uint64_t nonce_cursor = cfg_.index * 13; // arbitrary spread
    // Reusable mutable blob buffer (thread-local inside run loop)
    std::vector<uint8_t> mutable_blob;
    size_t cached_blob_size = 0;

    while(running_.load()){
        if(!job_manager_->has_job()) { std::this_thread::sleep_for(200ms); continue; }
        auto job = job_manager_->current_job();
        bool job_changed=false;
        if(job.epoch_id != local_epoch){
            local_epoch = job.epoch_id;
            nonce_cursor = (cfg_.index + 1) * 100003ULL;
            job_changed = true;
#ifdef ZION_HAVE_RANDOMX
            if(cfg_.use_randomx && !job.seed_hash.empty()){
                static thread_local std::string last_seed;
                if(last_seed != job.seed_hash){
                    zion::Hash key; key.fill(0);
                    auto hex_to_byte=[&](char c)->uint8_t{ if(c>='0'&&c<='9') return c-'0'; if(c>='a'&&c<='f') return 10+(c-'a'); if(c>='A'&&c<='F') return 10+(c-'A'); return 0; };
                    size_t out_i=0;
                    for(size_t i=0;i+1<job.seed_hash.size() && out_i<key.size(); i+=2){
                        uint8_t v = (hex_to_byte(job.seed_hash[i])<<4) | hex_to_byte(job.seed_hash[i+1]);
                        key[out_i++] = v;
                    }
                    auto& rx = zion::RandomXWrapper::instance();
                    rx.initialize(key, false);
                    last_seed = job.seed_hash;
                }
            }
#endif
        }
        // If blob size changed or new job, refresh mutable copy once
        if(job_changed || job.blob.size() != cached_blob_size){
            mutable_blob = job.blob; // single copy per job / size change
            cached_blob_size = mutable_blob.size();
        }
        if(mutable_blob.size() < job.nonce_offset + 4){ std::this_thread::sleep_for(50ms); continue; }

        const int batch = 128; // increased batch to amortize overhead
        uint32_t local_hashes = 0; // batch counter
        for(int i=0;i<batch && running_.load();++i){
            uint32_t nonce = compute_next_nonce(nonce_cursor, cfg_.index, cfg_.total_threads);
            nonce_cursor += cfg_.total_threads;
            // Write nonce directly into reusable buffer
            mutable_blob[job.nonce_offset+0] = (nonce >> 0) & 0xFF;
            mutable_blob[job.nonce_offset+1] = (nonce >> 8) & 0xFF;
            mutable_blob[job.nonce_offset+2] = (nonce >> 16) & 0xFF;
            mutable_blob[job.nonce_offset+3] = (nonce >> 24) & 0xFF;
            uint64_t difficulty_value = 0;
            bool submitted=false;
#ifdef ZION_HAVE_RANDOMX
            if(cfg_.use_randomx){
                auto& rx = zion::RandomXWrapper::instance();
                auto h = rx.hash(mutable_blob.data(), mutable_blob.size());
                bool mask_ok=false;
                if(!job.target_mask.empty()){
                    for(int bi=31; bi>=0; --bi){
                        uint8_t hv = h[bi];
                        uint8_t tv = job.target_mask[bi];
                        if(hv < tv){ mask_ok=true; break; }
                        if(hv > tv){ mask_ok=false; break; }
                        if(bi==0) mask_ok=true;
                    }
                }
                if(mask_ok){
                    uint64_t val=0; for(int ii=7; ii>=0; --ii) val = (val<<8) | h[ii];
                    difficulty_value = val? (UINT64_MAX/val) : UINT64_MAX;
                } else if(job.target_difficulty){
                    difficulty_value = rx.hash_to_difficulty(h);
                }
                if( (mask_ok || (job.target_difficulty && difficulty_value >= job.target_difficulty)) && submitter_){
                    static const char* hexchars = "0123456789abcdef";
                    char smallhex[17];
                    for(int b=0;b<8;++b){ uint8_t v = h[b]; smallhex[b*2] = hexchars[v>>4]; smallhex[b*2+1] = hexchars[v & 0xF]; }
                    smallhex[16]='\0';
                    submitter_->enqueue({job.job_id, nonce, std::string(smallhex, 16), difficulty_value});
                    submitted=true;
                }
            }
#else
            difficulty_value = (nonce & 0xFFFF);
            if(difficulty_value % 50000 == 0 && submitter_){
                submitter_->enqueue({job.job_id, nonce, "deadbeef", difficulty_value});
                submitted=true;
            }
#endif
            local_hashes++;
        }
        if(local_hashes) stats_.add_hashes(local_hashes); // single atomic add per batch
        last_instant_hs_ = stats_.current_hs();
        if(agg_) agg_->update_thread_hashrate(cfg_.index, last_instant_hs_);
    }
}
