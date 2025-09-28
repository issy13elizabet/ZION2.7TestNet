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

ZionCpuWorker::ZionCpuWorker(ZionJobManager* jm, ZionShareSubmitter* submitter, const ZionCpuWorkerConfig& cfg, ZionMiningStatsAggregator* stats)
    : job_manager_(jm), submitter_(submitter), cfg_(cfg), agg_(stats) {}

ZionCpuWorker::~ZionCpuWorker(){ stop(); }

void ZionCpuWorker::start(){ if(running_.exchange(true)) return; thread_ = std::thread(&ZionCpuWorker::run, this);} 
void ZionCpuWorker::stop(){ running_.store(false); if(thread_.joinable()) thread_.join(); }

uint32_t ZionCpuWorker::compute_next_nonce(uint64_t base_nonce, unsigned thread_index, unsigned total) const {
    return static_cast<uint32_t>(base_nonce + thread_index);
}

void ZionCpuWorker::run(){
    uint64_t local_epoch = 0;
    uint64_t nonce_cursor = cfg_.index * 13; // arbitrary spread

    while(running_.load()){
        if(!job_manager_->has_job()) { std::this_thread::sleep_for(200ms); continue; }
        auto job = job_manager_->current_job();
        if(job.epoch_id != local_epoch){
            local_epoch = job.epoch_id;
            // Reset per-job nonce base for this worker
            nonce_cursor = (cfg_.index + 1) * 100003ULL; // distinct starting space
#ifdef ZION_HAVE_RANDOMX
            // Reinitialize RandomX if seed hash changed (job.seed_hash expected hex)
            if(cfg_.use_randomx && !job.seed_hash.empty()){
                static thread_local std::string last_seed;
                if(last_seed != job.seed_hash){
                    // Convert hex seed to 32-byte key (truncate or pad)
                    zion::Hash key; key.fill(0);
                    auto hex_to_byte=[&](char c)->uint8_t{ if(c>='0'&&c<='9') return c-'0'; if(c>='a'&&c<='f') return 10+(c-'a'); if(c>='A'&&c<='F') return 10+(c-'A'); return 0; };
                    size_t out_i=0;
                    for(size_t i=0;i+1<job.seed_hash.size() && out_i<key.size(); i+=2){
                        uint8_t v = (hex_to_byte(job.seed_hash[i])<<4) | hex_to_byte(job.seed_hash[i+1]);
                        key[out_i++] = v;
                    }
                    auto& rx = zion::RandomXWrapper::instance();
                    rx.initialize(key, false); // light mode per job seed
                    last_seed = job.seed_hash;
                }
            }
#endif
        }
        // produce a batch of nonces
        const int batch = 64;
        for(int i=0;i<batch && running_.load();++i){
            uint32_t nonce = compute_next_nonce(nonce_cursor, cfg_.index, cfg_.total_threads);
            nonce_cursor += cfg_.total_threads; // stride

            // Insert nonce into blob copy
            if(job.blob.size() < job.nonce_offset + 4) continue; // safety
            std::vector<uint8_t> blob_copy = job.blob;
            blob_copy[job.nonce_offset+0] = (nonce >> 0) & 0xFF;
            blob_copy[job.nonce_offset+1] = (nonce >> 8) & 0xFF;
            blob_copy[job.nonce_offset+2] = (nonce >> 16) & 0xFF;
            blob_copy[job.nonce_offset+3] = (nonce >> 24) & 0xFF;

            uint64_t difficulty_value = 0;
#ifdef ZION_HAVE_RANDOMX
            auto compare_hash=[&](const zion::Hash& h, const std::array<uint8_t,32>& target_mask)->bool{
                // If mask all zero -> use numeric fallback
                bool any=false; for(auto b: target_mask) if(b){ any=true; break; }
                if(!any) return false; // can't decide here
                // Hash produced is big-endian? RandomX returns 32 bytes (little-endian internal). We'll treat it as little-endian integer.
                // Compare h <= target_mask (both little-endian). We'll convert to little-endian arrays if needed.
                // If target_mask is little-endian, compare from most significant byte (index 31) down.
                for(int i=31;i>=0;--i){
                    uint8_t hv = h[i];
                    uint8_t tv = target_mask[i];
                    if(hv < tv) return true;
                    if(hv > tv) return false;
                }
                return true; // equal
            };
#endif
            bool submitted=false;
#ifdef ZION_HAVE_RANDOMX
            if(cfg_.use_randomx){
                auto& rx = zion::RandomXWrapper::instance();
                auto h = rx.hash(blob_copy.data(), blob_copy.size());
                // Compute numeric diff only if no mask; else approximate difficulty from first 8B for UI
                bool mask_ok=false;
                if(!job.target_mask.empty()){
                    mask_ok = compare_hash(h, job.target_mask);
                }
                if(mask_ok){
                    // approximate difficulty: inverse of first 8 bytes
                    uint64_t val=0; for(int i=7;i>=0;--i) val = (val<<8) | h[i];
                    difficulty_value = val? (UINT64_MAX/val) : UINT64_MAX;
                } else if(job.target_difficulty){
                    difficulty_value = rx.hash_to_difficulty(h);
                }
                std::ostringstream oss; oss<<std::hex; for(int b=0;b<8;++b) oss<<std::setw(2)<<std::setfill('0')<<(int)h[b];
                if( (mask_ok || (job.target_difficulty && difficulty_value >= job.target_difficulty)) && submitter_){
                    submitter_->enqueue({job.job_id, nonce, oss.str(), difficulty_value});
                    submitted=true;
                }
            }
#endif
            if(!submitted){
#ifndef ZION_HAVE_RANDOMX
            // Stub fallback: simulate difficulty progression
            difficulty_value = (nonce & 0xFFFF);
            if(difficulty_value % 50000 == 0 && submitter_){
                submitter_->enqueue({job.job_id, nonce, "deadbeef", difficulty_value});
            }
            }
#endif // stub region
            stats_.add_hashes(1);
        }
        // rate calculation / throttle a bit
        last_instant_hs_ = stats_.current_hs();
        if(agg_) agg_->update_thread_hashrate(cfg_.index, last_instant_hs_);
    }
}
