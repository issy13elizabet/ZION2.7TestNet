#include "zion-share-submitter.h"
#include "stratum_client.h"
#include <iostream>
#include <chrono>

void ZionShareSubmitter::run(){
    while(running_.load()){
        ZionShare s; bool have=false;
        // ring first
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t pos = t & (RING_CAP -1);
        Slot& slot = ring_[pos];
        if(slot.full.load(std::memory_order_acquire)){
            s = slot.val; slot.full.store(false, std::memory_order_release); tail_.store(t+1,std::memory_order_release); have=true;
        }
        if(!have){
            std::unique_lock<std::mutex> lk(mutex_);
            if(fallback_.empty()){
                cv_.wait_for(lk, std::chrono::milliseconds(100), [&]{ return !running_.load() || !fallback_.empty(); });
            }
            if(!running_.load()) break;
            if(!fallback_.empty()){ s = fallback_.back(); fallback_.pop_back(); have=true; }
        }
        if(!have) continue;
        bool submitted=false;
        if(stratum_ && stratum_->running()){
            stratum_->submit_share(s.job_id, s.nonce, s.result_hex, s.difficulty);
            submitted=true; // assume ok; real impl would parse response-to-id mapping
        }
        if(submitted){ accepted_.fetch_add(1); std::cout << "[ShareSubmit] ✅ job="<<s.job_id<<" nonce=0x"<<std::hex<<s.nonce<<std::dec<<" diff="<<s.difficulty<<"\n"; }
        else { rejected_.fetch_add(1); std::cout << "[ShareSubmit] ❌ job="<<s.job_id<<" nonce=0x"<<std::hex<<s.nonce<<std::dec<<"\n"; }
    }
}
