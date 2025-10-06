#include "job_manager.h"
#include <algorithm>

ZionJobManager::ZionJobManager() = default;

void ZionJobManager::update_job(const ZionMiningJob& new_job) {
    std::lock_guard<std::mutex> lk(mutex_);
    job_ = new_job; // copy
    job_.epoch_id = current_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1; // increment then store in struct
    has_job_.store(true, std::memory_order_release);
}

ZionMiningJob ZionJobManager::current_job() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return job_; // copy snapshot
}
