#include <iostream>
#include <vector>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>
#include "../mining/core/job_manager.h"
#include "../mining/core/cpu_worker.h"
#include "../mining/core/share_submitter.h"
#include "../mining/core/mining_stats.h"
#include "../network/pool-connection.h"
#include "../network/stratum_client.h"
#ifdef ZION_HAVE_RANDOMX
#include "../../include/randomx_wrapper.h"
#endif

static std::atomic<bool> g_running{true};

void signal_handler(int){ g_running.store(false); }

int main(int argc, char* argv[]){
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string wallet = argc>1? argv[1]: "Z3ExampleWalletAddress";
    std::string pool_host = argc>2? argv[2]: "127.0.0.1";
    int pool_port = argc>3? std::atoi(argv[3]) : 3333;
    bool disable_stratum = false;
    for(int i=1;i<argc;i++){ if(std::string(argv[i]) == "--no-stratum") disable_stratum = true; }

    std::cout << "ZION Miner Core (CPU RandomX)\n";
    std::cout << "Wallet: " << wallet << "\nPool:   " << pool_host << ':' << pool_port << "\n";

#ifdef ZION_HAVE_RANDOMX
    // Initialize RandomX with dummy seed (replace with real seed hash parsing from job)
    zion::Hash dummy_seed; dummy_seed.fill(0x11);
    if(!zion::RandomXWrapper::instance().initialize(dummy_seed, false)){
        std::cerr << "Failed to init RandomX – exiting" << std::endl; return 1; }
#else
    std::cout << "RandomX not available (library missing) – running stub mode." << std::endl;
#endif

    PoolConnection pool(pool_host, pool_port, wallet);
    pool.connect();

    ZionJobManager job_manager;
    std::unique_ptr<StratumClient> stratum;
    if(!disable_stratum){
        stratum.reset(new StratumClient(&job_manager, pool_host, pool_port, wallet));
        if(!stratum->start()){
            std::cerr << "Failed to start Stratum client, falling back to bootstrap dummy job" << std::endl;
        }
    }
    if(disable_stratum || !stratum || !stratum->running()){
        ZionMiningJob job; job.job_id = "bootstrap"; job.seed_hash = "00"; job.target_difficulty = 100000; job.height=1; job.blob.resize(80, 0);
        job_manager.update_job(job);
    }

    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    ZionMiningStatsAggregator stats(threads);
    ZionShareSubmitter submitter(&pool, stratum.get(), &stats);
    submitter.set_result_callback([&](const ZionShare& sh, bool ok){
        // Could add additional logging hooks or UI forwarding here later.
    });
    submitter.start();

    std::vector<std::unique_ptr<ZionCpuWorker>> workers;
    workers.reserve(threads);
    for(unsigned i=0;i<threads;i++){
        ZionCpuWorkerConfig cfg; cfg.index=i; cfg.total_threads=threads; cfg.use_randomx=true;
        workers.emplace_back(new ZionCpuWorker(&job_manager, &submitter, cfg, &stats));
        workers.back()->start();
    }

    auto start = std::chrono::steady_clock::now();
    while(g_running.load()){
        std::this_thread::sleep_for(std::chrono::seconds(5));
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now-start).count();
        auto snap = stats.snapshot();
        std::cout << "[" << uptime << "s] Total ~" << std::fixed << std::setprecision(1) << snap.total_hashrate << " H/s  A=" << snap.shares_accepted << " R=" << snap.shares_rejected << " Found=" << snap.shares_found;
        if(stratum && stratum->running()){
            std::cout << "  (Stratum A=" << stratum->accepted() << " R=" << stratum->rejected() << ")";
        }
        std::cout << "\n";
    }

    for(auto& w: workers) w->stop();
    submitter.stop();
    if(stratum) stratum->stop();
    pool.disconnect();
#ifdef ZION_HAVE_RANDOMX
    zion::RandomXWrapper::instance().cleanup();
#endif
    std::cout << "Shutdown complete." << std::endl;
    return 0;
}
