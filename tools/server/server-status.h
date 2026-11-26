#pragma once
#include "server-status-hardware.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace server_status {
class monitor {
    using clock = std::chrono::steady_clock;
    static int64_t wall() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
    struct cell { std::atomic<int> phase{0}, used{0}, cap{0}; };
    std::unique_ptr<cell[]> cells;
    int count = 0;
    std::string display;
    std::mutex layout, control;
    std::condition_variable wake;
    bool stop = false;
    const int64_t epoch = wall();
    std::shared_ptr<const json> snapshot;
    std::thread worker;
    static json value(double v) { return v > 0 ? json(v) : json(nullptr); }
    void run() {
        hardware hw;
        uint64_t seq = 0, oldp = prompt.load(), oldg = generated.load();
        auto prev = clock::now();
        std::deque<json> ring;
        for (;;) {
            auto now = clock::now(); auto ms = wall();
            double dt = std::chrono::duration<double>(now-prev).count();
            auto p = prompt.load(), g = generated.load(); int np=0, ng=0;
            json slots = json::array(); std::string model;
            { std::lock_guard<std::mutex> lk(layout); model = display;
              for (int i=0;i<count;++i) { auto & c=cells[i]; int ph=c.phase.load(); np+=ph==1; ng+=ph==2;
                slots.push_back({{"id",i},{"state",ph==1?"prefill":ph==2?"decode":"idle"},{"context_used_tokens",c.used.load()},{"context_capacity_tokens",c.cap.load()}});
              }
            }
            json pp=np && seq && dt>0 ? value((p-oldp)/dt) : json(nullptr), de=ng && seq && dt>0 ? value((g-oldg)/dt) : json(nullptr);
            prev=now; oldp=p; oldg=g; ++seq;
            if (ring.size()==60) ring.pop_front();
            ring.push_back({{"t_ms",ms},{"prefill_tps",pp},{"decode_tps",de},{"active_slots",np+ng}});
            auto a=accepted.load(), d=proposed.load(); std::string status; auto gpu=hw.sample(status);
            json inf={{"phase",np&&ng?"mixed":np?"prefill":ng?"decode":slots.empty()?"loading":"idle"},{"active_slots",np+ng},{"prefill_tps",pp},{"decode_tps",de},{"last_prefill_tps",value(last_prefill.load())},{"last_decode_tps",value(last_decode.load())},{"ttft_ms",value(ttft.load())},{"draft_proposed",d},{"draft_accepted",a},{"draft_acceptance_rate",d?json(double(a)/d):json(nullptr)},{"history",ring}};
            json doc={{"schema_version",1},{"epoch",epoch},{"seq",seq},{"sampled_at_ms",ms},{"nvml_status",status},{"model",{{"display_name",model}}},{"inference",inf},{"slots",slots},{"gpus",gpu},{"host",memory()}};
            std::atomic_store(&snapshot,std::make_shared<const json>(std::move(doc)));
            std::unique_lock<std::mutex> lk(control); if(wake.wait_until(lk,now+std::chrono::seconds(1),[this]{return stop;})) break;
        }
    }
public:
    std::atomic<uint64_t> prompt{0},generated{0},proposed{0},accepted{0};
    std::atomic<double> last_prefill{0},last_decode{0},ttft{0};
    void configure(int n,const std::string & name) {
        std::lock_guard<std::mutex> lk(layout); count=n; cells.reset(new cell[n]); display=name.substr(name.find_last_of("/\\")+1);
        if(!worker.joinable()) worker=std::thread(&monitor::run,this);
    }
    void slot(int i,int phase,int used,int cap) { cells[i].used.store(used,std::memory_order_relaxed); cells[i].cap.store(cap,std::memory_order_relaxed); cells[i].phase.store(phase,std::memory_order_relaxed); }
    json read(const std::string & since) { auto s=std::atomic_load(&snapshot); if(!s) return nullptr; json out=*s;
        if (!since.empty() && since.size()<20 && since.find_first_not_of("0123456789")==std::string::npos) {
            auto & h=out["inference"]["history"]; uint64_t seq=out["seq"], after=std::stoull(since); auto keep=after>=seq?0:std::min<uint64_t>(h.size(),seq-after); h.erase(h.begin(),h.end()-keep);
        } return out;
    }
    ~monitor(){ {std::lock_guard<std::mutex> lk(control); stop=true;} wake.notify_all(); if(worker.joinable()) worker.join(); }
};
}
