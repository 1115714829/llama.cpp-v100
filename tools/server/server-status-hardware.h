#pragma once

#include "../../vendor/nlohmann/json.hpp"
#include <fstream>
#include <sstream>
#include <string>
#ifdef __linux__
#include <dlfcn.h>
#endif

namespace server_status {
using json = nlohmann::ordered_json;

inline json memory() {
    json host = {{"mem_total_bytes", nullptr}, {"mem_available_bytes", nullptr}};
#ifdef __linux__
    std::ifstream in("/proc/meminfo");
    std::string line, key, unit;
    unsigned long long kb;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        if (!(row >> key >> kb >> unit) || unit != "kB") continue;
        if (key == "MemTotal:") host["mem_total_bytes"] = kb * 1024;
        if (key == "MemAvailable:") host["mem_available_bytes"] = kb * 1024;
    }
#endif
    return host;
}

class hardware {
#ifdef __linux__
    void * ml = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    void * cu = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    template<class F> static F sym(void * lib, const char * key) {
        return lib ? reinterpret_cast<F>(dlsym(lib, key)) : nullptr;
    }
    using dev = void *;
    struct mem { unsigned version; unsigned long long total, reserved, free, used; };
    struct util { unsigned gpu, memory; };
    int (*init)() = sym<decltype(init)>(ml,"nvmlInit_v2");
    int (*shutdown)() = sym<decltype(shutdown)>(ml,"nvmlShutdown");
    int (*handle)(const char *, dev *) = sym<decltype(handle)>(ml,"nvmlDeviceGetHandleByPciBusId_v2");
    int (*name)(dev,char *,unsigned) = sym<decltype(name)>(ml,"nvmlDeviceGetName");
    int (*info)(dev,mem *) = sym<decltype(info)>(ml,"nvmlDeviceGetMemoryInfo_v2");
    int (*load)(dev,util *) = sym<decltype(load)>(ml,"nvmlDeviceGetUtilizationRates");
    int (*temp)(dev,unsigned,unsigned *) = sym<decltype(temp)>(ml,"nvmlDeviceGetTemperature");
    int (*power)(dev,unsigned *) = sym<decltype(power)>(ml,"nvmlDeviceGetPowerUsage");
    int (*limit)(dev,unsigned *) = sym<decltype(limit)>(ml,"nvmlDeviceGetEnforcedPowerLimit");
    int (*cu_init)(unsigned) = sym<decltype(cu_init)>(cu,"cuInit");
    int (*count)(int *) = sym<decltype(count)>(cu,"cuDeviceGetCount");
    int (*device)(int *,int) = sym<decltype(device)>(cu,"cuDeviceGet");
    int (*pci)(char *,int,int) = sym<decltype(pci)>(cu,"cuDeviceGetPCIBusId");
    bool ready = init && shutdown && init() == 0;
#endif
public:
    hardware() = default;
    hardware(const hardware &) = delete;
    hardware & operator=(const hardware &) = delete;
    ~hardware() {
#ifdef __linux__
        if (ready) shutdown();
        if (cu) dlclose(cu);
        if (ml) dlclose(ml);
#endif
    }
    json sample(std::string & status) const {
        json cards = json::array();
        status = "unavailable";
#ifdef __linux__
        if (!ml) return cards;
        status = "error";
        int n = 0;
        if (!ready || !handle || !cu_init || !count || !device || !pci || cu_init(0) || count(&n)) return cards;
        status = "ok";
        for (int i = 0; i < n; ++i) {
            json card = {{"id",i},{"name",""},{"mem_used_bytes",nullptr},{"mem_total_bytes",nullptr},{"gpu_util_pct",nullptr},{"temp_c",nullptr},{"power_w",nullptr},{"enforced_limit_w",nullptr},{"status","error"}};
            char bus[64] = {}, label[128] = {};
            int d; dev h = nullptr;
            if (!device(&d,i) && !pci(bus,sizeof(bus),d) && !handle(bus,&h)) {
                unsigned v, fields = 0; mem m{}; m.version = sizeof(m) | (2u << 24); util u;
                if (name && !name(h,label,sizeof(label))) { card["name"] = label; ++fields; }
                if (info && !info(h,&m)) { card["mem_used_bytes"] = m.used; card["mem_total_bytes"] = m.total; ++fields; }
                if (load && !load(h,&u)) { card["gpu_util_pct"] = u.gpu; ++fields; }
                if (temp && !temp(h,0,&v)) { card["temp_c"] = v; ++fields; }
                if (power && !power(h,&v)) { card["power_w"] = v/1000.; ++fields; }
                if (limit && !limit(h,&v)) { card["enforced_limit_w"] = v/1000.; ++fields; }
                card["status"] = fields == 6 ? "ok" : fields ? "partial" : "error";
            } else status = "error";
            cards.push_back(std::move(card));
        }
#endif
        return cards;
    }
};
}
