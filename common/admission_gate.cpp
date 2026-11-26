#include "admission_gate.h"
#include "admission_numa.h"
#include "admission_owned_math.h"
#include "admission_pin_math.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <limits>
#include <mutex>
#include <new>
#include <condition_variable>
#include <thread>
#include <utility>
#include <cerrno>
#include <sys/stat.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#endif

namespace {

constexpr uint64_t GiB = 1024ull*1024*1024;

bool checked_add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool parse_unsigned(const std::string & s, uint64_t & out) {
    if (s.empty()) return false;
    out = 0;
    for (const char c : s) {
        if (c < '0' || c > '9') return false;
        const uint64_t digit = (uint64_t) (c - '0');
        if (out > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
        out = out * 10 + digit;
    }
    return true;
}

std::string trim(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad() || ss.bad()) return false;
    out = ss.str();
    return true;
}

bool valid_id_list(const std::string & input) {
    const std::string text = trim(input);
    if (text.empty() || text.back() == ',') return false;
    std::istringstream in(text);
    std::string token;
    uint64_t previous = 0;
    bool have = false;
    while (std::getline(in, token, ',')) {
        const size_t dash = token.find('-');
        uint64_t first = 0, last = 0;
        if (!parse_unsigned(token.substr(0, dash), first)) return false;
        last = first;
        if (dash != std::string::npos && !parse_unsigned(token.substr(dash + 1), last)) return false;
        if (last < first || last > (uint64_t) std::numeric_limits<int>::max() || (have && first <= previous)) return false;
        previous = last;
        have = true;
    }
    return have;
}

bool meminfo_value(const std::string & text, const std::string & key, uint64_t & bytes) {
    std::istringstream in(text);
    std::string line;
    const std::string want = key + ":";
    bool found = false;
    while (std::getline(in, line)) {
        std::string t = trim(line);

        if (t.rfind("Node ", 0) == 0) {
            const size_t sp = t.find(' ', 5);
            if (sp == std::string::npos) {
                continue;
            }
            t = trim(t.substr(sp + 1));
        }
        if (t.rfind(want, 0) != 0) {
            continue;
        }
        if (found) return false;
        found = true;
        std::istringstream fields(t.substr(want.size()));
        uint64_t value = 0;
        std::string number, unit, extra;
        if (!(fields >> number) || !parse_unsigned(number, value)) return false;
        fields >> unit;
        if (fields >> extra) return false;
        uint64_t scale = 1;
        if (unit == "kB") scale = 1024;
        else if (unit == "mB" || unit == "MB") scale = 1024 * 1024;
        else if (!unit.empty() && unit != "B") return false;
        if (value > std::numeric_limits<uint64_t>::max() / scale) return false;
        bytes = value * scale;
    }
    return found;
}

bool cgroup_line_path(const std::string & line, bool & is_v2, std::string & path) {
    const size_t c1 = line.find(':');
    if (c1 == std::string::npos) {
        return false;
    }
    const size_t c2 = line.find(':', c1 + 1);
    if (c2 == std::string::npos) {
        return false;
    }
    const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1);
    path = line.substr(c2 + 1);

    is_v2 = controllers.empty();
    return true;
}

bool cgroup_v1_has_memory(const std::string & line, std::string & path) {
    const size_t c1 = line.find(':');
    if (c1 == std::string::npos) {
        return false;
    }
    const size_t c2 = line.find(':', c1 + 1);
    if (c2 == std::string::npos) {
        return false;
    }
    std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
    std::string c;
    bool has = false;
    while (std::getline(ctl, c, ',')) {
        if (c == "memory") {
            has = true;
        }
    }
    if (!has) {
        return false;
    }
    path = line.substr(c2 + 1);
    return true;
}

std::string join_path(const std::string & dir, const std::string & rel) {
    if (rel.empty()) {
        return dir;
    }
    std::string r = rel;
    if (r[0] == '/') {
        r = r.substr(1);
    }
    if (r.empty()) {
        return dir;
    }
    if (dir.empty()) {
        return r;
    }
    if (dir.back() == '/') {
        return dir + r;
    }
    return dir + "/" + r;
}

bool dir_exists(const std::string & path) {
    static const char * names[] = {
        "cgroup.procs", "cgroup.clone_children", "memory.max",
        "memory.limit_in_bytes", "cgroup.controllers",
    };
    for (const char * name : names) {
        std::ifstream f(path + "/" + name);
        if (f) {
            return true;
        }
    }
    return false;
}

std::string parent_path(const std::string & path) {
    const size_t p = path.find_last_of('/');
    if (p == std::string::npos) {
        return "";
    }
    if (p == 0) {
        return "/";
    }
    return path.substr(0, p);
}

struct cgroup_roots {
    std::string v2;
    std::string v1_mem;
    bool        found_v2  = false;
    bool        found_v1  = false;
};

cgroup_roots read_cgroup_roots(const common_admission_paths & paths) {
    cgroup_roots roots;
    std::string mountinfo;
    if (!read_file(join_path(paths.proc_dir, "self/mountinfo"), mountinfo)) {

        roots.v2      = paths.cgroup_root;
        roots.found_v2 = true;
        return roots;
    }
    std::istringstream in(mountinfo);
    std::string line;
    while (std::getline(in, line)) {

        const size_t sep = line.find(" - ");
        if (sep == std::string::npos) {
            continue;
        }
        std::istringstream head(line.substr(0, sep));
        std::vector<std::string> fields;
        std::string f;
        while (head >> f) {
            fields.push_back(f);
        }
        if (fields.size() < 5) {
            continue;
        }
        const std::string & mountpoint = fields[4];
        std::istringstream tail(line.substr(sep + 3));
        std::string fstype;
        std::string source;
        std::string superopts;
        tail >> fstype >> source >> superopts;
        if (fstype == "cgroup2") {
            roots.v2      = mountpoint;
            roots.found_v2 = true;
        } else if (fstype == "cgroup") {

            if (superopts.find("memory") != std::string::npos) {
                roots.v1_mem  = mountpoint;
                roots.found_v1 = true;
            }
        }
    }
    return roots;
}

struct cgroup_stat {
    uint64_t file           = 0;
    uint64_t inactive_file  = 0;
    uint64_t file_dirty     = 0;
    uint64_t file_writeback = 0;
    bool     valid          = false;
};

cgroup_stat read_cgroup_stat_v2(const std::string & dir) {
    cgroup_stat st;
    std::string text;
    if (!read_file(join_path(dir, "memory.stat"), text)) {
        return st;
    }
    std::istringstream in(text);
    std::string line;
    bool have_file = false, have_dirty = false, have_writeback = false, have_inactive = false;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string key, number;
        uint64_t value = 0;
        if (!(fields >> key >> number) || !parse_unsigned(number, value)) return st;
        if (key == "file") {
            if (have_file) return st;
            st.file = value;
            have_file = true;
        } else if (key == "inactive_file") {
            if (have_inactive) return st;
            st.inactive_file = value;
            have_inactive = true;
        } else if (key == "file_dirty") {
            if (have_dirty) return st;
            st.file_dirty = value;
            have_dirty = true;
        } else if (key == "file_writeback") {
            if (have_writeback) return st;
            st.file_writeback = value;
            have_writeback = true;
        }
    }
    st.valid = have_file && have_inactive && have_dirty && have_writeback;
    return st;
}

cgroup_stat read_cgroup_stat_v1(const std::string & dir) {
    cgroup_stat st;
    std::string text;
    if (!read_file(join_path(dir, "memory.stat"), text)) {
        return st;
    }
    std::istringstream in(text);
    std::string line;
    bool have_active = false, have_inactive = false, have_dirty = false, have_writeback = false;
    uint64_t active = 0, inactive = 0;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string key, number;
        uint64_t value = 0;
        if (!(fields >> key >> number)) {
            continue;
        }
        if (!parse_unsigned(number, value)) {
            return st;
        }
        if (key == "total_active_file") {
            if (have_active) return st;
            active = value;
            have_active = true;
        } else if (key == "total_inactive_file") {
            if (have_inactive) return st;
            inactive = value;
            have_inactive = true;
        } else if (key == "total_dirty") {
            if (have_dirty) return st;
            st.file_dirty = value;
            have_dirty = true;
        } else if (key == "total_writeback") {
            if (have_writeback) return st;
            st.file_writeback = value;
            have_writeback = true;
        }
    }
    if (!have_active || !have_inactive || !have_dirty || !have_writeback) {
        return st;
    }
    if (!checked_add(active, inactive, st.file)) {
        return st;
    }
    st.inactive_file = inactive;
    st.valid = true;
    return st;
}

uint64_t cgroup_reclaimable(const cgroup_stat & st, uint64_t current, uint64_t limit) {
    if (!st.valid) {
        return 0;
    }
    uint64_t reclaimable = std::min(std::max(st.file, st.inactive_file), current);
    reclaimable = st.file_dirty >= reclaimable ? 0 : reclaimable - st.file_dirty;
    reclaimable = st.file_writeback >= reclaimable ? 0 : reclaimable - st.file_writeback;

    reclaimable -= std::min(reclaimable / 2, limit / 64);
    return reclaimable;
}

enum cgroup_status {
    CGROUP_HAS_LIMIT = 0,
    CGROUP_NO_LIMIT  = 1,
    CGROUP_FAILED    = 2,
};

cgroup_status read_cgroup(const common_admission_paths & paths,
                           uint64_t & limit, uint64_t & usage, uint64_t & reclaimable) {
    std::string self_cgroup;
    if (!read_file(join_path(paths.proc_dir, "self/cgroup"), self_cgroup)) {
        return CGROUP_FAILED;
    }

    const cgroup_roots roots = read_cgroup_roots(paths);
    if (!roots.found_v2 && !roots.found_v1) {

        return CGROUP_NO_LIMIT;
    }

    std::istringstream in(self_cgroup);
    std::string line;
    bool have_cap   = false;
    bool have_level = false;
    bool any_error  = false;

    uint64_t min_cap   = 0;
    uint64_t bind_used = 0;
    uint64_t bind_rec  = 0;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        bool is_v2 = false;
        std::string path;
        if (!cgroup_line_path(line, is_v2, path)) {
            continue;
        }

        if (is_v2) {
            if (!roots.found_v2) {
                continue;
            }
            const std::string dir = join_path(roots.v2, path);
            if (!dir_exists(dir)) {

                any_error = true;
                continue;
            }
            have_level = true;

            std::string cur = dir;
            while (!cur.empty()) {
                std::string limit_text;
                std::string current_text;
                if (read_file(join_path(cur, "memory.max"), limit_text)) {
                    limit_text = trim(limit_text);
                    if (limit_text != "max") {
                        uint64_t cap = 0;
                        if (!parse_unsigned(limit_text, cap)) {
                            any_error = true;
                            break;
                        }
                        uint64_t used = 0;
                        if (!read_file(join_path(cur, "memory.current"), current_text) ||
                            !parse_unsigned(trim(current_text), used)) {
                            any_error = true;
                            break;
                        }
                        const cgroup_stat st = read_cgroup_stat_v2(cur);
                        if (!st.valid) { any_error = true; break; }
                        const uint64_t rec = cgroup_reclaimable(st, used, cap);

                        if (!have_cap || cap < min_cap) {
                            min_cap   = cap;
                            bind_used = used;
                            bind_rec  = rec;
                        }
                        have_cap = true;
                    }
                } else {

                    struct stat missing;
                    const std::string limit_path = join_path(cur, "memory.max");
                    if (cur != roots.v2 || stat(limit_path.c_str(), &missing) == 0 || errno != ENOENT) {
                        any_error = true;
                        break;
                    }
                }
                if (cur == roots.v2) {
                    break;
                }
                const std::string up = parent_path(cur);
                if (up == cur || up.empty()) {
                    break;
                }
                cur = up;
            }
        } else {
            std::string v1_path;
            if (!cgroup_v1_has_memory(line, v1_path)) {
                continue;
            }
            if (!roots.found_v1) {
                continue;
            }
            const std::string root = roots.v1_mem;
            std::string cur = join_path(root, v1_path);
            if (!dir_exists(cur)) {
                any_error = true;
                continue;
            }
            have_level = true;
            while (!cur.empty()) {
                std::string limit_text;
                std::string usage_text;
                if (read_file(join_path(cur, "memory.limit_in_bytes"), limit_text) &&
                    read_file(join_path(cur, "memory.usage_in_bytes"), usage_text)) {
                    uint64_t cap = 0, used = 0;
                    if (!parse_unsigned(trim(limit_text), cap) || !parse_unsigned(trim(usage_text), used)) {
                        any_error = true;
                        break;
                    }

                    if (cap < (1ull << 62)) {

                        const cgroup_stat st = read_cgroup_stat_v1(cur);
                        const uint64_t rec = st.valid ? cgroup_reclaimable(st, used, cap) : 0;
                        if (!have_cap || cap < min_cap) {
                            min_cap = cap;
                            bind_used = used;
                            bind_rec = rec;
                        }
                        have_cap = true;
                    }
                } else {
                    any_error = true;
                    break;
                }
                if (cur == root) {
                    break;
                }
                const std::string up = parent_path(cur);
                if (up == cur || up.empty()) {
                    break;
                }
                cur = up;
            }
        }
    }

    if (any_error) {
        return CGROUP_FAILED;
    }
    if (!have_cap) {

        return have_level ? CGROUP_NO_LIMIT : CGROUP_FAILED;
    }

    limit       = min_cap;
    usage       = bind_used;
    reclaimable = bind_rec;
    return CGROUP_HAS_LIMIT;
}

bool read_numa_nodes(const common_admission_paths & paths,
                     std::vector<common_admission_node> & nodes, bool & seen) {
    nodes.clear();
    seen = false;

    const std::string node_root = join_path(paths.sys_dir, "devices/system/node");
    struct stat root_stat;
    if (stat(node_root.c_str(), &root_stat) != 0) {
        return errno == ENOENT;
    }
    seen = true;
    if (!S_ISDIR(root_stat.st_mode)) return false;
    std::string probe;
    if (!read_file(join_path(node_root, "online"), probe)) return false;

    std::string online = trim(probe);
    if (!valid_id_list(online)) return false;

    std::vector<int> ids;
    {
        std::istringstream in(online);
        std::string tok;
        while (std::getline(in, tok, ',')) {
            tok = trim(tok);
            if (tok.empty()) return false;
            const size_t dash = tok.find('-');
            uint64_t first = 0, last = 0;
            if (!parse_unsigned(tok.substr(0, dash), first)) return false;
            last = first;
            if (dash != std::string::npos && !parse_unsigned(tok.substr(dash + 1), last)) return false;
            if (last < first || last > (uint64_t) std::numeric_limits<int>::max()) return false;

            if (last - first >= 65536 || ids.size() + (last - first + 1) > 65536) return false;
            for (uint64_t n = first; n <= last; ++n) {
                if (std::find(ids.begin(), ids.end(), (int) n) != ids.end()) return false;
                ids.push_back((int) n);
            }
        }
    }

    for (const int id : ids) {
        char base[64];
        snprintf(base, sizeof(base), "devices/system/node/node%d", id);
        const std::string dir = join_path(paths.sys_dir, base);

        std::string cpulist;
        if (!read_file(join_path(dir, "cpulist"), cpulist)) {
            return false;
        }
        if (trim(cpulist).empty()) {
            continue;
        }
        if (!valid_id_list(cpulist)) return false;

        std::string meminfo;
        if (!read_file(join_path(dir, "meminfo"), meminfo)) {
            return false;
        }

        common_admission_node node;
        node.id = id;
        uint64_t active_file   = 0;
        uint64_t inactive_file = 0;
        if (!meminfo_value(meminfo, "MemTotal", node.total)) return false;
        if (node.total == 0) continue;
        uint64_t dirty = 0, writeback = 0;
        if (!meminfo_value(meminfo, "MemFree", node.free) || node.free > node.total ||
            !meminfo_value(meminfo, "Active(file)", active_file) ||
            !meminfo_value(meminfo, "Inactive(file)", inactive_file) ||
            !meminfo_value(meminfo, "Dirty", dirty) ||
            !meminfo_value(meminfo, "Writeback", writeback) ||
            !checked_add(active_file, inactive_file, node.reclaimable)) return false;
        node.reclaimable -= std::min(node.reclaimable, dirty);
        node.reclaimable -= std::min(node.reclaimable, writeback);
        nodes.push_back(node);
    }

    if (nodes.empty()) {

        return false;
    }
    return true;
}

bool read_process_page_cache(const common_admission_paths & paths, uint64_t & page_cache, uint64_t & locked) {
    std::string status;
    if (!read_file(join_path(paths.proc_dir, "self/status"), status)) {
        return false;
    }
    uint64_t rss_file = 0;
    uint64_t vm_lock = 0;
    if (!meminfo_value(status, "RssFile", rss_file) ||
        !meminfo_value(status, "VmLck", vm_lock)) return false;
    page_cache = rss_file;
    locked    = vm_lock;
    return true;
}

}

uint64_t common_admission_headroom_bytes(uint64_t mem_total, int host_class) {

    constexpr uint64_t eight_gib   = 8ull  * 1024ull * 1024ull * 1024ull;
    constexpr uint64_t sixteen_gib = 16ull * 1024ull * 1024ull * 1024ull;
    if (host_class == COMMON_ADMISSION_HOST_OWN) {
        const uint64_t by_percent = (mem_total / 100) * 8 + ((mem_total % 100) * 8) / 100;
        return std::max(eight_gib, by_percent);
    }
    const uint64_t by_percent = (mem_total / 100) * 12 + ((mem_total % 100) * 12) / 100;
    return std::max(sixteen_gib, by_percent);
}

int common_admission_host_class_from_env() {

    const char * v = getenv("LLAMA_V100_HOST_CLASS");
    if (v != nullptr && (strcmp(v, "own") == 0 || strcmp(v, "OWN") == 0)) {
        return COMMON_ADMISSION_HOST_OWN;
    }
    return COMMON_ADMISSION_HOST_BORROWED;
}

common_admission_config common_admission_config_from_env() {
    common_admission_config cfg;
    cfg.host_class = common_admission_host_class_from_env();
    const char * v = getenv("LLAMA_V100_ADMISSION_HEADROOM_GIB");
    if (v != nullptr) {
        const double gib = atof(v);
        if (gib > 0) {
            cfg.headroom_override = (uint64_t) (gib * (double) GiB);
        }
    }
    return cfg;
}

common_admission_supply common_admission_probe(const common_admission_paths & paths) {
    common_admission_supply supply;
    supply.probe_ok = false;

    std::string meminfo;
    if (!read_file(join_path(paths.proc_dir, "meminfo"), meminfo)) {
        supply.probe_error = "the memory information of the machine could not be read";
        return supply;
    }
    if (!meminfo_value(meminfo, "MemTotal", supply.mem_total) ||
        !meminfo_value(meminfo, "MemAvailable", supply.mem_available)) {
        supply.probe_error = "the memory information of the machine is incomplete";
        return supply;
    }

    {
        uint64_t limit       = 0;
        uint64_t usage       = 0;
        uint64_t reclaimable = 0;
        const cgroup_status st = read_cgroup(paths, limit, usage, reclaimable);
        if (st == CGROUP_FAILED) {
            supply.probe_error = "the cgroup limit of the process could not be read";
            return supply;
        }
        if (st == CGROUP_HAS_LIMIT) {
            supply.cgroup_seen        = true;
            supply.cgroup_limit       = limit;
            supply.cgroup_usage       = usage;
            supply.cgroup_reclaimable = reclaimable;
        }

    }

    if (!read_numa_nodes(paths, supply.nodes, supply.numa_seen)) {
        supply.probe_error = "the NUMA nodes of the host could not be read";
        return supply;
    }

    if (!read_process_page_cache(paths, supply.own_page_cache, supply.own_locked)) {
        supply.probe_error = "the page cache of the process could not be read";
        return supply;
    }

    supply.probe_ok = true;
    return supply;
}

common_admission_result common_admission_check(
        const common_admission_supply & supply,
        const common_admission_demand & demand,
        const common_admission_config & config) {
    common_admission_result result;

    if (!supply.probe_ok) {
        result.verdict = COMMON_ADMISSION_REJECT;
        result.detail  = "the machine could not be measured: " + supply.probe_error;
        return result;
    }

    uint64_t mem_base = supply.mem_total;
    if (!supply.nodes.empty()) {
        mem_base = 0;
        for (const auto & node : supply.nodes) {
            if (!checked_add(mem_base, node.total, mem_base)) {
                result.detail = "NUMA byte count overflow";
                return result;
            }
        }
        if (mem_base == 0) {
            result.detail = "NUMA memory capacity is zero";
            return result;
        }
    }
    if (supply.cgroup_seen && supply.cgroup_limit > 0 && supply.cgroup_limit < mem_base) {
        mem_base = supply.cgroup_limit;
    }
    result.headroom = config.headroom_override > 0
        ? config.headroom_override
        : common_admission_headroom_bytes(mem_base, config.host_class);

    const uint64_t held_by_us = std::max(supply.own_page_cache, supply.own_locked);

    uint64_t cgroup_available = 0;
    if (supply.cgroup_seen) {
        uint64_t reclaimable = supply.cgroup_reclaimable;
        reclaimable = reclaimable > held_by_us ? reclaimable - held_by_us : 0;
        const uint64_t used = supply.cgroup_usage > reclaimable
            ? supply.cgroup_usage - reclaimable
            : 0;
        cgroup_available = supply.cgroup_limit > used ? supply.cgroup_limit - used : 0;
    }

    uint64_t numa_available = 0;
    {
        uint64_t total_free = 0;
        uint64_t total_size = 0;
        for (const common_admission_node & node : supply.nodes) {
            uint64_t node_free = 0;
            if (!checked_add(node.free, node.reclaimable, node_free) ||
                !checked_add(total_free, node_free, total_free) ||
                !checked_add(total_size, node.total, total_size)) {
                result.detail = "NUMA byte count overflow";
                return result;
            }
        }
        numa_available = total_free > held_by_us ? total_free - held_by_us : 0;

        uint64_t reserve_sum = 0;
        for (size_t i = 0; i < supply.nodes.size(); i++) {
            uint64_t share = 0;
            if (total_size > 0) {

                __extension__ const unsigned __int128 product =
                    (__extension__ (unsigned __int128) result.headroom) * supply.nodes[i].total;
                share = (uint64_t) (product / total_size);
                if (share > result.headroom - reserve_sum) {
                    share = result.headroom - reserve_sum;
                }
            }

            if (i + 1 == supply.nodes.size()) {
                share = result.headroom > reserve_sum ? result.headroom - reserve_sum : 0;
            }
            common_admission_node node = supply.nodes[i];
            node.os_reserve = share;
            reserve_sum += share;

            uint64_t node_avail = node.free + node.reclaimable;
            node_avail = node_avail > share ? node_avail - share : 0;
            node.avail = node_avail;
            result.nodes.push_back(node);
        }
    }

    uint64_t available = 0;
    bool have = false;
    if (supply.cgroup_seen) {
        available = cgroup_available;
        have = true;
    }
    if (!supply.nodes.empty()) {
        if (!have || numa_available < available) {
            available = numa_available;
        }
        have = true;
    }
    if (!have) {

        available = supply.mem_available > held_by_us ? supply.mem_available - held_by_us : 0;
    }

    result.available = available;

    result.usable    = available > result.headroom ? available - result.headroom : 0;

    const uint64_t resident_cap = std::max(supply.own_page_cache, supply.own_locked);
    result.resident = std::min(std::min(demand.resident_bytes, demand.committed_bytes), resident_cap);
    if (!checked_add(demand.committed_bytes - result.resident, demand.transient, result.demand)) {
        result.detail = "admission demand overflow";
        return result;
    }

    if (result.demand <= result.usable) {
        result.verdict = COMMON_ADMISSION_ADMIT;
        result.detail  = "the demand fits: " + std::to_string(result.demand) +
            " bytes of new memory, " + std::to_string(result.usable) + " usable bytes";
        return result;
    }

    result.deficit = result.demand > result.usable ? result.demand - result.usable : 0;
    if (config.degrade_allowed) {

        result.shed_bytes = result.deficit;
        result.verdict    = COMMON_ADMISSION_DEGRADE;
    } else {
        result.verdict = COMMON_ADMISSION_REJECT;
    }
    result.detail = "the demand does not fit: " + std::to_string(result.demand) +
        " bytes of new memory needed, " + std::to_string(result.usable) + " usable bytes, short by " +
        std::to_string(result.deficit) + " bytes";
    return result;
}

std::string common_admission_detail(const common_admission_supply & supply, const common_admission_result & result) {
    std::ostringstream ss;
    ss << "the host memory ledger: demand " << result.demand << " bytes of new memory (" <<
        result.resident << " bytes of the demand are already resident), available " <<
        result.available << " bytes, headroom " << result.headroom <<
        " bytes, usable " << result.usable << " bytes";
    if (supply.cgroup_seen) {
        ss << "; cgroup: limit " << supply.cgroup_limit << " bytes, usage " << supply.cgroup_usage <<
            " bytes, reclaimable " << supply.cgroup_reclaimable << " bytes";
    }
    for (const common_admission_node & node : result.nodes) {
        ss << "; node " << node.id << ": total " << node.total << " bytes, free " << node.free <<
            " bytes, reclaimable " << node.reclaimable << " bytes, reserve " << node.os_reserve <<
            " bytes, available " << node.avail << " bytes";
    }
    ss << "; the page cache of this process: " << supply.own_page_cache << " bytes, locked " <<
        supply.own_locked << " bytes";
    return ss.str();
}

namespace {
struct admission_commitments {
    std::mutex mutex;
    std::condition_variable scopes;
    std::thread::id scope_owner;
    uint64_t scope_depth = 0;
    common_admission_scope_kind scope_kind = common_admission_scope_kind::initialization;
    uint64_t pending = 0;
    struct owner_entry {
        void * mapping = nullptr;
        uint64_t bytes = 0;
        std::thread::id thread;
        bool occupied = false;
        bool locked = false;
        bool published = false;
    } owners[16];
    uint64_t owned_locked_anon = 0;

    uint64_t owned_pinned = 0;
    unsigned pin_settles = 0;

    uint64_t registry_generation = 0;

    unsigned materializing = 0;
#if defined(__linux__)
    const int64_t process = (int64_t) getpid();
#else
    const int64_t process = 0;
#endif
};
admission_commitments & commitments() {
    static admission_commitments state;
    return state;
}
int64_t admission_pid() noexcept {
#if defined(__linux__)
    return (int64_t) getpid();
#else
    return 0;
#endif
}
}

namespace {

common_admission_error pin_math_to_gate(admission_pin_math::error e) noexcept {
    switch (e) {
        case admission_pin_math::error::uncovered: return common_admission_error::rejected;
        case admission_pin_math::error::unproven:  return common_admission_error::invalid_owner_observation;
        case admission_pin_math::error::overflow:  return common_admission_error::overflow;
        default:                                   return common_admission_error::internal;
    }
}
}

common_admission_status common_admission_scope::enter(common_admission_scope_kind kind) noexcept {
    common_admission_status status;
    if (entered_) { status.error = common_admission_error::internal; return status; }
    try {
        auto & state = commitments();
        if (state.process != admission_pid()) {
            status.error = common_admission_error::wrong_process;
            return status;
        }
        std::unique_lock<std::mutex> lock(state.mutex);
        const auto self = std::this_thread::get_id();
        if (state.scope_depth && state.scope_owner == self) {
            if (kind != common_admission_scope_kind::initialization || state.scope_kind != kind ||
                state.scope_depth == UINT64_MAX) {
                status.error = common_admission_error::internal;
                return status;
            }
            ++state.scope_depth;
        } else {
            state.scopes.wait(lock, [&state] { return state.scope_depth == 0; });

            if (kind == common_admission_scope_kind::initialization && state.pending != 0) {
                status.error = common_admission_error::busy;
                return status;
            }
            state.scope_owner = self;
            state.scope_kind = kind;
            state.scope_depth = 1;
        }
        owner_pid_ = state.process;
        entered_ = true;
    } catch (...) { status.error = common_admission_error::internal; }
    return status;
}
void common_admission_scope::leave() noexcept {
    if (!entered_ || owner_pid_ != admission_pid()) return;
    try {
        auto & state = commitments();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.scope_owner != std::this_thread::get_id() || !state.scope_depth) {
            std::fputs("admission scope thread-affinity violation\n", stderr);
            std::abort();
        }
        if (--state.scope_depth == 0) {
            state.scope_owner = {};
            state.scopes.notify_all();
        }
        entered_ = false;
    } catch (...) {}
}
common_admission_scope::~common_admission_scope() noexcept { leave(); }

#ifdef COMMON_ADMISSION_OWNER_TESTING
static thread_local common_admission_owner_test_input * owner_test = nullptr;
void common_admission_owner_test_set(common_admission_owner_test_input * input) noexcept { owner_test = input; }
void common_admission_owner_test_counts(uint64_t & p, uint64_t & o, unsigned & entries) noexcept {
    auto & state = commitments();
    std::lock_guard<std::mutex> lock(state.mutex);
    p = state.pending; o = state.owned_locked_anon; entries = 0;
    for (const auto & entry : state.owners) if (entry.occupied) ++entries;
}
void common_admission_owner_test_materializing(unsigned & materializing) noexcept {
    auto & state = commitments();
    std::lock_guard<std::mutex> lock(state.mutex);
    materializing = state.materializing;
}
#endif

namespace {

bool observe_process_status(uint64_t & rss_file, uint64_t & vm_locked) {
#ifdef COMMON_ADMISSION_OWNER_TESTING
    if (owner_test != nullptr) {
        if (owner_test->throw_from_status) throw std::bad_alloc();
        if (owner_test->override_process_status) {
            rss_file = owner_test->override_rss_file;
            vm_locked = owner_test->override_vm_locked;
            return true;
        }
    }
    common_admission_paths paths;
    if (owner_test != nullptr && owner_test->override_paths) paths = owner_test->paths;
#else
    const common_admission_paths paths;
#endif
    uint64_t page_cache = 0;
    uint64_t locked = 0;
    if (!read_process_page_cache(paths, page_cache, locked)) return false;
    rss_file = page_cache;
    vm_locked = locked;
    return true;
}
}

common_admission_observation common_admission_observe() noexcept {
    common_admission_observation out;
    try {

        if (commitments().process != admission_pid()) {
            out.error = common_admission_error::wrong_process;
            return out;
        }
        constexpr unsigned max_attempts = 4;
        common_admission_error unsettled = common_admission_error::unstable;
        for (unsigned attempt = 1; attempt <= max_attempts; ++attempt) {
            out.attempts = attempt;
            uint64_t generation = 0;
            uint64_t owned = 0;
            unsigned materializing = 0;
            {
                auto & state = commitments();
                std::lock_guard<std::mutex> lock(state.mutex);
                generation = state.registry_generation;
                owned = state.owned_locked_anon;
                materializing = state.materializing;
            }

#ifdef COMMON_ADMISSION_OWNER_TESTING
            if (owner_test != nullptr && owner_test->observe_point != nullptr) owner_test->observe_point();
#endif
            uint64_t rss_file = 0;
            uint64_t vm_locked = 0;
            if (!observe_process_status(rss_file, vm_locked)) {

                out.error = common_admission_error::probe_failed;
                return out;
            }
            uint64_t generation_after = 0;
            uint64_t owned_after = 0;
            unsigned materializing_after = 0;
            {
                auto & state = commitments();
                std::lock_guard<std::mutex> lock(state.mutex);
                generation_after = state.registry_generation;
                owned_after = state.owned_locked_anon;
                materializing_after = state.materializing;
            }
            if (materializing != 0 || materializing_after != 0) {

                unsettled = common_admission_error::materializing;
                out.repeated = true;
                continue;
            }
            if (generation != generation_after || owned != owned_after) {
                unsettled = common_admission_error::unstable;
                out.repeated = true;
                continue;
            }
            admission_owned_math::observation window;
            window.rss_file = rss_file;
            window.vm_locked = vm_locked;
            window.owned_locked_anon = owned;
            window.registry_before = generation;
            window.registry_after = generation_after;
            const auto composed = admission_owned_math::normalize_observed(window);
            if (composed.status == admission_owned_math::observation_error::unstable) {
                unsettled = common_admission_error::unstable;
                out.repeated = true;
                continue;
            }
            out.rss_file = rss_file;
            out.vm_locked = vm_locked;
            out.owned_locked_anon = owned;
            if (composed.status == admission_owned_math::observation_error::invalid_owner_observation) {

                out.error = common_admission_error::invalid_owner_observation;
                return out;
            }
            if (composed.status != admission_owned_math::observation_error::ok) {
                out.error = common_admission_error::internal;
                return out;
            }
            out.error = common_admission_error::ok;
            out.unclassified_locked = composed.held.unclassified_locked;
            out.held_bytes = composed.held.held_file_credit;
            out.legacy_resident_cap = composed.held.legacy_resident_cap;
            return out;
        }

        out.repeated = true;
        out.error = unsettled;
        return out;
    } catch (...) {

        common_admission_observation failed;
        failed.error = common_admission_error::internal;
        failed.attempts = out.attempts;
        return failed;
    }
}
common_admission_status common_admission_lease::grow(uint64_t bytes) noexcept {
    common_admission_status status;
    status.requested = bytes;
    const int64_t pid = admission_pid();
    if (!pid || (owner_pid_ && owner_pid_ != pid)) {
        status.error = common_admission_error::wrong_process;
        return status;
    }
    try {
        auto & state = commitments();
        if (state.process != pid) {
            status.error = common_admission_error::wrong_process;
            return status;
        }
        std::unique_lock<std::mutex> guard(state.mutex);
        if (!state.scope_depth || state.scope_owner != std::this_thread::get_id() ||
            state.scope_kind != common_admission_scope_kind::transaction) {
            status.error = common_admission_error::scope_required;
            return status;
        }
        uint64_t total = 0, mine = 0;
        if (!checked_add(state.pending, bytes, total) || !checked_add(pending_, bytes, mine)) {
            status.error = common_admission_error::overflow;
            return status;
        }
        status.pending = state.pending;
        ++state.registry_generation;

        guard.unlock();
#ifdef COMMON_ADMISSION_OWNER_TESTING
        const auto supply = owner_test ? owner_test->supply : common_admission_probe();
#else
        const auto supply = common_admission_probe();
#endif
        if (!supply.probe_ok) {
            status.error = common_admission_error::probe_failed;
            return status;
        }
#ifdef COMMON_ADMISSION_OWNER_TESTING
        auto config = owner_test ? owner_test->config : common_admission_config_from_env();
#else
        auto config = common_admission_config_from_env();
#endif
        config.degrade_allowed = false;
        common_admission_demand demand;
        demand.transient = total;
        const auto result = common_admission_check(supply, demand, config);
        status.usable = result.usable;
        if (result.verdict != COMMON_ADMISSION_ADMIT) {
            status.error = common_admission_error::rejected;
            try {
                std::fprintf(stderr, "lease admission rejected: %s\n", common_admission_detail(supply, result).c_str());
            } catch (...) {
                std::fputs("lease admission rejected: snapshot formatting failed\n", stderr);
            }
            return status;
        }
        guard.lock();

        if (!checked_add(state.pending, bytes, state.pending)) {
            status.error = common_admission_error::overflow;
            return status;
        }
        ++state.registry_generation;
        pending_ = mine;
        owner_pid_ = pid;
    } catch (...) {
        status.error = common_admission_error::internal;
    }
    return status;
}

bool common_admission_lease::split_pending(uint64_t bytes, common_admission_lease & child) noexcept {
    if (!bytes || &child == this) { return false; }
    try {
        auto & state = commitments();
        std::lock_guard<std::mutex> guard(state.mutex);
        const auto pid = admission_pid();
        if (!pid || state.process != pid || owner_pid_ != pid || bytes > pending_ ||
                child.owner_pid_ || child.pending_ || child.settled_pinned_) {
            return false;
        }
        pending_ -= bytes;
        child.pending_ = bytes;
        child.owner_pid_ = pid;
        ++state.registry_generation;
        return true;
    } catch (...) {
        return false;
    }
}

void common_admission_lease::cancel_pending(uint64_t bytes) noexcept {
    if (!pending_ || owner_pid_ != admission_pid()) return;
    try {
        auto & state = commitments();
        std::lock_guard<std::mutex> guard(state.mutex);
        const uint64_t cancelled = std::min(bytes, pending_);
        if (!cancelled) return;
        state.pending -= cancelled;
        pending_ -= cancelled;
        ++state.registry_generation;
    } catch (...) {

    }
}
common_admission_lease::~common_admission_lease() noexcept { cancel_pending(pending_); }
common_admission_status common_admission_lease::settle_pin(uint64_t bytes) noexcept {
    common_admission_status status;
    status.requested = bytes;
    if (bytes == 0) {
        status.error = common_admission_error::ok;
        return status;
    }
    const int64_t pid = admission_pid();
    if (!pid || (owner_pid_ && owner_pid_ != pid)) {
        status.error = common_admission_error::wrong_process;
        return status;
    }
    try {
        auto & state = commitments();
        if (state.process != pid) {
            status.error = common_admission_error::wrong_process;
            return status;
        }

        constexpr unsigned max_attempts = 4;
        for (unsigned attempt = 1; attempt <= max_attempts; ++attempt) {
            uint64_t owned_before = 0;
            {
                std::lock_guard<std::mutex> guard(state.mutex);
                owned_before = state.owned_pinned;
            }

            uint64_t rss_file = 0;
            uint64_t vm_locked = 0;
            if (!observe_process_status(rss_file, vm_locked)) {
                status.error = common_admission_error::probe_failed;
                return status;
            }
            std::unique_lock<std::mutex> guard(state.mutex);
            if (state.owned_pinned != owned_before) {
                continue;
            }

            admission_pin_math::settle_input in;
            in.pin_bytes    = bytes;
            in.pending      = pending_;
            in.owned_pinned = owned_before;
            in.rss_file     = rss_file;
            in.vm_locked    = vm_locked;
            const admission_pin_math::settle_result verdict = admission_pin_math::settle(in);
            if (verdict.status != admission_pin_math::error::ok) {
                status.error = pin_math_to_gate(verdict.status);
                std::fprintf(stderr, "pack settle proof rejected math_rc=%d gate_rc=%d bytes=%llu rss_file=%llu vm_locked=%llu owned_pinned=%llu lease_pending=%llu total_pending=%llu\n",
                    int(verdict.status), int(status.error), (unsigned long long) bytes,
                    (unsigned long long) rss_file, (unsigned long long) vm_locked,
                    (unsigned long long) owned_before, (unsigned long long) pending_,
                    (unsigned long long) state.pending);
                return status;
            }

            state.pending -= verdict.retire_bytes;
            pending_ -= verdict.retire_bytes;
            settled_pinned_ += verdict.retire_bytes;
            state.owned_pinned = verdict.registered_after;
            ++state.pin_settles;
            ++state.registry_generation;
            status.pending = state.pending;
            status.error = common_admission_error::ok;
            return status;
        }

        status.error = common_admission_error::unstable;
        return status;
    } catch (...) {
        status.error = common_admission_error::internal;
        return status;
    }
}

void common_admission_lease::release_pin(uint64_t bytes) noexcept {
    if (bytes == 0) {
        return;
    }
    const int64_t pid = admission_pid();
    if (!pid || owner_pid_ != pid) {
        return;
    }
    try {
        auto & state = commitments();
        if (state.process != pid) {
            return;
        }
        std::lock_guard<std::mutex> guard(state.mutex);

        uint64_t release = bytes < settled_pinned_ ? bytes : settled_pinned_;
        if (release > state.owned_pinned) {
            release = state.owned_pinned;
        }
        if (release == 0) {
            return;
        }
        state.owned_pinned -= release;
        settled_pinned_ -= release;
        ++state.pin_settles;
        ++state.registry_generation;
    } catch (...) {

    }
}

common_admission_lease::common_admission_lease(common_admission_lease && other) noexcept
    : pending_(std::exchange(other.pending_, 0)),
      settled_pinned_(std::exchange(other.settled_pinned_, 0)),
      owner_pid_(std::exchange(other.owner_pid_, 0)) {}
common_admission_lease & common_admission_lease::operator=(common_admission_lease && other) noexcept {
    if (this != &other) {
        cancel_pending(pending_);
        if (pending_) return *this;
        pending_ = std::exchange(other.pending_, 0);
        settled_pinned_ = std::exchange(other.settled_pinned_, 0);
        owner_pid_ = std::exchange(other.owner_pid_, 0);
    }
    return *this;
}

namespace {
admission_numa::status owner_numa_policy(void * mapping,size_t bytes) noexcept {
#ifdef COMMON_ADMISSION_OWNER_TESTING
    if(owner_test && owner_test->numa_operations)
        return admission_numa::apply(mapping,bytes,*owner_test->numa_operations);
#endif
    return admission_numa::apply_system(mapping,bytes);
}
unsigned owner_touch_page() noexcept {
#ifdef COMMON_ADMISSION_OWNER_TESTING
    return owner_test ? owner_test->fail_touch_page : 1;
#else
    return 1;
#endif
}
bool owner_cleanup_failure() noexcept {
#ifdef COMMON_ADMISSION_OWNER_TESTING
    return owner_test && owner_test->fail_cleanup;
#else
    return false;
#endif
}
int owner_failure_stage() noexcept {
#ifdef COMMON_ADMISSION_OWNER_TESTING
    return owner_test ? owner_test->fail_stage : 0;
#else
    return 0;
#endif
}
}
common_admission_error common_admission_owner::create(uint64_t requested) noexcept {
    if (slot_ >= 0) return common_admission_error::busy;
    if (!requested) return common_admission_error::ok;
#ifndef COMMON_ADMISSION_OWNER_TESTING

    return common_admission_error::not_integrated;
#else
    if (!owner_test) return common_admission_error::not_integrated;
#endif
    const long page = sysconf(_SC_PAGESIZE);
    uint64_t rounded;
    if (page <= 0 || !checked_add(requested, uint64_t(page)-1, rounded)) return common_admission_error::overflow;
    rounded = rounded / uint64_t(page) * uint64_t(page);
    if (rounded > SIZE_MAX) return common_admission_error::overflow;
    common_admission_lease lease;
    auto status = lease.grow(rounded);
    if (status.error != common_admission_error::ok) return status.error;
    auto & state = commitments();
    int index = -1;
    try {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (rounded > UINT64_MAX-state.owned_locked_anon) return common_admission_error::overflow;
        for (int j=0;j<16;++j) if (!state.owners[j].occupied) { index=j; break; }
        if (index < 0) return common_admission_error::registry_full;
        auto & entry=state.owners[index];
        entry.occupied=true; entry.bytes=rounded; entry.thread=std::this_thread::get_id();
        ++state.registry_generation;
        ++state.materializing;
        slot_=index;bytes_=rounded;process_=admission_pid();
        lease.pending_=0;
    }

    struct materializing_guard {
        admission_commitments & state;
        bool active = true;
        ~materializing_guard() {
            if (!active) return;
            try {
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.materializing) --state.materializing;
                ++state.registry_generation;
            } catch (...) {}
        }
    } materializing{state};
    void * memory=MAP_FAILED;
    common_admission_error failure=common_admission_error::mapping_failed;
    if (owner_failure_stage()!=1)
        memory=mmap(nullptr,size_t(rounded),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (memory!=MAP_FAILED) {
        mapping_=memory;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.owners[index].mapping=memory;
            ++state.registry_generation;
        }
        failure=common_admission_error::policy_failed;
        if (owner_failure_stage()!=2 && madvise(memory,size_t(rounded),MADV_NOHUGEPAGE)==0 && (numa_status_=owner_numa_policy(memory,size_t(rounded))).error==admission_numa::code::ok) {
            volatile unsigned char * pages=static_cast<volatile unsigned char *>(memory);
            bool touched=true;
            for(uint64_t off=0;off<rounded;off+=uint64_t(page)) {
                pages[off]=0;
                if(owner_failure_stage()==3 && off/uint64_t(page)+1>=owner_touch_page()){touched=false;break;}
            }
            failure=common_admission_error::lock_failed;
            if(touched && owner_failure_stage()!=4 && mlock(memory,size_t(rounded))==0) {
#ifdef COMMON_ADMISSION_OWNER_TESTING

                if (owner_test != nullptr && owner_test->materialize_point != nullptr) owner_test->materialize_point();
#endif
                std::lock_guard<std::mutex> lock(state.mutex);
                auto & entry=state.owners[index];
                entry.mapping=memory;entry.locked=true;entry.published=true;

                state.pending-=rounded;
                state.owned_locked_anon+=rounded;
                if (state.materializing) --state.materializing;
                materializing.active=false;
                ++state.registry_generation;
                mapping_=memory;bytes_=rounded;slot_=index;process_=admission_pid();published_=true;
                return common_admission_error::ok;
            }
        }
    }

    if (owner_cleanup_failure()) return common_admission_error::cleanup_failed;
    const auto cleaned=close();
    return cleaned==common_admission_error::ok ? failure : cleaned;
    } catch (...) {

        if(slot_>=0) (void)close();
        return common_admission_error::internal;
    }
}

common_admission_error common_admission_owner::close() noexcept {
    if(slot_<0)return common_admission_error::ok;
    if(process_!=admission_pid())return common_admission_error::wrong_process;
    try {
    auto & state=commitments();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto & entry=state.owners[slot_];
    if(!entry.occupied || entry.bytes!=bytes_ ||
       (entry.mapping!=mapping_ && (entry.published || entry.mapping!=nullptr)))return common_admission_error::internal;
    if(entry.thread!=std::this_thread::get_id())return common_admission_error::wrong_thread;
#ifdef COMMON_ADMISSION_OWNER_TESTING
    if(owner_test && owner_test->fail_stage==5)return common_admission_error::cleanup_failed;
#endif
    if(mapping_ && munmap(mapping_,size_t(bytes_))!=0)return common_admission_error::cleanup_failed;
    if(entry.published)state.owned_locked_anon-=bytes_;else state.pending-=bytes_;
    ++state.registry_generation;
    entry={};mapping_=nullptr;bytes_=0;slot_=-1;published_=false;
    return common_admission_error::ok;
    } catch (...) { return common_admission_error::internal; }
}
common_admission_owner::~common_admission_owner() noexcept {
    if(close()!=common_admission_error::ok)
        std::fputs("admission owner cleanup failed; registry charge retained\n",stderr);
}
common_admission_owner::common_admission_owner(common_admission_owner && other) noexcept
    : mapping_(std::exchange(other.mapping_,nullptr)),published_(std::exchange(other.published_,false)),
      numa_status_(std::exchange(other.numa_status_,admission_numa::status{admission_numa::code::ok,0,"moved-from"})),bytes_(std::exchange(other.bytes_,0)),
      slot_(std::exchange(other.slot_,-1)),process_(std::exchange(other.process_,0)) {}

bool common_admission_drop_file_cache(const std::string & path) {
#if defined(__linux__)

    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const int rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    return rc == 0;
#else
    (void) path;
    return false;
#endif
}

#ifdef COMMON_ADMISSION_OWNER_TESTING
void common_admission_pin_counts(uint64_t & registered, unsigned & settles) noexcept {
    auto & state = commitments();
    std::lock_guard<std::mutex> lock(state.mutex);
    registered = state.owned_pinned;
    settles    = state.pin_settles;
}
#endif
