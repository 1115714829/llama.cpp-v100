#include "ggml-expert-store.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>

static bool ggml_expert_store_read_profile(const char * path, ggml_expert_store_profile & profile) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return false;
    }

    int64_t n_layer  = 0;
    int64_t n_expert = 0;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> rows;

    char line[256];
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        int64_t layer, expert, count;
        if (sscanf(line, "%" SCNd64 " %" SCNd64 " %" SCNd64, &layer, &expert, &count) != 3) {
            continue;
        }
        if (layer < 0 || expert < 0 || count < 0) {
            continue;
        }
        n_layer  = std::max(n_layer,  layer + 1);
        n_expert = std::max(n_expert, expert + 1);
        rows.emplace_back(layer, expert, count);
    }
    fclose(f);

    profile.n_layer  = n_layer;
    profile.n_expert = n_expert;
    profile.count.assign((size_t) (n_layer * n_expert), 0);
    for (const auto & row : rows) {
        profile.count[(size_t) (std::get<0>(row) * n_expert + std::get<1>(row))] = std::get<2>(row);
    }
    return true;
}

static std::mutex g_profile_mutex;
static std::string g_profile_path;
static std::string g_dump_path;
static std::string g_trace_path;
static int         g_trace_size_mib = 4;
static std::atomic<bool> g_trace_enabled{false};
static std::atomic<bool> g_dump_enabled{false};

void ggml_expert_store_set_profile(const char * path) {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_profile_path = path != nullptr ? path : "";
}

static std::string ggml_expert_store_get_profile() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_profile_path;
}

void ggml_expert_store_set_dump_path(const char * path) {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_dump_path = path != nullptr ? path : "";
    g_dump_enabled.store(!g_dump_path.empty(), std::memory_order_relaxed);
}

bool ggml_expert_store_dump_enabled() {
    return g_dump_enabled.load(std::memory_order_relaxed);
}

static std::string ggml_expert_store_get_dump_path() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_dump_path;
}

void ggml_expert_store_set_trace_path(const char * path, int size_mib) {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_trace_path     = path != nullptr ? path : "";
    g_trace_size_mib = size_mib > 0 ? size_mib : 4;
    g_trace_enabled.store(!g_trace_path.empty(), std::memory_order_relaxed);
}

bool ggml_expert_store_trace_enabled() {
    return g_trace_enabled.load(std::memory_order_relaxed);
}

std::string ggml_expert_store_get_trace_path() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_trace_path;
}

int ggml_expert_store_get_trace_size_mib() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_trace_size_mib;
}

void ggml_expert_store_dump_profile(ggml_expert_store * store, int n_device) {
    if (store == nullptr) {
        return;
    }
    std::string path = ggml_expert_store_get_dump_path();
    if (path.empty()) {
        return;
    }

    // the profile file is per (layer, expert) and does not carry the device; with several
    // stores (one per device) each one writes its own file next to the given path
    if (n_device > 1) {
        path += ".dev" + std::to_string(store->device);
    }

    // tensors of the same layer (gate/up/down) share one profile row: sum their counts.
    // With expert parallelism a store holds one device's segment of the experts; which global
    // expert a local one is comes from the table the loader laid the tensor out by, recorded by
    // the placement check (ggml_expert_store_tensor::global_of): the layout the kernels counted on.
    int64_t n_layer = 0;
    for (const ggml_expert_store_tensor & t : store->tensors) {
        if (!t.stat_experts.empty() && t.layer >= 0) {
            n_layer = std::max(n_layer, (int64_t) t.layer + 1);
        }
    }
    if (n_layer == 0) {
        return;
    }

    std::vector<std::vector<int64_t>> counts((size_t) n_layer);
    bool any = false;
    for (const ggml_expert_store_tensor & t : store->tensors) {
        if (t.stat_experts.empty() || t.layer < 0) {
            continue;
        }
        if ((int64_t) t.global_of.size() != t.n_expert || t.stat_experts.size() > t.global_of.size()) {
            GGML_LOG_ERROR("%s: device %d: layer %d: experts [%" PRId64 ", %" PRId64 ") have no global numbering (the loader did not place this tensor), the counts of this tensor are not written\n",
                __func__, store->device, t.layer, t.expert_base, t.expert_base + t.n_expert);
            continue;
        }
        std::vector<int64_t> & row = counts[(size_t) t.layer];
        for (size_t e = 0; e < t.stat_experts.size(); ++e) {
            const int64_t g = t.global_of[e];
            GGML_ASSERT(g >= 0);
            row.resize(std::max<size_t>(row.size(), (size_t) g + 1), 0);
            row[(size_t) g] += t.stat_experts[e];
            any = any || t.stat_experts[e] > 0;
        }
    }
    if (!any) {
        // nothing was routed through this store (e.g. a memory-fit load without inference);
        // keep whatever profile file exists instead of overwriting it with zeros
        return;
    }

    FILE * f = fopen(path.c_str(), "w");
    if (f == nullptr) {
        GGML_LOG_ERROR("%s: cannot write expert routing profile '%s'\n", __func__, path.c_str());
        return;
    }
    fprintf(f, "# expert routing profile: layer expert count\n");
    fprintf(f, "# generated from runtime routing statistics, one run can feed the next one\n");
    for (int64_t l = 0; l < n_layer; ++l) {
        const std::vector<int64_t> & row = counts[(size_t) l];
        for (size_t e = 0; e < row.size(); ++e) {
            if (row[e] > 0) {
                fprintf(f, "%" PRId64 " %zu %" PRId64 "\n", l, e, row[e]);
            }
        }
    }
    fclose(f);
    GGML_LOG_INFO("%s: device %d: wrote expert routing profile '%s'\n", __func__, store->device, path.c_str());

    for (ggml_expert_store_tensor & t : store->tensors) {
        t.stat_experts.clear();
    }
}

ggml_expert_store * ggml_expert_store_init(const ggml_expert_store_backend * backend, int device, const char * profile_path) {
    GGML_ASSERT(backend != nullptr);

    ggml_expert_store * store = new ggml_expert_store();
    store->backend = backend;
    store->device  = device;

    if (profile_path != nullptr && profile_path[0] != '\0') {
        store->profile_path = profile_path;
    } else {
        store->profile_path = ggml_expert_store_get_profile();
    }

    if (!store->profile_path.empty()) {
        GGML_LOG_INFO("%s: device %d: routing profile '%s'\n", __func__, device, store->profile_path.c_str());
    } else {
        GGML_LOG_INFO("%s: device %d: no routing profile, experts are placed uniformly within the layer quota\n",
            __func__, device);
    }

    return store;
}

void ggml_expert_store_free(ggml_expert_store * store) {
    if (store == nullptr) {
        return;
    }

    for (ggml_expert_store_tensor & t : store->tensors) {
        if (t.slots_dev != nullptr) {
            store->backend->free_device(store->backend->user, t.slots_dev);
        }
        if (t.host_host != nullptr) {
            store->backend->free_host(store->backend->user, t.host_host);
        }
        delete[] t.table_host;
    }
    delete store;
}

// What the loader registered about the experts of a tensor it wrote, by tensor name, for the
// placement check of the backends (ggml_expert_store_verify_placement): the hash and presence of
// every global expert, and the table it laid the experts out by (0: file order, the tensor holds the
// whole expert dim). Cleared once the backends checked the placement of the model.
struct ggml_expert_store_source_hashes {
    std::vector<uint64_t> hash;
    std::vector<uint8_t>  present;
    uint32_t              assignment = 0;
};

static std::mutex g_source_hash_mutex;
static std::map<std::string, ggml_expert_store_source_hashes> g_source_hash;

// Expert assignment: which device holds which expert when the routed experts of a layer are split
// over the devices along the expert dim.
//
// The proposal is computed from the routing profile while the expert buffers are planned
// (ggml_expert_store_plan -> ggml_expert_store_compute_assignment), and read through
// ggml_expert_store_owner() and friends: the plan ranks the experts of a store slice by the profile
// row of the global expert the proposal puts there.
//
// The tables (ggml_expert_assignment_*) are what loaded data follows. A model loader creates one
// table for the expert weights it splits over the devices - a snapshot of the proposal when its
// experts live in the expert stores, the contiguous split of its slices when they are plain device
// weights -, lays the experts out by it and activates it, which binds the weights of the model to
// the table. A MUL_MAT_ID node reads the table of its own weight (the meta backend writes it into the
// node from that binding, see ggml_mul_mat_id_set_assignment), so two models with different layouts
// - the target and its MTP draft - never read each other's table. A table never changes once it is
// created and lives until its model releases it; table ids are never reused.
struct ggml_expert_store_assignment {
    std::string          profile_path;
    std::vector<int64_t> profile_count; // counts the table was computed from
    bool     has_profile = false;
    int      n_dev       = 0;
    int      n_layer     = 0;
    int      n_expert    = 0;
    int      cap_max     = 0;
    std::vector<int>     cap;   // [n_dev] experts per device
    std::vector<int64_t> cap_prefix; // [n_dev + 1] experts of the devices before j: start of segment j
    std::vector<int32_t> owner; // [n_layer * n_expert] device of an expert
    std::vector<int16_t> local; // [n_dev * n_layer * n_expert] local number, -1 when not local
    std::vector<int32_t> rev;   // [n_dev * n_layer * cap_max] global number of a local expert
    double   imbalance_contiguous = 0.0;
    double   imbalance_permuted    = 0.0;
    int      worst_layer           = -1;
    double   worst_layer_imbalance = 0.0;
};

static std::mutex g_assign_mutex;
static ggml_expert_store_assignment g_assign; // the proposal
static bool g_assign_valid = false;
// most experts of any slice the plans have seen so far. Without a profile it is the only source
// of the global expert count: every device plans its own slice, the whole count is known once
// the last one did. A profile names the whole count itself.
static int64_t g_assign_n_expert_seen = 0;

// the table depends on nothing but the counts, the device count and the sizes, so an unchanged
// input reuses the table of the first computation
static bool ggml_expert_store_assignment_matches(
        const char * path, const ggml_expert_store_profile * profile,
        int n_dev, int n_layer, int n_expert) {
    if (g_assign.n_dev != n_dev || g_assign.n_layer != n_layer || g_assign.n_expert != n_expert) {
        return false;
    }
    const bool has = profile != nullptr && !profile->count.empty();
    if (g_assign.has_profile != has) {
        return false;
    }
    return !has || (g_assign.profile_count == profile->count &&
                    g_assign.profile_path == std::string(path != nullptr ? path : ""));
}

// mean over the layers of max(device load) / mean(device load); 0 for a layer without traffic
static double ggml_expert_store_imbalance(const std::vector<int64_t> & load, int n_dev) {
    int64_t total = 0;
    int64_t peak  = 0;
    for (const int64_t v : load) {
        total += v;
        peak = std::max(peak, v);
    }
    if (total <= 0 || n_dev <= 0) {
        return 0.0;
    }
    return (double) peak * n_dev / (double) total;
}

// sizes, quotas and empty tables of an assignment of n_expert experts over n_dev devices, device j
// holding cap[j] experts of every layer
static void ggml_expert_store_assignment_init_caps(ggml_expert_store_assignment & a, int n_dev, int n_layer, int n_expert,
        const std::vector<int> & cap) {
    GGML_ASSERT(n_dev > 0 && n_layer > 0 && n_expert > 0 && (int) cap.size() == n_dev);
    a.n_dev    = n_dev;
    a.n_layer  = n_layer;
    a.n_expert = n_expert;
    a.cap_max  = 0;
    a.cap      = cap;
    a.cap_prefix.assign((size_t) n_dev + 1, 0);
    for (int j = 0; j < n_dev; ++j) {
        GGML_ASSERT(cap[(size_t) j] >= 0);
        a.cap_max = std::max(a.cap_max, cap[(size_t) j]);
        a.cap_prefix[(size_t) j + 1] = a.cap_prefix[(size_t) j] + cap[(size_t) j];
    }
    GGML_ASSERT(a.cap_prefix[(size_t) n_dev] == n_expert);
    const size_t n_pair = (size_t) n_layer * n_expert;
    a.owner.assign(n_pair, 0);
    a.local.assign(n_pair * (size_t) n_dev, -1);
    a.rev.assign((size_t) n_dev * (size_t) n_layer * (size_t) a.cap_max, -1);
}

// the quota of device j is the integer slice of the weight split in llama-model.cpp, so the
// per-device byte counts of the contiguous split are kept whatever the owners are (512 experts
// over 6 devices: 85/85/86/85/85/86)
static void ggml_expert_store_assignment_init(ggml_expert_store_assignment & a, int n_dev, int n_layer, int n_expert) {
    GGML_ASSERT(n_dev > 0);
    std::vector<int> cap((size_t) n_dev, 0);
    for (int j = 0; j < n_dev; ++j) {
        const int64_t hi = (int64_t) n_expert * (j + 1) / n_dev;
        const int64_t lo = (int64_t) n_expert * j         / n_dev;
        cap[(size_t) j] = (int) (hi - lo);
    }
    ggml_expert_store_assignment_init_caps(a, n_dev, n_layer, n_expert, cap);
}

// owners of layer l of the contiguous split: device jj holds the experts of its own segment of the
// numbering, which is how the meta backend slices the weights
static void ggml_expert_store_assignment_contiguous_layer(ggml_expert_store_assignment & a, int l) {
    int base = 0;
    for (int jj = 0; jj < a.n_dev; ++jj) {
        for (int e = 0; e < a.cap[(size_t) jj]; ++e) {
            GGML_ASSERT(base + e < a.n_expert);
            a.owner[(size_t) l * a.n_expert + (size_t) (base + e)] = jj;
        }
        base += a.cap[(size_t) jj];
    }
}

// local numbers of layer l from its owners: the experts of one device in ascending global order,
// so a device that holds all of them keeps slot number == local number
static void ggml_expert_store_assignment_number_layer(ggml_expert_store_assignment & a, int l) {
    std::vector<int> nxt((size_t) a.n_dev, 0);
    for (int e = 0; e < a.n_expert; ++e) {
        const int j = a.owner[(size_t) l * a.n_expert + e];
        const int k = nxt[(size_t) j]++;
        GGML_ASSERT(k < a.cap[(size_t) j]);
        a.local[((size_t) j * a.n_layer + l) * a.n_expert + e] = (int16_t) k;
        a.rev [((size_t) j * a.n_layer + l) * a.cap_max + k] = e;
    }
}

static uint64_t ggml_expert_store_hash_range(uint64_t h, const uint8_t * p, size_t n);

// fingerprint of a whole table (every owner and local number), so two starts can be compared
// by one log line
static uint64_t ggml_expert_store_assignment_fingerprint(const ggml_expert_store_assignment & a) {
    uint64_t h = 0x243F6A8885A308D3ULL ^ ((uint64_t) a.n_dev << 40) ^ ((uint64_t) a.n_layer << 20) ^ (uint64_t) a.n_expert;
    h = ggml_expert_store_hash_range(h, (const uint8_t *) a.owner.data(), a.owner.size() * sizeof(int32_t));
    h = ggml_expert_store_hash_range(h, (const uint8_t *) a.local.data(), a.local.size() * sizeof(int16_t));
    return h;
}

// replace the contiguous split by a per-layer LPT split: the experts of a layer go to the least
// loaded device, the hottest first. The number of experts per device stays the one of the
// contiguous split (see cap above), so the bytes per device do not change. Without a profile
// the split stays contiguous: the routing counts are unknown then, and the contiguous split is
// the only placement that can be correct - a capability branch, not a fallback.
// The result is the proposal; a loader takes a snapshot of it as its table
// (ggml_expert_assignment_from_proposal) and lays its experts out by that.
void ggml_expert_store_compute_assignment(
        const char * path, const ggml_expert_store_profile * profile,
        int n_dev, int n_layer, int n_expert) {
    GGML_ASSERT(n_dev > 0 && n_layer > 0 && n_expert > 0);

    std::lock_guard<std::mutex> lock(g_assign_mutex);

    const bool has_profile = profile != nullptr && !profile->count.empty();

    // a plan covers the slice of one device: without a profile the global expert count is the
    // largest global index any of them has seen; a profile names the whole count itself
    if (has_profile) {
        g_assign_n_expert_seen = n_expert;
    } else {
        g_assign_n_expert_seen = std::max(g_assign_n_expert_seen, (int64_t) n_expert);
        n_expert = (int) g_assign_n_expert_seen;
    }

    if (g_assign_valid &&
        ggml_expert_store_assignment_matches(path, profile, n_dev, n_layer, n_expert)) {
        return;
    }

    // the tables grow with n_dev * n_layer * n_expert entries, so a caller that passes sizes of
    // a different model must be caught here instead of taking the process down with them
    if ((int64_t) n_layer * n_expert > (int64_t) 1 << 22 || n_expert > INT16_MAX) {
        GGML_LOG_ERROR("%s: %d layers x %d experts is too large for an expert assignment\n",
            __func__, n_layer, n_expert);
        return;
    }

    ggml_expert_store_assignment a;
    ggml_expert_store_assignment_init(a, n_dev, n_layer, n_expert);
    a.has_profile   = has_profile;
    a.profile_path  = a.has_profile ? (path != nullptr ? path : "") : "";
    a.profile_count = a.has_profile ? profile->count : std::vector<int64_t>();

    const auto count_of = [&](int l, int e) -> int64_t {
        if (!a.has_profile || l >= profile->n_layer || e >= profile->n_expert) {
            return 0;
        }
        return profile->at(l, e);
    };

    std::vector<int>      order((size_t) n_expert);
    std::vector<int64_t>  load((size_t) n_dev, 0);
    std::vector<int>      used((size_t) n_dev, 0);
    std::vector<int64_t>  load_split((size_t) n_dev, 0);
    std::vector<int64_t>  load_perm ((size_t) n_dev, 0);

    double sum_split = 0.0;
    double sum_perm  = 0.0;
    double worst     = 0.0;

    for (int l = 0; l < n_layer; ++l) {
        if (a.has_profile) {
            // hottest expert first, expert number breaks the ties: same profile, same table
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
                return count_of(l, x) > count_of(l, y);
            });
            std::fill(load.begin(), load.end(), 0);
            std::fill(used.begin(), used.end(), 0);
            for (const int e : order) {
                int best = -1;
                for (int j = 0; j < n_dev; ++j) {
                    if (used[(size_t) j] >= a.cap[(size_t) j]) {
                        continue; // this device already holds its quota of experts
                    }
                    if (best < 0 || load[(size_t) j] < load[(size_t) best]) {
                        best = j; // tie: the lowest device number stays
                    }
                }
                GGML_ASSERT(best >= 0);
                a.owner[(size_t) l * n_expert + e] = best;
                load  [(size_t) best] += count_of(l, e);
                used  [(size_t) best]++;
            }
        } else {
            ggml_expert_store_assignment_contiguous_layer(a, l);
        }
        ggml_expert_store_assignment_number_layer(a, l);

        std::fill(load_split.begin(), load_split.end(), 0);
        std::fill(load_perm.begin(),  load_perm.end(),  0);
        int base = 0;
        for (int jj = 0; jj < n_dev; ++jj) {
            for (int e = 0; e < a.cap[(size_t) jj]; ++e) {
                load_split[(size_t) jj] += count_of(l, base + e);
            }
            base += a.cap[(size_t) jj];
        }
        for (int e = 0; e < n_expert; ++e) {
            load_perm[(size_t) a.owner[(size_t) l * n_expert + e]] += count_of(l, e);
        }

        const double imb_split = ggml_expert_store_imbalance(load_split, n_dev);
        const double imb_perm  = ggml_expert_store_imbalance(load_perm,  n_dev);
        sum_split += imb_split;
        sum_perm  += imb_perm;
        if (a.worst_layer < 0 || imb_perm > worst) {
            a.worst_layer           = l;
            a.worst_layer_imbalance = imb_perm;
            worst = imb_perm;
        }
        GGML_LOG_DEBUG("%s: expert assignment: layer %d: max/avg %.3f -> %.3f\n",
            __func__, l, imb_split, imb_perm);
    }

    a.imbalance_contiguous = sum_split / n_layer;
    a.imbalance_permuted    = sum_perm  / n_layer;

    g_assign      = std::move(a);
    g_assign_valid = true;

    GGML_LOG_INFO("%s: expert assignment: profile=%s layers=%d experts=%d devs=%d max/avg %.3f -> %.3f (worst layer %d: %.3f)\n",
        __func__, g_assign.has_profile ? g_assign.profile_path.c_str() : "none",
        g_assign.n_layer, g_assign.n_expert, g_assign.n_dev,
        g_assign.imbalance_contiguous, g_assign.imbalance_permuted,
        g_assign.worst_layer, worst);
}

static int ggml_expert_store_table_owner(const ggml_expert_store_assignment & a, int layer, int expert) {
    if (layer < 0 || layer >= a.n_layer || expert < 0 || expert >= a.n_expert) {
        return -1;
    }
    return a.owner[(size_t) layer * a.n_expert + expert];
}

static const int16_t * ggml_expert_store_table_local_of(const ggml_expert_store_assignment & a, int dev, int layer) {
    if (dev < 0 || dev >= a.n_dev || layer < 0 || layer >= a.n_layer) {
        return nullptr;
    }
    return a.local.data() + ((size_t) dev * a.n_layer + layer) * a.n_expert;
}

static int ggml_expert_store_table_global_of(const ggml_expert_store_assignment & a, int dev, int layer, int local) {
    if (dev < 0 || dev >= a.n_dev || layer < 0 || layer >= a.n_layer) {
        return -1;
    }
    if (local < 0 || local >= a.cap[(size_t) dev]) {
        return -1;
    }
    return a.rev[((size_t) dev * a.n_layer + layer) * a.cap_max + local];
}

// the device whose segment is exactly [expert_base, expert_base + n_expert): >= 0 that device, -1
// the whole expert dim, -2 no segment
static int ggml_expert_store_table_slice_device(const ggml_expert_store_assignment & a, int64_t expert_base, int64_t n_expert) {
    if (expert_base == 0 && n_expert == a.n_expert) {
        return -1; // the whole expert dim: the tensor is not split over the devices
    }
    for (int j = 0; j < a.n_dev; ++j) {
        if (a.cap_prefix[(size_t) j] == expert_base && a.cap[(size_t) j] == n_expert) {
            return j;
        }
    }
    return -2;
}

int ggml_expert_store_owner(int layer, int expert) {
    std::lock_guard<std::mutex> lock(g_assign_mutex);
    return g_assign_valid ? ggml_expert_store_table_owner(g_assign, layer, expert) : -1;
}

const int16_t * ggml_expert_store_local_of(int dev, int layer) {
    std::lock_guard<std::mutex> lock(g_assign_mutex);
    return g_assign_valid ? ggml_expert_store_table_local_of(g_assign, dev, layer) : nullptr;
}

int ggml_expert_store_global_of(int dev, int layer, int local) {
    std::lock_guard<std::mutex> lock(g_assign_mutex);
    return g_assign_valid ? ggml_expert_store_table_global_of(g_assign, dev, layer, local) : -1;
}

int ggml_expert_store_slice_device(int64_t expert_base, int64_t n_expert) {
    std::lock_guard<std::mutex> lock(g_assign_mutex);
    return g_assign_valid ? ggml_expert_store_table_slice_device(g_assign, expert_base, n_expert) : -2;
}

bool ggml_expert_store_assignment_info_get(ggml_expert_store_assignment_info * info) {
    if (info == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_assign_mutex);
    if (!g_assign_valid) {
        return false;
    }
    info->n_dev       = g_assign.n_dev;
    info->n_layer     = g_assign.n_layer;
    info->n_expert    = g_assign.n_expert;
    info->cap_max     = g_assign.cap_max;
    info->has_profile = g_assign.has_profile;
    info->imbalance_contiguous = g_assign.imbalance_contiguous;
    info->imbalance_permuted    = g_assign.imbalance_permuted;
    info->worst_layer           = g_assign.worst_layer;
    info->worst_layer_imbalance = g_assign.worst_layer_imbalance;
    return true;
}

// ---- the tables

struct ggml_expert_assignment_table {
    uint32_t    id       = 0;
    bool        active   = false; // the loader placed the data and bound its weights
    std::string what;              // where the table came from, for the logs
    uint64_t    fingerprint = 0;
    ggml_expert_store_assignment a;
};

static std::mutex g_table_mutex;
static std::unordered_map<uint32_t, std::unique_ptr<ggml_expert_assignment_table>> g_tables;
static uint32_t g_table_next_id = 1;
// the table every bound expert weight (the tensor of the model, not a view of it) was laid out by
static std::unordered_map<const ggml_tensor *, uint32_t> g_weight_tables;

// the tensor that owns the data of a weight: a MUL_MAT_ID may read a view of the model tensor
static const ggml_tensor * ggml_expert_assignment_weight_base(const ggml_tensor * w) {
    int depth = 0;
    while (w != nullptr && w->view_src != nullptr && depth++ < 16) {
        w = w->view_src;
    }
    return w;
}

// every row of every device names exactly cap[dev] local experts, each local number once, and the
// reverse rows agree: the kernels index the weight slice of a node by these numbers. Checked once
// when the table is made, the node lookups then only compare the slice of the node.
static void ggml_expert_assignment_validate(const ggml_expert_store_assignment & a) {
    std::vector<uint8_t> seen((size_t) std::max(a.cap_max, 1));
    for (int j = 0; j < a.n_dev; ++j) {
        for (int l = 0; l < a.n_layer; ++l) {
            const int16_t * row = a.local.data() + ((size_t) j * a.n_layer + l) * a.n_expert;
            std::fill(seen.begin(), seen.end(), 0);
            int n = 0;
            for (int e = 0; e < a.n_expert; ++e) {
                const int v = row[e];
                if (v < 0) {
                    continue;
                }
                if (v >= a.cap[(size_t) j] || seen[(size_t) v] != 0 ||
                        a.rev[((size_t) j * a.n_layer + l) * a.cap_max + v] != e ||
                        a.owner[(size_t) l * a.n_expert + e] != j) {
                    GGML_ABORT("%s: expert assignment row (device %d, layer %d) numbers expert %d as local %d inconsistently\n",
                        __func__, j, l, e, v);
                }
                seen[(size_t) v] = 1;
                n++;
            }
            if (n != a.cap[(size_t) j]) {
                GGML_ABORT("%s: expert assignment row (device %d, layer %d) holds %d experts, the quota of the device is %d\n",
                    __func__, j, l, n, a.cap[(size_t) j]);
            }
        }
    }
}

static uint32_t ggml_expert_assignment_add(ggml_expert_store_assignment && a, const char * what) {
    ggml_expert_assignment_validate(a);

    auto t = std::make_unique<ggml_expert_assignment_table>();
    t->what        = what;
    t->fingerprint = ggml_expert_store_assignment_fingerprint(a);
    t->a           = std::move(a);

    std::lock_guard<std::mutex> lock(g_table_mutex);
    // the id travels in an int32 op param of the MUL_MAT_ID nodes
    GGML_ASSERT(g_table_next_id < (uint32_t) INT32_MAX);
    t->id = g_table_next_id++;
    const uint32_t id = t->id;
    GGML_LOG_DEBUG("%s: expert assignment %u: %s (devs=%d layers=%d experts=%d) table %016" PRIx64 "\n",
        __func__, id, t->what.c_str(), t->a.n_dev, t->a.n_layer, t->a.n_expert, t->fingerprint);
    g_tables.emplace(id, std::move(t));
    return id;
}

// the table of an id; the caller holds g_table_mutex
static const ggml_expert_assignment_table * ggml_expert_assignment_find(uint32_t id) {
    const auto it = g_tables.find(id);
    return it != g_tables.end() ? it->second.get() : nullptr;
}

uint32_t ggml_expert_assignment_from_proposal(int n_dev, int n_layer, int n_expert) {
    ggml_expert_store_assignment a;
    {
        std::lock_guard<std::mutex> lock(g_assign_mutex);
        if (!g_assign_valid || g_assign.n_dev != n_dev || g_assign.n_expert != n_expert || g_assign.n_layer < n_layer) {
            GGML_LOG_ERROR("%s: no expert assignment of %d devices, %d experts and at least %d layers was planned "
                "(the plan of the expert buffers has %s)\n", __func__, n_dev, n_expert, n_layer,
                g_assign_valid ? "other sizes" : "not run");
            if (g_assign_valid) {
                GGML_LOG_ERROR("%s: the planned assignment: devs=%d layers=%d experts=%d\n",
                    __func__, g_assign.n_dev, g_assign.n_layer, g_assign.n_expert);
            }
            return 0;
        }
        a = g_assign;
    }
    const char * what = a.has_profile ? "permuted by the routing profile" : "the proposal (contiguous, no profile)";
    return ggml_expert_assignment_add(std::move(a), what);
}

uint32_t ggml_expert_assignment_contiguous(int n_dev, int n_layer, int n_expert, const int64_t * caps) {
    if (n_dev <= 0 || n_layer <= 0 || n_expert <= 0 || n_expert > INT16_MAX ||
            (int64_t) n_layer * n_expert > (int64_t) 1 << 22 || caps == nullptr) {
        GGML_LOG_ERROR("%s: %d devices, %d layers x %d experts cannot be an expert assignment\n",
            __func__, n_dev, n_layer, n_expert);
        return 0;
    }
    std::vector<int> cap((size_t) n_dev, 0);
    int64_t sum = 0;
    for (int j = 0; j < n_dev; ++j) {
        if (caps[j] < 0) {
            GGML_LOG_ERROR("%s: device %d holds %" PRId64 " experts\n", __func__, j, caps[j]);
            return 0;
        }
        cap[(size_t) j] = (int) caps[j];
        sum += caps[j];
    }
    if (sum != n_expert) {
        GGML_LOG_ERROR("%s: the devices hold %" PRId64 " experts, the weight has %d\n", __func__, sum, n_expert);
        return 0;
    }
    ggml_expert_store_assignment a;
    ggml_expert_store_assignment_init_caps(a, n_dev, n_layer, n_expert, cap);
    for (int l = 0; l < n_layer; ++l) {
        ggml_expert_store_assignment_contiguous_layer(a, l);
        ggml_expert_store_assignment_number_layer(a, l);
    }
    return ggml_expert_assignment_add(std::move(a), "contiguous split");
}

bool ggml_expert_assignment_dims(uint32_t table, int * n_dev, int * n_layer, int * n_expert) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    if (t == nullptr) {
        return false;
    }
    if (n_dev    != nullptr) { *n_dev    = t->a.n_dev;    }
    if (n_layer  != nullptr) { *n_layer  = t->a.n_layer;  }
    if (n_expert != nullptr) { *n_expert = t->a.n_expert; }
    return true;
}

int ggml_expert_assignment_cap(uint32_t table, int dev) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    if (t == nullptr || dev < 0 || dev >= t->a.n_dev) {
        return -1;
    }
    return t->a.cap[(size_t) dev];
}

// where the loader puts expert `expert`: the segments of the devices follow each other in device
// order and segment j holds the cap[j] experts of device j in local order
int64_t ggml_expert_assignment_load_position(uint32_t table, int layer, int expert) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    if (t == nullptr) {
        return -1;
    }
    const int dev = ggml_expert_store_table_owner(t->a, layer, expert);
    if (dev < 0) {
        return -1;
    }
    const int16_t local = t->a.local[((size_t) dev * t->a.n_layer + layer) * t->a.n_expert + expert];
    GGML_ASSERT(local >= 0);
    return t->a.cap_prefix[(size_t) dev] + local;
}

void ggml_expert_assignment_activate(uint32_t table, const ggml_tensor * const * weights, size_t n_weights) {
    std::vector<std::string> names;
    names.reserve(n_weights);
    {
        std::lock_guard<std::mutex> lock(g_table_mutex);
        const auto it = g_tables.find(table);
        if (it == g_tables.end()) {
            GGML_ABORT("%s: there is no expert assignment %u\n", __func__, table);
        }
        ggml_expert_assignment_table & t = *it->second;
        if (t.active) {
            GGML_ABORT("%s: expert assignment %u is active already\n", __func__, table);
        }
        for (size_t i = 0; i < n_weights; ++i) {
            const ggml_tensor * base = ggml_expert_assignment_weight_base(weights[i]);
            GGML_ASSERT(base != nullptr);
            const auto res = g_weight_tables.emplace(base, table);
            if (!res.second && res.first->second != table) {
                GGML_ABORT("%s: weight %s was laid out by expert assignment %u already\n",
                    __func__, base->name, res.first->second);
            }
            names.emplace_back(weights[i]->name);
        }
        t.active = true;
        GGML_LOG_INFO("%s: expert assignment version %u active: %s (profile=%s devs=%d layers=%d experts=%d) max/avg %.3f table %016" PRIx64 ", %zu weights\n",
            __func__, table, t.what.c_str(), t.a.has_profile ? t.a.profile_path.c_str() : "none",
            t.a.n_dev, t.a.n_layer, t.a.n_expert, t.a.imbalance_permuted, t.fingerprint, n_weights);
    }

    // the placement check of the backends (ggml_expert_store_verify_placement) learns here which
    // table the loader laid every tensor out by
    std::lock_guard<std::mutex> lock(g_source_hash_mutex);
    for (const std::string & name : names) {
        g_source_hash[name].assignment = table;
    }
}

void ggml_expert_assignment_release(uint32_t table) {
    if (table == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_table_mutex);
    for (auto it = g_weight_tables.begin(); it != g_weight_tables.end(); ) {
        if (it->second == table) {
            it = g_weight_tables.erase(it);
        } else {
            ++it;
        }
    }
    g_tables.erase(table);
}

int32_t ggml_expert_assignment_of_weight(const ggml_tensor * weight) {
    const ggml_tensor * base = ggml_expert_assignment_weight_base(weight);
    if (base == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const auto it = g_weight_tables.find(base);
    return it != g_weight_tables.end() ? (int32_t) it->second : -1;
}

bool ggml_expert_assignment_is_active(uint32_t table) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    return t != nullptr && t->active;
}

int ggml_expert_assignment_owner(uint32_t table, int layer, int expert) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    return t != nullptr ? ggml_expert_store_table_owner(t->a, layer, expert) : -1;
}

const int16_t * ggml_expert_assignment_local_of(uint32_t table, int dev, int layer) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    return t != nullptr ? ggml_expert_store_table_local_of(t->a, dev, layer) : nullptr;
}

int ggml_expert_assignment_global_of(uint32_t table, int dev, int layer, int local) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    return t != nullptr ? ggml_expert_store_table_global_of(t->a, dev, layer, local) : -1;
}

int ggml_expert_assignment_slice_device(uint32_t table, int64_t expert_base, int64_t n_expert) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    return t != nullptr ? ggml_expert_store_table_slice_device(t->a, expert_base, n_expert) : -2;
}

uint64_t ggml_expert_assignment_fingerprint(uint32_t table) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find(table);
    return t != nullptr ? t->fingerprint : 0;
}

size_t ggml_expert_assignment_count(void) {
    std::lock_guard<std::mutex> lock(g_table_mutex);
    return g_tables.size();
}

// ---- the expert map of a MUL_MAT_ID node (see ggml-impl.h for the contract)

// contiguous segment rows, one per distinct (expert_offset, n_local, n_total); a row never changes,
// so the pointers stay valid for the process
static std::mutex g_segment_mutex;
static std::unordered_map<uint64_t, std::vector<int16_t>> g_segment_rows;

static const int16_t * ggml_expert_segment_row(int expert_offset, int n_local, int n_total) {
    GGML_ASSERT(n_local > 0 && n_total >= n_local && n_total <= INT16_MAX);
    GGML_ASSERT(expert_offset >= 0 && expert_offset + n_local <= n_total);
    const uint64_t key = (uint64_t) expert_offset << 32 | (uint64_t) n_local << 16 | (uint64_t) n_total;
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    auto it = g_segment_rows.find(key);
    if (it == g_segment_rows.end()) {
        std::vector<int16_t> row((size_t) n_total, (int16_t) -1);
        for (int e = 0; e < n_local; ++e) {
            row[(size_t) (expert_offset + e)] = (int16_t) e;
        }
        it = g_segment_rows.emplace(key, std::move(row)).first;
    }
    return it->second.data();
}

const int16_t * ggml_expert_store_node_lut(const ggml_tensor * node) {
    GGML_ASSERT(node != nullptr && node->op == GGML_OP_MUL_MAT_ID && node->src[0] != nullptr);
    const int n_local = (int) node->src[0]->ne[2];

    // without expert parallelism every id is a local expert and keeps its number
    if (!ggml_mul_mat_id_is_expert_parallel(node)) {
        return ggml_expert_segment_row(0, n_local, n_local);
    }

    const int     n_total = ggml_mul_mat_id_get_n_expert_total(node);
    const int     offset  = ggml_mul_mat_id_get_expert_offset(node);
    const int32_t table   = ggml_mul_mat_id_get_assignment(node);
    if (n_local <= 0 || n_total < n_local || n_total > INT16_MAX || offset < 0 || offset + n_local > n_total) {
        GGML_ABORT("%s: node %s: expert state offset %d, local %d, total %d is inconsistent\n",
            __func__, node->name, offset, n_local, n_total);
    }
    if (table < 0) {
        GGML_ABORT("%s: node %s: the weight %s is split over the devices along the expert dim but was laid out by no "
            "expert assignment (its loader binds every such weight to the table of its model)\n",
            __func__, node->name, node->src[0]->name);
    }
    if (table == 0) {
        // a node built with its expert range in the op params (the tests): the weight holds the
        // contiguous segment [offset, offset + n_local) of the numbering
        return ggml_expert_segment_row(offset, n_local, n_total);
    }

    const int device = ggml_mul_mat_id_get_device(node);
    const int layer  = ggml_mul_mat_id_get_layer(node);

    std::lock_guard<std::mutex> lock(g_table_mutex);
    const ggml_expert_assignment_table * t = ggml_expert_assignment_find((uint32_t) table);
    if (t == nullptr) {
        GGML_ABORT("%s: node %s reads expert assignment %d, which does not exist (released with its model)\n",
            __func__, node->name, table);
    }
    if (!t->active) {
        GGML_ABORT("%s: node %s reads expert assignment %d before its loader activated it\n", __func__, node->name, table);
    }
    const ggml_expert_store_assignment & a = t->a;
    if (device < 0 || device >= a.n_dev || layer < 0 || layer >= a.n_layer || n_total != a.n_expert) {
        GGML_ABORT("%s: node %s (device %d, layer %d, %d of %d experts) has no row in expert assignment %d "
            "(devs=%d layers=%d experts=%d)\n", __func__, node->name, device, layer, n_local, n_total,
            table, a.n_dev, a.n_layer, a.n_expert);
    }
    // the slice the meta backend gave the device must be the segment of the device in the table:
    // the row numbers exactly cap[device] experts and the weight slice has to hold that many
    if (n_local != a.cap[(size_t) device] || offset != a.cap_prefix[(size_t) device]) {
        GGML_ABORT("%s: node %s: device %d holds experts [%d, %d) of the weight, expert assignment %d gives it the segment [%" PRId64 ", %" PRId64 ")\n",
            __func__, node->name, device, offset, offset + n_local, table,
            a.cap_prefix[(size_t) device], a.cap_prefix[(size_t) device] + a.cap[(size_t) device]);
    }
    return a.local.data() + ((size_t) device * a.n_layer + layer) * a.n_expert;
}

void ggml_expert_store_plan(
        const ggml_expert_store_plan_tensor * tensors, size_t n_tensors,
        int n_device,
        int64_t slot_budget, const char * profile_path,
        ggml_expert_store_placement * placement) {
    GGML_ASSERT(placement != nullptr);

    placement->n_slot.assign(n_tensors, 0);
    placement->host_used.assign(n_tensors, 0);
    placement->slot_experts.assign(n_tensors, {});
    placement->slot_used  = 0;
    placement->host_bytes = 0;
    placement->host_peak  = 0;

    // per-expert span (data + row padding + alignment) and total span of every tensor. The
    // quota of a tensor is proportional to its span, so the slot count of a layer is its
    // quota divided by its stride, rounded down to whole experts.
    std::vector<size_t> stride(n_tensors, 0);
    std::vector<size_t> tensor_bytes(n_tensors, 0);
    size_t expert_bytes = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        stride[i] = ggml_expert_store_stride(tensors[i].blob_bytes, tensors[i].blob_pad, tensors[i].type_size);
        tensor_bytes[i] = (size_t) tensors[i].n_expert * stride[i];
        expert_bytes += tensor_bytes[i];
    }
    if (expert_bytes == 0) {
        return;
    }

    // canonical tensor order for the redistribution below: fit and the real allocation may
    // collect the same tensors in a different order, but the plan must be byte-identical for
    // both. The sort key uses only the tensor metadata, so the result does not depend on the
    // input order.
    std::vector<size_t> order_redist(n_tensors);
    for (size_t i = 0; i < n_tensors; ++i) {
        order_redist[i] = i;
    }
    std::stable_sort(order_redist.begin(), order_redist.end(), [&](size_t a, size_t b) {
        if (tensors[a].layer     != tensors[b].layer)     return tensors[a].layer     < tensors[b].layer;
        if (tensors[a].n_expert  != tensors[b].n_expert)  return tensors[a].n_expert  < tensors[b].n_expert;
        if (stride[a]            != stride[b])            return stride[a]            < stride[b];
        return tensor_bytes[a]   < tensor_bytes[b];
    });

    // slots per tensor. Whole experts only, so a byte share that does not hold a full
    // expert is not used. The rounding is always DOWN (floor of the per-tensor quota), so
    // the sum can never cross the budget. slot_budget < 0 means every expert gets a slot.
    std::vector<int64_t> n_slots(n_tensors, 0);
    size_t slot_used = 0;
    if (slot_budget < 0) {
        for (size_t i = 0; i < n_tensors; ++i) {
            n_slots[i] = tensors[i].n_expert;
            slot_used += (size_t) n_slots[i] * stride[i];
        }
    } else {
        // floor(slot_budget * tensor_bytes[i] / expert_bytes / stride[i]) in exact 64-bit
        // arithmetic. Dividing by expert_bytes first does not change the result:
        // floor(floor(x / c) / d) == floor(x / (c * d)) for positive integers.
        for (size_t i = 0; i < n_tensors; ++i) {
            const uint64_t quota = ggml_expert_store_mul_div_floor(
                (uint64_t) slot_budget, (uint64_t) tensor_bytes[i], (uint64_t) expert_bytes);
            int64_t n = (int64_t) (quota / (uint64_t) stride[i]);
            if (n > tensors[i].n_expert) {
                n = tensors[i].n_expert;
            }
            // a rounding artifact must never push the running total past the budget: take
            // whole experts off this layer until it fits (integer-safe by construction)
            while (n > 0 && slot_used + (size_t) n * stride[i] > (size_t) slot_budget) {
                --n;
            }
            n_slots[i] = n;
            slot_used += (size_t) n * stride[i];
        }

        // the bytes left by rounding down are given back as whole experts: every round adds
        // one expert to the layer that is furthest below its exact byte share. The check on
        // slot_budget keeps the running total within the budget, so the result uses as much
        // of the budget as whole experts allow and never more.
        while (true) {
            size_t  best = SIZE_MAX;
            int64_t best_gap = 0;
            for (const size_t i : order_redist) {
                if (n_slots[i] >= tensors[i].n_expert) {
                    continue;
                }
                if (slot_used + stride[i] > (size_t) slot_budget) {
                    continue;
                }
                // exact byte share of the layer (floor(slot_budget * tensor_bytes[i] /
                // expert_bytes)) minus the bytes it already holds. The quota is at most the
                // budget and the held bytes at most the expert bytes of the tensor, so the
                // difference fits in int64_t; ordering by it is ordering by the gap, because
                // both gaps share the denominator expert_bytes and the fraction that floor
                // drops is below one, which cannot change the order.
                const int64_t gap = (int64_t) ggml_expert_store_mul_div_floor(
                    (uint64_t) slot_budget, (uint64_t) tensor_bytes[i], (uint64_t) expert_bytes)
                    - (int64_t) ((size_t) n_slots[i] * stride[i]);
                if (best == SIZE_MAX || gap > best_gap) {
                    best     = i;
                    best_gap = gap;
                }
            }
            if (best == SIZE_MAX) {
                break;
            }
            n_slots[best]++;
            slot_used += stride[best];
        }
    }

    // static profile (Strata A3): experts with the highest hit counts get the slots. A profile
    // that the user asked for but that cannot be read (or carries no entry) is a configuration
    // error: loading it as a uniform placement would silently change where the experts live.
    const std::string path = profile_path != nullptr ? profile_path : ggml_expert_store_get_profile();
    ggml_expert_store_profile profile;
    const bool have_profile = !path.empty() && ggml_expert_store_read_profile(path.c_str(), profile);
    if (!path.empty() && (!have_profile || profile.n_layer <= 0 || profile.n_expert <= 0)) {
        GGML_ABORT("%s: cannot read expert profile '%s' (missing file or no entry)\n",
            __func__, path.c_str());
    }

    // which device holds which expert: the same profile and the same layer/expert sizes as the
    // plan above, computed once for the whole process and read through ggml_expert_store_owner()
    if (n_device > 0) {
        int64_t n_expert_total = 0;
        int64_t n_layer_total  = 0;
        for (size_t i = 0; i < n_tensors; ++i) {
            n_expert_total = std::max(n_expert_total, tensors[i].expert_base + tensors[i].n_expert);
            n_layer_total  = std::max(n_layer_total, tensors[i].layer >= 0 ? (int64_t) tensors[i].layer + 1 : (int64_t) 0);
        }
        if (have_profile) {
            n_expert_total = std::max(n_expert_total, profile.n_expert);
            n_layer_total  = std::max(n_layer_total,  profile.n_layer);
        }
        if (n_expert_total > 0 && n_layer_total > 0) {
            ggml_expert_store_compute_assignment(path.c_str(), have_profile ? &profile : nullptr,
                n_device, (int) n_layer_total, (int) n_expert_total);
        }
    }

    for (size_t i = 0; i < n_tensors; ++i) {
        const ggml_expert_store_plan_tensor & in = tensors[i];
        GGML_ASSERT(in.n_expert > 0 && in.blob_bytes > 0);

        const int64_t n_slot   = n_slots[i];
        const int64_t aligned  = (int64_t) stride[i];

        std::vector<int64_t> order((size_t) in.n_expert);
        for (int64_t e = 0; e < in.n_expert; ++e) {
            order[(size_t) e] = e;
        }
        if (have_profile && in.layer >= 0 && profile.n_layer > in.layer && profile.n_expert > 0) {
            // the profile row of a local expert is the one of the global expert the assignment
            // puts there: the loader lays the slice out by the proposal computed above, so the
            // proposal (not the active table of the layout still in memory) is the right map here
            const int dev = ggml_expert_store_slice_device(in.expert_base, in.n_expert);
            if (dev == -2) {
                GGML_ABORT("%s: layer %d: experts [%" PRId64 ", %" PRId64 ") are no segment of the expert assignment\n",
                    __func__, in.layer, in.expert_base, in.expert_base + in.n_expert);
            }
            const auto global_of = [&](int64_t local) -> int64_t {
                if (dev == -1) {
                    return local; // the whole expert dim on one device
                }
                const int g = ggml_expert_store_global_of(dev, in.layer, (int) local);
                GGML_ASSERT(g >= 0);
                return g;
            };
            const auto count_of = [&](int64_t local) -> int64_t {
                const int64_t g = global_of(local);
                return g < profile.n_expert ? profile.at(in.layer, g) : 0;
            };
            std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                return count_of(a) > count_of(b);
            });
        }

        placement->n_slot[i]    = n_slot;
        placement->host_used[i] = (in.n_expert - n_slot) * aligned;
        placement->slot_experts[i].assign(order.begin(), order.begin() + (size_t) n_slot);
        // slot k holds the k-th selected expert in ascending expert order. The profile decides
        // which experts get a slot, not in which order: numbering the slots by expert number
        // makes a fully resident tensor an identity map (expert e at slot e), so its experts can
        // be addressed arithmetically instead of through the pointer table.
        std::sort(placement->slot_experts[i].begin(), placement->slot_experts[i].end());
        placement->slot_used += n_slot * aligned;
        placement->host_bytes += placement->host_used[i];
        placement->host_peak = std::max(placement->host_peak, placement->host_used[i]);
    }

    GGML_LOG_DEBUG("%s: %zu tensors, slot budget %.2f MiB: slots %.2f MiB, host %.2f MiB, host peak %.2f MiB\n",
        __func__, n_tensors, slot_budget / 1024.0 / 1024.0, placement->slot_used / 1024.0 / 1024.0,
        placement->host_bytes / 1024.0 / 1024.0, placement->host_peak / 1024.0 / 1024.0);
}

int ggml_expert_store_add_tensor(
        ggml_expert_store * store,
        int layer, int kind,
        int64_t expert_base, int64_t n_expert, size_t blob_bytes, size_t blob_pad, size_t type_size,
        const int64_t * slot_experts, int64_t n_slot,
        int64_t host_budget) {
    GGML_ASSERT(store != nullptr);
    GGML_ASSERT(n_expert > 0);
    GGML_ASSERT(blob_bytes > 0);
    GGML_ASSERT(n_slot >= 0 && n_slot <= n_expert);
    GGML_ASSERT(n_slot == 0 || slot_experts != nullptr);

    if (store->tensors.size() >= (size_t) GGML_EXPERT_STORE_MAX_TENSORS) {
        GGML_LOG_ERROR("%s: too many expert tensors (%zu)\n", __func__, store->tensors.size());
        return -1;
    }

    const size_t stride = ggml_expert_store_stride(blob_bytes, blob_pad, type_size);
    const int64_t n_host = n_expert - n_slot;

    // the host tier stores the complement of the slot set; it must fit into the budget,
    // otherwise the placement is impossible and the caller has to make room.
    if (host_budget >= 0 && n_host > 0 && host_budget < n_host * (int64_t) stride) {
        GGML_LOG_ERROR("%s: layer %d kind %d: %" PRId64 " host experts need %.2f MiB, budget is %.2f MiB\n",
            __func__, layer, kind, n_host, n_host * stride / 1024.0 / 1024.0, host_budget / 1024.0 / 1024.0);
        return -1;
    }

    ggml_expert_store_tensor t;
    t.layer      = layer;
    t.kind       = kind;
    t.expert_base = expert_base;
    t.n_expert   = n_expert;
    t.blob_bytes = blob_bytes;
    t.blob_pad   = blob_pad;
    t.type_size  = type_size;
    t.stride     = stride;
    t.n_slot     = n_slot;
    t.n_host     = n_host;
    t.residency.assign((size_t) n_expert, GGML_EXPERT_STORE_INVALID);
    t.host_offset.assign((size_t) n_expert, GGML_EXPERT_STORE_INVALID);
    t.write_hash.assign((size_t) n_expert, 0);
    t.write_hash_state.assign((size_t) n_expert, 0);

    // the placement comes from ggml_expert_store_plan: slot s holds expert slot_experts[s],
    // the rest is packed into the host arena in expert order
    for (int64_t s = 0; s < n_slot; ++s) {
        const int64_t e = slot_experts[s];
        GGML_ASSERT(e >= 0 && e < n_expert);
        t.residency[(size_t) e] = (int32_t) s;
    }
    int64_t host_next = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        if (t.residency[(size_t) e] < 0) {
            t.host_offset[(size_t) e] = host_next;
            host_next += (int64_t) stride;
        }
    }
    t.host_used = host_next;

    GGML_LOG_INFO("%s: layer %d kind %d: %" PRId64 " experts, %.2f MiB data + %.2f MiB pad each, slots %" PRId64 ", host %" PRId64 "\n",
        __func__, layer, kind, n_expert, blob_bytes / 1024.0 / 1024.0, blob_pad / 1024.0 / 1024.0, n_slot, n_host);

    // allocate the slot array and the host arena slice
    if (n_slot > 0) {
        const size_t slot_bytes = (size_t) n_slot * stride;
        t.slots_dev = (char *) store->backend->alloc_device(store->backend->user, slot_bytes);
        if (t.slots_dev == nullptr) {
            GGML_LOG_ERROR("%s: layer %d kind %d: cannot allocate %.2f MiB of slots\n",
                __func__, layer, kind, slot_bytes / 1024.0 / 1024.0);
            return -1;
        }
        store->slot_used += (int64_t) slot_bytes;
    }
    if (n_host > 0) {
        t.host_host = (char *) store->backend->alloc_host(store->backend->user, (size_t) t.host_used, (void **) &t.host_dev);
        if (t.host_host == nullptr || t.host_dev == nullptr) {
            GGML_LOG_ERROR("%s: layer %d kind %d: cannot allocate %.2f MiB of host arena\n",
                __func__, layer, kind, t.host_used / 1024.0 / 1024.0);
            if (t.slots_dev != nullptr) {
                store->backend->free_device(store->backend->user, t.slots_dev);
            }
            return -1;
        }
        store->host_used += t.host_used;
    }

    // host shadow of the main pointer table; slot addresses are fixed from here on
    t.table_host = new char *[(size_t) n_expert];
    for (int64_t e = 0; e < n_expert; ++e) {
        if (t.residency[(size_t) e] >= 0) {
            t.table_host[(size_t) e] = t.slots_dev + (size_t) t.residency[(size_t) e] * stride;
        } else {
            t.table_host[(size_t) e] = t.host_dev + t.host_offset[(size_t) e];
        }
    }

    // sampling for the startup byte check: one expert served from the slots and one from
    // the host arena, the loader keeps their source bytes in ggml_expert_store_write()
    for (int64_t e = 0; e < n_expert && t.sample_expert[0] < 0; ++e) {
        if (t.residency[(size_t) e] >= 0) {
            t.sample_expert[0] = e;
            t.sample_src[0].resize(blob_bytes);
        }
    }
    for (int64_t e = 0; e < n_expert && t.sample_expert[1] < 0; ++e) {
        if (t.residency[(size_t) e] < 0) {
            t.sample_expert[1] = e;
            t.sample_src[1].resize(blob_bytes);
        }
    }

    GGML_LOG_INFO("%s: layer %d kind %d: slots %" PRId64 " (%.2f MiB), host %" PRId64 " (%.2f MiB)\n",
        __func__, layer, kind,
        n_slot, n_slot * (double) stride / 1024.0 / 1024.0,
        n_host, host_next / 1024.0 / 1024.0);

    store->tensors.push_back(std::move(t));

    return (int) store->tensors.size() - 1;
}

static inline int64_t ggml_expert_store_expert_of(const ggml_expert_store_tensor & t, size_t offset) {
    return (int64_t) (offset / t.blob_bytes);
}

// record_hash: the bytes are weight data (a write by the loader) and their hash is kept for the
// placement check; false for a memset, which carries no weight data
static void ggml_expert_store_write_impl(
        ggml_expert_store * store, int tensor_id,
        const void * data, size_t offset, size_t size, bool record_hash) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < (int) store->tensors.size());
    ggml_expert_store_tensor & t = store->tensors[(size_t) tensor_id];

    const uint8_t * src = (const uint8_t *) data;
    size_t written = 0;
    while (written < size) {
        const size_t cur = offset + written;
        const int64_t expert = ggml_expert_store_expert_of(t, cur);
        GGML_ASSERT(expert >= 0 && expert < t.n_expert);
        const size_t in_expert = cur - (size_t) expert * t.blob_bytes;
        const size_t n = std::min(size - written, t.blob_bytes - in_expert);

        if (t.residency[(size_t) expert] >= 0) {
            // only the data bytes are copied; the padding after them stays zero from the
            // allocation-time clear (see ggml_expert_store_add_tensor)
            char * dst = t.slots_dev + (size_t) t.residency[(size_t) expert] * t.stride + in_expert;
            store->backend->copy_h2d(store->backend->user, dst, src + written, n);
        } else {
            char * dst = t.host_host + t.host_offset[(size_t) expert] + in_expert;
            memcpy(dst, src + written, n);
        }
        // keep the source bytes of the sampled experts for the startup byte check
        for (int s = 0; s < 2; ++s) {
            if (t.sample_expert[s] == expert && !t.sample_skip[s] && !t.sample_ready[s]) {
                if (in_expert == 0 && n == t.blob_bytes) {
                    memcpy(t.sample_src[s].data(), src + written, t.blob_bytes);
                    t.sample_ready[s] = true;
                } else {
                    // loaded in several pieces, the sample cannot be extracted safely
                    t.sample_skip[s] = true;
                }
            }
        }
        // the hash of what this local expert holds, for the placement check: a whole expert in
        // one write is hashed, an expert that arrives in pieces has no hash
        if (record_hash) {
            if (in_expert == 0 && n == t.blob_bytes) {
                t.write_hash[(size_t) expert]       = ggml_expert_store_hash_expert(src + written, n);
                t.write_hash_state[(size_t) expert] = 1;
            } else {
                t.write_hash_state[(size_t) expert] = 2;
            }
        }
        t.load_bytes += (int64_t) n;
        written += n;
    }
}

void ggml_expert_store_write(
        ggml_expert_store * store, int tensor_id,
        const void * data, size_t offset, size_t size) {
    ggml_expert_store_write_impl(store, tensor_id, data, offset, size, /*record_hash =*/ true);
}

void ggml_expert_store_read(
        ggml_expert_store * store, int tensor_id,
        void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < (int) store->tensors.size());
    ggml_expert_store_tensor & t = store->tensors[(size_t) tensor_id];

    uint8_t * dst = (uint8_t *) data;
    size_t read = 0;
    while (read < size) {
        const size_t cur = offset + read;
        const int64_t expert = ggml_expert_store_expert_of(t, cur);
        GGML_ASSERT(expert >= 0 && expert < t.n_expert);
        const size_t in_expert = cur - (size_t) expert * t.blob_bytes;
        const size_t n = std::min(size - read, t.blob_bytes - in_expert);

        if (t.residency[(size_t) expert] >= 0) {
            const char * src = t.slots_dev + (size_t) t.residency[(size_t) expert] * t.stride + in_expert;
            store->backend->copy_d2h(store->backend->user, dst + read, src, n);
        } else {
            memcpy(dst + read, t.host_host + t.host_offset[(size_t) expert] + in_expert, n);
        }
        read += n;
    }
}

void ggml_expert_store_memset(
        ggml_expert_store * store, int tensor_id,
        uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < (int) store->tensors.size());
    ggml_expert_store_tensor & t = store->tensors[(size_t) tensor_id];

    if (size == 0) {
        return;
    }
    // clearing does not carry weight data: keep the verification samples untouched
    const bool skip0  = t.sample_skip[0];
    const bool skip1  = t.sample_skip[1];
    const bool ready0 = t.sample_ready[0];
    const bool ready1 = t.sample_ready[1];

    std::vector<uint8_t> zeros(std::min(size, (size_t) (1 << 20)), value);
    size_t done = 0;
    while (done < size) {
        const size_t n = std::min(size - done, zeros.size());
        ggml_expert_store_write_impl(store, tensor_id, zeros.data(), offset + done, n, /*record_hash =*/ false);
        done += n;
    }

    t.sample_skip[0]  = skip0;
    t.sample_skip[1]  = skip1;
    t.sample_ready[0] = ready0;
    t.sample_ready[1] = ready1;
}

bool ggml_expert_store_verify(
        ggml_expert_store * store, int tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < (int) store->tensors.size());
    ggml_expert_store_tensor & t = store->tensors[(size_t) tensor_id];

    std::vector<uint8_t> buf(t.blob_bytes);
    bool ok = true;

    for (int s = 0; s < 2; ++s) {
        const int64_t expert = t.sample_expert[s];
        if (expert < 0 || t.sample_skip[s] || !t.sample_ready[s]) {
            GGML_LOG_INFO("%s: layer %d kind %d: sample %d not available, skipped\n", __func__, t.layer, t.kind, s);
            continue;
        }

        if (t.residency[(size_t) expert] >= 0) {
            const char * dev = t.slots_dev + (size_t) t.residency[(size_t) expert] * t.stride;
            store->backend->copy_d2h(store->backend->user, buf.data(), dev, t.blob_bytes);
        } else {
            memcpy(buf.data(), t.host_host + t.host_offset[(size_t) expert], t.blob_bytes);
        }

        if (memcmp(buf.data(), t.sample_src[s].data(), t.blob_bytes) != 0) {
            GGML_LOG_ERROR("%s: layer %d kind %d expert %" PRId64 " (%s tier) does not match the source bytes\n",
                __func__, t.layer, t.kind, expert, t.residency[(size_t) expert] >= 0 ? "slot" : "host");
            ok = false;
        }
    }

    // the source copies are only needed for this check
    for (int s = 0; s < 2; ++s) {
        t.sample_src[s].clear();
        t.sample_src[s].shrink_to_fit();
    }

    if (ok) {
        GGML_LOG_INFO("%s: layer %d kind %d: byte check passed\n", __func__, t.layer, t.kind);
    }
    return ok;
}

// 8 bytes per step, multiply and xor-shift mixing; the tail is zero padded and the length is
// folded into the seed by the caller, so a shorter range never aliases a longer one
static uint64_t ggml_expert_store_hash_range(uint64_t h, const uint8_t * p, size_t n) {
    while (n >= 8) {
        uint64_t w;
        memcpy(&w, p, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
        p += 8;
        n -= 8;
    }
    if (n > 0) {
        uint64_t w = 0;
        memcpy(&w, p, n);
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
    }
    return h;
}

uint64_t ggml_expert_store_hash_expert(const void * data, size_t size) {
    const size_t    W = GGML_EXPERT_STORE_HASH_WINDOW;
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0x243F6A8885A308D3ULL ^ ((uint64_t) size * 0x9E3779B97F4A7C15ULL);
    if (size <= 3*W) {
        return ggml_expert_store_hash_range(h, p, size);
    }
    h = ggml_expert_store_hash_range(h, p, W);
    h = ggml_expert_store_hash_range(h, p + size/2 - W/2, W);
    h = ggml_expert_store_hash_range(h, p + size - W, W);
    return h;
}

void ggml_expert_store_set_source_hash(const char * tensor_name, int64_t expert, uint64_t hash) {
    GGML_ASSERT(tensor_name != nullptr && expert >= 0);
    std::lock_guard<std::mutex> lock(g_source_hash_mutex);
    ggml_expert_store_source_hashes & s = g_source_hash[tensor_name];
    if ((size_t) expert >= s.hash.size()) {
        s.hash.resize((size_t) expert + 1, 0);
        s.present.resize((size_t) expert + 1, 0);
    }
    s.hash[(size_t) expert]    = hash;
    s.present[(size_t) expert] = 1;
}

void ggml_expert_store_clear_source_hashes(void) {
    std::lock_guard<std::mutex> lock(g_source_hash_mutex);
    g_source_hash.clear();
}

bool ggml_expert_store_verify_placement(
        ggml_expert_store * store, int tensor_id, const char * tensor_name,
        int64_t * n_checked, int64_t * n_skipped) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < (int) store->tensors.size());
    ggml_expert_store_tensor & t = store->tensors[(size_t) tensor_id];

    int64_t checked = 0;
    int64_t skipped = 0;
    bool ok = true;

    ggml_expert_store_source_hashes src;
    bool have_src = false;
    if (tensor_name != nullptr) {
        std::lock_guard<std::mutex> lock(g_source_hash_mutex);
        const auto it = g_source_hash.find(tensor_name);
        if (it != g_source_hash.end()) {
            src      = it->second;
            have_src = true;
        }
    }

    if (!have_src) {
        // the bytes did not come from the loader (another writer filled the store): nothing
        // to compare against, every expert is skipped and the summary says so
        skipped = t.n_expert;
    } else {
        // the global expert of every local expert of this slice: through the table the loader laid
        // the tensor out by, or the identity for a tensor it wrote in file order, which can only be
        // a tensor that holds the whole expert dim
        std::vector<int64_t> global_of((size_t) t.n_expert, -1);
        int dev = -1;
        if (src.assignment != 0) {
            std::lock_guard<std::mutex> lock(g_table_mutex);
            const ggml_expert_assignment_table * tab = ggml_expert_assignment_find(src.assignment);
            if (tab == nullptr) {
                GGML_LOG_ERROR("%s: %s: the loader laid the tensor out by expert assignment %u, which does not exist\n",
                    __func__, tensor_name, src.assignment);
                ok = false;
            } else {
                dev = ggml_expert_store_table_slice_device(tab->a, t.expert_base, t.n_expert);
                if (dev == -2 || t.layer < 0 || t.layer >= tab->a.n_layer) {
                    GGML_LOG_ERROR("%s: %s: layer %d, experts [%" PRId64 ", %" PRId64 ") are no segment of expert assignment %u (devs=%d layers=%d experts=%d)\n",
                        __func__, tensor_name, t.layer, t.expert_base, t.expert_base + t.n_expert, src.assignment,
                        tab->a.n_dev, tab->a.n_layer, tab->a.n_expert);
                    ok = false;
                }
                for (int64_t k = 0; ok && k < t.n_expert; ++k) {
                    global_of[(size_t) k] = dev == -1 ? k : ggml_expert_store_table_global_of(tab->a, dev, t.layer, (int) k);
                    if (global_of[(size_t) k] < 0) {
                        GGML_LOG_ERROR("%s: %s: local expert %" PRId64 " of device segment %d has no global expert in expert assignment %u\n",
                            __func__, tensor_name, k, dev, src.assignment);
                        ok = false;
                    }
                }
            }
        } else if (t.expert_base != 0 || (int64_t) src.present.size() != t.n_expert) {
            GGML_LOG_ERROR("%s: %s: experts [%" PRId64 ", %" PRId64 ") of a tensor of %zu experts were written in file order, without an expert assignment\n",
                __func__, tensor_name, t.expert_base, t.expert_base + t.n_expert, src.present.size());
            ok = false;
        } else {
            for (int64_t k = 0; k < t.n_expert; ++k) {
                global_of[(size_t) k] = k;
            }
        }

        if (ok) {
            // the routing dump writes the rows of this tensor through the same numbering
            t.global_of  = global_of;
            t.assignment = src.assignment;
        }

        for (int64_t k = 0; ok && k < t.n_expert; ++k) {
            const int64_t g = global_of[(size_t) k];
            if (t.write_hash_state[(size_t) k] != 1) {
                // written in pieces or not at all: no hash to compare
                skipped++;
                continue;
            }
            if ((size_t) g >= src.present.size() || !src.present[(size_t) g]) {
                GGML_LOG_ERROR("%s: %s: the loader registered no source hash for global expert %" PRId64 " (local %" PRId64 " of device segment %d)\n",
                    __func__, tensor_name, g, k, dev);
                ok = false;
                break;
            }
            if (src.hash[(size_t) g] != t.write_hash[(size_t) k]) {
                GGML_LOG_ERROR("%s: %s: local expert %" PRId64 " of device segment %d holds other bytes than global expert %" PRId64 " of the file (layer %d, %s tier)\n",
                    __func__, tensor_name, k, dev, g, t.layer, t.residency[(size_t) k] >= 0 ? "slot" : "host");
                ok = false;
                break;
            }
            checked++;
        }
    }

    if (n_checked != nullptr) {
        *n_checked = checked;
    }
    if (n_skipped != nullptr) {
        *n_skipped = skipped;
    }
    return ok;
}

void ggml_expert_store_log_stats(const ggml_expert_store * store, const char * tag) {
    int64_t hits_vram = 0;
    int64_t hits_host = 0;
    for (const ggml_expert_store_tensor & t : store->tensors) {
        hits_vram += t.stat_hits_vram;
        hits_host += t.stat_hits_host;
    }
    const int64_t total = hits_vram + hits_host;
    if (total == 0) {
        return;
    }
    GGML_LOG_DEBUG("%s: device %d: expert tier hits %" PRId64 "/%" PRId64 " (%.2f%% vram, %.2f%% host)\n",
        tag != nullptr ? tag : __func__, store->device,
        hits_vram, total, 100.0 * hits_vram / total, 100.0 * hits_host / total);
}

void ggml_expert_store_print_plan(const ggml_expert_store * store) {
    int64_t n_slot = 0;
    int64_t n_host = 0;
    for (const ggml_expert_store_tensor & t : store->tensors) {
        n_slot += t.n_slot;
        n_host += t.n_host;
    }
    GGML_LOG_INFO("%s: device %d: %d tensors, %" PRId64 " slots (%.2f MiB), %" PRId64 " host experts (%.2f MiB)\n",
        __func__, store->device, (int) store->tensors.size(),
        n_slot, store->slot_used / 1024.0 / 1024.0,
        n_host, store->host_used / 1024.0 / 1024.0);
}

void ggml_expert_store_reset_stats(ggml_expert_store * store) {
    for (ggml_expert_store_tensor & t : store->tensors) {
        t.stat_hits_vram = 0;
        t.stat_hits_host = 0;
    }
}

const ggml_expert_store_tensor * ggml_expert_store_tensor_at(const ggml_expert_store * store, int tensor_id) {
    if (tensor_id < 0 || tensor_id >= (int) store->tensors.size()) {
        return nullptr;
    }
    return &store->tensors[(size_t) tensor_id];
}

int ggml_expert_store_tensor_id(const ggml_expert_store * store, const ggml_tensor * tensor) {
    for (const auto & kv : store->tensor_map) {
        if (kv.first == tensor) {
            return kv.second;
        }
    }
    return -1;
}

void ggml_expert_store_set_tensor_id(ggml_expert_store * store, const ggml_tensor * tensor, int tensor_id) {
    for (auto & kv : store->tensor_map) {
        if (kv.first == tensor) {
            kv.second = tensor_id;
            return;
        }
    }
    store->tensor_map.emplace_back(tensor, tensor_id);
}

void ggml_expert_store_set_tensor_stats(ggml_expert_store * store, int tensor_id, const uint32_t * counts, size_t n) {
    if (tensor_id < 0 || tensor_id >= (int) store->tensors.size()) {
        return;
    }
    ggml_expert_store_tensor & t = store->tensors[(size_t) tensor_id];
    t.stat_experts.assign(counts, counts + n);
}
