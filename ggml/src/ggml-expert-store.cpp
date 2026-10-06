#include "ggml-expert-store.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#define GGML_EXPERT_STORE_ALIGN 256

static size_t ggml_expert_store_align_up(size_t v) {
    return (v + GGML_EXPERT_STORE_ALIGN - 1) & ~(size_t) (GGML_EXPERT_STORE_ALIGN - 1);
}

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
}

static std::string ggml_expert_store_get_dump_path() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_dump_path;
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

    // tensors of the same layer (gate/up/down) share one profile row: sum their counts
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
        std::vector<int64_t> & row = counts[(size_t) t.layer];
        row.resize(std::max<size_t>(row.size(), t.stat_experts.size()), 0);
        for (size_t e = 0; e < t.stat_experts.size(); ++e) {
            row[e] += t.stat_experts[e];
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

int ggml_expert_store_add_tensor(
        ggml_expert_store * store,
        int layer, int kind,
        int64_t n_expert, size_t blob_bytes,
        int64_t slot_budget, int64_t host_budget) {
    GGML_ASSERT(store != nullptr);
    GGML_ASSERT(n_expert > 0);
    GGML_ASSERT(blob_bytes > 0);

    if (store->tensors.size() >= (size_t) GGML_EXPERT_STORE_MAX_TENSORS) {
        GGML_LOG_ERROR("%s: too many expert tensors (%zu)\n", __func__, store->tensors.size());
        return -1;
    }

    // per-layer quota: this tensor is one layer, so every slot belongs to this layer.
    // slot_budget < 0 means "all experts get a slot" (fully resident tensor).
    int64_t n_slot = n_expert;
    if (slot_budget >= 0) {
        n_slot = std::min<int64_t>(n_expert, slot_budget / (int64_t) blob_bytes);
    }
    const int64_t n_host = n_expert - n_slot;

    // the host tier stores the complement of the slot set; it must fit into the budget,
    // otherwise the placement is impossible and the caller has to make room.
    if (host_budget >= 0 && n_host > 0 && host_budget < n_host * (int64_t) blob_bytes) {
        GGML_LOG_ERROR("%s: layer %d kind %d: %" PRId64 " host experts need %.2f MiB, budget is %.2f MiB\n",
            __func__, layer, kind, n_host, n_host * blob_bytes / 1024.0 / 1024.0, host_budget / 1024.0 / 1024.0);
        return -1;
    }

    ggml_expert_store_tensor t;
    t.layer      = layer;
    t.kind       = kind;
    t.n_expert   = n_expert;
    t.blob_bytes = blob_bytes;
    t.n_slot     = n_slot;
    t.n_host     = n_host;
    t.residency.assign((size_t) n_expert, GGML_EXPERT_STORE_INVALID);
    t.host_offset.assign((size_t) n_expert, GGML_EXPERT_STORE_INVALID);

    GGML_LOG_INFO("%s: layer %d kind %d: %" PRId64 " experts, %.2f MiB each, slots %" PRId64 ", host %" PRId64 "\n",
        __func__, layer, kind, n_expert, blob_bytes / 1024.0 / 1024.0, n_slot, n_host);

    // static profile (Strata A3): experts with the highest hit counts get the slots.
    // Without a profile the first experts by index are placed, which is the uniform
    // fallback of the task description and is logged as such.
    ggml_expert_store_profile profile;
    bool have_profile = false;
    if (!store->profile_path.empty()) {
        have_profile = ggml_expert_store_read_profile(store->profile_path.c_str(), profile);
        if (!have_profile) {
            GGML_LOG_ERROR("%s: cannot read expert profile '%s', falling back to uniform placement\n",
                __func__, store->profile_path.c_str());
        }
    }

    std::vector<int64_t> order((size_t) n_expert);
    for (int64_t i = 0; i < n_expert; ++i) {
        order[i] = i;
    }

    int64_t ranked = 0; // experts with a non-zero profile count
    if (have_profile && profile.n_layer > layer && profile.n_expert > 0) {
        for (int64_t e = 0; e < n_expert; ++e) {
            if (e < profile.n_expert && profile.at(layer, e) > 0) {
                ranked++;
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            const int64_t ca = a < profile.n_expert ? profile.at(layer, a) : 0;
            const int64_t cb = b < profile.n_expert ? profile.at(layer, b) : 0;
            return ca > cb;
        });
    } else {
        have_profile = false;
    }

    int64_t slot_next = 0;
    int64_t host_next = 0;
    for (int64_t rank = 0; rank < n_expert; ++rank) {
        const int64_t e = order[(size_t) rank];
        if (rank < n_slot) {
            t.residency[(size_t) e] = (int32_t) slot_next++;
        } else {
            t.host_offset[(size_t) e] = host_next;
            host_next += (int64_t) ggml_expert_store_align_up(blob_bytes);
        }
    }
    t.host_used = host_next;

    // allocate the slot array and the host arena slice
    if (n_slot > 0) {
        const size_t slot_bytes = (size_t) n_slot * ggml_expert_store_align_up(blob_bytes);
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
            t.table_host[(size_t) e] = t.slots_dev + (size_t) t.residency[(size_t) e] * ggml_expert_store_align_up(blob_bytes);
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

    GGML_LOG_INFO("%s: layer %d kind %d: profile %s, slots %" PRId64 " (%.2f MiB), host %" PRId64 " (%.2f MiB)\n",
        __func__, layer, kind,
        have_profile ? "ranked" : "uniform (no profile)",
        n_slot, n_slot * (double) ggml_expert_store_align_up(blob_bytes) / 1024.0 / 1024.0,
        n_host, host_next / 1024.0 / 1024.0);

    store->tensors.push_back(std::move(t));
    if (have_profile) {
        GGML_LOG_INFO("%s: layer %d kind %d: profile covers %" PRId64 "/%" PRId64 " experts\n",
            __func__, layer, kind, ranked, n_expert);
    } else if (!store->profile_path.empty()) {
        GGML_LOG_INFO("%s: layer %d kind %d: uniform placement within the layer quota\n", __func__, layer, kind);
    }

    return (int) store->tensors.size() - 1;
}

static inline int64_t ggml_expert_store_expert_of(const ggml_expert_store_tensor & t, size_t offset) {
    return (int64_t) (offset / t.blob_bytes);
}

void ggml_expert_store_write(
        ggml_expert_store * store, int tensor_id,
        const void * data, size_t offset, size_t size) {
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
            char * dst = t.slots_dev + (size_t) t.residency[(size_t) expert] * ggml_expert_store_align_up(t.blob_bytes) + in_expert;
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
        t.load_bytes += (int64_t) n;
        written += n;
    }
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
            const char * src = t.slots_dev + (size_t) t.residency[(size_t) expert] * ggml_expert_store_align_up(t.blob_bytes) + in_expert;
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
        ggml_expert_store_write(store, tensor_id, zeros.data(), offset + done, n);
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
            const char * dev = t.slots_dev + (size_t) t.residency[(size_t) expert] * ggml_expert_store_align_up(t.blob_bytes);
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
