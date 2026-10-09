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
#include <tuple>

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
    // With expert parallelism a store holds a contiguous slice of the experts, so the local
    // expert numbers are shifted by the tensor's global expert base.
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
        row.resize(std::max<size_t>(row.size(), (size_t) t.expert_base + t.stat_experts.size()), 0);
        for (size_t e = 0; e < t.stat_experts.size(); ++e) {
            row[(size_t) t.expert_base + e] += t.stat_experts[e];
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

void ggml_expert_store_plan(
        const ggml_expert_store_plan_tensor * tensors, size_t n_tensors,
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
            std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                const int64_t ca = a + in.expert_base < profile.n_expert ? profile.at(in.layer, a + in.expert_base) : 0;
                const int64_t cb = b + in.expert_base < profile.n_expert ? profile.at(in.layer, b + in.expert_base) : 0;
                return ca > cb;
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
