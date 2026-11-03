#pragma once

// Host-side expert store for MoE weights spread over per-device VRAM slots and a
// device-readable host arena.
//
// Layout follows the expert cache of Strata (https://github.com/Niko1221/Strata,
// include/strata/core/expert_cache.hpp, src/core/expert_cache.cpp, src/core/expert_source.cpp):
//   - every registered tensor owns a contiguous array of VRAM slots, one expert per slot;
//   - layers are isolated by construction because a tensor spans one layer only, so a slot
//     array is exactly the per-layer slot quota of Strata's per-layer admission;
//   - a (expert) -> slot residency table per tensor says where each expert lives;
//   - experts without a slot are packed into the host arena (the complement of the slots,
//     like the compact host copy of Strata's resident RAM mode);
//   - a static routing profile (as Strata's --expert-profile) decides which experts get
//     slots, otherwise the first experts of each tensor are placed in the slots;
//   - every expert is stored exactly once, either in a slot or in the host arena.
//
// The kernels read the weights through a per-tensor device pointer table whose address is
// fixed for the lifetime of the tensor; the table content is written by this module at load
// time and may be rewritten for prefill staging (the main table stays untouched).
//
// This file is internal to ggml. The CUDA backend provides the memory operations through
// ggml_expert_store_backend, so the store itself stays platform independent and other
// platforms can reuse it (for example computing host experts in place on the CPU, or
// copying them to VRAM).

#include "ggml.h"
#include "ggml-impl.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


#define GGML_EXPERT_STORE_INVALID (-1)


#define GGML_EXPERT_STORE_MAX_TENSORS 512



#define GGML_EXPERT_STORE_ALIGN 256



static inline size_t ggml_expert_store_align_up(size_t v, size_t align) {
    return (v + align - 1) / align * align;
}




static inline uint64_t ggml_expert_store_mul_div_floor(uint64_t a, uint64_t b, uint64_t c) {
    GGML_ASSERT(c > 0);



    const uint64_t a_lo = a & 0xFFFFFFFFULL;
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = b & 0xFFFFFFFFULL;
    const uint64_t b_hi = b >> 32;
    const uint64_t p0   = a_lo * b_lo;
    const uint64_t p1   = a_lo * b_hi;
    const uint64_t p2   = a_hi * b_lo;
    const uint64_t p3   = a_hi * b_hi;
    const uint64_t mid  = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    const uint64_t lo   = (mid << 32) | (p0 & 0xFFFFFFFFULL);
    const uint64_t hi   = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);



    GGML_ASSERT(hi < c);




    uint64_t q = 0;
    uint64_t r = hi;
    for (int i = 63; i >= 0; --i) {
        const uint64_t bit = (lo >> i) & 1;
        if (r >> 63) {

            q |= (uint64_t) 1 << i;
            r = (r << 1 | bit) - c;
        } else {
            r = r << 1 | bit;
            if (r >= c) {
                q |= (uint64_t) 1 << i;
                r -= c;
            }
        }
    }
    return q;
}






static inline size_t ggml_expert_store_stride_align(size_t type_size) {
    size_t a = GGML_EXPERT_STORE_ALIGN;
    size_t b = type_size;
    while (b != 0) {
        const size_t t = a % b;
        a = b;
        b = t;
    }
    return GGML_EXPERT_STORE_ALIGN / a * type_size;
}





static inline size_t ggml_expert_store_stride(size_t blob_bytes, size_t blob_pad, size_t type_size) {
    return ggml_expert_store_align_up(blob_bytes + blob_pad, ggml_expert_store_stride_align(type_size));
}


enum ggml_expert_tier {
    GGML_EXPERT_TIER_VRAM = 0,
    GGML_EXPERT_TIER_HOST = 1,
    GGML_EXPERT_TIER_CPU  = 2,
};


struct ggml_expert_store_backend {
    void * user = nullptr;


    void * (*alloc_device)(void * user, size_t size) = nullptr;
    void   (*free_device) (void * user, void * ptr)   = nullptr;



    void * (*alloc_host)(void * user, size_t size, void ** dev_ptr) = nullptr;
    void   (*free_host) (void * user, void * ptr)                   = nullptr;


    void (*copy_h2d)(void * user, void * dst_dev, const void * src_host, size_t size) = nullptr;
    void (*copy_d2h)(void * user, void * dst_host, const void * src_dev, size_t size) = nullptr;


    void (*table_upload)(void * user, void * dst_dev, const char * const * src_host, size_t n) = nullptr;
};


struct ggml_expert_store_tensor {
    int      layer      = -1;
    int      kind       = -1;





    int64_t  expert_base = 0;
    int64_t  n_expert   = 0;
    size_t   blob_bytes = 0;
    size_t   blob_pad   = 0;
    size_t   type_size  = 1;
    size_t   stride     = 0;


    std::vector<int32_t> residency;
    std::vector<int64_t> host_offset;
    int64_t n_slot       = 0;
    int64_t n_host       = 0;
    int64_t host_used    = 0;
    int64_t load_bytes   = 0;
    int64_t swap_count   = 0;
    int64_t swap_bytes   = 0;


    char * slots_dev  = nullptr;
    char * host_host  = nullptr;
    char * host_dev   = nullptr;
    char ** table_host = nullptr;




    std::vector<char *> slot_addr;




    int64_t sample_expert[2] = { GGML_EXPERT_STORE_INVALID, GGML_EXPERT_STORE_INVALID };
    std::vector<uint8_t> sample_src[2];
    bool    sample_ready[2] = { false, false };
    bool    sample_skip[2]  = { false, false };






    std::vector<uint64_t> write_hash;
    std::vector<uint8_t>  write_hash_state;





    std::vector<int64_t> global_of;
    uint32_t             assignment = 0;


    int64_t stat_hits_vram = 0;
    int64_t stat_hits_host = 0;
    std::vector<uint32_t> stat_experts;
    std::vector<uint32_t> stat_launches;
};


struct ggml_expert_store {
    const ggml_expert_store_backend * backend = nullptr;
    int device = -1;

    std::string profile_path;

    std::vector<ggml_expert_store_tensor> tensors;

    std::vector<std::pair<const ggml_tensor *, int>> tensor_map;

    int64_t slot_used = 0;
    int64_t host_used = 0;
};










struct ggml_expert_store_profile {
    int64_t n_layer  = 0;
    int64_t n_expert = 0;
    std::vector<int64_t> count;
    std::vector<int64_t> launch;

    int64_t at(int64_t layer, int64_t expert) const {
        return count[layer * n_expert + expert];
    }
    bool has_launch() const {
        return !launch.empty();
    }
    int64_t launch_at(int64_t layer, int64_t expert) const {
        return launch[layer * n_expert + expert];
    }
};

GGML_API ggml_expert_store * ggml_expert_store_init(const ggml_expert_store_backend * backend, int device, const char * profile_path);
GGML_API void ggml_expert_store_free(ggml_expert_store * store);




#ifdef __cplusplus
extern "C" {
#endif








#define GGML_EXPERT_S3PRIME_MIN_TOKENS 4096

GGML_API void ggml_expert_store_set_profile(const char * path);





GGML_API void ggml_expert_store_set_dump_path(const char * path);






GGML_API void ggml_expert_store_set_trace_path(const char * path, int size_mib);
#ifdef __cplusplus
}
#endif







#ifndef GGML_EXPERT_ADAPTIVE
#define GGML_EXPERT_ADAPTIVE 1
#endif

struct ggml_expert_store_adaptive_cfg {
    bool    enabled = GGML_EXPERT_ADAPTIVE != 0;

    int64_t window  = 16;

    double  decay   = 0.7;




    double  ratio    = 1.5;
    double  min_gain = 2.0;


    int64_t pairs_per_token = 10;
    int64_t pairs_step_max  = 24;




    int64_t inflight_tokens = 3;





    int64_t min_host_per_pool = 32;
};

GGML_API ggml_expert_store_adaptive_cfg ggml_expert_store_get_adaptive();


GGML_API bool ggml_expert_store_adaptive_enabled();



GGML_API void ggml_expert_store_set_adaptive_window(int64_t window);



GGML_API void ggml_expert_store_set_adaptive_min_host_per_pool(int64_t ratio);


GGML_API std::string ggml_expert_store_get_trace_path();
GGML_API int         ggml_expert_store_get_trace_size_mib();


GGML_API bool ggml_expert_store_trace_enabled();



GGML_API bool ggml_expert_store_dump_enabled();



GGML_API void ggml_expert_store_dump_profile(ggml_expert_store * store, int n_device);


struct ggml_expert_store_plan_tensor {
    int     layer      = -1;
    int64_t n_expert   = 0;




    int64_t expert_base = 0;
    size_t  blob_bytes = 0;




    size_t  blob_pad   = 0;


    size_t  type_size  = 1;


    int     kind       = -1;



    int     n_expert_used = 0;
};


struct ggml_expert_store_placement {
    std::vector<int64_t> n_slot;
    std::vector<int64_t> host_used;
    std::vector<std::vector<int64_t>> slot_experts;
    int64_t slot_used  = 0;
    int64_t host_bytes = 0;



    int64_t host_peak  = 0;





    double  routed_share   = 0.0;
    bool    has_routed_share = false;
};









GGML_API void ggml_expert_store_plan(
        const ggml_expert_store_plan_tensor * tensors, size_t n_tensors,
        int n_device,
        int64_t slot_budget, const char * profile_path,
        ggml_expert_store_placement * placement);





GGML_API void ggml_expert_store_compute_assignment(
        const char * path, const ggml_expert_store_profile * profile,
        int n_device, int n_layer, int n_expert);


struct ggml_expert_store_assignment_info {
    int    n_dev    = 0;
    int    n_layer  = 0;
    int    n_expert = 0;
    int    cap_max  = 0;
    bool   has_profile = false;

    double imbalance_contiguous = 0.0;
    double imbalance_permuted    = 0.0;
    int    worst_layer           = -1;
    double worst_layer_imbalance = 0.0;
};


GGML_API int ggml_expert_store_owner(int layer, int expert);




GGML_API const int16_t * ggml_expert_store_local_of(int dev, int layer);


GGML_API int ggml_expert_store_global_of(int dev, int layer, int local);


GGML_API bool ggml_expert_store_assignment_info_get(ggml_expert_store_assignment_info * info);





GGML_API int ggml_expert_store_slice_device(int64_t expert_base, int64_t n_expert);





GGML_API bool            ggml_expert_assignment_is_active(uint32_t table);
GGML_API int             ggml_expert_assignment_owner    (uint32_t table, int layer, int expert);
GGML_API const int16_t * ggml_expert_assignment_local_of (uint32_t table, int dev, int layer);
GGML_API int             ggml_expert_assignment_global_of(uint32_t table, int dev, int layer, int local);


GGML_API int             ggml_expert_assignment_slice_device(uint32_t table, int64_t expert_base, int64_t n_expert);

GGML_API uint64_t        ggml_expert_assignment_fingerprint(uint32_t table);

GGML_API size_t          ggml_expert_assignment_count(void);






#define GGML_EXPERT_STORE_HASH_WINDOW (16*1024)
GGML_API uint64_t ggml_expert_store_hash_expert(const void * data, size_t size);










GGML_API bool ggml_expert_store_verify_placement(
        ggml_expert_store * store, int tensor_id, const char * tensor_name,
        int64_t * n_checked, int64_t * n_skipped);


GGML_API void ggml_expert_store_clear_source_hashes(void);







GGML_API int ggml_expert_store_add_tensor(
        ggml_expert_store * store,
        int layer, int kind,
        int64_t expert_base, int64_t n_expert, size_t blob_bytes, size_t blob_pad, size_t type_size,
        const int64_t * slot_experts, int64_t n_slot,
        int64_t host_budget);




GGML_API void ggml_expert_store_write(
        ggml_expert_store * store, int tensor_id,
        const void * data, size_t offset, size_t size);


GGML_API void ggml_expert_store_read(
        ggml_expert_store * store, int tensor_id,
        void * data, size_t offset, size_t size);


GGML_API void ggml_expert_store_memset(
        ggml_expert_store * store, int tensor_id,
        uint8_t value, size_t offset, size_t size);





GGML_API bool ggml_expert_store_verify(
        ggml_expert_store * store, int tensor_id);


GGML_API void ggml_expert_store_print_plan(const ggml_expert_store * store);


GGML_API void ggml_expert_store_log_stats(const ggml_expert_store * store, const char * tag);


GGML_API void ggml_expert_store_reset_stats(ggml_expert_store * store);
GGML_API const ggml_expert_store_tensor * ggml_expert_store_tensor_at(const ggml_expert_store * store, int tensor_id);








GGML_API bool ggml_expert_store_tensor_swap(ggml_expert_store_tensor * st, int64_t expert_in, int64_t expert_out);







GGML_API bool ggml_expert_store_tensor_swap_to(ggml_expert_store_tensor * st, int64_t expert_in, int64_t expert_out,
        char * slot_dev, char ** slot_old);










GGML_API int64_t ggml_expert_store_swap_epoch(const ggml_expert_store * store);



GGML_API int32_t ggml_expert_store_tensor_residency(const ggml_expert_store_tensor * st, int64_t expert);
GGML_API int64_t ggml_expert_store_tensor_host_offset(const ggml_expert_store_tensor * st, int64_t expert);


GGML_API int64_t ggml_expert_store_tensor_n_slot(const ggml_expert_store_tensor * st);
GGML_API int64_t ggml_expert_store_tensor_n_host(const ggml_expert_store_tensor * st);





GGML_API bool ggml_expert_store_tensor_check(const ggml_expert_store_tensor * st);

GGML_API int ggml_expert_store_tensor_id(const ggml_expert_store * store, const ggml_tensor * tensor);
GGML_API void ggml_expert_store_set_tensor_id(ggml_expert_store * store, const ggml_tensor * tensor, int tensor_id);



GGML_API void ggml_expert_store_set_tensor_stats(ggml_expert_store * store, int tensor_id,
        const uint32_t * counts, const uint32_t * launches, size_t n);
