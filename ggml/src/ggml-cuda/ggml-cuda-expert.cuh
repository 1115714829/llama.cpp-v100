#pragma once

// CUDA side of the expert store (ggml-expert-store.h): implements the memory operations
// with device allocations + device-readable host memory, owns the per-tensor device
// pointer tables, and stages the host tier experts into VRAM for prefill.
//
// Prefill staging (the streamed ring of Strata src/prefill/prefill.cpp, MIT,
// licenses/LICENSE-strata): the host experts of every layer of a tensor kind stream through a ring
// of expert slots, in layer and expert order, and the consumer of a layer takes the experts of its
// layer out of the ring as they arrive. A ring holds one layer's worth of experts, so a block of
// the prompt fills it again; the slot of an expert is a static function of its place in the order,
// so the pointer table of a tensor stays static. A producer thread per ring copies the arcs on the
// copy stream of the ring, gated on the release of the arc that held the slot before, so the copy
// of a layer's experts runs beside the attention half of that layer; the consumer waits for the
// copied event of its arc on the compute stream. Decode batches stage only the routed host experts
// with one kernel (ggml_cuda_expert_stage_prepare_decode). Details in ggml-cuda-expert.cu.

#include "common.cuh"

#include "../ggml-expert-store.h"





static inline size_t ggml_cuda_expert_blob_pad(ggml_type type, int64_t ne0) {
    if (!ggml_is_quantized(type) || ne0 % MATRIX_ROW_PADDING == 0) {
        return 0;
    }
    return ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
}


struct ggml_cuda_expert_table {
    const char * const * main  = nullptr;
    const char * const * stage = nullptr;




    char * const * main_host = nullptr;
    int64_t     n_expert   = 0;
    size_t      blob_bytes = 0;
    const char * stage_dev = nullptr;
    size_t      stage_size = 0;
    bool        has_host   = false;
    bool        has_stage  = false;
    int         tensor_id  = -1;




    bool        uniform     = false;
    const char * plain_base = nullptr;
    size_t       plain_pitch = 0;





    int64_t     expert_start = 0;
};

bool ggml_cuda_expert_buft_is_expert(ggml_backend_buffer_type_t buft);





void ggml_cuda_expert_mark_tensor(ggml_backend_buffer_type_t buft, ggml_tensor * t);





bool ggml_cuda_expert_is_store(const ggml_tensor * t);



bool ggml_cuda_expert_supported(void);





bool ggml_cuda_expert_table_get(const ggml_tensor * src0, ggml_cuda_expert_table * table);









ggml_cuda_expert_lut ggml_cuda_mul_mat_id_lut(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);








const char * const * ggml_cuda_expert_stage_prepare(const ggml_tensor * src0, int64_t n_tokens,
        const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream);






void ggml_cuda_expert_stage_consumed(const ggml_tensor * src0, cudaStream_t stream);



struct ggml_cuda_expert_stage_scope {
    const ggml_tensor * src0  = nullptr;
    cudaStream_t        stream = nullptr;
    ggml_cuda_expert_stage_scope(const ggml_tensor * s, cudaStream_t st) : src0(s), stream(st) {}
    ~ggml_cuda_expert_stage_scope() {
        if (src0 != nullptr) {
            ggml_cuda_expert_stage_consumed(src0, stream);
        }
    }
    ggml_cuda_expert_stage_scope(const ggml_cuda_expert_stage_scope &) = delete;
    ggml_cuda_expert_stage_scope & operator=(const ggml_cuda_expert_stage_scope &) = delete;
};




const char * const * ggml_cuda_expert_stage_prepare_decode(
        const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream);




void ggml_cuda_expert_record_usage(const ggml_tensor * src0, const ggml_tensor * ids, const ggml_cuda_expert_lut & lut, cudaStream_t stream);







int ggml_cuda_expert_counts_read(const ggml_tensor * weight, uint32_t * out, bool reset, cudaStream_t stream);




void ggml_backend_cuda_expert_verify_all(void);


void ggml_cuda_expert_stats_tick(const ggml_tensor * src0, cudaStream_t stream);






void ggml_cuda_expert_note_reader(const ggml_tensor * src0, int tensor_id, cudaStream_t stream);





int64_t ggml_cuda_expert_spare_slots(void);












int ggml_cuda_expert_swap_test(ggml_tensor * weight, const int32_t * pairs, int n_pairs, int32_t * pairs_out);











void ggml_cuda_expert_maintain(ggml_backend_cuda_context & cuda_ctx, const ggml_cgraph * cgraph);




int ggml_cuda_expert_residency(const ggml_tensor * weight, int32_t * out);





const char * ggml_backend_cuda_expert_buffer_type_name(ggml_backend_buffer_type_t buft);






void ggml_cuda_expert_set_budget(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t stage_bytes);





void ggml_cuda_expert_record_plan(ggml_backend_buffer_type_t buft, size_t slot_bytes, size_t host_bytes);





void ggml_cuda_expert_set_tensor_base(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor, int64_t expert_base);











void ggml_cuda_expert_plan(
        ggml_backend_buffer_type_t buft,
        const int * layer, const int64_t * expert_base, const int64_t * n_expert, const size_t * blob_bytes,
        const int * type, const int64_t * ne0, const int * kind, int n_expert_used, size_t n_tensors,
        size_t slot_budget,
        size_t * slot_bytes, size_t * host_bytes, size_t * host_peak, size_t * host_bytes_each,
        double * routed_share);




bool ggml_backend_cuda_expert_get_allocation(
        ggml_backend_buffer_type_t buft, size_t * slot_bytes, size_t * host_bytes, size_t * stage_bytes);



void ggml_cuda_expert_reset_allocation(ggml_backend_buffer_type_t buft);





void ggml_backend_cuda_expert_device_ready(ggml_backend_buffer_type_t buft);




bool ggml_backend_cuda_expert_get_host_peak(ggml_backend_buffer_type_t buft, size_t * peak_bytes);



void ggml_cuda_expert_set_slot_limit(ggml_backend_buffer_type_t buft, size_t bytes);



void ggml_cuda_expert_reset_streams(ggml_backend_buffer_type_t buft);




bool ggml_cuda_expert_last_stage_hits(ggml_backend_buffer_type_t buft, size_t * n_hit);



bool ggml_cuda_expert_stage_counters(ggml_backend_buffer_type_t buft, size_t * cold, size_t * hits);







void ggml_cuda_expert_comm_done(int device, cudaStream_t stream);









#define GGML_CUDA_NUMA_NODE_ANY  (-2)
#define GGML_CUDA_NUMA_NODE_FAIL (-1)




int ggml_cuda_numa_prepare_host_plan(const size_t * dev_bytes, size_t n_dev);




int ggml_cuda_numa_reserve(int device, size_t size);


void ggml_cuda_numa_release(int device, int node, size_t size);







int ggml_cuda_numa_fixed_alloc(size_t size);



void * ggml_cuda_numa_host_alloc(int device, size_t size, int node);



void ggml_cuda_numa_log_placement(const char * what);


void ggml_cuda_numa_log_final_placement(void);










enum ggml_cuda_host_pool_kind {
    GGML_CUDA_HOST_POOL_EXPERT = 0,
    GGML_CUDA_HOST_POOL_MAPPED = 1,
    GGML_CUDA_HOST_POOL_PINNED = 2,
    GGML_CUDA_HOST_POOL_COUNT  = 3,
};



void ggml_cuda_host_pool_add(int pool, int node, size_t size);


void ggml_cuda_host_pool_sub(int pool, int node, size_t size);



std::string ggml_cuda_host_pool_report(void);









bool ggml_cuda_numa_bind_cpus(int node);
void ggml_cuda_numa_unbind_cpus(void);
