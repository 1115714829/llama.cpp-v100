#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        // the rows of the tensor are rewritten from the concatenated gate/up order of the file
        // into the interleaved order read by ggml_mul_mat_id_swiglu, see TENSOR_GATE_UP_INTERLEAVE
        bool interleave_gate_up = false;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // one file tensor whose bytes go into a synthesized tensor, see llama_tensor_merged
    struct llama_tensor_weight_src {
        uint16_t idx; // source file index
        size_t   offs; // tensor data offset in the original file
        size_t   size; // size of the source tensor data in the file
    };

    // a tensor that the file does not have, synthesized while loading from the gate and the up
    // weight of the file: their rows are interleaved into it (row 2r = gate row r, row 2r+1 = up
    // row r), see TENSOR_GATE_UP_INTERLEAVE and llama_load_merged_gate_up
    struct llama_tensor_merged {
        llama_tensor_weight_src gate;
        llama_tensor_weight_src up;
        // the sources are BF16 in the file and the tensor is F16 (bf16_weights_as_f16)
        bool bf16_as_f16 = false;
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;
    static const int TENSOR_READ_LAZY       = 1 << 5; // read rows on demand instead of loading whole tensor; requires mmap for now
    // the tensor holds the rows of a gate and an up projection concatenated (n_ff gate rows followed
    // by n_ff up rows per expert) and is rewritten into the interleaved order (row 2r = gate row r,
    // row 2r+1 = up row r) that ggml_mul_mat_id_swiglu reads; see llama_interleave_gate_up_rows
    static const int TENSOR_GATE_UP_INTERLEAVE = 1 << 6;
    // an F32 weight of the file is stored as BF16 (rounded to nearest even while loading): the MoE
    // router of qwen4exp, whose F32 values in the GGUF are the BF16 values of the checkpoint, so the
    // conversion is exact for them and halves the bytes every token reads. BF16 is kept on devices
    // without native BF16 support too: the CUDA kernels read it into F32 and accumulate in F32
    // (mul_mat_vec_f, cuBLAS SGEMM after the BF16->F32 conversion), as for the F32 weight
    static const int TENSOR_F32_AS_BF16     = 1 << 7;

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;
    bool load_mtp;
    // do not mark the routed experts: they stay plain device weights instead of going into
    // the expert store (draft models, which use every expert every step)
    bool no_expert_store = false;
    // the n-gram table is read from the model file row by row: the page cache of the experts is
    // released after they are loaded, so it does not take the memory of the nodes they live on
    bool ple_host_read = false;

    // handle TENSOR_READ_LAZY
    // use case: keep PLE / engrams embd tensors on disk, read them on demand
    struct lazy_read {
        // set by the caller before the create_tensor() calls
        enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;

        // decide whether this tensor is read lazily
        // pass w to also record it, or nullptr to only ask
        bool add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w);

        bool any() const {
            return !ranges.empty();
        }

        bool has(const ggml_tensor * t) const {
            return tensors.count(ggml_get_name(t)) > 0;
        }

        const llama_mmap::ranges & for_file(uint32_t idx) const {
            static const llama_mmap::ranges none;

            const auto it = ranges.find(idx);
            return it == ranges.end() ? none : it->second;
        }

        // device that consumes the lazy tensors, set by the model before the tensors are created.
        // when it is a CUDA device the lazy table is loaded into device-readable mapped host
        // memory (mapped weights buffer); otherwise it stays in a CPU mmap buffer
        ggml_backend_dev_t dev = nullptr;

        // buffer type of a lazy tensor: a device-readable mapped weights buffer when the
        // consuming device offers one, otherwise the CPU buffer type
        ggml_backend_buffer_type_t buft() const;

    private:
        std::map<uint32_t, llama_mmap::ranges> ranges;
        std::set<std::string>                  tensors;
    } lazy;

    // handle the token embedding table of the input layer: the input layer stays on the CPU,
    // but the table is loaded into device-readable mapped host memory and read on the device
    // that holds layer 0, so that GET_ROWS runs there instead of splitting the graph back to
    // the CPU for it
    struct embd_read {
        // set by the model before the create_tensor() calls
        ggml_backend_dev_t dev = nullptr;

        // buffer type of the token embedding table: the mapped weights buffer of dev, or
        // nullptr when the device does not offer one
        ggml_backend_buffer_type_t buft() const;
    } embd;

    // target model a draft head borrows the shared tensors from, see borrow_shared_tensor()
    // from unsloth PR #142
    const struct llama_model * model_shared = nullptr;

    // true when the file's metadata says it left the shared tensors to the target
    // (nextn_shared_target_tensors). a draft that ships its own tokens or does not set the
    // key keeps its own tensors, see borrow_shared_tensor()
    bool nextn_shared_target_tensors = false;

    llama_files files;
    // the path of every file of the model, in the order of files: the reader of the n-gram table
    // opens the file the table is in
    std::vector<std::string> file_names;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    // tensors that the file does not have and that are synthesized from the gate and up weights
    // of this file while loading, keyed by tensor name
    std::map<std::string, llama_tensor_merged> merged_tensors;
    // file tensors consumed by a synthesized tensor: they are neither created nor loaded on their
    // own, their bytes are read into the synthesized tensor instead
    int n_merged_sources = 0;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // BF16 weights loaded as F16 on devices without native BF16 support
    size_t n_bf16_converted = 0; // elements
    size_t n_bf16_clamped   = 0; // elements clamped to +-65504

    // F32 weights stored as BF16 (TENSOR_F32_AS_BF16)
    size_t n_f32_as_bf16         = 0; // elements
    size_t n_f32_as_bf16_rounded = 0; // elements that BF16 does not represent exactly

    // The routed expert weights this loader created (by their role), whether they go into the
    // expert store or stay plain device weights (no_expert_store): both are laid out by the expert
    // assignment table of the model when they are split over the devices along the expert dim.
    std::set<const ggml_tensor *> expert_weights;

    // Expert tensors that are split over the devices along the expert dim, with the number of
    // experts every device holds (the slice lengths of the meta split, in device order). The
    // model fills this before load_all_data(), together with expert_assignment: the loader lays
    // every expert of these tensors out at the position of that table
    // (ggml_expert_assignment_load_position), the backend checks the placement after the load. A
    // tensor that is not in the map holds the whole expert dim and keeps the file order.
    std::map<const ggml_tensor *, std::vector<int64_t>> expert_dev_slices;
    // the expert assignment table of the model (0: no tensor is split along the expert dim)
    uint32_t expert_assignment = 0;
    // expert tensors laid out by the table in this load; the model activates the table for them
    // once they are all written
    std::vector<const ggml_tensor *> expert_tensors_by_table;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    // lazy tensors need dedicated context
    struct ctx_key {
        ggml_backend_buffer_type_t buft;
        bool lazy;
    };

    struct ctx_key_comparator {
        bool operator()(const ctx_key & lhs, const ctx_key & rhs) const {
            if (lhs.lazy != rhs.lazy) {
                return lhs.lazy < rhs.lazy;
            }
            return strcmp(ggml_backend_buft_name(lhs.buft), ggml_backend_buft_name(rhs.buft)) < 0;
        }
    };

    std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        bool no_expert_store,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    // true if the devices in the list have no native BF16 support, so BF16 weights should be loaded as F16
    static bool bf16_weights_as_f16(const buft_list_t * buft_list);

    // true if the tensor has another type than its weight in the file (a BF16 weight loaded as
    // F16, an F32 weight stored as BF16): it is rewritten while loading and cannot point into the mapping
    bool is_type_converted(const struct ggml_tensor * tensor) const;

    // tn_src_gate and tn_src_up name the two file tensors a gate/up tensor is synthesized from,
    // instead of looking the tensor up in the file. only valid when they are set
    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags,
        const LLM_TN_IMPL * tn_src_gate = nullptr, const LLM_TN_IMPL * tn_src_up = nullptr);

    // a draft head that shares the target's embeddings does not carry its own token_embd,
    // output or output_norm; take them from the target model instead. returns null unless the
    // file is missing token_embd, so a model that ships its own tensors is never affected
    // from unsloth PR #142
    struct ggml_tensor * borrow_shared_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne);

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // release a weight's mmap pages
    void unmap_weight(const llama_tensor_weight & w) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
