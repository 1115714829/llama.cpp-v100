#pragma once

#include "llama.h"
#include "common.h"
#include "sampling.h"

struct common_speculative;

std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

const char * common_speculative_all_types_str();

std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

enum common_speculative_type common_speculative_type_from_name(const std::string & name);

std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

int32_t common_speculative_n_max(const common_speculative * spec);

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative_output_limits common_get_output_limits(const common_params & params);

bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {

    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    const llama_tokens * prompt;

    llama_tokens * result;

    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    bool use_rejection = false;

    std::vector<common_sampler_draft_q> * result_q_dflash = nullptr;
    size_t * result_mtp_size = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

void common_speculative_set_prompt_end(common_speculative * spec, llama_seq_id seq_id, llama_pos pos_end);

bool common_speculative_process(common_speculative * spec, const common_batch & batch);

void common_speculative_draft(common_speculative * spec);

void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
