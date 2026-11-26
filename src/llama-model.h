#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-ple-reader.h"
#include "llama-vocab.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;
struct llama_model;

enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_14M,
    LLM_TYPE_17M,
    LLM_TYPE_22M,
    LLM_TYPE_33M,
    LLM_TYPE_47M,
    LLM_TYPE_60M,
    LLM_TYPE_70M,
    LLM_TYPE_80M,
    LLM_TYPE_109M,
    LLM_TYPE_137M,
    LLM_TYPE_140M,
    LLM_TYPE_149M,
    LLM_TYPE_160M,
    LLM_TYPE_190M,
    LLM_TYPE_220M,
    LLM_TYPE_230M,
    LLM_TYPE_250M,
    LLM_TYPE_256M,
    LLM_TYPE_270M,
    LLM_TYPE_335M,
    LLM_TYPE_350M,
    LLM_TYPE_360M,
    LLM_TYPE_395M,
    LLM_TYPE_410M,
    LLM_TYPE_450M,
    LLM_TYPE_475M,
    LLM_TYPE_558M,
    LLM_TYPE_700M,
    LLM_TYPE_770M,
    LLM_TYPE_780M,
    LLM_TYPE_950M,
    LLM_TYPE_0_3B,
    LLM_TYPE_0_5B,
    LLM_TYPE_0_6B,
    LLM_TYPE_0_8B,
    LLM_TYPE_1B,
    LLM_TYPE_1_2B,
    LLM_TYPE_1_3B,
    LLM_TYPE_1_4B,
    LLM_TYPE_1_5B,
    LLM_TYPE_1_6B,
    LLM_TYPE_1_7B,
    LLM_TYPE_1_8B,
    LLM_TYPE_2B,
    LLM_TYPE_2_6B,
    LLM_TYPE_2_8B,
    LLM_TYPE_2_9B,
    LLM_TYPE_3B,
    LLM_TYPE_4B,
    LLM_TYPE_6B,
    LLM_TYPE_6_9B,
    LLM_TYPE_7B,
    LLM_TYPE_8B,
    LLM_TYPE_9B,
    LLM_TYPE_11B,
    LLM_TYPE_12B,
    LLM_TYPE_13B,
    LLM_TYPE_14B,
    LLM_TYPE_15B,
    LLM_TYPE_16B,
    LLM_TYPE_20B,
    LLM_TYPE_26B,
    LLM_TYPE_27B,
    LLM_TYPE_30B,
    LLM_TYPE_31B,
    LLM_TYPE_32B,
    LLM_TYPE_34B,
    LLM_TYPE_35B,
    LLM_TYPE_36B,
    LLM_TYPE_40B,
    LLM_TYPE_65B,
    LLM_TYPE_70B,
    LLM_TYPE_120B,
    LLM_TYPE_142B,
    LLM_TYPE_236B,
    LLM_TYPE_290B,
    LLM_TYPE_314B,
    LLM_TYPE_405B,
    LLM_TYPE_456B,
    LLM_TYPE_671B,
    LLM_TYPE_SMALL,
    LLM_TYPE_MEDIUM,
    LLM_TYPE_LARGE,
    LLM_TYPE_XL,
    LLM_TYPE_A1_7B,
    LLM_TYPE_A2_7B,
    LLM_TYPE_8x7B,
    LLM_TYPE_8x22B,
    LLM_TYPE_16x12B,
    LLM_TYPE_16x3_8B,
    LLM_TYPE_10B_128x3_66B,
    LLM_TYPE_57B_A14B,
    LLM_TYPE_17B_16E,
    LLM_TYPE_17B_128E,
    LLM_TYPE_A13B,
    LLM_TYPE_1B_A400M,
    LLM_TYPE_3B_A800M,
    LLM_TYPE_7B_A1B,
    LLM_TYPE_8B_A1B,
    LLM_TYPE_7_9B_A1_3B,
    LLM_TYPE_12B_A2_5B,
    LLM_TYPE_16B_A1B,
    LLM_TYPE_21B_A3B,
    LLM_TYPE_24B_A2B,
    LLM_TYPE_26B_A4B,
    LLM_TYPE_30B_A3B,
    LLM_TYPE_31B_A3_5B,
    LLM_TYPE_32B_A9B,
    LLM_TYPE_35B_A3B,
    LLM_TYPE_48B_A3B,
    LLM_TYPE_75B_A9B,
    LLM_TYPE_80B_A3B,
    LLM_TYPE_A3B,
    LLM_TYPE_100B_A6B,
    LLM_TYPE_102B_A12B,
    LLM_TYPE_106B_A12B,
    LLM_TYPE_118B_A8B,
    LLM_TYPE_120B_A12B,
    LLM_TYPE_122B_A10B,
    LLM_TYPE_124B_A5_1B,
    LLM_TYPE_196B_A11B,
    LLM_TYPE_230B_A10B,
    LLM_TYPE_428B_A23B,
    LLM_TYPE_235B_A22B,
    LLM_TYPE_288B_A19B,
    LLM_TYPE_300B_A47B,
    LLM_TYPE_310B_A15B,
    LLM_TYPE_355B_A32B,
    LLM_TYPE_397B_A17B,
    LLM_TYPE_685B_A37B,
    LLM_TYPE_744B_A40B,
    LLM_TYPE_2_8T_A50B,
    LLM_TYPE_320B_A18B,
    LLM_TYPE_E2B,
    LLM_TYPE_E4B,
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order);

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback);

struct llama_layer_posnet {

    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer_shortconv {
    struct ggml_tensor * in_proj  = nullptr;
    struct ggml_tensor * conv     = nullptr;
    struct ggml_tensor * out_proj = nullptr;
};

struct llama_layer_nextn {
    struct ggml_tensor * eh_proj               = nullptr;
    struct ggml_tensor * eh_proj_s             = nullptr;
    struct ggml_tensor * eh_proj_in_s          = nullptr;
    struct ggml_tensor * embed_tokens          = nullptr;
    struct ggml_tensor * enorm                 = nullptr;
    struct ggml_tensor * hnorm                 = nullptr;
    struct ggml_tensor * shared_head_head      = nullptr;
    struct ggml_tensor * shared_head_head_s    = nullptr;
    struct ggml_tensor * shared_head_head_in_s = nullptr;
    struct ggml_tensor * shared_head_norm      = nullptr;

    struct ggml_tensor * hc_head_norm          = nullptr;
    struct ggml_tensor * hc_head_down          = nullptr;
    struct ggml_tensor * hc_head_up            = nullptr;
};

struct llama_layer_switch_lora {
    struct ggml_tensor * a_q    = nullptr;
    struct ggml_tensor * b_q    = nullptr;
    struct ggml_tensor * a_k    = nullptr;
    struct ggml_tensor * b_k    = nullptr;
    struct ggml_tensor * a_v    = nullptr;
    struct ggml_tensor * b_v    = nullptr;
    struct ggml_tensor * a_o    = nullptr;
    struct ggml_tensor * b_o    = nullptr;

    struct ggml_tensor * a_gate = nullptr;
    struct ggml_tensor * b_gate = nullptr;
    struct ggml_tensor * a_up   = nullptr;
    struct ggml_tensor * b_up   = nullptr;
    struct ggml_tensor * a_down = nullptr;
    struct ggml_tensor * b_down = nullptr;
};

struct llama_layer {

    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_norm_b     = nullptr;
    struct ggml_tensor * attn_norm_2     = nullptr;
    struct ggml_tensor * attn_norm_2_b   = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_q_norm_b   = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_k_norm_b   = nullptr;
    struct ggml_tensor * attn_out_norm   = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm   = nullptr;
    struct ggml_tensor * attn_kv_a_norm  = nullptr;
    struct ggml_tensor * attn_sub_norm   = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_sub_norm    = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc   = nullptr;
    struct ggml_tensor * ssm_norm        = nullptr;
    struct ggml_tensor * ssm_dt_norm     = nullptr;
    struct ggml_tensor * ssm_b_norm      = nullptr;
    struct ggml_tensor * ssm_c_norm      = nullptr;

    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wg        = nullptr;
    struct ggml_tensor * wq_a      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkv_b     = nullptr;
    struct ggml_tensor * wkv       = nullptr;
    struct ggml_tensor * wk_b      = nullptr;
    struct ggml_tensor * wv_b      = nullptr;
    struct ggml_tensor * wqkv_b    = nullptr;
    struct ggml_tensor * wo_a      = nullptr;
    struct ggml_tensor * wo_b      = nullptr;
    struct ggml_tensor * wq_cross  = nullptr;
    struct ggml_tensor * wk_cross  = nullptr;
    struct ggml_tensor * wv_cross  = nullptr;
    struct ggml_tensor * wo_cross  = nullptr;
    struct ggml_tensor * wq_enc    = nullptr;
    struct ggml_tensor * wk_enc    = nullptr;
    struct ggml_tensor * wv_enc    = nullptr;
    struct ggml_tensor * wo_enc    = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;

    struct ggml_tensor * attn_v_gate   = nullptr;
    struct ggml_tensor * attn_v_gate_b = nullptr;
    struct ggml_tensor * attn_v_exps   = nullptr;

    struct ggml_tensor * attn_rel_b       = nullptr;
    struct ggml_tensor * attn_rel_b_enc   = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_norm_b       = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;
    struct ggml_tensor * ffn_post_norm_1  = nullptr;
    struct ggml_tensor * ffn_post_norm_2  = nullptr;
    struct ggml_tensor * ffn_pre_norm_2   = nullptr;
    struct ggml_tensor * layer_out_norm   = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps    = nullptr;
    struct ggml_tensor * ffn_norm_enc     = nullptr;

    struct ggml_tensor * ffn_gate     = nullptr;
    struct ggml_tensor * ffn_down     = nullptr;
    struct ggml_tensor * ffn_up       = nullptr;
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc   = nullptr;

    struct ggml_tensor * ffn_gate_inp      = nullptr;
    struct ggml_tensor * ffn_gate_inp_s    = nullptr;
    struct ggml_tensor * ffn_gate_exps     = nullptr;
    struct ggml_tensor * ffn_down_exps     = nullptr;
    struct ggml_tensor * ffn_up_exps       = nullptr;
    struct ggml_tensor * ffn_gate_up_exps  = nullptr;
    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
    struct ggml_tensor * ffn_gate_exps_b   = nullptr;
    struct ggml_tensor * ffn_down_exps_b   = nullptr;
    struct ggml_tensor * ffn_up_exps_b     = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_b = nullptr;

    struct ggml_tensor * ffn_gate_exps_s   = nullptr;
    struct ggml_tensor * ffn_down_exps_s   = nullptr;
    struct ggml_tensor * ffn_up_exps_s     = nullptr;

    struct ggml_tensor * ffn_latent_down = nullptr;
    struct ggml_tensor * ffn_latent_up   = nullptr;

    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    struct ggml_tensor * ffn_gate_chexps     = nullptr;
    struct ggml_tensor * ffn_down_chexps     = nullptr;
    struct ggml_tensor * ffn_up_chexps       = nullptr;

    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr;
    struct ggml_tensor * ffn_up_b   = nullptr;
    struct ggml_tensor * ffn_act    = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;
    struct ggml_tensor * ffn_exp_probs_b_vl = nullptr;
    struct ggml_tensor * ffn_gate_tid2eid = nullptr;

    struct ggml_tensor * dflash_attn_conv_base = nullptr;
    struct ggml_tensor * dflash_attn_conv_proj = nullptr;
    struct ggml_tensor * dflash_ffn_conv_base  = nullptr;
    struct ggml_tensor * dflash_ffn_conv_proj  = nullptr;

    struct ggml_tensor * lilicorr_attn_norm       = nullptr;
    struct ggml_tensor * lilicorr_mlp_norm        = nullptr;
    struct ggml_tensor * lilicorr_attn_in_proj    = nullptr;
    struct ggml_tensor * lilicorr_attn_in_proj_b  = nullptr;
    struct ggml_tensor * lilicorr_attn_out_proj   = nullptr;
    struct ggml_tensor * lilicorr_attn_out_proj_b = nullptr;
    struct ggml_tensor * lilicorr_attn_out_proj_s = nullptr;
    struct ggml_tensor * lilicorr_mlp_0           = nullptr;
    struct ggml_tensor * lilicorr_mlp_0_b         = nullptr;
    struct ggml_tensor * lilicorr_mlp_0_s         = nullptr;
    struct ggml_tensor * lilicorr_mlp_2           = nullptr;
    struct ggml_tensor * lilicorr_mlp_2_b         = nullptr;
    struct ggml_tensor * lilicorr_mlp_2_s         = nullptr;

    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;

    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    struct ggml_tensor * ssm_beta_alpha = nullptr;

    struct ggml_tensor * ssm_alpha = nullptr;

    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    struct ggml_tensor * time_mix_w0         = nullptr;
    struct ggml_tensor * time_mix_a0         = nullptr;
    struct ggml_tensor * time_mix_a1         = nullptr;
    struct ggml_tensor * time_mix_a2         = nullptr;
    struct ggml_tensor * time_mix_v0         = nullptr;
    struct ggml_tensor * time_mix_v1         = nullptr;
    struct ggml_tensor * time_mix_v2         = nullptr;
    struct ggml_tensor * time_mix_g1         = nullptr;
    struct ggml_tensor * time_mix_g2         = nullptr;
    struct ggml_tensor * time_mix_k_k        = nullptr;
    struct ggml_tensor * time_mix_k_a        = nullptr;
    struct ggml_tensor * time_mix_r_k        = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    struct ggml_tensor * wq_s       = nullptr;
    struct ggml_tensor * wk_s       = nullptr;
    struct ggml_tensor * wv_s       = nullptr;
    struct ggml_tensor * wo_s       = nullptr;
    struct ggml_tensor * wqkv_s     = nullptr;
    struct ggml_tensor * wqkv_gate_s = nullptr;
    struct ggml_tensor * ffn_gate_s = nullptr;
    struct ggml_tensor * ffn_up_s   = nullptr;
    struct ggml_tensor * ffn_down_s = nullptr;
    struct ggml_tensor * ffn_gate_shexp_s = nullptr;
    struct ggml_tensor * ffn_up_shexp_s   = nullptr;
    struct ggml_tensor * ffn_down_shexp_s = nullptr;
    struct ggml_tensor * ssm_in_s    = nullptr;
    struct ggml_tensor * ssm_out_s   = nullptr;
    struct ggml_tensor * ssm_alpha_s = nullptr;
    struct ggml_tensor * ssm_beta_s  = nullptr;

    struct ggml_tensor * wq_in_s            = nullptr;
    struct ggml_tensor * wk_in_s            = nullptr;
    struct ggml_tensor * wv_in_s            = nullptr;
    struct ggml_tensor * wo_in_s            = nullptr;
    struct ggml_tensor * wqkv_in_s          = nullptr;
    struct ggml_tensor * wqkv_gate_in_s     = nullptr;
    struct ggml_tensor * ffn_gate_in_s      = nullptr;
    struct ggml_tensor * ffn_up_in_s        = nullptr;
    struct ggml_tensor * ffn_down_in_s      = nullptr;
    struct ggml_tensor * ffn_gate_exps_in_s = nullptr;
    struct ggml_tensor * ffn_down_exps_in_s = nullptr;
    struct ggml_tensor * ffn_up_exps_in_s   = nullptr;
    struct ggml_tensor * ffn_gate_shexp_in_s= nullptr;
    struct ggml_tensor * ffn_up_shexp_in_s  = nullptr;
    struct ggml_tensor * ffn_down_shexp_in_s= nullptr;
    struct ggml_tensor * ssm_in_in_s        = nullptr;
    struct ggml_tensor * ssm_out_in_s       = nullptr;
    struct ggml_tensor * ssm_alpha_in_s     = nullptr;
    struct ggml_tensor * ssm_beta_in_s      = nullptr;

    struct ggml_tensor * per_layer_inp_gate   = nullptr;
    struct ggml_tensor * per_layer_proj       = nullptr;
    struct ggml_tensor * per_layer_post_norm  = nullptr;
    struct ggml_tensor * altup_correct_coef   = nullptr;
    struct ggml_tensor * altup_correct_scale  = nullptr;
    struct ggml_tensor * altup_predict_coef   = nullptr;
    struct ggml_tensor * altup_router         = nullptr;
    struct ggml_tensor * altup_router_norm    = nullptr;
    struct ggml_tensor * laurel_l             = nullptr;
    struct ggml_tensor * laurel_r             = nullptr;
    struct ggml_tensor * laurel_post_norm     = nullptr;

    struct ggml_tensor * attn_sinks = nullptr;

    struct ggml_tensor * attn_kv_norm = nullptr;
    struct ggml_tensor * hc_attn_fn   = nullptr;
    struct ggml_tensor * hc_attn_base = nullptr;
    struct ggml_tensor * hc_attn_scale = nullptr;
    struct ggml_tensor * hc_ffn_fn    = nullptr;
    struct ggml_tensor * hc_ffn_base  = nullptr;
    struct ggml_tensor * hc_ffn_scale = nullptr;
    struct ggml_tensor * attn_comp_wkv   = nullptr;
    struct ggml_tensor * attn_comp_wgate = nullptr;
    struct ggml_tensor * attn_comp_ape   = nullptr;
    struct ggml_tensor * attn_comp_norm  = nullptr;
    struct ggml_tensor * indexer_comp_wkv   = nullptr;
    struct ggml_tensor * indexer_comp_wgate = nullptr;
    struct ggml_tensor * indexer_comp_ape   = nullptr;
    struct ggml_tensor * indexer_comp_norm  = nullptr;

    struct ggml_tensor * visexp_attn_wqkv = nullptr;
    struct ggml_tensor * visexp_attn_wo   = nullptr;
    struct ggml_tensor * visexp_ffn_gate  = nullptr;
    struct ggml_tensor * visexp_ffn_down  = nullptr;
    struct ggml_tensor * visexp_ffn_up    = nullptr;

    struct ggml_tensor * ffn_act_alpha_n = nullptr;
    struct ggml_tensor * ffn_act_alpha_p = nullptr;
    struct ggml_tensor * ffn_act_beta    = nullptr;
    struct ggml_tensor * ffn_act_eps     = nullptr;

    struct ggml_tensor * ssm_q_conv = nullptr;
    struct ggml_tensor * ssm_k_conv = nullptr;
    struct ggml_tensor * ssm_v_conv = nullptr;
    struct ggml_tensor * ssm_f_a    = nullptr;
    struct ggml_tensor * ssm_f_b    = nullptr;
    struct ggml_tensor * ssm_beta   = nullptr;
    struct ggml_tensor * ssm_g_a    = nullptr;
    struct ggml_tensor * ssm_g_b    = nullptr;
    struct ggml_tensor * ssm_o_norm = nullptr;

    struct ggml_tensor * ssm_g           = nullptr;
    struct ggml_tensor * attn_res_score  = nullptr;
    struct ggml_tensor * ffn_res_score   = nullptr;
    struct ggml_tensor * ffn_routed_down = nullptr;
    struct ggml_tensor * ffn_routed_up   = nullptr;
    struct ggml_tensor * ffn_routed_norm = nullptr;

    struct ggml_tensor * indexer_k_norm   = nullptr;
    struct ggml_tensor * indexer_k_norm_b = nullptr;
    struct ggml_tensor * indexer_proj     = nullptr;
    struct ggml_tensor * indexer_attn_k   = nullptr;
    struct ggml_tensor * indexer_attn_q_b = nullptr;

    struct ggml_tensor * indexer_kpool_gate = nullptr;
    struct ggml_tensor * indexer_kpool_ape  = nullptr;

    struct ggml_tensor * index_q_proj = nullptr;
    struct ggml_tensor * index_k_proj = nullptr;
    struct ggml_tensor * index_q_norm = nullptr;
    struct ggml_tensor * index_k_norm = nullptr;

    struct ggml_tensor * hc_attn_norm   = nullptr;
    struct ggml_tensor * hc_attn_down   = nullptr;
    struct ggml_tensor * hc_attn_up     = nullptr;
    struct ggml_tensor * hc_attn_inject = nullptr;
    struct ggml_tensor * hc_ffn_norm    = nullptr;
    struct ggml_tensor * hc_ffn_down    = nullptr;
    struct ggml_tensor * hc_ffn_up      = nullptr;
    struct ggml_tensor * hc_ffn_inject  = nullptr;

    struct ggml_tensor * ple_key        = nullptr;
    struct ggml_tensor * ple_value      = nullptr;
    struct ggml_tensor * ple_norm_key   = nullptr;
    struct ggml_tensor * ple_norm_query = nullptr;
    struct ggml_tensor * ple_norm_conv  = nullptr;
    struct ggml_tensor * ple_conv1d     = nullptr;

    struct ggml_tensor * out_scale = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;

    struct llama_layer_shortconv shortconv;

    struct llama_layer_nextn nextn;

    struct llama_layer_switch_lora switch_lora;
};

struct llama_device {
    bool is_meta;

    ggml_backend_dev_t dev;
};

struct llama_meta_device_get_split_state_userdata {
    size_t                     n_devices;
    const struct llama_model * model;
};

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata);

bool attn_kv_overlap_3card(const llama_hparams & hparams, const llama_model & model, uint32_t il);

struct llama_prec_policy {

    std::unordered_map<const ggml_tensor *, ggml_prec> prec_acc;

    std::unordered_map<const ggml_tensor *, ggml_prec> prec_src1;

    bool apply(ggml_tensor * res) const;

    void load(llama_model_loader & ml, const llama_model & model);
};

struct llama_hadamard_transform {
    ggml_tensor * rot;
    ggml_tensor * signs;

    int64_t perm_hd  = 0;
    int64_t perm_nk  = 0;
    int64_t perm_rep = 0;

    bool set_gdn_v_perm(int64_t n_in, int64_t n_v, int64_t n_k) {
        if (n_k <= 0 || n_v <= 0 || n_v % n_k != 0 || n_in % n_v != 0) {
            return false;
        }
        perm_hd  = n_in / n_v;
        perm_nk  = n_k;
        perm_rep = n_v / n_k;
        return true;
    }
};
using llama_hadamard_rotations = std::unordered_map<const ggml_tensor *, llama_hadamard_transform>;

struct llama_hadamard {

    std::unordered_map<std::string, uint32_t> weight_blocks;
    std::unordered_map<std::string, uint32_t> inverse_blocks;

    std::map<uint32_t, std::vector<int32_t>> sign_data;

    bool gdn_v_grouped = false;
    bool tied_output = false;

    llama_hadamard_rotations rot;
    llama_hadamard_rotations inv;
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    bool h_nextn_sink_ok = false;

    bool draft_ring_ok = false;

    bool gate_up_fused_swiglu = false;

    llama_hparams hparams = {};
    llama_vocab   vocab;

    llama_prec_policy prec_policy;

    llama_hadamard hdmd;

    bool attn_rot = true;

    std::vector<std::string> classifier_labels;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_res_score = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    struct ggml_tensor * output_s    = nullptr;
    struct ggml_tensor * output_in_s = nullptr;

    struct ggml_tensor * nextn_proj_pre  = nullptr;
    struct ggml_tensor * nextn_proj_post = nullptr;

    struct ggml_tensor * hrm_z_l_init = nullptr;

    struct ggml_tensor * hc_head_fn    = nullptr;
    struct ggml_tensor * hc_head_base  = nullptr;
    struct ggml_tensor * hc_head_scale = nullptr;

    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;
    struct ggml_tensor * cls_norm  = nullptr;
    struct ggml_tensor * cls_norm_b = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    struct ggml_tensor * altup_proj           = nullptr;
    struct ggml_tensor * altup_unembd_proj    = nullptr;
    struct ggml_tensor * per_layer_tok_embd   = nullptr;

    struct ggml_tensor * hc_head_norm = nullptr;
    struct ggml_tensor * hc_head_down = nullptr;
    struct ggml_tensor * hc_head_up   = nullptr;
    struct ggml_tensor * per_layer_model_proj = nullptr;
    struct ggml_tensor * per_layer_proj_norm  = nullptr;

    struct ggml_tensor * fc   = nullptr;
    struct ggml_tensor * fc_s = nullptr;
    struct ggml_tensor * d2t = nullptr;

    struct ggml_tensor * dspark_markov_w1   = nullptr;
    struct ggml_tensor * dspark_markov_w2   = nullptr;
    struct ggml_tensor * dspark_markov_w2_s = nullptr;
    struct ggml_tensor * dspark_conf_proj   = nullptr;
    struct ggml_tensor * dspark_conf_proj_b = nullptr;

    struct ggml_tensor * dflash_selector_prev   = nullptr;
    struct ggml_tensor * dflash_selector_next   = nullptr;
    struct ggml_tensor * dflash_selector_hidden = nullptr;

    struct ggml_tensor * lilicorr_token_proj          = nullptr;
    struct ggml_tensor * lilicorr_token_proj_b        = nullptr;
    struct ggml_tensor * lilicorr_token_proj_s        = nullptr;
    struct ggml_tensor * lilicorr_pass_hidden_proj    = nullptr;
    struct ggml_tensor * lilicorr_pass_hidden_proj_b  = nullptr;
    struct ggml_tensor * lilicorr_pass_hidden_proj_s  = nullptr;
    struct ggml_tensor * lilicorr_feature_norm        = nullptr;
    struct ggml_tensor * lilicorr_feature_norm_b      = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_1       = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_1_b     = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_3       = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_3_b     = nullptr;
    struct ggml_tensor * lilicorr_slot_embedding      = nullptr;
    struct ggml_tensor * lilicorr_rank_embedding      = nullptr;
    struct ggml_tensor * lilicorr_attention_bias      = nullptr;
    struct ggml_tensor * lilicorr_context_proj        = nullptr;
    struct ggml_tensor * lilicorr_context_proj_b      = nullptr;
    struct ggml_tensor * lilicorr_context_proj_s      = nullptr;
    struct ggml_tensor * lilicorr_output_norm         = nullptr;
    struct ggml_tensor * lilicorr_anchor_norm         = nullptr;
    struct ggml_tensor * lilicorr_factor_input_proj   = nullptr;
    struct ggml_tensor * lilicorr_factor_input_proj_b = nullptr;
    struct ggml_tensor * lilicorr_out_head            = nullptr;
    struct ggml_tensor * lilicorr_out_head_b          = nullptr;
    struct ggml_tensor * lilicorr_in_head             = nullptr;
    struct ggml_tensor * lilicorr_in_head_b           = nullptr;
    struct ggml_tensor * lilicorr_anchor_out_head     = nullptr;
    struct ggml_tensor * lilicorr_anchor_out_head_b   = nullptr;

    std::vector<int32_t> target_layer_ids;

    std::vector<llama_layer> layers;

    //Dense linear projections for SentenceTransformers models like embeddinggemma
    // For Sentence Transformers models structure see
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    struct ggml_tensor * dense_2_out_layers   = nullptr;
    struct ggml_tensor * dense_2_out_layers_b = nullptr;
    struct ggml_tensor * dense_3_out_layers   = nullptr;

    std::unordered_map<std::string, std::string> gguf_kv;

    std::vector<llama_device> devices;

    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    std::unordered_set<llama_adapter_lora *> loras;

    std::unordered_set<const ggml_tensor *> can_prefetch;

    std::unique_ptr<llama_ple_reader> ple_reader;

    struct llama_meta_device_get_split_state_userdata get_split_state_ud;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const llama_model_params & params);
    virtual ~llama_model();

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    llama_ftype ftype() const;

    size_t size() const;
    size_t n_tensors() const;
    size_t n_devices() const;
    const float * tensor_split() const;

    uint32_t n_gpu_layers() const;
    llama_split_mode split_mode() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(int il) const;

    bool has_tensor_overrides() const;

    bool has_host_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    bool kq_range_mask_supported() const;

    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;

    bool expert_store_disabled() const { return params.no_expert_store; }
    bool mtp_host() const { return params.mtp_host; }

    ggml_cgraph * build_graph(const llm_graph_params & params) const;

    virtual void load_stats  (llama_model_loader & ml) = 0;
    virtual void load_hparams(llama_model_loader & ml) = 0;
    virtual void load_vocab  (llama_model_loader & ml) = 0;
    virtual bool load_tensors(llama_model_loader & ml) = 0;

    virtual void load_arch_hparams(llama_model_loader & ml) = 0;
    virtual void load_arch_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const = 0;

protected:
    llama_model_params params;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params);
llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params);

struct llama_model_base : public llama_model {
    friend struct llama_model;

    llama_model * model;
    llama_model_loader * ml = nullptr;
    const LLM_TN tn;

    const int TENSOR_DUPLICATED;
    const int TENSOR_NOT_REQUIRED;
    const int TENSOR_SKIP;
    const int TENSOR_SKIP_IF_VIRTUAL;
    const int TENSOR_ALLOW_RESHAPE;
    const int TENSOR_READ_LAZY;
    const int TENSOR_GATE_UP_INTERLEAVE;
    const int TENSOR_F32_AS_BF16;

    explicit llama_model_base(const llama_model_params & params);
    virtual ~llama_model_base() = default;

    ggml_tensor * create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags,
            const LLM_TN_IMPL * tn_src_gate = nullptr, const LLM_TN_IMPL * tn_src_up = nullptr);

    ggml_tensor * create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags,
            const LLM_TN_IMPL * tn_src_gate = nullptr, const LLM_TN_IMPL * tn_src_up = nullptr);

    void create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_,
                int64_t n_ff_, int64_t n_expert_, int flags, int flags_extra = 0);

    void create_tensor_qkv(llama_layer & layer, int bid,
                int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
                int flags);

    struct nextn_flags_t {
        int trunk;
        int mtp;
    };
    nextn_flags_t nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe = LLM_TENSOR_ATTN_NORM) const;

    void load_hparams_hadamard(llama_model_loader & ml);

    void load_tensors_hadamard();

    void load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first = false);

    void load_stats  (llama_model_loader & ml) override;
    void load_hparams(llama_model_loader & ml) override;
    void load_vocab  (llama_model_loader & ml) override;
    bool load_tensors(llama_model_loader & ml) override;

    void load_arch_hparams(llama_model_loader & ml) override = 0;
    void load_arch_tensors(llama_model_loader & ml) override = 0;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override = 0;
};

const char * llm_type_name(llm_type type);

#define LLAMA_LOAD_LOCALS \
    const int     n_layer        = hparams.n_layer();        GGML_UNUSED(n_layer); \
    const int     n_layer_all    = hparams.n_layer_all;      GGML_UNUSED(n_layer_all); \
    const int     n_layer_nextn  = hparams.n_layer_nextn;    GGML_UNUSED(n_layer_nextn); \
    const int64_t n_head         = hparams.n_head();         GGML_UNUSED(n_head); \
    const int64_t n_head_kv      = hparams.n_head_kv();      GGML_UNUSED(n_head_kv); \
    const int64_t n_embd         = hparams.n_embd;           GGML_UNUSED(n_embd); \
    const int64_t n_embd_k_gqa   = hparams.n_embd_k_gqa();   GGML_UNUSED(n_embd_k_gqa); \
    const int64_t n_embd_v_gqa   = hparams.n_embd_v_gqa();   GGML_UNUSED(n_embd_v_gqa); \
    const int64_t n_embd_head_k  = hparams.n_embd_head_k();  GGML_UNUSED(n_embd_head_k); \
    const int64_t n_embd_head_v  = hparams.n_embd_head_v();  GGML_UNUSED(n_embd_head_v); \
    const int64_t n_ff           = hparams.n_ff();           GGML_UNUSED(n_ff); \
    const int64_t n_embd_gqa     = n_embd_v_gqa;             GGML_UNUSED(n_embd_gqa); \
    const int64_t n_vocab        = vocab.n_tokens();         GGML_UNUSED(n_vocab); \
    const int64_t n_token_types  = vocab.n_token_types();    GGML_UNUSED(n_token_types); \
    const int64_t n_rot          = hparams.n_rot();          GGML_UNUSED(n_rot); \
    const int64_t n_expert       = hparams.n_expert;         GGML_UNUSED(n_expert); \
    const int64_t n_expert_used  = hparams.n_expert_used();  GGML_UNUSED(n_expert_used); \
    const int64_t n_ctx_train    = hparams.n_ctx_train;      GGML_UNUSED(n_ctx_train);

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
