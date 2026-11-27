#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_gdn_conv_silu_l2norm(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * conv_states_all,
    const ggml_tensor * state_ids,
    const ggml_tensor * qkv_mixed,
    const ggml_tensor * conv_kernel,
    ggml_tensor *       state_cell_dst,
    ggml_tensor *       v_silu_dst,
    ggml_tensor *       q_norm_dst,
    ggml_tensor *       k_norm_dst,
    int64_t             head_k_dim,
    int64_t             num_k_heads,
    float               eps_over_n,
    float               inv_sqrt_n);

void ggml_cuda_op_gdn_alpha_beta(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * beta_in,
    const ggml_tensor * alpha_in,
    const ggml_tensor * ssm_dt,
    const ggml_tensor * ssm_a,
    ggml_tensor *       beta_out,
    ggml_tensor *       gate_out);

void ggml_cuda_op_gdn_conv_silu_l2norm_ab(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * conv_states_all,
    const ggml_tensor * state_ids,
    const ggml_tensor * qkv_mixed,
    const ggml_tensor * conv_kernel,
    ggml_tensor *       state_cell_dst,
    ggml_tensor *       v_silu_dst,
    ggml_tensor *       q_norm_dst,
    ggml_tensor *       k_norm_dst,
    int64_t             head_k_dim,
    int64_t             num_k_heads,
    float               eps_over_n,
    float               inv_sqrt_n,
    const ggml_tensor * beta_in,
    const ggml_tensor * alpha_in,
    const ggml_tensor * ssm_dt,
    const ggml_tensor * ssm_a,
    ggml_tensor *       beta_out,
    ggml_tensor *       gate_out);
