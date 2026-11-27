#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx,
                       ggml_tensor *               dst,
                       ggml_tensor *               mul = nullptr,
                       ggml_tensor *               add = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, bool write_q8_1 = true);

void ggml_cuda_op_tower_eh_concat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * embd_w,
    const ggml_tensor * code,
    const ggml_tensor * mask,
    const ggml_tensor * extra,
    const ggml_tensor * enorm_w,
    const ggml_tensor * x,
    const ggml_tensor * hnorm_w,
    ggml_tensor * dst,
    float eps_e,
    float eps_h);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [MUL by w ->] L2_NORM -> SCALE [-> MUL by f], w broadcasts over the rows, f is one value per channel/sample
void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * l2_norm,
        const ggml_tensor * scale, const ggml_tensor * f, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_mul_silu_fused(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * rms_node,
    ggml_tensor * norm_mul_node,
    ggml_tensor * silu_node,
    ggml_tensor * out_mul_node);
