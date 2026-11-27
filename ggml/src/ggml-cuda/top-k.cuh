#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_top_k_top_p_sample_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor *         logits,
    const ggml_tensor *         rand_val,
    ggml_tensor *               dst,
    int                         k,
    float                       top_p);
