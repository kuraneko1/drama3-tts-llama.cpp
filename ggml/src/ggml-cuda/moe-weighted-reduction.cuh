#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

void ggml_cuda_op_moe_reduce_shared_add(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor *         experts,
                                        const ggml_tensor *         weights,
                                        const ggml_tensor *         shared_out,
                                        const ggml_tensor *         shared_gate_raw,
                                        const ggml_tensor *         residual,
                                        ggml_tensor *               dst);
