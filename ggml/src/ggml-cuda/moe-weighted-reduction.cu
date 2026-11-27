#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void moe_reduce_shared_add_f32(
        const float * __restrict__ experts,
        const float * __restrict__ weights,
        const float * __restrict__ shared_out,
        const float * __restrict__ shared_gate_raw,
        const float * __restrict__ residual,
        float * __restrict__       dst,
        const int64_t              n_embd,
        const int                  n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row = (uint64_t) token * n_expert_used;
    float sum = experts[first_row * n_embd + col] * weights[first_row];
#pragma unroll
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        sum += experts[row * n_embd + col] * weights[row];
    }

    const float g   = shared_gate_raw[token];
    const float sig = 1.0f / (1.0f + expf(-g));
    sum += shared_out[token * n_embd + col] * sig;

    if (residual != nullptr) {
        sum += residual[token * n_embd + col];
    }
    dst[token * n_embd + col] = sum;
}

void ggml_cuda_op_moe_reduce_shared_add(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor *         experts,
                                        const ggml_tensor *         weights,
                                        const ggml_tensor *         shared_out,
                                        const ggml_tensor *         shared_gate_raw,
                                        const ggml_tensor *         residual,
                                        ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(shared_out->type == GGML_TYPE_F32 && shared_gate_raw->type == GGML_TYPE_F32);
    GGML_ASSERT(residual == nullptr || residual->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    constexpr int threads       = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);

    moe_reduce_shared_add_f32<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) experts->data,
        (const float *) weights->data,
        (const float *) shared_out->data,
        (const float *) shared_gate_raw->data,
        residual ? (const float *) residual->data : nullptr,
        (float *) dst->data,
        n_embd,
        (int) n_expert_used);
    CUDA_CHECK(cudaGetLastError());
}
