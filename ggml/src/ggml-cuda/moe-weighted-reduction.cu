#include "moe-weighted-reduction.cuh"
#include "convert.cuh"

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

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int n_exp>
__launch_bounds__(512, 2)
static __global__ void moe_reduce_shared_add_norm_q8_f32(
        const float * __restrict__ experts,
        const float * __restrict__ weights,
        const float * __restrict__ shared_out,
        const float * __restrict__ shared_gate_raw,
        const float * __restrict__ residual,
        const float * __restrict__ norm_w,
        float * __restrict__       res_dst,
        float * __restrict__       norm_dst,
        block_q8_1 * __restrict__  q8_dst,
        uint32_t * __restrict__    q8_p_qs,
        half2 * __restrict__       q8_p_ds,
        const float                eps) {
    const int tid   = threadIdx.x; // 0..511, each handles 4 floats (512 * 4 = 2048)
    const int lane8 = tid & 7;
    const int blk   = tid >> 3;    // 0..63
    __shared__ float s_sum[32];

    const float4 * __restrict__ exp4 = reinterpret_cast<const float4 *>(experts);
    const float w0 = weights[0];
    const float4 e0 = exp4[tid];
    float4 acc = make_float4(e0.x * w0, e0.y * w0, e0.z * w0, e0.w * w0);
#pragma unroll
    for (int ex = 1; ex < n_exp; ++ex) {
        const float  we = weights[ex];
        const float4 ee = exp4[ex * 512 + tid];
        acc.x = fmaf(ee.x, we, acc.x);
        acc.y = fmaf(ee.y, we, acc.y);
        acc.z = fmaf(ee.z, we, acc.z);
        acc.w = fmaf(ee.w, we, acc.w);
    }

    const float  sig = 1.0f / (1.0f + expf(-shared_gate_raw[0]));
    const float4 sh4 = reinterpret_cast<const float4 *>(shared_out)[tid];
    const float4 rs4 = reinterpret_cast<const float4 *>(residual)[tid];
    acc.x += sh4.x * sig + rs4.x;
    acc.y += sh4.y * sig + rs4.y;
    acc.z += sh4.z * sig + rs4.z;
    acc.w += sh4.w * sig + rs4.w;

    reinterpret_cast<float4 *>(res_dst)[tid] = acc;

    float sq = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
    sq = block_reduce<block_reduce_method::SUM, 512>(sq, s_sum);
    const float scale = rsqrtf(sq / 2048.0f + eps);

    const float4 nw4 = reinterpret_cast<const float4 *>(norm_w)[tid];
    const float4 r   = make_float4(
        scale * acc.x * nw4.x,
        scale * acc.y * nw4.y,
        scale * acc.z * nw4.z,
        scale * acc.w * nw4.w);
    reinterpret_cast<float4 *>(norm_dst)[tid] = r;

    if (q8_dst != nullptr) {
        float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));

        const float d  = amax / 127.0f;
        const int   q0 = (amax == 0.0f) ? 0 : (int) roundf(r.x / d);
        const int   q1 = (amax == 0.0f) ? 0 : (int) roundf(r.y / d);
        const int   q2 = (amax == 0.0f) ? 0 : (int) roundf(r.z / d);
        const int   q3 = (amax == 0.0f) ? 0 : (int) roundf(r.w / d);

        const uint32_t packed =
            ((uint32_t) (uint8_t) (int8_t) q0) |
            (((uint32_t) (uint8_t) (int8_t) q1) << 8) |
            (((uint32_t) (uint8_t) (int8_t) q2) << 16) |
            (((uint32_t) (uint8_t) (int8_t) q3) << 24);
        reinterpret_cast<uint32_t *>(q8_dst[blk].qs)[lane8] = packed;
        if (q8_p_qs != nullptr) {
            q8_p_qs[blk * 8 + lane8] = packed;
        }

        int sum_q = q0 + q1 + q2 + q3;
        sum_q += __shfl_xor_sync(0xffffffff, sum_q, 4);
        sum_q += __shfl_xor_sync(0xffffffff, sum_q, 2);
        sum_q += __shfl_xor_sync(0xffffffff, sum_q, 1);
        if (lane8 == 0) {
            const half2 dsh = make_half2(d, d * sum_q);
            q8_dst[blk].ds = dsh;
            if (q8_p_ds != nullptr) {
                q8_p_ds[blk] = dsh;
            }
        }
    }
}
#endif

void ggml_cuda_op_moe_reduce_shared_add_norm_q8(ggml_backend_cuda_context & ctx,
                                                const ggml_tensor *         experts,
                                                const ggml_tensor *         weights,
                                                const ggml_tensor *         shared_out,
                                                const ggml_tensor *         shared_gate_raw,
                                                const ggml_tensor *         residual,
                                                const ggml_tensor *         norm_w,
                                                float                       eps,
                                                ggml_tensor *               res_dst,
                                                ggml_tensor *               norm_dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_ASSERT(experts->ne[0] == 2048 && experts->ne[1] == 8);
    GGML_ASSERT(experts->ne[2] * experts->ne[3] == 1);

    block_q8_1 * dst_q8   = nullptr;
    char *       dst_q8_p = nullptr;
    if (ctx.curr_stream_no == 0 && ctx.q8_1_cache_buf != nullptr) {
        constexpr int64_t ne10_padded = GGML_PAD(2048, MATRIX_ROW_PADDING);
        constexpr size_t  q8_1_bytes  = (size_t) ne10_padded * sizeof(block_q8_1) / QK8_1;
        constexpr size_t  aligned_b   = (q8_1_bytes + 255) & ~size_t(255);
        constexpr size_t  planar_b    = 2048 + (2048 / 32) * sizeof(half2);
        constexpr size_t  aligned_p   = (planar_b + 255) & ~size_t(255);
        if (ctx.q8_1_cache_used + aligned_b + aligned_p <= ctx.q8_1_cache_cap) {
            ctx.q8_1_cache_invalidate_range(res_dst->data, ggml_nbytes(res_dst));
            ctx.q8_1_cache_invalidate_range(norm_dst->data, ggml_nbytes(norm_dst));
            char * q8_ptr = (char *) ctx.q8_1_cache_buf + ctx.q8_1_cache_used;
            dst_q8_p      = q8_ptr + aligned_b;
            ctx.q8_1_cache_used += aligned_b + aligned_p;
            ctx.q8_1_cache.push_back({
                norm_dst->data, ggml_nbytes(norm_dst), GGML_TYPE_F32,
                2048, 1, 1, 1, 2048, 2048, 2048, q8_ptr, dst_q8_p
            });
            dst_q8 = reinterpret_cast<block_q8_1 *>(q8_ptr);
        }
    }

    moe_reduce_shared_add_norm_q8_f32<8><<<1, 512, 0, ctx.stream()>>>(
        (const float *) experts->data,
        (const float *) weights->data,
        (const float *) shared_out->data,
        (const float *) shared_gate_raw->data,
        (const float *) residual->data,
        (const float *) norm_w->data,
        (float *)       res_dst->data,
        (float *)       norm_dst->data,
        dst_q8,
        (uint32_t *)    dst_q8_p,
        dst_q8_p ? (half2 *) (dst_q8_p + 2048) : nullptr,
        eps);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, experts, weights, shared_out, shared_gate_raw, residual, norm_w, eps, res_dst, norm_dst);
    GGML_ABORT("unsupported");
#endif
}
