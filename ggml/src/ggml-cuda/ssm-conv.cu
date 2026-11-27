#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 2:  launch_kernel(std::integral_constant<int, 2 >{}); break;
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 2, 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

template <int d_conv, int head_dim, bool WITH_AB = false>
static __global__ void k_gdn_conv_silu_l2norm_f32(
        const float * __restrict__ conv_states_all,
        const int32_t * __restrict__ state_ids,
        const float * __restrict__ qkv_mixed,
        const float * __restrict__ conv_kernel,
        float * __restrict__       state_cell_dst,
        float * __restrict__       v_silu_dst,
        float * __restrict__       q_norm_dst,
        float * __restrict__       k_norm_dst,
        const int64_t              state_in_row_stride,
        const int64_t              conv_w_row_stride,
        const int                  num_k_heads,
        const float                eps_over_n,
        const float                inv_sqrt_n,
        const float * __restrict__ beta_in,
        const float * __restrict__ alpha_in,
        const float * __restrict__ ssm_dt,
        const float * __restrict__ ssm_a,
        float * __restrict__       beta_out,
        float * __restrict__       gate_out,
        const int                  n_ab) {
    const int tid = threadIdx.x;
    if constexpr (WITH_AB) {
        if (blockIdx.x == gridDim.x - 1) {
            if (tid < n_ab) {
                const float b  = beta_in[tid];
                beta_out[tid]  = 1.0f / (1.0f + expf(-b));
                const float ab = alpha_in[tid] + ssm_dt[tid];
                const float sp = (ab > 20.0f) ? ab : log1pf(expf(ab));
                gate_out[tid]  = sp * ssm_a[tid];
            }
            return;
        }
    }

    const int head = blockIdx.x;
    const int c    = head * head_dim + tid;

    const int32_t row_idx = state_ids[0];
    const float * st_in   = row_idx >= 0 ? conv_states_all + (int64_t) row_idx * state_in_row_stride : nullptr;

    constexpr int n_hist = d_conv - 1;
    const float h0 = st_in ? st_in[c * n_hist + 0] : 0.0f;
    const float h1 = st_in ? st_in[c * n_hist + 1] : 0.0f;
    const float h2 = st_in ? st_in[c * n_hist + 2] : 0.0f;
    const float x  = qkv_mixed[c];

    state_cell_dst[c * n_hist + 0] = h1;
    state_cell_dst[c * n_hist + 1] = h2;
    state_cell_dst[c * n_hist + 2] = x;

    const float * wc = conv_kernel + (int64_t) c * conv_w_row_stride;
    const float sum  = h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x * wc[3];
    const float y    = ggml_cuda_op_silu_single(sum);

    v_silu_dst[c] = y;

    if (head < 2 * num_k_heads) {
        __shared__ float s_warp[4];
        const float sq   = y * y;
        const float wsum = warp_reduce_sum<32>(sq);
        const int warp   = tid >> 5;
        const int lane   = tid & 31;
        if (lane == 0) {
            s_warp[warp] = wsum;
        }
        __syncthreads();

        float total_sq = 0.0f;
        if (warp == 0 && lane < 4) {
            total_sq = s_warp[lane];
        }
        total_sq = warp_reduce_sum<4>(total_sq);
        if (tid == 0) {
            s_warp[0] = rsqrtf(total_sq / (float) head_dim + eps_over_n) * inv_sqrt_n;
        }
        __syncthreads();

        const float normed = y * s_warp[0];
        if (head < num_k_heads) {
            q_norm_dst[c] = normed;
        } else {
            k_norm_dst[c - num_k_heads * head_dim] = normed;
        }
    }
}

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
        float               inv_sqrt_n) {
    GGML_ASSERT(head_k_dim == 128);
    GGML_ASSERT(conv_kernel->ne[0] == 4);
    const int64_t conv_channels = conv_kernel->ne[1];
    GGML_ASSERT(conv_channels % head_k_dim == 0);
    const int num_blocks = (int) (conv_channels / head_k_dim);

    const int64_t state_in_row_stride = (int64_t) (conv_states_all->nb[1] / sizeof(float));
    const int64_t conv_w_row_stride   = (int64_t) (conv_kernel->nb[1] / sizeof(float));

    k_gdn_conv_silu_l2norm_f32<4, 128, false><<<num_blocks, 128, 0, ctx.stream()>>>(
        (const float *) conv_states_all->data,
        (const int32_t *) state_ids->data,
        (const float *) qkv_mixed->data,
        (const float *) conv_kernel->data,
        (float *) state_cell_dst->data,
        (float *) v_silu_dst->data,
        (float *) q_norm_dst->data,
        (float *) k_norm_dst->data,
        state_in_row_stride,
        conv_w_row_stride,
        (int) num_k_heads,
        eps_over_n,
        inv_sqrt_n,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
}

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
        ggml_tensor *       gate_out) {
    GGML_ASSERT(head_k_dim == 128);
    GGML_ASSERT(conv_kernel->ne[0] == 4);
    const int64_t conv_channels = conv_kernel->ne[1];
    GGML_ASSERT(conv_channels % head_k_dim == 0);
    const int num_blocks = (int) (conv_channels / head_k_dim);
    const int n_ab = (int) ggml_nelements(beta_out);
    GGML_ASSERT(n_ab > 0 && n_ab <= 128);

    const int64_t state_in_row_stride = (int64_t) (conv_states_all->nb[1] / sizeof(float));
    const int64_t conv_w_row_stride   = (int64_t) (conv_kernel->nb[1] / sizeof(float));

    k_gdn_conv_silu_l2norm_f32<4, 128, true><<<num_blocks + 1, 128, 0, ctx.stream()>>>(
        (const float *) conv_states_all->data,
        (const int32_t *) state_ids->data,
        (const float *) qkv_mixed->data,
        (const float *) conv_kernel->data,
        (float *) state_cell_dst->data,
        (float *) v_silu_dst->data,
        (float *) q_norm_dst->data,
        (float *) k_norm_dst->data,
        state_in_row_stride,
        conv_w_row_stride,
        (int) num_k_heads,
        eps_over_n,
        inv_sqrt_n,
        (const float *) beta_in->data,
        (const float *) alpha_in->data,
        (const float *) ssm_dt->data,
        (const float *) ssm_a->data,
        (float *) beta_out->data,
        (float *) gate_out->data,
        n_ab);
}

static __global__ void k_gdn_alpha_beta_f32(
        const float * __restrict__ beta_in,
        const float * __restrict__ alpha_in,
        const float * __restrict__ ssm_dt,
        const float * __restrict__ ssm_a,
        float * __restrict__       beta_out,
        float * __restrict__       gate_out,
        const int                  n) {
    const int i = threadIdx.x;
    if (i < n) {
        const float b = beta_in[i];
        beta_out[i] = 1.0f / (1.0f + expf(-b));
        const float ab = alpha_in[i] + ssm_dt[i];
        const float sp = (ab > 20.0f) ? ab : log1pf(expf(ab));
        gate_out[i] = sp * ssm_a[i];
    }
}

void ggml_cuda_op_gdn_alpha_beta(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * beta_in,
        const ggml_tensor * alpha_in,
        const ggml_tensor * ssm_dt,
        const ggml_tensor * ssm_a,
        ggml_tensor *       beta_out,
        ggml_tensor *       gate_out) {
    const int n = (int) ggml_nelements(beta_out);
    GGML_ASSERT(n > 0 && n <= 32);
    k_gdn_alpha_beta_f32<<<1, 32, 0, ctx.stream()>>>(
        (const float *) beta_in->data,
        (const float *) alpha_in->data,
        (const float *) ssm_dt->data,
        (const float *) ssm_a->data,
        (float *) beta_out->data,
        (float *) gate_out->data,
        n);
}
