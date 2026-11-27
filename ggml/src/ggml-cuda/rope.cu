#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        } else if constexpr (std::is_same_v<nv_bfloat16, D>) {
            const __align__(4) nv_bfloat16 v[2] = { ggml_cuda_cast<nv_bfloat16>(x0), ggml_cuda_cast<nv_bfloat16>(x1) };
            ggml_cuda_memcpy_1<sizeof(v)>(dst + idst, v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(ggml_cuda_cast<float>(x[ix + 0]), ggml_cuda_cast<float>(x[ix + 1]));
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = ggml_cuda_cast<float>(x[ix + 0]);
    const float x1 = ggml_cuda_cast<float>(x[ix + 1]);

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = ggml_cuda_cast<float>(x[ix + n_offs/2 + 0]);
    const float x1 = ggml_cuda_cast<float>(x[ix + n_offs/2 + n_dims/2]);

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

static __device__ __forceinline__ int rope_mrope_pos_idx(
        const int iw, const int channel, const int nchannels,
        const mrope_sections sections, const bool is_imrope) {
    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (iw / 2) % sect_dims;

    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
            return channel + nchannels * 1;
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
            return channel + nchannels * 2;
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
            return channel;
        } else {
            return channel + nchannels * 3;
        }
    } else {
        if (sector < sections.v[0]) {
            return channel;
        } else if (sector < sec_w) {
            return channel + nchannels * 1;
        } else if (sector < sec_w + sections.v[2]) {
            return channel + nchannels * 2;
        } else {
            return channel + nchannels * 3;
        }
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_multi(const T *            x,
                                  D *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const int64_t *      row_indices,
                                  const int            set_rows_stride,
                                  const bool           inplace) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = ggml_cuda_cast<D>(x[ix + i0/2 + 0]);
        dst[idst + i0/2 + 1] = ggml_cuda_cast<D>(x[ix + i0/2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[rope_mrope_pos_idx(iw, i2, ne02, sections, is_imrope)] * powf(theta_scale, iw / 2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = ggml_cuda_cast<float>(x[ix + n_offs/2 + 0]);
    const float x1 = ggml_cuda_cast<float>(x[ix + n_offs/2 + n_dims/2]);

    dst[idst + n_offs/2 + 0]        = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
    dst[idst + n_offs/2 + n_dims/2] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = ggml_cuda_cast<float>(x[ix + 0]);
    const float x1 = ggml_cuda_cast<float>(x[ix + n_dims]);

    dst[idst + 0]      = ggml_cuda_cast<T>(x0*cos_theta - x1*sin_theta);
    dst[idst + n_dims] = ggml_cuda_cast<T>(x0*sin_theta + x1*cos_theta);
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_multi_cuda(const T *            x,
                            D *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const int64_t *      row_indices,
                            const int            set_rows_stride,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(dst), corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_BF16) {
            rope_neox_cuda<forward, nv_bfloat16, nv_bfloat16>((const nv_bfloat16 *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01, s02,
                                                              s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                              ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                              set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, nv_bfloat16, half>((const nv_bfloat16 *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                       s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                       ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                       set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_multi_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                   s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                   ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                   row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                  row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                 row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_BF16) {
            rope_multi_cuda<forward, nv_bfloat16, nv_bfloat16>((const nv_bfloat16 *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01, s02,
                                                               s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                               ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                               row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, nv_bfloat16, half>((const nv_bfloat16 *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                        s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                        ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                        row_indices, set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            rope_vision_cuda<forward>((const nv_bfloat16 *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_BF16) {
            rope_norm_cuda<forward, nv_bfloat16, nv_bfloat16>((const nv_bfloat16 *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01, s02,
                                                              s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                              ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                              set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_BF16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, nv_bfloat16, half>((const nv_bfloat16 *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                       s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                       ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                       set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
template <int block_size, bool has_ff, typename D, bool supports_mrope>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const mrope_sections sections,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const bool is_mrope, const bool is_imrope) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc = x + sample*s03 + channel*s02 + row*s01;

            const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
            const float * mulc = mul + mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

            float tmp = 0.0f;

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            const float scale = rsqrtf(tmp/ncols + eps);

            int64_t idst = sample*s3 + channel*s2 + row*s1;
            if (set_rows_stride != 0) {
                idst = row*s1 + row_indices[channel]*set_rows_stride;
            }
            D * dstc = dst + idst;

            for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
                int ix0;
                int ix1;
                if ((is_neox || (supports_mrope && is_mrope)) && i0 < n_dims) {
                    ix0 = i0/2;
                    ix1 = i0/2 + n_dims/2;
                } else {
                    ix0 = i0 + 0;
                    ix1 = i0 + 1;
                }

                const float x0 = scale * xc[ix0] * mulc[fastmodulo(ix0, mul_ncols_packed)];
                const float x1 = scale * xc[ix1] * mulc[fastmodulo(ix1, mul_ncols_packed)];

                if (i0 >= n_dims) {
                    dstc[ix0] = ggml_cuda_cast<D>(x0);
                    dstc[ix1] = ggml_cuda_cast<D>(x1);
                    continue;
                }

                const int pos_idx = supports_mrope && is_mrope ? rope_mrope_pos_idx(i0, channel, nchannels, sections, is_imrope) : channel;

                const float theta_base  = pos[pos_idx]*powf(theta_scale, i0/2.0f);
                const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

                float cos_theta;
                float sin_theta;
                rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

                dstc[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
                dstc[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const mrope_sections sections,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const bool is_mrope, const bool is_imrope, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    auto launch_block = [&](auto bs_tag, auto mrope_tag) {
        constexpr int block_size = decltype(bs_tag)::value;
        constexpr bool supports_mrope = decltype(mrope_tag)::value;
        const dim3 block_dims(block_size, 1, 1);
        const size_t shmem = block_size > WARP_SIZE ? 32 * sizeof(float) : 0;
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, shmem, stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<block_size, false, D, supports_mrope>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, sections, row_indices, set_rows_stride, is_neox, is_mrope, is_imrope);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<block_size, true, D, supports_mrope>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, sections, row_indices, set_rows_stride, is_neox, is_mrope, is_imrope);
        }
    };

    // Keep the wider launch and compile out multi-axis indexing for single-token NEOX.
    if (is_neox && !is_mrope && nchannels == 1 && nsamples == 1 && ncols <= 8 * WARP_SIZE) {
        launch_block(std::integral_constant<int, 256>{}, std::false_type{});
    } else if (ncols <= 2 * WARP_SIZE) {
        launch_block(std::integral_constant<int, WARP_SIZE>{}, std::true_type{});
    } else if (ncols <= 4 * WARP_SIZE) {
        launch_block(std::integral_constant<int, 2 * WARP_SIZE>{}, std::true_type{});
    } else if (ncols <= 8 * WARP_SIZE) {
        if (nchannels == 1 && nsamples == 1) {
            launch_block(std::integral_constant<int, 256>{}, std::true_type{});
        } else {
            launch_block(std::integral_constant<int, 2 * WARP_SIZE>{}, std::true_type{});
        }
    } else if (ncols < 1024) {
        launch_block(std::integral_constant<int, 256>{}, std::true_type{});
    } else {
        launch_block(std::integral_constant<int, 1024>{}, std::true_type{});
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    mrope_sections sections;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int) * 4);

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(rope), corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, sections, row_indices, set_rows_stride, is_neox, is_mrope, is_imrope, stream);
    } else if (dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, sections, row_indices, set_rows_stride, is_neox, is_mrope, is_imrope, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}

static __global__ void k_qkv_prep_d256_f32_f16(
        const float * __restrict__   q_in,
        const float * __restrict__   q_norm_w,
        float * __restrict__         q_out,
        const float * __restrict__   k_in,
        const float * __restrict__   k_norm_w,
        half * __restrict__          k_cache,
        const float * __restrict__   v_in,
        half * __restrict__          v_cache,
        const int64_t                q_in_s1,
        const int64_t                q_out_s1,
        const int64_t                k_in_s1,
        const int64_t                k_cache_s1,
        const int64_t                v_cache_s1,
        const int                    n_head_q,
        const int                    n_head_kv,
        const float                  eps_q,
        const float                  eps_k,
        const int                    n_dims,
        const int32_t * __restrict__ pos,
        const float                  freq_scale,
        const float                  ext_factor,
        const float                  attn_factor,
        const rope_corr_dims         corr_dims,
        const float                  theta_scale,
        const mrope_sections         sections,
        const bool                   is_neox,
        const bool                   is_mrope,
        const bool                   is_imrope,
        const void * __restrict__    k_row_indices,
        const void * __restrict__    v_row_indices,
        const bool                   row_idx_is_i32) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x; // 0..127 (each thread processes 2 of D=256)

    if (bid >= n_head_q + n_head_kv) {
        const int vh = bid - n_head_q - n_head_kv;
        const int64_t row_v = row_idx_is_i32
            ? (int64_t) ((const int32_t *) v_row_indices)[0]
            :           ((const int64_t *) v_row_indices)[0];
        const float2 v2 = reinterpret_cast<const float2 *>(v_in + vh * 256)[tid];
        reinterpret_cast<half2 *>(v_cache + row_v * v_cache_s1 + vh * 256)[tid] = __float22half2_rn(v2);
        return;
    }

    const bool is_q = (bid < n_head_q);
    const int  h    = is_q ? bid : (bid - n_head_q);
    const float * __restrict__ xc   = is_q ? (q_in + (int64_t) h * q_in_s1) : (k_in + (int64_t) h * k_in_s1);
    const float * __restrict__ norm = is_q ? q_norm_w : k_norm_w;
    const float                eps  = is_q ? eps_q : eps_k;

    const float2 x_pair = reinterpret_cast<const float2 *>(xc)[tid];
    float sq = x_pair.x * x_pair.x + x_pair.y * x_pair.y;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        sq += __shfl_xor_sync(0xffffffff, sq, m);
    }
    __shared__ float s_w[4];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        s_w[warp] = sq;
    }
    __syncthreads();

    float sum_sq = 0.0f;
    if (warp == 0 && lane < 4) {
        sum_sq = s_w[lane];
    }
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, 2);
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, 1);
    if (tid == 0) {
        s_w[0] = rsqrtf(sum_sq / 256.0f + eps);
    }
    __syncthreads();
    const float scale = s_w[0];

    const int i0 = 2 * tid;
    int ix0, ix1;
    if ((is_neox || is_mrope) && i0 < n_dims) {
        ix0 = i0 / 2;
        ix1 = i0 / 2 + n_dims / 2;
    } else {
        ix0 = i0 + 0;
        ix1 = i0 + 1;
    }

    const float x0 = scale * xc[ix0] * norm[ix0];
    const float x1 = scale * xc[ix1] * norm[ix1];

    float o0 = x0;
    float o1 = x1;
    if (i0 < n_dims) {
        const int pos_idx = is_mrope ? rope_mrope_pos_idx(i0, 0, 1, sections, is_imrope) : 0;
        const float theta_base = pos[pos_idx] * powf(theta_scale, i0 / 2.0f);
        float cos_theta, sin_theta;
        rope_yarn<true>(theta_base, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
        o0 = x0 * cos_theta - x1 * sin_theta;
        o1 = x0 * sin_theta + x1 * cos_theta;
    }

    if (is_q) {
        float * __restrict__ dst_q = q_out + (int64_t) h * q_out_s1;
        dst_q[ix0] = o0;
        dst_q[ix1] = o1;
    } else {
        const int64_t row_k = row_idx_is_i32
            ? (int64_t) ((const int32_t *) k_row_indices)[0]
            :           ((const int64_t *) k_row_indices)[0];
        half * __restrict__ dst_k = k_cache + row_k * k_cache_s1 + h * 256;
        dst_k[ix0] = __float2half(o0);
        dst_k[ix1] = __float2half(o1);
    }
}

void ggml_cuda_op_qkv_prep_d256_fused(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * q_rms_norm, ggml_tensor * q_mul, ggml_tensor * q_rope,
        ggml_tensor * k_rms_norm, ggml_tensor * k_mul, ggml_tensor * k_rope, ggml_tensor * k_set_rows,
        ggml_tensor * v_set_rows) {
    const ggml_tensor * x_q = q_rms_norm->src[0];
    const ggml_tensor * w_q = (q_mul->src[0] == q_rms_norm) ? q_mul->src[1] : q_mul->src[0];
    const ggml_tensor * x_k = k_rms_norm->src[0];
    const ggml_tensor * w_k = (k_mul->src[0] == k_rms_norm) ? k_mul->src[1] : k_mul->src[0];
    const ggml_tensor * x_v = v_set_rows->src[0];

    float eps_q = 0.0f, eps_k = 0.0f;
    memcpy(&eps_q, q_rms_norm->op_params, sizeof(float));
    memcpy(&eps_k, k_rms_norm->op_params, sizeof(float));

    const int n_dims     = ((const int32_t *) q_rope->op_params)[1];
    const int mode       = ((const int32_t *) q_rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) q_rope->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    mrope_sections sections;
    memcpy(&freq_base,   (const int32_t *) q_rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) q_rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) q_rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) q_rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) q_rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) q_rope->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (const int32_t *) q_rope->op_params + 11, sizeof(int) * 4);

    const bool is_neox   = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool is_mrope  = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(q_rope), corr_dims.v);

    const int n_head_q  = (int) x_q->ne[1];
    const int n_head_kv = (int) x_k->ne[1];
    const int64_t q_in_s1    = x_q->nb[1] / sizeof(float);
    const int64_t q_out_s1   = q_rope->nb[1] / sizeof(float);
    const int64_t k_in_s1    = x_k->nb[1] / sizeof(float);
    const int64_t k_cache_s1 = k_set_rows->nb[1] / sizeof(half);
    const int64_t v_cache_s1 = v_set_rows->nb[1] / sizeof(half);
    const bool row_idx_is_i32 = (k_set_rows->src[1]->type == GGML_TYPE_I32);

    const int total_blocks = n_head_q + 2 * n_head_kv;
    k_qkv_prep_d256_f32_f16<<<total_blocks, 128, 0, ctx.stream()>>>(
        (const float *) x_q->data,
        (const float *) w_q->data,
        (float *) q_rope->data,
        (const float *) x_k->data,
        (const float *) w_k->data,
        (half *) k_set_rows->data,
        (const float *) x_v->data,
        (half *) v_set_rows->data,
        q_in_s1,
        q_out_s1,
        k_in_s1,
        k_cache_s1,
        v_cache_s1,
        n_head_q,
        n_head_kv,
        eps_q,
        eps_k,
        n_dims,
        (const int32_t *) q_rope->src[1]->data,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_dims,
        theta_scale,
        sections,
        is_neox,
        is_mrope,
        is_imrope,
        k_set_rows->src[1]->data,
        v_set_rows->src[1]->data,
        row_idx_is_i32);
    (void) k_rope;
}
