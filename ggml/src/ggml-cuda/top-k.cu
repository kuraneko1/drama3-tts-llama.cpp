#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// one state per cache line, so the gather atomics of different rows do not compete
struct alignas(128) top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int block_cols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const int64_t col_begin = (int64_t) row_block * block_cols;
    const int64_t col_end   = min(col_begin + block_cols, (int64_t) ncols);
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = col_begin + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int block_cols,
        int k,
        int blocks_per_row) {
    constexpr int NBINS     = 1 << RADIX_BITS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    __shared__ int warp_ties[nwarps];

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];
    const uint32_t threshold = state->prefix;
    const int rank = state->rank;

    // each warp reads a contiguous part of the contiguous range of the block
    const int     warp_cols  = block_cols / nwarps;
    const int64_t warp_begin = (int64_t) row_block * block_cols + warp * warp_cols;
    const int64_t warp_end   = min(warp_begin + warp_cols, (int64_t) ncols);

    int nties = 0;
    for (int64_t col = warp_begin + lane; col < warp_end; col += warp_size) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > threshold) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == threshold) {
            nties++;
        }
    }

    // the ties fill the last rank slots in column order, the lowest columns win.
    // the histograms of the last pass count the ties of each block in the bin of the threshold.
    const int * block_ties = block_histograms + (size_t) row * blocks_per_row * NBINS + (threshold & (NBINS - 1));
    int pos = 0;
    for (int b = lane; b < row_block; b += warp_size) {
        pos += block_ties[b * NBINS];
    }
    pos = warp_reduce_sum<warp_size>(pos);
    if (pos >= rank || block_ties[row_block * NBINS] == 0) {
        return; // the same result in all threads of the block
    }

    nties = warp_reduce_sum<warp_size>(nties);
    if (lane == 0) {
        warp_ties[warp] = nties;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        pos += warp_ties[w];
    }

    // pos is the same in all lanes of the warp, so the lanes leave the loop together
    const int pos_end = min(pos + nties, rank);
    for (int64_t tile = warp_begin; tile < warp_end && pos < pos_end; tile += warp_size) {
        const int64_t col    = tile + lane;
        const bool    is_tie = col < warp_end && top_k_float_to_ordered(row_src[col]) == threshold;

        const ggml_cuda_thread_mask_t ties = ggml_cuda_ballot(is_tie);
        const int tie_pos = pos + ggml_cuda_popc(ties & (((ggml_cuda_thread_mask_t) 1 << lane) - 1));
        if (is_tie && tie_pos < pos_end) {
            row_dst[k - rank + tie_pos] = col;
        }
        pos += ggml_cuda_popc(ties);
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    // each block of a row reads a contiguous range of whole tiles
    const int block_cols = (int) (((int64_t) ncols + blocks_per_row * BLOCK_SIZE - 1) / (blocks_per_row * BLOCK_SIZE) * BLOCK_SIZE);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, block_cols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_gather<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, histograms, states, ncols, block_cols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}

__device__ __forceinline__ bool topk_cand_better(float va, int ia, float vb, int ib) {
    return (ib < 0) || (ia >= 0 && (va > vb || (va == vb && ia < ib)));
}

template <int BLOCK_SIZE = 256, int MAX_CHUNKS = 16, int MAX_K = 64>
static __global__ void k_fused_top_k_top_p_sample_f32(
        const float * __restrict__ logits,
        const float * __restrict__ rand_ptr,
        int32_t * __restrict__     dst,
        const int                  n_vocab,
        const int                  k,
        const float                top_p) {
    constexpr int N_WARPS = BLOCK_SIZE / 32;

    __shared__ float s_tmax[BLOCK_SIZE];
    __shared__ float s_cval[BLOCK_SIZE];
    __shared__ int   s_cidx[BLOCK_SIZE];
    __shared__ int   s_warp_cnt[N_WARPS];
    __shared__ int   s_warp_off[N_WARPS];
    __shared__ float s_warp_val[N_WARPS];
    __shared__ int   s_warp_idx[N_WARPS];
    __shared__ int   s_win_idx;
    __shared__ float s_work[MAX_K];
    __shared__ float s_keep[MAX_K];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float r_val[MAX_CHUNKS];
    float my_max = -INFINITY;
#pragma unroll
    for (int m = 0; m < MAX_CHUNKS; ++m) {
        const int idx = tid + m * BLOCK_SIZE;
        const float v = (idx < n_vocab) ? logits[idx] : -INFINITY;
        r_val[m] = v;
        if (idx < n_vocab && v > my_max) {
            my_max = v;
        }
    }

    // Step 1: exact lower bound t_lb = k-th largest thread maximum via descending bitonic sort
    s_tmax[tid] = my_max;
    for (int size = 2; size <= BLOCK_SIZE; size <<= 1) {
        const bool inter_warp = (size >= 64);
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (inter_warp && stride >= 16) {
                __syncthreads();
            } else {
                __syncwarp();
            }
            const int ixj = tid ^ stride;
            if (ixj > tid) {
                const float a = s_tmax[tid];
                const float b = s_tmax[ixj];
                const bool want_desc = (tid & size) == 0;
                if (want_desc ? (a < b) : (a > b)) {
                    s_tmax[tid] = b;
                    s_tmax[ixj] = a;
                }
            }
        }
    }
    __syncthreads();
    const float t_lb = s_tmax[k - 1];

    // Step 2: lock-free candidate count and prefix sum for r_val[m] >= t_lb
    int my_cnt = 0;
#pragma unroll
    for (int m = 0; m < MAX_CHUNKS; ++m) {
        const int idx = tid + m * BLOCK_SIZE;
        if (idx < n_vocab && r_val[m] >= t_lb) {
            my_cnt++;
        }
    }

    int warp_sum = my_cnt;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int n = __shfl_up_sync(0xffffffff, warp_sum, d);
        if (lane >= d) {
            warp_sum += n;
        }
    }
    if (lane == 31) {
        s_warp_cnt[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        int wc = (lane < N_WARPS) ? s_warp_cnt[lane] : 0;
#pragma unroll
        for (int d = 1; d < N_WARPS; d <<= 1) {
            const int n = __shfl_up_sync(0xffffffff, wc, d);
            if (lane >= d) {
                wc += n;
            }
        }
        if (lane < N_WARPS) {
            s_warp_off[lane] = wc;
        }
    }
    __syncthreads();

    const int total_cand = s_warp_off[N_WARPS - 1];
    const int warp_base  = (warp == 0) ? 0 : s_warp_off[warp - 1];
    int       my_off     = warp_base + (warp_sum - my_cnt);

    if (total_cand <= BLOCK_SIZE) {
        // Step 3A (fast path): gather <= 256 candidates and descending bitonic sort
        const int need = max(total_cand, k);
        const int P    = (need <= 1) ? 1 : (1 << (32 - __clz((unsigned) (need - 1))));
        if (tid < P) {
            s_cval[tid] = -INFINITY;
            s_cidx[tid] = -1;
        }
        __syncthreads();

#pragma unroll
        for (int m = 0; m < MAX_CHUNKS; ++m) {
            const int idx = tid + m * BLOCK_SIZE;
            if (idx < n_vocab && r_val[m] >= t_lb) {
                s_cval[my_off] = r_val[m];
                s_cidx[my_off] = idx;
                my_off++;
            }
        }

        for (int size = 2; size <= P; size <<= 1) {
            const bool inter_warp = (size >= 64);
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                if (inter_warp && stride >= 16) {
                    __syncthreads();
                } else {
                    __syncwarp();
                }
                if (tid < P) {
                    const int ixj = tid ^ stride;
                    if (ixj > tid) {
                        const float va = s_cval[tid];
                        const int   ia = s_cidx[tid];
                        const float vb = s_cval[ixj];
                        const int   ib = s_cidx[ixj];
                        const bool want_desc = (tid & size) == 0;
                        const bool swap = want_desc ? !topk_cand_better(va, ia, vb, ib)
                                                    :  topk_cand_better(va, ia, vb, ib);
                        if (swap) {
                            s_cval[tid] = vb;
                            s_cidx[tid] = ib;
                            s_cval[ixj] = va;
                            s_cidx[ixj] = ia;
                        }
                    }
                }
            }
        }
        __syncthreads();
    } else {
        // Step 3B (rare fallback when > 256 candidates tied/clustered >= t_lb): exact register argmax
        uint32_t picked_mask = 0;
        float    my_best     = -INFINITY;
        int      my_best_idx = -1;
#pragma unroll
        for (int m = 0; m < MAX_CHUNKS; ++m) {
            const int idx = tid + m * BLOCK_SIZE;
            if (idx < n_vocab) {
                const float v = r_val[m];
                if (my_best_idx < 0 || v > my_best || (v == my_best && idx < my_best_idx)) {
                    my_best     = v;
                    my_best_idx = idx;
                }
            }
        }

        for (int step = 0; step < k; ++step) {
            float w_val = my_best;
            int   w_idx = my_best_idx;
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float o_val = __shfl_xor_sync(0xffffffff, w_val, mask);
                const int   o_idx = __shfl_xor_sync(0xffffffff, w_idx, mask);
                if (o_idx >= 0 && (w_idx < 0 || o_val > w_val || (o_val == w_val && o_idx < w_idx))) {
                    w_val = o_val;
                    w_idx = o_idx;
                }
            }
            if (lane == 0) {
                s_warp_val[warp] = w_val;
                s_warp_idx[warp] = w_idx;
            }
            __syncthreads();

            if (warp == 0) {
                float b_val = (lane < N_WARPS) ? s_warp_val[lane] : -INFINITY;
                int   b_idx = (lane < N_WARPS) ? s_warp_idx[lane] : -1;
#pragma unroll
                for (int mask = N_WARPS / 2; mask > 0; mask >>= 1) {
                    const float o_val = __shfl_xor_sync(0xffffffff, b_val, mask);
                    const int   o_idx = __shfl_xor_sync(0xffffffff, b_idx, mask);
                    if (o_idx >= 0 && (b_idx < 0 || o_val > b_val || (o_val == b_val && o_idx < b_idx))) {
                        b_val = o_val;
                        b_idx = o_idx;
                    }
                }
                if (lane == 0) {
                    s_win_idx    = b_idx;
                    s_cval[step] = b_val;
                    s_cidx[step] = b_idx;
                }
            }
            __syncthreads();

            const int win_idx = s_win_idx;
            if (win_idx >= 0 && (win_idx % BLOCK_SIZE) == tid) {
                picked_mask |= (1u << (win_idx / BLOCK_SIZE));
                my_best     = -INFINITY;
                my_best_idx = -1;
#pragma unroll
                for (int m = 0; m < MAX_CHUNKS; ++m) {
                    if ((picked_mask & (1u << m)) == 0) {
                        const int idx = tid + m * BLOCK_SIZE;
                        if (idx < n_vocab) {
                            const float v = r_val[m];
                            if (my_best_idx < 0 || v > my_best || (v == my_best && idx < my_best_idx)) {
                                my_best     = v;
                                my_best_idx = idx;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (k <= 32) {
        if (warp == 0) {
            const bool active = lane < k;
            const float val   = active ? s_cval[lane] : -INFINITY;
            const float max_v1 = s_cval[0];
            const float e1     = active ? expf(val - max_v1) : 0.0f;

            float sum_e1 = e1;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                sum_e1 += __shfl_xor_sync(0xffffffff, sum_e1, m);
            }
            const float p1 = e1 / sum_e1;

            float cdf1 = p1;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const float n = __shfl_up_sync(0xffffffff, cdf1, d);
                if (lane >= d) {
                    cdf1 += n;
                }
            }

            int keep = (active && (top_p - cdf1 > 0.0f)) ? 1 : 0;
            int sum_keep = keep;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                sum_keep += __shfl_xor_sync(0xffffffff, sum_keep, m);
            }
            const int force_idx = min(max(sum_keep, 0), k - 1);
            if (lane == force_idx) {
                keep = 1;
            }

            float max_v2 = keep ? val : -INFINITY;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                max_v2 = fmaxf(max_v2, __shfl_xor_sync(0xffffffff, max_v2, m));
            }

            const float e2 = keep ? expf(val - max_v2) : 0.0f;
            float sum_e2 = e2;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                sum_e2 += __shfl_xor_sync(0xffffffff, sum_e2, m);
            }
            const float p2 = e2 / sum_e2;

            float cdf2 = p2;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const float n = __shfl_up_sync(0xffffffff, cdf2, d);
                if (lane >= d) {
                    cdf2 += n;
                }
            }

            const float r = *rand_ptr;
            int cross = (active && (cdf2 - r > 0.0f)) ? 1 : 0;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                cross += __shfl_xor_sync(0xffffffff, cross, m);
            }
            const int draw_pos = min(max(k - cross, 0), k - 1);
            if (lane == 0) {
                *dst = s_cidx[draw_pos];
            }
        }
    } else if (tid == 0) {
        const float max_v1 = s_cval[0];
        float sum_e1 = 0.0f;
        for (int j = 0; j < k; ++j) {
            const float e = expf(s_cval[j] - max_v1);
            s_work[j] = e;
            sum_e1 += e;
        }
        const float inv_s1 = 1.0f / sum_e1;
        float cdf1     = 0.0f;
        float sum_keep = 0.0f;
        for (int j = 0; j < k; ++j) {
            cdf1 += s_work[j] * inv_s1;
            const float km = (top_p - cdf1 > 0.0f) ? 1.0f : 0.0f;
            s_keep[j]  = km;
            sum_keep  += km;
        }
        const int force_idx = (int) fminf(fmaxf(sum_keep, 0.0f), (float) (k - 1));
        s_keep[force_idx] = 1.0f;

        float max_v2 = -INFINITY;
        for (int j = 0; j < k; ++j) {
            if (s_keep[j] > 0.0f && s_cval[j] > max_v2) {
                max_v2 = s_cval[j];
            }
        }
        float sum_e2 = 0.0f;
        for (int j = 0; j < k; ++j) {
            const float e = (s_keep[j] > 0.0f) ? expf(s_cval[j] - max_v2) : 0.0f;
            s_work[j] = e;
            sum_e2 += e;
        }
        const float inv_s2 = 1.0f / sum_e2;
        const float r = *rand_ptr;
        float cdf2      = 0.0f;
        float sum_cross = 0.0f;
        for (int j = 0; j < k; ++j) {
            cdf2 += s_work[j] * inv_s2;
            sum_cross += (cdf2 - r > 0.0f) ? 1.0f : 0.0f;
        }
        const int draw_pos = (int) fminf(fmaxf((float) k - sum_cross, 0.0f), (float) (k - 1));
        *dst = s_cidx[draw_pos];
    }
}

void ggml_cuda_op_top_k_top_p_sample_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         logits,
        const ggml_tensor *         rand_val,
        ggml_tensor *               dst,
        const int                   k,
        const float                 top_p) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32 && ggml_is_contiguous(logits));
    GGML_ASSERT(rand_val->type == GGML_TYPE_F32 && ggml_is_contiguous(rand_val));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ggml_nelements(dst) == 1);
    const int n_vocab = (int) logits->ne[0];
    GGML_ASSERT(n_vocab > k && n_vocab <= 8192 && k >= 1 && k <= 64);

    if (n_vocab <= 4096) {
        k_fused_top_k_top_p_sample_f32<256, 16, 64><<<1, 256, 0, ctx.stream()>>>(
            (const float *) logits->data,
            (const float *) rand_val->data,
            (int32_t *)     dst->data,
            n_vocab,
            k,
            top_p);
    } else {
        k_fused_top_k_top_p_sample_f32<256, 32, 64><<<1, 256, 0, ctx.stream()>>>(
            (const float *) logits->data,
            (const float *) rand_val->data,
            (int32_t *)     dst->data,
            n_vocab,
            k,
            top_p);
    }
}
