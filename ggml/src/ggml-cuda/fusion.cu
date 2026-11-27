#include "fusion.cuh"
#include "graph.cuh"
#include "binbcast.cuh"
#include "dsv4-hc.cuh"
#include "fp8.cuh"
#include "gated_delta_net.cuh"
#include "inkling2.cuh"
#include "mmq.cuh"
#include "mmq-few.cuh"
#include "mmvf.cuh"
#include "mmvq.cuh"
#include "moe-weighted-reduction.cuh"
#include "norm.cuh"
#include "rope.cuh"
#include "snake.cuh"
#include "softcap.cuh"
#include "ssm-conv.cuh"
#include "ssm-scan.cuh"
#include "top-k.cuh"
#include "topk-moe.cuh"
#include "unary.cuh"

#include <cstring>
#include <initializer_list>

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor, const ggml_tensor * gate = nullptr) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
         src0->type == GGML_TYPE_F8_E4M3) &&
        (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16) && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int64_t ne11  = is_mul_mat_id ? src1->ne[2] : src1->ne[1];
    // a MUL_MAT_ID with an F32 src1 keeps the MUL_MAT rule here, with a BF16 src1 it takes MMVF where
    // ggml_cuda_mul_mat_id does, since MMF can not read a BF16 src1
    const bool mul_mat_id_rule = is_mul_mat_id && src1->type == GGML_TYPE_BF16;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0, src1->type, cc, warp_size, ne11, mul_mat_id_rule);

    if (gate && !ggml_cuda_should_use_mmvf(gate->src[0], src1->type, cc, warp_size, ne11, mul_mat_id_rule)) {
        return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_mul_mat_vec_q = ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1]) && !bad_padding_clear &&
                             (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16) && dst->type == GGML_TYPE_F32;

    // fusion is not universally faster on Pascal
    if (cc < GGML_CUDA_CC_VOLTA) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    // do not fuse the matmuls that few-row MMQ takes in ggml_cuda_mul_mat_id
    if (tensor->op == GGML_OP_MUL_MAT_ID && ggml_cuda_should_use_mmq_few(src0, src1, tensor->src[2], cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// The weight scale that scale_node (MUL) applies to the output of mm_node (MUL_MAT), or nullptr.
// Shared by ggml_cuda_try_fuse and graph_optimize so that both accept the same patterns.
static const ggml_tensor * ggml_cuda_get_mul_mat_scale(const ggml_tensor * scale_node, const ggml_tensor * mm_node) {
    const bool scale_lhs_mm = scale_node->src[0] == mm_node;
    const bool scale_rhs_mm = scale_node->src[1] == mm_node;
    if (!scale_lhs_mm && !scale_rhs_mm) {
        return nullptr;
    }

    const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
    if ((mm_node->src[0]->type != GGML_TYPE_NVFP4 && mm_node->src[0]->type != GGML_TYPE_F8_E4M3) || scale_node->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
            !ggml_are_same_shape(scale_node, mm_node)) {
        return nullptr;
    }

    return scale;
}

// The same for MUL_MAT_ID with one scale per expert: RESHAPE, REPEAT and GET_ROWS (by the ids) of the scales, then the MUL.
static const ggml_tensor * ggml_cuda_get_mul_mat_id_scale(const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
        const ggml_tensor * scale_node, const ggml_tensor * mm_node) {
    if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
        return nullptr;
    }
    if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
            (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
        return nullptr;
    }

    const ggml_tensor * scale = reshape->src[0];
    if ((mm_node->src[0]->type != GGML_TYPE_NVFP4 && mm_node->src[0]->type != GGML_TYPE_F8_E4M3) || scale_node->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
            !ggml_are_same_shape(scale_node, mm_node)) {
        return nullptr;
    }

    return scale;
}

// Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (see #29986).
// up_scale/gate_scale: the MUL nodes that apply the NVFP4 weight scales (per tensor or per expert) between the MUL_MATs and the GLU.
static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device,
        const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];

    if ((up->op != GGML_OP_MUL_MAT && up->op != GGML_OP_MUL_MAT_ID) || !ggml_cuda_mmq_fusion_supported(x_up->type) ||
            (y->type != GGML_TYPE_F32 && y->type != GGML_TYPE_BF16)) {
        return false;
    }

    // The fused kernel is only compiled for weights whose rows fill whole tiles (no fallback).
    if (ggml_cuda_mmq_needs_fallback(x_up->ne[1])) {
        return false;
    }

    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu, nullptr, nullptr, up_scale, gate_scale)) {
        return false;
    }

    // The fused kernel runs both MUL_MATs at one src1 precision, see ggml_cuda_mul_mat_q.
    if (ggml_prec_get_src(up, 1) != ggml_prec_get_src(gate, 1)) {
        return false;
    }

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu,
        const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }

    // The padding of a compute-buffer weight that is a view cannot be cleared, see ggml_cuda_mul_mat.
    const auto bad_padding_clear = [](const ggml_tensor * x) {
        return ggml_backend_buffer_get_usage(x->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
    };
    const int device = ggml_cuda_get_device();
    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device, up_scale, gate_scale) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
        return false;
    }

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];
    const int           cc   = ggml_cuda_info().devices[device].cc;

    // For MUL_MAT_ID: the MMVQ/MMQ choice of ggml_cuda_mul_mat_id, with the nominal tiles of all experts (the per-expert token counts are only known on the GPU).
    const bool    is_id     = up->op == GGML_OP_MUL_MAT_ID;
    const int64_t ncols     = is_id ? y->ne[2]    : y->ne[1];
    const int64_t nchannels = is_id ? x_up->ne[2] : y->ne[2]*y->ne[3];
    const bool    use_mmvq  = is_id ? ncols <= get_mmvq_mmid_max_batch(x_up->type, cc) : ggml_cuda_should_use_mmvq(x_up->type, cc, ncols);
    // do not fuse the matmuls that few-row MMQ takes in ggml_cuda_mul_mat; for MUL_MAT_ID the fused MMQ keeps the expert gate and up
    if (!is_id && ggml_cuda_should_use_mmq_few(x_up, y, nullptr, cc)) {
        return false;
    }
    if (use_mmvq || !ggml_cuda_should_use_mmq(x_up->type, cc, ncols, /*n_experts =*/ is_id ? x_up->ne[2] : 0)) {
        return false;
    }

    // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, ggml_prec_get_src(up, 1), x_up->ne[1], ncols, nchannels, device);
}

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_BF16) {
        return false;
    }

    // BF16 rope fuses into an F16 or BF16 cache, other ropes not into a BF16 cache
    if (rope->type == GGML_TYPE_BF16 ? set_rows->type == GGML_TYPE_F32 : set_rows->type == GGML_TYPE_BF16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    if (mode & GGML_ROPE_TYPE_MROPE) {
        int sections[4];
        memcpy(sections, (const int32_t *) rope->op_params + 11, sizeof(int) * 4);
        const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        if ((sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) || sect_dims <= 0) {
            return false;
        }
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;

    // rs_replay: the next 4 real nodes may store the k, v, g, beta inputs into the pending rows
    if (gdn->src[6] != nullptr && gdn->src[3]->ne[0] == 1) {
        int    j = node_idx + skip + 1;
        int    skip_in = skip;
        float * in_dst[4];
        int64_t in_nb2[4];
        int64_t in_nb3[4];
        int i = 0;
        for (; i < 4 && j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(n)) {
                continue;
            }
            const ggml_tensor * in = gdn->src[1 + i];
            const ggml_tensor * d  = n->src[1];
            // the cpy reads the whole input, directly or through a view of the same data
            const ggml_tensor * s0 = n->src[0];
            const bool whole_in = s0 == in || (s0 != nullptr && s0->data == in->data && ggml_are_same_shape(s0, in) &&
                                               ggml_are_same_stride(s0, in));
            if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT) || !whole_in ||
                d->type != GGML_TYPE_F32 || d->data == nullptr || !ggml_are_same_shape(d, in) ||
                d->nb[0] != sizeof(float) || d->nb[1] != d->ne[0]*sizeof(float) || d->nb[2] != d->ne[0]*d->ne[1]*sizeof(float)) {
                break;
            }
            in_dst[i] = (float *) d->data;
            in_nb2[i] = d->nb[2] / sizeof(float);
            in_nb3[i] = d->nb[3] / sizeof(float);
            skip_in   = j - node_idx;
            ++i;
        }
        if (i == 4) {
            for (int c = 0; c < 4; ++c) {
                fused_state_cpy.in_dst[c] = in_dst[c];
                fused_state_cpy.in_nb2[c] = in_nb2[c];
                fused_state_cpy.in_nb3[c] = in_nb3[c];
            }
            skip = skip_in;
        }
    }

    return skip;
}

// match GET_ROWS(ssm_states_all, s_copy) -> [views/empty] -> GATED_DELTA_NET -> [views] -> CPY

static bool ggml_cuda_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b);

struct ggml_cuda_gdn_conv_match {
    const ggml_tensor * conv_states_all = nullptr;
    const ggml_tensor * state_ids       = nullptr;
    const ggml_tensor * qkv_mixed       = nullptr;
    const ggml_tensor * conv_kernel     = nullptr;
    ggml_tensor *       state_cell_dst  = nullptr;
    ggml_tensor *       v_silu_dst      = nullptr;
    ggml_tensor *       q_norm_dst      = nullptr;
    ggml_tensor *       k_norm_dst      = nullptr;
    int64_t             head_k_dim      = 0;
    int64_t             num_k_heads     = 0;
    float               eps_over_n      = 0.0f;
    float               inv_sqrt_n      = 1.0f;
    int                 node_count      = 0;
};

static bool ggml_cuda_match_gdn_conv_silu_l2norm(
        const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m, bool check_ptrs) {
    if (i + 10 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * get_rows = cgraph->nodes[i];
    if (get_rows->op != GGML_OP_GET_ROWS || get_rows->type != GGML_TYPE_F32 ||
            (get_rows->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            get_rows->ne[1] != 1 || get_rows->ne[2] != 1 || get_rows->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * conv_states_all = get_rows->src[0];
    const ggml_tensor * state_ids       = get_rows->src[1];
    if (!conv_states_all || !state_ids ||
            conv_states_all->type != GGML_TYPE_F32 || state_ids->type != GGML_TYPE_I32 ||
            conv_states_all->nb[0] != sizeof(float) || state_ids->nb[0] != sizeof(int32_t) ||
            ggml_nelements(state_ids) != 1 || get_rows->ne[0] != conv_states_all->ne[0]) {
        return false;
    }

    auto traces_to = [](const ggml_tensor * t, const ggml_tensor * target) -> bool {
        while (t) {
            if (t == target) {
                return true;
            }
            if (ggml_cuda_is_view_or_noop(t)) {
                t = t->view_src ? t->view_src : t->src[0];
            } else {
                break;
            }
        }
        return false;
    };

    int idxs[24];
    int n_sub = 0;
    auto push_node = [&](int idx) -> bool {
        if (n_sub >= 24) return false;
        idxs[n_sub] = idx;
        n_sub++;
        return true;
    };

    if (!push_node(i)) return false;

    int k = i + 1;
    // skip view/reshape/transpose between get_rows and concat
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if (!push_node(k)) return false;
        k++;
    }
    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_CONCAT) {
        return false;
    }
    const ggml_tensor * concat = cgraph->nodes[k];
    if (ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
            concat->ne[0] != 4 || concat->ne[2] != 1 || concat->ne[3] != 1 ||
            !traces_to(concat->src[0], get_rows)) {
        return false;
    }
    const ggml_tensor * tr_qkv = concat->src[1];
    if (!tr_qkv || tr_qkv->op != GGML_OP_TRANSPOSE || !tr_qkv->src[0]) {
        return false;
    }
    const ggml_tensor * qkv_mixed = tr_qkv->src[0];
    const int64_t conv_channels = concat->ne[1];
    if (qkv_mixed->type != GGML_TYPE_F32 || !ggml_is_contiguous(qkv_mixed) ||
            qkv_mixed->ne[0] != conv_channels || qkv_mixed->ne[1] != 1 || qkv_mixed->ne[2] != 1 || qkv_mixed->ne[3] != 1 ||
            get_rows->ne[0] != 3 * conv_channels) {
        return false;
    }
    if (!push_node(k)) return false;
    k++;

    // views leading to CPY (build_conv_state_store)
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if (!push_node(k)) return false;
        k++;
    }
    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_CPY) {
        return false;
    }
    const int cpy_idx = k;
    ggml_tensor * cpy = cgraph->nodes[k];
    const ggml_tensor * last_view = cpy->src[0];
    ggml_tensor * cell = cpy->src[1];
    if (!last_view || !cell || last_view->op != GGML_OP_VIEW || last_view->view_src != concat ||
            last_view->view_offs != sizeof(float) || last_view->ne[0] != 3 || last_view->ne[1] != conv_channels ||
            cell->type != GGML_TYPE_F32 || !ggml_is_contiguous(cell) ||
            cell->ne[0] != 3 * conv_channels || cell->ne[1] != 1 || cell->ne[2] != 1 || cell->ne[3] != 1) {
        return false;
    }
    if (!push_node(k)) return false;
    k++;

    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_SSM_CONV) {
        return false;
    }
    const ggml_tensor * ssm_conv = cgraph->nodes[k];
    const ggml_tensor * conv_kernel = ssm_conv->src[1];
    if (ssm_conv->src[0] != concat || ssm_conv->type != GGML_TYPE_F32 ||
            !conv_kernel || conv_kernel->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv_kernel) ||
            conv_kernel->ne[0] != 4 || conv_kernel->ne[1] != conv_channels) {
        return false;
    }
    if (!push_node(k)) return false;
    k++;

    if (k >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(cgraph->nodes[k]) != GGML_UNARY_OP_SILU) {
        return false;
    }
    const int silu_idx = k;
    ggml_tensor * silu = cgraph->nodes[k];
    if (silu->src[0] != ssm_conv || silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu) ||
            silu->ne[0] != conv_channels || silu->ne[1] != 1) {
        return false;
    }
    if (!push_node(k)) return false;
    k++;

    // Q branch: VIEW(silu, off=0) -> RMS_NORM -> SCALE
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if (!push_node(k)) return false;
        k++;
    }
    if (k + 1 >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_RMS_NORM ||
            cgraph->nodes[k + 1]->op != GGML_OP_SCALE) {
        return false;
    }
    const ggml_tensor * q_rms   = cgraph->nodes[k];
    ggml_tensor *       q_scale = cgraph->nodes[k + 1];
    const ggml_tensor * q_view  = q_rms->src[0];
    if (!q_view || q_view->op != GGML_OP_VIEW || q_view->view_src != silu || q_view->view_offs != 0 ||
            q_view->ne[0] != 128 || q_view->ne[2] != 1 || q_view->ne[3] != 1 ||
            q_scale->src[0] != q_rms || q_rms->type != GGML_TYPE_F32 || q_scale->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(q_scale)) {
        return false;
    }
    const int64_t num_k_heads = q_view->ne[1];
    if (conv_channels < 2 * num_k_heads * 128) {
        return false;
    }
    float q_eps = 0.0f, q_sc = 0.0f, q_bias = 0.0f;
    memcpy(&q_eps, q_rms->op_params, sizeof(float));
    memcpy(&q_sc,  q_scale->op_params, sizeof(float));
    memcpy(&q_bias, (const float *) q_scale->op_params + 1, sizeof(float));
    if (q_bias != 0.0f) {
        return false;
    }
    if (!push_node(k) || !push_node(k + 1)) return false;
    const int q_scale_idx = k + 1;
    k += 2;

    // K branch: VIEW(silu, off=num_k_heads*128*4) -> RMS_NORM -> SCALE
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if (!push_node(k)) return false;
        k++;
    }
    if (k + 1 >= cgraph->n_nodes || cgraph->nodes[k]->op != GGML_OP_RMS_NORM ||
            cgraph->nodes[k + 1]->op != GGML_OP_SCALE) {
        return false;
    }
    const ggml_tensor * k_rms   = cgraph->nodes[k];
    ggml_tensor *       k_scale = cgraph->nodes[k + 1];
    const ggml_tensor * k_view  = k_rms->src[0];
    if (!k_view || k_view->op != GGML_OP_VIEW || k_view->view_src != silu ||
            k_view->view_offs != (size_t) num_k_heads * 128 * sizeof(float) ||
            k_view->ne[0] != 128 || k_view->ne[1] != num_k_heads || k_view->ne[2] != 1 || k_view->ne[3] != 1 ||
            k_scale->src[0] != k_rms || k_rms->type != GGML_TYPE_F32 || k_scale->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(k_scale)) {
        return false;
    }
    float k_eps = 0.0f, k_sc = 0.0f, k_bias = 0.0f;
    memcpy(&k_eps, k_rms->op_params, sizeof(float));
    memcpy(&k_sc,  k_scale->op_params, sizeof(float));
    memcpy(&k_bias, (const float *) k_scale->op_params + 1, sizeof(float));
    if (k_bias != 0.0f || k_eps != q_eps || k_sc != q_sc) {
        return false;
    }
    if (!push_node(k) || !push_node(k + 1)) return false;
    const int k_scale_idx = k + 1;
    k += 2;

    const ggml_tensor * internal[5] = { get_rows, concat, ssm_conv, q_rms, k_rms };
    for (const ggml_tensor * t : internal) {
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
    }
    for (int j = k; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            for (const ggml_tensor * t : internal) {
                if (n->src[s] == t) {
                    return false;
                }
            }
        }
        if (n->view_src) {
            for (const ggml_tensor * t : internal) {
                if (n->view_src == t) {
                    return false;
                }
            }
        }
    }
    for (int idx_i = 0; idx_i < n_sub; ++idx_i) {
        const int ni = idxs[idx_i];
        if (ni == cpy_idx || ni == silu_idx || ni == q_scale_idx || ni == k_scale_idx) {
            continue;
        }
        const ggml_tensor * nt = cgraph->nodes[ni];
        if ((nt->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
        for (int j = k; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * ext = cgraph->nodes[j];
            if (ext->view_src == nt) {
                return false;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (ext->src[s] == nt) {
                    return false;
                }
            }
        }
    }

    if (check_ptrs) {
        if (!conv_states_all->data || !state_ids->data || !qkv_mixed->data || !conv_kernel->data ||
                !cell->data || !silu->data || !q_scale->data || !k_scale->data) {
            return false;
        }
        // note: cell IS allowed to overlap conv_states_all (in-place per-channel update)
        const ggml_tensor * read_only_srcs[3] = { state_ids, qkv_mixed, conv_kernel };
        const ggml_tensor * out_tensors[4]    = { cell, silu, q_scale, k_scale };
        for (const ggml_tensor * out_t : out_tensors) {
            for (const ggml_tensor * in_t : read_only_srcs) {
                if (ggml_cuda_tensors_overlap(out_t, in_t)) {
                    return false;
                }
            }
        }
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                if (ggml_cuda_tensors_overlap(out_tensors[a], out_tensors[b])) {
                    return false;
                }
            }
        }
        for (int b = 1; b < 4; ++b) {
            if (ggml_cuda_tensors_overlap(conv_states_all, out_tensors[b])) {
                return false;
            }
        }
    }

    m.conv_states_all = conv_states_all;
    m.state_ids       = state_ids;
    m.qkv_mixed       = qkv_mixed;
    m.conv_kernel     = conv_kernel;
    m.state_cell_dst  = cell;
    m.v_silu_dst      = silu;
    m.q_norm_dst      = q_scale;
    m.k_norm_dst      = k_scale;
    m.head_k_dim      = 128;
    m.num_k_heads     = num_k_heads;
    m.eps_over_n      = q_eps;
    m.inv_sqrt_n      = q_sc;
    m.node_count      = k - i;
    return true;
}


struct ggml_cuda_gdn_alpha_beta_match {
    const ggml_tensor * beta_in    = nullptr;
    const ggml_tensor * alpha_in   = nullptr;
    const ggml_tensor * ssm_dt     = nullptr;
    const ggml_tensor * ssm_a      = nullptr;
    ggml_tensor *       beta_out   = nullptr;
    ggml_tensor *       gate_out   = nullptr;
    int                 node_count = 0;
};


static bool ggml_cuda_match_gdn_alpha_beta(
        const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_alpha_beta_match & m, bool check_ptrs) {
    if (i + 3 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * beta_sig = cgraph->nodes[i];
    if (beta_sig->op != GGML_OP_UNARY || ggml_get_unary_op(beta_sig) != GGML_UNARY_OP_SIGMOID ||
            beta_sig->type != GGML_TYPE_F32 || !ggml_is_contiguous(beta_sig) ||
            ggml_nelements(beta_sig) == 0 || ggml_nelements(beta_sig) > 32) {
        return false;
    }
    const ggml_tensor * beta_in = beta_sig->src[0];
    if (!beta_in || beta_in->type != GGML_TYPE_F32 || !ggml_is_contiguous(beta_in) ||
            ggml_nelements(beta_in) != ggml_nelements(beta_sig)) {
        return false;
    }
    const int64_t n = ggml_nelements(beta_sig);

    int k = i + 1;
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if ((cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
        k++;
    }
    if (k + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add_n = cgraph->nodes[k];
    const ggml_tensor * sp_n  = cgraph->nodes[k + 1];
    ggml_tensor *       mul_n = cgraph->nodes[k + 2];
    if (add_n->op != GGML_OP_ADD || sp_n->op != GGML_OP_UNARY ||
            ggml_get_unary_op(sp_n) != GGML_UNARY_OP_SOFTPLUS || mul_n->op != GGML_OP_MUL) {
        return false;
    }
    if ((add_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || (sp_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    if (sp_n->src[0] != add_n || (mul_n->src[0] != sp_n && mul_n->src[1] != sp_n)) {
        return false;
    }
    const ggml_tensor * alpha_in = add_n->src[0];
    const ggml_tensor * ssm_dt   = add_n->src[1];
    const ggml_tensor * ssm_a    = (mul_n->src[0] == sp_n) ? mul_n->src[1] : mul_n->src[0];
    if (!alpha_in || !ssm_dt || !ssm_a ||
            add_n->type != GGML_TYPE_F32 || sp_n->type != GGML_TYPE_F32 || mul_n->type != GGML_TYPE_F32 ||
            alpha_in->type != GGML_TYPE_F32 || ssm_dt->type != GGML_TYPE_F32 || ssm_a->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(alpha_in) || !ggml_is_contiguous(ssm_dt) || !ggml_is_contiguous(ssm_a) ||
            !ggml_is_contiguous(mul_n) ||
            ggml_nelements(alpha_in) != n || ggml_nelements(ssm_dt) != n ||
            ggml_nelements(ssm_a) != n || ggml_nelements(mul_n) != n) {
        return false;
    }

    const int end_idx = k + 3;
    for (int j = end_idx; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * ext = cgraph->nodes[j];
        if (ext->view_src == add_n || ext->view_src == sp_n) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (ext->src[s] == add_n || ext->src[s] == sp_n) return false;
        }
    }

    if (check_ptrs) {
        if (!beta_in->data || !alpha_in->data || !ssm_dt->data || !ssm_a->data ||
                !beta_sig->data || !mul_n->data) {
            return false;
        }
        const ggml_tensor * ins[4]  = { beta_in, alpha_in, ssm_dt, ssm_a };
        const ggml_tensor * outs[2] = { beta_sig, mul_n };
        if (ggml_cuda_tensors_overlap(beta_sig, mul_n)) return false;
        for (const ggml_tensor * out_t : outs) {
            for (const ggml_tensor * in_t : ins) {
                if (ggml_cuda_tensors_overlap(out_t, in_t)) return false;
            }
        }
    }

    m.beta_in    = beta_in;
    m.alpha_in   = alpha_in;
    m.ssm_dt     = ssm_dt;
    m.ssm_a      = ssm_a;
    m.beta_out   = beta_sig;
    m.gate_out   = mul_n;
    m.node_count = end_idx - i;
    return true;
}

struct ggml_cuda_gdn_norm_gated_match {
    ggml_tensor * rms_node      = nullptr;
    ggml_tensor * norm_mul_node = nullptr;
    ggml_tensor * silu_node     = nullptr;
    ggml_tensor * out_mul_node  = nullptr;
    int           node_count    = 0;
};

static bool ggml_cuda_match_gdn_norm_gated(
        const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_norm_gated_match & m, bool check_ptrs) {
    if (i + 3 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * rms_n      = cgraph->nodes[i];
    ggml_tensor * norm_mul_n = cgraph->nodes[i + 1];
    if (rms_n->op != GGML_OP_RMS_NORM || norm_mul_n->op != GGML_OP_MUL ||
            rms_n->type != GGML_TYPE_F32 || norm_mul_n->type != GGML_TYPE_F32 ||
            rms_n->ne[0] != 128 || rms_n->ne[2] != 1 || rms_n->ne[3] != 1 ||
            !ggml_is_contiguous(rms_n) || !ggml_is_contiguous(norm_mul_n) ||
            (rms_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            (norm_mul_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const ggml_tensor * x = rms_n->src[0];
    const ggml_tensor * w = (norm_mul_n->src[0] == rms_n) ? norm_mul_n->src[1] :
                            (norm_mul_n->src[1] == rms_n) ? norm_mul_n->src[0] : nullptr;
    if (!x || !w || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(w) ||
            !ggml_are_same_shape(rms_n, norm_mul_n) ||
            w->ne[0] != 128 || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }

    int k = i + 2;
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if ((cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
        k++;
    }
    if (k + 1 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * silu_n    = cgraph->nodes[k];
    ggml_tensor * out_mul_n = cgraph->nodes[k + 1];
    if (silu_n->op != GGML_OP_UNARY || ggml_get_unary_op(silu_n) != GGML_UNARY_OP_SILU ||
            out_mul_n->op != GGML_OP_MUL ||
            silu_n->type != GGML_TYPE_F32 || out_mul_n->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(silu_n) || !ggml_is_contiguous(out_mul_n) ||
            !ggml_are_same_shape(silu_n, norm_mul_n) || !ggml_are_same_shape(out_mul_n, norm_mul_n) ||
            (silu_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    if (!((out_mul_n->src[0] == norm_mul_n && out_mul_n->src[1] == silu_n) ||
          (out_mul_n->src[1] == norm_mul_n && out_mul_n->src[0] == silu_n))) {
        return false;
    }
    const ggml_tensor * z = silu_n->src[0];
    if (!z || z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || !ggml_are_same_shape(z, norm_mul_n)) {
        return false;
    }

    const int end_idx = k + 2;
    for (int j = end_idx; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * ext = cgraph->nodes[j];
        if (ext->view_src == rms_n || ext->view_src == norm_mul_n || ext->view_src == silu_n) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (ext->src[s] == rms_n || ext->src[s] == norm_mul_n || ext->src[s] == silu_n) return false;
        }
    }

    if (check_ptrs) {
        if (!x->data || !w->data || !z->data || !out_mul_n->data) {
            return false;
        }
        if (ggml_cuda_tensors_overlap(out_mul_n, x) ||
                ggml_cuda_tensors_overlap(out_mul_n, w) ||
                ggml_cuda_tensors_overlap(out_mul_n, z)) {
            return false;
        }
    }

    m.rms_node      = rms_n;
    m.norm_mul_node = norm_mul_n;
    m.silu_node     = silu_n;
    m.out_mul_node  = out_mul_n;
    m.node_count    = end_idx - i;
    return true;
}

static int ggml_cuda_try_gdn_rs_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy,
        int & out_gdn_idx) {
    const ggml_tensor * get_rows = cgraph->nodes[node_idx];
    if (get_rows->op != GGML_OP_GET_ROWS || get_rows->type != GGML_TYPE_F32 ||
            (get_rows->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            get_rows->ne[1] != 1 || get_rows->ne[2] != 1 || get_rows->ne[3] != 1) {
        return 0;
    }
    const ggml_tensor * states_src = get_rows->src[0];
    const ggml_tensor * ids_src    = get_rows->src[1];
    if (!states_src || !ids_src || states_src->type != GGML_TYPE_F32 || ids_src->type != GGML_TYPE_I32 ||
            !states_src->data || !ids_src->data ||
            states_src->nb[0] != sizeof(float) || ids_src->nb[0] != sizeof(int32_t) ||
            ggml_nelements(ids_src) != 1 ||
            get_rows->ne[0] != states_src->ne[0]) {
        return 0;
    }

    int gdn_idx = -1;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op == GGML_OP_GATED_DELTA_NET) {
            gdn_idx = j;
        }
        break;
    }
    if (gdn_idx < 0) {
        return 0;
    }

    const ggml_tensor * gdn = cgraph->nodes[gdn_idx];
    if (gdn->src[6] != nullptr || gdn->src[2]->ne[2] != 1 || gdn->src[2]->ne[3] != 1) {
        return 0;
    }
    const ggml_tensor * s_root = gdn->src[5];
    if (!s_root || !ggml_is_contiguous(s_root)) {
        return 0;
    }
    while (s_root && s_root != get_rows && (s_root->op == GGML_OP_VIEW || s_root->op == GGML_OP_RESHAPE)) {
        if (s_root->view_offs != 0) {
            return 0;
        }
        s_root = s_root->view_src;
    }
    if (s_root != get_rows) {
        return 0;
    }

    // ensure neither get_rows nor any view of it has consumers past gdn_idx
    auto traces_to_get_rows = [&](const ggml_tensor * t) -> bool {
        while (t) {
            if (t == get_rows) {
                return true;
            }
            if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) {
                t = t->view_src;
            } else {
                break;
            }
        }
        return false;
    };
    for (int j = gdn_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->view_src && traces_to_get_rows(n->view_src)) {
            return 0;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] && traces_to_get_rows(n->src[s])) {
                return 0;
            }
        }
    }

    const int cache_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, gdn_idx, fused_state_cpy);
    if (cache_skip <= 0) {
        return 0;
    }
    fused_state_cpy.state_in_ptr        = (const float *) states_src->data;
    fused_state_cpy.state_ids           = (const int32_t *) ids_src->data;
    fused_state_cpy.state_in_row_stride = (int64_t) (states_src->nb[1] / sizeof(float));

    out_gdn_idx = gdn_idx;
    return (gdn_idx - node_idx) + cache_skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.norm_bias       = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP|SCALE->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // the sum is clamped from below, or a SCALE with scale 1 adds a bias (DeepSeek-V4.1, Inkling 2)
        const ggml_tensor * denom = node_idx < n_nodes ? nodes[node_idx] : nullptr;
        const bool is_clamp = denom && denom->op == GGML_OP_CLAMP;
        const bool is_bias  = denom && denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) == 1.0f;
        if ((!is_clamp && !is_bias) || denom->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }
        args.norm_bias = is_bias;
        node_idx++;

        // DIV <- CLAMP|SCALE, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor * norm   = cgraph->nodes[node_idx];
        const ggml_tensor * mul    = cgraph->nodes[node_idx + 1];
        const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        if (norm->src[0]->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            weight->type != GGML_TYPE_F32 || !ggml_is_vector(weight) || weight->ne[0] != norm->ne[0] ||
            !ggml_is_contiguous(weight)) {
            return false;
        }
        if (ops.size() == 3) {
            const ggml_tensor * add  = cgraph->nodes[node_idx + 2];
            const ggml_tensor * bias = add->src[0] == mul ? add->src[1] : add->src[0];
            if (ops.begin()[2] != GGML_OP_ADD || add->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
                !ggml_is_vector(bias) || bias->ne[0] != norm->ne[0] || !ggml_is_contiguous(bias)) {
                return false;
            }
        }
        return true;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        // the mul operand is a norm weight, which model files store as F32;
        // the add operand is an F32 bias or a residual with the type of x
        const ggml_type type = rms_norm->type;
        if ((type != GGML_TYPE_F32 && type != GGML_TYPE_BF16) || rms_norm->src[0]->type != type) {
            return false;
        }

        const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
        if (mul_src->type != GGML_TYPE_F32 || mul->type != type) {
            return false;
        }

        if (add) {
            const ggml_tensor * add_src = add->src[0] == mul ? add->src[1] : add->src[0];
            if ((add_src->type != GGML_TYPE_F32 && add_src->type != type) || add->type != type) {
                return false;
            }
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) {
            return false;
        }

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        // TODO: make the fused softcap kernel stride aware, it reads src and writes dst as dense
        if (!ggml_is_contiguous(scale->src[0]) || !ggml_is_contiguous(scale2)) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// fused op chains of the Inkling 2 graph (inkling2.cuh, ggml_cuda_op_l2_clip)

static bool ggml_cuda_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    const char * a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const char * b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return a0 < b1 && b0 < a1;
}

// a dimension of src broadcasts to dst if it is 1 or equal
static bool ggml_cuda_can_bcast_to(const ggml_tensor * src, const ggml_tensor * dst, int first_dim) {
    for (int d = first_dim; d < GGML_MAX_DIMS; ++d) {
        if (src->ne[d] != 1 && src->ne[d] != dst->ne[d]) {
            return false;
        }
    }
    return true;
}

// [MUL by a per-column weight] -> L2_NORM -> SCALE [-> MUL by a per-channel/sample factor]
static int ggml_cuda_try_fuse_l2_clip(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node  = cgraph->nodes[i];
    const bool          has_w = node->op == GGML_OP_MUL;
    if (!has_w && node->op != GGML_OP_L2_NORM) {
        return 0;
    }

    const ggml_tensor * x = node->src[0];
    const ggml_tensor * w = has_w ? node->src[1] : nullptr;
    if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) {
        return 0;
    }
    if (has_w && (!ggml_are_same_shape(x, node) || w->type != GGML_TYPE_F32 || w->nb[0] != sizeof(float) ||
                  w->ne[0] != x->ne[0] || !ggml_cuda_can_bcast_to(w, x, 1))) {
        return 0;
    }

    for (const bool has_f : { true, false }) {
        ggml_op ops[4];
        int     n = 0;
        if (has_w) {
            ops[n++] = GGML_OP_MUL;
        }
        ops[n++] = GGML_OP_L2_NORM;
        ops[n++] = GGML_OP_SCALE;
        if (has_f) {
            ops[n++] = GGML_OP_MUL;
        }

        if (!ggml_can_fuse(cgraph, i, ops, n)) {
            continue;
        }

        const ggml_tensor * l2  = cgraph->nodes[i + n - (has_f ? 3 : 2)];
        const ggml_tensor * sc  = cgraph->nodes[i + n - (has_f ? 2 : 1)];
        ggml_tensor *       dst = cgraph->nodes[i + n - 1];
        const ggml_tensor * f   = nullptr;

        if (l2->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 || sc->src[0] != l2 || (has_w && l2->src[0] != node) ||
            !ggml_is_contiguous(dst)) {
            continue;
        }
        if (has_f) {
            f = dst->src[1];
            if (dst->src[0] != sc || f->type != GGML_TYPE_F32 || f->ne[0] != 1 || f->ne[1] != 1 || !ggml_cuda_can_bcast_to(f, dst, 2)) {
                continue;
            }
        }

        // a block reads its row before it writes it, so dst may be x itself but must not overlap it otherwise
        const bool x_in_place = x->data == dst->data && ggml_is_contiguous(x);
        if ((!x_in_place && ggml_cuda_tensors_overlap(x, dst)) || (w && ggml_cuda_tensors_overlap(w, dst)) ||
            (f && ggml_cuda_tensors_overlap(f, dst))) {
            continue;
        }

        ggml_cuda_op_l2_clip(*cuda_ctx, x, w, l2, sc, f, dst);
        return n - 1;
    }

    return 0;
}

// situ on the halves of a merged gate_up: SCALE -> TANH -> SCALE, SIGMOID, MUL, (up view) SCALE -> TANH -> SCALE, MUL
static int ggml_cuda_try_fuse_situ(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static constexpr ggml_op ops[] = {
        GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_VIEW,
        GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL,
    };
    constexpr int n = sizeof(ops)/sizeof(ops[0]);

    if (i + n > cgraph->n_nodes) {
        return 0;
    }
    for (int k = 0; k < n; ++k) {
        if (cgraph->nodes[i + k]->op != ops[k]) {
            return 0;
        }
    }

    ggml_tensor ** nd = cgraph->nodes + i;

    const ggml_tensor * gate = nd[0]->src[0];
    const ggml_tensor * up   = nd[5];

    const auto is_mul_of = [](const ggml_tensor * t, const ggml_tensor * a, const ggml_tensor * b) {
        return (t->src[0] == a && t->src[1] == b) || (t->src[0] == b && t->src[1] == a);
    };

    if (ggml_get_unary_op(nd[1]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nd[3]) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_unary_op(nd[7]) != GGML_UNARY_OP_TANH ||
        nd[1]->src[0] != nd[0] || nd[2]->src[0] != nd[1] || nd[3]->src[0] != gate || !is_mul_of(nd[4], nd[2], nd[3]) ||
        nd[6]->src[0] != up || nd[7]->src[0] != nd[6] || nd[8]->src[0] != nd[7] || !is_mul_of(nd[9], nd[4], nd[8])) {
        return 0;
    }

    if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || gate->nb[0] != sizeof(float) || up->nb[0] != sizeof(float) ||
        !ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, nd[9]) || !ggml_is_contiguous(nd[9])) {
        return 0;
    }
    for (int k = 0; k < n; ++k) {
        if (nd[k]->type != GGML_TYPE_F32) {
            return 0;
        }
    }

    // the up view is a view of a node outside the group, it is not part of the subgraph check
    const int     idxs[]   = { i, i + 1, i + 2, i + 3, i + 4, i + 6, i + 7, i + 8, i + 9 };
    const ggml_op ops_ext[] = { ops[0], ops[1], ops[2], ops[3], ops[4], ops[6], ops[7], ops[8], ops[9] };
    const int     out       = i + n - 1;
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 9, ops_ext, &out, 1) || !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1)) {
        return 0;
    }

    const float s[4] = {
        ggml_get_op_params_f32(nd[0], 0), ggml_get_op_params_f32(nd[2], 0), ggml_get_op_params_f32(nd[6], 0), ggml_get_op_params_f32(nd[8], 0),
    };
    const float b[4] = {
        ggml_get_op_params_f32(nd[0], 1), ggml_get_op_params_f32(nd[2], 1), ggml_get_op_params_f32(nd[6], 1), ggml_get_op_params_f32(nd[8], 1),
    };

    ggml_cuda_op_situ(*cuda_ctx, gate, up, s, b, nd[9]);
    return n - 1;
}

// depth mix: [RMS_NORM(cur)] -> CONCAT(stack, norm) -> [REPEAT(query)] -> MUL_MAT -> SOFT_MAX, TRANSPOSE -> CONT -> MUL_MAT
static int ggml_cuda_try_fuse_depth_mix(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node     = cgraph->nodes[i];
    const bool          has_norm = node->op == GGML_OP_RMS_NORM;

    const int j = i + (has_norm ? 1 : 0);
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }

    const ggml_tensor * cc = cgraph->nodes[j];
    if (cc->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cc, 0) != 1 || (has_norm && cc->src[1] != node) ||
        (!has_norm && cc->src[1]->op != GGML_OP_RMS_NORM)) {
        return 0;
    }

    const bool has_rep = cgraph->nodes[j + 1]->op == GGML_OP_REPEAT;

    ggml_op ops[8];
    int     n = 0;
    if (has_norm) {
        ops[n++] = GGML_OP_RMS_NORM;
    }
    ops[n++] = GGML_OP_CONCAT;
    if (has_rep) {
        ops[n++] = GGML_OP_REPEAT;
    }
    ops[n++] = GGML_OP_MUL_MAT;
    ops[n++] = GGML_OP_SOFT_MAX;
    ops[n++] = GGML_OP_TRANSPOSE;
    ops[n++] = GGML_OP_CONT;
    ops[n++] = GGML_OP_MUL_MAT;

    if (i + n > cgraph->n_nodes) {
        return 0;
    }
    for (int k = 0; k < n; ++k) {
        if (cgraph->nodes[i + k]->op != ops[k]) {
            return 0;
        }
    }

    const ggml_tensor * rep = has_rep ? cgraph->nodes[j + 1] : nullptr;
    const ggml_tensor * mm1 = cgraph->nodes[i + n - 5];
    const ggml_tensor * sm  = cgraph->nodes[i + n - 4];
    const ggml_tensor * tr  = cgraph->nodes[i + n - 3];
    const ggml_tensor * ct  = cgraph->nodes[i + n - 2];
    ggml_tensor *       mm2 = cgraph->nodes[i + n - 1];

    const ggml_tensor * stack = cc->src[0];
    const ggml_tensor * cur   = has_norm ? node->src[0] : cc->src[1];
    const ggml_tensor * query = has_rep ? rep->src[0] : mm1->src[1];

    if (mm1->src[0] != cc || mm1->src[1] != (has_rep ? rep : query) || sm->src[0] != mm1 || sm->src[1] || sm->src[2] ||
        ggml_get_op_params_f32(sm, 1) != 0.0f || tr->src[0] != cc || ct->src[0] != tr || mm2->src[0] != sm || mm2->src[1] != ct) {
        return 0;
    }

    const int64_t ne = cur->ne[0];
    const int64_t T  = cur->ne[2];
    const int64_t B  = stack->ne[1];
    const int64_t Q  = query->ne[1];

    for (const ggml_tensor * t : std::initializer_list<const ggml_tensor *>{ stack, cur, query, cc, mm1, sm, mm2 }) {
        if (t->type != GGML_TYPE_F32) {
            return 0;
        }
    }
    if (cur->nb[0] != sizeof(float) || cur->ne[1] != 1 || cur->ne[3] != 1 || !ggml_is_contiguous(stack) ||
        stack->ne[0] != ne || stack->ne[2] != T || stack->ne[3] != 1 || B < 1 || B > GGML_CUDA_DEPTH_MIX_MAX_ROWS ||
        !ggml_is_contiguous(query) || query->ne[0] != ne || query->ne[2] != 1 || query->ne[3] != 1 || (!has_rep && T != 1) ||
        mm2->ne[0] != Q || mm2->ne[1] != ne || mm2->ne[2] != T || !ggml_is_contiguous(mm2)) {
        return 0;
    }

    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1) || !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1)) {
        return 0;
    }

    const float eps = has_norm ? ggml_get_op_params_f32(node, 0) : 0.0f;
    ggml_cuda_op_depth_mix(*cuda_ctx, stack, cur, has_norm, eps, query, ggml_get_op_params_f32(sm, 0), mm2);
    return n - 1;
}

// loop controller: CONCAT(cur, msg) -> RMS_NORM -> RESHAPE -> MUL_MAT(down) -> MUL_MAT(forget) -> TANH -> MUL(cur) -> SUB(cur),
// MUL_MAT(gain) -> TANH -> MUL(msg) -> ADD
static int ggml_cuda_try_fuse_loop_ctrl(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static constexpr ggml_op ops[] = {
        GGML_OP_CONCAT, GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB,
        GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD,
    };
    constexpr int n = sizeof(ops)/sizeof(ops[0]);

    if (i + n > cgraph->n_nodes) {
        return 0;
    }
    for (int k = 0; k < n; ++k) {
        if (cgraph->nodes[i + k]->op != ops[k]) {
            return 0;
        }
    }

    ggml_tensor ** nd = cgraph->nodes + i;

    const ggml_tensor * cc = nd[0];
    if (ggml_get_op_params_i32(cc, 0) != 1 || cc->src[0]->op != GGML_OP_RESHAPE || cc->src[1]->op != GGML_OP_RESHAPE) {
        return 0;
    }

    const ggml_tensor * cur  = cc->src[0]->src[0];
    const ggml_tensor * msg  = cc->src[1]->src[0];
    const ggml_tensor * down = nd[3];
    const ggml_tensor * wd   = down->src[0];
    const ggml_tensor * wf   = nd[4]->src[0];
    const ggml_tensor * wg   = nd[8]->src[0];
    ggml_tensor *       dst  = nd[11];

    if (nd[1]->src[0] != cc || nd[2]->src[0] != nd[1] || down->src[1] != nd[2] || nd[4]->src[1] != down ||
        ggml_get_unary_op(nd[5]) != GGML_UNARY_OP_TANH || nd[5]->src[0] != nd[4] ||
        nd[6]->src[0] != nd[5] || nd[6]->src[1] != cur || nd[7]->src[0] != cur || nd[7]->src[1] != nd[6] ||
        nd[8]->src[1] != down || ggml_get_unary_op(nd[9]) != GGML_UNARY_OP_TANH || nd[9]->src[0] != nd[8] ||
        nd[10]->src[0] != nd[9] || nd[10]->src[1] != msg || dst->src[0] != nd[7] || dst->src[1] != nd[10]) {
        return 0;
    }

    const int64_t ne   = cur->ne[0];
    const int64_t T    = cur->ne[1];
    const int64_t rank = wd->ne[1];

    if (T > GGML_CUDA_LOOP_CTRL_MAX_TOKENS || !ggml_is_contiguous(cur) || !ggml_is_contiguous(msg) || !ggml_are_same_shape(cur, msg) ||
        cur->ne[2] != 1 || cur->ne[3] != 1 || !ggml_are_same_shape(cur, dst) || !ggml_is_contiguous(dst) ||
        !ggml_is_contiguous(down) || down->ne[0] != rank || down->ne[1] != T) {
        return 0;
    }
    for (const ggml_tensor * t : std::initializer_list<const ggml_tensor *>{ cur, msg, wd, wf, wg, down, dst }) {
        if (t->type != GGML_TYPE_F32) {
            return 0;
        }
    }
    for (const ggml_tensor * w : { wd, wf, wg }) {
        if (!ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1) {
            return 0;
        }
    }
    if (wd->ne[0] != 2*ne || wf->ne[0] != rank || wg->ne[0] != rank || wf->ne[1] != ne || wg->ne[1] != ne) {
        return 0;
    }

    // the down product is written by the first kernel and read by the second, so the output must not overlap it
    const int outs[] = { i + 3, i + n - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs, 2) || !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, outs, 2) ||
        ggml_cuda_tensors_overlap(dst, down)) {
        return 0;
    }

    ggml_cuda_op_loop_ctrl(*cuda_ctx, cur, msg, ggml_get_op_params_f32(nd[1], 0), wd, wg, wf, nd[3], dst);
    return n - 1;
}

// short conv: CONCAT(past, cur^T) -> the state store -> SSM_CONV -> RESHAPE -> [taps] -> ADD(cur) -> SCALE [-> ADD]
// each tap: VIEW(x) -> TRANSPOSE -> CONT -> RESHAPE, [RESHAPE(dyn)], VIEW(dyn) -> MUL -> RESHAPE -> ADD
static int ggml_cuda_try_fuse_short_conv(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    // the subgraph nodes: 5 for the concat, state store and conv, 7 per tap, 3 for the adds and the scale
    constexpr int MAX_NODES = 8 + 7*GGML_CUDA_SHORT_CONV_MAX_K;

    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cc, 0) != 0 || i + 8 > cgraph->n_nodes) {
        return 0;
    }

    const ggml_tensor * past = cc->src[0];
    const ggml_tensor * xin  = cc->src[1];

    if (cc->type != GGML_TYPE_F32 || past->type != GGML_TYPE_F32 || past->nb[0] != sizeof(float) || cc->ne[3] != 1 ||
        !ggml_is_contiguous(cc) || xin->op != GGML_OP_TRANSPOSE || xin->src[0]->op != GGML_OP_RESHAPE) {
        return 0;
    }

    const int64_t K1 = past->ne[0];
    const int64_t C  = cc->ne[1];
    const int64_t S  = cc->ne[2];
    const int64_t Ts = cc->ne[0] - K1;
    const int64_t K  = K1 + 1;

    const ggml_tensor * cur = xin->src[0]->src[0];
    if (K < 2 || K > GGML_CUDA_SHORT_CONV_MAX_K || Ts < 1 || past->ne[1] != C || past->ne[2] != S || xin->ne[0] != Ts ||
        cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) || cur->ne[0] != C || ggml_nelements(cur) != C*Ts*S) {
        return 0;
    }

    int     idxs[MAX_NODES];
    ggml_op ops[MAX_NODES];
    int     n_sub = 0;
    auto    add_node = [&](int k) {
        idxs[n_sub] = k;
        ops[n_sub]  = cgraph->nodes[k]->op;
        n_sub++;
    };

    int k = i;
    add_node(k++);

    // the state store: the last K - 1 columns of x copied to the cell of each sequence (VIEW, VIEW, CPY), or with rollback
    // slots the windows that end at the last kw tokens set into slots kw - 1 .. 0 (VIEW, ARANGE, SCALE, CPY, VIEW, VIEW, SET_ROWS)
    const ggml_tensor * cell    = nullptr;
    int                 st_idx  = -1;
    int64_t             kw      = 1;
    int64_t             slot_s  = 0;
    int64_t             ch_s    = K1;
    int64_t             seq_s   = 0;
    if (cgraph->nodes[k + 1]->op == GGML_OP_ARANGE) {
        if (k + 7 > cgraph->n_nodes) {
            return 0;
        }
        const ggml_tensor * win = cgraph->nodes[k];
        const ggml_tensor * ar  = cgraph->nodes[k + 1];
        const ggml_tensor * sb  = cgraph->nodes[k + 2];
        const ggml_tensor * cs  = cgraph->nodes[k + 3];
        const ggml_tensor * ord = cgraph->nodes[k + 4];
        const ggml_tensor * sl  = cgraph->nodes[k + 5];
        const ggml_tensor * sr  = cgraph->nodes[k + 6];

        const int64_t n_slots = ar->ne[0];
        kw = ord->ne[0];

        // the slot order {n_slots - 1, ..., 0}, its last kw entries
        if (win->op != GGML_OP_VIEW || sb->op != GGML_OP_SCALE || cs->op != GGML_OP_CPY || ord->op != GGML_OP_VIEW ||
            sl->op != GGML_OP_VIEW || sr->op != GGML_OP_SET_ROWS ||
            ggml_get_op_params_f32(ar, 0) != 0.0f || ggml_get_op_params_f32(ar, 2) != 1.0f || sb->src[0] != ar ||
            ggml_get_op_params_f32(sb, 0) != -1.0f || ggml_get_op_params_f32(sb, 1) != (float) (n_slots - 1) ||
            cs->src[0] != sb || cs->type != GGML_TYPE_I32 || ord->view_src != cs ||
            ord->view_offs != (size_t) (n_slots - kw)*sizeof(int32_t) || kw < 1 || kw > n_slots || kw > Ts) {
            return 0;
        }
        // window w is x[Ts - (kw - 1) + w, +K - 1) of each channel
        if (win->view_src != cc || win->ne[0] != K1 || win->ne[1] != kw || win->ne[2] != C || win->ne[3] != S ||
            win->nb[0] != sizeof(float) || win->nb[1] != sizeof(float) || win->nb[2] != cc->nb[1] || win->nb[3] != cc->nb[2] ||
            win->view_offs != (size_t) (Ts - (kw - 1))*sizeof(float)) {
            return 0;
        }
        // SET_ROWS: src[0] the rows, src[1] the indices, src[2] the destination
        if (sr->src[0] != win || sr->src[1] != ord || sr->src[2] != sl || sl->type != GGML_TYPE_F32 || sl->nb[0] != sizeof(float) ||
            sl->ne[0] != K1 || sl->ne[1] != n_slots || sl->ne[2] != C || sl->ne[3] != S) {
            return 0;
        }
        // the slot order is a constant the kernel does not need; the cast is its own src[1], so its uses are checked here
        if (ggml_node_get_use_count(cgraph, k + 1) != 1 || ggml_node_get_use_count(cgraph, k + 2) != 1 ||
            ggml_node_get_use_count(cgraph, k + 3) != 1 + (cs->src[1] == cs ? 1 : 0) || ggml_node_get_use_count(cgraph, k + 4) != 1) {
            return 0;
        }
        cell   = sl;
        st_idx = k + 6;
        slot_s = sl->nb[1]/sizeof(float);
        ch_s   = sl->nb[2]/sizeof(float);
        seq_s  = sl->nb[3]/sizeof(float);
        add_node(k);     // windows
        add_node(k + 6); // set_rows, the slots are a view of the state outside the group
        k += 7;
    } else {
        const ggml_tensor * tail = cgraph->nodes[k];
        const ggml_tensor * cpy  = cgraph->nodes[k + 2];
        cell = cgraph->nodes[k + 1];
        if (tail->op != GGML_OP_VIEW || cell->op != GGML_OP_VIEW || cpy->op != GGML_OP_CPY ||
            tail->view_src != cc || tail->view_offs != (size_t) Ts*sizeof(float) || tail->ne[0] != K1 || tail->ne[1] != C ||
            tail->ne[2] != S || tail->nb[1] != cc->nb[1] || tail->nb[2] != cc->nb[2] || cpy->src[0] != tail || cpy->src[1] != cell ||
            cell->type != GGML_TYPE_F32 || cell->nb[0] != sizeof(float) || cell->ne[0] != K1*C || ggml_nrows(cell) != S ||
            ggml_get_op_params_i32(cpy, 0) != 0) {
            return 0;
        }
        st_idx = k + 2;
        seq_s  = cell->nb[1]/sizeof(float);
        add_node(k);     // tail
        add_node(k + 2); // cpy, the cell is a view of the state outside the group
        k += 3;
    }
    const int cpy_idx = st_idx;

    if (k + 4 > cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * conv = cgraph->nodes[k];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * out  = cgraph->nodes[k + 1];
    if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != cc || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
        w->ne[0] != K || w->ne[1] != C || out->op != GGML_OP_RESHAPE || out->src[0] != conv || out->ne[0] != C || out->ne[1] != Ts*S) {
        return 0;
    }
    add_node(k);
    add_node(k + 1);
    k += 2;

    // the input-dependent taps, one per lag
    const ggml_tensor * dyn     = nullptr;
    int64_t             n_heads = 0;
    if (cgraph->nodes[k]->op == GGML_OP_VIEW && cgraph->nodes[k]->view_src == cc) {
        for (int64_t lag = 0; lag < K; ++lag) {
            if (k + 9 > cgraph->n_nodes) {
                return 0;
            }
            const ggml_tensor * win = cgraph->nodes[k];
            const ggml_tensor * tr  = cgraph->nodes[k + 1];
            const ggml_tensor * ct  = cgraph->nodes[k + 2];
            const ggml_tensor * wr  = cgraph->nodes[k + 3];
            if (win->op != GGML_OP_VIEW || tr->op != GGML_OP_TRANSPOSE || ct->op != GGML_OP_CONT || wr->op != GGML_OP_RESHAPE ||
                win->view_src != cc || win->view_offs != (size_t) (K1 - lag)*sizeof(float) || win->ne[0] != Ts || win->ne[1] != C ||
                win->ne[2] != S || win->nb[1] != cc->nb[1] || win->nb[2] != cc->nb[2] ||
                tr->src[0] != win || ct->src[0] != tr || wr->src[0] != ct) {
                return 0;
            }
            for (int m = 0; m < 4; ++m) {
                add_node(k + m);
            }
            k += 4;

            if (lag == 0) {
                const ggml_tensor * d3 = cgraph->nodes[k];
                if (d3->op != GGML_OP_RESHAPE || d3->ne[0] != K || d3->ne[2] != Ts*S) {
                    return 0;
                }
                dyn     = d3->src[0];
                n_heads = d3->ne[1];
                if (dyn->type != GGML_TYPE_F32 || !ggml_is_contiguous(dyn) || dyn->ne[0] != K*n_heads || n_heads < 1 || C % n_heads != 0) {
                    return 0;
                }
                k += 1; // a view of the taps input outside the group
            }

            const ggml_tensor * tap  = cgraph->nodes[k];
            const ggml_tensor * mul  = cgraph->nodes[k + 1];
            const ggml_tensor * term = cgraph->nodes[k + 2];
            const ggml_tensor * add  = cgraph->nodes[k + 3];
            if (tap->op != GGML_OP_VIEW || mul->op != GGML_OP_MUL || term->op != GGML_OP_RESHAPE || add->op != GGML_OP_ADD ||
                tap->view_src != dyn || tap->view_offs != dyn->view_offs + (size_t) lag*sizeof(float) || tap->ne[0] != 1 ||
                tap->ne[1] != n_heads || tap->nb[1] != dyn->nb[0]*K || tap->nb[2] != dyn->nb[1] ||
                wr->ne[0] != C/n_heads || wr->ne[1] != n_heads ||
                mul->src[0] != wr || mul->src[1] != tap || term->src[0] != mul || add->src[0] != out || add->src[1] != term) {
                return 0;
            }
            add_node(k + 1);
            add_node(k + 2);
            add_node(k + 3);
            k += 4;
            out = add;
        }
    }

    if (k + 2 > cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * res_add = cgraph->nodes[k];
    ggml_tensor *       sc      = cgraph->nodes[k + 1];
    if (res_add->op != GGML_OP_ADD || sc->op != GGML_OP_SCALE || res_add->src[0] != cur || res_add->src[1] != out ||
        sc->src[0] != res_add || !ggml_is_contiguous(sc) || sc->type != GGML_TYPE_F32) {
        return 0;
    }
    add_node(k);
    add_node(k + 1);
    k += 2;

    // the residual add that follows the attention and FFN output convs
    ggml_tensor *       dst   = sc;
    const ggml_tensor * resid = nullptr;
    if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_ADD &&
        (cgraph->nodes[k]->src[0] == sc || cgraph->nodes[k]->src[1] == sc)) {
        ggml_tensor *       add = cgraph->nodes[k];
        const ggml_tensor * r   = add->src[0] == sc ? add->src[1] : add->src[0];
        if (r != sc && r->type == GGML_TYPE_F32 && ggml_is_contiguous(r) && ggml_are_same_shape(r, sc) &&
            ggml_is_contiguous(add) && add->type == GGML_TYPE_F32) {
            int     outs[] = { cpy_idx, k };
            add_node(k);
            if (ggml_can_fuse_subgraph_ext(cgraph, idxs, n_sub, ops, outs, 2)) {
                dst   = add;
                resid = r;
                k += 1;
            } else {
                n_sub--;
            }
        }
    }

    const int n      = k - i;
    const int outs[] = { cpy_idx, k - 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n_sub, ops, outs, 2)) {
        return 0;
    }

    // the kernel reads past, cur, w, the taps and the residual and writes the state and dst. a thread reads the residual
    // and cur at a token right before it writes dst there, so dst may be either of them in place when one thread has
    // the whole sequence; any other overlap is rejected
    const bool cur_in_place = cur->data == dst->data;
    if ((cur_in_place && Ts > 64) || (!cur_in_place && ggml_cuda_tensors_overlap(cur, dst)) ||
        (resid && resid->data != dst->data && ggml_cuda_tensors_overlap(resid, dst)) ||
        ggml_cuda_tensors_overlap(cell, dst) || (resid && ggml_cuda_tensors_overlap(resid, cell))) {
        return 0;
    }
    for (const ggml_tensor * t : { cell, (const ggml_tensor *) dst }) {
        for (const ggml_tensor * src : { past, w, dyn }) {
            if (src && ggml_cuda_tensors_overlap(t, src)) {
                return 0;
            }
        }
    }
    if (ggml_cuda_tensors_overlap(cell, cur)) {
        return 0;
    }

    ggml_cuda_short_conv_args args = {};
    args.past         = (const float *) past->data;
    args.past_s1      = past->nb[1]/sizeof(float);
    args.past_s2      = past->nb[2]/sizeof(float);
    args.cur          = (const float *) cur->data;
    args.w            = (const float *) w->data;
    args.dyn          = dyn ? (const float *) dyn->data : nullptr;
    args.dyn_s1       = dyn ? dyn->nb[1]/sizeof(float) : 0;
    args.n_heads      = n_heads;
    args.res          = resid ? (const float *) resid->data : nullptr;
    args.dst          = (float *) dst->data;
    args.state        = (float *) cell->data;
    args.state_slot_s = slot_s;
    args.state_ch_s   = ch_s;
    args.state_seq_s  = seq_s;
    args.kw           = kw;
    args.run          = cur_in_place || Ts <= 8 ? Ts : 1;
    args.C            = C;
    args.K            = K;
    args.Ts           = Ts;
    args.S            = S;
    args.scale        = ggml_get_op_params_f32(sc, 0);
    args.bias         = ggml_get_op_params_f32(sc, 1);

    ggml_cuda_op_short_conv(*cuda_ctx, args);
    return n - 1;
}

static int ggml_cuda_try_fuse_inkling2(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    switch (cgraph->nodes[i]->op) {
        case GGML_OP_CONCAT:
            if (int n = ggml_cuda_try_fuse_short_conv(cuda_ctx, cgraph, i)) {
                return n;
            }
            if (int n = ggml_cuda_try_fuse_loop_ctrl(cuda_ctx, cgraph, i)) {
                return n;
            }
            return ggml_cuda_try_fuse_depth_mix(cuda_ctx, cgraph, i);
        case GGML_OP_RMS_NORM:
            return ggml_cuda_try_fuse_depth_mix(cuda_ctx, cgraph, i);
        case GGML_OP_MUL:
        case GGML_OP_L2_NORM:
            return ggml_cuda_try_fuse_l2_clip(cuda_ctx, cgraph, i);
        case GGML_OP_SCALE:
            return ggml_cuda_try_fuse_situ(cuda_ctx, cgraph, i);
        default:
            return 0;
    }
}

// try and fuse nodes and return the number of nodes to skip

struct ggml_cuda_top_k_top_p_sample_match {
    const ggml_tensor * logits     = nullptr;
    const ggml_tensor * rand_val   = nullptr;
    const ggml_tensor * logits_2d  = nullptr;
    ggml_tensor *       dst        = nullptr;
    int                 k          = 0;
    float               top_p      = 1.0f;
    int                 node_count = 0;
};



struct ggml_cuda_moe_reduce_shared_add_match {
    const ggml_tensor * experts         = nullptr;
    const ggml_tensor * weights         = nullptr;
    const ggml_tensor * shared_out      = nullptr;
    const ggml_tensor * shared_gate_raw = nullptr;
    const ggml_tensor * residual        = nullptr;
    ggml_tensor *       dst             = nullptr;
    int                 node_count      = 0;
};

static bool ggml_cuda_match_moe_reduce_shared_add(
        const ggml_cgraph * cgraph, int i, ggml_cuda_moe_reduce_shared_add_match & m, bool check_ptrs) {
    ggml_cuda_moe_weighted_reduction_match red_match;
    if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, red_match)) {
        return false;
    }
    if (red_match.expert_scale != nullptr || red_match.experts->ne[2] * red_match.experts->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * moe_out = red_match.dst;
    const int64_t n_embd = moe_out->ne[0];

    int k = i + red_match.node_count;
    const int red_end = k;
    // Expect: [views] -> SIGMOID(shared_gate) -> MUL(shared_out, sig) -> ADD(moe_out, mul_shared) [-> ADD(add1, residual)]
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if ((cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
        k++;
    }
    if (k + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * sig_n  = cgraph->nodes[k];
    const ggml_tensor * mul_n  = cgraph->nodes[k + 1];
    ggml_tensor *       add1_n = cgraph->nodes[k + 2];
    if (sig_n->op != GGML_OP_UNARY || ggml_get_unary_op(sig_n) != GGML_UNARY_OP_SIGMOID ||
            mul_n->op != GGML_OP_MUL || add1_n->op != GGML_OP_ADD ||
            sig_n->type != GGML_TYPE_F32 || mul_n->type != GGML_TYPE_F32 || add1_n->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(sig_n) || !ggml_is_contiguous(mul_n) || !ggml_is_contiguous(add1_n) ||
            ggml_nelements(sig_n) != 1 ||
            mul_n->ne[0] != n_embd || mul_n->ne[1] != 1 || mul_n->ne[2] != 1 || mul_n->ne[3] != 1 ||
            add1_n->ne[0] != n_embd || add1_n->ne[1] != 1 || add1_n->ne[2] != 1 || add1_n->ne[3] != 1 ||
            (sig_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            (mul_n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const ggml_tensor * gate_raw = sig_n->src[0];
    if (!gate_raw || gate_raw->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate_raw) || ggml_nelements(gate_raw) != 1) {
        return false;
    }
    const ggml_tensor * shared_out = (mul_n->src[0] == sig_n) ? mul_n->src[1] :
                                     (mul_n->src[1] == sig_n) ? mul_n->src[0] : nullptr;
    if (!shared_out || shared_out->type != GGML_TYPE_F32 || !ggml_is_contiguous(shared_out) ||
            shared_out->ne[0] != n_embd || shared_out->ne[1] != 1 || shared_out->ne[2] != 1 || shared_out->ne[3] != 1) {
        return false;
    }
    if (!((add1_n->src[0] == moe_out && add1_n->src[1] == mul_n) ||
          (add1_n->src[1] == moe_out && add1_n->src[0] == mul_n))) {
        return false;
    }

    int end_idx = k + 3;
    ggml_tensor * final_dst = add1_n;
    const ggml_tensor * residual = nullptr;

    // Optional outer residual ADD(add1_n, residual)
    if (end_idx < cgraph->n_nodes && cgraph->nodes[end_idx]->op == GGML_OP_ADD &&
            (add1_n->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
        ggml_tensor * add2_n = cgraph->nodes[end_idx];
        const ggml_tensor * res_cand = (add2_n->src[0] == add1_n) ? add2_n->src[1] :
                                       (add2_n->src[1] == add1_n) ? add2_n->src[0] : nullptr;
        if (res_cand && add2_n->type == GGML_TYPE_F32 && res_cand->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(add2_n) && ggml_is_contiguous(res_cand) &&
                add2_n->ne[0] == n_embd && add2_n->ne[1] == 1 && add2_n->ne[2] == 1 && add2_n->ne[3] == 1 &&
                res_cand->ne[0] == n_embd && res_cand->ne[1] == 1 && res_cand->ne[2] == 1 && res_cand->ne[3] == 1) {
            // Check add1_n has only 1 consumer (add2_n)
            bool add1_ext = false;
            for (int j = end_idx + 1; j < cgraph->n_nodes; ++j) {
                const ggml_tensor * ext = cgraph->nodes[j];
                if (ext->view_src == add1_n) { add1_ext = true; break; }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (ext->src[s] == add1_n) { add1_ext = true; break; }
                }
                if (add1_ext) break;
            }
            if (!add1_ext) {
                residual  = res_cand;
                final_dst = add2_n;
                end_idx++;
            }
        }
    }

    // Ensure moe_out, views between red_end and sig_n, sig_n, and mul_n are unused outside [i .. end_idx-1]
    if ((moe_out->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    for (int j = end_idx; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * ext = cgraph->nodes[j];
        for (int vi = red_end; vi <= k + 1; ++vi) {
            const ggml_tensor * t = cgraph->nodes[vi];
            if (ext->view_src == t) return false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (ext->src[s] == t) return false;
            }
        }
        if (ext->view_src == moe_out) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (ext->src[s] == moe_out) return false;
        }
    }

    if (check_ptrs) {
        if (!red_match.experts->data || !red_match.weights->data || !shared_out->data ||
                !gate_raw->data || !final_dst->data || (residual && !residual->data)) {
            return false;
        }
        // residual IS safe to overlap final_dst (per-thread elementwise read before write)
        if (ggml_cuda_tensors_overlap(final_dst, red_match.experts) ||
                ggml_cuda_tensors_overlap(final_dst, red_match.weights) ||
                ggml_cuda_tensors_overlap(final_dst, shared_out) ||
                ggml_cuda_tensors_overlap(final_dst, gate_raw)) {
            return false;
        }
    }

    m.experts         = red_match.experts;
    m.weights         = red_match.weights;
    m.shared_out      = shared_out;
    m.shared_gate_raw = gate_raw;
    m.residual        = residual;
    m.dst             = final_dst;
    m.node_count      = end_idx - i;
    return true;
}


static bool ggml_cuda_match_top_k_top_p_sample(
        const ggml_cgraph *                    cgraph,
        int                                    node_idx,
        ggml_cuda_top_k_top_p_sample_match &   match) {
    const ggml_tensor * top_k = cgraph->nodes[node_idx];
    if (top_k->op != GGML_OP_TOP_K || top_k->type != GGML_TYPE_I32 ||
            top_k->ne[1] != 1 || top_k->ne[2] != 1 || top_k->ne[3] != 1) {
        return false;
    }
    const int k = (int) top_k->ne[0];
    if (k < 1 || k > 64) {
        return false;
    }
    const ggml_tensor * logits = top_k->src[0];
    if (!logits || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
            logits->ne[1] != 1 || logits->ne[2] != 1 || logits->ne[3] != 1) {
        return false;
    }
    const int n_vocab = (int) logits->ne[0];
    if (n_vocab <= k || n_vocab > 8192) {
        return false;
    }

    int end_idx = -1;
    const int max_end = std::min(node_idx + 45, cgraph->n_nodes - 1);
    for (int p = node_idx + 1; p <= max_end; ++p) {
        const ggml_tensor * cand_dst = cgraph->nodes[p];
        if (cand_dst->op == GGML_OP_GET_ROWS && cand_dst->type == GGML_TYPE_I32 &&
                cand_dst->ne[0] == 1 && cand_dst->ne[1] == 1 && cand_dst->ne[2] == 1 && cand_dst->ne[3] == 1 &&
                cand_dst->src[0] && cand_dst->src[0]->op == GGML_OP_RESHAPE &&
                cand_dst->src[1] && cand_dst->src[1]->op == GGML_OP_CPY && cand_dst->src[1]->type == GGML_TYPE_I32) {
            const ggml_tensor * cand_1d = cand_dst->src[0]->src[0];
            if (cand_1d && cand_1d->op == GGML_OP_RESHAPE &&
                    cand_1d->src[0] && cand_1d->src[0]->op == GGML_OP_GET_ROWS &&
                    cand_1d->src[0]->src[0] && cand_1d->src[0]->src[0]->op == GGML_OP_RESHAPE &&
                    cand_1d->src[0]->src[0]->src[0] == top_k) {
                end_idx = p;
                break;
            }
        }
    }
    if (end_idx < 0) {
        return false;
    }

    ggml_tensor * final_gr = cgraph->nodes[end_idx];
    const ggml_tensor * cand_2d       = final_gr->src[0];
    const ggml_tensor * candidates    = cand_2d ? cand_2d->src[0] : nullptr;
    const ggml_tensor * cand_gr       = candidates ? candidates->src[0] : nullptr;
    const ggml_tensor * cand_top_k_2d = cand_gr ? cand_gr->src[0] : nullptr;
    const ggml_tensor * sorted_idx    = cand_gr ? cand_gr->src[1] : nullptr;
    if (!cand_2d || !candidates || !cand_gr || !cand_top_k_2d || !sorted_idx ||
            cand_top_k_2d->src[0] != top_k ||
            sorted_idx->op != GGML_OP_ARGSORT ||
            ggml_get_op_params_i32(sorted_idx, 0) != GGML_SORT_ORDER_DESC) {
        return false;
    }

    const ggml_tensor * cur_topk_1d = sorted_idx->src[0];
    const ggml_tensor * topk_gr     = (cur_topk_1d && cur_topk_1d->op == GGML_OP_RESHAPE) ? cur_topk_1d->src[0] : nullptr;
    const ggml_tensor * logits_2d   = (topk_gr && topk_gr->op == GGML_OP_GET_ROWS && topk_gr->src[1] == top_k) ? topk_gr->src[0] : nullptr;
    if (!logits_2d || logits_2d->op != GGML_OP_RESHAPE || logits_2d->src[0] != logits ||
            logits_2d->ne[0] != 1 || logits_2d->ne[1] != n_vocab) {
        return false;
    }

    const ggml_tensor * idx_cast   = final_gr->src[1];
    const ggml_tensor * idx_clamp  = (idx_cast && idx_cast->op == GGML_OP_CPY && idx_cast->type == GGML_TYPE_I32) ? idx_cast->src[0] : nullptr;
    const ggml_tensor * idx_raw    = (idx_clamp && idx_clamp->op == GGML_OP_CLAMP) ? idx_clamp->src[0] : nullptr;
    const ggml_tensor * idxf_d     = (idx_raw && idx_raw->op == GGML_OP_SCALE) ? idx_raw->src[0] : nullptr;
    const ggml_tensor * cross_mask = (idxf_d && idxf_d->op == GGML_OP_SUM) ? idxf_d->src[0] : nullptr;
    const ggml_tensor * diff       = (cross_mask && cross_mask->op == GGML_OP_UNARY && ggml_get_unary_op(cross_mask) == GGML_UNARY_OP_STEP) ? cross_mask->src[0] : nullptr;
    if (!diff || diff->op != GGML_OP_SUB) {
        return false;
    }
    if (ggml_get_op_params_f32(idx_clamp, 0) != 0.0f || ggml_get_op_params_f32(idx_clamp, 1) != (float) (k - 1) ||
            ggml_get_op_params_f32(idx_raw, 0) != -1.0f || ggml_get_op_params_f32(idx_raw, 1) != (float) k) {
        return false;
    }

    const ggml_tensor * cumsum_d  = diff->src[0];
    const ggml_tensor * rand_view = diff->src[1];
    if (!rand_view || rand_view->type != GGML_TYPE_F32 || !ggml_is_contiguous(rand_view) || ggml_nelements(rand_view) != 1) {
        return false;
    }
    const ggml_tensor * probs_d  = (cumsum_d && cumsum_d->op == GGML_OP_CUMSUM) ? cumsum_d->src[0] : nullptr;
    const ggml_tensor * cur_topp = (probs_d && probs_d->op == GGML_OP_SOFT_MAX && probs_d->src[1] == nullptr &&
                                    ggml_get_op_params_f32(probs_d, 0) == 1.0f) ? probs_d->src[0] : nullptr;
    if (!cur_topp || cur_topp->op != GGML_OP_ADD) {
        return false;
    }

    const ggml_tensor * sorted_logits = cur_topp->src[0];
    const ggml_tensor * bias          = cur_topp->src[1];
    const ggml_tensor * sl_gr         = (sorted_logits && sorted_logits->op == GGML_OP_RESHAPE) ? sorted_logits->src[0] : nullptr;
    const ggml_tensor * sl_2d         = (sl_gr && sl_gr->op == GGML_OP_GET_ROWS && sl_gr->src[1] == sorted_idx) ? sl_gr->src[0] : nullptr;
    if (!sl_2d || sl_2d->op != GGML_OP_RESHAPE || sl_2d->src[0] != cur_topk_1d) {
        return false;
    }

    const ggml_tensor * keep_mask_1    = (bias && bias->op == GGML_OP_LOG) ? bias->src[0] : nullptr;
    const ggml_tensor * keep_mask_2d_1 = (keep_mask_1 && keep_mask_1->op == GGML_OP_RESHAPE) ? keep_mask_1->src[0] : nullptr;
    if (!keep_mask_2d_1 || keep_mask_2d_1->op != GGML_OP_SET_ROWS) {
        return false;
    }
    const ggml_tensor * ones           = keep_mask_2d_1->src[0];
    const ggml_tensor * cast_p         = keep_mask_2d_1->src[1];
    const ggml_tensor * keep_mask_2d_0 = keep_mask_2d_1->src[2];
    const ggml_tensor * idxf_clamp_p   = (ones && ones->op == GGML_OP_SCALE &&
                                          ggml_get_op_params_f32(ones, 0) == 0.0f &&
                                          ggml_get_op_params_f32(ones, 1) == 1.0f) ? ones->src[0] : nullptr;
    if (!cast_p || cast_p->op != GGML_OP_CPY || cast_p->type != GGML_TYPE_I32 || cast_p->src[0] != idxf_clamp_p ||
            !idxf_clamp_p || idxf_clamp_p->op != GGML_OP_CLAMP ||
            ggml_get_op_params_f32(idxf_clamp_p, 0) != 0.0f || ggml_get_op_params_f32(idxf_clamp_p, 1) != (float) (k - 1)) {
        return false;
    }
    const ggml_tensor * idxf_p      = idxf_clamp_p->src[0];
    const ggml_tensor * keep_mask_0 = (idxf_p && idxf_p->op == GGML_OP_SUM) ? idxf_p->src[0] : nullptr;
    if (!keep_mask_2d_0 || keep_mask_2d_0->op != GGML_OP_RESHAPE || keep_mask_2d_0->src[0] != keep_mask_0) {
        return false;
    }
    const ggml_tensor * cdf_scaled = (keep_mask_0 && keep_mask_0->op == GGML_OP_UNARY && ggml_get_unary_op(keep_mask_0) == GGML_UNARY_OP_STEP) ? keep_mask_0->src[0] : nullptr;
    const ggml_tensor * cdf_p      = (cdf_scaled && cdf_scaled->op == GGML_OP_SCALE && ggml_get_op_params_f32(cdf_scaled, 0) == -1.0f) ? cdf_scaled->src[0] : nullptr;
    const float top_p = cdf_scaled ? ggml_get_op_params_f32(cdf_scaled, 1) : 1.0f;
    if (!(top_p > 0.0f && top_p < 1.0f)) {
        return false;
    }
    const ggml_tensor * probs_p = (cdf_p && cdf_p->op == GGML_OP_CUMSUM) ? cdf_p->src[0] : nullptr;
    if (!probs_p || probs_p->op != GGML_OP_SOFT_MAX || probs_p->src[0] != sorted_logits ||
            probs_p->src[1] != nullptr || ggml_get_op_params_f32(probs_p, 0) != 1.0f) {
        return false;
    }

    const ggml_tensor * expected[] = {
        top_k, logits_2d, topk_gr, cur_topk_1d, sorted_idx, cand_top_k_2d, cand_gr, candidates, cand_2d,
        sl_2d, sl_gr, sorted_logits, probs_p, cdf_p, cdf_scaled, keep_mask_0, idxf_p, idxf_clamp_p, ones,
        cast_p, keep_mask_2d_0, keep_mask_2d_1, keep_mask_1, bias, cur_topp, probs_d, cumsum_d,
        rand_view, diff, cross_mask, idxf_d, idx_raw, idx_clamp, idx_cast, final_gr
    };

    for (int p = node_idx; p <= end_idx; ++p) {
        const ggml_tensor * t = cgraph->nodes[p];
        bool in_expected = false;
        for (const ggml_tensor * e : expected) {
            if (t == e) {
                in_expected = true;
                break;
            }
        }
        if (!in_expected) {
            bool is_sched_src_view = (t->op == GGML_OP_VIEW && t->type == GGML_TYPE_F32 && ggml_nelements(t) == 1 &&
                                      (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
            if (is_sched_src_view) {
                for (int q = 0; q < cgraph->n_nodes; ++q) {
                    const ggml_tensor * outer = cgraph->nodes[q];
                    if (outer->view_src == t && q != p) {
                        is_sched_src_view = false;
                        break;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        if (outer->src[s] == t) {
                            is_sched_src_view = false;
                            break;
                        }
                    }
                    if (!is_sched_src_view) {
                        break;
                    }
                }
            }
            if (!is_sched_src_view) {
                return false;
            }
        }
        if (t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW && (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (p < end_idx) {
            if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
                return false;
            }
            for (int q = 0; q < cgraph->n_nodes; ++q) {
                if (q >= node_idx && q <= end_idx) {
                    continue;
                }
                const ggml_tensor * outer = cgraph->nodes[q];
                if (outer->view_src == t) {
                    return false;
                }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (outer->src[s] == t) {
                        return false;
                    }
                }
            }
        }
    }

    for (const ggml_tensor * e : expected) {
        bool found = false;
        for (int p = node_idx; p <= end_idx; ++p) {
            if (cgraph->nodes[p] == e) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (e == rand_view) {
                continue;
            }
            if (e == logits_2d) {
                for (int q = 0; q < cgraph->n_nodes; ++q) {
                    const ggml_tensor * outer = cgraph->nodes[q];
                    if (outer != topk_gr && outer->view_src == e) {
                        return false;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        if (outer->src[s] == e && outer != topk_gr) {
                            return false;
                        }
                    }
                }
                continue;
            }
            return false;
        }
    }

    match.logits     = logits;
    match.rand_val   = rand_view;
    match.logits_2d  = logits_2d;
    match.dst        = final_gr;
    match.k          = k;
    match.top_p      = top_p;
    match.node_count = end_idx - node_idx + 1;
    return true;
}

struct ggml_cuda_ffn_down_shared_match {
    ggml_tensor *       routed_down = nullptr;
    ggml_tensor *       shared_down = nullptr;
    int                 node_count  = 0;
};

static bool ggml_cuda_match_ffn_down_shared(
        const ggml_cgraph * cgraph, int i, ggml_cuda_ffn_down_shared_match & m, bool check_ptrs) {
    if (i + 1 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * routed = cgraph->nodes[i];
    if (routed->op != GGML_OP_MUL_MAT_ID || routed->type != GGML_TYPE_F32 ||
            !routed->src[0] || !routed->src[1] || !routed->src[2] ||
            (routed->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const ggml_type wtype = routed->src[0]->type;
    if (wtype != GGML_TYPE_Q4_K && wtype != GGML_TYPE_Q6_K) {
        return false;
    }
    if (routed->src[0]->ne[0] != 512 || routed->src[1]->type != GGML_TYPE_F32) {
        return false;
    }

    int k = i + 1;
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        if ((cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
        k++;
    }
    if (k >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * shared = cgraph->nodes[k];
    if (shared->op != GGML_OP_MUL_MAT || shared->type != GGML_TYPE_F32 ||
            !shared->src[0] || !shared->src[1] ||
            shared->src[0]->type != wtype ||
            shared->src[0]->ne[0] != 512 ||
            shared->src[0]->ne[1] != routed->src[0]->ne[1] ||
            shared->src[1]->type != GGML_TYPE_F32) {
        return false;
    }

    // Ensure shared does not depend on any node in [i, k) and intermediate views are not externally used
    for (int t_idx = i + 1; t_idx < k; ++t_idx) {
        const ggml_tensor * v = cgraph->nodes[t_idx];
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            if (j >= i && j < k) continue;
            const ggml_tensor * ext = cgraph->nodes[j];
            if (ext->view_src == v) return false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (ext->src[s] == v) return false;
            }
        }
    }

    if (check_ptrs) {
        // Check single-token decode shape and alignment
        if (routed->src[1]->ne[2] != 1 || routed->src[1]->ne[3] != 1 ||
                routed->ne[2] != 1 || routed->ne[3] != 1 ||
                routed->src[2]->type != GGML_TYPE_I32 || routed->src[2]->ne[1] != 1 ||
                shared->src[1]->ne[1] != 1 || shared->src[1]->ne[2] != 1 || shared->src[1]->ne[3] != 1 ||
                shared->ne[1] != 1 || shared->ne[2] != 1 || shared->ne[3] != 1) {
            return false;
        }
        if (!ggml_is_contiguous(routed->src[0]) || !ggml_is_contiguous(shared->src[0]) ||
                !ggml_is_contiguous(routed->src[1]) || !ggml_is_contiguous(shared->src[1]) ||
                !ggml_is_contiguous(routed->src[2]) ||
                !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
            return false;
        }
        if (((uintptr_t) routed->src[1]->data & 15) != 0 || ((uintptr_t) shared->src[1]->data & 15) != 0) {
            return false;
        }
        if (wtype == GGML_TYPE_Q4_K &&
                (((uintptr_t) routed->src[0]->data & 15) != 0 || ((uintptr_t) shared->src[0]->data & 15) != 0)) {
            return false;
        }
        if (ggml_cuda_tensors_overlap(routed, shared) ||
                ggml_cuda_tensors_overlap(routed, shared->src[1]) ||
                ggml_cuda_tensors_overlap(routed, routed->src[1]) ||
                ggml_cuda_tensors_overlap(routed, routed->src[2]) ||
                ggml_cuda_tensors_overlap(shared, routed->src[1]) ||
                ggml_cuda_tensors_overlap(shared, routed->src[2]) ||
                ggml_cuda_tensors_overlap(shared, shared->src[1])) {
            return false;
        }
    }

    m.routed_down = routed;
    m.shared_down = shared;
    m.node_count  = k - i + 1;
    return true;
}

struct ggml_cuda_tower_eh_concat_match {
    const ggml_tensor * embd_w  = nullptr;
    const ggml_tensor * code    = nullptr;
    const ggml_tensor * mask    = nullptr;
    const ggml_tensor * extra   = nullptr;
    const ggml_tensor * enorm_w = nullptr;
    const ggml_tensor * x       = nullptr;
    const ggml_tensor * hnorm_w = nullptr;
    ggml_tensor *       dst     = nullptr;
    float               eps_e   = 0.0f;
    float               eps_h   = 0.0f;
    int                 node_count = 0;
};

static bool ggml_cuda_match_tower_eh_concat(
        const ggml_cgraph * cgraph, int i, ggml_cuda_tower_eh_concat_match & m, bool check_ptrs) {
    if (i + 6 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * n_gr = cgraph->nodes[i];
    if (n_gr->op != GGML_OP_GET_ROWS || !n_gr->src[0] || !n_gr->src[1] ||
            n_gr->src[0]->type != GGML_TYPE_F16 || n_gr->type != GGML_TYPE_F32 ||
            n_gr->src[0]->ne[0] != 2048 || n_gr->src[1]->type != GGML_TYPE_I32 ||
            (n_gr->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const ggml_tensor * n_mask_mul = cgraph->nodes[i + 1];
    if (n_mask_mul->op != GGML_OP_MUL || n_mask_mul->type != GGML_TYPE_F32 ||
            (n_mask_mul->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const ggml_tensor * mask = (n_mask_mul->src[0] == n_gr) ? n_mask_mul->src[1] :
                               (n_mask_mul->src[1] == n_gr) ? n_mask_mul->src[0] : nullptr;
    if (!mask || mask->type != GGML_TYPE_F32 || mask->ne[0] != 1) {
        return false;
    }

    int k = i + 2;
    const ggml_tensor * cur_e = n_mask_mul;
    const ggml_tensor * extra = nullptr;
    if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_ADD) {
        const ggml_tensor * n_add = cgraph->nodes[k];
        if ((n_add->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || n_add->type != GGML_TYPE_F32) {
            return false;
        }
        extra = (n_add->src[0] == n_mask_mul) ? n_add->src[1] :
                (n_add->src[1] == n_mask_mul) ? n_add->src[0] : nullptr;
        if (!extra || extra->type != GGML_TYPE_F32 || extra->ne[0] != 2048) {
            return false;
        }
        cur_e = n_add;
        k++;
    }

    if (k + 4 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * n_rms_e = cgraph->nodes[k + 0];
    const ggml_tensor * n_mul_e = cgraph->nodes[k + 1];
    const ggml_tensor * n_rms_h = cgraph->nodes[k + 2];
    const ggml_tensor * n_mul_h = cgraph->nodes[k + 3];
    ggml_tensor *       n_cat   = cgraph->nodes[k + 4];

    if (n_rms_e->op != GGML_OP_RMS_NORM || n_rms_e->src[0] != cur_e || n_rms_e->type != GGML_TYPE_F32 ||
            n_mul_e->op != GGML_OP_MUL || n_mul_e->type != GGML_TYPE_F32 ||
            n_rms_h->op != GGML_OP_RMS_NORM || n_rms_h->type != GGML_TYPE_F32 ||
            n_mul_h->op != GGML_OP_MUL || n_mul_h->type != GGML_TYPE_F32 ||
            n_cat->op   != GGML_OP_CONCAT || n_cat->type != GGML_TYPE_F32) {
        return false;
    }
    if ((n_rms_e->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            (n_mul_e->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            (n_rms_h->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
            (n_mul_h->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const ggml_tensor * enorm_w = (n_mul_e->src[0] == n_rms_e) ? n_mul_e->src[1] :
                                  (n_mul_e->src[1] == n_rms_e) ? n_mul_e->src[0] : nullptr;
    const ggml_tensor * hnorm_w = (n_mul_h->src[0] == n_rms_h) ? n_mul_h->src[1] :
                                  (n_mul_h->src[1] == n_rms_h) ? n_mul_h->src[0] : nullptr;
    const ggml_tensor * x       = n_rms_h->src[0];
    if (!enorm_w || !hnorm_w || !x ||
            enorm_w->type != GGML_TYPE_F32 || enorm_w->ne[0] != 2048 || !ggml_is_vector(enorm_w) ||
            hnorm_w->type != GGML_TYPE_F32 || hnorm_w->ne[0] != 2048 || !ggml_is_vector(hnorm_w) ||
            x->type != GGML_TYPE_F32 || x->ne[0] != 2048) {
        return false;
    }
    if (n_cat->src[0] != n_mul_e || n_cat->src[1] != n_mul_h || ggml_get_op_params_i32(n_cat, 0) != 0) {
        return false;
    }

    // Ensure no internal node in [i, k+3] is referenced outside [i, k+4]
    const int cat_idx = k + 4;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        if (j >= i && j <= cat_idx) continue;
        const ggml_tensor * ext = cgraph->nodes[j];
        for (int p = i; p < cat_idx; ++p) {
            if (ext->view_src == cgraph->nodes[p]) return false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (ext->src[s] == cgraph->nodes[p]) return false;
            }
        }
    }

    if (check_ptrs) {
        if (n_gr->src[1]->ne[0] != 1 || n_gr->src[1]->ne[1] != 1 || n_gr->src[1]->ne[2] != 1 || n_gr->src[1]->ne[3] != 1 ||
                mask->ne[1] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
                x->ne[1] != 1 || x->ne[2] != 1 || x->ne[3] != 1 ||
                (extra && (extra->ne[1] != 1 || extra->ne[2] != 1 || extra->ne[3] != 1 || !ggml_is_contiguous(extra))) ||
                !ggml_is_contiguous(n_gr->src[0]) || !ggml_is_contiguous(x) ||
                !ggml_is_contiguous(enorm_w) || !ggml_is_contiguous(hnorm_w) || !ggml_is_contiguous(n_cat)) {
            return false;
        }
        if (((uintptr_t) n_gr->src[0]->data & 7) != 0 ||
                ((uintptr_t) x->data & 15) != 0 ||
                ((uintptr_t) enorm_w->data & 15) != 0 ||
                ((uintptr_t) hnorm_w->data & 15) != 0 ||
                ((uintptr_t) n_cat->data & 15) != 0 ||
                (extra && ((uintptr_t) extra->data & 15) != 0)) {
            return false;
        }
        if (ggml_cuda_tensors_overlap(n_cat, n_gr->src[0]) ||
                ggml_cuda_tensors_overlap(n_cat, n_gr->src[1]) ||
                ggml_cuda_tensors_overlap(n_cat, mask) ||
                (extra && ggml_cuda_tensors_overlap(n_cat, extra)) ||
                ggml_cuda_tensors_overlap(n_cat, enorm_w) ||
                ggml_cuda_tensors_overlap(n_cat, x) ||
                ggml_cuda_tensors_overlap(n_cat, hnorm_w)) {
            return false;
        }
    }

    float eps_e = 0.0f, eps_h = 0.0f;
    memcpy(&eps_e, n_rms_e->op_params, sizeof(float));
    memcpy(&eps_h, n_rms_h->op_params, sizeof(float));
    if (eps_e < 0.0f || eps_h < 0.0f) {
        return false;
    }

    m.embd_w     = n_gr->src[0];
    m.code       = n_gr->src[1];
    m.mask       = mask;
    m.extra      = extra;
    m.enorm_w    = enorm_w;
    m.x          = x;
    m.hnorm_w    = hnorm_w;
    m.dst        = n_cat;
    m.eps_e      = eps_e;
    m.eps_h      = eps_h;
    m.node_count = cat_idx - i + 1;
    return true;
}

static bool ggml_cuda_has_quantized_mm_consumer(
        const ggml_cgraph * cgraph, int start_idx, const ggml_tensor * root) {
    const ggml_tensor * aliases[16];
    int n_alias = 0;
    aliases[n_alias++] = root;
    const auto is_alias = [&](const ggml_tensor * t) {
        for (int a = 0; a < n_alias; ++a) {
            if (aliases[a] == t) return true;
        }
        return false;
    };
    const int limit = std::min(start_idx + 64, cgraph->n_nodes);
    for (int k = start_idx; k < limit; ++k) {
        const ggml_tensor * cn = cgraph->nodes[k];
        if ((cn->op == GGML_OP_MUL_MAT || cn->op == GGML_OP_MUL_MAT_ID) &&
                cn->src[0] && cn->src[1] &&
                ggml_is_quantized(cn->src[0]->type) && is_alias(cn->src[1])) {
            return true;
        }
        if ((cn->op == GGML_OP_VIEW || cn->op == GGML_OP_RESHAPE ||
                cn->op == GGML_OP_TRANSPOSE || cn->op == GGML_OP_PERMUTE) &&
                cn->src[0] && is_alias(cn->src[0]) && n_alias < 16) {
            aliases[n_alias++] = cn;
        }
    }
    return false;
}

int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (int n = ggml_cuda_try_fuse_inkling2(cuda_ctx, cgraph, i)) {
        return n;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID &&
            ggml_cuda_info().devices[cuda_ctx->device].cc == GGML_CUDA_CC_VOLTA) {
        ggml_cuda_ffn_down_shared_match fd_match;
        if (ggml_cuda_match_ffn_down_shared(cgraph, i, fd_match, true)) {
            ggml_cuda_mul_mat_vec_q_k512_f32_shared(
                *cuda_ctx,
                fd_match.routed_down->src[0],
                fd_match.routed_down->src[1],
                fd_match.routed_down->src[2],
                fd_match.routed_down,
                fd_match.shared_down->src[0],
                fd_match.shared_down->src[1],
                fd_match.shared_down);
            return fd_match.node_count - 1;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_reduce_shared_add_match moe_sa_match;
        if (ggml_cuda_match_moe_reduce_shared_add(cgraph, i, moe_sa_match, true)) {
            ggml_cuda_op_moe_reduce_shared_add(
                *cuda_ctx,
                moe_sa_match.experts,
                moe_sa_match.weights,
                moe_sa_match.shared_out,
                moe_sa_match.shared_gate_raw,
                moe_sa_match.residual,
                moe_sa_match.dst);
            return moe_sa_match.node_count - 1;
        }

        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // get_rows(embd, code) -> mul(mask) -> [add(extra)] -> rms_norm -> mul(enorm) -> rms_norm(x) -> mul(hnorm) -> concat
    if (node->op == GGML_OP_GET_ROWS) {
        ggml_cuda_tower_eh_concat_match eh_match;
        if (ggml_cuda_match_tower_eh_concat(cgraph, i, eh_match, true)) {
            ggml_cuda_op_tower_eh_concat(
                *cuda_ctx,
                eh_match.embd_w,
                eh_match.code,
                eh_match.mask,
                eh_match.extra,
                eh_match.enorm_w,
                eh_match.x,
                eh_match.hnorm_w,
                eh_match.dst,
                eh_match.eps_e,
                eh_match.eps_h);
            return eh_match.node_count - 1;
        }
    }

    // get_rows -> concat -> cpy + ssm_conv -> silu -> Q/K L2-norm
    if (node->op == GGML_OP_GET_ROWS) {
        ggml_cuda_gdn_conv_match conv_match;
        if (ggml_cuda_match_gdn_conv_silu_l2norm(cgraph, i, conv_match, true)) {
            ggml_cuda_op_gdn_conv_silu_l2norm(
                *cuda_ctx,
                conv_match.conv_states_all,
                conv_match.state_ids,
                conv_match.qkv_mixed,
                conv_match.conv_kernel,
                conv_match.state_cell_dst,
                conv_match.v_silu_dst,
                conv_match.q_norm_dst,
                conv_match.k_norm_dst,
                conv_match.head_k_dim,
                conv_match.num_k_heads,
                conv_match.eps_over_n,
                conv_match.inv_sqrt_n);
            return conv_match.node_count - 1;
        }
    }

    // get_rows -> [views] -> gated_delta_net -> [views] -> cpy
    if (node->op == GGML_OP_GET_ROWS) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        int gdn_idx = -1;
        const int nodes_to_skip = ggml_cuda_try_gdn_rs_cache_fusion(cgraph, i, fused_state_cpy, gdn_idx);
        if (nodes_to_skip > 0) {
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, cgraph->nodes[gdn_idx], fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * denom   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                ops.push_back(GGML_OP_RESHAPE);
                if (args.prob_bias) {
                    ops.push_back(GGML_OP_ADD);
                    bias = cgraph->nodes[i + ops.size() - 1]->src[1];
                }
                if (args.n_groups > 0) {
                    for (const auto & p : topk_moe_group_pattern) {
                        ops.push_back(p.op);
                    }
                }
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                out_nodes[0] = i + ops.size() - 2;
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, args.norm_bias ? GGML_OP_SCALE : GGML_OP_CLAMP,
                                            GGML_OP_DIV, GGML_OP_RESHAPE });
                    denom = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, denom, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, denom, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = ggml_cuda_get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = ggml_cuda_get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n, gate_n)) {
                    ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (!with_bias && ggml_cuda_should_fuse_mul_mat_q(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
                    fusion_data.prec_src1 = ggml_prec_get_src(up_n, 1);
                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n, gate_n)) {
                    ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (!with_bias && ggml_cuda_should_fuse_mul_mat_q(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
                    fusion_data.prec_src1 = ggml_prec_get_src(up_n, 1);
                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n, gate_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up, gate)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_q(up, gate, glu)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);
                fusion_data.prec_src1 = ggml_prec_get_src(up, 1);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = ggml_cuda_get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }

            if (op == GGML_OP_MUL_MAT && !with_bias && src0->type == GGML_TYPE_F8_E4M3 &&
                    ggml_get_op_params_i32(mm_node, 3) != GGML_PREC_F32 &&
                    ggml_cuda_mul_mat_fp8(*cuda_ctx, src0, src1, scale_node, scale)) {
                return 1;
            }

            if (!with_bias && src0->type == GGML_TYPE_NVFP4 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16) &&
                    mm_node->type == GGML_TYPE_F32) {
                const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
                const bool is_id = op == GGML_OP_MUL_MAT_ID;
                const int64_t ncols = is_id ? src1->ne[2] : src1->ne[1];
                const bool use_mmvq = is_id ? ncols <= get_mmvq_mmid_max_batch(src0->type, cc) :
                                            ggml_cuda_should_use_mmvq(src0->type, cc, ncols);
                const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                               ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                               src0->view_src;
                if (!use_mmvq && !bad_padding_clear &&
                        ggml_cuda_should_use_mmq(src0->type, cc, ncols, /*n_experts =*/ is_id ? src0->ne[2] : 0)) {
                    fusion_data.prec_src1 = ggml_prec_get_src(mm_node, 1);
                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                    return n_ops - 1;
                }
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_norm(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_norm(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_gdn_norm_gated_match ng_match;
        if (ggml_cuda_match_gdn_norm_gated(cgraph, i, ng_match, true)) {
            ggml_cuda_op_rms_norm_mul_silu_fused(
                *cuda_ctx, ng_match.rms_node, ng_match.norm_mul_node, ng_match.silu_node, ng_match.out_mul_node);
            return ng_match.node_count - 1;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        const bool write_q8_1 = ggml_cuda_has_quantized_mm_consumer(cgraph, i + 2, cgraph->nodes[i + 1]);
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1], write_q8_1);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID) {
        ggml_cuda_gdn_alpha_beta_match ab_match;
        if (ggml_cuda_match_gdn_alpha_beta(cgraph, i, ab_match, true)) {
            ggml_cuda_op_gdn_alpha_beta(
                *cuda_ctx, ab_match.beta_in, ab_match.alpha_in, ab_match.ssm_dt, ab_match.ssm_a,
                ab_match.beta_out, ab_match.gate_out);
            return ab_match.node_count - 1;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    if (node->op == GGML_OP_TOP_K) {
        ggml_cuda_top_k_top_p_sample_match match;
        if (ggml_cuda_match_top_k_top_p_sample(cgraph, i, match)) {
            const int out_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &out_idx, 1)) {
                ggml_cuda_op_top_k_top_p_sample_fused(*cuda_ctx, match.logits, match.rand_val, match.dst, match.k, match.top_p);
                return match.node_count - 1;
            }
        }
    }

    return 0;
}

void ggml_cuda_fusion_add_alloc_deps(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    ggml_cuda_set_device(cuda_ctx->device);
    for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
            if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                break;
            }
            if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                continue;
            }
            // Group both outputs before allocation so the shared result cannot alias intervening nodes.
            std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
            ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
            params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
            params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
            i += 5;
            break;
        }
    }
    // Move ffn_down_shexp (MUL_MAT consuming shared GLU) right before moe_weighted_reduction so epilogue mega-fusion is contiguous
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op != GGML_OP_MUL) {
            continue;
        }
        ggml_cuda_moe_weighted_reduction_match red_m;
        if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, red_m) ||
                red_m.expert_scale != nullptr || red_m.experts->ne[2] * red_m.experts->ne[3] != 1) {
            continue;
        }
        const int shexp_idx = i + red_m.node_count;
        if (shexp_idx < cgraph->n_nodes && cgraph->nodes[shexp_idx]->op == GGML_OP_MUL_MAT) {
            const ggml_tensor * mm = cgraph->nodes[shexp_idx];
            bool src_before_i = true;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (!mm->src[s]) continue;
                for (int k = i; k < shexp_idx; ++k) {
                    if (cgraph->nodes[k] == mm->src[s]) {
                        src_before_i = false;
                        break;
                    }
                }
            }
            if (src_before_i) {
                std::rotate(cgraph->nodes + i, cgraph->nodes + shexp_idx, cgraph->nodes + shexp_idx + 1);
                i++; // red_m now starts at i + 1
            }
        }
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
            ggml_cuda_ffn_down_shared_match fd_m;
            if (ggml_cuda_match_ffn_down_shared(cgraph, i, fd_m, false)) {
                ggml_tensor * last = cgraph->nodes[i + fd_m.node_count - 1];
                params->add_alloc_dep(params->user_data, fd_m.routed_down, last);
                params->add_alloc_dep(params->user_data, fd_m.routed_down->src[1], last);
                params->add_alloc_dep(params->user_data, fd_m.routed_down->src[2], last);
                params->add_alloc_dep(params->user_data, fd_m.shared_down->src[1], last);
                i += fd_m.node_count - 1;
                continue;
            }
        }
        if (cgraph->nodes[i]->op == GGML_OP_GET_ROWS) {
            ggml_cuda_tower_eh_concat_match eh_m;
            if (ggml_cuda_match_tower_eh_concat(cgraph, i, eh_m, false)) {
                add_alloc_deps(i, i + eh_m.node_count - 1);
                i += eh_m.node_count - 1;
                continue;
            }
        }
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op == GGML_OP_MUL) {
            ggml_cuda_moe_reduce_shared_add_match moe_sa;
            if (ggml_cuda_match_moe_reduce_shared_add(cgraph, i, moe_sa, false)) {
                ggml_tensor * last = cgraph->nodes[i + moe_sa.node_count - 1];
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(moe_sa.experts), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(moe_sa.weights), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(moe_sa.shared_out), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(moe_sa.shared_gate_raw), last);
                if (moe_sa.residual) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(moe_sa.residual), last);
                }
                i += moe_sa.node_count - 1;
                continue;
            }
        }
        if (cgraph->nodes[i]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i]) == GGML_UNARY_OP_SIGMOID) {
            ggml_cuda_gdn_alpha_beta_match ab_match;
            if (ggml_cuda_match_gdn_alpha_beta(cgraph, i, ab_match, false)) {
                ggml_tensor * last = cgraph->nodes[i + ab_match.node_count - 1];
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(ab_match.beta_in), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(ab_match.alpha_in), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(ab_match.ssm_dt), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(ab_match.ssm_a), last);
                params->add_alloc_dep(params->user_data, ab_match.beta_out, last);
                params->add_alloc_dep(params->user_data, ab_match.gate_out, last);
            }
        }
        if (cgraph->nodes[i]->op == GGML_OP_RMS_NORM) {
            ggml_cuda_gdn_norm_gated_match ng_match;
            if (ggml_cuda_match_gdn_norm_gated(cgraph, i, ng_match, false)) {
                ggml_tensor * w = (ng_match.norm_mul_node->src[0] == ng_match.rms_node) ?
                                  ng_match.norm_mul_node->src[1] : ng_match.norm_mul_node->src[0];
                ggml_tensor * last = cgraph->nodes[i + ng_match.node_count - 1];
                params->add_alloc_dep(params->user_data, ng_match.rms_node->src[0], last);
                params->add_alloc_dep(params->user_data, w, last);
                params->add_alloc_dep(params->user_data, ng_match.silu_node->src[0], last);
                params->add_alloc_dep(params->user_data, ng_match.out_mul_node, last);
            }
        }
        if (cgraph->nodes[i]->op == GGML_OP_GET_ROWS) {
            ggml_cuda_gdn_conv_match conv_match;
            if (ggml_cuda_match_gdn_conv_silu_l2norm(cgraph, i, conv_match, false)) {
                ggml_tensor * last = cgraph->nodes[i + conv_match.node_count - 1];
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(conv_match.qkv_mixed), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(conv_match.state_ids), last);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(conv_match.conv_kernel), last);
                params->add_alloc_dep(params->user_data, conv_match.v_silu_dst, last);
                params->add_alloc_dep(params->user_data, conv_match.q_norm_dst, last);
                params->add_alloc_dep(params->user_data, conv_match.k_norm_dst, last);
                params->add_alloc_dep(params->user_data, conv_match.state_cell_dst, last);
            }
        }
        // Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
        if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];
            const bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);
            if (ok && ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, cuda_ctx->device)) {
                add_alloc_deps(i, i + 2);
                i += 2;
                continue;
            }
        }
        // Same with the NVFP4 scales: MUL_MAT(gate), MUL(gate scale), MUL_MAT(up), MUL(up scale), GLU.
        if (i + 4 < cgraph->n_nodes && cgraph->nodes[i + 4]->op == GGML_OP_GLU) {
            ggml_tensor * glu        = cgraph->nodes[i + 4];
            ggml_tensor * gate_scale = cgraph->nodes[i + 1];
            ggml_tensor * up_scale   = cgraph->nodes[i + 3];
            if (glu->src[0] == gate_scale && glu->src[1] == up_scale &&
                    ggml_cuda_mul_mat_q_fusion_matches(cgraph->nodes[i + 2], cgraph->nodes[i], glu, cuda_ctx->device, up_scale, gate_scale) &&
                    ggml_cuda_get_mul_mat_scale(gate_scale, cgraph->nodes[i]) && ggml_cuda_get_mul_mat_scale(up_scale, cgraph->nodes[i + 2])) {
                add_alloc_deps(i, i + 4);
                i += 4;
                continue;
            }
        }
        // Same with per-expert NVFP4 scales: MUL_MAT_ID, RESHAPE, REPEAT, GET_ROWS, MUL for gate and for up, GLU.
        if (i + 10 < cgraph->n_nodes && cgraph->nodes[i + 10]->op == GGML_OP_GLU) {
            ggml_tensor * glu        = cgraph->nodes[i + 10];
            ggml_tensor * gate_scale = cgraph->nodes[i + 4];
            ggml_tensor * up_scale   = cgraph->nodes[i + 9];
            if (glu->src[0] == gate_scale && glu->src[1] == up_scale &&
                    ggml_cuda_mul_mat_q_fusion_matches(cgraph->nodes[i + 5], cgraph->nodes[i], glu, cuda_ctx->device, up_scale, gate_scale) &&
                    ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], gate_scale, cgraph->nodes[i]) &&
                    ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 6], cgraph->nodes[i + 7], cgraph->nodes[i + 8], up_scale, cgraph->nodes[i + 5])) {
                add_alloc_deps(i, i + 10);
                i += 10;
                continue;
            }
        }

        if (cgraph->nodes[i]->op == GGML_OP_TOP_K) {
            ggml_cuda_top_k_top_p_sample_match match;
            if (ggml_cuda_match_top_k_top_p_sample(cgraph, i, match)) {
                add_alloc_deps(i, i + match.node_count - 1);
                for (const ggml_tensor * dep = match.logits; dep; dep = dep->view_src) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(dep), match.dst);
                }
                for (const ggml_tensor * dep = match.rand_val; dep; dep = dep->view_src) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(dep), match.dst);
                }
                i += match.node_count - 1;
                continue;
            }
        }

        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }

        if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
            ggml_cuda_topk_moe_args args;
            const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
            std::vector<ggml_op>    ops;
            ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

            const ggml_tensor * node = cgraph->nodes[i];

            if (can_fuse) {
                const ggml_tensor * logits  = node->src[0];
                ggml_tensor *       weights = nullptr;
                ggml_tensor *       ids     = nullptr;

                if (!args.delayed_softmax) {
                    int out_nodes[2];  // nodes which can't be elided

                    if (args.sigmoid) {
                        ops.insert(ops.end(), { GGML_OP_UNARY });
                    } else if (args.sqrt_softplus) {
                        ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                    } else {
                        ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                    }
                    const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                    if (args.prob_bias) {
                        ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                GGML_OP_GET_ROWS });
                        out_nodes[0] = i_probs + 4;
                    } else {
                        ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                        out_nodes[0] = i_probs + 3;
                    }
                    ids = cgraph->nodes[out_nodes[0]];

                    if (args.norm) {
                        ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS,
                                                args.norm_bias ? GGML_OP_SCALE : GGML_OP_CLAMP, GGML_OP_DIV,
                                                GGML_OP_RESHAPE });
                    }
                    if (args.scale) {
                        ops.insert(ops.end(), { GGML_OP_SCALE });
                    }

                    weights      = cgraph->nodes[i + ops.size() - 1];
                    out_nodes[1] = i + ops.size() - 1;

                    if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                            ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                        add_alloc_deps(i, i + ops.size());
                        i += ops.size() - 1;
                    }
                }
            }
        }
    }
}
