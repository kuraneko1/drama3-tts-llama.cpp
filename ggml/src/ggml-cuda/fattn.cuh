#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_can_fuse_gate(int device, const ggml_tensor * fa_node, const ggml_tensor * gate, const ggml_tensor * dst, bool check_ptrs);

void ggml_cuda_flash_attn_ext_gated(ggml_backend_cuda_context & ctx, ggml_tensor * fa_node, const ggml_tensor * gate, ggml_tensor * dst);
