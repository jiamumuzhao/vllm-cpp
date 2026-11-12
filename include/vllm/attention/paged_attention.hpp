#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm {

struct PagedAttentionParams {
    float* output;                // [num_tokens, num_heads, head_dim]
    const float* query;           // [num_tokens, num_heads, head_dim]
    const float* key_cache;       // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* value_cache;     // [num_blocks, block_size, num_kv_heads, head_dim]
    const int32_t* block_tables;  // [num_seqs, max_blocks_per_seq]
    const int32_t* context_lens;  // [num_seqs]
    int32_t num_seqs;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t block_size;
    int32_t max_context_len;
    int32_t max_blocks_per_seq;
    float scale;                  // 1/sqrt(head_dim)
    cudaStream_t stream;
};

// PagedAttention forward pass.
void paged_attention_forward(const PagedAttentionParams& params);

}  // namespace vllm
