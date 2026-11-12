#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm {

struct FlashAttentionParams {
    float* output;             // [batch, seq_len, num_heads, head_dim]
    const float* query;        // [batch, seq_len, num_heads, head_dim]
    const float* key;          // [batch, seq_len, num_kv_heads, head_dim]
    const float* value;        // [batch, seq_len, num_kv_heads, head_dim]
    int32_t batch;
    int32_t seq_len_q;
    int32_t seq_len_kv;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    float scale;
    bool is_causal;
    cudaStream_t stream;
};

// FlashAttention-3 forward pass (Blackwell-optimized).
void flash_attention_forward(const FlashAttentionParams& params);

}  // namespace vllm
