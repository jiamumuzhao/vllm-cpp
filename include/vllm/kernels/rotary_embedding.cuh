#pragma once

namespace vllm {
namespace kernels {

// RoPE: Rotary Position Embedding for query/key tensors.
// q/k shape: [num_tokens, num_heads, head_dim]
// cos_sin: precomputed [max_seq_len, head_dim/2] interleaved cos/sin values.
void rope_forward(float* q, float* k, const float* cos_sin,
                  int num_tokens, int num_heads, int num_kv_heads, int head_dim,
                  int start_pos, float theta = 10000.0f,
                  cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace vllm
