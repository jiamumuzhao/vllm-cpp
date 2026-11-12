#include "vllm/attention/paged_attention.hpp"

#include <cuda_runtime.h>

namespace vllm {

__global__ void paged_attention_kernel(
    float* out, const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* context_lens,
    int32_t num_seqs, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t block_size, int32_t max_context_len, int32_t max_blocks_per_seq, float scale) {

    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int ctx_len = min(context_lens[seq_idx], max_context_len);
    int num_blocks = (ctx_len + block_size - 1) / block_size;

    int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;

    float acc = 0.0f;
    float max_logit = -1e30f;

    // Dot-product attention with paged KV cache.
    for (int b = 0; b < num_blocks; b++) {
        int phys_block = block_tables[seq_idx * max_blocks_per_seq + b];
        int tokens_in_block = min(block_size, ctx_len - b * block_size);

        for (int t = 0; t < tokens_in_block; t++) {
            int kv_offset = (phys_block * block_size + t) * num_kv_heads * head_dim
                            + kv_head_idx * head_dim;

            float dot = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                dot += query[q_offset + d] * key_cache[kv_offset + d];
            }

            // Warp reduction for dot product.
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }

            if (tid == 0) {
                dot *= scale;
                float prev_max = max_logit;
                max_logit = fmaxf(max_logit, dot);
                acc = acc * expf(prev_max - max_logit) + expf(dot - max_logit);
            }
        }
    }

    // Weighted sum over values.
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        int phys_block = block_tables[seq_idx * max_blocks_per_seq + b];
        int tokens_in_block = min(block_size, ctx_len - b * block_size);

        for (int t = 0; t < tokens_in_block; t++) {
            int qk_offset = (phys_block * block_size + t) * num_kv_heads * head_dim
                            + kv_head_idx * head_dim;

            float dot = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                dot += query[q_offset + d] * key_cache[qk_offset + d];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }

            if (tid == 0) {
                float weight = expf(dot * scale - max_logit) / (acc + 1e-10f);
                sum += weight;
                for (int d = 0; d < head_dim; d++) {
                    out[q_offset + d] += weight * value_cache[qk_offset + d];
                }
            }
        }
    }
}

void paged_attention_forward(const PagedAttentionParams& params) {
    dim3 grid(params.num_seqs, params.num_heads);
    dim3 block(32);
    paged_attention_kernel<<<grid, block, 0, params.stream>>>(
        params.output, params.query, params.key_cache, params.value_cache,
        params.block_tables, params.context_lens,
        params.num_seqs, params.num_heads, params.num_kv_heads, params.head_dim,
        params.block_size, params.max_context_len, params.max_blocks_per_seq, params.scale);
}

}  // namespace vllm
