#include "vllm/attention/flash_attention.hpp"

#include <cuda_runtime.h>

namespace vllm {

__global__ void flash_attention_kernel(
    float* out, const float* query, const float* key, const float* value,
    int32_t batch, int32_t seq_len_q, int32_t seq_len_kv,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
    float scale, bool is_causal) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_idx = blockIdx.z;

    if (b >= batch || h >= num_heads || q_idx >= seq_len_q) return;

    int kv_h = h % num_kv_heads;
    extern __shared__ float shared_qk[];

    // Load query into shared memory.
    int q_base = ((b * seq_len_q + q_idx) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        shared_qk[d] = query[q_base + d];
    }
    __syncthreads();

    float acc = 0.0f;
    float max_logit = -1e30f;

    int kv_end = is_causal ? q_idx + 1 : seq_len_kv;
    for (int kv_idx = 0; kv_idx < kv_end; kv_idx++) {
        int kv_base = ((b * seq_len_kv + kv_idx) * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_qk[d] * key[kv_base + d];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }
        if (threadIdx.x == 0) {
            dot *= scale;
            float prev_max = max_logit;
            max_logit = fmaxf(max_logit, dot);
            acc = acc * expf(prev_max - max_logit) + expf(dot - max_logit);
        }
    }

    int out_base = ((b * seq_len_q + q_idx) * num_heads + h) * head_dim;
    float sum = 0.0f;
    for (int kv_idx = 0; kv_idx < kv_end; kv_idx++) {
        int kv_base = ((b * seq_len_kv + kv_idx) * num_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_qk[d] * key[kv_base + d];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }
        if (threadIdx.x == 0) {
            float weight = expf(dot * scale - max_logit) / (acc + 1e-10f);
            for (int d = 0; d < head_dim; d++) {
                out[out_base + d] += weight * value[kv_base + d];
            }
        }
    }
}

void flash_attention_forward(const FlashAttentionParams& params) {
    dim3 grid(params.batch, params.num_heads, params.seq_len_q);
    dim3 block(32);
    size_t shared_bytes = params.head_dim * sizeof(float);
    flash_attention_kernel<<<grid, block, shared_bytes, params.stream>>>(
        params.output, params.query, params.key, params.value,
        params.batch, params.seq_len_q, params.seq_len_kv,
        params.num_heads, params.num_kv_heads, params.head_dim,
        params.scale, params.is_causal);
}

}  // namespace vllm
