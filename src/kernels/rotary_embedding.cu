#include "vllm/kernels/rotary_embedding.cuh"

#include <cuda_runtime.h>

namespace vllm {
namespace kernels {

__global__ void rope_kernel(float* q, float* k, const float* cos_sin,
                             int num_tokens, int num_heads, int num_kv_heads, int head_dim,
                             int start_pos) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int half_dim = head_dim / 2;
    int dim_idx = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads || dim_idx >= half_dim) return;

    int pos = start_pos + token_idx;
    float cos_val = cos_sin[pos * half_dim * 2 + dim_idx * 2];
    float sin_val = cos_sin[pos * half_dim * 2 + dim_idx * 2 + 1];

    int q_offset = token_idx * num_heads * head_dim + head_idx * head_dim;
    float q0 = q[q_offset + dim_idx];
    float q1 = q[q_offset + dim_idx + half_dim];
    q[q_offset + dim_idx] = q0 * cos_val - q1 * sin_val;
    q[q_offset + dim_idx + half_dim] = q0 * sin_val + q1 * cos_val;

    if (head_idx < num_kv_heads) {
        int k_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim;
        float k0 = k[k_offset + dim_idx];
        float k1 = k[k_offset + dim_idx + half_dim];
        k[k_offset + dim_idx] = k0 * cos_val - k1 * sin_val;
        k[k_offset + dim_idx + half_dim] = k0 * sin_val + k1 * cos_val;
    }
}

void rope_forward(float* q, float* k, const float* cos_sin,
                  int num_tokens, int num_heads, int num_kv_heads, int head_dim,
                  int start_pos, float /*theta*/,
                  cudaStream_t stream) {
    int half_dim = head_dim / 2;
    dim3 grid(num_tokens, num_heads);
    dim3 block(half_dim);
    rope_kernel<<<grid, block, 0, stream>>>(q, k, cos_sin, num_tokens, num_heads, num_kv_heads, head_dim, start_pos);
}

}  // namespace kernels
}  // namespace vllm
