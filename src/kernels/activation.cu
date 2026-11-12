#include "vllm/kernels/activation.cuh"

#include <cuda_runtime.h>

namespace vllm {
namespace kernels {

__device__ inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ inline float gelu_f(float x) {
    const float sqrt_2_over_pi = 0.79788456f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void silu_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu_f(in[idx]);
    }
}

__global__ void gelu_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = gelu_f(in[idx]);
    }
}

__global__ void swiglu_kernel(float* out, const float* gate, const float* up, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu_f(gate[idx]) * up[idx];
    }
}

void silu_forward(float* out, const float* in, int n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid = (n + kBlockSize - 1) / kBlockSize;
    silu_kernel<<<grid, kBlockSize, 0, stream>>>(out, in, n);
}

void gelu_forward(float* out, const float* in, int n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid = (n + kBlockSize - 1) / kBlockSize;
    gelu_kernel<<<grid, kBlockSize, 0, stream>>>(out, in, n);
}

void swiglu_forward(float* out, const float* gate, const float* up, int n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid = (n + kBlockSize - 1) / kBlockSize;
    swiglu_kernel<<<grid, kBlockSize, 0, stream>>>(out, gate, up, n);
}

}  // namespace kernels
}  // namespace vllm
