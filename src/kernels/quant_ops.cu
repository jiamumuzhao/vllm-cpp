#include "vllm/kernels/quant_ops.cuh"

#include <cuda_runtime.h>

namespace vllm {
namespace kernels {

__global__ void dequant_w8_kernel(float* out, const int8_t* w_int8, const float* scale,
                                   int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += static_cast<float>(w_int8[col * K + k]) * scale[col];
    }
    out[row * N + col] = acc;
}

void dequant_w8(float* out, const int8_t* weight_int8, const float* scale,
                int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    dequant_w8_kernel<<<grid, block, 0, stream>>>(out, weight_int8, scale, M, N, K);
}

__global__ void dequant_w4_kernel(float* out, const int8_t* w_int4, const float* scale,
                                   const float* zero, int M, int N, int K, int groupsize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    int group_idx = col / groupsize;
    float s = scale[group_idx * N + col];
    float z = zero[group_idx * N + col];
    float acc = 0.0f;
    for (int k = 0; k < K; k += 2) {
        int8_t packed = w_int4[col * (K / 2) + (k / 2)];
        float w0 = (static_cast<float>(packed & 0x0F) - 8.0f) * s + z;
        float w1 = (static_cast<float>((packed >> 4) & 0x0F) - 8.0f) * s + z;
        acc += w0;
        if (k + 1 < K) acc += w1;
    }
    out[row * N + col] = acc;
}

void dequant_w4(float* out, const int8_t* weight_int4, const float* scale, const float* zero,
                int M, int N, int K, int groupsize, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    dequant_w4_kernel<<<grid, block, 0, stream>>>(out, weight_int4, scale, zero, M, N, K, groupsize);
}

__global__ void dequant_fp8_kernel(float* out, const uint8_t* w_fp8, const float* scale,
                                    int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        uint8_t raw = w_fp8[col * K + k];
        int exp = (raw >> 4) & 0x07;
        int mant = raw & 0x0F;
        float sign = (raw & 0x80) ? -1.0f : 1.0f;
        float val = sign * (static_cast<float>(mant) / 16.0f + 1.0f) * exp2f(static_cast<float>(exp));
        acc += val * scale[col];
    }
    out[row * N + col] = acc;
}

void dequant_fp8(float* out, const uint8_t* weight_fp8, const float* scale,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    dequant_fp8_kernel<<<grid, block, 0, stream>>>(out, weight_fp8, scale, M, N, K);
}

}  // namespace kernels
}  // namespace vllm
