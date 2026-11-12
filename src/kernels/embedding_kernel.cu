#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace vllm {

__global__ void f32_to_fp16_kernel(uint16_t* dst, const float* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half_as_ushort(__float2half(src[i]));
}

void launch_f32_to_fp16(uint16_t* dst, const float* src, size_t n, cudaStream_t stream) {
    f32_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, n);
}

__device__ inline float fp16_to_float(uint16_t v) {
    return __half2float(__ushort_as_half(v));
}

__global__ void embedding_lookup_kernel(
    float* output, const uint16_t* weight, const int32_t* input_ids,
    int batch_size, int hidden_size) {
    int b = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch_size || d >= hidden_size) return;
    int32_t token_id = input_ids[b];
    output[b * hidden_size + d] = fp16_to_float(weight[token_id * hidden_size + d]);
}

void launch_embedding_lookup_fp16(float* output, const uint16_t* weight, const int32_t* input_ids,
                              int batch_size, int hidden_size, cudaStream_t stream) {
    constexpr int kBlockSize = 64;
    dim3 grid(batch_size, (hidden_size + kBlockSize - 1) / kBlockSize);
    dim3 block(kBlockSize);
    embedding_lookup_kernel<<<grid, block, 0, stream>>>(
        output, weight, input_ids, batch_size, hidden_size);
}

__global__ void fp16_to_f32_kernel(float* dst, const uint16_t* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(__ushort_as_half(src[i]));
}

}  // namespace vllm

extern "C" void launch_fp16_to_f32(float* dst, const uint16_t* src, size_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    size_t grid = (n + kBlock - 1) / kBlock;
    vllm::fp16_to_f32_kernel<<<grid, kBlock, 0, stream>>>(dst, src, n);
}

namespace vllm {

__global__ void add_bias_kernel(float* out, const float* bias, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    if (row >= rows || col >= cols) return;
    out[row * cols + col] += bias[col];
}

void launch_add_bias(float* out, const float* bias, int rows, int cols, cudaStream_t stream) {
    constexpr int kBlockSize = 64;
    dim3 grid(rows, (cols + kBlockSize - 1) / kBlockSize);
    dim3 block(kBlockSize);
    add_bias_kernel<<<grid, block, 0, stream>>>(out, bias, rows, cols);
}

}  // namespace vllm

namespace vllm {

__global__ void add_inplace_kernel(float* dst, const float* src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

}

namespace vllm {

void launch_add_inplace(float* dst, const float* src, size_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    size_t grid = (n + kBlock - 1) / kBlock;
    add_inplace_kernel<<<grid, kBlock, 0, stream>>>(dst, src, n);
}

}  // namespace vllm
