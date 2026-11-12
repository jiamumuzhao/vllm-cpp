#include "vllm/kernels/layernorm.cuh"

#include <cuda_runtime.h>

namespace vllm {
namespace kernels {

constexpr int kWarpSize = 32;

template <int kBlockSize>
__global__ void rms_norm_kernel(float* out, const float* in, const float* weight,
                                 int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += kBlockSize) {
        float val = in[row * cols + i];
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    if (tid % kWarpSize == 0) {
        sdata[tid / kWarpSize] = sum_sq;
    }
    __syncthreads();

    float rms;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < (kBlockSize + kWarpSize - 1) / kWarpSize; i++) {
            sum += sdata[i];
        }
        rms = rsqrtf(sum / static_cast<float>(cols) + eps);
        sdata[0] = rms;
    }
    __syncthreads();

    rms = sdata[0];
    for (int i = tid; i < cols; i += kBlockSize) {
        out[row * cols + i] = in[row * cols + i] * rms * weight[i];
    }
}

void rms_norm(float* out, const float* in, const float* weight,
              int rows, int cols, float eps, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int shared_bytes = ((kBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(float);
    rms_norm_kernel<kBlockSize><<<rows, kBlockSize, shared_bytes, stream>>>(out, in, weight, rows, cols, eps);
}

__global__ void layer_norm_kernel(float* out, const float* in, const float* weight,
                                   const float* bias, int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* s_mean = shared;
    float* s_var = shared + blockDim.x;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = in[row * cols + i];
        sum += val;
        sum_sq += val * val;
    }
    s_mean[tid] = sum;
    s_var[tid] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        float total_sq = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            total += s_mean[i];
            total_sq += s_var[i];
        }
        float mean = total / static_cast<float>(cols);
        float var = total_sq / static_cast<float>(cols) - mean * mean;
        s_mean[0] = mean;
        s_var[0] = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean[0];
    float inv_std = s_var[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = (in[row * cols + i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void layer_norm(float* out, const float* in, const float* weight, const float* bias,
                int rows, int cols, float eps, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int shared_bytes = 2 * kBlockSize * sizeof(float);
    layer_norm_kernel<<<rows, kBlockSize, shared_bytes, stream>>>(out, in, weight, bias, rows, cols, eps);
}

}  // namespace kernels
}  // namespace vllm
