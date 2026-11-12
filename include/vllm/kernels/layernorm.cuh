#pragma once

namespace vllm {
namespace kernels {

// RMS Normalization: out = in * weight * rsqrt(mean(in^2) + eps)
// Shape: [rows, cols]. Each row is normalized independently.
void rms_norm(float* out, const float* in, const float* weight,
              int rows, int cols, float eps = 1e-6f, cudaStream_t stream = 0);

// Layer Normalization: same interface as rms_norm but with bias.
void layer_norm(float* out, const float* in, const float* weight, const float* bias,
                int rows, int cols, float eps = 1e-5f, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace vllm
