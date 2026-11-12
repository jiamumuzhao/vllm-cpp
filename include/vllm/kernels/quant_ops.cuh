#include <cstdint>
#pragma once

namespace vllm {
namespace kernels {

// INT8 weight dequantization: W_fp16 = scale * W_int8
// out: [M, K] in float, weight_int8: [K, N] in int8, scale: [N] in float
void dequant_w8(float* out, const int8_t* weight_int8, const float* scale,
                int M, int N, int K, cudaStream_t stream = 0);

// INT4 weight dequantization (GPTQ/AWQ format).
// Two int4 values packed per int8 byte. groupsize controls granularity.
void dequant_w4(float* out, const int8_t* weight_int4, const float* scale, const float* zero,
                int M, int N, int K, int groupsize, cudaStream_t stream = 0);

// FP8 dequantization (Blackwell native, simulated in software for back compat).
void dequant_fp8(float* out, const uint8_t* weight_fp8, const float* scale,
                 int M, int N, int K, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace vllm
#include <cstdint>
#include <cuda_runtime.h>
