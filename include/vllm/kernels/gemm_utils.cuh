#pragma once

namespace vllm {
namespace kernels {

// Fused quantize + transpose for weight preprocessing.
// Converts float weights → INT8 with per-channel scales.
void quantize_w8_per_channel(int8_t* out, float* scale, const float* in,
                             int rows, int cols, cudaStream_t stream = 0);

// Pack float weights into INT4 format (2 values per byte, even/odd pairing).
void pack_w4(int8_t* out, const float* in, int rows, int cols, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace vllm
