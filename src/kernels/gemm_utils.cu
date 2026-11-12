#include "vllm/kernels/gemm_utils.cuh"

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

namespace vllm {
namespace kernels {

__global__ void quantize_w8_kernel(int8_t* out, float* scale, const float* in,
                                    int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float max_abs = 0.0f;
    for (int r = 0; r < rows; r++) {
        max_abs = fmaxf(max_abs, fabsf(in[r * cols + col]));
    }
    scale[col] = max_abs / 127.0f;
    float inv_scale = (max_abs > 0.0f) ? 127.0f / max_abs : 1.0f;

    for (int r = 0; r < rows; r++) {
        float val = in[r * cols + col] * inv_scale;
        val = fminf(fmaxf(val, -128.0f), 127.0f);
        out[r * cols + col] = static_cast<int8_t>(__float2int_rn(val));
    }
}

void quantize_w8_per_channel(int8_t* out, float* scale, const float* in,
                             int rows, int cols, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid = (cols + kBlockSize - 1) / kBlockSize;
    quantize_w8_kernel<<<grid, kBlockSize, 0, stream>>>(out, scale, in, rows, cols);
}

__global__ void pack_w4_kernel(int8_t* out, const float* in, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row_pair = blockIdx.y;
    if (col >= cols || row_pair >= (rows + 1) / 2) return;

    int r0 = row_pair * 2;
    int r1 = r0 + 1;
    float v0 = r0 < rows ? in[r0 * cols + col] : 0.0f;
    float v1 = r1 < rows ? in[r1 * cols + col] : 0.0f;

    int q0 = __float2int_rn(fminf(fmaxf(v0, -8.0f), 7.0f)) & 0x0F;
    int q1 = __float2int_rn(fminf(fmaxf(v1, -8.0f), 7.0f)) & 0x0F;
    out[row_pair * cols + col] = static_cast<int8_t>((q1 << 4) | q0);
}

void pack_w4(int8_t* out, const float* in, int rows, int cols, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, ((rows + 1) / 2 + block.y - 1) / block.y);
    pack_w4_kernel<<<grid, block, 0, stream>>>(out, in, rows, cols);
}

}  // namespace kernels
}  // namespace vllm
