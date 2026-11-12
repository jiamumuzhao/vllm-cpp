#pragma once

namespace vllm {
namespace kernels {

// SiLU activation: f(x) = x * sigmoid(x)
// Launches with block_size threads, processes n elements.
void silu_forward(float* out, const float* in, int n, cudaStream_t stream = 0);

// GELU activation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_forward(float* out, const float* in, int n, cudaStream_t stream = 0);

// SwiGLU: gated SiLU. gate = SiLU(x_gate), out = gate * x_up
void swiglu_forward(float* out, const float* gate, const float* up, int n, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace vllm
