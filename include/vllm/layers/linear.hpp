#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace vllm {

class Linear {
  public:
    Linear(int in_features, int out_features, bool has_bias = false);
    ~Linear();

    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;

    // Load weights from host memory (FP16/FP32).
    void set_weights(const float* weight, const float* bias = nullptr);
    void set_weights(const int8_t* weight, const float* scale, const float* bias = nullptr);


    // Use an existing device pointer (for tied weights)
    void use_device_pointer(uint16_t* device_weight, float* device_bias = nullptr);

    // Attach pool slices; set_weights then writes into these buffers.
    void attach_buffers(uint16_t* weight_buf, float* bias_buf);

    // Forward pass: out = x @ W^T + bias
    // x: [batch, in_features], out: [batch, out_features]
    void forward(float* out, const float* x, int batch, cudaStream_t stream = 0);

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

  private:
    int in_features_;
    int out_features_;
    bool has_bias_;

    // GPU buffers
    uint16_t* weight_fp16_ = nullptr;
    uint16_t* input_fp16_ = nullptr;
    size_t input_capacity_ = 0;
    bool owns_weights_ = true;

    float* bias_fp32_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
};

}  // namespace vllm
