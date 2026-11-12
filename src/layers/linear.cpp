#include "vllm/layers/linear.hpp"
#include "vllm/utils/cuda_utils.hpp"
#include <cuda_fp16.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cublas_v2.h>
namespace vllm {

void launch_add_bias(float* out, const float* bias, int rows, int cols, cudaStream_t stream);
void launch_f32_to_fp16(uint16_t* out, const float* in, size_t n, cudaStream_t stream);

Linear::Linear(int in_features, int out_features, bool has_bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(has_bias) {
    cublasCreate(&cublas_handle_);
}

Linear::~Linear() {
    if (input_fp16_) cudaFree(input_fp16_);
    if (owns_weights_) {
        if (weight_fp16_) cudaFree(weight_fp16_);
        if (bias_fp32_) cudaFree(bias_fp32_);
    }
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

void Linear::attach_buffers(uint16_t* weight_buf, float* bias_buf) {
    weight_fp16_ = weight_buf;
    if (has_bias_) bias_fp32_ = bias_buf;
    owns_weights_ = false;
}

void Linear::set_weights(const float* weight, const float* bias) {
    if (!weight_fp16_) {
        cudaMalloc(&weight_fp16_, static_cast<size_t>(in_features_) * static_cast<size_t>(out_features_) * sizeof(uint16_t));
        if (has_bias_) cudaMalloc(&bias_fp32_, static_cast<size_t>(out_features_) * sizeof(float));
    }
    size_t n = static_cast<size_t>(in_features_) * static_cast<size_t>(out_features_);
    std::vector<uint16_t> fp16(n);
    for (size_t i = 0; i < n; i++) {
        fp16[i] = __half_as_ushort(__float2half(weight[i]));
    }
    VLLM_CUDA_CHECK(cudaMemcpy(weight_fp16_, fp16.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));
    if (has_bias_ && bias && bias_fp32_) {
        VLLM_CUDA_CHECK(cudaMemcpy(bias_fp32_, bias, static_cast<size_t>(out_features_) * sizeof(float),
                                   cudaMemcpyHostToDevice));
    }
}

void Linear::set_weights(const int8_t*, const float*, const float*) {}

void Linear::use_device_pointer(uint16_t* device_weight, float* device_bias) {
    if (owns_weights_ && weight_fp16_) cudaFree(weight_fp16_);
    if (owns_weights_ && bias_fp32_ && device_bias) cudaFree(bias_fp32_);
    weight_fp16_ = device_weight;
    if (has_bias_ && device_bias) bias_fp32_ = device_bias;
    owns_weights_ = false;
}

void Linear::forward(float* out, const float* x, int batch, cudaStream_t stream) {
    size_t input_count = static_cast<size_t>(batch) * in_features_;
    if (input_count > input_capacity_) {
        if (input_fp16_) VLLM_CUDA_CHECK(cudaFree(input_fp16_));
        VLLM_CUDA_CHECK(cudaMalloc(&input_fp16_, input_count * sizeof(uint16_t)));
        input_capacity_ = input_count;
    }
    launch_f32_to_fp16(input_fp16_, x, input_count, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(cublas_handle_, stream);
    cublasStatus_t status = cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                 out_features_, batch, in_features_,
                 &alpha,
                 weight_fp16_, CUDA_R_16F, in_features_,
                 input_fp16_, CUDA_R_16F, in_features_,
                 &beta,
                 out, CUDA_R_32F, out_features_,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM failed: status=%d, dimensions=%d x %d x %d\n",
                static_cast<int>(status), out_features_, batch, in_features_);
        std::abort();
    }

    if (has_bias_ && bias_fp32_) {
        launch_add_bias(out, bias_fp32_, batch, out_features_, stream);
    }
}

}  // namespace vllm
