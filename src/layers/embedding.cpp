#include "vllm/layers/embedding.hpp"
#include "vllm/utils/cuda_utils.hpp"
#include <cuda_fp16.h>
#include <vector>
#include <cstring>

namespace vllm {

void launch_embedding_lookup_fp16(float* output, const uint16_t* weight, const int32_t* input_ids,
                                   int batch_size, int hidden_size, cudaStream_t stream);

Embedding::Embedding(int vocab_size, int hidden_size)
    : vocab_size_(vocab_size), hidden_size_(hidden_size) {
    (void)vocab_size_; (void)hidden_size_;  // allocated from pool via attach_buffers
}

Embedding::~Embedding() {
    // weight_fp16_ owned by model GPU pool
}

void Embedding::attach_buffers(uint16_t* weight_buf) { weight_fp16_ = weight_buf; }

void Embedding::set_weights(const float* weight) {
    if (!weight_fp16_) {
        cudaMalloc(&weight_fp16_, static_cast<size_t>(vocab_size_) * static_cast<size_t>(hidden_size_) * sizeof(uint16_t));
    }
    size_t n = static_cast<size_t>(vocab_size_) * static_cast<size_t>(hidden_size_);
    std::vector<uint16_t> fp16(n);
    for (size_t i = 0; i < n; i++) {
        fp16[i] = __half_as_ushort(__float2half(weight[i]));
    }
    VLLM_CUDA_CHECK(cudaMemcpy(weight_fp16_, fp16.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

void Embedding::forward(float* output, const int32_t* input_ids, int batch_size, cudaStream_t stream) {
    launch_embedding_lookup_fp16(output, weight_fp16_, input_ids, batch_size, hidden_size_, stream);
}

}  // namespace vllm
