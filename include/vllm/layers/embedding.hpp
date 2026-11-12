#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace vllm {

class Embedding {
  public:
    Embedding(int vocab_size, int hidden_size);
    ~Embedding();

    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;

    void set_weights(const float* weight);
    void attach_buffers(uint16_t* weight_buf);
    void forward(float* output, const int32_t* input_ids, int batch_size, cudaStream_t stream = 0);

    int vocab_size() const { return vocab_size_; }
    int hidden_size() const { return hidden_size_; }

  private:
    int vocab_size_;
    int hidden_size_;
    uint16_t* weight_fp16_ = nullptr;
    // public accessor for tied weights
  public:
    uint16_t* weight_device_ptr() const { return weight_fp16_; }
  private:
};

}  // namespace vllm
