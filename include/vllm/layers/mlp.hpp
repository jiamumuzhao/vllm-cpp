#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <string>

#include <memory>

namespace vllm {

class Linear;

class MLP {
  public:
    MLP(int hidden_size, int intermediate_size, const std::string& act_fn = "silu");
    ~MLP();

    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;

        void set_weights(const float* gate_weight, const float* up_weight, const float* down_weight);
    void attach_weight_buffers(uint16_t* gw, uint16_t* uw, uint16_t* dw);
    void attach_work_buffers(float* gate_buf, float* up_buf, float* out_buf);    void forward(float* output, const float* hidden_states, int batch_size, cudaStream_t stream = 0);

  private:
    int hidden_size_;
    int intermediate_size_;
    std::string act_fn_;  // "silu" or "gelu"

    std::unique_ptr<Linear> gate_proj_;
    std::unique_ptr<Linear> up_proj_;
    std::unique_ptr<Linear> down_proj_;

    float* gate_buffer_ = nullptr;
    float* up_buffer_ = nullptr;
    bool owns_work_buffers_ = true;
    float* out_buffer_ = nullptr;
};

}  // namespace vllm
