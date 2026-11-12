#include "vllm/layers/mlp.hpp"

#include "vllm/layers/linear.hpp"
#include "vllm/kernels/activation.cuh"
#include "vllm/utils/cuda_utils.hpp"

namespace vllm {

MLP::MLP(int hidden_size, int intermediate_size, const std::string& act_fn)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size), act_fn_(act_fn) {
    gate_proj_ = std::make_unique<Linear>(hidden_size_, intermediate_size_, false);
    up_proj_ = std::make_unique<Linear>(hidden_size_, intermediate_size_, false);
    down_proj_ = std::make_unique<Linear>(intermediate_size_, hidden_size_, false);
    // work buffers attached from model GPU pool
}

MLP::~MLP() {
    if (owns_work_buffers_) {
        if (gate_buffer_) cudaFree(gate_buffer_);
        if (up_buffer_) cudaFree(up_buffer_);
    }
}



void MLP::attach_work_buffers(float* gate_buf, float* up_buf, float* out_buf) {
    gate_buffer_ = gate_buf;
    up_buffer_ = up_buf;
    out_buffer_ = out_buf;
    owns_work_buffers_ = false;
}

void MLP::attach_weight_buffers(uint16_t* gw, uint16_t* uw, uint16_t* dw) {
    gate_proj_->attach_buffers(gw, nullptr);
    up_proj_->attach_buffers(uw, nullptr);
    down_proj_->attach_buffers(dw, nullptr);
}

void MLP::set_weights(const float* gate_weight, const float* up_weight, const float* down_weight) {
    gate_proj_->set_weights(gate_weight);
    up_proj_->set_weights(up_weight);
    down_proj_->set_weights(down_weight);
}

void MLP::forward(float* output, const float* hidden_states, int batch_size, cudaStream_t stream) {
    // gate = SiLU(x @ gate_W)
    // up   = x @ up_W
    // out  = (gate * up) @ down_W
    gate_proj_->forward(gate_buffer_, hidden_states, batch_size, stream);
    up_proj_->forward(up_buffer_, hidden_states, batch_size, stream);

    if (act_fn_ != "silu") {
        kernels::gelu_forward(gate_buffer_, gate_buffer_, batch_size * intermediate_size_, stream);
    }

    // Element-wise multiply: gate * up
    kernels::swiglu_forward(out_buffer_, gate_buffer_, up_buffer_, batch_size * intermediate_size_, stream);

    // Keep GEMM input and output separate; cuBLAS does not guarantee in-place GEMM.
    down_proj_->forward(gate_buffer_, out_buffer_, batch_size, stream);

    // output += out_buffer (residual)
    extern void launch_add_inplace(float* dst, const float* src, size_t n, cudaStream_t s);
    launch_add_inplace(output, gate_buffer_, static_cast<size_t>(batch_size) * static_cast<size_t>(hidden_size_), stream);
}

}  // namespace vllm
