#include "vllm/model/llama.hpp"
#include "vllm/model/model_registry.hpp"
#include "vllm/layers/embedding.hpp"
#include "vllm/layers/attention.hpp"
#include "vllm/layers/mlp.hpp"
#include "vllm/layers/linear.hpp"
#include "vllm/memory/kv_cache.hpp"
#include "vllm/kernels/layernorm.cuh"
#include "vllm/utils/cuda_utils.hpp"

namespace vllm {

LlamaModel::LlamaModel(const EngineConfig& config) : config_(config) {
    // Default LLaMA-3 8B architecture.
    vocab_size_ = 128256;
    hidden_size_ = 4096;
    num_layers_ = 32;
    num_heads_ = 32;
    num_kv_heads_ = 8;
    head_dim_ = 128;
    intermediate_size_ = 14336;
    max_seq_len_ = 8192;

    embed_ = std::make_unique<Embedding>(vocab_size_, hidden_size_);

    for (int i = 0; i < num_layers_; i++) {
        attn_layers_.push_back(std::make_unique<AttentionLayer>(
            num_heads_, num_kv_heads_, head_dim_, hidden_size_, max_seq_len_));
        mlp_layers_.push_back(std::make_unique<MLP>(hidden_size_, intermediate_size_, "silu"));
    }

    lm_head_ = std::make_unique<Linear>(hidden_size_, vocab_size_, false);

    size_t num_blocks = 256;  // Conservative for 12GB VRAM
    kv_cache_ = std::make_unique<KVCache>(num_layers_, num_blocks, config_.block_size,
                                           num_kv_heads_, head_dim_);

    VLLM_CUDA_CHECK(cudaMalloc(&hidden_buffer_, static_cast<size_t>(hidden_size_ * config_.max_num_seqs) * sizeof(float)));
    VLLM_CUDA_CHECK(cudaMalloc(&norm_buffer_, static_cast<size_t>(hidden_size_ * config_.max_num_seqs) * sizeof(float)));
}

LlamaModel::~LlamaModel() {
    if (hidden_buffer_) cudaFree(hidden_buffer_);
    if (norm_buffer_) cudaFree(norm_buffer_);
}

bool LlamaModel::load_weights(const std::string& path) {
    // MVP: stub — real impl would use ModelLoader + safetensors parsing.
    (void)path;
    return true;
}

void LlamaModel::forward(float* logits, const int32_t* input_ids, int batch_size, int* context_len,
                         const std::vector<int32_t>& block_table) {
    cudaStream_t stream = 0;

    // Embedding lookup.
    embed_->forward(hidden_buffer_, input_ids, batch_size, stream);

    // Transformer layers.
    for (int layer = 0; layer < num_layers_; layer++) {
        // RMSNorm (pre-norm).
        kernels::rms_norm(norm_buffer_, hidden_buffer_, nullptr, batch_size, hidden_size_, 1e-6f, stream);

        // Self-attention with residual.
        attn_layers_[layer]->forward(hidden_buffer_, norm_buffer_, *kv_cache_,
                                      block_table, *context_len + batch_size,
                                      batch_size, layer, stream);

        // RMSNorm (pre-norm).
        kernels::rms_norm(norm_buffer_, hidden_buffer_, nullptr, batch_size, hidden_size_, 1e-6f, stream);

        // MLP with residual.
        mlp_layers_[layer]->forward(hidden_buffer_, norm_buffer_, batch_size, stream);
    }

    // Final RMSNorm + LM head.
    kernels::rms_norm(norm_buffer_, hidden_buffer_, nullptr, batch_size, hidden_size_, 1e-6f, stream);
    lm_head_->forward(logits, norm_buffer_, batch_size, stream);

    *context_len += batch_size;
}

// Auto-register.
REGISTER_MODEL("llama", LlamaModel);

}  // namespace vllm
