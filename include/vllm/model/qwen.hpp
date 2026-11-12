#pragma once
#include <memory>
#include <vector>
#include "vllm/model/model_registry.hpp"
#include "vllm/layers/linear.hpp"
#include "vllm/memory/gpu_pool.hpp"

namespace vllm {

class Embedding;
class AttentionLayer;
class MLP;
class KVCache;

class QwenModel : public ModelBase {
  public:
    explicit QwenModel(const EngineConfig& config);
    ~QwenModel() override;
    bool load_weights(const std::string& path) override;
    void forward(float* logits, const int32_t* input_ids, int batch_size, int* context_len,
                 const std::vector<int32_t>& block_table) override;
    int vocab_size() const override { return vocab_size_; }
    int hidden_size() const override { return hidden_size_; }
    int num_layers() const override { return num_layers_; }
    int num_heads() const override { return num_heads_; }
    int num_kv_heads() const override { return num_kv_heads_; }
    int head_dim() const override { return head_dim_; }
    int intermediate_size() const override { return intermediate_size_; }
    int max_seq_len() const override { return max_seq_len_; }
    size_t num_kv_blocks() const override { return kv_block_count_; }

    void* gpu_pool_alloc(size_t bytes) { return gpu_pool_->alloc(bytes); }
  private:
    EngineConfig config_;
    size_t kv_block_count_ = 0;
    int vocab_size_ = 0, hidden_size_ = 0, num_layers_ = 0, num_heads_ = 0;
    int num_kv_heads_ = 0, head_dim_ = 0, intermediate_size_ = 0, max_seq_len_ = 0;
    float rope_theta_ = 1000000.0f;
    std::unique_ptr<Embedding> embed_;
    std::vector<std::unique_ptr<AttentionLayer>> attn_layers_;
    std::vector<std::unique_ptr<MLP>> mlp_layers_;
    std::unique_ptr<Linear> lm_head_;
    std::unique_ptr<KVCache> kv_cache_;
        float* hidden_buffer_ = nullptr;
    float* norm_buffer_ = nullptr;
    std::vector<float*> input_norm_weights_;
    std::vector<float*> post_norm_weights_;
    float* final_norm_weight_ = nullptr;
    std::vector<float*> attn_q_buf_;
    std::vector<float*> attn_k_buf_;
    std::vector<float*> attn_v_buf_;
    std::vector<float*> attn_out_buf_;
    std::unique_ptr<GpuMemoryPool> gpu_pool_;
};

}  // namespace vllm
