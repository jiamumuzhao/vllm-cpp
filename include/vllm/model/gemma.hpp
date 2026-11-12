#pragma once
#include <memory>
#include <vector>
#include "vllm/model/model_registry.hpp"
#include "vllm/layers/embedding.hpp"
#include "vllm/layers/attention.hpp"
#include "vllm/layers/mlp.hpp"
#include "vllm/layers/linear.hpp"
#include "vllm/memory/kv_cache.hpp"

namespace vllm {

class GemmaModel : public ModelBase {
  public:
    explicit GemmaModel(const EngineConfig& config);
    ~GemmaModel() override;
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
    size_t num_kv_blocks() const override { return 0; }
  private:
    EngineConfig config_;
    int vocab_size_ = 0, hidden_size_ = 0, num_layers_ = 0, num_heads_ = 0;
    int num_kv_heads_ = 0, head_dim_ = 0, intermediate_size_ = 0, max_seq_len_ = 0;
};

}  // namespace vllm
