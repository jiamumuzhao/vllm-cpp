#pragma once

#include <cuda_runtime.h>
#include <vector>

#include <cstdint>
#include <memory>

#include "vllm/attention/attention_backend.hpp"

namespace vllm {

class Linear;
class KVCache;

class AttentionLayer {
  public:
    AttentionLayer(int num_heads, int num_kv_heads, int head_dim, int hidden_size,
                   int max_seq_len, float rope_theta = 10000.0f);
    ~AttentionLayer();

    AttentionLayer(const AttentionLayer&) = delete;
    AttentionLayer& operator=(const AttentionLayer&) = delete;

    void set_weights(const float* q_weight, const float* k_weight, const float* v_weight,
                     const float* o_weight, const float* q_bias = nullptr,
                     const float* k_bias = nullptr, const float* v_bias = nullptr);

            void attach_rope_table(float* buf);
    void attach_context_buffers(int32_t* cl_d, int32_t* bt_d, int max_blocks_per_seq);
    void attach_work_buffers(float* q_buf, float* k_buf, float* v_buf, float* out_buf);    void attach_weight_buffers(uint16_t* qw, uint16_t* kw, uint16_t* vw, uint16_t* ow,
                               float* qb, float* kb, float* vb);

    void forward(float* output, const float* hidden_states, KVCache& kv_cache,
                 const std::vector<int32_t>& block_table, int32_t context_len,
                 int batch_size, int layer_idx, cudaStream_t stream = 0);

  private:
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int hidden_size_;
    int max_seq_len_;
    float rope_theta_;
    AttentionBackend backend_ = AttentionBackend::kPagedAttention;

    std::unique_ptr<Linear> q_proj_;
    std::unique_ptr<Linear> k_proj_;
    std::unique_ptr<Linear> v_proj_;
    std::unique_ptr<Linear> o_proj_;

        float* cos_sin_ = nullptr;  // precomputed RoPE table
    float* q_buf_ = nullptr;
    float* k_buf_ = nullptr;
    float* v_buf_ = nullptr;
        float* attn_out_buf_ = nullptr;
    int32_t context_lens_[64] = {};
    int32_t block_table_arr_[64] = {};
    int32_t* context_lens_d_ = nullptr;
    int32_t* block_table_d_ = nullptr;
    int max_blocks_per_seq_ = 0;
};

}  // namespace vllm
