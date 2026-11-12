#include "vllm/layers/attention.hpp"

#include "vllm/layers/linear.hpp"
#include "vllm/memory/kv_cache.hpp"
#include "vllm/attention/paged_attention.hpp"
#include "vllm/kernels/rotary_embedding.cuh"
#include "vllm/utils/cuda_utils.hpp"
#include <cstdio>
#include <cstdlib>

namespace vllm {

AttentionLayer::AttentionLayer(int num_heads, int num_kv_heads, int head_dim, int hidden_size,
                               int max_seq_len, float rope_theta)
    : num_heads_(num_heads), num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      hidden_size_(hidden_size), max_seq_len_(max_seq_len), rope_theta_(rope_theta) {
    int qkv_size = num_heads_ * head_dim_;
    int kv_size = num_kv_heads_ * head_dim_;
    q_proj_ = std::make_unique<Linear>(hidden_size_, qkv_size, true);
    k_proj_ = std::make_unique<Linear>(hidden_size_, kv_size, true);
    v_proj_ = std::make_unique<Linear>(hidden_size_, kv_size, true);
    o_proj_ = std::make_unique<Linear>(qkv_size, hidden_size_, false);

    // RoPE table + context buffers from model GPU pool
}

AttentionLayer::~AttentionLayer() {
    // context buffers from pool
}

void AttentionLayer::attach_rope_table(float* buf) { cos_sin_ = buf; }

void AttentionLayer::attach_context_buffers(int32_t* cl_d, int32_t* bt_d, int max_blocks_per_seq) {
    context_lens_d_ = cl_d;
    block_table_d_ = bt_d;
    max_blocks_per_seq_ = max_blocks_per_seq;
}



void AttentionLayer::attach_work_buffers(float* q_buf, float* k_buf, float* v_buf, float* out_buf) {
    q_buf_ = q_buf;
    k_buf_ = k_buf;
    v_buf_ = v_buf;
    attn_out_buf_ = out_buf;
}

void AttentionLayer::attach_weight_buffers(uint16_t* qw, uint16_t* kw, uint16_t* vw, uint16_t* ow,
                                            float* qb, float* kb, float* vb) {
    q_proj_->attach_buffers(qw, qb);
    k_proj_->attach_buffers(kw, kb);
    v_proj_->attach_buffers(vw, vb);
    o_proj_->attach_buffers(ow, nullptr);
}

void AttentionLayer::set_weights(const float* q_weight, const float* k_weight, const float* v_weight,
                                  const float* o_weight, const float* q_bias,
                                  const float* k_bias, const float* v_bias) {
    q_proj_->set_weights(q_weight, q_bias);
    k_proj_->set_weights(k_weight, k_bias);
    v_proj_->set_weights(v_weight, v_bias);
    o_proj_->set_weights(o_weight, nullptr);
}

void AttentionLayer::forward(float* output, const float* hidden_states, KVCache& kv_cache,
                              const std::vector<int32_t>& block_table, int32_t context_len,
                              int batch_size, int layer_idx, cudaStream_t stream) {
    int qkv_size = num_heads_ * head_dim_;
    int kv_size = num_kv_heads_ * head_dim_;

    // Project queries, keys, values.
    float *q = q_buf_, *k = k_buf_, *v = v_buf_;

    q_proj_->forward(q, hidden_states, batch_size, stream);
    k_proj_->forward(k, hidden_states, batch_size, stream);
    v_proj_->forward(v, hidden_states, batch_size, stream);

    // RoPE.
    kernels::rope_forward(q, k, cos_sin_, batch_size, num_heads_, num_kv_heads_, head_dim_, context_len, rope_theta_, stream);

    // Store K/V into cache. Each token in the batch gets the next slot.
    // The engine currently submits one sequence at a time. Write each token to
    // its logical position, including the correct physical block and slot.
    for (int token = 0; token < batch_size; ++token) {
        int position = context_len + token;
        size_t logical_block = static_cast<size_t>(position) / kv_cache.block_size();
        if (logical_block >= block_table.size()) {
            fprintf(stderr, "KV block table missing position %d\n", position);
            std::abort();
        }
        int32_t phys_block = block_table[logical_block];
        int32_t slot = position % static_cast<int32_t>(kv_cache.block_size());
        float* k_slot = static_cast<float*>(kv_cache.k_ptr(layer_idx, phys_block)) +
                        static_cast<size_t>(slot) * kv_cache.num_kv_heads() * kv_cache.head_dim();
        float* v_slot = static_cast<float*>(kv_cache.v_ptr(layer_idx, phys_block)) +
                        static_cast<size_t>(slot) * kv_cache.num_kv_heads() * kv_cache.head_dim();
        size_t kv_bytes = kv_cache.num_kv_heads() * kv_cache.head_dim() * sizeof(float);
        VLLM_CUDA_CHECK(cudaMemcpyAsync(k_slot, k + static_cast<size_t>(token) * kv_size,
                                        kv_bytes, cudaMemcpyDeviceToDevice, stream));
        VLLM_CUDA_CHECK(cudaMemcpyAsync(v_slot, v + static_cast<size_t>(token) * kv_size,
                                        kv_bytes, cudaMemcpyDeviceToDevice, stream));
    }

    // PagedAttention into dedicated attention output buffer.
    context_lens_[0] = context_len + batch_size;
    VLLM_CUDA_CHECK(cudaMemcpyAsync(context_lens_d_, context_lens_, sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream));
    VLLM_CUDA_CHECK(cudaMemcpyAsync(block_table_d_, block_table.data(),
                                     block_table.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream));
    PagedAttentionParams pa_params{};
    pa_params.output = attn_out_buf_;
    pa_params.query = q;
    pa_params.key_cache = static_cast<float*>(kv_cache.k_ptr(layer_idx, 0));
    pa_params.value_cache = static_cast<float*>(kv_cache.v_ptr(layer_idx, 0));
    pa_params.block_tables = block_table_d_;
    pa_params.context_lens = context_lens_d_;
    pa_params.num_seqs = batch_size;
    pa_params.num_heads = num_heads_;
    pa_params.num_kv_heads = num_kv_heads_;
    pa_params.head_dim = head_dim_;
    pa_params.block_size = static_cast<int32_t>(kv_cache.block_size());
    pa_params.max_context_len = static_cast<int32_t>(std::min<size_t>(
        static_cast<size_t>(max_seq_len_), kv_cache.num_blocks() * kv_cache.block_size()));
    pa_params.max_blocks_per_seq = max_blocks_per_seq_;
    pa_params.scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
    pa_params.stream = stream;
    cudaMemsetAsync(attn_out_buf_, 0, static_cast<size_t>(batch_size) * qkv_size * sizeof(float), stream);
    paged_attention_forward(pa_params);

    // Output projection into dedicated attention output buffer.
    // Use q_buf_ as a separate output for the projection; cuBLAS does not
    // guarantee correctness when input and output alias.
    o_proj_->forward(q_buf_, attn_out_buf_, batch_size, stream);
    // output += attn_out_buf (residual)
    size_t n = static_cast<size_t>(batch_size) * static_cast<size_t>(hidden_size_);
    extern void launch_add_inplace(float* dst, const float* src, size_t n, cudaStream_t s);
    launch_add_inplace(output, q_buf_, n, stream);
}

}  // namespace vllm
