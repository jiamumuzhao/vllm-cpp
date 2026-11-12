#include "vllm/model/qwen.hpp"
#include "vllm/model/model_registry.hpp"
#include "vllm/model/model_loader.hpp"
#include "vllm/layers/embedding.hpp"
#include "vllm/layers/attention.hpp"
#include "vllm/layers/mlp.hpp"
#include "vllm/layers/linear.hpp"
#include "vllm/memory/kv_cache.hpp"
#include "vllm/kernels/layernorm.cuh"
#include "vllm/utils/cuda_utils.hpp"
#include "vllm/utils/logger.hpp"
#include "vllm/utils/json_utils.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace vllm {

QwenModel::QwenModel(const EngineConfig& config) : config_(config) {}

QwenModel::~QwenModel() {
    // GPU memory owned by gpu_pool_
}

bool QwenModel::load_weights(const std::string& path) {
    // 1. Read config.json from the model directory
    std::string config_path = path + "/config.json";
    std::ifstream cfg_file(config_path);
    if (!cfg_file) {
        VLLM_ERROR("Cannot open config: {}", config_path);
        return false;
    }
    auto j = vllm::json::parse_file(config_path);
    hidden_size_       = j.contains("hidden_size") ? static_cast<int>(j["hidden_size"].get_int()) : 896;
    num_layers_        = j.contains("num_hidden_layers") ? static_cast<int>(j["num_hidden_layers"].get_int()) : 24;
    num_heads_         = j.contains("num_attention_heads") ? static_cast<int>(j["num_attention_heads"].get_int()) : 14;
    num_kv_heads_      = j.contains("num_key_value_heads") ? static_cast<int>(j["num_key_value_heads"].get_int()) : 2;
    intermediate_size_ = j.contains("intermediate_size") ? static_cast<int>(j["intermediate_size"].get_int()) : 4864;
    max_seq_len_       = j.contains("max_position_embeddings") ? static_cast<int>(j["max_position_embeddings"].get_int()) : 32768;
    rope_theta_        = j.contains("rope_theta") ? static_cast<float>(j["rope_theta"].get_number()) : 1000000.0f;
    if (max_seq_len_ <= 0 || rope_theta_ <= 0.0f) {
        VLLM_ERROR("Invalid Qwen RoPE configuration: max_position_embeddings={}, rope_theta={}",
                   max_seq_len_, rope_theta_);
        return false;
    }
    vocab_size_        = j.contains("vocab_size") ? static_cast<int>(j["vocab_size"].get_int()) : 151936;
    head_dim_          = hidden_size_ / num_heads_;
    if (config_.block_size == 0 || config_.gpu_memory_utilization == 0 ||
        config_.gpu_memory_utilization > 100 || config_.max_num_seqs == 0) {
        VLLM_ERROR("Invalid KV cache configuration");
        return false;
    }

    VLLM_INFO("QwenModel: {}L, h={}, heads={}/{}, dim={}, vocab={}",
              num_layers_, hidden_size_, num_heads_, num_kv_heads_, head_dim_, vocab_size_);

    // 2. Calculate GPU memory budget and create one pool
    size_t qkv_size = static_cast<size_t>(num_heads_) * static_cast<size_t>(head_dim_);
    size_t kv_size  = static_cast<size_t>(num_kv_heads_) * static_cast<size_t>(head_dim_);
    size_t weight_bytes = 0;
    weight_bytes += static_cast<size_t>(vocab_size_) * static_cast<size_t>(hidden_size_) * sizeof(uint16_t);  // embed
    weight_bytes += static_cast<size_t>(num_layers_) * (
        (static_cast<size_t>(hidden_size_) * qkv_size * 2) +          // q + o weights
        (static_cast<size_t>(hidden_size_) * kv_size * 2) +           // k + v weights
        (static_cast<size_t>(hidden_size_) * intermediate_size_ * 3 * 2) +  // mlp weights (2x due to gate/up/down)
        0);  // no extra
    weight_bytes += static_cast<size_t>(num_layers_) * 3 * sizeof(float) * 1024;  // bias slack
    weight_bytes += static_cast<size_t>(num_layers_) * static_cast<size_t>(max_seq_len_) *
                    static_cast<size_t>(head_dim_) * sizeof(float);  // RoPE table
    weight_bytes += 2 * static_cast<size_t>(config_.max_num_seqs) * static_cast<size_t>(hidden_size_) * sizeof(float);  // buffers
    weight_bytes += 2 * static_cast<size_t>(num_layers_) * static_cast<size_t>(intermediate_size_) * sizeof(float);  // mlp work
    const size_t kv_bytes_per_block = 2 * config_.block_size * static_cast<size_t>(num_kv_heads_)
                    * static_cast<size_t>(head_dim_) * sizeof(float) * static_cast<size_t>(num_layers_);
    weight_bytes += static_cast<size_t>(num_layers_) *
        (3 * intermediate_size_ + 3 * qkv_size + 2 * kv_size + 2 * hidden_size_) * sizeof(float);
    weight_bytes += static_cast<size_t>(hidden_size_) * sizeof(float) + 64 * 1024 * 1024;

    weight_bytes += 64 * sizeof(int32_t) + 64 * static_cast<size_t>(vocab_size_) * sizeof(float);  // engine buffers
    size_t max_blocks_per_seq = (config_.max_model_len + config_.block_size - 1) / config_.block_size;
    weight_bytes += static_cast<size_t>(num_layers_) * (max_blocks_per_seq + 1) * sizeof(int32_t);
    // The pool owns both model/runtime buffers and the KV cache. Size KV from
    // the actual pre-load free memory budget, leaving the existing 64 MiB
    // workspace allowance in the non-KV estimate.
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    VLLM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t non_kv_bytes = weight_bytes;
    const size_t budget_bytes = static_cast<size_t>(
        static_cast<double>(free_bytes) * static_cast<double>(config_.gpu_memory_utilization) / 100.0);
    if (budget_bytes <= non_kv_bytes || kv_bytes_per_block == 0) {
        VLLM_ERROR("GPU memory budget is insufficient for model buffers and KV cache");
        return false;
    }
    const size_t blocks_per_seq = config_.max_model_len / config_.block_size +
                                  (config_.max_model_len % config_.block_size != 0);
    const size_t max_blocks = blocks_per_seq > static_cast<size_t>(-1) / config_.max_num_seqs
                                  ? static_cast<size_t>(-1)
                                  : blocks_per_seq * config_.max_num_seqs;
    kv_block_count_ = std::min(max_blocks, (budget_bytes - non_kv_bytes) / kv_bytes_per_block);
    if (kv_block_count_ == 0) {
        VLLM_ERROR("GPU memory budget cannot fit one KV cache block");
        return false;
    }
    weight_bytes = non_kv_bytes + kv_block_count_ * kv_bytes_per_block;
    gpu_pool_ = std::make_unique<GpuMemoryPool>(weight_bytes);

    // 3. Initialize layers from pool slices
    size_t weight_elements = static_cast<size_t>(vocab_size_) * hidden_size_ +
        static_cast<size_t>(num_layers_) * (2 * static_cast<size_t>(hidden_size_) * qkv_size +
            2 * static_cast<size_t>(hidden_size_) * kv_size +
            3 * static_cast<size_t>(hidden_size_) * intermediate_size_);
    uint16_t* wbase = static_cast<uint16_t*>(gpu_pool_->alloc(weight_elements * sizeof(uint16_t)));
    float* fbase = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * 3 * 1024 * sizeof(float)));
    float* rope_base = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * static_cast<size_t>(max_seq_len_) *
        static_cast<size_t>(head_dim_) * sizeof(float)));
    // Precompute RoPE cos/sin table on CPU and upload once.
    {
        int half_dim = static_cast<int>(head_dim_) / 2;
        std::vector<float> rope_table(static_cast<size_t>(max_seq_len_) * static_cast<size_t>(head_dim_));
        for (int pos = 0; pos < max_seq_len_; pos++) {
            for (int d = 0; d < half_dim; d++) {
                float freq = pos / powf(rope_theta_, 2.0f * static_cast<float>(d) / static_cast<float>(head_dim_));
                rope_table[static_cast<size_t>(pos * static_cast<int>(head_dim_) + d * 2)] = cosf(freq);
                rope_table[static_cast<size_t>(pos * static_cast<int>(head_dim_) + d * 2 + 1)] = sinf(freq);
            }
        }
        for (int i = 0; i < num_layers_; i++) {
            VLLM_CUDA_CHECK(cudaMemcpy(rope_base + static_cast<size_t>(i) * static_cast<size_t>(max_seq_len_) *
                                            static_cast<size_t>(head_dim_),
                                        rope_table.data(), rope_table.size() * sizeof(float),
                                        cudaMemcpyHostToDevice));
        }
    }
    float* mlp_gate_base = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * static_cast<size_t>(intermediate_size_) * sizeof(float)));
    float* mlp_up_base = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * static_cast<size_t>(intermediate_size_) * sizeof(float)));
    float* mlp_out_base = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * static_cast<size_t>(intermediate_size_) * sizeof(float)));
    float* attn_work_base = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * (2 * qkv_size + 2 * kv_size) * sizeof(float)));
    float* attn_out_base = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * qkv_size * sizeof(float)));
    int32_t* attn_dev_base = static_cast<int32_t*>(gpu_pool_->alloc(
        static_cast<size_t>(num_layers_) * (max_blocks_per_seq + 1) * sizeof(int32_t)));
    float* norm_base = static_cast<float*>(gpu_pool_->alloc(
        (2 * static_cast<size_t>(num_layers_) + 1) * static_cast<size_t>(hidden_size_) * sizeof(float)));

    input_norm_weights_.resize(num_layers_);
    post_norm_weights_.resize(num_layers_);
    for (int i = 0; i < num_layers_; i++) {
        input_norm_weights_[static_cast<size_t>(i)] = norm_base + static_cast<size_t>(i) * hidden_size_;
        post_norm_weights_[static_cast<size_t>(i)] = norm_base + (static_cast<size_t>(num_layers_) + static_cast<size_t>(i)) * hidden_size_;
    }
    final_norm_weight_ = norm_base + (2 * static_cast<size_t>(num_layers_)) * hidden_size_;
    hidden_buffer_ = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(config_.max_num_seqs) * static_cast<size_t>(hidden_size_) * sizeof(float)));
    norm_buffer_ = static_cast<float*>(gpu_pool_->alloc(
        static_cast<size_t>(config_.max_num_seqs) * static_cast<size_t>(hidden_size_) * sizeof(float)));

    embed_ = std::make_unique<Embedding>(vocab_size_, hidden_size_);
    embed_->attach_buffers(wbase);
    uint16_t* wcur = wbase + static_cast<size_t>(vocab_size_) * static_cast<size_t>(hidden_size_);
    float* bcur = fbase;
    for (int i = 0; i < num_layers_; i++) {
        attn_layers_.push_back(std::make_unique<AttentionLayer>(
            num_heads_, num_kv_heads_, head_dim_, hidden_size_, max_seq_len_, rope_theta_));
        attn_layers_.back()->attach_rope_table(rope_base +
            static_cast<size_t>(i) * static_cast<size_t>(max_seq_len_) * static_cast<size_t>(head_dim_));
        int32_t* layer_dev = attn_dev_base + static_cast<size_t>(i) * (max_blocks_per_seq + 1);
        attn_layers_.back()->attach_context_buffers(
            layer_dev + max_blocks_per_seq, layer_dev, static_cast<int>(max_blocks_per_seq));
        uint16_t* qw = wcur; wcur += static_cast<size_t>(hidden_size_) * qkv_size;
        uint16_t* kw = wcur; wcur += static_cast<size_t>(hidden_size_) * kv_size;
        uint16_t* vw = wcur; wcur += static_cast<size_t>(hidden_size_) * kv_size;
        uint16_t* ow = wcur; wcur += static_cast<size_t>(hidden_size_) * qkv_size;
        float* qb = bcur; bcur += qkv_size;
        float* kb = bcur; bcur += kv_size;
        float* vb = bcur; bcur += kv_size;
        attn_layers_.back()->attach_weight_buffers(qw, kw, vw, ow, qb, kb, vb);
        size_t aw = static_cast<size_t>(i) * (2 * qkv_size + 2 * kv_size);
        attn_layers_.back()->attach_work_buffers(
            attn_work_base + aw,
            attn_work_base + aw + qkv_size,
            attn_work_base + aw + qkv_size + kv_size,
            attn_out_base + static_cast<size_t>(i) * qkv_size);


        mlp_layers_.push_back(std::make_unique<MLP>(hidden_size_, intermediate_size_, "silu"));
        uint16_t* gw = wcur; wcur += static_cast<size_t>(hidden_size_) * static_cast<size_t>(intermediate_size_);
        uint16_t* uw = wcur; wcur += static_cast<size_t>(hidden_size_) * static_cast<size_t>(intermediate_size_);
        uint16_t* dw = wcur; wcur += static_cast<size_t>(hidden_size_) * static_cast<size_t>(intermediate_size_);
        mlp_layers_.back()->attach_weight_buffers(gw, uw, dw);
        mlp_layers_.back()->attach_work_buffers(
            mlp_gate_base + static_cast<size_t>(i) * static_cast<size_t>(intermediate_size_),
            mlp_up_base + static_cast<size_t>(i) * static_cast<size_t>(intermediate_size_),
            mlp_out_base + static_cast<size_t>(i) * static_cast<size_t>(intermediate_size_));

    }
    lm_head_ = std::make_unique<Linear>(hidden_size_, vocab_size_, false);


    // 2. Initialize layers (old placeholder removed)

    // KV cache from pool
    size_t num_blocks = kv_block_count_;
    size_t kv_layer_bytes = num_blocks * config_.block_size * static_cast<size_t>(num_kv_heads_)
                            * static_cast<size_t>(head_dim_) * sizeof(float);
    void* kv_k = gpu_pool_->alloc(kv_layer_bytes * static_cast<size_t>(num_layers_));
    void* kv_v = gpu_pool_->alloc(kv_layer_bytes * static_cast<size_t>(num_layers_));
    cudaMemset(kv_k, 0, kv_layer_bytes * static_cast<size_t>(num_layers_));
    cudaMemset(kv_v, 0, kv_layer_bytes * static_cast<size_t>(num_layers_));
    kv_cache_ = std::make_unique<KVCache>(num_layers_, num_blocks, config_.block_size,
                                           num_kv_heads_, head_dim_, kv_k, kv_v);

    // Load safetensors weights
    std::string safetensors_path = path + "/model.safetensors";
    ModelLoader loader;
    WeightMap wm;
    if (!loader.load_auto(safetensors_path, wm)) {
        VLLM_ERROR("Failed to load: {}", safetensors_path);
        return false;
    }

    // Embedding
    auto it = wm.tensors.find("model.embed_tokens.weight");
    if (it != wm.tensors.end()) {
        embed_->set_weights(it->second.data());
        // Share embed's GPU buffer with lm_head for tied weights
        lm_head_->use_device_pointer(embed_->weight_device_ptr());
    }

    // Per-layer weights
    for (int i = 0; i < num_layers_; i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto& attn = attn_layers_[static_cast<size_t>(i)];
        auto& mlp  = mlp_layers_[static_cast<size_t>(i)];

        // Attention weights (Qwen has bias in Q/K/V proj)
        auto* qw = wm.tensors[prefix + "self_attn.q_proj.weight"].data();
        auto* kw = wm.tensors[prefix + "self_attn.k_proj.weight"].data();
        auto* vw = wm.tensors[prefix + "self_attn.v_proj.weight"].data();
        auto* ow = wm.tensors[prefix + "self_attn.o_proj.weight"].data();
        float* qb = nullptr, *kb = nullptr, *vb = nullptr;
        auto qb_it = wm.tensors.find(prefix + "self_attn.q_proj.bias");
        auto kb_it = wm.tensors.find(prefix + "self_attn.k_proj.bias");
        auto vb_it = wm.tensors.find(prefix + "self_attn.v_proj.bias");
        if (qb_it != wm.tensors.end()) qb = qb_it->second.data();
        if (kb_it != wm.tensors.end()) kb = kb_it->second.data();
        if (vb_it != wm.tensors.end()) vb = vb_it->second.data();
        attn->set_weights(qw, kw, vw, ow, qb, kb, vb);

        // MLP weights
        auto* gw = wm.tensors[prefix + "mlp.gate_proj.weight"].data();
        auto* uw = wm.tensors[prefix + "mlp.up_proj.weight"].data();
        auto* dw = wm.tensors[prefix + "mlp.down_proj.weight"].data();
        mlp->set_weights(gw, uw, dw);
    }

    // RMSNorm weights
    for (int i = 0; i < num_layers_; i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto* in_w = wm.tensors[prefix + "input_layernorm.weight"].data();
        auto* post_w = wm.tensors[prefix + "post_attention_layernorm.weight"].data();
        VLLM_CUDA_CHECK(cudaMemcpy(input_norm_weights_[static_cast<size_t>(i)], in_w,
            static_cast<size_t>(hidden_size_) * sizeof(float), cudaMemcpyHostToDevice));
        VLLM_CUDA_CHECK(cudaMemcpy(post_norm_weights_[static_cast<size_t>(i)], post_w,
            static_cast<size_t>(hidden_size_) * sizeof(float), cudaMemcpyHostToDevice));
    }
    auto fin_it = wm.tensors.find("model.norm.weight");
    if (fin_it != wm.tensors.end()) {
        VLLM_CUDA_CHECK(cudaMemcpy(final_norm_weight_, fin_it->second.data(),
            static_cast<size_t>(hidden_size_) * sizeof(float), cudaMemcpyHostToDevice));
    }

    // lm_head (shared with embed_tokens if not present)
    auto lm_it = wm.tensors.find("lm_head.weight");
    if (lm_it != wm.tensors.end()) {
        lm_head_->set_weights(lm_it->second.data());
    } else {
        // Tied: share embed's GPU buffer to avoid double allocation
        VLLM_INFO("QwenModel: lm_head tied with embed, sharing buffer\n");
    }

    VLLM_INFO("QwenModel weights loaded ({} layers)", num_layers_);
    return true;
}

void QwenModel::forward(float* logits, const int32_t* input_ids, int batch_size, int* context_len,
                        const std::vector<int32_t>& block_table) {
    cudaStream_t stream = 0;

    // 1. Embedding
    embed_->forward(hidden_buffer_, input_ids, batch_size, stream);

    // 2. Transformer layers
    int ctx = *context_len;
    for (int layer = 0; layer < num_layers_; layer++) {
        // input RMSNorm
        kernels::rms_norm(norm_buffer_, hidden_buffer_, input_norm_weights_[static_cast<size_t>(layer)],
                          batch_size, hidden_size_, 1e-6f, stream);

        // self-attention with residual
        attn_layers_[static_cast<size_t>(layer)]->forward(
            hidden_buffer_, norm_buffer_, *kv_cache_,
            block_table, ctx, batch_size, layer, stream);

        // post RMSNorm
        kernels::rms_norm(norm_buffer_, hidden_buffer_, post_norm_weights_[static_cast<size_t>(layer)],
                          batch_size, hidden_size_, 1e-6f, stream);

        // MLP with residual
        mlp_layers_[static_cast<size_t>(layer)]->forward(
            hidden_buffer_, norm_buffer_, batch_size, stream);

    }

    // 3. Final RMSNorm
    kernels::rms_norm(norm_buffer_, hidden_buffer_, final_norm_weight_,
                      batch_size, hidden_size_, 1e-6f, stream);

    // 4. LM head
    lm_head_->forward(logits, norm_buffer_, batch_size, stream);

    *context_len = ctx + batch_size;
}
REGISTER_MODEL("qwen", QwenModel);

}  // namespace vllm
