#include "vllm/memory/kv_cache.hpp"

#include "vllm/utils/cuda_utils.hpp"
#include "vllm/utils/logger.hpp"

namespace vllm {

KVCache::KVCache(size_t num_layers, size_t num_blocks, size_t block_size,
                 size_t num_kv_heads, size_t head_dim, void* external_k, void* external_v)
    : num_layers_(num_layers), num_blocks_(num_blocks), block_size_(block_size),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim) {
    size_t bytes_per_block = block_size_ * num_kv_heads_ * head_dim_ * sizeof(float);
    size_t layer_bytes = num_blocks_ * bytes_per_block;

    k_buffers_.resize(num_layers_);
    v_buffers_.resize(num_layers_);

    void* k_all = external_k;
    void* v_all = external_v;
    if (!k_all || !v_all) {
        size_t total = layer_bytes * num_layers_;
        VLLM_CUDA_CHECK(cudaMalloc(&k_all, total));
        VLLM_CUDA_CHECK(cudaMalloc(&v_all, total));
        VLLM_CUDA_CHECK(cudaMemset(k_all, 0, total));
        VLLM_CUDA_CHECK(cudaMemset(v_all, 0, total));
    }
    for (size_t i = 0; i < num_layers_; i++) {
        k_buffers_[i] = static_cast<uint8_t*>(k_all) + i * layer_bytes;
        v_buffers_[i] = static_cast<uint8_t*>(v_all) + i * layer_bytes;
    }

    VLLM_INFO("KVCache: {} layers x {} blocks x {} tokens, {} kv-heads, {} dim = {}",
              num_layers_, num_blocks_, block_size_, num_kv_heads_, head_dim_,
              format_bytes(total_bytes()));
}

KVCache::~KVCache() {
    // buffers owned by model GPU pool when external buffers were supplied
}

void* KVCache::k_ptr(size_t layer, size_t physical_block) {
    size_t block_stride = block_size_ * num_kv_heads_ * head_dim_;
    size_t offset = physical_block * block_stride;
    return static_cast<float*>(k_buffers_[layer]) + offset;
}

void* KVCache::v_ptr(size_t layer, size_t physical_block) {
    size_t block_stride = block_size_ * num_kv_heads_ * head_dim_;
    size_t offset = physical_block * block_stride;
    return static_cast<float*>(v_buffers_[layer]) + offset;
}

void KVCache::clear_block(size_t layer, size_t physical_block) {
    size_t block_bytes = block_size_ * num_kv_heads_ * head_dim_ * sizeof(float);
    VLLM_CUDA_CHECK(cudaMemset(k_ptr(layer, physical_block), 0, block_bytes));
    VLLM_CUDA_CHECK(cudaMemset(v_ptr(layer, physical_block), 0, block_bytes));
}

size_t KVCache::total_bytes() const {
    size_t per_layer = num_blocks_ * block_size_ * num_kv_heads_ * head_dim_ * sizeof(float);
    return 2 * num_layers_ * per_layer;
}

}  // namespace vllm
