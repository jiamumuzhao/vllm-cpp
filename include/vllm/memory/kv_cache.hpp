#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vllm/core/config.hpp"

namespace vllm {

class KVCache {
  public:
    KVCache(size_t num_layers, size_t num_blocks, size_t block_size,
            size_t num_kv_heads, size_t head_dim, void* external_k = nullptr,
            void* external_v = nullptr);
    ~KVCache();

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    // Get device pointers for block/head.
    void* k_ptr(size_t layer, size_t physical_block);
    void* v_ptr(size_t layer, size_t physical_block);

    // Zero out a block.
    void clear_block(size_t layer, size_t physical_block);

    size_t num_layers() const { return num_layers_; }
    size_t num_blocks() const { return num_blocks_; }
    size_t block_size() const { return block_size_; }
    size_t num_kv_heads() const { return num_kv_heads_; }
    size_t head_dim() const { return head_dim_; }

    // Total GPU memory used by KV cache in bytes.
    size_t total_bytes() const;

  private:
    size_t num_layers_;
    size_t num_blocks_;
    size_t block_size_;
    size_t num_kv_heads_;
    size_t head_dim_;

    // Block storage: contiguous allocation per layer for K and V.
    // Each layer: [num_blocks * block_size * num_kv_heads * head_dim] elements.
    std::vector<void*> k_buffers_;   // one GPU ptr per layer
    std::vector<void*> v_buffers_;   // one GPU ptr per layer
};

}  // namespace vllm
