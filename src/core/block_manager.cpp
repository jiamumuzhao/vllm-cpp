#include "vllm/core/block_manager.hpp"
#include <algorithm>
#include <stdexcept>

namespace vllm {

BlockManager::BlockManager(size_t block_size, size_t num_gpu_blocks, size_t num_cpu_blocks)
    : block_size_(block_size), num_gpu_blocks_(num_gpu_blocks), num_cpu_blocks_(num_cpu_blocks) {
    if (block_size_ == 0) throw std::invalid_argument("KV block size must be greater than zero");
    gpu_blocks_.resize(num_gpu_blocks_);
    cpu_blocks_.resize(num_cpu_blocks_);
    for (size_t i = 0; i < num_gpu_blocks_; i++) {
        gpu_free_list_.push_back(static_cast<int32_t>(i));
    }
    for (size_t i = 0; i < num_cpu_blocks_; i++) {
        cpu_free_list_.push_back(static_cast<int32_t>(i));
    }
}

std::optional<int32_t> BlockManager::acquire_block() {
    if (gpu_free_list_.empty()) return std::nullopt;
    int32_t id = gpu_free_list_.back();
    gpu_free_list_.pop_back();
    gpu_blocks_[static_cast<size_t>(id)].is_free = false;
    gpu_blocks_[static_cast<size_t>(id)].ref_count = 1;
    return id;
}

void BlockManager::release_block(int32_t physical_id) {
    if (physical_id < 0 || static_cast<size_t>(physical_id) >= num_gpu_blocks_) return;
    auto& block = gpu_blocks_[static_cast<size_t>(physical_id)];
    block.is_free = true;
    block.ref_count = 0;
    gpu_free_list_.push_back(physical_id);
}

void BlockManager::inc_ref(int32_t physical_id) {
    if (physical_id >= 0 && static_cast<size_t>(physical_id) < num_gpu_blocks_) {
        gpu_blocks_[static_cast<size_t>(physical_id)].ref_count++;
    }
}

bool BlockManager::dec_ref(int32_t physical_id) {
    if (physical_id < 0 || static_cast<size_t>(physical_id) >= num_gpu_blocks_) return false;
    auto& block = gpu_blocks_[static_cast<size_t>(physical_id)];
    block.ref_count--;
    if (block.ref_count <= 0) {
        block.is_free = true;
        gpu_free_list_.push_back(physical_id);
        return true;
    }
    return false;
}

size_t BlockManager::num_free_gpu_blocks() const {
    return gpu_free_list_.size();
}

size_t BlockManager::num_free_cpu_blocks() const {
    return cpu_free_list_.size();
}

bool BlockManager::can_allocate(size_t num_blocks) const {
    return num_free_gpu_blocks() >= num_blocks;
}

std::vector<int32_t> BlockManager::allocate_for_sequence(size_t num_tokens) {
    size_t num_blocks = num_tokens / block_size_ + (num_tokens % block_size_ != 0);
    std::vector<int32_t> blocks;
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; i++) {
        auto blk = acquire_block();
        if (!blk) {
            free_sequence(blocks);
            return {};
        }
        blocks.push_back(*blk);
    }
    return blocks;
}

bool BlockManager::ensure_sequence_capacity(std::vector<int32_t>& block_table, size_t num_tokens) {
    size_t required = num_tokens / block_size_ + (num_tokens % block_size_ != 0);
    const size_t original_size = block_table.size();
    while (block_table.size() < required) {
        auto block = acquire_block();
        if (!block) {
            while (block_table.size() > original_size) {
                release_block(block_table.back());
                block_table.pop_back();
            }
            return false;
        }
        block_table.push_back(*block);
    }
    return true;
}

void BlockManager::free_sequence(const std::vector<int32_t>& block_table) {
    for (auto id : block_table) {
        dec_ref(id);
    }
}

std::vector<int32_t> BlockManager::find_prefix_blocks(const std::vector<int32_t>&, size_t) const {
    // Stub: full implementation requires hash table for prefix caching.
    return {};
}

}  // namespace vllm
