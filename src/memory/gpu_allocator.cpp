#include "vllm/memory/gpu_allocator.hpp"

#include <cstring>

#include "vllm/utils/cuda_utils.hpp"
#include "vllm/utils/logger.hpp"

namespace vllm {

namespace {
constexpr size_t kMinBlockSize = 4096;

size_t next_pow2(size_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}
}  // namespace

GPUAllocator::GPUAllocator(size_t total_bytes, size_t min_block_size)
    : total_bytes_(total_bytes), min_block_size_(std::max(min_block_size, kMinBlockSize)) {
    VLLM_CUDA_CHECK(cudaMalloc(&pool_, total_bytes_));
    VLLM_INFO("GPUAllocator: allocated {} pool", format_bytes(total_bytes_));

    max_order_ = 0;
    size_t sz = total_bytes_;
    while (sz > min_block_size_) {
        sz >>= 1;
        max_order_++;
    }
    free_lists_.resize(static_cast<size_t>(max_order_) + 1);

    // Root block at max_order.
    Block root{0, total_bytes_, true, max_order_};
    blocks_.push_back(root);
    free_lists_[static_cast<size_t>(max_order_)].push_back(0);
}

GPUAllocator::~GPUAllocator() {
    if (pool_) {
        VLLM_CUDA_CHECK(cudaFree(pool_));
        pool_ = nullptr;
    }
}

void* GPUAllocator::allocate(size_t bytes) {
    size_t required = std::max(bytes, min_block_size_);
    int32_t order = 0;
    size_t sz = min_block_size_;
    while (sz < required) {
        sz <<= 1;
        order++;
    }
    if (order > max_order_) {
        VLLM_ERROR("GPUAllocator: requested {} > total pool", format_bytes(bytes));
        return nullptr;
    }

    // Find smallest free block >= required order.
    int32_t found_order = -1;
    for (int32_t o = order; o <= max_order_; o++) {
        if (!free_lists_[static_cast<size_t>(o)].empty()) {
            found_order = o;
            break;
        }
    }
    if (found_order < 0) {
        VLLM_ERROR("GPUAllocator: out of memory (requested {})", format_bytes(bytes));
        return nullptr;
    }

    // Split larger blocks down to required order.
    while (found_order > order) {
        int32_t parent_idx = free_lists_[static_cast<size_t>(found_order)].back();
        free_lists_[static_cast<size_t>(found_order)].pop_back();
        blocks_[static_cast<size_t>(parent_idx)].is_free = false;

        Block left;
        left.offset = blocks_[static_cast<size_t>(parent_idx)].offset;
        left.size = blocks_[static_cast<size_t>(parent_idx)].size / 2;
        left.order = found_order - 1;
        left.is_free = true;
        int32_t left_idx = static_cast<int32_t>(blocks_.size());
        blocks_.push_back(left);

        Block right;
        right.offset = left.offset + left.size;
        right.size = left.size;
        right.order = left.order;
        right.is_free = true;
        int32_t right_idx = static_cast<int32_t>(blocks_.size());
        blocks_.push_back(right);

        free_lists_[static_cast<size_t>(left.order)].push_back(left_idx);
        free_lists_[static_cast<size_t>(right.order)].push_back(right_idx);
        found_order--;
    }

    int32_t idx = free_lists_[static_cast<size_t>(order)].back();
    free_lists_[static_cast<size_t>(order)].pop_back();
    blocks_[static_cast<size_t>(idx)].is_free = false;
    return pool_ + blocks_[static_cast<size_t>(idx)].offset;
}

void GPUAllocator::deallocate(void* ptr) {
    size_t offset = static_cast<uint8_t*>(ptr) - pool_;
    // Find the block by offset.
    for (size_t i = 0; i < blocks_.size(); i++) {
        if (blocks_[i].offset == offset && !blocks_[i].is_free) {
            blocks_[i].is_free = true;
            free_lists_[static_cast<size_t>(blocks_[i].order)].push_back(static_cast<int32_t>(i));
            return;
        }
    }
    VLLM_WARN("GPUAllocator: deallocate unknown pointer {}", static_cast<void*>(ptr));
}

size_t GPUAllocator::free_bytes() const {
    size_t free = 0;
    for (const auto& b : blocks_) {
        if (b.is_free) free += b.size;
    }
    return free;
}

bool GPUAllocator::can_allocate(size_t bytes) const {
    size_t required = std::max(bytes, min_block_size_);
    return required <= free_bytes();
}

}  // namespace vllm
