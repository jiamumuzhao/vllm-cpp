#include "vllm/memory/host_allocator.hpp"

#include <cstring>

#include "vllm/utils/cuda_utils.hpp"
#include "vllm/utils/logger.hpp"

namespace vllm {

HostAllocator::HostAllocator(size_t pool_bytes) : total_bytes_(pool_bytes), offset_(0) {
    VLLM_CUDA_CHECK(cudaMallocHost(&pool_, pool_bytes));
    VLLM_INFO("HostAllocator: allocated {} pinned memory pool", format_bytes(pool_bytes));
}

HostAllocator::~HostAllocator() {
    if (pool_) {
        VLLM_CUDA_CHECK(cudaFreeHost(pool_));
        pool_ = nullptr;
    }
}

void* HostAllocator::allocate(size_t bytes) {
    // Simple bump allocator + free-list for deallocated blocks.
    if (!free_ptrs_.empty()) {
        void* ptr = free_ptrs_.back();
        free_ptrs_.pop_back();
        return ptr;
    }
    if (offset_ + bytes > total_bytes_) {
        VLLM_ERROR("HostAllocator: out of pinned memory");
        return nullptr;
    }
    void* ptr = pool_ + offset_;
    offset_ += bytes;
    return ptr;
}

void HostAllocator::deallocate(void* ptr) {
    free_ptrs_.push_back(ptr);
}

size_t HostAllocator::free_bytes() const {
    return total_bytes_ - offset_ + free_ptrs_.size() * 4096;  // approximate
}

size_t HostAllocator::used_bytes() const {
    return total_bytes_ - free_bytes();
}

}  // namespace vllm
