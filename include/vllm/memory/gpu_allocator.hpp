#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vllm {

class GPUAllocator {
  public:
    GPUAllocator(size_t total_bytes, size_t min_block_size = 4096);
    ~GPUAllocator();

    GPUAllocator(const GPUAllocator&) = delete;
    GPUAllocator& operator=(const GPUAllocator&) = delete;

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    size_t total_bytes() const { return total_bytes_; }
    size_t free_bytes() const;
    size_t used_bytes() const { return total_bytes_ - free_bytes(); }
    bool can_allocate(size_t bytes) const;

  private:
    struct Block {
        size_t offset;
        size_t size;
        bool is_free = true;
        int32_t order = 0;
    };

    size_t total_bytes_;
    size_t min_block_size_;
    int32_t max_order_;
    uint8_t* pool_ = nullptr;
    std::vector<std::vector<int32_t>> free_lists_;
    std::vector<Block> blocks_;
};

}  // namespace vllm
