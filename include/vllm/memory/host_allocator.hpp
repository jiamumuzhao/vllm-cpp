#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vllm {

class HostAllocator {
  public:
    HostAllocator(size_t pool_bytes);
    ~HostAllocator();

    HostAllocator(const HostAllocator&) = delete;
    HostAllocator& operator=(const HostAllocator&) = delete;

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    size_t total_bytes() const { return total_bytes_; }
    size_t free_bytes() const;
    size_t used_bytes() const;

  private:
    size_t total_bytes_;
    uint8_t* pool_ = nullptr;
    size_t offset_ = 0;
    std::vector<void*> free_ptrs_;
};

}  // namespace vllm
