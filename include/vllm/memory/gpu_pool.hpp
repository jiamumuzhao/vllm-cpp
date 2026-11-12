#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdio>

namespace vllm {

// Simple linear GPU memory pool: one cudaMalloc, slices for every consumer.
// Avoids CUDA 13 driver's large per-allocation granularity.
class GpuMemoryPool {
  public:
    explicit GpuMemoryPool(size_t total_bytes) : total_bytes_(total_bytes) {
        if (cudaMalloc(&pool_, total_bytes_) != cudaSuccess) { fprintf(stderr, "[POOL] FATAL: cudaMalloc %zu bytes failed\n", total_bytes_); pool_ = nullptr; }
    }
    ~GpuMemoryPool() { if (pool_) cudaFree(pool_); }

    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;

    void* alloc(size_t bytes, size_t alignment = 256) {
        size_t offset = (offset_ + alignment - 1) & ~(alignment - 1);
        if (offset + bytes > total_bytes_) {
            fprintf(stderr, "[POOL] OOM: need %zu more bytes, pool %zu, used %zu\n",
                    bytes, total_bytes_, offset);
            return nullptr;
        }
        void* ptr = static_cast<uint8_t*>(pool_) + offset;
        offset_ = offset + bytes;
        return ptr;
    }

    size_t used() const { return offset_; }
    size_t total() const { return total_bytes_; }

  private:
    size_t total_bytes_;
    size_t offset_ = 0;
    void* pool_ = nullptr;
};

}  // namespace vllm
