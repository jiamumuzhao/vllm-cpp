#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace vllm {

class Sequence;

// ── LogicalBlock: a block of KV cache slots ────────────────
// Each block holds block_size_ tokens worth of KV data.
struct LogicalBlock {
    int32_t block_id;         // logical index (per-sequence numbering)
    int32_t physical_id;      // physical index in global KV pool (−1 = unallocated)
    int32_t ref_count;        // reference count (for prefix caching / sharing)
    bool is_committed;        // true if KV data has been written
};

// ── BlockTable maps logical blocks → physical blocks ───────
class BlockTable {
  public:
    explicit BlockTable(size_t block_size, size_t max_blocks);

    // Allocate a new physical block for the given logical position.
    // Returns physical block id, or −1 if out of memory.
    int32_t allocate_block(int32_t logical_idx);

    // Lookup physical block for logical index.
    int32_t physical_id(int32_t logical_idx) const;

    // Release all blocks owned by this table.
    void release_all();

    size_t num_blocks() const { return blocks_.size(); }
    size_t block_size() const { return block_size_; }

    // Snapshot the block table (for preemption / swap).
    std::vector<int32_t> to_id_list() const;

  private:
    size_t block_size_;
    std::vector<LogicalBlock> blocks_;
};

// ── BlockManager: global KV cache block pool ───────────────
// Manages allocation, deallocation, and sharing of physical KV blocks
// across all active sequences.
class BlockManager {
  public:
    BlockManager(size_t block_size, size_t num_gpu_blocks, size_t num_cpu_blocks);

    // Reserve a free physical block. Returns physical_id or nullopt if full.
    std::optional<int32_t> acquire_block();

    // Release a physical block back to the free pool.
    void release_block(int32_t physical_id);

    // Increment reference count (prefix sharing).
    void inc_ref(int32_t physical_id);

    // Decrement reference count; if reaches 0, return to free pool.
    // Returns true if the block was freed.
    bool dec_ref(int32_t physical_id);

    // Allocate blocks for a new sequence (prompt length dependent).
    std::vector<int32_t> allocate_for_sequence(size_t num_tokens);

    // Grow an existing sequence block table to cover num_tokens.
    bool ensure_sequence_capacity(std::vector<int32_t>& block_table, size_t num_tokens);

    // Free all blocks for a sequence (full cleanup).
    void free_sequence(const std::vector<int32_t>& block_table);

    // Capacity queries
    size_t block_size() const { return block_size_; }
    size_t num_total_gpu_blocks() const { return num_gpu_blocks_; }
    size_t num_free_gpu_blocks() const;
    size_t num_total_cpu_blocks() const { return num_cpu_blocks_; }
    size_t num_free_cpu_blocks() const;
    bool can_allocate(size_t num_blocks) const;

    // Prefix caching: find matching prefix blocks from cache.
    // Returns list of physical block IDs that match, or empty if no match.
    std::vector<int32_t> find_prefix_blocks(const std::vector<int32_t>& token_ids,
                                             size_t max_blocks) const;

  private:
    struct PhysicalBlock {
        int32_t ref_count = 0;
        bool is_free = true;
    };

    size_t block_size_;
    size_t num_gpu_blocks_;
    size_t num_cpu_blocks_;
    std::vector<PhysicalBlock> gpu_blocks_;
    std::vector<PhysicalBlock> cpu_blocks_;
    std::vector<int32_t> gpu_free_list_;
    std::vector<int32_t> cpu_free_list_;
};

}  // namespace vllm
