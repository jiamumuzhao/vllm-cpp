#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "vllm/core/config.hpp"
#include "vllm/core/sequence.hpp"

namespace vllm {

class BlockManager;

// ── SchedulerOutput: the result of one scheduling decision ─
struct SchedulerOutput {
    // Sequences to run prefill for (prompt not yet fully processed).
    std::vector<std::shared_ptr<Sequence>> prefill_seqs;

    // Sequences to run decode for (1 token generation each).
    std::vector<std::shared_ptr<Sequence>> decode_seqs;

    // Total number of tokens across all scheduled prefill chunks.
    size_t num_prefill_tokens = 0;

    // Total number of tokens across all scheduled decode steps.
    size_t num_decode_tokens = 0;

    bool is_empty() const {
        return prefill_seqs.empty() && decode_seqs.empty();
    }
};

// ── Scheduler: request queue + prefill/decode orchestration ─
// Implements continuous batching: new requests can join decode batch
// at any step, and finished requests drop out immediately.
class Scheduler {
  public:
    Scheduler(const EngineConfig& config, BlockManager& block_manager);

    // Add a new request (prompt tokens) to the waiting queue.
    // Returns the sequence pointer for the caller to track.
    std::shared_ptr<SequenceGroup> add_request(std::vector<TokenId> prompt_tokens,
                                                size_t max_output_len,
                                                float temperature = 1.0f,
                                                float top_p = 1.0f,
                                                int32_t top_k = -1);

    // Schedule one iteration: decides which sequences get prefill/decode.
    // Must be called each engine step.
    SchedulerOutput schedule();

    // Mark a sequence as finished (after EOS or stop token).
    void finish_sequence(int64_t seq_id);

    // Abort a request (timeout, client disconnect).
    void abort_request(int64_t seq_id);

    // Query
    size_t waiting_queue_size() const { return waiting_queue_.size(); }
    size_t running_seq_count() const { return running_seqs_.size(); }
    bool has_pending_work() const;

  private:
    // Chunked-prefill: split a long prompt into chunks to bound prefill latency.
    std::vector<TokenId> chunk_prompt(const Sequence& seq, size_t chunk_size) const;

    // Check if there are enough free KV blocks to schedule a sequence.
    bool can_schedule(const Sequence& seq) const;

    // Move a sequence from waiting → running, allocate initial blocks.
    bool promote_to_running(std::shared_ptr<SequenceGroup> group);
    bool preempt_one(const std::shared_ptr<SequenceGroup>& except = nullptr);

    const EngineConfig& config_;
    BlockManager& block_manager_;

    // Request queues
    std::deque<std::shared_ptr<SequenceGroup>> waiting_queue_;
    std::vector<std::shared_ptr<SequenceGroup>> running_seqs_;

    // Monotonic ID counters
    int64_t next_seq_id_ = 0;
    int64_t next_group_id_ = 0;

    // Token budget tracking
    size_t tokens_scheduled_this_step_ = 0;
};

}  // namespace vllm
