#include "vllm/core/scheduler.hpp"
#include "vllm/core/block_manager.hpp"
#include "vllm/utils/logger.hpp"
#include <stdexcept>

namespace vllm {

namespace {
size_t blocks_for_tokens(size_t tokens, size_t block_size) {
    return tokens / block_size + (tokens % block_size != 0);
}
}  // namespace

Scheduler::Scheduler(const EngineConfig& config, BlockManager& block_manager)
    : config_(config), block_manager_(block_manager) {}

std::shared_ptr<SequenceGroup> Scheduler::add_request(std::vector<TokenId> prompt_tokens,
                                                       size_t max_output_len,
                                                       float temperature, float top_p, int32_t top_k) {
    if (prompt_tokens.empty()) throw std::invalid_argument("prompt must contain at least one token");
    if (max_output_len > static_cast<size_t>(-1) - prompt_tokens.size())
        throw std::invalid_argument("requested sequence length overflows size_t");
    const size_t planned_tokens = prompt_tokens.size() + max_output_len;
    const size_t required_blocks = blocks_for_tokens(planned_tokens, block_manager_.block_size());
    if (required_blocks > block_manager_.num_total_gpu_blocks())
        throw std::invalid_argument("request exceeds total KV cache block capacity");
    int64_t seq_id = next_seq_id_++;
    int64_t group_id = next_group_id_++;
    auto seq = std::make_shared<Sequence>(seq_id, std::move(prompt_tokens),
                                           max_output_len, temperature, top_p, top_k);
    auto group = std::make_shared<SequenceGroup>(group_id);
    group->add_seq(seq);
    waiting_queue_.push_back(group);
    return group;
}

SchedulerOutput Scheduler::schedule() {
    SchedulerOutput out;
    tokens_scheduled_this_step_ = 0;

    // Grow the cache as requests make progress. If prompt/replay admission
    // needs space, preempt a running request and put it at the back of FCFS.
    while (!waiting_queue_.empty()) {
        auto group = waiting_queue_.front();
        if (group->all_finished()) {
            waiting_queue_.pop_front();
            continue;
        }
        size_t required_blocks = 0;
        for (const auto& seq : group->seqs()) {
            const size_t tokens_to_rebuild = seq->needs_recompute() ? seq->replay_len() : seq->prompt_len();
            const size_t seq_blocks = blocks_for_tokens(tokens_to_rebuild, block_manager_.block_size());
            if (seq_blocks > static_cast<size_t>(-1) - required_blocks)
                throw std::overflow_error("request KV block count overflows size_t");
            required_blocks += seq_blocks;
        }
        while (!block_manager_.can_allocate(required_blocks) && preempt_one()) {}
        if (!block_manager_.can_allocate(required_blocks)) break;
        waiting_queue_.pop_front();
        if (promote_to_running(group)) {
            running_seqs_.push_back(group);
        } else {
            // Allocation may fail if another allocator changed availability;
            // keep the request queued and try again on a later schedule step.
            waiting_queue_.push_front(group);
            break;
        }
    }

    // Ensure decode has a physical slot before scheduling its forward pass.
    // If the pool is full, evict another sequence; its output history is kept
    // and its KV state is rebuilt when it is admitted again.
    for (size_t i = 0; i < running_seqs_.size(); ++i) {
        auto group = running_seqs_[i];
        for (auto& seq : group->seqs()) {
            if (seq->is_finished() || seq->needs_recompute() ||
                seq->num_processed_tokens() < seq->prompt_len()) continue;
            const size_t next_position = seq->num_processed_tokens() + 1;
            while (!block_manager_.ensure_sequence_capacity(seq->block_table(), next_position)) {
                if (!preempt_one(group)) break;
            }
        }
    }

    // Classify running sequences: new arrivals and preempted sequences rebuild
    // their prefix; active sequences decode one token.
    for (auto& group : running_seqs_) {
        for (auto& seq : group->seqs()) {
            if (seq->is_finished()) continue;

            const size_t prefill_len = seq->needs_recompute() ? seq->replay_len() : seq->prompt_len();
            if (seq->num_processed_tokens() < prefill_len) {
                // Still in prefill phase.
                if (tokens_scheduled_this_step_ < config_.max_tokens_per_step) {
                    out.prefill_seqs.push_back(seq);
                    out.num_prefill_tokens++;
                    tokens_scheduled_this_step_++;
                }
            } else if (block_manager_.ensure_sequence_capacity(
                           seq->block_table(), seq->num_processed_tokens() + 1)) {
                // Decode phase.
                out.decode_seqs.push_back(seq);
                out.num_decode_tokens++;
            }
        }
    }

    return out;
}

void Scheduler::finish_sequence(int64_t seq_id) {
    for (auto& group : running_seqs_) {
        for (auto& seq : group->seqs()) {
            if (seq->seq_id() == seq_id) {
                seq->set_status(SequenceStatus::kFinished);
                block_manager_.free_sequence(seq->block_table());
                seq->block_table().clear();
                return;
            }
        }
    }
    for (auto& group : waiting_queue_) {
        for (auto& seq : group->seqs()) {
            if (seq->seq_id() == seq_id) {
                seq->set_status(SequenceStatus::kFinished);
                if (!seq->block_table().empty()) {
                    block_manager_.free_sequence(seq->block_table());
                    seq->block_table().clear();
                }
                return;
            }
        }
    }
}

void Scheduler::abort_request(int64_t seq_id) {
    finish_sequence(seq_id);
}

bool Scheduler::has_pending_work() const {
    for (const auto& group : running_seqs_) {
        if (!group->all_finished()) return true;
    }
    return !waiting_queue_.empty();
}

bool Scheduler::promote_to_running(std::shared_ptr<SequenceGroup> group) {
    std::vector<std::pair<std::shared_ptr<Sequence>, std::vector<int32_t>>> allocations;
    for (auto& seq : group->seqs()) {
        if (seq->is_finished()) continue;
        const size_t tokens_to_rebuild = seq->needs_recompute() ? seq->replay_len() : seq->prompt_len();
        const size_t required_blocks = blocks_for_tokens(tokens_to_rebuild, block_manager_.block_size());
        auto blocks = block_manager_.allocate_for_sequence(tokens_to_rebuild);
        if (blocks.size() != required_blocks) {
            for (auto& allocation : allocations) block_manager_.free_sequence(allocation.second);
            if (!blocks.empty()) block_manager_.free_sequence(blocks);
            return false;
        }
        allocations.emplace_back(seq, std::move(blocks));
    }
    for (auto& allocation : allocations) {
        allocation.first->block_table() = std::move(allocation.second);
        allocation.first->set_status(SequenceStatus::kRunning);
    }
    return true;
}

bool Scheduler::preempt_one(const std::shared_ptr<SequenceGroup>& except) {
    for (size_t i = running_seqs_.size(); i > 0; --i) {
        auto group = running_seqs_[i - 1];
        if (group == except || group->all_finished()) continue;
        bool has_blocks = false;
        for (auto& seq : group->seqs()) {
            if (seq->block_table().empty()) continue;
            block_manager_.free_sequence(seq->block_table());
            seq->block_table().clear();
            seq->set_num_processed_tokens(0);
            seq->set_needs_recompute(true);
            seq->set_status(SequenceStatus::kPreempted);
            has_blocks = true;
        }
        if (!has_blocks) continue;
        waiting_queue_.push_back(group);
        running_seqs_.erase(running_seqs_.begin() + static_cast<std::ptrdiff_t>(i - 1));
        return true;
    }
    return false;
}

}  // namespace vllm
