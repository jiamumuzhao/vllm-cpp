#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace vllm {

using TokenId = int32_t;

enum class SequenceStatus {
    kWaiting = 0,
    kRunning,
    kFinished,
    kPreempted,
};

struct LogprobEntry {
    TokenId token;
    float logprob;
    bool is_chosen;
};

class Sequence {
  public:
    Sequence(int64_t seq_id, std::vector<TokenId> prompt_tokens, size_t max_output_len,
             float temperature = 1.0f, float top_p = 1.0f, int32_t top_k = -1);

    int64_t seq_id() const { return seq_id_; }
    SequenceStatus status() const { return status_; }
    const std::vector<TokenId>& prompt_tokens() const { return prompt_tokens_; }
    const std::vector<TokenId>& output_tokens() const { return output_tokens_; }
    size_t prompt_len() const { return prompt_tokens_.size(); }
    size_t output_len() const { return output_tokens_.size(); }
    size_t total_len() const { return prompt_tokens_.size() + output_tokens_.size(); }
    size_t max_output_len() const { return max_output_len_; }
    size_t replay_len() const { return prompt_len() + (output_tokens_.empty() ? 0 : output_tokens_.size() - 1); }
    TokenId replay_token(size_t position) const {
        return position < prompt_len() ? prompt_tokens_[position] : output_tokens_[position - prompt_len()];
    }
    bool needs_recompute() const { return needs_recompute_; }
    void set_needs_recompute(bool value) { needs_recompute_ = value; }
    float temperature() const { return temperature_; }
    float top_p() const { return top_p_; }
    int32_t top_k() const { return top_k_; }

    void set_status(SequenceStatus s) { status_ = s; }
    void append_token(TokenId token);
    bool token_generated_this_step() const { return token_generated_this_step_; }
    void clear_token_generated_this_step() { token_generated_this_step_ = false; }
    void clear_output_tokens() { output_tokens_.clear(); }
    bool is_finished() const { return status_ == SequenceStatus::kFinished; }

    const std::vector<int32_t>& block_table() const { return block_table_; }
    std::vector<int32_t>& block_table() { return block_table_; }

    const std::vector<LogprobEntry>& logprobs() const { return logprobs_; }
    void set_logprobs(std::vector<LogprobEntry> lps) { logprobs_ = std::move(lps); }

    size_t num_processed_tokens() const { return num_processed_tokens_; }
    void set_num_processed_tokens(size_t n) { num_processed_tokens_ = n; }

  private:
    int64_t seq_id_;
    SequenceStatus status_ = SequenceStatus::kWaiting;
    std::vector<TokenId> prompt_tokens_;
    std::vector<TokenId> output_tokens_;
    size_t max_output_len_;
    float temperature_;
    float top_p_;
    int32_t top_k_;
    std::vector<int32_t> block_table_;
    std::vector<LogprobEntry> logprobs_;
    size_t num_processed_tokens_ = 0;
    bool needs_recompute_ = false;
    bool token_generated_this_step_ = false;
};

class SequenceGroup {
  public:
    explicit SequenceGroup(int64_t group_id);

    int64_t group_id() const { return group_id_; }
    size_t num_seqs() const { return seqs_.size(); }
    const std::vector<std::shared_ptr<Sequence>>& seqs() const { return seqs_; }

    void add_seq(std::shared_ptr<Sequence> seq);
    bool all_finished() const;
    std::shared_ptr<Sequence> best_seq() const;

  private:
    int64_t group_id_;
    std::vector<std::shared_ptr<Sequence>> seqs_;
};

}  // namespace vllm
