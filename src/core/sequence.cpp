#include "vllm/core/sequence.hpp"

namespace vllm {

Sequence::Sequence(int64_t seq_id, std::vector<TokenId> prompt_tokens, size_t max_output_len,
                   float temperature, float top_p, int32_t top_k)
    : seq_id_(seq_id), prompt_tokens_(std::move(prompt_tokens)), max_output_len_(max_output_len),
      temperature_(temperature), top_p_(top_p), top_k_(top_k) {}

void Sequence::append_token(TokenId token) {
    output_tokens_.push_back(token);
    token_generated_this_step_ = true;
}

SequenceGroup::SequenceGroup(int64_t group_id) : group_id_(group_id) {}

void SequenceGroup::add_seq(std::shared_ptr<Sequence> seq) {
    seqs_.push_back(std::move(seq));
}

bool SequenceGroup::all_finished() const {
    for (const auto& s : seqs_) {
        if (!s->is_finished()) return false;
    }
    return true;
}

std::shared_ptr<Sequence> SequenceGroup::best_seq() const {
    return seqs_.empty() ? nullptr : seqs_[0];
}

}  // namespace vllm
