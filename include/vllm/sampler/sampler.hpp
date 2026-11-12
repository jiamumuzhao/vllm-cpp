#pragma once
#include <cstdint>
#include <vector>
#include "vllm/core/config.hpp"

namespace vllm {

using TokenId = int32_t;

struct SamplerOutput {
    TokenId token;
    float logprob;
    std::vector<std::pair<TokenId, float>> top_logprobs;
};

class Sampler {
  public:
    Sampler();

    SamplerOutput sample(const float* logits, int vocab_size,
                         float temperature, float top_p, int32_t top_k,
                         float repetition_penalty, const std::vector<TokenId>& prev_tokens);

    // Greedy: argmax.
    static TokenId greedy(const float* logits, int vocab_size);

  private:
    void apply_repetition_penalty(float* logits, int vocab_size,
                                   const std::vector<TokenId>& prev_tokens, float penalty);
    void apply_temperature(float* logits, int vocab_size, float temperature);
};

}  // namespace vllm
