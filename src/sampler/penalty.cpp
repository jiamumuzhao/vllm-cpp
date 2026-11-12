#include "vllm/sampler/penalty.hpp"
#include <algorithm>

namespace vllm {

float apply_repetition_penalty(float* logits, const std::vector<int32_t>& input_ids,
                                int vocab_size, float penalty) {
    if (penalty == 1.0f) return penalty;
    for (auto id : input_ids) {
        if (id >= 0 && id < vocab_size) {
            logits[id] = logits[id] > 0 ? logits[id] / penalty : logits[id] * penalty;
        }
    }
    return penalty;
}

float apply_frequency_presence_penalty(float* logits, const std::vector<int32_t>& output_ids,
                                        int vocab_size, float freq_penalty, float pres_penalty) {
    if (freq_penalty == 0.0f && pres_penalty == 0.0f) return 0.0f;
    std::vector<int> counts(vocab_size, 0);
    for (auto id : output_ids) {
        if (id >= 0 && id < vocab_size) counts[id]++;
    }
    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] > 0) {
            logits[i] -= static_cast<float>(counts[i]) * freq_penalty + pres_penalty;
        }
    }
    return freq_penalty + pres_penalty;
}

}  // namespace vllm
