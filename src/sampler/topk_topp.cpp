#include "vllm/sampler/topk_topp.hpp"
#include <algorithm>
#include <cmath>

namespace vllm {
namespace sampler {

int32_t top_k_filter(float* logits, int32_t* indices, int vocab_size, int32_t k) {
    k = std::max(1, std::min(k, vocab_size));
    auto better = [logits](int32_t a, int32_t b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    };
    if (k < vocab_size)
        std::partial_sort(indices, indices + k, indices + vocab_size, better);
    else
        std::sort(indices, indices + vocab_size, better);
    return k;
}

int32_t top_p_filter(float* logits, int32_t* indices, int vocab_size, float p) {
    std::sort(indices, indices + vocab_size, [logits](int32_t a, int32_t b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    if (p >= 1.0f) return vocab_size;
    float total = 0.0f;
    for (int i = 0; i < vocab_size; ++i) total += logits[indices[i]];
    float sum = 0.0f;
    int kept = 0;
    for (int i = 0; i < vocab_size; i++) {
        sum += logits[indices[i]];
        kept++;
        if (total > 0.0f && sum / total >= p) break;
    }
    return kept;
}

void softmax(float* logits, int n) {
    float max_val = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    for (int i = 0; i < n; i++) {
        logits[i] /= sum;
    }
}

int32_t random_sample(const float* probs, int vocab_size, float random_val) {
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (random_val < cumsum) return i;
    }
    return vocab_size - 1;
}

}  // namespace sampler
}  // namespace vllm
