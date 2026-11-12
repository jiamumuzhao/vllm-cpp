#include "vllm/sampler/sampler.hpp"
#include "vllm/sampler/topk_topp.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace vllm {

Sampler::Sampler() = default;

TokenId Sampler::greedy(const float* logits, int vocab_size) {
    TokenId best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

SamplerOutput Sampler::sample(const float* logits, int vocab_size,
                                float temperature, float top_p, int32_t top_k,
                                float repetition_penalty,
                                const std::vector<TokenId>& prev_tokens) {
    std::vector<float> probs(logits, logits + vocab_size);

    if (repetition_penalty != 1.0f && !prev_tokens.empty()) {
        apply_repetition_penalty(probs.data(), vocab_size, prev_tokens, repetition_penalty);
    }

    if (temperature > 0.0f && temperature != 1.0f) {
        apply_temperature(probs.data(), vocab_size, temperature);
    }

    SamplerOutput out;
    if (temperature <= 0.0f) {
        out.token = greedy(probs.data(), vocab_size);
        float max_logit = probs[out.token];
        float sum_exp = 0.0f;
        for (float logit : probs) sum_exp += std::exp(logit - max_logit);
        out.logprob = -std::log(sum_exp);
        return out;
    }

    sampler::softmax(probs.data(), vocab_size);

    std::vector<int32_t> indices(vocab_size);
    for (int i = 0; i < vocab_size; i++) indices[i] = i;

    int kept = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        kept = sampler::top_k_filter(probs.data(), indices.data(), vocab_size, top_k);
    }
    if (top_p < 1.0f || top_k > 0) {
        kept = sampler::top_p_filter(probs.data(), indices.data(), kept, top_p);
    }

    std::vector<float> selected(static_cast<size_t>(kept));
    float total = 0.0f;
    for (int i = 0; i < kept; ++i) {
        selected[i] = probs[indices[i]];
        total += selected[i];
    }
    for (float& probability : selected) probability /= total;

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int32_t sampled = sampler::random_sample(selected.data(), kept, dist(rng));
    out.token = indices[sampled];
    out.logprob = std::log(selected[sampled]);

    for (int i = 0; i < std::min(5, kept); i++) {
        out.top_logprobs.emplace_back(indices[i], std::log(selected[i]));
    }

    return out;
}

void Sampler::apply_repetition_penalty(float* logits, int vocab_size,
                                        const std::vector<TokenId>& prev_tokens, float penalty) {
    std::vector<bool> seen(static_cast<size_t>(vocab_size), false);
    for (auto t : prev_tokens) {
        if (t >= 0 && t < vocab_size) {
            if (seen[t]) continue;
            seen[t] = true;
            if (logits[t] > 0) logits[t] /= penalty;
            else logits[t] *= penalty;
        }
    }
}

void Sampler::apply_temperature(float* logits, int vocab_size, float temperature) {
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}

}  // namespace vllm
