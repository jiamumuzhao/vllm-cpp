#pragma once
#include <cstdint>

namespace vllm {
namespace sampler {

// Top-K filtering in-place on CPU. Returns number of kept tokens.
int32_t top_k_filter(float* logits, int32_t* indices, int vocab_size, int32_t k);

// Top-P (nucleus) filtering in-place on CPU. Returns number of kept tokens.
int32_t top_p_filter(float* logits, int32_t* indices, int vocab_size, float p);

// Softmax in-place.
void softmax(float* logits, int vocab_size);

// Random sample from log-probabilities.
int32_t random_sample(const float* probs, int vocab_size, float random_val);

}  // namespace sampler
}  // namespace vllm
