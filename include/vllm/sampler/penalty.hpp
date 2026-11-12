#pragma once
#include <cstdint>
#include <vector>

namespace vllm {

float apply_repetition_penalty(float* logits, const std::vector<int32_t>& input_ids,
                                int vocab_size, float penalty);

float apply_frequency_presence_penalty(float* logits, const std::vector<int32_t>& output_ids,
                                        int vocab_size, float freq_penalty, float pres_penalty);

}  // namespace vllm
