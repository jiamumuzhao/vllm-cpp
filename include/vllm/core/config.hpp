#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vllm {

// ── Quantization mode ──────────────────────────────────────
enum class QuantMode {
    kNone = 0,    // FP16 full precision
    kW8A8,        // INT8 weights + INT8 activations
    kW4A16,       // INT4 weights (GPTQ/AWQ) + FP16 activations
    kFP8,         // Blackwell native FP8
};

// ── Sampling strategy ──────────────────────────────────────
enum class SamplingStrategy {
    kGreedy = 0,
    kTopK,
    kTopP,
    kTopKTopP,
    kMinP,
};

// ── Scheduling policy ──────────────────────────────────────
enum class SchedulePolicy {
    kFCFS = 0,     // First-Come First-Served
    kPriority,     // Priority-based
    kFairShare,    // Fair-share across request groups
};

// ── Engine configuration ───────────────────────────────────
struct EngineConfig {
    // Model
    std::string model_path;
    std::string model_arch = "llama";     // "llama", "qwen", "gemma", "mistral"
    std::string tokenizer_path;

    // Precision
    QuantMode quant_mode = QuantMode::kNone;

    // Memory
    size_t gpu_memory_utilization = 90;    // percentage (0-100)
    size_t block_size = 16;                // tokens per KV cache block
    size_t max_num_seqs = 64;              // max concurrent sequences
    size_t max_model_len = 8192;           // max context length
    size_t max_num_batched_tokens = 16384; // max tokens in one batch

    // Scheduling
    SchedulePolicy schedule_policy = SchedulePolicy::kFCFS;
    size_t max_tokens_per_step = 512;      // chunked-prefill chunk size

    // Sampling defaults
    float temperature = 1.0f;
    float top_p = 1.0f;
    int32_t top_k = -1;                    // -1 means disabled
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;

    // Server
    int32_t port = 8000;
    bool enable_prefix_caching = true;
};

}  // namespace vllm
