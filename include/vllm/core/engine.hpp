#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/core/config.hpp"
#include "vllm/memory/gpu_pool.hpp"
#include "vllm/core/sequence.hpp"

namespace vllm {

class Scheduler;
class BlockManager;
class ModelBase;
class Sampler;
class Tokenizer;

// ── Generation output for a single request ─────────────────
struct GenerationOutput {
    int64_t seq_id;
    std::vector<TokenId> tokens;
    std::string text;                    // decoded text
    bool finished = false;
    std::vector<LogprobEntry> logprobs;
};

// ── Streaming callback: invoked per generated token ────────
using StreamCallback = std::function<void(GenerationOutput)>;

// ── Engine: top-level inference orchestrator ───────────────
// Encapsulates the full pipeline: prefill → decode → sample.
class Engine {
  public:
    explicit Engine(const EngineConfig& config);
    ~Engine();

    // Non-copyable, movable.
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    // Initialize: load model weights, allocate KV cache, warm up.
    bool initialize();

    // Add a generation request. Returns request id for tracking.
    int64_t add_request(const std::string& prompt,
                        size_t max_output_len,
                        float temperature = 1.0f,
                        float top_p = 1.0f,
                        int32_t top_k = -1);

    // Run one engine step (one iteration of the scheduling loop).
    // Returns list of newly generated outputs (may be empty).
    std::vector<GenerationOutput> step();

    // Run the engine loop until all pending requests are completed.
    // Invokes the callback for each generated token.
    void run(StreamCallback callback = nullptr);

    // Abort a specific request.
    void abort_request(int64_t seq_id);

    // Query
    bool is_initialized() const { return initialized_; }
    size_t pending_requests() const;
    const EngineConfig& config() const { return config_; }

    // Decode token IDs to text.
    std::string decode(const std::vector<TokenId>& tokens) const;

  private:
    // Pipeline stages
    void prefill(const std::vector<std::shared_ptr<Sequence>>& seqs);
    void decode(const std::vector<std::shared_ptr<Sequence>>& seqs);
    std::vector<GenerationOutput> sample(const std::vector<std::shared_ptr<Sequence>>& seqs);
    void sample_one(Sequence& seq, int ctx);

    EngineConfig config_;
    bool initialized_ = false;

    std::unique_ptr<BlockManager> block_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelBase> model_;
    std::unique_ptr<Sampler> sampler_;
            std::unique_ptr<Tokenizer> tokenizer_;
    int32_t* input_ids_d_ = nullptr;
    float* logits_d_ = nullptr;
    float* logits_cpu_ = nullptr;
    size_t max_batch_tokens_ = 64;
};

}  // namespace vllm
