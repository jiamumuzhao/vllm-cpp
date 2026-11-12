#include "vllm/core/engine.hpp"
#include "vllm/core/scheduler.hpp"
#include "vllm/core/block_manager.hpp"
#include "vllm/model/llama.hpp"
#include "vllm/model/qwen.hpp"
#include "vllm/model/gemma.hpp"
#include "vllm/model/model_registry.hpp"
#include "vllm/sampler/sampler.hpp"
#include "vllm/tokenizer/tokenizer.hpp"
#include "vllm/utils/cuda_utils.hpp"
#include "vllm/utils/logger.hpp"
#include <algorithm>
#include <stdexcept>

namespace vllm {

Engine::Engine(const EngineConfig& config) : config_(config) {}

Engine::~Engine() {
    if (input_ids_d_) cudaFree(input_ids_d_);
    if (logits_d_) cudaFree(logits_d_);
    if (logits_cpu_) cudaFreeHost(logits_cpu_);
}

bool Engine::initialize() {
    ModelRegistry::instance().register_model("llama", [](const EngineConfig& c) { return std::make_unique<LlamaModel>(c); });
    ModelRegistry::instance().register_model("qwen",  [](const EngineConfig& c) { return std::make_unique<QwenModel>(c); });
    ModelRegistry::instance().register_model("gemma", [](const EngineConfig& c) { return std::make_unique<GemmaModel>(c); });
    VLLM_INFO("Initializing vLLM-CPP Engine");

    // Create tokenizer; fall back to model directory if not specified.
    tokenizer_ = std::make_unique<SimpleTokenizer>();
    if (!config_.tokenizer_path.empty()) {
        tokenizer_->load(config_.tokenizer_path);
    } else if (!config_.model_path.empty()) {
        tokenizer_->load(config_.model_path);
    }

    // Create model via registry.
    model_ = ModelRegistry::instance().create(config_.model_arch, config_);
    if (!model_) {
        VLLM_ERROR("Unknown model architecture: {}", config_.model_arch);
        return false;
    }

    // Load weights.
    if (!model_->load_weights(config_.model_path)) {
        VLLM_ERROR("Failed to load model weights from {}", config_.model_path);
        return false;
    }

    // The model owns the backing KV storage. Its block count is computed from
    // the same GPU memory budget when that storage is allocated, so the block
    // manager must expose exactly that capacity instead of estimating a second
    // pool from the already-reduced free-memory figure.
    size_t num_gpu_blocks = model_->num_kv_blocks();
    block_manager_ = std::make_unique<BlockManager>(config_.block_size, num_gpu_blocks, 0);

    size_t block_bytes = config_.block_size * model_->num_kv_heads() * model_->head_dim() * sizeof(float);
    VLLM_INFO("Block manager: {} GPU blocks ({} each)", num_gpu_blocks, format_bytes(block_bytes));

    // Create scheduler and sampler.
    scheduler_ = std::make_unique<Scheduler>(config_, *block_manager_);
    sampler_ = std::make_unique<Sampler>();

    // GPU buffers for single-request inference loop.
    // Allocate from model pool to avoid extra cudaMalloc.
    auto* pool_ptr = dynamic_cast<QwenModel*>(model_.get());
    if (!pool_ptr) {
        VLLM_ERROR("engine buffers require QwenModel pool access\n");
        return false;
    }
    input_ids_d_ = static_cast<int32_t*>(pool_ptr->gpu_pool_alloc(max_batch_tokens_ * sizeof(int32_t)));
    logits_d_ = static_cast<float*>(pool_ptr->gpu_pool_alloc(
        max_batch_tokens_ * static_cast<size_t>(model_->vocab_size()) * sizeof(float)));
    VLLM_CUDA_CHECK(cudaMallocHost(&logits_cpu_, static_cast<size_t>(model_->vocab_size()) * sizeof(float)));

    initialized_ = true;
    VLLM_INFO("Engine initialization complete");
    return true;
}

int64_t Engine::add_request(const std::string& prompt, size_t max_output_len,
                             float temperature, float top_p, int32_t top_k) {
    auto tokens = tokenizer_->encode(prompt);
    if (tokens.empty()) {
        throw std::invalid_argument("prompt must contain at least one token");
    }

    const size_t kv_capacity = block_manager_->num_total_gpu_blocks() * block_manager_->block_size();
    const size_t max_context = std::min({config_.max_model_len,
                                         static_cast<size_t>(model_->max_seq_len()),
                                         kv_capacity});
    if (tokens.size() > max_context) {
        throw std::invalid_argument("prompt exceeds the available context length of " +
                                    std::to_string(max_context) + " tokens");
    }
    if (max_output_len > max_context - tokens.size()) {
        throw std::invalid_argument("prompt tokens plus max_tokens exceeds the available context length of " +
                                    std::to_string(max_context) + " tokens");
    }
    auto group = scheduler_->add_request(std::move(tokens), max_output_len, temperature, top_p, top_k);
    return group->group_id();
}

std::vector<GenerationOutput> Engine::step() {
    auto scheduled = scheduler_->schedule();
    std::vector<GenerationOutput> results;

    if (scheduled.is_empty()) return results;

    for (auto& seq : scheduled.prefill_seqs) seq->clear_token_generated_this_step();
    for (auto& seq : scheduled.decode_seqs) seq->clear_token_generated_this_step();

    // Prefill.
    prefill(scheduled.prefill_seqs);

    // Decode.
    decode(scheduled.decode_seqs);

    // Collect generated tokens from both phases.
    auto all = scheduled.prefill_seqs;
    all.insert(all.end(), scheduled.decode_seqs.begin(), scheduled.decode_seqs.end());
    results = sample(all);

    return results;
}

void Engine::run(StreamCallback callback) {
    while (scheduler_->has_pending_work()) {
        auto outputs = step();
        for (auto& out : outputs) {
            if (callback) callback(out);
        }
    }
}

void Engine::abort_request(int64_t seq_id) {
    scheduler_->abort_request(seq_id);
}

size_t Engine::pending_requests() const {
    return scheduler_->waiting_queue_size() + scheduler_->running_seq_count();
}

std::string Engine::decode(const std::vector<TokenId>& tokens) const {
    return tokenizer_->decode(tokens);
}

void Engine::prefill(const std::vector<std::shared_ptr<Sequence>>& seqs) {
    for (auto& seq : seqs) {
        int ctx = static_cast<int>(seq->num_processed_tokens());
        const bool replaying = seq->needs_recompute();
        const size_t input_len = replaying ? seq->replay_len() : seq->prompt_len();
        fprintf(stderr, "[ENG] prefill seq=%lld len=%zu replay=%d\n", (long long)seq->seq_id(), input_len, replaying);
        // Recompute prompt plus previously consumed output tokens after preemption.
        while (seq->num_processed_tokens() < input_len) {
            TokenId tok = seq->replay_token(seq->num_processed_tokens());
            ctx = static_cast<int>(seq->num_processed_tokens());
            VLLM_CUDA_CHECK(cudaMemcpyAsync(input_ids_d_, &tok, sizeof(TokenId),
                                             cudaMemcpyHostToDevice, 0));
            model_->forward(logits_d_, input_ids_d_, 1, &ctx, seq->block_table());
            seq->set_num_processed_tokens(seq->num_processed_tokens() + 1);
        }
        if (replaying && !seq->output_tokens().empty()) {
            seq->set_needs_recompute(false);
        } else {
            // Sample first output token from final prompt position.
            seq->set_needs_recompute(false);
            sample_one(*seq, ctx);
        }
    }
}

void Engine::decode(const std::vector<std::shared_ptr<Sequence>>& seqs) {
    for (auto& seq : seqs) {
        if (seq->is_finished()) continue;
        fprintf(stderr, "[ENG] decode seq=%lld ctx=%zu\n", (long long)seq->seq_id(), seq->num_processed_tokens());
        TokenId tok = seq->output_tokens().empty() ? seq->prompt_tokens().back()
                                                   : seq->output_tokens().back();
        int ctx = static_cast<int>(seq->num_processed_tokens());
        VLLM_CUDA_CHECK(cudaMemcpyAsync(input_ids_d_, &tok, sizeof(TokenId),
                                         cudaMemcpyHostToDevice, 0));
        model_->forward(logits_d_, input_ids_d_, 1, &ctx, seq->block_table());
        seq->set_num_processed_tokens(static_cast<size_t>(ctx) + 1);
        sample_one(*seq, ctx);
    }
}

std::vector<GenerationOutput> Engine::sample(const std::vector<std::shared_ptr<Sequence>>& seqs) {
    std::vector<GenerationOutput> results;
    for (auto& seq : seqs) {
        if (seq->output_tokens().empty() || !seq->token_generated_this_step()) continue;
        GenerationOutput out;
        out.seq_id = seq->seq_id();
        out.tokens = { seq->output_tokens().back() };
        if (!tokenizer_->is_stop_token(out.tokens.back()))
            out.text = tokenizer_->decode(out.tokens);
        out.finished = seq->is_finished();
        results.push_back(out);
        seq->clear_token_generated_this_step();
    }
    return results;
}

void Engine::sample_one(Sequence& seq, int ctx) {
    fprintf(stderr, "[ENG] sample_one ctx=%d vocab=%d\n", ctx, model_->vocab_size());
    int vocab = model_->vocab_size();
    VLLM_CUDA_CHECK(cudaMemcpyAsync(logits_cpu_, logits_d_, static_cast<size_t>(vocab) * sizeof(float),
                                     cudaMemcpyDeviceToHost, 0));
    VLLM_CUDA_CHECK(cudaDeviceSynchronize());

    auto out = sampler_->sample(logits_cpu_, vocab, seq.temperature(), seq.top_p(),
                                seq.top_k(), config_.repetition_penalty, seq.output_tokens());
    seq.append_token(out.token);

    bool eos = tokenizer_->is_stop_token(out.token);
    bool maxed = seq.output_len() >= seq.max_output_len();
    if (eos || maxed) {
        seq.set_status(SequenceStatus::kFinished);
        scheduler_->finish_sequence(seq.seq_id());
    }
}

}  // namespace vllm
