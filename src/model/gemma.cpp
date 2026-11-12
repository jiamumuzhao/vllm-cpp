#include "vllm/model/gemma.hpp"
namespace vllm {
GemmaModel::GemmaModel(const EngineConfig& config) : config_(config) {
    vocab_size_ = 256000;
    hidden_size_ = 3072;
    num_layers_ = 26;
    num_heads_ = 16;
    num_kv_heads_ = 8;
    head_dim_ = 256;
    intermediate_size_ = 24576;
    max_seq_len_ = 4096;
}
GemmaModel::~GemmaModel() = default;
bool GemmaModel::load_weights(const std::string& path) { (void)path; return true; }
void GemmaModel::forward(float*, const int32_t*, int, int*, const std::vector<int32_t>&) {}
REGISTER_MODEL("gemma", GemmaModel);
}  // namespace vllm
