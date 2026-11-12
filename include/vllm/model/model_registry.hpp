#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "vllm/core/config.hpp"

namespace vllm {

class ModelBase {
  public:
    virtual ~ModelBase() = default;
    virtual bool load_weights(const std::string& path) = 0;
    virtual void forward(float* logits, const int32_t* input_ids, int batch_size,
                         int* context_len, const std::vector<int32_t>& block_table) = 0;
    virtual int vocab_size() const = 0;
    virtual int hidden_size() const = 0;
    virtual int num_layers() const = 0;
    virtual int num_heads() const = 0;
    virtual int num_kv_heads() const = 0;
    virtual int head_dim() const = 0;
    virtual int intermediate_size() const = 0;
    virtual int max_seq_len() const = 0;
    virtual size_t num_kv_blocks() const = 0;
};

using ModelFactory = std::function<std::unique_ptr<ModelBase>(const EngineConfig&)>;

class ModelRegistry {
  public:
    static ModelRegistry& instance();
    void register_model(const std::string& arch, ModelFactory factory);
    std::unique_ptr<ModelBase> create(const std::string& arch, const EngineConfig& config) const;
    bool has(const std::string& arch) const;
  private:
    ModelRegistry() = default;
    std::unordered_map<std::string, ModelFactory> factories_;
};

#define REGISTER_MODEL(arch_name, ModelClass)                                       \
    inline bool _registered_##ModelClass = []() {                                    \
        ModelRegistry::instance().register_model(                                    \
            arch_name,                                                               \
            [](const EngineConfig& cfg) -> std::unique_ptr<ModelBase> {              \
                return std::make_unique<ModelClass>(cfg);                            \
            });                                                                      \
        return true;                                                                 \
    }()

}  // namespace vllm
