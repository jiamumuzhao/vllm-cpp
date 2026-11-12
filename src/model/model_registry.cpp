#include "vllm/model/model_registry.hpp"

namespace vllm {

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry registry;
    return registry;
}

void ModelRegistry::register_model(const std::string& arch, ModelFactory factory) {
    factories_[arch] = std::move(factory);
}

std::unique_ptr<ModelBase> ModelRegistry::create(const std::string& arch, const EngineConfig& config) const {
    auto it = factories_.find(arch);
    if (it == factories_.end()) {
        return nullptr;
    }
    return it->second(config);
}

bool ModelRegistry::has(const std::string& arch) const {
    return factories_.find(arch) != factories_.end();
}

}  // namespace vllm
