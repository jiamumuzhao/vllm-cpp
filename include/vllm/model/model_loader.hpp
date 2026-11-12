#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm {

struct WeightMap {
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, std::vector<int32_t>> shapes;  // name -> [dim0, dim1, ...]
};

class ModelLoader {
  public:
    ModelLoader() = default;

    // Load from safetensors file(s). Supports both single-file and sharded.
    bool load_safetensors(const std::string& path, WeightMap& weights);

    // Load from PyTorch .bin checkpoint.
    bool load_pytorch_bin(const std::string& path, WeightMap& weights);

    // Load from GGUF file.
    bool load_gguf(const std::string& path, WeightMap& weights);

    // Auto-detect format from file extension and load.
    bool load_auto(const std::string& path, WeightMap& weights);

  private:
    bool load_safetensors_single(const std::string& path, WeightMap& weights);
};

}  // namespace vllm
