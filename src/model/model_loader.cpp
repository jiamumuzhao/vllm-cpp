#include "vllm/model/model_loader.hpp"
#include "vllm/utils/logger.hpp"
#include "vllm/utils/cuda_utils.hpp"
#include <fstream>
#include <cmath>
#include <cstring>
#include "vllm/utils/json_utils.hpp"

namespace vllm {

// ── Safetensors parser ─────────────────────────────────────
// Format: 8-byte header_size (u64 LE) + JSON header + tensor data

static bool load_safetensors_file(const std::string& path, WeightMap& weights) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        VLLM_ERROR("Cannot open safetensors: {}", path);
        return false;
    }

    // Read 8-byte header size
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    if (!file) {
        VLLM_ERROR("Failed to read safetensors header size");
        return false;
    }

    // Read JSON header
    std::string header_json(header_size, '\0');
    file.read(header_json.data(), static_cast<std::streamsize>(header_size));
    if (!file) {
        VLLM_ERROR("Failed to read safetensors header");
        return false;
    }

    auto header = vllm::json::parse(header_json);
    size_t data_offset = 8 + header_size;
    size_t tensor_count = 0;

    for (auto& [name, meta] : header.get_object()) {
        if (name == "__metadata__") continue;

        std::string dtype_str = meta.contains("dtype") ? meta["dtype"].get_string() : "F32";
        auto shape_arr = meta["shape"];
        std::vector<int32_t> shape;
        size_t numel = 1;
        for (auto& d : shape_arr.get_array()) {
            int32_t dim = d.get_int();
            shape.push_back(dim);
            numel *= static_cast<size_t>(dim);
        }
        auto offsets = meta["data_offsets"];
        size_t start = offsets[0].get_int();
        size_t end   = offsets[1].get_int();

        size_t elem_size = 4;  // default F32
        if (dtype_str == "F16" || dtype_str == "BF16") elem_size = 2;
        else if (dtype_str == "F32") elem_size = 4;

        std::vector<float> tensor_data(numel);

        // Read tensor data from file
        file.seekg(static_cast<std::streamoff>(data_offset + start));
        if (dtype_str == "F32") {
            file.read(reinterpret_cast<char*>(tensor_data.data()),
                      static_cast<std::streamsize>(numel * elem_size));
        } else if (dtype_str == "F16") {
            std::vector<uint16_t> fp16_data(numel);
            file.read(reinterpret_cast<char*>(fp16_data.data()),
                      static_cast<std::streamsize>(numel * elem_size));
            for (size_t i = 0; i < numel; i++) {
                // Convert FP16 to FP32
                uint16_t h = fp16_data[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                float val;
                if (exp == 0) {
                    val = (sign ? -1.0f : 1.0f) * static_cast<float>(mant) / 1024.0f * 6.103515625e-05f;
                } else if (exp == 31) {
                    val = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                } else {
                    val = (sign ? -1.0f : 1.0f) * (1.0f + static_cast<float>(mant) / 1024.0f)
                          * exp2f(static_cast<float>(exp) - 15.0f);
                }
                tensor_data[i] = val;
            }
        } else if (dtype_str == "BF16") {
            std::vector<uint16_t> bf16_data(numel);
            file.read(reinterpret_cast<char*>(bf16_data.data()),
                      static_cast<std::streamsize>(numel * elem_size));
            for (size_t i = 0; i < numel; i++) {
                // BF16 to FP32: just pad mantissa
                uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
                tensor_data[i] = *reinterpret_cast<float*>(&bits);
            }
        }

        weights.tensors[name] = std::move(tensor_data);
        weights.shapes[name] = std::move(shape);
        tensor_count++;
    }

    VLLM_INFO("Safetensors: loaded {} tensors from {}", tensor_count, path);
    return true;
}

// ── Public API ─────────────────────────────────────────────

bool ModelLoader::load_auto(const std::string& path, WeightMap& weights) {
    if (path.find(".safetensors") != std::string::npos) {
        return load_safetensors(path, weights);
    }
    VLLM_ERROR("Unknown format for: {}", path);
    return false;
}

bool ModelLoader::load_safetensors(const std::string& path, WeightMap& weights) {
    return load_safetensors_file(path, weights);
}

bool ModelLoader::load_pytorch_bin(const std::string& path, WeightMap& weights) {
    (void)path; (void)weights;
    return false;
}

bool ModelLoader::load_gguf(const std::string& path, WeightMap& weights) {
    (void)path; (void)weights;
    return false;
}

}  // namespace vllm
