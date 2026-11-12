#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace vllm {

enum class AttentionBackend {
    kPagedAttention,
    kFlashAttention,
    kAuto,
};

inline std::string to_string(AttentionBackend b) {
    switch (b) {
        case AttentionBackend::kPagedAttention: return "PagedAttention";
        case AttentionBackend::kFlashAttention: return "FlashAttention";
        case AttentionBackend::kAuto: return "Auto";
    }
    return "Unknown";
}

}  // namespace vllm
