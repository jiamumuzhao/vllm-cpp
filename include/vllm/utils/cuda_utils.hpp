#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace vllm {

// ── CUDA error checking macro ──────────────────────────────
// Usage: VLLM_CUDA_CHECK(cudaMalloc(&ptr, size));
#define VLLM_CUDA_CHECK(call)                                                               \
    do {                                                                                    \
        cudaError_t _err = (call);                                                          \
        if (_err != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                          \
                         __FILE__, __LINE__, cudaGetErrorString(_err), cudaGetErrorName(_err)); \
            std::abort();                                                                   \
        }                                                                                   \
    } while (0)

// ── CUDA kernel launch error check ─────────────────────────
#define VLLM_KERNEL_CHECK() VLLM_CUDA_CHECK(cudaGetLastError())

// ── Utility: get current GPU device properties ─────────────
inline cudaDeviceProp get_device_props(int device = 0) {
    cudaDeviceProp props{};
    VLLM_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props;
}

// ── Utility: bytes to human-readable string ────────────────
inline std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit_idx]);
    return buf;
}

// ── Utility: round up to multiple ──────────────────────────
template <typename T>
constexpr T round_up(T value, T multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// ── Utility: div up (ceiling division) ─────────────────────
template <typename T>
constexpr T div_up(T a, T b) {
    return (a + b - 1) / b;
}

}  // namespace vllm
