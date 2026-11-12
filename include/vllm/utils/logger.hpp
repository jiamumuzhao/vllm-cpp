#pragma once

#ifdef VLLM_USE_SPDLOG
#include <spdlog/spdlog.h>
#include <memory>

namespace vllm {

void init_logger(const std::string& level = "info");

#define VLLM_TRACE(...) ::spdlog::trace(__VA_ARGS__)
#define VLLM_DEBUG(...) ::spdlog::debug(__VA_ARGS__)
#define VLLM_INFO(...)  ::spdlog::info(__VA_ARGS__)
#define VLLM_WARN(...)  ::spdlog::warn(__VA_ARGS__)
#define VLLM_ERROR(...) ::spdlog::error(__VA_ARGS__)
#define VLLM_CRITICAL(...) ::spdlog::critical(__VA_ARGS__)

}  // namespace vllm

#else  // fallback logging for offline builds

#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vllm {

void init_logger(const std::string& level = "info");

template <typename T>
std::string log_string(T&& value) {
    std::ostringstream out;
    out << std::forward<T>(value);
    return out.str();
}

template <typename... Args>
void log_fallback(const char* level, std::string_view format, Args&&... args) {
    const std::vector<std::string> values{log_string(std::forward<Args>(args))...};
    std::string message;
    message.reserve(format.size() + values.size() * 16);
    size_t arg = 0;
    for (size_t i = 0; i < format.size();) {
        if (i + 1 < format.size() && format[i] == '{' && format[i + 1] == '}' && arg < values.size()) {
            message += values[arg++];
            i += 2;
        } else {
            message += format[i++];
        }
    }
    while (!message.empty() && message.back() == '\n') message.pop_back();
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::fprintf(stderr, "[%s] %s\n", level, message.c_str());
}

#define VLLM_TRACE(...) ::vllm::log_fallback("TRACE", __VA_ARGS__)
#define VLLM_DEBUG(...) ::vllm::log_fallback("DEBUG", __VA_ARGS__)
#define VLLM_INFO(...)  ::vllm::log_fallback("INFO", __VA_ARGS__)
#define VLLM_WARN(...)  ::vllm::log_fallback("WARN", __VA_ARGS__)
#define VLLM_ERROR(...) ::vllm::log_fallback("ERROR", __VA_ARGS__)
#define VLLM_CRITICAL(...) ::vllm::log_fallback("CRIT", __VA_ARGS__)

}  // namespace vllm

#endif  // VLLM_USE_SPDLOG
