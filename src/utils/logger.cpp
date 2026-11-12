#include "vllm/utils/logger.hpp"

#ifdef VLLM_USE_SPDLOG
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace vllm {

void init_logger(const std::string& level) {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("vllm", sink);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->set_level(spdlog::level::from_str(level));
    spdlog::set_default_logger(logger);
}

}  // namespace vllm

#else
#include <cstdio>

namespace vllm {

void init_logger(const std::string& level) {
    std::fprintf(stderr, "[INFO]  Logger initialized (level: %s, printf fallback)\n", level.c_str());
}

}  // namespace vllm
#endif
