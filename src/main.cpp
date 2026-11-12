#include <cuda_runtime.h>
#include <csignal>
#include <cstdlib>
#include <iostream>

#include "vllm/core/engine.hpp"
#include "vllm/server/http_server.hpp"
#include "vllm/utils/logger.hpp"

namespace {
volatile sig_atomic_t g_running = 1;
void signal_handler(int) { g_running = 0; }


}  // namespace

int main(int argc, char* argv[]) {
    vllm::init_logger("info");

    vllm::EngineConfig config;
    config.model_path = "./models/test-model";
    config.model_arch = "llama";
    config.port = 8000;

    // Parse CLI args (minimal).
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) config.model_path = argv[++i];
        else if (arg == "--port" && i + 1 < argc) config.port = std::stoi(argv[++i]);
        else if (arg == "--arch" && i + 1 < argc) config.model_arch = argv[++i];
        else if (arg == "--quantization" && i + 1 < argc) {
            std::string q = argv[++i];
            if (q == "w4a16") config.quant_mode = vllm::QuantMode::kW4A16;
            else if (q == "w8a8") config.quant_mode = vllm::QuantMode::kW8A8;
            else if (q == "fp8") config.quant_mode = vllm::QuantMode::kFP8;
        }
    }

    VLLM_INFO("vLLM-CPP v0.1.0 starting");
    VLLM_INFO("Model: {} Arch: {}", config.model_path, config.model_arch);

        // Force CUDA context creation
    cudaSetDevice(0);
    cudaFree(0);

    vllm::Engine engine(config);
    cudaSetDevice(0); cudaFree(0);
    if (!engine.initialize()) {
        VLLM_ERROR("Engine initialization failed");
        return 1;
    }

    vllm::HttpServer server(config, engine);
    if (!server.start()) {
        VLLM_ERROR("Server start failed");
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    VLLM_INFO("Server listening on port {}", config.port);
    server.join(&g_running);

    server.stop();
    VLLM_INFO("vLLM-CPP shutting down");
    return 0;
}
