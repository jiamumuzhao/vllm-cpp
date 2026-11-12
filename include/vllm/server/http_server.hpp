#pragma once
#include <csignal>
#include <memory>
#include <string>
#include "vllm/core/config.hpp"

namespace vllm {

class Engine;

class HttpServer {
  public:
    HttpServer(const EngineConfig& config, Engine& engine);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    bool start();
    void stop();
    void join(const volatile sig_atomic_t* stop_flag = nullptr);

    int port() const { return config_.port; }

  private:
    void setup_routes();

    EngineConfig config_;
    Engine& engine_;
    int listen_fd_ = -1;  // cpp-httplib Server opaque ptr
    bool running_ = false;
};

}  // namespace vllm
