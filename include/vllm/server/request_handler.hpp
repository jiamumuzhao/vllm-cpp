#pragma once
#include <string>
#include "vllm/core/config.hpp"
#include "vllm/core/engine.hpp"

namespace vllm {

class Engine;



class RequestHandler {
  public:
    RequestHandler(Engine& engine);

    struct CompletionRequest {
        std::string prompt;
        int max_tokens = 256;
        float temperature = 1.0f;
        float top_p = 1.0f;
        int top_k = -1;
        bool stream = false;
    };

    struct CompletionResponse {
        std::string text;
        int64_t request_id;
        bool finished = false;
    };

    CompletionResponse handle_completion(const CompletionRequest& req, StreamCallback callback = nullptr);

  private:
    Engine& engine_;
};

}  // namespace vllm
