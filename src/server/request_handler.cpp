#include "vllm/server/request_handler.hpp"
#include "vllm/core/engine.hpp"

namespace vllm {

RequestHandler::RequestHandler(Engine& engine) : engine_(engine) {}

RequestHandler::CompletionResponse RequestHandler::handle_completion(
    const CompletionRequest& req, StreamCallback) {
    int64_t id = engine_.add_request(req.prompt, static_cast<size_t>(req.max_tokens),
                                      req.temperature, req.top_p, req.top_k);
    CompletionResponse resp;
    resp.request_id = id;
    resp.text = "[generated]";
    resp.finished = true;
    return resp;
}

}  // namespace vllm
