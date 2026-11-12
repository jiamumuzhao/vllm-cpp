#include "vllm/server/http_server.hpp"
#include "vllm/core/engine.hpp"
#include "vllm/utils/json_utils.hpp"
#include "vllm/utils/logger.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <cstring>
#include <string>
#include <sstream>

namespace vllm {

HttpServer::HttpServer(const EngineConfig& config, Engine& engine)
    : config_(config), engine_(engine) {}

HttpServer::~HttpServer() { stop(); }

static std::string build_response(const std::string& body, const std::string& content_type = "application/json") {
    std::ostringstream ss;
    ss << "HTTP/1.1 200 OK\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n\r\n"
       << body;
    return ss.str();
}

static std::string build_error(const std::string& msg) {
    return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n"
           "Content-Length: " + std::to_string(msg.size()) + "\r\nConnection: close\r\n\r\n" + msg;
}

bool HttpServer::start() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        VLLM_ERROR("socket failed: {}\n", strerror(errno));
        return false;
    }
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));
    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        VLLM_ERROR("bind failed: {}\n", strerror(errno));
        return false;
    }
    if (listen(listen_fd_, 16) < 0) {
        VLLM_ERROR("listen failed: {}\n", strerror(errno));
        return false;
    }
    running_ = true;
    VLLM_INFO("HTTP server listening on port {}\n", config_.port);
    return true;
}

void HttpServer::stop() {
    running_ = false;
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
}

static std::string read_http_request(int fd) {
    std::string data;
    char buf[8192];
    size_t header_end = std::string::npos;
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        data.append(buf, static_cast<size_t>(n));
        if (data.size() > 64 * 1024) break;
    }
    if (header_end == std::string::npos) return data;

    // Parse Content-Length from header.
    size_t content_length = 0;
    std::string header = data.substr(0, header_end);
    size_t cl_pos = header.find("Content-Length:");
    if (cl_pos != std::string::npos) {
        content_length = static_cast<size_t>(std::strtoul(header.c_str() + cl_pos + 15, nullptr, 10));
    }
    size_t body_start = header_end + 4;
    while (data.size() - body_start < content_length) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        data.append(buf, static_cast<size_t>(n));
    }
    return data;
}

static std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 16);
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    escaped += buf;
                } else {
                    escaped += static_cast<char>(ch);
                }
        }
    }
    return escaped;
}

static std::string run_generation(Engine& engine, const std::string& prompt,
                                  int max_tokens, float temperature, float top_p, int top_k,
                                  bool chat_response) {
    if (max_tokens < 0) throw std::runtime_error("max_tokens must be non-negative");
    if (max_tokens == 0) {
        if (chat_response) {
            return build_response(
                "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":\"length\"}]}");
        }
        return build_response("{\"object\":\"text_completion\",\"choices\":[{\"text\":\"\"}]}");
    }
    engine.add_request(prompt, static_cast<size_t>(max_tokens), temperature, top_p, top_k);
    std::string generated;
    bool done = false;
    int guard = 0;
    while (!done && guard++ < 10000) {
        auto outs = engine.step();
        for (auto& out : outs) {
            generated += out.text;
            if (out.finished) done = true;
        }
    }
    if (!done) throw std::runtime_error("generation did not finish before the request step limit");
    const std::string escaped = json_escape(generated);
    if (chat_response) {
        return build_response(
            "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
            escaped + "\"},\"finish_reason\":\"stop\"}]}");
    }
    return build_response(
        "{\"object\":\"text_completion\",\"choices\":[{\"text\":\"" + escaped + "\"}]}");
}

static std::string handle_completion(Engine& engine, const std::string& body) {
    try {
        auto j = json::parse(body);
        if (!j["prompt"].is_string()) throw std::runtime_error("prompt must be a string");
        int max_tokens = j.contains("max_tokens") ? static_cast<int>(j["max_tokens"].get_int()) : 64;
        float temperature = j.contains("temperature") ? static_cast<float>(j["temperature"].get_number()) : 1.0f;
        float top_p = j.contains("top_p") ? static_cast<float>(j["top_p"].get_number()) : 1.0f;
        int top_k = j.contains("top_k") ? static_cast<int>(j["top_k"].get_int()) : -1;
        return run_generation(engine, j["prompt"].get_string(), max_tokens, temperature, top_p, top_k, false);
    } catch (const std::exception& e) {
        return build_error(std::string("{\"error\":\"") + json_escape(e.what()) + "\"}");
    }
}

static std::string message_content(const json::Value& message) {
    const auto& content = message["content"];
    if (content.is_string()) return content.get_string();
    if (!content.is_array()) throw std::runtime_error("message.content must be a string or text parts");
    std::string result;
    for (const auto& part : content.get_array()) {
        if (!part.is_object() || part["type"].get_string() != "text")
            throw std::runtime_error("only text message content is supported");
        result += part["text"].get_string();
    }
    return result;
}

static std::string handle_chat_completion(Engine& engine, const std::string& body) {
    try {
        auto j = json::parse(body);
        if (!j["messages"].is_array() || j["messages"].size() == 0)
            throw std::runtime_error("messages must be a non-empty array");
        if (j.contains("stream") && j["stream"].get_int() != 0)
            throw std::runtime_error("streaming chat completions are not supported");
        std::string prompt;
        for (const auto& message : j["messages"].get_array()) {
            if (!message.is_object()) throw std::runtime_error("each message must be an object");
            const std::string role = message["role"].get_string();
            if (role != "system" && role != "user" && role != "assistant")
                throw std::runtime_error("message.role must be system, user, or assistant");
            prompt += "<|im_start|>" + role + "\n" + message_content(message) + "<|im_end|>\n";
        }
        prompt += "<|im_start|>assistant\n";
        int max_tokens = j.contains("max_tokens") ? static_cast<int>(j["max_tokens"].get_int()) : 64;
        float temperature = j.contains("temperature") ? static_cast<float>(j["temperature"].get_number()) : 1.0f;
        float top_p = j.contains("top_p") ? static_cast<float>(j["top_p"].get_number()) : 1.0f;
        int top_k = j.contains("top_k") ? static_cast<int>(j["top_k"].get_int()) : -1;
        return run_generation(engine, prompt, max_tokens, temperature, top_p, top_k, true);
    } catch (const std::exception& e) {
        return build_error(std::string("{\"error\":\"") + json_escape(e.what()) + "\"}");
    }
}

void HttpServer::join(const volatile sig_atomic_t* stop_flag) {
    while (running_ && (!stop_flag || *stop_flag)) {
        pollfd listener{listen_fd_, POLLIN, 0};
        int ready = poll(&listener, 1, 200);
        if (ready <= 0) continue;
        if (!(listener.revents & POLLIN)) continue;
        int client = accept(listen_fd_, nullptr, nullptr);
        if (client < 0) {
            if (!running_) break;
            continue;
        }
        std::string req = read_http_request(client);
        std::string response;
        if (req.find("GET /health") == 0) {
            response = build_response("{\"status\":\"ok\"}");
        } else if (req.find("POST /v1/completions") == 0) {
            size_t pos = req.find("\r\n\r\n");
            std::string body = (pos == std::string::npos) ? "" : req.substr(pos + 4);
            response = handle_completion(engine_, body);
        } else if (req.find("POST /v1/chat/completions") == 0) {
            size_t pos = req.find("\r\n\r\n");
            std::string body = (pos == std::string::npos) ? "" : req.substr(pos + 4);
            response = handle_chat_completion(engine_, body);
        } else {
            response = build_error("{\"error\":\"not found\"}");
        }
        send(client, response.data(), response.size(), 0);
        close(client);
    }
}

}  // namespace vllm
