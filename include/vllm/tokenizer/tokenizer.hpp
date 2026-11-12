#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace vllm {

using TokenId = int32_t;

class Tokenizer {
  public:
    Tokenizer() = default;
    virtual ~Tokenizer() = default;

    virtual bool load(const std::string& path) = 0;
    virtual std::vector<TokenId> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<TokenId>& tokens) = 0;
    virtual std::string decode(TokenId token) = 0;
    virtual int vocab_size() const = 0;
    virtual TokenId eos_token() const = 0;
    virtual bool is_stop_token(TokenId token) const;
};

class SimpleTokenizer : public Tokenizer {
  public:
    SimpleTokenizer();
    bool load(const std::string& path) override;
    std::vector<TokenId> encode(const std::string& text) override;
    std::string decode(const std::vector<TokenId>& tokens) override;
    std::string decode(TokenId token) override;
    int vocab_size() const override { return static_cast<int>(id_to_token_.size()); }
    TokenId eos_token() const override { return eos_token_; }
    bool is_stop_token(TokenId token) const override;

  private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::vector<std::string> special_tokens_;
    TokenId eos_token_ = -1;
    std::vector<TokenId> stop_tokens_;
};

}  // namespace vllm
