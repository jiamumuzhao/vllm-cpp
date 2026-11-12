#include <gtest/gtest.h>
#include "vllm/tokenizer/tokenizer.hpp"
#include <fstream>

TEST(TokenizerTest, MatchesQwenReferenceIds) {
    const std::string path = VLLM_TEST_MODEL_PATH;
    if (!std::ifstream(path + "/tokenizer.json")) GTEST_SKIP() << "Qwen tokenizer.json is not installed";
    vllm::SimpleTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(path));
    EXPECT_EQ(tokenizer.encode("Hello"), (std::vector<vllm::TokenId>{9707}));
    EXPECT_EQ(tokenizer.encode("什么是LLM？"),
              (std::vector<vllm::TokenId>{106582, 4086, 44, 11319}));
    EXPECT_EQ(tokenizer.encode("<|im_start|>user\n你好<|im_end|>"),
              (std::vector<vllm::TokenId>{151644, 872, 198, 108386, 151645}));
    EXPECT_EQ(tokenizer.encode("e\xcc\x81"), (std::vector<vllm::TokenId>{963}));
    EXPECT_EQ(tokenizer.encode("A\n\nB"), (std::vector<vllm::TokenId>{32, 271, 33}));
}
