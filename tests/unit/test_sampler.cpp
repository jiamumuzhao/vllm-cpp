#include <gtest/gtest.h>
#include "vllm/sampler/sampler.hpp"
#include "vllm/sampler/topk_topp.hpp"
#include <cmath>

TEST(SamplerTest, Greedy) {
    std::vector<float> logits = {0.1f, 0.9f, 0.5f, 0.3f};
    auto token = vllm::Sampler::greedy(logits.data(), 4);
    EXPECT_EQ(token, 1);
}

TEST(SamplerTest, GreedyLarger) {
    std::vector<float> logits(100, 0.0f);
    logits[42] = 10.0f;
    auto token = vllm::Sampler::greedy(logits.data(), 100);
    EXPECT_EQ(token, 42);
}

TEST(SamplerTest, TopPUsesProbabilityOrder) {
    float probabilities[] = {0.05f, 0.85f, 0.10f};
    int32_t indices[] = {0, 1, 2};
    EXPECT_EQ(vllm::sampler::top_p_filter(probabilities, indices, 3, 0.8f), 1);
    EXPECT_EQ(indices[0], 1);
}

TEST(SamplerTest, TopKAtVocabSize) {
    float probabilities[] = {0.1f, 0.7f, 0.2f};
    int32_t indices[] = {0, 1, 2};
    EXPECT_EQ(vllm::sampler::top_k_filter(probabilities, indices, 3, 3), 3);
    EXPECT_EQ(indices[0], 1);
}

TEST(SamplerTest, NucleusSamplingPreservesTokenIds) {
    vllm::Sampler sampler;
    float logits[] = {-5.0f, 5.0f, -4.0f};
    for (int i = 0; i < 20; ++i) {
        auto out = sampler.sample(logits, 3, 1.0f, 0.5f, -1, 1.0f, {});
        EXPECT_EQ(out.token, 1);
        EXPECT_NEAR(out.logprob, 0.0f, 1e-6f);
    }
}
