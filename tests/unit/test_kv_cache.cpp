#include <gtest/gtest.h>
#include "vllm/memory/kv_cache.hpp"

TEST(KVCacheTest, BasicAllocation) {
    vllm::KVCache cache(4, 16, 16, 8, 128);
    EXPECT_EQ(cache.num_layers(), 4);
    EXPECT_EQ(cache.num_blocks(), 16);
    EXPECT_EQ(cache.block_size(), 16);
    EXPECT_GT(cache.total_bytes(), 0);
}

TEST(KVCacheTest, PointersValid) {
    vllm::KVCache cache(2, 4, 8, 4, 64);
    EXPECT_NE(cache.k_ptr(0, 0), nullptr);
    EXPECT_NE(cache.v_ptr(0, 0), nullptr);
}
