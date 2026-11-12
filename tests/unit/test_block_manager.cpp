#include <gtest/gtest.h>
#include "vllm/core/block_manager.hpp"

TEST(BlockManagerTest, AcquireRelease) {
    vllm::BlockManager bm(16, 10, 0);
    EXPECT_EQ(bm.num_free_gpu_blocks(), 10);
    auto blk = bm.acquire_block();
    EXPECT_TRUE(blk.has_value());
    EXPECT_EQ(bm.num_free_gpu_blocks(), 9);
    bm.release_block(*blk);
    EXPECT_EQ(bm.num_free_gpu_blocks(), 10);
}

TEST(BlockManagerTest, RefCounting) {
    vllm::BlockManager bm(16, 5, 0);
    auto blk = bm.acquire_block();
    bm.inc_ref(*blk);
    EXPECT_FALSE(bm.dec_ref(*blk));  // still held
    EXPECT_TRUE(bm.dec_ref(*blk));   // now freed
}

TEST(BlockManagerTest, CanAllocate) {
    vllm::BlockManager bm(16, 3, 0);
    EXPECT_TRUE(bm.can_allocate(2));
    EXPECT_FALSE(bm.can_allocate(5));
}
