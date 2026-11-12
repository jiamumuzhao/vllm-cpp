#include <gtest/gtest.h>
#include "vllm/core/scheduler.hpp"
#include "vllm/core/block_manager.hpp"
#include <stdexcept>

TEST(SchedulerTest, AddRequest) {
    vllm::EngineConfig cfg;
    vllm::BlockManager bm(16, 100, 0);
    vllm::Scheduler sched(cfg, bm);
    auto grp = sched.add_request({1, 2, 3, 4}, 100, 1.0f, 1.0f, -1);
    EXPECT_EQ(sched.waiting_queue_size(), 1);
    EXPECT_EQ(sched.running_seq_count(), 0);
    auto output = sched.schedule();
    EXPECT_GT(output.num_prefill_tokens, 0);
    EXPECT_EQ(sched.running_seq_count(), 1);
}

TEST(SchedulerTest, HasPendingWork) {
    vllm::EngineConfig cfg;
    vllm::BlockManager bm(16, 100, 0);
    vllm::Scheduler sched(cfg, bm);
    EXPECT_FALSE(sched.has_pending_work());
    sched.add_request({1, 2}, 50);
    EXPECT_TRUE(sched.has_pending_work());
}

TEST(SchedulerTest, AllocatesPromptBlocksAndGrowsOnDecode) {
    vllm::EngineConfig cfg;
    vllm::BlockManager bm(16, 4, 0);
    vllm::Scheduler sched(cfg, bm);
    sched.add_request(std::vector<vllm::TokenId>(16, 1), 32);
    auto output = sched.schedule();
    ASSERT_EQ(output.prefill_seqs.size(), 1);
    EXPECT_EQ(output.prefill_seqs[0]->block_table().size(), 1);
    EXPECT_EQ(bm.num_free_gpu_blocks(), 3);
}

TEST(SchedulerTest, RejectsRequestLargerThanEntirePool) {
    vllm::EngineConfig cfg;
    vllm::BlockManager bm(16, 2, 0);
    vllm::Scheduler sched(cfg, bm);
    EXPECT_THROW(sched.add_request(std::vector<vllm::TokenId>(16, 1), 32), std::invalid_argument);
    EXPECT_EQ(sched.waiting_queue_size(), 0);
    EXPECT_EQ(sched.running_seq_count(), 0);
}

TEST(SchedulerTest, PreemptsAndRequeuesToMakeDecodeProgress) {
    vllm::EngineConfig cfg;
    vllm::BlockManager bm(16, 3, 0);
    vllm::Scheduler sched(cfg, bm);
    auto first_group = sched.add_request(std::vector<vllm::TokenId>(16, 1), 32);
    auto first = sched.schedule();
    ASSERT_EQ(first.prefill_seqs.size(), 1);
    auto first_seq = first_group->seqs()[0];
    first_seq->set_num_processed_tokens(16);
    auto decoding = sched.schedule();
    ASSERT_EQ(decoding.decode_seqs.size(), 1);
    EXPECT_EQ(first_seq->block_table().size(), 2);

    auto second_group = sched.add_request(std::vector<vllm::TokenId>(16, 2), 32);
    auto with_second = sched.schedule();
    ASSERT_EQ(with_second.prefill_seqs.size(), 1);
    auto second_seq = second_group->seqs()[0];
    second_seq->append_token(42);
    second_seq->append_token(43);
    EXPECT_EQ(second_seq->replay_len(), 17);
    EXPECT_EQ(second_seq->replay_token(16), 42);
    second_seq->clear_token_generated_this_step();
    EXPECT_EQ(bm.num_free_gpu_blocks(), 0);

    // The first request reaches the next block boundary while the second owns
    // the remaining physical block. It is preempted and queued for replay.
    first_seq->set_num_processed_tokens(32);
    auto after_preempt = sched.schedule();
    ASSERT_EQ(after_preempt.decode_seqs.size(), 1);
    EXPECT_EQ(after_preempt.decode_seqs[0]->seq_id(), first_seq->seq_id());
    EXPECT_TRUE(second_seq->needs_recompute());
    EXPECT_TRUE(second_seq->block_table().empty());
    EXPECT_EQ(sched.waiting_queue_size(), 1);
    EXPECT_EQ(sched.running_seq_count(), 1);
    EXPECT_EQ(first_seq->block_table().size(), 3);

    sched.finish_sequence(first_seq->seq_id());
    auto restored = sched.schedule();
    ASSERT_EQ(restored.prefill_seqs.size(), 1);
    EXPECT_EQ(restored.prefill_seqs[0]->seq_id(), second_seq->seq_id());
    EXPECT_TRUE(second_seq->needs_recompute());
    EXPECT_EQ(second_seq->block_table().size(), 2);
}

TEST(SchedulerTest, RejectsZeroBlockSize) {
    EXPECT_THROW(vllm::BlockManager(0, 4, 0), std::invalid_argument);
}
