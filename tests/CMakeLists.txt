add_executable(test_kv_cache unit/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE vllm::vllm GTest::gtest_main)
add_test(NAME kv_cache COMMAND test_kv_cache)

add_executable(test_block_manager unit/test_block_manager.cpp)
target_link_libraries(test_block_manager PRIVATE vllm::vllm GTest::gtest_main)
add_test(NAME block_manager COMMAND test_block_manager)

add_executable(test_scheduler unit/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE vllm::vllm GTest::gtest_main)
add_test(NAME scheduler COMMAND test_scheduler)

add_executable(test_sampler unit/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE vllm::vllm GTest::gtest_main)
add_test(NAME sampler COMMAND test_sampler)

add_executable(test_tokenizer unit/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE vllm::vllm GTest::gtest_main)
target_compile_definitions(test_tokenizer PRIVATE VLLM_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/Qwen2.5-0.5B-Instruct")
add_test(NAME tokenizer COMMAND test_tokenizer)
