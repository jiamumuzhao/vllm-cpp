add_executable(throughput_bench throughput_bench.cpp)
target_link_libraries(throughput_bench PRIVATE vllm::vllm benchmark::benchmark)

add_executable(latency_bench latency_bench.cpp)
target_link_libraries(latency_bench PRIVATE vllm::vllm benchmark::benchmark)
