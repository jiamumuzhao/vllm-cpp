#include <benchmark/benchmark.h>
static void BM_Latency(benchmark::State& state) {
    for (auto _ : state) {}
}
BENCHMARK(BM_Latency);
BENCHMARK_MAIN();
