#include <benchmark/benchmark.h>
static void BM_Empty(benchmark::State& state) {
    for (auto _ : state) {}
}
BENCHMARK(BM_Empty);
BENCHMARK_MAIN();
