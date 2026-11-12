#include "vllm/utils/profiler.hpp"

#include <cstdio>

namespace vllm {

// ── Timer ──────────────────────────────────────────────────
Timer::Timer(std::string name) : name_(std::move(name)), start_(std::chrono::high_resolution_clock::now()) {}

Timer::~Timer() {
    if (!moved_) {
        Profiler::instance().record(name_, elapsed_ms());
    }
}

double Timer::elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

// ── Profiler ───────────────────────────────────────────────
Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::record(const std::string& phase, double ms) {
    auto& s = stats_[phase];
    s.total_ms += ms;
    s.count++;
    if (ms < s.min_ms) s.min_ms = ms;
    if (ms > s.max_ms) s.max_ms = ms;
}

void Profiler::report() const {
    std::fprintf(stderr, "\n=== vLLM Profiler Report ===\n");
    std::fprintf(stderr, "%-20s %8s %8s %8s %8s\n", "Phase", "Count", "Total(ms)", "Avg(ms)", "Min/Max(ms)");
    for (const auto& [phase, s] : stats_) {
        double avg = s.count > 0 ? s.total_ms / static_cast<double>(s.count) : 0.0;
        std::fprintf(stderr, "%-20s %8zu %8.2f %8.2f %8.2f/%.2f\n",
                     phase.c_str(), s.count, s.total_ms, avg, s.min_ms, s.max_ms);
    }
    std::fprintf(stderr, "==============================\n\n");
}

void Profiler::reset() {
    stats_.clear();
}

}  // namespace vllm
