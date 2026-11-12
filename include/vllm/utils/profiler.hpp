#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace vllm {

// ── Timer: RAII scope-based wall-clock timer ───────────────
// Usage:
//   {
//       Timer t("prefill");
//       ... prefill work ...
//   }  // prints "[prefill] took X.XX ms"
class Timer {
  public:
    explicit Timer(std::string name);
    ~Timer();

    // Return elapsed in milliseconds without printing.
    double elapsed_ms() const;

  private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    bool moved_ = false;
};

// ── Profiler: accumulates timing stats across engine steps ─
class Profiler {
  public:
    static Profiler& instance();

    // Record a timing sample for a named phase.
    void record(const std::string& phase, double ms);

    // Print a summary to stderr.
    void report() const;

    // Reset all accumulators.
    void reset();

  private:
    Profiler() = default;

    struct Stats {
        double total_ms = 0.0;
        size_t count = 0;
        double min_ms = 1e9;
        double max_ms = 0.0;
    };
    std::unordered_map<std::string, Stats> stats_;
};

}  // namespace vllm
