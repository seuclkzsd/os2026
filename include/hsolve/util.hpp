// Small utilities: timing, memory accounting, threading info, version.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace hs {

class Timer {
 public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}
  void reset() { t0_ = std::chrono::steady_clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

/// Peak resident set size of this process (getrusage ru_maxrss), bytes.
std::int64_t peak_rss_bytes();
/// Current resident set size (/proc/self/statm), bytes; -1 if unavailable.
std::int64_t current_rss_bytes();
/// Threads the library will use for OpenMP regions (omp_get_max_threads(), or 1 without OpenMP).
int max_threads();

/// Library version string, e.g. "0.1.0".
const char* version();
/// Build description: compiler, flags, MUMPS/BLAS versions (for reports).
std::string build_info();

}  // namespace hs
