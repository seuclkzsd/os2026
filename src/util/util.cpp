#include "hsolve/util.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HSOLVE_VERSION
#define HSOLVE_VERSION "0.0.0"
#endif
#ifndef HSOLVE_BUILD_FLAGS
#define HSOLVE_BUILD_FLAGS ""
#endif

namespace hs {

std::int64_t peak_rss_bytes() {
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
  return static_cast<std::int64_t>(ru.ru_maxrss) * 1024;  // Linux reports KiB
}

std::int64_t current_rss_bytes() {
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return -1;
  long pages_total = 0, pages_rss = 0;
  const int got = std::fscanf(f, "%ld %ld", &pages_total, &pages_rss);
  std::fclose(f);
  if (got != 2) return -1;
  return static_cast<std::int64_t>(pages_rss) * sysconf(_SC_PAGESIZE);
}

int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

const char* version() { return HSOLVE_VERSION; }

std::string build_info() {
  std::ostringstream os;
  os << "hsolve " << HSOLVE_VERSION;
#if defined(__clang__)
  os << ", clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
  os << ", gcc " << __GNUC__ << "." << __GNUC_MINOR__;
#endif
  os << ", flags: " << HSOLVE_BUILD_FLAGS;
#ifdef _OPENMP
  os << ", openmp " << _OPENMP;
#endif
  return os.str();
}

}  // namespace hs
