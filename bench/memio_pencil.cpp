// Memory and speed of hs::ShiftedPencil (module memio): what the pencil keeps, what constructing
// and assembling it costs at peak, and how long assemble() takes.
//
//   bench_memio_pencil --A A.npz [--B B.npz] [--reps R]
//
// Loads A (and B) with hs::load_matrix (A complex and B real, or vice versa, are promoted), then:
//   build:    construct the pencil; retained = VmRSS growth, peak = VmHWM growth (high-water mark
//             reset through /proc/self/clear_refs, as in bench_memio_load);
//   assemble: R calls of assemble(1, -sigma, vals) into a preallocated double (and float) array;
//             peak = VmHWM growth over the calls (the transient workspace of assemble);
//   check:    the assembled values against assemble_full (to 4e-16 relative on the upper pattern),
//             plus an FNV-1a hash of the assembled bits, to compare implementations exactly.
// Prints one JSON line. Run through tools/run/guard.py.
#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;

namespace {

struct ProcMem {
  std::int64_t rss = 0;
  std::int64_t hwm = 0;
};

ProcMem proc_mem() {
  std::ifstream in("/proc/self/status");
  ProcMem m;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) m.rss = std::atoll(line.c_str() + 6) * 1024;
    if (line.rfind("VmHWM:", 0) == 0) m.hwm = std::atoll(line.c_str() + 6) * 1024;
  }
  return m;
}

ProcMem fresh_baseline() {
  malloc_trim(0);
  std::ofstream("/proc/self/clear_refs") << "5";
  return proc_mem();
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class T>
std::int64_t bytes_of(const CsrMatrix<T>& a) {
  return static_cast<std::int64_t>(a.rowptr.capacity() * 8 + a.colind.capacity() * 4 + a.values.capacity() * sizeof(T));
}

template <class T>
std::string run(const CsrMatrix<T>& a, const CsrMatrix<T>* b, int reps) {
  const double sigma = 0.37;
  std::ostringstream os;
  os.precision(4);
  os << "\"scalar\": \"" << (hs::is_complex_v<T> ? "complex128" : "float64") << "\", \"n\": " << a.nrows
     << ", \"nnz_a\": " << a.nnz() << ", \"nnz_b\": " << (b ? b->nnz() : 0)
     << ", \"input_mb\": " << static_cast<double>(bytes_of(a) + (b ? bytes_of(*b) : 0)) / 1e6;

  const ProcMem m0 = fresh_baseline();
  auto t0 = std::chrono::steady_clock::now();
  std::optional<hs::ShiftedPencil<T>> pencil;
  pencil.emplace(a, b);
  const double build_s = seconds_since(t0);
  const ProcMem m1 = proc_mem();
  const index_t nnz = pencil->nnz();
  os << ", \"pattern_nnz\": " << nnz << ", \"build_s\": " << build_s
     << ", \"pencil_retained_mb\": " << static_cast<double>(m1.rss - m0.rss) / 1e6
     << ", \"pencil_build_peak_mb\": " << static_cast<double>(m1.hwm - m0.rss) / 1e6;

  std::vector<T> vals(static_cast<std::size_t>(nnz));
  std::vector<hs::rebind_t<T, float>> vals32(static_cast<std::size_t>(nnz));
  pencil->assemble(1.0, -sigma, vals.data());  // warm up (first touch)
  pencil->assemble(1.0, -sigma, vals32.data());
  const ProcMem m2 = fresh_baseline();
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) pencil->assemble(1.0, -sigma, vals.data());
  const double t64 = seconds_since(t0) / reps;
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) pencil->assemble(1.0, -sigma, vals32.data());
  const double t32 = seconds_since(t0) / reps;
  const ProcMem m3 = proc_mem();
  os << ", \"assemble_ms\": " << t64 * 1e3 << ", \"assemble_f32_ms\": " << t32 * 1e3
     << ", \"assemble_peak_mb\": " << static_cast<double>(m3.hwm - m2.rss) / 1e6;

  // Check against the full assembly (upper entries; entries absent from the full matrix must be
  // exact zeros in the pattern, e.g. the diagonal of a row without one). Entries stored in both
  // A and B may differ in the last bit (the two assemblies may contract a*x + b*y differently).
  // The tolerance is relative to |a_ij| + sigma |b_ij| (cancellation between A and B).
  const CsrMatrix<T> full = pencil->assemble_full(1.0, -sigma);
  const CsrMatrix<T> full_a = pencil->assemble_full(1.0, 0.0), full_b = pencil->assemble_full(0.0, 1.0);
  const auto at = [](const CsrMatrix<T>& m, index_t i, std::int32_t j) {
    const auto* b = m.colind.data() + m.rowptr[i];
    const auto* e = m.colind.data() + m.rowptr[i + 1];
    const auto* it = std::lower_bound(b, e, j);
    return it != e && *it == j ? m.values[m.rowptr[i] + (it - b)] : T(0);
  };
  index_t bad = 0;
  for (index_t i = 0; i < pencil->order(); ++i) {
    for (index_t p = pencil->rowptr()[i]; p < pencil->rowptr()[i + 1]; ++p) {
      const std::int32_t j = pencil->colind()[p];
      const double scale = std::abs(at(full_a, i, j)) + sigma * std::abs(at(full_b, i, j));
      bad += !(std::abs(vals[p] - at(full, i, j)) <= 4e-16 * scale);
    }
  }
  // Bitwise fingerprint of the assembled values, to compare pencil implementations exactly.
  std::uint64_t h = 1469598103934665603ull;
  const auto* bytes = reinterpret_cast<const unsigned char*>(vals.data());
  for (std::size_t k = 0; k < vals.size() * sizeof(T); ++k) h = (h ^ bytes[k]) * 1099511628211ull;
  const auto* bytes32 = reinterpret_cast<const unsigned char*>(vals32.data());
  for (std::size_t k = 0; k < vals32.size() * sizeof(vals32[0]); ++k) h = (h ^ bytes32[k]) * 1099511628211ull;
  char hex[32];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
  os << ", \"check\": \"" << (bad == 0 ? "ok" : "FAIL " + std::to_string(bad)) << "\", \"values_fnv1a\": \"" << hex
     << "\"";
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string a_path, b_path;
    int reps = 10;
    for (int k = 1; k < argc; ++k) {
      const std::string arg = argv[k];
      if (arg == "--A" && k + 1 < argc) {
        a_path = argv[++k];
      } else if (arg == "--B" && k + 1 < argc) {
        b_path = argv[++k];
      } else if (arg == "--reps" && k + 1 < argc) {
        reps = std::max(1, std::atoi(argv[++k]));
      } else {
        HS_THROW(InvalidInput, "usage: bench_memio_pencil --A A.npz [--B B.npz] [--reps R]");
      }
    }
    if (a_path.empty()) HS_THROW(InvalidInput, "--A is required");
    mallopt(M_MMAP_THRESHOLD, 1 << 20);
    mallopt(M_TRIM_THRESHOLD, 1 << 20);
    hs::LoadedMatrix la = hs::load_matrix(a_path);
    std::optional<hs::LoadedMatrix> lb;
    if (!b_path.empty()) lb = hs::load_matrix(b_path);
    std::string body;
    if (la.is_complex() || (lb && lb->is_complex())) {
      CsrMatrix<cplx> a = la.is_complex() ? std::get<1>(std::move(la.matrix)) : hs::to_complex(std::get<0>(std::move(la.matrix)));
      CsrMatrix<cplx> b;
      if (lb) b = lb->is_complex() ? std::get<1>(std::move(lb->matrix)) : hs::to_complex(std::get<0>(std::move(lb->matrix)));
      body = run<cplx>(a, lb ? &b : nullptr, reps);
    } else {
      body = run<double>(std::get<0>(la.matrix), lb ? &std::get<0>(lb->matrix) : nullptr, reps);
    }
    std::cout << "{\"A\": \"" << a_path << "\", \"B\": \"" << b_path << "\", " << body << "}" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_memio_pencil: %s\n", e.what());
    return 2;
  }
}
