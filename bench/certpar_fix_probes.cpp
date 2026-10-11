// Probes behind the certpar review fixes (src/factor): small, synthetic, seconds each.
//
//   bench_certpar_fix_probes --exp lock
//   bench_certpar_fix_probes --exp nullpivot [--nx 100] [--ny 101] [--ordering 0] [--threads 2]
//                            [--tol 1e-8] [--shifts 24] [--complex 0|1]
//
// Experiments:
//   lock       the main thread holds hs::mumps::call_mutex() while a second std::thread locks it;
//              "excludes" must be true. With call_mutex() a std::mutex it was false in executables
//              linked without libpthread (the ASan debug preset: libasan provides pthread_create,
//              so --as-needed drops libpthread, the weak __pthread_key_create resolves to 0 and
//              libstdc++'s __gthread_active_p() turns std::mutex::lock() into a no-op). The probe
//              itself avoids sleep_for and std::stoi, which would link libpthread (nanosleep and
//              __errno_location bind to it first) and so hide the problem.
//   nullpivot  shifts exactly on eigenvalues of the nx x ny 2-D Dirichlet Laplacian (--shifts of
//              them, spread over the spectrum): does null-pivot detection (CNTL(3) = --tol) flag
//              them, on the main object (analysed at a generic shift) and on an InertiaCounter (its
//              own elimination: given pivot order, no compressed 2x2 ordering)? A count reported
//              valid must lie in [#below, #below + multiplicity]: the missed eigenvalue is then
//              counted on one side or the other.
// Output: one JSON object per line on stdout. Run through tools/run/guard.py, e.g.
//   python3 tools/run/guard.py --skip-busy-check --cores 36,37 --mem-gb 4 --timeout 600 --
//     build/release/bench_certpar_fix_probes --exp nullpivot --nx 100 --ny 101 --ordering 0
// (one command line). The lock probe is meaningful in the debug preset (build/debug).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "factor/mumps_api.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;

struct Args {
  std::vector<std::string> exps;
  index_t nx = 100, ny = 101;
  int ordering = 0;
  int threads = 2;
  double tol = 1e-8;
  int shifts = 24;
  bool complex = false;
};

void emit(const json& j) { std::cout << j.dump() << std::endl; }

// Integer argument. Not std::stoi: its errno check references __errno_location, which (like
// nanosleep) binds to libpthread first and would link it, hiding what --exp lock looks for.
long long to_int(const std::string& s) { return std::strtoll(s.c_str(), nullptr, 10); }

json lock_probe() {
  auto& m = hs::mumps::call_mutex();
  std::atomic<bool> trying{false}, acquired{false};
  m.lock();
  std::thread t([&] {
    trying = true;
    m.lock();
    acquired = true;
    m.unlock();
  });
  while (!trying) std::this_thread::yield();
  // No sleep_for: its nanosleep would bind to libpthread and so link it, hiding the problem.
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  while (!acquired && std::chrono::steady_clock::now() < until) std::this_thread::yield();
  const bool excludes = !acquired;
  m.unlock();
  t.join();
  return {{"exp", "lock"}, {"excludes", excludes}};
}

template <class T>
hs::CsrMatrix<T> laplacian_2d(index_t nx, index_t ny) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  auto id = [nx](index_t i, index_t j) { return i + j * nx; };
  for (index_t j = 0; j < ny; ++j)
    for (index_t i = 0; i < nx; ++i) {
      const index_t p = id(i, j);
      r.push_back(p), c.push_back(p), v.push_back(T(4));
      if (i + 1 < nx) r.push_back(p), c.push_back(id(i + 1, j)), v.push_back(T(-1));
      if (i > 0) r.push_back(p), c.push_back(id(i - 1, j)), v.push_back(T(-1));
      if (j + 1 < ny) r.push_back(p), c.push_back(id(i, j + 1)), v.push_back(T(-1));
      if (j > 0) r.push_back(p), c.push_back(id(i, j - 1)), v.push_back(T(-1));
    }
  return hs::csr_from_coo<T>(nx * ny, nx * ny, r, c, v);
}

std::vector<double> laplacian_2d_eigenvalues(index_t nx, index_t ny) {
  const double pi = std::acos(-1.0);
  std::vector<double> w;
  for (index_t i = 1; i <= nx; ++i)
    for (index_t j = 1; j <= ny; ++j)
      w.push_back(4.0 - 2.0 * std::cos(i * pi / (nx + 1)) - 2.0 * std::cos(j * pi / (ny + 1)));
  std::sort(w.begin(), w.end());
  return w;
}

template <class T>
json nullpivot_probe(const Args& args) {
  const auto a = laplacian_2d<T>(args.nx, args.ny);
  const auto w = laplacian_2d_eigenvalues(args.nx, args.ny);
  const index_t n = a.nrows;
  hs::ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions o;
  o.ordering = args.ordering;
  o.threads = args.threads;
  o.null_pivot_tol = args.tol;
  auto f = hs::make_mumps_factorization<T>(o);
  // Generic analysis shift: the middle of the gap above the median eigenvalue.
  const double generic = 0.5 * (w[n / 2] + w[n / 2 + 1]) + 1e-3;
  f->analyze(pencil, 1.0, -generic);
  hs::InertiaCounter<T> counter(*f, o);

  int main_missed = 0, counter_missed = 0, unbracketed = 0;
  json rows = json::array();
  for (int t = 1; t <= args.shifts; ++t) {
    const index_t k = n * t / (args.shifts + 1);
    const double lambda = w[k];
    const index_t below = std::lower_bound(w.begin(), w.end(), lambda) - w.begin();
    const index_t mult = (std::upper_bound(w.begin(), w.end(), lambda) - w.begin()) - below;
    const bool main_valid = f->factorize(1.0, -lambda);
    const index_t main_neg = f->inertia().neg;
    const hs::Inertia c = counter.count(1.0, -lambda);
    auto bracketed = [&](index_t neg) { return neg >= below && neg <= below + mult; };
    main_missed += main_valid;
    counter_missed += c.valid;
    unbracketed += (main_valid && !bracketed(main_neg)) + (c.valid && !bracketed(c.neg));
    rows.push_back({{"k", k}, {"lambda", lambda}, {"below", below}, {"mult", mult},
                    {"main_valid", main_valid}, {"main_neg", main_neg}, {"counter_valid", c.valid},
                    {"counter_neg", c.neg}});
  }
  return {{"exp", "nullpivot"},          {"complex", hs::is_complex_v<T>},
          {"nx", args.nx},               {"ny", args.ny},
          {"ordering", args.ordering},   {"threads", args.threads},
          {"tol", args.tol},             {"shifts", args.shifts},
          {"main_missed", main_missed},  {"counter_missed", counter_missed},
          {"valid_but_outside_bracket", unbracketed}, {"rows", rows}};
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << argv[i] << "\n";
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--exp") {
      std::stringstream ss(need(i));
      for (std::string e; std::getline(ss, e, ',');) args.exps.push_back(e);
    } else if (k == "--nx") args.nx = to_int(need(i));
    else if (k == "--ny") args.ny = to_int(need(i));
    else if (k == "--ordering") args.ordering = static_cast<int>(to_int(need(i)));
    else if (k == "--threads") args.threads = static_cast<int>(to_int(need(i)));
    else if (k == "--tol") args.tol = std::strtod(need(i).c_str(), nullptr);
    else if (k == "--shifts") args.shifts = static_cast<int>(to_int(need(i)));
    else if (k == "--complex") args.complex = static_cast<int>(to_int(need(i))) != 0;
    else {
      std::cerr << "unknown argument " << k << "\n";
      return 2;
    }
  }
  if (args.exps.empty()) {
    std::cerr << "usage: bench_certpar_fix_probes --exp lock|nullpivot [options]\n";
    return 2;
  }
  try {
    for (const auto& e : args.exps) {
      if (e == "lock") emit(lock_probe());
      else if (e == "nullpivot")
        emit(args.complex ? nullpivot_probe<hs::cplx>(args) : nullpivot_probe<double>(args));
      else {
        std::cerr << "unknown experiment " << e << "\n";
        return 2;
      }
    }
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
