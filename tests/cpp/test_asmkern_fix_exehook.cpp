// The BLAS hook linked into an executable (CMake function hsolve_link_blas_hook - what
// -DHSOLVE_BLAS_HOOK=ON does to hsolve; module blas, P5c; docs/BLAS.md section 4): this test
// executable carries the hook's dgemm_ / dtrsm_ / ... and dmumps_c / smumps_c wrappers itself.
//   * MUMPS stays loaded (the executable's own dmumps_c satisfies libhsolve's references, and a
//     link with --as-needed used to drop the MUMPS libraries: the wrapper then aborted with "no
//     definition of dmumps_c"), and the wrappers' next definitions are MUMPS';
//   * the executable exports its BLAS entry points, so MUMPS' BLAS calls land in the hook: a
//     factorization and a block solve through hsolve's MUMPS backend record dgemm / dtrsm calls in
//     the factorization and solve phases (profiler enabled for this process, output /dev/null);
//   * with HSOLVE_BLAS_HOOK=ON, the hooked hsolve executable solves a small problem.
#include <dlfcn.h>
#include <dmumps_c.h>
#include <gtest/gtest.h>
#include <smumps_c.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "blas/blas_hook.hpp"
#include "blas/blas_profile.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/blas_lapack.hpp"
#include "support/fixtures.hpp"

namespace {

namespace prof = hs::blas::prof;

// Before main() and any BLAS call: the profiler reads HS_BLAS_PROFILE once.
const bool kProfileOn = [] {
  setenv("HS_BLAS_PROFILE", "/dev/null", 0);
  return true;
}();

std::string object_of(const void* p) {
  Dl_info info{};
  if (p == nullptr || dladdr(p, &info) == 0 || info.dli_fname == nullptr) return "";
  return info.dli_fname;
}

std::string self_exe() {
  char buf[4096] = {0};
  const ssize_t len = readlink("/proc/self/exe", buf, sizeof buf - 1);
  return len > 0 ? std::string(buf, static_cast<std::size_t>(len)) : std::string();
}

bool same_file(const std::string& a, const std::string& b) {
  std::error_code ec;
  return !a.empty() && !b.empty() && std::filesystem::equivalent(a, b, ec);
}

TEST(AsmkernFixExeHook, MumpsIsLoadedAndTheWrappersFindTheRealEntryPoints) {
  const std::string exe = self_exe();
  for (const auto& [name, own] :
       {std::pair<const char*, const void*>{"dmumps_c", reinterpret_cast<const void*>(&dmumps_c)},
        std::pair<const char*, const void*>{"smumps_c", reinterpret_cast<const void*>(&smumps_c)}}) {
    ASSERT_TRUE(same_file(object_of(own), exe)) << name << " is not the executable's wrapper";
    void* real = hs::blas::next_definition(name);
    ASSERT_NE(real, nullptr) << "no MUMPS library defines " << name
                             << " (dropped from DT_NEEDED by --as-needed?)";
    EXPECT_NE(real, own);
    EXPECT_NE(object_of(real).find("mumps"), std::string::npos) << object_of(real);
  }
}

TEST(AsmkernFixExeHook, TheExecutableExportsTheBlasEntryPoints) {
  const std::string exe = self_exe();
  for (const char* name : {"dgemm_", "dtrsm_", "dgemv_", "daxpy_", "dcopy_", "dscal_", "dswap_",
                           "dnrm2_", "idamax_"}) {
    // The global scope's first definition is what MUMPS' references bind to.
    void* global = dlsym(RTLD_DEFAULT, name);
    ASSERT_NE(global, nullptr) << name;
    EXPECT_TRUE(same_file(object_of(global), exe))
        << name << " resolves to " << object_of(global) << ", not to the hook in " << exe;
    void* real = hs::blas::next_definition(name);
    ASSERT_NE(real, nullptr) << name;
    EXPECT_FALSE(same_file(object_of(real), exe)) << name;
  }
  EXPECT_FALSE(same_file(hs::blas::real_blas().library, exe));
}

// A shifted 2-D Laplacian (fronts large enough for dense BLAS) factorized and solved with four
// right-hand sides through hsolve's MUMPS backend: correct, and MUMPS' BLAS calls were recorded
// by the executable's hook in the phases its dmumps_c wrapper set.
TEST(AsmkernFixExeHook, MumpsBlasCallsGoThroughTheHook) {
  ASSERT_TRUE(kProfileOn && prof::enabled());
  const hs::index_t nx = 60, n = nx * nx, nrhs = 4;
  const double sigma = 0.1234;
  const auto a = hs::test::laplacian_2d<double>(nx, nx);
  hs::ShiftedPencil<double> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<double>();
  f->analyze(pencil, 1.0, -sigma);
  ASSERT_TRUE(f->factorize(1.0, -sigma));
  std::mt19937_64 g(91);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> x(static_cast<std::size_t>(n) * nrhs), b(x.size());
  for (double& v : x) v = u(g);
  for (hs::index_t j = 0; j < nrhs; ++j) {  // b = (A - sigma I) x
    const double* xj = x.data() + j * n;
    double* bj = b.data() + j * n;
    hs::spmv(a, xj, bj);
    for (hs::index_t i = 0; i < n; ++i) bj[i] -= sigma * xj[i];
  }
  f->solve(b.data(), nrhs, n);
  double err = 0.0, nrm = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    err = std::max(err, std::abs(b[i] - x[i]));
    nrm = std::max(nrm, std::abs(x[i]));
  }
  EXPECT_LE(err, 1e-8 * nrm);
  EXPECT_GE(prof::recorded_mumps_calls(prof::Phase::Factorization), 1);
  EXPECT_GE(prof::recorded_mumps_calls(prof::Phase::SolveP), 1);
  const std::int64_t fact = prof::recorded_calls(prof::Op::Dgemm, prof::Phase::Factorization) +
                            prof::recorded_calls(prof::Op::Dtrsm, prof::Phase::Factorization);
  const std::int64_t solve = prof::recorded_calls(prof::Op::Dgemm, prof::Phase::SolveP) +
                             prof::recorded_calls(prof::Op::Dtrsm, prof::Phase::SolveP);
  EXPECT_GT(fact, 0) << "MUMPS' factorization BLAS calls bypassed the hook";
  EXPECT_GT(solve, 0) << "MUMPS' solve BLAS calls bypassed the hook";
}

// HSOLVE_BLAS_HOOK=ON: the hooked hsolve executable (route from HSOLVE_BLAS_ROUTE_DEFAULT or
// HS_BLAS_ROUTE) solves the shifted 2-D Laplacian, written as Matrix Market, nearest to a shift.
TEST(AsmkernFixExeHook, HookedHsolveExecutableSolvesASmallProblem) {
#ifndef HSOLVE_HOOKED_CLI
  GTEST_SKIP() << "built without HSOLVE_BLAS_HOOK";
#else
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
                       ("hsolve_asmkern_exehook_" + std::to_string(static_cast<long>(getpid())));
  fs::create_directories(dir);
  const auto a = hs::test::laplacian_2d<double>(40, 37);  // nx != ny: no multiple eigenvalues
  {
    std::ofstream out(dir / "A.mtx");
    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    hs::index_t lower = 0;
    for (hs::index_t i = 0; i < a.nrows; ++i)
      for (hs::index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) lower += a.colind[p] <= i;
    out << a.nrows << " " << a.ncols << " " << lower << "\n";
    out.precision(17);
    for (hs::index_t i = 0; i < a.nrows; ++i)
      for (hs::index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
        if (a.colind[p] <= i) out << i + 1 << " " << a.colind[p] + 1 << " " << a.values[p] << "\n";
  }
  const std::string cmd = std::string("'") + HSOLVE_HOOKED_CLI + "' --A '" +
                          (dir / "A.mtx").string() + "' --sigma 0.1234 --k 6 --out '" +
                          (dir / "out").string() + "' --threads 2 -q > '" +
                          (dir / "log.txt").string() + "' 2>&1";
  const int rc = std::system(cmd.c_str());
  std::ifstream log(dir / "log.txt");
  const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(WIFEXITED(rc) && WEXITSTATUS(rc) == 0) << "rc " << rc << ": " << text;
  EXPECT_TRUE(fs::exists(dir / "out" / "report.json")) << text;
  fs::remove_all(dir);
#endif
}

}  // namespace
