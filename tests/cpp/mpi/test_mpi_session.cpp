// Multi-process solves over MPI (src/mpi, DESIGN §3.9): run by ctest as MPI jobs of 1, 2 and 3 ranks
// (CMakeLists.txt: test_mpi_session.ranks<R>). Rank 0 runs the tests below - each a solve_parallel
// of the session - while ranks 1..R-1 serve them (hs::serve) until rank 0 releases them at the end.
// What the helpers do depends on R: with 3 ranks the certificate's two counts and the slices go to
// the two helpers, with 2 (one helper) and 1 the leader solves alone; the results must not depend
// on it. The same logic over thread transports: test_parallel_session.cpp.
#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "driver/internal.hpp"
#include "hsolve/parallel.hpp"
#include "hsolve/solver.hpp"
#include "mpi/mpi_transport.hpp"
#include "support/parallel_problems.hpp"

namespace {

using hs::cplx;
using hs::index_t;
using namespace hs::partest;

hs::Transport* g_peers = nullptr;  // rank 0's session (main)

TEST(MpiSession, RealMpiIsLinkedNotTheMumpsStub) {
  const std::string lib = hs::mpi::mpi_library();
  EXPECT_EQ(lib.find("mpiseq"), std::string::npos) << lib;
  EXPECT_EQ(hs::mpi::world_size(), g_peers->size());
  EXPECT_EQ(g_peers->rank(), 0);
}

template <class T>
class MpiSessionTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(MpiSessionTyped, Scalars);

TYPED_TEST(MpiSessionTyped, NearestCertificate) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  const auto ref = hs::solve(a, nullptr, nearest_cfg());
  const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, *g_peers);
  expect_nearest_certified(sol);
  expect_same_set(sol.eigenvalues, ref.eigenvalues, 1e-12);
  const auto& p = sol.report.parallel;
  EXPECT_EQ(p.processes, g_peers->size());
  EXPECT_EQ(p.helper_counts, g_peers->size() >= 3 ? 2 : 0) << warnings(sol.report);
  EXPECT_EQ(p.fallbacks, 0);
}

TYPED_TEST(MpiSessionTyped, IntervalSlices) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  const auto ref = hs::solve(a, nullptr, interval_cfg());
  const auto sol = hs::solve_parallel(a, nullptr, interval_cfg(), nullptr, *g_peers);
  expect_interval_certified(sol);
  expect_same_set(sol.eigenvalues, ref.eigenvalues, 1e-12);
  const auto& p = sol.report.parallel;
  EXPECT_EQ(p.helper_slices, g_peers->size() >= 3 ? static_cast<index_t>(sol.report.slices.size()) : 0)
      << warnings(sol.report);
}

// Every helper task fails (fault bit 128): the leader does the work, the results are certified.
TYPED_TEST(MpiSessionTyped, FailingHelpersAreReplacedByTheLeader) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(hs::driver::kFaultHelperTasks), nullptr, *g_peers);
  expect_nearest_certified(sol);
  EXPECT_EQ(sol.report.parallel.helper_counts, 0);
  EXPECT_EQ(sol.report.parallel.fallbacks, g_peers->size() >= 3 ? 2 : 0) << warnings(sol.report);
  const auto iv = hs::solve_parallel(a, nullptr, interval_cfg(hs::driver::kFaultHelperTasks | hs::driver::kFaultOtherError),
                                     nullptr, *g_peers);
  expect_interval_certified(iv);
  EXPECT_EQ(iv.report.parallel.helper_slices, 0);
  EXPECT_EQ(iv.report.parallel.fallbacks, g_peers->size() >= 3 ? 2 : 0) << warnings(iv.report);
}

}  // namespace

int main(int argc, char** argv) {
  if (!hs::mpi::init(&argc, &argv)) return 2;
  int rc = 0;
  {
    hs::mpi::MpiTransport peers;
    if (peers.rank() > 0) {
      hs::serve(peers);
    } else {
      ::testing::InitGoogleTest(&argc, argv);
      g_peers = &peers;
      rc = RUN_ALL_TESTS();
      hs::release_helpers(peers);
    }
  }
  hs::mpi::finalize();
  return rc;
}
