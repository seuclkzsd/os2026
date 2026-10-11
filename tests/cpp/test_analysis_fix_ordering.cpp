// Regression tests for the review fixes of hsolve's nested-dissection ordering (src/factor/
// ordering*.cpp, compute_ordering): coarsening with small targets on graphs with many components;
// calls from inside an OpenMP parallel region (threads = 1 included).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::ShiftedPencil;
namespace ord = hs::ordering;
using Edges = std::vector<std::pair<std::int32_t, std::int32_t>>;

// ------------------------------------------------------------------------------- helpers

// Graph of a symmetric pattern given as an edge list (both directions are added).
ord::Graph graph_from_edges(std::int32_t n, const Edges& edges) {
  std::vector<std::vector<std::int32_t>> up(static_cast<std::size_t>(n));
  for (auto [a, b] : edges) {
    if (a == b) continue;
    up[std::min(a, b)].push_back(std::max(a, b));
  }
  std::vector<std::int64_t> rowptr(1, 0);
  std::vector<std::int32_t> col;
  for (std::int32_t i = 0; i < n; ++i) {
    auto& r = up[i];
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    col.push_back(i);
    col.insert(col.end(), r.begin(), r.end());
    rowptr.push_back(static_cast<std::int64_t>(col.size()));
  }
  return ord::graph_from_upper(n, rowptr.data(), col.data());
}

// Edges of an nx x ny (x nz) grid with the 5-point (7-point) stencil.
Edges grid_edges(int nx, int ny, int nz = 1) {
  Edges e;
  auto id = [&](int i, int j, int k) { return i + nx * (j + ny * k); };
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        if (i + 1 < nx) e.emplace_back(id(i, j, k), id(i + 1, j, k));
        if (j + 1 < ny) e.emplace_back(id(i, j, k), id(i, j + 1, k));
        if (k + 1 < nz) e.emplace_back(id(i, j, k), id(i, j, k + 1));
      }
  return e;
}

// `count` disjoint paths of `len` vertices each.
Edges paths(int count, int len) {
  Edges e;
  for (int p = 0; p < count; ++p)
    for (int i = 0; i + 1 < len; ++i) e.emplace_back(p * len + i, p * len + i + 1);
  return e;
}

// Hub vertex 0 joined to every other vertex, plus the path 1 - 2 - ... - (n-1).
Edges hub_and_path(int n) {
  Edges e;
  for (int v = 1; v < n; ++v) {
    e.emplace_back(0, v);
    if (v + 1 < n) e.emplace_back(v, v + 1);
  }
  return e;
}

// Symmetric matrix (diagonal 4, off-diagonal -1) with the pattern of `edges`.
template <class T>
CsrMatrix<T> matrix_from_edges(std::int32_t n, const Edges& edges) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t p = 0; p < n; ++p) r.push_back(p), c.push_back(p), v.push_back(T(4.0));
  for (auto [a, b] : edges) {
    r.push_back(a), c.push_back(b), v.push_back(T(-1.0));
    r.push_back(b), c.push_back(a), v.push_back(T(-1.0));
  }
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

bool is_pivot_order(const std::vector<int>& perm, index_t m) {
  if (static_cast<index_t>(perm.size()) != m) return false;
  std::vector<char> seen(static_cast<std::size_t>(m), 0);
  for (const int p : perm) {
    if (p < 1 || p > m || seen[p - 1]) return false;
    seen[p - 1] = 1;
  }
  return true;
}

// ------------------------------------------------------------------- coarsening targets

// coarsen_to below 10 on graphs with more components than that: once every component is one
// coarse vertex, a contraction removes nothing; it used to be pushed again and again (unbounded
// time and memory).
TEST(FactorOrderingCoarsening, SmallTargetsTerminateOnManyComponents) {
  struct Case {
    std::int32_t n;
    Edges edges;
    int refine_passes;
  };
  std::vector<Case> cases;
  cases.push_back({150, paths(3, 50), 10});
  cases.push_back({210, paths(7, 30), 10});
  cases.push_back({480, paths(24, 20), 10});
  cases.push_back({2100, hub_and_path(2100), 0});   // the fuzz-found case
  cases.push_back({1630, grid_edges(40, 40), 10});  // a grid plus 30 isolated vertices
  for (const Case& cs : cases) {
    const auto a = matrix_from_edges<double>(cs.n, cs.edges);
    const ShiftedPencil<double> pencil(a, nullptr);
    for (index_t to = 2; to <= 10; ++to) {
      hs::OrderingOptions o;
      o.coarsen_to = to;
      o.refine_passes = cs.refine_passes;
      o.threads = 2;
      EXPECT_TRUE(is_pivot_order(hs::compute_ordering(pencil, o), cs.n))
          << "n=" << cs.n << " coarsen_to=" << to;
    }
  }
  // The separator itself, at the smallest target.
  const ord::Graph g = graph_from_edges(150, paths(3, 50));
  ord::SeparatorParams p;
  p.coarsen_to = 2;
  ord::SeparatorTrace tr;
  const auto where = ord::vertex_separator(g, 7, p, &tr);
  EXPECT_TRUE(ord::check_separator(g, where).valid);
  for (std::size_t k = 1; k < tr.vertices.size(); ++k)
    EXPECT_LT(tr.vertices[k], tr.vertices[k - 1]) << "level " << k << " removed nothing";
}

// ------------------------------------------------------------- inside a parallel region

// compute_ordering with threads = 1 used to run without a parallel region of its own: called
// from inside an active team, its dissection tasks were deferred to that team and outlived the
// call (invalid permutations, crashes).
TEST(FactorOrderingThreads, CallableFromInsideAParallelRegion) {
  const auto a = hs::test::laplacian_2d<double>(120, 120);
  const ShiftedPencil<double> pencil(a, nullptr);
  hs::OrderingOptions o;
  o.threads = 1;
  const std::vector<int> ref = hs::compute_ordering(pencil, o);
  ASSERT_TRUE(is_pivot_order(ref, pencil.order()));
  std::vector<std::vector<int>> single_one(2), single_two(2), each(4);
#pragma omp parallel num_threads(2) default(shared)
  {
#pragma omp single
    for (int rep = 0; rep < 2; ++rep) {
      hs::OrderingOptions oo = o;
      single_one[rep] = hs::compute_ordering(pencil, oo);
      oo.threads = 2;
      single_two[rep] = hs::compute_ordering(pencil, oo);
    }
    // One ordering per thread of the team (the one-thread-per-instance pattern).
#pragma omp for schedule(static)
    for (int k = 0; k < 4; ++k) each[k] = hs::compute_ordering(pencil, o);
  }
  for (int rep = 0; rep < 2; ++rep) {
    EXPECT_EQ(single_one[rep], ref) << "threads = 1 inside single, rep " << rep;
    EXPECT_EQ(single_two[rep], ref) << "threads = 2 inside single, rep " << rep;
  }
  for (int k = 0; k < 4; ++k) EXPECT_EQ(each[k], ref) << "omp for, iteration " << k;
}

using Scalars = ::testing::Types<double, cplx>;
template <class T>
class FactorOrderingThreadsPencil : public ::testing::Test {};
TYPED_TEST_SUITE(FactorOrderingThreadsPencil, Scalars);

// FactorOptions{ordering = kOrderingNestedDissection, threads = 1} analysed concurrently from an
// OpenMP team (MUMPS calls are serialized by the factor module; the ordering runs outside).
TYPED_TEST(FactorOrderingThreadsPencil, FactorizationsInsideAParallelRegion) {
  using T = TypeParam;
  const index_t nx = 120, ny = 90;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  const ShiftedPencil<T> pencil(a, nullptr);
  index_t k = 2000;  // a shift in a gap of the (sorted) spectrum, k eigenvalues below it
  while (w[k] - w[k - 1] < 1e-6) ++k;
  const double sigma = 0.5 * (w[k - 1] + w[k]);
  std::vector<index_t> neg(4, -1);
  std::vector<int> ok(4, 0);
#pragma omp parallel for num_threads(2) schedule(static) default(shared)
  for (int it = 0; it < 4; ++it) {
    hs::FactorOptions fo;
    fo.ordering = hs::kOrderingNestedDissection;
    fo.threads = 1;
    auto f = hs::make_mumps_factorization<T>(fo);
    f->analyze(pencil, 1.0, -sigma);
    ok[it] = f->factorize(1.0, -sigma) ? 1 : 0;
    neg[it] = f->inertia().neg;
  }
  for (int it = 0; it < 4; ++it) {
    EXPECT_EQ(ok[it], 1) << "iteration " << it;
    EXPECT_EQ(neg[it], k) << "iteration " << it;
  }
}

}  // namespace
