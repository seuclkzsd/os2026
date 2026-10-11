// Unit tests for hsolve's own fill-reducing ordering (src/factor/ordering*.cpp, compute_ordering,
// FactorOptions::ordering = kOrderingNestedDissection): graph construction, vertex separators,
// minimum degree, the parallel nested dissection (permutation validity, independence of the
// thread count), the PERM_IN handed to MUMPS (real and complex embedding), and factorizations
// with it (exact inertia, backward-stable solves, fill comparable to METIS).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "factor/mumps_factorization.hpp"
#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::ShiftedPencil;
namespace ord = hs::ordering;

// ------------------------------------------------------------------------------- helpers

// Graph of a symmetric pattern given as an edge list (both directions are added).
ord::Graph graph_from_edges(std::int32_t n, const std::vector<std::pair<std::int32_t, std::int32_t>>& edges) {
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
    col.push_back(i);  // diagonal, ignored by graph_from_upper
    col.insert(col.end(), r.begin(), r.end());
    rowptr.push_back(static_cast<std::int64_t>(col.size()));
  }
  return ord::graph_from_upper(n, rowptr.data(), col.data());
}

// Edges of an nx x ny (x nz) grid with the 5-point (7-point) stencil.
std::vector<std::pair<std::int32_t, std::int32_t>> grid_edges(int nx, int ny, int nz = 1, int offset = 0) {
  std::vector<std::pair<std::int32_t, std::int32_t>> e;
  auto id = [&](int i, int j, int k) { return offset + i + nx * (j + ny * k); };
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        if (i + 1 < nx) e.emplace_back(id(i, j, k), id(i + 1, j, k));
        if (j + 1 < ny) e.emplace_back(id(i, j, k), id(i, j + 1, k));
        if (k + 1 < nz) e.emplace_back(id(i, j, k), id(i, j, k + 1));
      }
  return e;
}

// 3-D 7-point Dirichlet Laplacian on an m^3 grid.
template <class T>
CsrMatrix<T> laplacian_3d(int m) {
  const index_t n = static_cast<index_t>(m) * m * m;
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t p = 0; p < n; ++p) {
    r.push_back(p), c.push_back(p), v.push_back(T(6.0));
  }
  for (auto [a, b] : grid_edges(m, m, m)) {
    r.push_back(a), c.push_back(b), v.push_back(T(-1.0));
    r.push_back(b), c.push_back(a), v.push_back(T(-1.0));
  }
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

bool is_permutation(const std::vector<std::int32_t>& order, std::int32_t n) {
  if (static_cast<std::int32_t>(order.size()) != n) return false;
  std::vector<char> seen(static_cast<std::size_t>(n), 0);
  for (const std::int32_t v : order) {
    if (v < 0 || v >= n || seen[v]) return false;
    seen[v] = 1;
  }
  return true;
}

// Fill-in (entries of L below the diagonal not in the pattern) of eliminating g in `order`,
// by explicit symbolic elimination (small graphs only).
std::int64_t fill_in(const ord::Graph& g, const std::vector<std::int32_t>& order) {
  std::vector<std::set<std::int32_t>> adj(static_cast<std::size_t>(g.n));
  for (std::int32_t v = 0; v < g.n; ++v)
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) adj[v].insert(g.adj[e]);
  std::int64_t fill = 0;
  std::vector<char> done(static_cast<std::size_t>(g.n), 0);
  for (const std::int32_t v : order) {
    std::vector<std::int32_t> nb;
    for (const std::int32_t u : adj[v])
      if (!done[u]) nb.push_back(u);
    for (std::size_t a = 0; a < nb.size(); ++a)
      for (std::size_t b = a + 1; b < nb.size(); ++b)
        if (adj[nb[a]].insert(nb[b]).second) {
          adj[nb[b]].insert(nb[a]);
          ++fill;
        }
    done[v] = 1;
  }
  return fill;
}

template <class T>
double solve_backward_error(hs::SparseFactorization<T>& f, const CsrMatrix<T>& k) {
  const index_t n = k.nrows;
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<T> b(static_cast<std::size_t>(n)), x;
  for (auto& bi : b) {
    if constexpr (hs::is_complex_v<T>) bi = T(u(rng), u(rng));
    else bi = u(rng);
  }
  x = b;
  f.solve(x.data(), 1, n);
  std::vector<T> r(static_cast<std::size_t>(n));
  hs::spmv(k, x.data(), r.data());
  for (index_t i = 0; i < n; ++i) r[i] -= b[i];
  return hs::linalg::norm2(n, r.data()) / (hs::norm_one(k) * hs::linalg::norm2(n, x.data()));
}

index_t count_below(const std::vector<double>& w, double s) {
  return std::lower_bound(w.begin(), w.end(), s) - w.begin();
}

// --------------------------------------------------------------------------------- graphs

TEST(FactorOrderingGraph, FromUpperBuildsSortedSymmetricAdjacency) {
  // Upper pattern of a 4x4 matrix with entries (0,1), (0,3), (1,2), (2,3) plus diagonals.
  const std::vector<std::int64_t> rowptr{0, 3, 5, 7, 8};
  const std::vector<std::int32_t> col{0, 1, 3, 1, 2, 2, 3, 3};
  const ord::Graph g = ord::graph_from_upper(4, rowptr.data(), col.data());
  ASSERT_EQ(g.n, 4);
  EXPECT_EQ(g.edges(), 8);
  EXPECT_EQ(g.total_vwgt, 4);
  const std::vector<std::vector<std::int32_t>> want{{1, 3}, {0, 2}, {1, 3}, {0, 2}};
  for (std::int32_t v = 0; v < 4; ++v) {
    std::vector<std::int32_t> row(g.adj.begin() + g.xadj[v], g.adj.begin() + g.xadj[v + 1]);
    EXPECT_EQ(row, want[v]) << "row " << v;
  }
  EXPECT_TRUE(std::all_of(g.ewgt.begin(), g.ewgt.end(), [](int w) { return w == 1; }));
}

TEST(FactorOrderingGraph, InducedSubgraphKeepsInternalEdges) {
  const ord::Graph g = graph_from_edges(9, grid_edges(3, 3));
  std::vector<std::uint8_t> where{0, 0, 2, 0, 0, 2, 2, 2, 1};
  std::vector<std::int32_t> to_parent;
  const ord::Graph s = ord::induced_subgraph(g, where, 0, to_parent);
  EXPECT_EQ(to_parent, (std::vector<std::int32_t>{0, 1, 3, 4}));
  EXPECT_EQ(s.n, 4);
  EXPECT_EQ(s.edges(), 8);  // the 2x2 square
  const ord::SeparatorCheck c = ord::check_separator(g, where);
  EXPECT_TRUE(c.valid);
  EXPECT_EQ(c.weight[2], 4);
  where[6] = 1;  // now (3,6) joins halves 0 and 1 directly
  EXPECT_FALSE(ord::check_separator(g, where).valid);
}

// ------------------------------------------------------------------------------ separators

TEST(FactorOrderingSeparator, GridSeparatorsAreValidBalancedAndSmall) {
  struct Case {
    int nx, ny, nz;
    std::int64_t max_sep;  // optimum: the smallest cross-section (nx*ny for the 3-D grid)
  };
  for (const Case cs : {Case{40, 40, 1, 60}, Case{90, 30, 1, 45}, Case{14, 14, 14, 280}}) {
    const ord::Graph g = graph_from_edges(cs.nx * cs.ny * cs.nz, grid_edges(cs.nx, cs.ny, cs.nz));
    const ord::SeparatorParams p;
    for (std::uint64_t seed : {1ULL, 2ULL, 3ULL}) {
      const auto where = ord::vertex_separator(g, seed, p);
      const ord::SeparatorCheck c = ord::check_separator(g, where);
      ASSERT_TRUE(c.valid);
      EXPECT_LE(c.weight[2], cs.max_sep) << cs.nx << "x" << cs.ny << "x" << cs.nz << " seed " << seed;
      EXPECT_GT(c.weight[0], 0);
      EXPECT_GT(c.weight[1], 0);
      const double bound = p.imbalance * 0.5 * static_cast<double>(g.n) + 2;
      EXPECT_LE(c.weight[0], bound);
      EXPECT_LE(c.weight[1], bound);
    }
  }
}

TEST(FactorOrderingSeparator, DeterministicForAFixedSeed) {
  const ord::Graph g = graph_from_edges(12 * 12 * 12, grid_edges(12, 12, 12));
  const ord::SeparatorParams p;
  const auto a = ord::vertex_separator(g, 42, p);
  const auto b = ord::vertex_separator(g, 42, p);
  EXPECT_EQ(a, b);
}

TEST(FactorOrderingSeparator, LargeGraphUsesTheTwoStageSchemeAndTrace) {
  // 6000 vertices: above the two-stage threshold; the trace lists the levels finest first.
  const ord::Graph g = graph_from_edges(100 * 60, grid_edges(100, 60));
  ord::SeparatorTrace tr;
  const auto where = ord::vertex_separator(g, 9, ord::SeparatorParams{}, &tr);
  const ord::SeparatorCheck c = ord::check_separator(g, where);
  ASSERT_TRUE(c.valid);
  EXPECT_LE(c.weight[2], 90);  // optimum 60
  ASSERT_GE(tr.vertices.size(), 3u);
  EXPECT_EQ(tr.vertices.front(), g.n);
  EXPECT_TRUE(std::is_sorted(tr.vertices.rbegin(), tr.vertices.rend()));
  EXPECT_EQ(tr.refined.front(), c.weight[2]);
  EXPECT_EQ(tr.refined.size(), tr.vertices.size());
}

TEST(FactorOrderingSeparator, DisconnectedAndEdgelessGraphs) {
  // Two disjoint 20x20 grids: an empty separator is optimal; any valid answer is accepted, but
  // it must be no larger than one grid's cross-section.
  auto e = grid_edges(20, 20);
  const auto e2 = grid_edges(20, 20, 1, 400);
  e.insert(e.end(), e2.begin(), e2.end());
  const ord::Graph g = graph_from_edges(800, e);
  const ord::SeparatorCheck c = ord::check_separator(g, ord::vertex_separator(g, 5, ord::SeparatorParams{}));
  EXPECT_TRUE(c.valid);
  EXPECT_LE(c.weight[2], 20);
  // No edges at all.
  const ord::Graph empty = graph_from_edges(50, {});
  const ord::SeparatorCheck ce = ord::check_separator(empty, ord::vertex_separator(empty, 5, ord::SeparatorParams{}));
  EXPECT_TRUE(ce.valid);
  EXPECT_EQ(ce.weight[2], 0);
}

// -------------------------------------------------------------------------- minimum degree

TEST(FactorOrderingMinimumDegree, TreesAreEliminatedWithoutFill) {
  // Path, star and a random tree: a minimum-degree order eliminates leaves first (no fill).
  std::vector<std::pair<std::int32_t, std::int32_t>> path, star, tree;
  for (int i = 0; i + 1 < 40; ++i) path.emplace_back(i, i + 1);
  for (int i = 1; i < 40; ++i) star.emplace_back(0, i);
  std::mt19937 rng(3);
  for (int i = 1; i < 40; ++i) tree.emplace_back(i, static_cast<int>(rng() % i));
  for (const auto* edges : {&path, &star, &tree}) {
    const ord::Graph g = graph_from_edges(40, *edges);
    const auto order = ord::minimum_degree(g);
    ASSERT_TRUE(is_permutation(order, g.n));
    EXPECT_EQ(fill_in(g, order), 0);
  }
}

TEST(FactorOrderingMinimumDegree, BeatsNaturalOrderOnAGrid) {
  const ord::Graph g = graph_from_edges(12 * 12, grid_edges(12, 12));
  std::vector<std::int32_t> natural(144);
  std::iota(natural.begin(), natural.end(), 0);
  const auto md = ord::minimum_degree(g);
  ASSERT_TRUE(is_permutation(md, g.n));
  EXPECT_LT(fill_in(g, md), fill_in(g, natural));
}

// -------------------------------------------------------------------------- dissection

TEST(FactorOrderingDissection, IsAPermutationOnAssortedGraphs) {
  std::vector<std::pair<std::int32_t, std::vector<std::pair<std::int32_t, std::int32_t>>>> cases;
  cases.push_back({1, {}});
  cases.push_back({7, {}});
  cases.push_back({60 * 50, grid_edges(60, 50)});
  cases.push_back({16 * 16 * 16, grid_edges(16, 16, 16)});
  {
    auto e = grid_edges(30, 30);  // grid plus isolated vertices and a separate path
    for (int i = 0; i + 1 < 200; ++i) e.emplace_back(1000 + i, 1001 + i);
    cases.push_back({1300, e});
  }
  {
    std::vector<std::pair<std::int32_t, std::int32_t>> clique;  // dense: no useful separator
    for (int i = 0; i < 300; ++i)
      for (int j = i + 1; j < 300; ++j) clique.emplace_back(i, j);
    cases.push_back({300, clique});
  }
  {
    std::vector<std::pair<std::int32_t, std::int32_t>> star;
    for (int i = 1; i < 3000; ++i) star.emplace_back(0, i);
    cases.push_back({3000, star});
  }
  for (const auto& [n, edges] : cases) {
    ord::DissectionStats st;
    const auto order = ord::nested_dissection(graph_from_edges(n, edges), ord::DissectionParams{}, 2, &st);
    EXPECT_TRUE(is_permutation(order, n)) << "n=" << n << ", " << edges.size() << " edges";
    EXPECT_EQ(st.threads, 2);
  }
}

TEST(FactorOrderingDissection, IndependentOfTheThreadCount) {
  // 20^3 = 8000 vertices: two-stage separators with parallel runs and dissection tasks.
  const ord::Graph g = graph_from_edges(8000, grid_edges(20, 20, 20));
  ord::DissectionParams p;
  p.task_size = 500;  // many tasks
  ord::DissectionStats s1;
  const auto ref = ord::nested_dissection(g, p, 1, &s1);
  ASSERT_TRUE(is_permutation(ref, g.n));
  EXPECT_GT(s1.separators, 10);
  EXPECT_EQ(s1.top_separator, s1.depth_separator.at(0));
  for (int threads : {2, 3, 4}) {
    const auto o = ord::nested_dissection(g, p, threads);
    EXPECT_EQ(o, ref) << threads << " threads";
  }
  EXPECT_EQ(ord::nested_dissection(g, p, 2), ref) << "repeated call";
  p.seed += 1;
  EXPECT_NE(ord::nested_dissection(g, p, 2), ref) << "the seed should matter";
}

TEST(FactorOrderingDissection, OrdersSeparatorsAfterTheirHalves) {
  // On a 2-D grid, nested dissection fills far less than the natural (banded) order.
  const ord::Graph g = graph_from_edges(30 * 30, grid_edges(30, 30));
  ord::DissectionParams p;
  p.leaf_size = 20;
  const auto nd = ord::nested_dissection(g, p, 1);
  std::vector<std::int32_t> natural(900);
  std::iota(natural.begin(), natural.end(), 0);
  EXPECT_LT(fill_in(g, nd), fill_in(g, natural));
}

// ------------------------------------------------------------------- pencils and MUMPS

using Scalars = ::testing::Types<double, cplx>;
template <class T>
class FactorOrderingPencil : public ::testing::Test {};
TYPED_TEST_SUITE(FactorOrderingPencil, Scalars);

TYPED_TEST(FactorOrderingPencil, PivotOrderIsAValidPermutation) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(70, 40);
  const auto b = hs::test::random_hpd<T>(70 * 40, 3.0 / 2800, 11);  // union pattern A + B
  const ShiftedPencil<T> pencil(a, &b);
  hs::OrderingStats st;
  hs::OrderingOptions o;
  o.threads = 2;
  o.embedding = hs::OrderingOptions::Embedding::Compressed;  // pairs (complex T)
  const std::vector<int> perm = hs::compute_ordering(pencil, o, &st);
  const index_t n = pencil.order();
  const index_t m = hs::is_complex_v<T> ? 2 * n : n;
  ASSERT_EQ(static_cast<index_t>(perm.size()), m);
  std::vector<char> seen(static_cast<std::size_t>(m), 0);
  for (const int p : perm) {
    ASSERT_GE(p, 1);
    ASSERT_LE(p, m);
    ASSERT_FALSE(seen[p - 1]);
    seen[p - 1] = 1;
  }
  if constexpr (hs::is_complex_v<T>) {
    for (index_t v = 0; v < n; ++v) {  // Re and Im of a vertex are eliminated back to back
      EXPECT_EQ(perm[v] % 2, 1);
      EXPECT_EQ(perm[n + v], perm[v] + 1);
    }
  }
  EXPECT_EQ(st.vertices, n);
  EXPECT_EQ(st.threads, 2);
  EXPECT_GT(st.separators, 0);
  EXPECT_GT(st.seconds, 0.0);
  // Deterministic, also across thread counts.
  o.threads = 1;
  EXPECT_EQ(hs::compute_ordering(pencil, o), perm);
}

TYPED_TEST(FactorOrderingPencil, InertiaAndSolvesAreExact) {
  using T = TypeParam;
  const index_t nx = 60, ny = 45;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  const ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.ordering = hs::kOrderingNestedDissection;
  fo.ordering_threads = 2;
  auto f = hs::make_mumps_factorization<T>(fo);
  f->analyze(pencil, 1.0, -4.0);
  EXPECT_GT(f->stats().ordering_s, 0.0);
  EXPECT_LE(f->stats().ordering_s, f->stats().analyze_s);
  for (double sigma : {0.1, 1.3, 2.9, 4.05, 6.7}) {
    // Shifts between eigenvalues (the spectrum is fine-grained: move off any eigenvalue).
    const index_t k = count_below(w, sigma);
    const double s = (k > 0 && k < static_cast<index_t>(w.size())) ? 0.5 * (w[k - 1] + w[k]) : sigma;
    ASSERT_TRUE(f->factorize(1.0, -s)) << "sigma=" << s;
    EXPECT_EQ(f->inertia().neg, count_below(w, s)) << "sigma=" << s;
    const CsrMatrix<T> k_mat = hs::add(T(1.0), a, T(-s), hs::identity<T>(a.nrows));
    EXPECT_LT(solve_backward_error(*f, k_mat), 1e-13) << "sigma=" << s;
  }
}

TYPED_TEST(FactorOrderingPencil, GenuinelyComplexSpectrumCountsMatch) {
  using T = TypeParam;
  const index_t n = 1200;
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) w[i] = -2.0 + 5.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
  const auto a = hs::test::prescribed_spectrum<T>(w, 12, 2026);
  const ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions nd;
  nd.ordering = hs::kOrderingNestedDissection;
  hs::FactorOptions amd;
  amd.ordering = 0;
  auto fn = hs::make_mumps_factorization<T>(nd);
  auto fa = hs::make_mumps_factorization<T>(amd);
  fn->analyze(pencil, 1.0, 0.0);
  fa->analyze(pencil, 1.0, 0.0);
  for (index_t k : {100, 600, 1100}) {
    const double s = 0.5 * (w[k - 1] + w[k]);
    ASSERT_TRUE(fn->factorize(1.0, -s));
    ASSERT_TRUE(fa->factorize(1.0, -s));
    EXPECT_EQ(fn->inertia().neg, k);
    EXPECT_EQ(fa->inertia().neg, k);
  }
}

TYPED_TEST(FactorOrderingPencil, FillIsComparableToMetis) {
  using T = TypeParam;
  const auto a = laplacian_3d<T>(18);  // 5832 unknowns
  const ShiftedPencil<T> pencil(a, nullptr);
  auto estimate = [&](int ordering) {
    hs::FactorOptions o;
    o.ordering = ordering;
    auto f = hs::make_mumps_factorization<T>(o);
    f->analyze(pencil, 1.0, -0.5);
    return f->estimate();
  };
  const hs::FactorEstimate nd = estimate(hs::kOrderingNestedDissection);
  const hs::FactorEstimate metis = estimate(5);
  const hs::FactorEstimate amd = estimate(0);
  EXPECT_LE(nd.flops, 1.25 * metis.flops) << "nd " << nd.flops << " metis " << metis.flops;
  EXPECT_LT(nd.flops, amd.flops) << "nd " << nd.flops << " amd " << amd.flops;
  EXPECT_LE(nd.factor_entries, 1.15 * metis.factor_entries);
}

TYPED_TEST(FactorOrderingPencil, InertiaCounterReusesTheOrder) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(50, 50);
  const auto w = hs::test::laplacian_2d_eigenvalues(50, 50);
  const ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.ordering = hs::kOrderingNestedDissection;
  auto f = hs::make_mumps_factorization<T>(fo);
  f->analyze(pencil, 1.0, -1.0);
  hs::InertiaCounter<T> counter(*f, fo);
  EXPECT_TRUE(counter.stats().reused_pivot_order);
  const index_t k = 700;
  const double s = 0.5 * (w[k - 1] + w[k]);
  const hs::Inertia in = counter.count(1.0, -s);
  EXPECT_TRUE(in.valid);
  EXPECT_EQ(in.neg, k);
}

// Null-pivot detection depends on the elimination order (factor.hpp): shifts exactly on
// eigenvalues must still be flagged with the nested-dissection order, and perturbed shifts
// factorize with the exact inertia (as FactorSingular checks for the MUMPS orderings).
TYPED_TEST(FactorOrderingPencil, ShiftOnAnEigenvalueIsSingular) {
  using T = TypeParam;
  for (auto [nx, ny] : {std::pair<index_t, index_t>{50, 51}, {60, 60}}) {
    const auto a = hs::test::laplacian_2d<T>(nx, ny);
    const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
    const ShiftedPencil<T> pencil(a, nullptr);
    hs::FactorOptions fo;
    fo.ordering = hs::kOrderingNestedDissection;
    auto f = hs::make_mumps_factorization<T>(fo);
    f->analyze(pencil, 1.0, -3.0);
    const index_t n = static_cast<index_t>(w.size());
    for (int t = 0; t < 16; ++t) {
      const double lambda = w[(n - 1) * t / 15];
      EXPECT_FALSE(f->factorize(1.0, -lambda)) << nx << "x" << ny << " lambda=" << lambda;
      EXPECT_FALSE(f->inertia().valid);
      const double sigma = lambda + 8e-8;
      ASSERT_TRUE(f->factorize(1.0, -sigma)) << "perturbed shift " << sigma;
      EXPECT_EQ(f->inertia().neg, count_below(w, sigma));
    }
  }
}

// ---------------------------------------------------------------- complex embedding choices

// 2-D Laplacian whose x-couplings carry a phase on every fifth grid row (Hermitian; about 10% of
// the couplings complex), and the same with a phase on every x-coupling (about 50%).
CsrMatrix<cplx> phased_laplacian(index_t nx, index_t ny, int every_row) {
  std::vector<index_t> r, c;
  std::vector<cplx> v;
  auto add_edge = [&](index_t p, index_t q, cplx x) {
    r.push_back(p), c.push_back(q), v.push_back(x);
    r.push_back(q), c.push_back(p), v.push_back(std::conj(x));
  };
  for (index_t j = 0; j < ny; ++j)
    for (index_t i = 0; i < nx; ++i) {
      const index_t p = i + j * nx;
      r.push_back(p), c.push_back(p), v.push_back(cplx(4.0));
      const cplx x = (j % every_row == 0) ? -std::polar(1.0, 0.3 * static_cast<double>(i + 1)) : cplx(-1.0);
      if (i + 1 < nx) add_edge(p, p + 1, x);
      if (j + 1 < ny) add_edge(p, p + nx, cplx(-1.0));
    }
  return hs::csr_from_coo<cplx>(nx * ny, nx * ny, r, c, v);
}

TEST(FactorOrderingEmbedding, CouplingFractionCountsComplexCouplings) {
  const auto real_valued = hs::test::laplacian_2d<cplx>(20, 20);
  const ShiftedPencil<cplx> p0(real_valued, nullptr);
  EXPECT_EQ(ord::complex_coupling_fraction(p0), 0.0);
  const auto partly = phased_laplacian(20, 20, 5);  // 4 of 20 rows: 4*19 of 2*20*19 couplings
  const ShiftedPencil<cplx> p1(partly, nullptr);
  EXPECT_NEAR(ord::complex_coupling_fraction(p1), 4.0 * 19 / (2.0 * 20 * 19), 1e-12);
  std::vector<double> w(300);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = -1.0 + 0.01 * static_cast<double>(i);
  const auto full = hs::test::prescribed_spectrum<cplx>(w, 6, 3);
  const ShiftedPencil<cplx> p2(full, nullptr);
  EXPECT_GT(ord::complex_coupling_fraction(p2), 0.9);
  const auto real_matrix = hs::test::laplacian_2d<double>(20, 20);
  const ShiftedPencil<double> p3(real_matrix, nullptr);
  EXPECT_EQ(ord::complex_coupling_fraction(p3), 0.0);
}

TEST(FactorOrderingEmbedding, AutoOrdersTheFullEmbeddingWhenFewCouplingsAreComplex) {
  const auto a = phased_laplacian(40, 30, 5);
  const ShiftedPencil<cplx> pencil(a, nullptr);
  const index_t n = pencil.order();
  hs::OrderingStats st;
  hs::OrderingOptions o;
  const std::vector<int> perm = hs::compute_ordering(pencil, o, &st);
  EXPECT_FALSE(st.compressed);
  EXPECT_EQ(st.vertices, 2 * n);
  std::vector<int> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  for (index_t k = 0; k < 2 * n; ++k) ASSERT_EQ(sorted[k], k + 1);
  o.embedding = hs::OrderingOptions::Embedding::Compressed;
  const std::vector<int> paired = hs::compute_ordering(pencil, o, &st);
  EXPECT_TRUE(st.compressed);
  EXPECT_EQ(st.vertices, n);
  for (index_t v = 0; v < n; ++v) EXPECT_EQ(paired[n + v], paired[v] + 1);
  // Both orders factorize the embedding with the inertia of an AMD factorization, and solve.
  hs::FactorOptions amd;
  amd.ordering = 0;
  auto ref = hs::make_mumps_factorization<cplx>(amd);
  ref->analyze(pencil, 1.0, -2.0);
  ASSERT_TRUE(ref->factorize(1.0, -2.0));
  const CsrMatrix<cplx> km = hs::add(cplx(1.0), a, cplx(-2.0), hs::identity<cplx>(n));
  for (auto mode : {hs::OrderingOptions::Embedding::Full, hs::OrderingOptions::Embedding::Compressed}) {
    hs::OrderingOptions m;
    m.embedding = mode;
    hs::mumps::MumpsFactorization<cplx, double> f{hs::FactorOptions{}};
    f.set_pivot_order(hs::compute_ordering(pencil, m));
    f.analyze(pencil, 1.0, -2.0);
    ASSERT_TRUE(f.factorize(1.0, -2.0));
    EXPECT_EQ(f.inertia().neg, ref->inertia().neg);
    EXPECT_LT(solve_backward_error(f, km), 1e-13);
  }
}

TEST(FactorBlockAnalysis, DeclaredBlocksKeepTheInertiaAndSingularDetection) {
  const index_t n = 1500;
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) w[i] = -2.0 + 5.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
  const auto a = hs::test::prescribed_spectrum<cplx>(w, 12, 2026);
  const ShiftedPencil<cplx> pencil(a, nullptr);
  for (int ordering : {0, 3, 5}) {
    for (int blocks : {-1, 1}) {
      hs::FactorOptions fo;
      fo.ordering = ordering;
      fo.block_analysis = blocks;
      auto f = hs::make_mumps_factorization<cplx>(fo);
      f->analyze(pencil, 1.0, 0.0);
      EXPECT_TRUE(f->stats().block_analysis) << "ordering " << ordering << " blocks " << blocks;
      for (index_t k : {150, 700, 1350}) {
        const double s = 0.5 * (w[k - 1] + w[k]);
        ASSERT_TRUE(f->factorize(1.0, -s));
        EXPECT_EQ(f->inertia().neg, k) << "ordering " << ordering;
        const CsrMatrix<cplx> km = hs::add(cplx(1.0), a, cplx(-s), hs::identity<cplx>(n));
        EXPECT_LT(solve_backward_error(*f, km), 1e-13);
      }
      for (index_t k : {100, 900}) {
        EXPECT_FALSE(f->factorize(1.0, -w[k])) << "exact eigenvalue " << w[k] << ", ordering " << ordering;
        EXPECT_FALSE(f->inertia().valid);
      }
    }
  }
}

TEST(FactorBlockAnalysis, AutoFollowsTheComplexFractionAndIgnoresGivenOrders) {
  const auto partly = phased_laplacian(40, 40, 5);
  const ShiftedPencil<cplx> pencil(partly, nullptr);
  hs::FactorOptions fo;
  fo.ordering = 0;
  fo.block_analysis = -1;
  auto f = hs::make_mumps_factorization<cplx>(fo);
  f->analyze(pencil, 1.0, -1.0);
  EXPECT_FALSE(f->stats().block_analysis);  // about 10% of the couplings are complex
  fo.block_analysis = 1;
  auto g = hs::make_mumps_factorization<cplx>(fo);
  g->analyze(pencil, 1.0, -1.0);
  EXPECT_TRUE(g->stats().block_analysis);
  ASSERT_TRUE(f->factorize(1.0, -1.0));
  ASSERT_TRUE(g->factorize(1.0, -1.0));
  EXPECT_EQ(f->inertia().neg, g->inertia().neg);
  // hsolve's own ordering is a given order: no MUMPS blocks.
  fo.ordering = hs::kOrderingNestedDissection;
  auto h = hs::make_mumps_factorization<cplx>(fo);
  h->analyze(pencil, 1.0, -1.0);
  EXPECT_FALSE(h->stats().block_analysis);
  // Real T: nothing to declare.
  const auto real_matrix = hs::test::laplacian_2d<double>(30, 30);
  const ShiftedPencil<double> rp(real_matrix, nullptr);
  hs::FactorOptions ro;
  ro.block_analysis = 1;
  auto r = hs::make_mumps_factorization<double>(ro);
  r->analyze(rp, 1.0, 0.0);
  EXPECT_FALSE(r->stats().block_analysis);
}

TEST(FactorOrderingOptions, InvalidValuesAreRejected) {
  const auto a = hs::test::laplacian_2d<double>(10, 10);
  const ShiftedPencil<double> pencil(a, nullptr);
  auto code_of = [](const std::function<void()>& fn) {
    try {
      fn();
    } catch (const hs::Error& e) {
      return e.code();
    }
    return hs::Error::Code::Internal;
  };
  const auto bad = [&](auto mutate) {
    hs::OrderingOptions o;
    mutate(o);
    return code_of([&] { static_cast<void>(hs::compute_ordering(pencil, o)); });
  };
  EXPECT_EQ(bad([](hs::OrderingOptions& o) { o.threads = -1; }), hs::Error::Code::InvalidInput);
  EXPECT_EQ(bad([](hs::OrderingOptions& o) { o.leaf_size = 0; }), hs::Error::Code::InvalidInput);
  EXPECT_EQ(bad([](hs::OrderingOptions& o) { o.imbalance = 0.9; }), hs::Error::Code::InvalidInput);
  EXPECT_EQ(bad([](hs::OrderingOptions& o) { o.coarsen_to = 1; }), hs::Error::Code::InvalidInput);
  EXPECT_EQ(bad([](hs::OrderingOptions& o) { o.init_tries = 0; }), hs::Error::Code::InvalidInput);
  hs::FactorOptions fo;
  fo.ordering_threads = -2;
  EXPECT_EQ(code_of([&] { hs::make_mumps_factorization<double>(fo); }), hs::Error::Code::InvalidInput);
  fo.ordering_threads = 0;
  fo.ordering = 9;
  EXPECT_EQ(code_of([&] { hs::make_mumps_factorization<double>(fo); }), hs::Error::Code::InvalidInput);
  fo.ordering = hs::kOrderingNestedDissection;
  EXPECT_NO_THROW(hs::make_mumps_factorization<double>(fo));
}

TEST(FactorOrderingOptions, MumpsOrderingsDoNotCountOrderingTime) {
  const auto a = hs::test::laplacian_2d<double>(30, 30);
  const ShiftedPencil<double> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.ordering = 0;
  auto f = hs::make_mumps_factorization<double>(fo);
  f->analyze(pencil, 1.0, 0.0);
  EXPECT_EQ(f->stats().ordering_s, 0.0);
  EXPECT_GT(f->stats().analyze_s, 0.0);
}

}  // namespace
