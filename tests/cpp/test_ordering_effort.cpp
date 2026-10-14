// Unit tests of the P9 additions to hsolve's nested dissection (src/factor/ordering*.cpp,
// compute_ordering): merging indistinguishable vertices (compress_graph, OrderingOptions::compress),
// weighted minimum degree on the leaves, trial separators refined down to their graph
// (SeparatorParams::finalists) and the effort on the top separators (OrderingOptions::top_*),
// the options' ranges, FactorOptions::nested_dissection, and the driver's choice of the effort
// (nd_ordering_options, OrderingPricer).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "driver/internal.hpp"
#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::ShiftedPencil;
namespace ord = hs::ordering;
using Edges = std::vector<std::pair<std::int32_t, std::int32_t>>;

ord::Graph graph_from_edges(std::int32_t n, const Edges& edges, std::vector<std::int32_t> vwgt = {}) {
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
  ord::Graph g = ord::graph_from_upper(n, rowptr.data(), col.data());
  if (!vwgt.empty()) {
    g.vwgt = std::move(vwgt);
    g.total_vwgt = std::accumulate(g.vwgt.begin(), g.vwgt.end(), std::int64_t{0});
  }
  return g;
}

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

// A field of `c` components on the nodes of the grid: variable c*v + a couples with every
// component of v and of v's neighbours, so the c variables of a node are indistinguishable.
Edges vector_field_edges(std::int32_t nodes, const Edges& grid, int c) {
  Edges e;
  for (std::int32_t v = 0; v < nodes; ++v)
    for (int a = 0; a < c; ++a)
      for (int b = a + 1; b < c; ++b) e.emplace_back(c * v + a, c * v + b);
  for (auto [u, v] : grid)
    for (int a = 0; a < c; ++a)
      for (int b = 0; b < c; ++b) e.emplace_back(c * u + a, c * v + b);
  return e;
}

// Pencil (K = A, B = I) with the pattern of `edges`: diagonally dominant.
template <class T>
CsrMatrix<T> matrix_from_edges(std::int32_t n, const Edges& edges) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  std::vector<double> deg(static_cast<std::size_t>(n), 1.0);
  for (auto [a, b] : edges) {
    r.push_back(a), c.push_back(b), v.push_back(T(-1.0));
    r.push_back(b), c.push_back(a), v.push_back(T(-1.0));
    deg[a] += 1.0, deg[b] += 1.0;
  }
  for (std::int32_t i = 0; i < n; ++i) r.push_back(i), c.push_back(i), v.push_back(T(deg[i]));
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

// ----------------------------------------------------------------------------- compression

TEST(FactorOrderingCompression, MergesTheComponentsOfAVectorField) {
  const int nx = 12, ny = 9, comps = 3;
  const std::int32_t nodes = nx * ny;
  const Edges grid = grid_edges(nx, ny);
  const ord::Graph g = graph_from_edges(comps * nodes, vector_field_edges(nodes, grid, comps));
  ord::Graph c;
  std::vector<std::int32_t> first, members;
  ASSERT_TRUE(ord::compress_graph(g, ord::kCompressionRatio, 2, c, first, members));
  ASSERT_EQ(c.n, nodes);
  EXPECT_EQ(c.total_vwgt, g.total_vwgt);
  ASSERT_EQ(first.size(), static_cast<std::size_t>(nodes) + 1);
  ASSERT_EQ(members.size(), static_cast<std::size_t>(g.n));
  // Class k is node k (classes are numbered by their smallest member), members increasing.
  for (std::int32_t k = 0; k < nodes; ++k) {
    ASSERT_EQ(first[k + 1] - first[k], comps);
    EXPECT_EQ(c.vwgt[k], comps);
    for (int a = 0; a < comps; ++a) EXPECT_EQ(members[first[k] + a], comps * k + a);
  }
  // The compressed graph is the grid itself: sorted rows, unit edge weights.
  const ord::Graph ref = graph_from_edges(nodes, grid);
  EXPECT_EQ(c.xadj, ref.xadj);
  EXPECT_EQ(c.adj, ref.adj);
  EXPECT_EQ(c.ewgt, std::vector<std::int32_t>(c.adj.size(), 1));
  // The same with any thread count.
  for (int threads : {1, 4}) {
    ord::Graph c2;
    std::vector<std::int32_t> f2, m2;
    ASSERT_TRUE(ord::compress_graph(g, ord::kCompressionRatio, threads, c2, f2, m2));
    EXPECT_EQ(c2.adj, c.adj);
    EXPECT_EQ(m2, members);
  }
}

TEST(FactorOrderingCompression, LeavesGraphsWithoutIndistinguishableVerticesAlone) {
  // A grid: no two vertices share their closed neighbourhood (neighbours u, v differ by v's other
  // neighbours) - and a graph merging only a few pairs stays above the ratio.
  const ord::Graph g = graph_from_edges(10 * 10, grid_edges(10, 10));
  ord::Graph c;
  std::vector<std::int32_t> first{1}, members{1};
  EXPECT_FALSE(ord::compress_graph(g, ord::kCompressionRatio, 2, c, first, members));
  EXPECT_TRUE(first.empty());
  EXPECT_TRUE(members.empty());
  EXPECT_EQ(c.n, 0);
  // Two isolated adjacent vertices (an edge 0-1 alone) are indistinguishable: 2 of 3 vertices
  // merge, 2 classes of 3 = 0.67 <= 0.85.
  const ord::Graph h = graph_from_edges(3, {{0, 1}});
  ASSERT_TRUE(ord::compress_graph(h, ord::kCompressionRatio, 1, c, first, members));
  EXPECT_EQ(c.n, 2);
  EXPECT_EQ(first, (std::vector<std::int32_t>{0, 2, 3}));
  EXPECT_EQ(c.vwgt, (std::vector<std::int32_t>{2, 1}));
  EXPECT_EQ(c.edges(), 0);
  // Below the ratio only: the same graph with max_ratio 0.5 is not compressed.
  EXPECT_FALSE(ord::compress_graph(h, 0.5, 1, c, first, members));
}

TEST(FactorOrderingCompression, ComputeOrderingEliminatesEachClassAsABlock) {
  const int nx = 30, ny = 25, comps = 2;
  const std::int32_t nodes = nx * ny, n = comps * nodes;
  const auto a = matrix_from_edges<double>(n, vector_field_edges(nodes, grid_edges(nx, ny), comps));
  const ShiftedPencil<double> pencil(a, nullptr);
  hs::OrderingOptions o;
  o.threads = 1;
  hs::OrderingStats st;
  const std::vector<int> perm = hs::compute_ordering(pencil, o, &st);
  EXPECT_EQ(st.vertices, n);
  EXPECT_EQ(st.ordered_vertices, nodes);
  ASSERT_EQ(perm.size(), static_cast<std::size_t>(n));
  std::vector<std::int32_t> order(static_cast<std::size_t>(n));
  for (std::int32_t v = 0; v < n; ++v) {
    ASSERT_GE(perm[v], 1);
    ASSERT_LE(perm[v], n);
    order[perm[v] - 1] = v;
  }
  ASSERT_TRUE(is_permutation(order, n));
  for (std::int32_t v = 0; v < n; v += comps) EXPECT_EQ(perm[v + 1], perm[v] + 1) << v;
  // Thread-independent, and compress = false is the uncompressed dissection.
  o.threads = 3;
  EXPECT_EQ(hs::compute_ordering(pencil, o), perm);
  o.compress = false;
  hs::OrderingStats su;
  const auto uncompressed = hs::compute_ordering(pencil, o, &su);
  EXPECT_EQ(su.ordered_vertices, n);
  EXPECT_NE(uncompressed, perm);
}

// --------------------------------------------------------------------- weighted min degree

TEST(FactorOrderingMinimumDegree, DegreesCountNeighbourWeights) {
  // 0 - 1 and the triangle 2 3 4: with unit weights 0 (degree 1) goes first; with vertex 1 of
  // weight 10 (ten indistinguishable variables) vertex 1 (degree w(0) = 1) goes first.
  const Edges e = {{0, 1}, {2, 3}, {2, 4}, {3, 4}};
  const auto unit = ord::minimum_degree(graph_from_edges(5, e));
  ASSERT_TRUE(is_permutation(unit, 5));
  EXPECT_EQ(unit.front(), 0);
  const auto weighted = ord::minimum_degree(graph_from_edges(5, e, {1, 10, 1, 1, 1}));
  ASSERT_TRUE(is_permutation(weighted, 5));
  EXPECT_EQ(weighted.front(), 1);
}

// ------------------------------------------------------------------------------- finalists

TEST(FactorOrderingSeparator, FinalistsAreValidDeterministicAndThreadIndependent) {
  const ord::Graph g = graph_from_edges(30 * 30 * 30, grid_edges(30, 30, 30));
  ASSERT_GE(g.n, ord::kTwoStageSize);
  ord::SeparatorParams p;
  p.runs = 8;
  p.finalists = 4;
  const auto ref = ord::vertex_separator(g, 7, p);
  const ord::SeparatorCheck chk = ord::check_separator(g, ref);
  ASSERT_TRUE(chk.valid);
  EXPECT_LE(std::max(chk.weight[0], chk.weight[1]), static_cast<std::int64_t>(0.6 * g.n) + 1);
  EXPECT_LE(chk.weight[2], 2 * 30 * 30) << "a 30^3 grid has plane separators of 900";
  EXPECT_EQ(ord::vertex_separator(g, 7, p), ref);
  for (int threads : {2, 4}) {
    std::vector<std::uint8_t> where;
#pragma omp parallel num_threads(threads) default(shared)
#pragma omp single
    where = ord::vertex_separator(g, 7, p);
    EXPECT_EQ(where, ref) << threads << " threads";
  }
  // finalists = 1 keeps METIS' scheme: the best mid-graph run carried down alone.
  p.finalists = 1;
  EXPECT_TRUE(ord::check_separator(g, ord::vertex_separator(g, 7, p)).valid);
}

TEST(FactorOrderingDissection, TopEffortIsThreadIndependentAndOnlyChangesTheTop) {
  const ord::Graph g = graph_from_edges(30 * 30 * 30, grid_edges(30, 30, 30));
  ord::DissectionParams p;
  const auto plain = ord::nested_dissection(g, p, 2);
  p.top_depth = 2;
  p.top_runs = 8;
  p.top_finalists = 8;
  ord::DissectionStats st;
  const auto ref = ord::nested_dissection(g, p, 1, &st);
  ASSERT_TRUE(is_permutation(ref, g.n));
  EXPECT_NE(ref, plain);
  for (int threads : {2, 4}) EXPECT_EQ(ord::nested_dissection(g, p, threads), ref) << threads;
  // Size threshold instead of depth: every subgraph of at least the whole graph = the top one.
  p.top_depth = 0;
  p.top_size = g.total_vwgt;
  const auto by_size = ord::nested_dissection(g, p, 2);
  ASSERT_TRUE(is_permutation(by_size, g.n));
  p.top_depth = 1;
  p.top_size = std::numeric_limits<std::int64_t>::max();
  EXPECT_EQ(ord::nested_dissection(g, p, 3), by_size) << "depth 1 = the root alone";
}

TEST(FactorOrderingOptions, EffortRangesAreEnforced) {
  const auto a = matrix_from_edges<double>(100, grid_edges(10, 10));
  const ShiftedPencil<double> pencil(a, nullptr);
  using O = hs::OrderingOptions;
  auto bad = [&](void (*set)(O&)) {
    O o;
    set(o);
    try {
      static_cast<void>(hs::compute_ordering(pencil, o));
    } catch (const hs::Error& e) {
      return e.code() == hs::Error::Code::InvalidInput;
    }
    return false;
  };
  EXPECT_TRUE(bad([](O& o) { o.separator_runs = 0; }));
  EXPECT_TRUE(bad([](O& o) { o.separator_runs = 65; }));
  EXPECT_TRUE(bad([](O& o) { o.separator_finalists = 6; }));
  EXPECT_TRUE(bad([](O& o) { o.separator_finalists = 0; }));
  EXPECT_TRUE(bad([](O& o) { o.top_depth = -1; }));
  EXPECT_TRUE(bad([](O& o) { o.top_size = -1; }));
  EXPECT_TRUE(bad([](O& o) { o.top_runs = 0; }));
  EXPECT_TRUE(bad([](O& o) { o.top_finalists = 6; }));
  EXPECT_FALSE(bad([](O& o) { o.separator_runs = 64, o.separator_finalists = 64; }));
  EXPECT_FALSE(bad([](O& o) { o.top_depth = 3, o.top_size = 50, o.top_runs = 16, o.top_finalists = 16; }));
}

// ------------------------------------------------------------- FactorOptions and the driver

template <class T>
class FactorOrderingOptionsPencil : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(FactorOrderingOptionsPencil, Scalars);

TYPED_TEST(FactorOrderingOptionsPencil, FactorOptionsCarryTheDissectionOptions) {
  using T = TypeParam;
  const auto a = matrix_from_edges<T>(16 * 16 * 16, grid_edges(16, 16, 16));
  const ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.ordering = hs::kOrderingNestedDissection;
  fo.threads = 2;
  auto analysed_order = [&](const hs::FactorOptions& o) {
    auto f = hs::make_mumps_factorization<T>(o);
    f->analyze(pencil, 1.0, 0.0);
    return f->pivot_order();
  };
  const auto base = analysed_order(fo);
  EXPECT_EQ(analysed_order(fo), base);
  fo.nested_dissection.seed = 77;
  const auto seeded = analysed_order(fo);
  EXPECT_NE(seeded, base) << "the options must reach compute_ordering";
  fo.nested_dissection.threads = 1;  // replaced by ordering_threads / threads: no effect
  EXPECT_EQ(analysed_order(fo), seeded);
}

TEST(DriverOrderingEffort, OptionsAndPricing) {
  const hs::OrderingOptions plain = hs::driver::nd_ordering_options(false);
  EXPECT_EQ(plain.top_depth, 0);
  const hs::OrderingOptions effort = hs::driver::nd_ordering_options(true);
  EXPECT_EQ(effort.top_depth, hs::driver::kNdEffortDepth);
  EXPECT_EQ(effort.top_runs, hs::driver::kNdEffortRuns);
  EXPECT_EQ(effort.top_finalists, hs::driver::kNdEffortRuns);
  // The pricer takes the effort where the numeric work dwarfs the ordering (kNdEffortWorkRatio):
  // a huge 3-D AMD estimate (fill 200, 1e16 flops) yes, a small 2-D one (fill 10) no.
  hs::SolverConfig cfg;
  cfg.target.k = 10;
  const hs::driver::PlannedWork work = hs::driver::plan_work(cfg, hs::ProblemType::Standard, 1000000, 1.0);
  auto nd_candidate = [&](double order, double entries, double amd_fe, double amd_flops) {
    hs::driver::OrderingInput in;
    in.mumps = {order, entries};
    in.nd = in.mumps;
    hs::driver::OrderingPricer pricer{hs::driver::ordering_model(), cfg, work, in, entries, 16, 0.0, {}};
    std::vector<hs::driver::OrderingCandidate> cand(2);
    cand[0].ordering = 0;
    cand[1].ordering = hs::kOrderingNestedDissection;
    hs::FactorEstimate e;
    e.factor_entries = static_cast<std::int64_t>(amd_fe);
    e.flops = amd_flops;
    pricer.analysed(cand, 0, e);
    pricer.predict(cand, 0);
    return cand[1];
  };
  const auto big = nd_candidate(1e6, 1.4e7, 200 * 1.4e7, 1e16);
  EXPECT_TRUE(big.nd_effort);
  const auto small = nd_candidate(1e5, 5e5, 10 * 5e5, 2e9);
  EXPECT_FALSE(small.nd_effort);
}

}  // namespace
