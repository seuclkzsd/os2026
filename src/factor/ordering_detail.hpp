// Building blocks of hsolve's own fill-reducing ordering (module-internal to src/factor; the
// public entry point is hs::compute_ordering in include/hsolve/factor.hpp).
//
// Nested dissection in the style of METIS_NodeND (multilevel vertex separators: sorted
// heavy-edge matching, greedy graph growing on the coarsest graph, one-sided node FM refinement
// during uncoarsening, best of 5 bisections of a 4x-coarsened graph for large graphs; small
// subgraphs by minimum degree), with two differences that make it parallel and reproducible:
//   * the two halves of every separator, and the 5 trial bisections of a large graph, are
//     independent OpenMP tasks, each writing its own, precomputed part of the result;
//   * every random choice draws from a counter-based generator seeded by the subgraph's position
//     in the dissection tree (and the trial's number), never from a shared state (METIS 5.1
//     uses libc rand(): concurrent calls would interleave their draws). The permutation is
//     therefore a function of the input graph and the parameters alone - identical for every
//     thread count.
// Thread safety: no shared state. vertex_separator creates OpenMP tasks (run inline outside a
// parallel region); everything else is serial except nested_dissection. Vertex counts are 32-bit.
//
// Measured (bench/analysis_orderings.cpp, 2 threads, separator found / fill vs METIS inside
// MUMPS): the plane separator 46^2 = 2116 of the 27-point 46^3 grid in 4 of 5 seeds (the two-sided
// FM of the first version ended 30-45% above it, flops 1.6x METIS: one-sided passes and random
// tie-breaking in the gain queues fixed it).
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/sparse.hpp"

namespace hs::ordering {

/// Share of the off-diagonal couplings (entries of the pencil's upper pattern) whose imaginary
/// part is nonzero in A or B; 1 for a complex pencil without off-diagonal couplings, 0 for real
/// T. Decides between the compressed and the full embedding (OrderingOptions::embedding, and
/// FactorOptions::block_analysis = -1): compressed from kCompressedFraction on. O(nnz).
template <class T>
double complex_coupling_fraction(const ShiftedPencil<T>& pencil);
inline constexpr double kCompressedFraction = 0.5;

/// Undirected graph in CSR form: symmetric adjacency, no self-loops, positive integer vertex
/// and edge weights (edge weights stored per adjacency entry, equal in both directions). Rows
/// are sorted ascending in graphs from graph_from_upper and induced_subgraph (coarse graphs
/// inside vertex_separator: any order).
struct Graph {
  std::int32_t n = 0;
  std::vector<std::int64_t> xadj;  ///< n + 1 row pointers
  std::vector<std::int32_t> adj;   ///< neighbours
  std::vector<std::int32_t> ewgt;  ///< edge weights, adj.size()
  std::vector<std::int32_t> vwgt;  ///< vertex weights, n
  std::int64_t total_vwgt = 0;     ///< sum of vwgt

  std::int64_t edges() const noexcept { return static_cast<std::int64_t>(adj.size()); }
  std::int32_t degree(std::int32_t v) const noexcept {
    return static_cast<std::int32_t>(xadj[v + 1] - xadj[v]);
  }
};

/// The graph of a symmetric pattern given by its upper triangle (row i lists columns j >= i,
/// ascending; the diagonal, if present, is ignored) with unit weights. O(nnz), serial.
Graph graph_from_upper(std::int32_t n, const std::int64_t* rowptr, const std::int32_t* colind);

/// Counter-based random numbers (SplitMix64): value k of stream `seed` is a pure function of
/// (seed, k), so results never depend on which thread draws them.
class Random {
 public:
  explicit Random(std::uint64_t seed) noexcept : state_(seed) {}
  std::uint64_t next() noexcept;
  /// Uniform in [0, bound) for bound > 0 (multiply-shift, negligible bias for 32-bit bounds).
  std::uint32_t below(std::uint32_t bound) noexcept;

 private:
  std::uint64_t state_;
};

/// Seed of a child subgraph (side 0 or 1) of the subgraph seeded with `parent`.
std::uint64_t child_seed(std::uint64_t parent, int side) noexcept;

/// Part labels of a vertex bisection: 0 and 1 are the halves, kSeparator the separator (no
/// edge joins part 0 and part 1).
inline constexpr std::uint8_t kSeparator = 2;

/// vertex_separator's two-stage scheme (METIS MlevelNodeBisectionL2) for graphs of at least
/// kTwoStageSize vertices: kTwoStageLevels contractions to a "mid" graph, the best of
/// kTwoStageRuns complete bisections of it, refinement back up. The runs are concurrent OpenMP
/// tasks (each with its own seed) when the mid graph has at least kParallelRunSize vertices
/// (SeparatorTrace::vertices[kTwoStageLevels] shows its size).
inline constexpr std::int32_t kTwoStageSize = 5000;
inline constexpr int kTwoStageLevels = 4;
inline constexpr int kTwoStageRuns = 5;
inline constexpr std::int32_t kParallelRunSize = 1000;

/// Largest subgraph the dissection orders by minimum degree (bitset elimination graph: O(n^2/64)
/// memory, up to O(n^3/64) time), hence the largest DissectionParams::leaf_size (and
/// OrderingOptions::leaf_size). A larger subgraph that cannot be split (no separator leaves both
/// halves smaller, e.g. a clique) keeps its numbering: it is dense, every order fills it.
inline constexpr std::int32_t kMaxMinimumDegree = 2048;

/// Largest OrderingOptions::imbalance compute_ordering accepts (nested_dissection itself takes
/// [1, 2), for experiments). The bigger half then holds at most 3/4 of a subgraph, so the
/// dissection is at most log_{4/3}(n) deep. Measured (bench/analysis_fix_imbalance.cpp: MUMPS
/// flops of the order on 60^2, 300^2 and 20^3 grids, relative to 1.2): 1.3-1.5 0.67-1.0x;
/// 1.7-1.8 0.75-1.7x; 1.9 1.6-5x; 1.999 170-250x (the halves peel a few vertices off a corner,
/// thousands of separators deep); from 2 on a half may hold the whole graph: an empty separator,
/// i.e. the natural order.
inline constexpr double kMaxImbalance = 1.5;

struct SeparatorParams {
  double imbalance = 1.2;  ///< each half weighs at most imbalance * total / 2 (METIS ND: 1.2);
                           ///< vertex_separator takes any value >= 1 (>= 2: no bound at all)
  std::int32_t coarsen_to = 100;  ///< stop coarsening at this many vertices
  int init_tries = 7;             ///< greedy-growing attempts on the coarsest graph
  int refine_passes = 10;         ///< node-FM passes per level (stop early without gain)
  int refine_mode = 1;            ///< 0 two-sided FM, 1 one-sided (METIS ND default), 2 both
};

/// What vertex_separator did, level by level (index 0 = the input graph), for benches and tests.
struct SeparatorTrace {
  std::vector<std::int32_t> vertices;   ///< vertices of each level
  std::vector<std::int64_t> edges;      ///< adjacency entries of each level
  std::vector<std::int64_t> projected;  ///< separator weight projected onto each level
  std::vector<std::int64_t> refined;    ///< ... and after refinement there
  double coarsen_s = 0.0;               ///< time in matching + contraction
  double init_s = 0.0;                  ///< time in the initial separators
  double refine_s = 0.0;                ///< time in projection + FM refinement
};

/// Multilevel vertex separator of g (g.n >= 1). Returns the part label of every vertex. The
/// separator is minimal in the FM sense (no single move reduces its weight within the balance
/// constraint) but not optimal. Deterministic in (g, seed, params). O(edges * levels).
std::vector<std::uint8_t> vertex_separator(const Graph& g, std::uint64_t seed,
                                           const SeparatorParams& params,
                                           SeparatorTrace* trace = nullptr);

/// Weight of the separator and of the two halves of `where`, and whether it is a valid vertex
/// separator of g (no edge between parts 0 and 1). For tests and assertions. O(edges).
struct SeparatorCheck {
  std::int64_t weight[3] = {0, 0, 0};
  bool valid = false;
};
SeparatorCheck check_separator(const Graph& g, const std::vector<std::uint8_t>& where);

/// Subgraph induced by the vertices with where[v] == part, renumbered in increasing order of
/// the parent's numbering; `to_parent[k]` receives the parent vertex of local vertex k.
Graph induced_subgraph(const Graph& g, const std::vector<std::uint8_t>& where, std::uint8_t part,
                       std::vector<std::int32_t>& to_parent);

/// Minimum-degree elimination order of g (exact external degrees on an explicit elimination
/// graph kept as bitsets: O(n^2 / 64) memory, meant for the small leaves of the dissection).
/// order[k] = vertex eliminated k-th; ties go to the lowest vertex number.
std::vector<std::int32_t> minimum_degree(const Graph& g);

struct DissectionParams {
  SeparatorParams separator;     ///< imbalance in [1, 2) (from 2 on a half may take the whole
                                 ///< subgraph); compute_ordering admits [1, kMaxImbalance]
  std::int32_t leaf_size = 120;  ///< subgraphs with at most this many vertices: minimum degree
                                 ///< (1..kMaxMinimumDegree)
  std::int32_t task_size = 512;  ///< subgraphs below this size are ordered inline, not as tasks
                                 ///< (syn-mem27-46, 2 threads: 0.76 s at 150..500, 0.79 at 1000,
                                 ///< 0.84 at 2000, 1.11 at 4096, 1.30 at 20000; serial 1.40 s)
  std::uint64_t seed = 0x6873'6f6c'7665'2026ULL;
};

struct DissectionStats {
  std::int64_t separators = 0;                ///< nonleaf subgraphs (separators computed)
  std::int64_t leaves = 0;                    ///< subgraphs ordered by minimum degree
  std::int64_t top_separator = 0;             ///< vertices in the first (largest) separator
  std::vector<std::int64_t> depth_separator;  ///< separator vertices summed per dissection depth
  double top_separator_s = 0.0;               ///< time of the first separator (the serial part)
  double work_s = 0.0;           ///< time in separators, subgraph extraction and leaves, summed
                                 ///< over the pieces
  double critical_path_s = 0.0;  ///< longest root-to-leaf chain of that work, i.e. the wall
                                 ///< time with unlimited threads (Brent: T_p <= work/p +
                                 ///< critical). Exact with 1 thread; with more, a piece's
                                 ///< time includes waiting for its stolen trial bisections
  int threads = 1;               ///< OpenMP threads that ran the dissection
};

/// Nested-dissection elimination order of g with `threads` OpenMP threads (>= 1): order[k] =
/// vertex eliminated k-th. The result does not depend on `threads`. Always opens a parallel
/// region of its own (also for one thread), whose end waits for every dissection task: safe to
/// call from inside a parallel region (without nested parallelism it then runs on one thread).
/// Precondition (HS_ASSERT): params within the documented ranges.
std::vector<std::int32_t> nested_dissection(Graph g, const DissectionParams& params, int threads,
                                            DissectionStats* stats = nullptr);

}  // namespace hs::ordering
