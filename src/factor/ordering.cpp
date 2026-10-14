// hsolve's own fill-reducing ordering: parallel nested dissection (include/hsolve/factor.hpp,
// compute_ordering; building blocks in ordering_detail.hpp).
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/util.hpp"

namespace hs {

namespace ordering {

namespace {

// One subgraph of the dissection: local graph, the original vertex of each local vertex (empty:
// identity), the first position of its range in the elimination order and its seed.
struct Piece {
  Graph g;
  std::vector<std::int32_t> label;
  std::int64_t first = 0;
  std::uint64_t seed = 0;
  int depth = 0;
  std::int64_t path_ns = 0;  ///< work on the ancestors (critical-path accounting)
};

class Dissector {
 public:
  Dissector(const DissectionParams& p, std::int32_t n)
      : p_(p), order_(static_cast<std::size_t>(n), -1) {}

  void run(std::unique_ptr<Piece> piece) noexcept {
    try {
      dissect(std::move(piece));
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex_);
      if (!error_) error_ = std::current_exception();
    }
  }

  std::vector<std::int32_t> take(DissectionStats* stats) {
    if (error_) std::rethrow_exception(error_);
    if (stats) {
      stats->separators = separators_.load();
      stats->leaves = leaves_.load();
      stats->top_separator = top_separator_.load();
      stats->top_separator_s = top_separator_s_;
      stats->work_s = static_cast<double>(work_ns_.load()) * 1e-9;
      stats->critical_path_s = static_cast<double>(critical_ns_.load()) * 1e-9;
      stats->separator.coarsen_s = static_cast<double>(sep_ns_[0].load()) * 1e-9;
      stats->separator.runs_s = static_cast<double>(sep_ns_[1].load()) * 1e-9;
      stats->separator.uncoarsen_s = static_cast<double>(sep_ns_[2].load()) * 1e-9;
      stats->separator.small_s = static_cast<double>(sep_ns_[3].load()) * 1e-9;
      stats->extract_s = static_cast<double>(extract_ns_.load()) * 1e-9;
      stats->leaf_s = static_cast<double>(leaf_ns_.load()) * 1e-9;
      stats->depth_separator.clear();
      for (const auto& d : depth_separator_) stats->depth_separator.push_back(d.load());
      while (!stats->depth_separator.empty() && stats->depth_separator.back() == 0)
        stats->depth_separator.pop_back();
    }
    return std::move(order_);
  }

 private:
  void place(const Piece& pc, std::int32_t local, std::int64_t pos) {
    order_[static_cast<std::size_t>(pos)] = pc.label.empty() ? local : pc.label[local];
  }

  static std::int64_t ns(double seconds) { return static_cast<std::int64_t>(seconds * 1e9); }

  void leaf(const Piece& pc) {
    Timer t;
    ++leaves_;
    if (pc.g.n <= kMaxMinimumDegree && pc.g.edges() > 0) {
      const std::vector<std::int32_t> md = minimum_degree(pc.g);
      for (std::int32_t k = 0; k < pc.g.n; ++k) place(pc, md[k], pc.first + k);
    } else {
      for (std::int32_t k = 0; k < pc.g.n; ++k) place(pc, k, pc.first + k);
    }
    leaf_ns_ += ns(t.seconds());
  }

  // Accounts the work of one piece (started at t, after `path_ns` of work on its ancestors):
  // total work and the longest root-to-leaf chain of work (the critical path). Returns the
  // path including this piece.
  std::int64_t add_work(const Timer& t, std::int64_t path_ns) {
    const auto own = static_cast<std::int64_t>(t.seconds() * 1e9);
    work_ns_ += own;
    const std::int64_t path = path_ns + own;
    std::int64_t seen = critical_ns_.load();
    while (path > seen && !critical_ns_.compare_exchange_weak(seen, path)) {
    }
    return path;
  }

  void dissect(std::unique_ptr<Piece> pc) {
    const Graph& g = pc->g;
    if (g.n == 0) return;
    Timer work;
    // Leaf size in variables (vertex weight): a compressed graph keeps the tree of the original.
    if (g.total_vwgt <= p_.leaf_size || g.edges() == 0) {
      leaf(*pc);
      add_work(work, pc->path_ns);
      return;
    }
    SeparatorTiming timing;
    SeparatorParams sp = p_.separator;
    if (pc->depth < p_.top_depth || g.total_vwgt >= p_.top_size) {
      sp.runs = p_.top_runs;
      sp.finalists = p_.top_finalists;
    }
    const std::vector<std::uint8_t> where = vertex_separator(g, pc->seed, sp, nullptr, &timing);
    sep_ns_[0] += ns(timing.coarsen_s);
    sep_ns_[1] += ns(timing.runs_s);
    sep_ns_[2] += ns(timing.uncoarsen_s);
    sep_ns_[3] += ns(timing.small_s);
    if (pc->depth == 0) top_separator_s_ = work.seconds();
    std::int32_t count[3] = {0, 0, 0};
    for (const std::uint8_t w : where) ++count[w];
    if (count[0] == g.n || count[1] == g.n || count[kSeparator] == g.n) {
      leaf(*pc);  // no progress possible
      add_work(work, pc->path_ns);
      return;
    }
    ++separators_;
    if (pc->depth == 0) top_separator_ = count[kSeparator];
    depth_separator_[std::min(pc->depth, kMaxDepth - 1)] += count[kSeparator];

    // Separator last, in increasing local (hence original) order.
    std::int64_t pos = pc->first + count[0] + count[1];
    for (std::int32_t v = 0; v < g.n; ++v)
      if (where[v] == kSeparator) place(*pc, v, pos++);

    Timer extract;
    std::unique_ptr<Piece> half[2];
    for (int side = 0; side < 2; ++side) {
      auto h = std::make_unique<Piece>();
      std::vector<std::int32_t> to_parent;
      h->g = induced_subgraph(g, where, static_cast<std::uint8_t>(side), to_parent);
      if (pc->label.empty()) {
        h->label = std::move(to_parent);
      } else {
        h->label.resize(to_parent.size());
        for (std::size_t k = 0; k < to_parent.size(); ++k) h->label[k] = pc->label[to_parent[k]];
      }
      h->first = pc->first + (side == 0 ? 0 : count[0]);
      h->seed = child_seed(pc->seed, side);
      h->depth = pc->depth + 1;
      half[side] = std::move(h);
    }
    extract_ns_ += ns(extract.seconds());
    const std::int64_t path = pc->path_ns;
    pc.reset();  // free the parent before descending
    half[0]->path_ns = half[1]->path_ns = add_work(work, path);

    const bool spawn = half[0]->g.n >= p_.task_size;
    Piece* first_half = half[0].release();
#pragma omp task if (spawn) firstprivate(first_half)
    run(std::unique_ptr<Piece>(first_half));
    dissect(std::move(half[1]));
  }

  const DissectionParams& p_;
  std::vector<std::int32_t> order_;
  static constexpr int kMaxDepth = 64;
  std::atomic<std::int64_t> separators_{0}, leaves_{0}, top_separator_{0}, work_ns_{0},
      critical_ns_{0}, extract_ns_{0}, leaf_ns_{0};
  std::atomic<std::int64_t> sep_ns_[4] = {};
  double top_separator_s_ = 0.0;  // written by the task at depth 0 only
  std::atomic<std::int64_t> depth_separator_[kMaxDepth] = {};
  std::mutex error_mutex_;
  std::exception_ptr error_;
};

}  // namespace

std::vector<std::int32_t> nested_dissection(Graph g, const DissectionParams& params, int threads,
                                            DissectionStats* stats) {
  HS_ASSERT(threads >= 1, "nested_dissection: threads = " << threads);
  HS_ASSERT(params.leaf_size >= 1 && params.leaf_size <= kMaxMinimumDegree,
            "nested_dissection: leaf_size = " << params.leaf_size);
  HS_ASSERT(params.separator.imbalance >= 1.0 && params.separator.imbalance < 2.0,
            "nested_dissection: imbalance = " << params.separator.imbalance);
  Dissector d(params, g.n);
  auto root = std::make_unique<Piece>();
  root->g = std::move(g);
  root->seed = params.seed;
  // A region of its own even for one thread: dissect() never waits for the tasks it spawns, the
  // end of this region does. Without it, a call from inside an active parallel region would
  // defer them to the caller's team, and they would outlive `d`.
  Piece* r = root.release();
#pragma omp parallel num_threads(threads) default(none) shared(d) firstprivate(r)
#pragma omp single
  d.run(std::unique_ptr<Piece>(r));
  if (stats) stats->threads = threads;
  return d.take(stats);
}

}  // namespace ordering

namespace {

constexpr std::uint8_t kHasRe = 1;
constexpr std::uint8_t kHasIm = 2;

// For every entry of the pencil's upper pattern: which parts (kHasRe, kHasIm) A or B store a
// nonzero for. Real T: kHasRe everywhere. O(nnz).
template <class T>
std::vector<std::uint8_t> embedding_parts(const ShiftedPencil<T>& pencil) {
  const auto& prow = pencil.rowptr();
  const auto& pcol = pencil.colind();
  const index_t n = pencil.order();
  std::vector<std::uint8_t> flags(static_cast<std::size_t>(pencil.nnz()),
                                  is_complex_v<T> ? 0 : kHasRe);
  if constexpr (is_complex_v<T>) {
    auto mark = [&](const CsrMatrix<T>& m) {
      // The pattern contains upper(m) and both are sorted: one merge walk per row.
      for (index_t i = 0; i < n; ++i) {
        index_t o = prow[i];
        for (index_t q = m.rowptr[i]; q < m.rowptr[i + 1]; ++q) {
          if (m.colind[q] < i) continue;
          while (pcol[o] != m.colind[q]) ++o;
          if (m.values[q].real() != 0.0) flags[o] |= kHasRe;
          if (m.values[q].imag() != 0.0) flags[o] |= kHasIm;
        }
      }
    };
    mark(pencil.a());
    if (pencil.b()) mark(*pencil.b());
  }
  return flags;
}

// Share of the off-diagonal couplings that have an imaginary part (1 without couplings).
template <class T>
double coupling_fraction(const ShiftedPencil<T>& pencil, const std::vector<std::uint8_t>& flags) {
  std::int64_t with_im = 0, coupled = 0;
  for (index_t i = 0; i < pencil.order(); ++i)
    for (index_t p = pencil.rowptr()[i] + 1; p < pencil.rowptr()[i + 1]; ++p) {  // diagonal first
      coupled += flags[p] != 0;
      with_im += (flags[p] & kHasIm) != 0;
    }
  return coupled > 0 ? static_cast<double>(with_im) / static_cast<double>(coupled) : 1.0;
}

// Graph of the real embedding E = [[Re K, -Im K], [Im K, Re K]] (2n vertices) of a complex
// pencil: coupling (i, j) of K contributes (i, j) and (n+i, n+j) when it has a real part, and
// (i, n+j) and (j, n+i) when it has an imaginary part (the pattern MUMPS factorizes).
template <class T>
ordering::Graph embedding_graph(const ShiftedPencil<T>& pencil,
                                const std::vector<std::uint8_t>& flags) {
  const auto& prow = pencil.rowptr();
  const auto& pcol = pencil.colind();
  const auto n = static_cast<std::int32_t>(pencil.order());
  ordering::Graph g;
  g.n = 2 * n;
  g.xadj.assign(static_cast<std::size_t>(g.n) + 1, 0);
  auto for_each_edge = [&](auto&& edge) {
    for (std::int32_t i = 0; i < n; ++i)
      for (index_t p = prow[i]; p < prow[i + 1]; ++p) {
        const std::int32_t j = pcol[p];
        if (j == i) continue;
        if (flags[p] & kHasRe) {
          edge(i, j);
          edge(n + i, n + j);
        }
        if (flags[p] & kHasIm) {
          edge(i, n + j);
          edge(j, n + i);
        }
      }
  };
  for_each_edge([&](std::int32_t a, std::int32_t b) {
    ++g.xadj[a + 1];
    ++g.xadj[b + 1];
  });
  for (std::int32_t v = 0; v < g.n; ++v) g.xadj[v + 1] += g.xadj[v];
  g.adj.resize(static_cast<std::size_t>(g.xadj[g.n]));
  std::vector<std::int64_t> fill(g.xadj.begin(), g.xadj.end() - 1);
  for_each_edge([&](std::int32_t a, std::int32_t b) {
    g.adj[static_cast<std::size_t>(fill[a]++)] = b;
    g.adj[static_cast<std::size_t>(fill[b]++)] = a;
  });
  for (std::int32_t v = 0; v < g.n; ++v)
    std::sort(g.adj.begin() + g.xadj[v], g.adj.begin() + g.xadj[v + 1]);
  g.ewgt.assign(g.adj.size(), 1);
  g.vwgt.assign(static_cast<std::size_t>(g.n), 1);
  g.total_vwgt = g.n;
  return g;
}

}  // namespace

template <class T>
double ordering::complex_coupling_fraction(const ShiftedPencil<T>& pencil) {
  if constexpr (!is_complex_v<T>) {
    return 0.0;
  } else {
    return coupling_fraction(pencil, embedding_parts(pencil));
  }
}

template <class T>
std::vector<int> compute_ordering(const ShiftedPencil<T>& pencil, const OrderingOptions& o,
                                  OrderingStats* stats) {
  if (o.threads < 0)
    HS_THROW(InvalidInput, "OrderingOptions.threads must be >= 0, got " << o.threads);
  if (o.leaf_size < 1 || o.leaf_size > ordering::kMaxMinimumDegree)
    HS_THROW(InvalidInput, "OrderingOptions.leaf_size must lie in [1, "
                               << ordering::kMaxMinimumDegree << "], got " << o.leaf_size);
  if (!(o.imbalance >= 1.0 && o.imbalance <= ordering::kMaxImbalance))
    HS_THROW(InvalidInput, "OrderingOptions.imbalance must lie in [1, "
                               << ordering::kMaxImbalance << "], got " << o.imbalance);
  if (o.coarsen_to < 2 || o.coarsen_to > INT_MAX)
    HS_THROW(InvalidInput, "OrderingOptions.coarsen_to must be >= 2, got " << o.coarsen_to);
  if (o.separator_runs < 1 || o.separator_runs > 64 || o.separator_finalists < 1 ||
      o.separator_finalists > o.separator_runs)
    HS_THROW(InvalidInput, "OrderingOptions: need 1 <= separator_finalists <= separator_runs <= 64, got "
                               << o.separator_finalists << ", " << o.separator_runs);
  if (o.top_depth < 0 || o.top_size < 0 || o.top_runs < 1 || o.top_runs > 64 || o.top_finalists < 1 ||
      o.top_finalists > o.top_runs)
    HS_THROW(InvalidInput, "OrderingOptions: need top_depth, top_size >= 0 and 1 <= top_finalists <= top_runs <= 64, got "
                               << o.top_depth << ", " << o.top_finalists << ", " << o.top_runs);
  if (o.flow_depth < 0 || o.flow_depth > 16)
    HS_THROW(InvalidInput, "OrderingOptions.flow_depth must lie in [0, 16], got " << o.flow_depth);
  if (o.init_tries < 1 || o.refine_passes < 0)
    HS_THROW(InvalidInput, "OrderingOptions: init_tries must be >= 1 and refine_passes >= 0, got "
                               << o.init_tries << ", " << o.refine_passes);
  constexpr int kFactor = is_complex_v<T> ? 2 : 1;
  const index_t n = pencil.order();
  if (n * kFactor > INT_MAX)
    HS_THROW(Unsupported,
             "compute_ordering: MUMPS order " << n * kFactor << " exceeds the 32-bit range");
  Timer timer;
  bool compressed = kFactor == 2;
  double complex_fraction = 0.0;
  ordering::Graph g;
  if constexpr (kFactor == 2) {
    const std::vector<std::uint8_t> flags = embedding_parts(pencil);
    complex_fraction = coupling_fraction(pencil, flags);
    compressed = o.embedding == OrderingOptions::Embedding::Compressed ||
                 (o.embedding == OrderingOptions::Embedding::Auto &&
                  complex_fraction >= ordering::kCompressedFraction);
    if (!compressed) g = embedding_graph(pencil, flags);
  }
  if (kFactor == 1 || compressed)
    g = ordering::graph_from_upper(static_cast<std::int32_t>(n), pencil.rowptr().data(),
                                   pencil.colind().data());
  const std::int64_t edges = g.edges() / 2;
  const index_t vertices = g.n;
  const int threads = o.threads > 0 ? o.threads : max_threads();
  // Indistinguishable vertices (identical closed neighbourhoods) as one weighted vertex.
  ordering::Graph cg;
  std::vector<std::int32_t> class_first, class_members;
  const bool graph_compressed =
      o.compress &&
      ordering::compress_graph(g, ordering::kCompressionRatio, threads, cg, class_first, class_members);
  if (graph_compressed) g = std::move(cg);
  const index_t ordered_vertices = g.n;
  ordering::DissectionParams p;
  p.separator.imbalance = o.imbalance;
  p.separator.coarsen_to = static_cast<std::int32_t>(o.coarsen_to);
  p.separator.init_tries = o.init_tries;
  p.separator.refine_passes = o.refine_passes;
  p.separator.runs = o.separator_runs;
  p.separator.finalists = o.separator_finalists;
  p.separator.flow_depth = o.flow_depth;
  p.top_depth = o.top_depth;
  p.top_size = o.top_size > 0 ? o.top_size : std::numeric_limits<std::int64_t>::max();
  p.top_runs = o.top_runs;
  p.top_finalists = o.top_finalists;
  p.leaf_size = static_cast<std::int32_t>(o.leaf_size);
  p.seed = o.seed;
  ordering::DissectionStats ds;
  std::vector<std::int32_t> order = ordering::nested_dissection(std::move(g), p, threads, &ds);
  if (graph_compressed) {
    // Each class eliminated as a block, its members in increasing order.
    std::vector<std::int32_t> expanded;
    expanded.reserve(static_cast<std::size_t>(vertices));
    for (const std::int32_t q : order)
      for (std::int32_t m = class_first[q]; m < class_first[q + 1]; ++m) expanded.push_back(class_members[m]);
    order = std::move(expanded);
  }

  std::vector<int> perm(static_cast<std::size_t>(n * kFactor), 0);
  for (index_t k = 0; k < vertices; ++k) {
    const std::int32_t v = order[static_cast<std::size_t>(k)];
    HS_ASSERT(v >= 0 && v < vertices && perm[v] == 0,
              "compute_ordering: not a permutation at " << k);
    if (compressed) {
      perm[v] = static_cast<int>(2 * k + 1);      // Re part of vertex v
      perm[n + v] = static_cast<int>(2 * k + 2);  // Im part, eliminated right after
    } else {
      perm[v] = static_cast<int>(k + 1);
    }
  }
  if (stats) {
    stats->seconds = timer.seconds();
    stats->threads = threads;
    stats->vertices = vertices;
    stats->complex_fraction = complex_fraction;
    stats->compressed = compressed;
    stats->edges = edges;
    stats->ordered_vertices = ordered_vertices;
    stats->separators = ds.separators;
    stats->leaves = ds.leaves;
    stats->top_separator = ds.top_separator;
  }
  return perm;
}

#define HS_INSTANTIATE_ORDERING(T)                                                               \
  template std::vector<int> compute_ordering<T>(const ShiftedPencil<T>&, const OrderingOptions&, \
                                                OrderingStats*);                                 \
  template double ordering::complex_coupling_fraction<T>(const ShiftedPencil<T>&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_ORDERING)
#undef HS_INSTANTIATE_ORDERING

}  // namespace hs
