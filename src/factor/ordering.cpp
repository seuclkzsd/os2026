// hsolve's own fill-reducing ordering: parallel nested dissection (include/hsolve/factor.hpp,
// compute_ordering; building blocks in ordering_detail.hpp).
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <exception>
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

// Largest leaf the bitset minimum degree takes (O(n^2/64) memory and up to O(n^3/64) time); a
// subgraph that cannot be split (no separator leaves both halves smaller, e.g. a clique) and is
// larger keeps its numbering: it is dense, every order fills it completely.
constexpr std::int32_t kMaxMinimumDegree = 2048;

// One subgraph of the dissection: local graph, the original vertex of each local vertex (empty:
// identity), the first position of its range in the elimination order and its seed.
struct Piece {
  Graph g;
  std::vector<std::int32_t> label;
  std::int64_t first = 0;
  std::uint64_t seed = 0;
  int depth = 0;
};

class Dissector {
 public:
  Dissector(const DissectionParams& p, std::int32_t n) : p_(p), order_(static_cast<std::size_t>(n), -1) {}

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
      stats->depth_separator.clear();
      for (const auto& d : depth_separator_) stats->depth_separator.push_back(d.load());
      while (!stats->depth_separator.empty() && stats->depth_separator.back() == 0) stats->depth_separator.pop_back();
    }
    return std::move(order_);
  }

 private:
  void place(const Piece& pc, std::int32_t local, std::int64_t pos) {
    order_[static_cast<std::size_t>(pos)] = pc.label.empty() ? local : pc.label[local];
  }

  void leaf(const Piece& pc) {
    ++leaves_;
    if (pc.g.n <= kMaxMinimumDegree && pc.g.edges() > 0) {
      const std::vector<std::int32_t> md = minimum_degree(pc.g);
      for (std::int32_t k = 0; k < pc.g.n; ++k) place(pc, md[k], pc.first + k);
    } else {
      for (std::int32_t k = 0; k < pc.g.n; ++k) place(pc, k, pc.first + k);
    }
  }

  void dissect(std::unique_ptr<Piece> pc) {
    const Graph& g = pc->g;
    if (g.n == 0) return;
    if (g.n <= p_.leaf_size || g.edges() == 0) {
      leaf(*pc);
      return;
    }
    Timer separator_timer;
    const std::vector<std::uint8_t> where = vertex_separator(g, pc->seed, p_.separator);
    if (pc->depth == 0) top_separator_s_ = separator_timer.seconds();
    std::int32_t count[3] = {0, 0, 0};
    for (const std::uint8_t w : where) ++count[w];
    if (count[0] == g.n || count[1] == g.n || count[kSeparator] == g.n) {
      leaf(*pc);  // no progress possible
      return;
    }
    ++separators_;
    if (pc->depth == 0) top_separator_ = count[kSeparator];
    depth_separator_[std::min(pc->depth, kMaxDepth - 1)] += count[kSeparator];

    // Separator last, in increasing local (hence original) order.
    std::int64_t pos = pc->first + count[0] + count[1];
    for (std::int32_t v = 0; v < g.n; ++v)
      if (where[v] == kSeparator) place(*pc, v, pos++);

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
    pc.reset();  // free the parent before descending

    const bool spawn = half[0]->g.n >= p_.task_size;
    Piece* first_half = half[0].release();
#pragma omp task if (spawn) firstprivate(first_half)
    run(std::unique_ptr<Piece>(first_half));
    dissect(std::move(half[1]));
  }

  const DissectionParams& p_;
  std::vector<std::int32_t> order_;
  static constexpr int kMaxDepth = 64;
  std::atomic<std::int64_t> separators_{0}, leaves_{0}, top_separator_{0};
  double top_separator_s_ = 0.0;  // written by the task at depth 0 only
  std::atomic<std::int64_t> depth_separator_[kMaxDepth] = {};
  std::mutex error_mutex_;
  std::exception_ptr error_;
};

}  // namespace

std::vector<std::int32_t> nested_dissection(Graph g, const DissectionParams& params, int threads,
                                            DissectionStats* stats) {
  HS_ASSERT(threads >= 1, "nested_dissection: threads = " << threads);
  Dissector d(params, g.n);
  auto root = std::make_unique<Piece>();
  root->g = std::move(g);
  root->seed = params.seed;
  if (threads == 1) {
    d.run(std::move(root));
  } else {
    Piece* r = root.release();
#pragma omp parallel num_threads(threads) default(none) shared(d) firstprivate(r)
#pragma omp single
    d.run(std::unique_ptr<Piece>(r));
  }
  if (stats) stats->threads = threads;
  return d.take(stats);
}

}  // namespace ordering

template <class T>
std::vector<int> compute_ordering(const ShiftedPencil<T>& pencil, const OrderingOptions& o,
                                  OrderingStats* stats) {
  if (o.threads < 0) HS_THROW(InvalidInput, "OrderingOptions.threads must be >= 0, got " << o.threads);
  if (o.leaf_size < 1 || o.leaf_size > INT_MAX)
    HS_THROW(InvalidInput, "OrderingOptions.leaf_size must be >= 1, got " << o.leaf_size);
  if (!(o.imbalance >= 1.0) || !std::isfinite(o.imbalance))
    HS_THROW(InvalidInput, "OrderingOptions.imbalance must be finite and >= 1, got " << o.imbalance);
  if (o.coarsen_to < 2 || o.coarsen_to > INT_MAX)
    HS_THROW(InvalidInput, "OrderingOptions.coarsen_to must be >= 2, got " << o.coarsen_to);
  if (o.init_tries < 1 || o.refine_passes < 0)
    HS_THROW(InvalidInput, "OrderingOptions: init_tries must be >= 1 and refine_passes >= 0, got "
                               << o.init_tries << ", " << o.refine_passes);
  constexpr int kFactor = is_complex_v<T> ? 2 : 1;
  const index_t n = pencil.order();
  if (n * kFactor > INT_MAX)
    HS_THROW(Unsupported, "compute_ordering: MUMPS order " << n * kFactor << " exceeds the 32-bit range");
  Timer timer;
  ordering::Graph g = ordering::graph_from_upper(static_cast<std::int32_t>(n), pencil.rowptr().data(),
                                                 pencil.colind().data());
  const std::int64_t edges = g.edges() / 2;
  ordering::DissectionParams p;
  p.separator.imbalance = o.imbalance;
  p.separator.coarsen_to = static_cast<std::int32_t>(o.coarsen_to);
  p.separator.init_tries = o.init_tries;
  p.separator.refine_passes = o.refine_passes;
  p.leaf_size = static_cast<std::int32_t>(o.leaf_size);
  p.seed = o.seed;
  const int threads = o.threads > 0 ? o.threads : max_threads();
  ordering::DissectionStats ds;
  const std::vector<std::int32_t> order = ordering::nested_dissection(std::move(g), p, threads, &ds);

  std::vector<int> perm(static_cast<std::size_t>(n * kFactor), 0);
  for (index_t k = 0; k < n; ++k) {
    const std::int32_t v = order[static_cast<std::size_t>(k)];
    HS_ASSERT(v >= 0 && v < n && perm[v] == 0, "compute_ordering: not a permutation at " << k);
    if constexpr (kFactor == 1) {
      perm[v] = static_cast<int>(k + 1);
    } else {
      perm[v] = static_cast<int>(2 * k + 1);      // Re part of vertex v
      perm[n + v] = static_cast<int>(2 * k + 2);  // Im part, eliminated right after
    }
  }
  if (stats) {
    stats->seconds = timer.seconds();
    stats->threads = threads;
    stats->vertices = n;
    stats->edges = edges;
    stats->separators = ds.separators;
    stats->leaves = ds.leaves;
    stats->top_separator = ds.top_separator;
  }
  return perm;
}

#define HS_INSTANTIATE_ORDERING(T)                                                             \
  template std::vector<int> compute_ordering<T>(const ShiftedPencil<T>&, const OrderingOptions&, \
                                                OrderingStats*);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_ORDERING)
#undef HS_INSTANTIATE_ORDERING

}  // namespace hs
