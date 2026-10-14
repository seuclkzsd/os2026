// Multilevel vertex separators (ordering_detail.hpp): sorted heavy-edge matching, greedy graph
// growing with boundary separators on the coarsest graph, node FM refinement during
// uncoarsening, and for large graphs the best of several complete bisections of a once-coarsened
// graph - the scheme of METIS_NodeND (Karypis & Kumar, SIAM J. Sci. Comput. 20(1), 1998) - with
// deterministic random numbers drawn per call (never from a shared state).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/util.hpp"

namespace hs::ordering {

namespace {

// The two-stage scheme for large graphs: kTwoStageSize, kTwoStageLevels, kTwoStageRuns and
// kParallelRunSize in ordering_detail.hpp.

// Indexed binary max-heap of vertices keyed by gain; ties go to the lower rank, then the lower
// vertex number, so the order of extraction is a function of keys and ranks alone.
class GainHeap {
 public:
  void reset(std::int32_t n) {
    pos_.assign(static_cast<std::size_t>(n), -1);
    heap_.clear();
  }
  bool empty() const noexcept { return heap_.empty(); }
  bool contains(std::int32_t v) const noexcept { return pos_[v] >= 0; }
  std::int32_t top() const noexcept { return heap_.front().v; }
  std::int64_t top_key() const noexcept { return heap_.front().key; }

  void insert(std::int32_t v, std::int64_t key, std::uint32_t rank) {
    heap_.push_back({key, rank, v});
    const std::int32_t i = static_cast<std::int32_t>(heap_.size() - 1);
    pos_[v] = i;
    up(i);
  }
  void update(std::int32_t v, std::int64_t key) {
    const std::int32_t i = pos_[v];
    const std::int64_t old = heap_[i].key;
    heap_[i].key = key;
    if (key > old)
      up(i);
    else
      down(i);
  }
  void remove(std::int32_t v) {
    const std::int32_t i = pos_[v];
    if (i < 0) return;
    const std::int32_t last = static_cast<std::int32_t>(heap_.size() - 1);
    pos_[v] = -1;
    if (i == last) {
      heap_.pop_back();
      return;
    }
    const Item moved = heap_[last];
    heap_.pop_back();
    place(i, moved);
    if (i > 0 && above(moved, heap_[(i - 1) / 2]))
      up(i);
    else
      down(i);
  }
  void clear() {
    for (const Item& it : heap_) pos_[it.v] = -1;
    heap_.clear();
  }

 private:
  struct Item {
    std::int64_t key;
    std::uint32_t rank;
    std::int32_t v;
  };
  static bool above(const Item& a, const Item& b) noexcept {
    if (a.key != b.key) return a.key > b.key;
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.v < b.v;
  }
  void place(std::int32_t i, const Item& it) {
    heap_[i] = it;
    pos_[it.v] = i;
  }
  void up(std::int32_t i) {
    const Item it = heap_[i];
    while (i > 0) {
      const std::int32_t parent = (i - 1) / 2;
      if (!above(it, heap_[parent])) break;
      place(i, heap_[parent]);
      i = parent;
    }
    place(i, it);
  }
  void down(std::int32_t i) {
    const std::int32_t n = static_cast<std::int32_t>(heap_.size());
    const Item it = heap_[i];
    for (;;) {
      std::int32_t c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && above(heap_[c + 1], heap_[c])) ++c;
      if (!above(heap_[c], it)) break;
      place(i, heap_[c]);
      i = c;
    }
    place(i, it);
  }
  std::vector<Item> heap_;
  std::vector<std::int32_t> pos_;
};

// ------------------------------------------------------------------------------- coarsening

// Sorted heavy-edge matching (vertices visited by increasing degree, random among equal
// degrees; each unmatched vertex takes its heaviest unmatched neighbour, scanning its row from a
// random rotation so that unit weights do not favour low numbers) and contraction. Returns the
// coarse graph, numbered in the order of the fine graph; cmap[v] is the coarse vertex of v.
Graph coarsen_once(const Graph& g, Random& rng, std::int64_t max_vwgt,
                   std::vector<std::int32_t>& cmap) {
  const std::int32_t n = g.n;
  // Visit order as in METIS' Match_SHEM: a partial shuffle (n/8 swaps of 4-element groups: the
  // order stays mostly local, which keeps the row accesses cache-friendly), then a stable bucket
  // sort by degree capped at 0.7x the average degree (low-degree vertices first, the bulk of a
  // mesh keeps the shuffled order).
  std::vector<std::int32_t> visit(static_cast<std::size_t>(n));
  {
    std::vector<std::int32_t> perm(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) perm[i] = i;
    if (n < 10) {
      for (std::int32_t i = n - 1; i > 0; --i)
        std::swap(perm[i], perm[rng.below(static_cast<std::uint32_t>(i + 1))]);
    } else {
      const auto range = static_cast<std::uint32_t>(n - 3);
      for (std::int32_t s = 0; s < n / 8; ++s) {
        const std::uint32_t a = rng.below(range), b = rng.below(range);
        std::swap(perm[a + 0], perm[b + 2]);
        std::swap(perm[a + 1], perm[b + 3]);
        std::swap(perm[a + 2], perm[b + 0]);
        std::swap(perm[a + 3], perm[b + 1]);
      }
    }
    const std::int32_t cap =
        std::max<std::int32_t>(1, static_cast<std::int32_t>(0.7 * static_cast<double>(g.edges()) /
                                                            std::max<std::int32_t>(1, n)));
    std::vector<std::int32_t> bucket(static_cast<std::size_t>(cap) + 2, 0);
    for (std::int32_t v = 0; v < n; ++v) ++bucket[std::min(g.degree(v), cap) + 1];
    for (std::int32_t d = 0; d <= cap; ++d) bucket[d + 1] += bucket[d];
    for (const std::int32_t v : perm) visit[bucket[std::min(g.degree(v), cap)]++] = v;
  }

  std::vector<std::int32_t> match(static_cast<std::size_t>(n), -1);
  for (const std::int32_t v : visit) {
    if (match[v] >= 0) continue;
    std::int32_t best = -1;
    std::int32_t best_w = 0;
    const std::int64_t b = g.xadj[v], deg = g.xadj[v + 1] - b;
    if (deg > 0 && g.vwgt[v] < max_vwgt) {
      const std::int64_t rot = rng.below(static_cast<std::uint32_t>(deg));
      const std::int64_t room = max_vwgt - g.vwgt[v];
      for (std::int64_t k = 0; k < deg; ++k) {
        std::int64_t e = b + rot + k;
        if (e >= b + deg) e -= deg;
        const std::int32_t u = g.adj[e];
        if (match[u] >= 0 || g.vwgt[u] > room || g.ewgt[e] <= best_w) continue;
        best = u;
        best_w = g.ewgt[e];
      }
    }
    if (best < 0) {
      match[v] = v;
    } else {
      match[v] = best;
      match[best] = v;
    }
  }
  visit = {};

  cmap.assign(static_cast<std::size_t>(n), -1);
  std::int32_t nc = 0;
  for (std::int32_t v = 0; v < n; ++v) {
    if (cmap[v] >= 0) continue;
    cmap[v] = nc;
    cmap[match[v]] = nc;
    ++nc;
  }

  Graph c;
  c.n = nc;
  c.xadj.assign(static_cast<std::size_t>(nc) + 1, 0);
  c.vwgt.resize(static_cast<std::size_t>(nc));
  c.total_vwgt = g.total_vwgt;
  // Coarse rows are appended one after the other (capacity reserved for the fine graph, an upper
  // bound that is never touched beyond what is written); slot[cu] is the position of coarse
  // neighbour cu if it lies in the current row.
  c.adj.reserve(g.adj.size());
  c.ewgt.reserve(g.adj.size());
  std::vector<std::int64_t> slot(static_cast<std::size_t>(nc), -1);
  std::int64_t q = 0;
  std::int32_t cv = 0;
  std::int64_t row = 0;
  auto merge_row = [&](std::int32_t x) {
    for (std::int64_t e = g.xadj[x]; e < g.xadj[x + 1]; ++e) {
      const std::int32_t cu = cmap[g.adj[e]];
      if (cu == cv) continue;
      const std::int64_t s = slot[cu];
      if (s < row) {
        slot[cu] = q++;
        c.adj.push_back(cu);
        c.ewgt.push_back(g.ewgt[e]);
      } else {
        c.ewgt[static_cast<std::size_t>(s)] += g.ewgt[e];
      }
    }
  };
  for (std::int32_t v = 0; v < n; ++v) {
    const std::int32_t partner = match[v];
    if (partner < v) continue;  // the pair was handled at its lower vertex
    row = q;
    merge_row(v);
    std::int64_t w = g.vwgt[v];
    if (partner != v) {
      merge_row(partner);
      w += g.vwgt[partner];
    }
    c.vwgt[cv] = static_cast<std::int32_t>(w);
    c.xadj[++cv] = q;
  }
  HS_ASSERT(cv == nc, "coarsen_once: " << cv << " coarse rows for " << nc << " coarse vertices");
  return c;
}

// ------------------------------------------------------------------------------ refinement

std::uint32_t mix_rank(std::uint64_t salt, std::int32_t v) noexcept {
  std::uint64_t z = salt + 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(v) + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return static_cast<std::uint32_t>((z ^ (z >> 31)) >> 32);
}

// Node FM refinement of a separator (METIS FM_2WayNodeRefine2Sided / 1Sided): repeatedly move
// the separator vertex with the best gain into a half, pulling its neighbours in the other half
// into the separator; hill-climb past local minima (a bounded number of moves without
// improvement), then roll back to the best separator seen. A vertex leaves the separator at most
// once per pass. Ties between equal gains go to a per-vertex random rank (lowest vertex first
// would drag separators of structured meshes in one direction).
// The workspace serves every graph of up to max_n vertices; per refinement it costs one pass over
// the vertices plus work proportional to the separator and the moves.
class NodeFm {
 public:
  explicit NodeFm(std::int32_t max_n) {
    const std::size_t n = static_cast<std::size_t>(max_n);
    ed_[0].reset(new std::int64_t[n]);  // written before read: no initialization
    ed_[1].reset(new std::int64_t[n]);
    locked_.assign(n, 0);
    listed_.assign(n, 0);
    heap_[0].reset(max_n);
    heap_[1].reset(max_n);
  }

  // Two-sided passes (both heaps; the better top moves, ties towards the lighter half).
  void refine_two_sided(const Graph& g, std::uint64_t salt, std::vector<std::uint8_t>& where,
                        std::int64_t max_part, int passes) {
    start(g, salt, where);
    for (int pass = 0; pass < passes && !sep_.empty(); ++pass) {
      for (const std::int32_t v : sep_) {
        separator_degrees(where, v);
        heap_[0].insert(v, g_->vwgt[v] - ed_[1][v], rank(v));
        heap_[1].insert(v, g_->vwgt[v] - ed_[0][v], rank(v));
      }
      const std::int64_t limit =
          std::clamp<std::int64_t>(2 * static_cast<std::int64_t>(sep_.size()), 25, 300);
      const bool improved = run_pass(where, max_part, limit, -1);
      if (!improved) break;
    }
  }

  // One-sided passes: every pass moves vertices into one half only, alternating halves (the
  // lighter one first); up to 2 * passes passes, stopping after a pass into the second half that
  // gained nothing (METIS' default for nested dissection).
  void refine_one_sided(const Graph& g, std::uint64_t salt, std::vector<std::uint8_t>& where,
                        std::int64_t max_part, int passes) {
    start(g, salt, where);
    int to = pw_[0] < pw_[1] ? 1 : 0;
    for (int pass = 0; pass < 2 * passes && !sep_.empty(); ++pass) {
      to = 1 - to;
      for (const std::int32_t v : sep_) {
        separator_degrees(where, v);
        heap_[to].insert(v, g_->vwgt[v] - ed_[1 - to][v], rank(v));
      }
      const std::int64_t limit =
          std::clamp<std::int64_t>(3 * static_cast<std::int64_t>(sep_.size()), 25, 300);
      const bool improved = run_pass(where, max_part, limit, to);
      if (pass % 2 == 1 && !improved) break;
    }
  }

 private:
  std::uint32_t rank(std::int32_t v) const noexcept { return mix_rank(salt_, v); }

  // Current graph, part weights and separator list.
  void start(const Graph& g, std::uint64_t salt, const std::vector<std::uint8_t>& where) {
    HS_ASSERT(static_cast<std::size_t>(g.n) <= locked_.size(),
              "NodeFm: graph larger than the workspace");
    g_ = &g;
    salt_ = salt;
    pw_[0] = pw_[1] = pw_[2] = 0;
    sep_.clear();
    for (std::int32_t v = 0; v < g.n; ++v) {
      pw_[where[v]] += g.vwgt[v];
      if (where[v] == kSeparator) sep_.push_back(v);
    }
  }

  void separator_degrees(const std::vector<std::uint8_t>& where, std::int32_t v) {
    std::int64_t d[3] = {0, 0, 0};
    for (std::int64_t e = g_->xadj[v]; e < g_->xadj[v + 1]; ++e) {
      const std::int32_t u = g_->adj[e];
      d[where[u]] += g_->vwgt[u];
    }
    ed_[0][v] = d[0];
    ed_[1][v] = d[1];
  }

  // One pass over the filled heap(s) (only_to = -1: two-sided). Returns whether the separator
  // improved; leaves the heaps empty and the separator list updated.
  bool run_pass(std::vector<std::uint8_t>& where, std::int64_t max_part, std::int64_t limit,
                int only_to) {
    const std::int64_t initial = pw_[kSeparator];
    log_.clear();
    std::int64_t best = pw_[kSeparator];
    std::int64_t best_diff = std::llabs(pw_[0] - pw_[1]);
    std::size_t best_len = 0;
    std::int64_t since_best = 0;
    for (;;) {
      int to = only_to;
      std::int32_t v = -1;
      if (only_to >= 0) {
        if (heap_[to].empty()) break;
        v = heap_[to].top();
        if (pw_[to] + g_->vwgt[v] > max_part) break;
      } else if (!pick(max_part, to, v)) {
        break;
      }
      move(where, v, to, only_to < 0);
      const std::int64_t diff = std::llabs(pw_[0] - pw_[1]);
      if (pw_[kSeparator] < best || (pw_[kSeparator] == best && diff < best_diff)) {
        best = pw_[kSeparator];
        best_diff = diff;
        best_len = log_.size();
        since_best = 0;
      } else if (++since_best > limit) {
        break;
      }
    }
    heap_[0].clear();
    heap_[1].clear();
    for (std::size_t k = log_.size(); k-- > best_len;) {
      const auto [u, old] = log_[k];
      pw_[where[u]] -= g_->vwgt[u];
      pw_[old] += g_->vwgt[u];
      where[u] = old;
    }
    // New separator list: the old one minus the vertices that left, plus those that entered.
    std::size_t kept = 0;
    for (const std::int32_t v : sep_) {
      if (where[v] == kSeparator) {
        sep_[kept++] = v;
        listed_[v] = 1;
      }
    }
    sep_.resize(kept);
    for (const auto& [u, old] : log_) {
      locked_[u] = 0;
      if (where[u] == kSeparator && !listed_[u]) {
        listed_[u] = 1;
        sep_.push_back(u);
      }
    }
    for (const std::int32_t v : sep_) listed_[v] = 0;
    return best < initial;
  }

  // Two-sided choice: the better of the two heap tops (ties: towards the lighter half) that
  // keeps the target half within max_part.
  bool pick(std::int64_t max_part, int& to, std::int32_t& v) const {
    const bool h0 = !heap_[0].empty(), h1 = !heap_[1].empty();
    if (!h0 && !h1) return false;
    int first;
    if (h0 && h1) {
      const std::int64_t g0 = heap_[0].top_key(), g1 = heap_[1].top_key();
      first = g0 > g1 ? 0 : g0 < g1 ? 1 : (pw_[0] <= pw_[1] ? 0 : 1);
    } else {
      first = h0 ? 0 : 1;
    }
    for (const int side : {first, 1 - first}) {
      if (heap_[side].empty()) continue;
      const std::int32_t u = heap_[side].top();
      if (pw_[side] + g_->vwgt[u] <= max_part) {
        to = side;
        v = u;
        return true;
      }
    }
    return false;
  }

  // Moves separator vertex v into half `to`; its neighbours in the other half join the
  // separator (and the heap(s) in use, unless they already left it in this pass).
  void move(std::vector<std::uint8_t>& where, std::int32_t v, int to, bool two_sided) {
    const Graph& g = *g_;
    const int other = 1 - to;
    heap_[0].remove(v);
    heap_[1].remove(v);
    log_.push_back({v, kSeparator});
    locked_[v] = 1;
    where[v] = static_cast<std::uint8_t>(to);
    pw_[kSeparator] -= g.vwgt[v];
    pw_[to] += g.vwgt[v];
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const std::int32_t u = g.adj[e];
      if (where[u] == kSeparator) {
        ed_[to][u] += g.vwgt[v];
        if (heap_[other].contains(u)) heap_[other].update(u, g.vwgt[u] - ed_[to][u]);
      } else if (where[u] == other) {
        log_.push_back({u, static_cast<std::uint8_t>(other)});
        where[u] = kSeparator;
        pw_[other] -= g.vwgt[u];
        pw_[kSeparator] += g.vwgt[u];
        separator_degrees(where, u);
        for (std::int64_t f = g.xadj[u]; f < g.xadj[u + 1]; ++f) {
          const std::int32_t w = g.adj[f];
          if (where[w] != kSeparator) continue;
          ed_[other][w] -= g.vwgt[u];
          if (heap_[to].contains(w)) heap_[to].update(w, g.vwgt[w] - ed_[other][w]);
        }
        if (!locked_[u]) {
          if (two_sided || to == 0) heap_[0].insert(u, g.vwgt[u] - ed_[1][u], rank(u));
          if (two_sided || to == 1) heap_[1].insert(u, g.vwgt[u] - ed_[0][u], rank(u));
        }
      }
    }
  }

  const Graph* g_ = nullptr;
  std::uint64_t salt_ = 0;
  std::int64_t pw_[3] = {0, 0, 0};  ///< weights of half 0, half 1, separator
  std::vector<std::int32_t> sep_;   ///< separator vertices
  std::unique_ptr<std::int64_t[]>
      ed_[2];                         ///< separator vertex: weight of its neighbours in half 0 / 1
  std::vector<std::uint8_t> locked_;  ///< left the separator in this pass
  std::vector<std::uint8_t> listed_;  ///< scratch mark for the separator list update
  GainHeap heap_[2];                  ///< heap_[s]: gain of moving a separator vertex into half s
  std::vector<std::pair<std::int32_t, std::uint8_t>> log_;  ///< (vertex, previous part)
};

// Refinement by SeparatorParams::refine_mode.
void refine(NodeFm& fm, const Graph& g, Random& rng, std::vector<std::uint8_t>& where,
            std::int64_t max_part, const SeparatorParams& p) {
  const std::uint64_t salt = rng.next();
  if (p.refine_mode != 1) fm.refine_two_sided(g, salt, where, max_part, p.refine_passes);
  if (p.refine_mode != 0) fm.refine_one_sided(g, salt, where, max_part, p.refine_passes);
}

// ------------------------------------------------------------------------- initial separator

// Greedy graph growing: breadth-first from a random vertex (restarting at a random unreached
// vertex in another component) until half 0 holds half of the weight; the boundary vertices of
// both halves form the separator.
void grow_separator(const Graph& g, Random& rng, std::int64_t max_part,
                    std::vector<std::uint8_t>& where) {
  const std::int32_t n = g.n;
  where.assign(static_cast<std::size_t>(n), 1);
  std::vector<std::uint8_t> queued(static_cast<std::size_t>(n), 0);
  std::vector<std::int32_t> queue;
  queue.reserve(static_cast<std::size_t>(n));
  const std::int64_t target = g.total_vwgt / 2;
  std::int64_t pw0 = 0;
  std::size_t head = 0;
  std::int32_t scan = static_cast<std::int32_t>(rng.below(static_cast<std::uint32_t>(n)));
  std::int32_t scanned = 0;
  while (pw0 < target) {
    if (head == queue.size()) {
      while (scanned < n && queued[scan]) {
        scan = scan + 1 == n ? 0 : scan + 1;
        ++scanned;
      }
      if (scanned == n) break;
      queued[scan] = 1;
      queue.push_back(scan);
    }
    const std::int32_t v = queue[head++];
    if (pw0 + g.vwgt[v] > max_part) continue;
    where[v] = 0;
    pw0 += g.vwgt[v];
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const std::int32_t u = g.adj[e];
      if (!queued[u]) {
        queued[u] = 1;
        queue.push_back(u);
      }
    }
  }
  std::vector<std::int32_t> boundary;
  for (std::int32_t v = 0; v < n; ++v) {
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      if (where[g.adj[e]] != where[v]) {
        boundary.push_back(v);
        break;
      }
    }
  }
  for (const std::int32_t v : boundary) where[v] = kSeparator;
}

// Separator weight of `where` on g; diff receives the weight difference of the halves.
std::int64_t separator_score(const Graph& g, const std::vector<std::uint8_t>& where,
                             std::int64_t& diff) {
  std::int64_t pw[3] = {0, 0, 0};
  for (std::int32_t v = 0; v < g.n; ++v) pw[where[v]] += g.vwgt[v];
  diff = std::llabs(pw[0] - pw[1]);
  return pw[kSeparator];
}

std::int64_t separator_weight(const Graph& g, const std::vector<std::uint8_t>& where) {
  std::int64_t diff = 0;
  return separator_score(g, where, diff);
}

// Bound on the weight of each half: imbalance * total / 2, but never below what a vertex-level
// split can guarantee. Clamped to the total weight in floating point (imbalance >= 2 already
// allows a half to hold everything; a huge value would overflow the conversion to int64).
std::int64_t max_part_weight(const Graph& g, double imbalance) {
  std::int32_t heaviest = 1;
  for (const std::int32_t w : g.vwgt) heaviest = std::max(heaviest, w);
  const double total = static_cast<double>(g.total_vwgt);
  const double half = imbalance * 0.5 * total;
  const std::int64_t bound =
      half < total ? static_cast<std::int64_t>(std::max(0.0, half)) : g.total_vwgt;  // NaN: total
  return std::max(bound, (g.total_vwgt + 1) / 2 + heaviest);
}

// Best of `tries` refined greedy growings on a (small) graph. METIS' ConstructSeparator refines
// the boundary separator two-sided, then one-sided.
std::vector<std::uint8_t> initial_separator(const Graph& g, Random& rng, std::int64_t max_part,
                                            int tries, const SeparatorParams& p, NodeFm& fm) {
  std::vector<std::uint8_t> where, trial;
  std::int64_t best_w = -1, best_diff = 0;
  for (int t = 0; t < std::max(1, tries); ++t) {
    grow_separator(g, rng, max_part, trial);
    const std::uint64_t salt = rng.next();
    fm.refine_two_sided(g, salt, trial, max_part, p.refine_passes);
    if (p.refine_mode != 0) fm.refine_one_sided(g, salt, trial, max_part, p.refine_passes);
    std::int64_t diff = 0;
    const std::int64_t w = separator_score(g, trial, diff);
    if (best_w < 0 || w < best_w || (w == best_w && diff < best_diff)) {
      best_w = w;
      best_diff = diff;
      where.swap(trial);
    }
  }
  return where;
}

// ------------------------------------------------------------------------------ multilevel

// Stopwatch for the trace: lap() returns the time since the previous lap.
struct Lap {
  Timer t;
  double lap() {
    const double s = t.seconds();
    t.reset();
    return s;
  }
};

// Coarsening hierarchy above a base graph: coarse[k] is the graph after k + 1 contractions,
// cmaps[k] maps the vertices of level k (0 = base) to those of level k + 1.
struct Hierarchy {
  std::vector<Graph> coarse;
  std::vector<std::vector<std::int32_t>> cmaps;

  const Graph& top(const Graph& base) const { return coarse.empty() ? base : coarse.back(); }

  // Contracts until the top has at most `target` vertices, `max_levels` contractions were done
  // or a contraction removes less than 10% of the vertices - and at least one: below 10
  // vertices n / 10 is 0, and a graph whose vertices cannot be matched (isolated, or heavier
  // than max_vwgt) contracts to itself, which would be pushed forever (coarsen_to < 10 on a
  // graph with more components than that).
  void build(const Graph& base, Random& rng, std::int64_t max_vwgt, std::int32_t target,
             int max_levels) {
    while (static_cast<int>(coarse.size()) < max_levels) {
      const Graph& cur = top(base);
      if (cur.n <= target) break;
      std::vector<std::int32_t> cmap;
      Graph c = coarsen_once(cur, rng, max_vwgt, cmap);
      if (c.n > cur.n - std::max<std::int32_t>(1, cur.n / 10)) break;
      coarse.push_back(std::move(c));
      cmaps.push_back(std::move(cmap));
    }
  }

  // Projects `where` (on the top) level by level down to the base, refining at every level and
  // freeing the coarse graphs on the way. Appends the separator weights per level to the trace
  // (coarsest first).
  void uncoarsen(const Graph& base, Random& rng, NodeFm& fm, std::vector<std::uint8_t>& where,
                 std::int64_t max_part, const SeparatorParams& p, SeparatorTrace* trace) {
    for (std::size_t k = coarse.size(); k-- > 0;) {
      const Graph& fine = k == 0 ? base : coarse[k - 1];
      std::vector<std::uint8_t> projected(static_cast<std::size_t>(fine.n));
      const std::vector<std::int32_t>& cmap = cmaps[k];
      for (std::int32_t v = 0; v < fine.n; ++v) projected[v] = where[cmap[v]];
      where.swap(projected);
      coarse.pop_back();
      cmaps.pop_back();
      if (trace) trace->projected.push_back(separator_weight(fine, where));
      refine(fm, fine, rng, where, max_part, p);
      if (trace) {
        trace->refined.push_back(separator_weight(fine, where));
        trace->vertices.push_back(fine.n);
        trace->edges.push_back(fine.edges());
      }
    }
  }

  // The same without freeing anything, so that several separators of the top can be carried
  // down the same hierarchy (concurrently: const).
  void refine_down(const Graph& base, Random& rng, NodeFm& fm, std::vector<std::uint8_t>& where,
                   std::int64_t max_part, const SeparatorParams& p) const {
    for (std::size_t k = coarse.size(); k-- > 0;) {
      const Graph& fine = k == 0 ? base : coarse[k - 1];
      std::vector<std::uint8_t> projected(static_cast<std::size_t>(fine.n));
      const std::vector<std::int32_t>& cmap = cmaps[k];
      for (std::int32_t v = 0; v < fine.n; ++v) projected[v] = where[cmap[v]];
      where.swap(projected);
      refine(fm, fine, rng, where, max_part, p);
    }
  }
};

// One multilevel bisection (METIS MlevelNodeBisectionL1): coarsen to min(coarsen_to,
// max(n/8, 40)) vertices, initial separator there, refine back up.
std::vector<std::uint8_t> bisect_once(const Graph& g, Random& rng, const SeparatorParams& p,
                                      std::int64_t max_part, int tries, SeparatorTrace* trace) {
  const std::int32_t target = std::min(p.coarsen_to, std::max<std::int32_t>(g.n / 8, 40));
  const std::int64_t max_vwgt = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(1.5 * static_cast<double>(g.total_vwgt) / target));
  Lap timer;
  Hierarchy h;
  h.build(g, rng, max_vwgt, target, 1 << 30);
  if (trace) trace->coarsen_s += timer.lap();
  const Graph& top = h.top(g);
  NodeFm fm(g.n);
  std::vector<std::uint8_t> where = initial_separator(top, rng, max_part, tries, p, fm);
  if (trace) {
    trace->init_s += timer.lap();
    trace->vertices.push_back(top.n);
    trace->edges.push_back(top.edges());
    trace->projected.push_back(-1);
    trace->refined.push_back(separator_weight(top, where));
  }
  h.uncoarsen(g, rng, fm, where, max_part, p, trace);
  if (trace) trace->refine_s += timer.lap();
  return where;
}

}  // namespace

std::vector<std::uint8_t> vertex_separator(const Graph& g, std::uint64_t seed,
                                           const SeparatorParams& p, SeparatorTrace* trace,
                                           SeparatorTiming* timing) {
  HS_ASSERT(g.n >= 1, "vertex_separator: empty graph");
  Lap stage;
  Random rng(seed);
  const std::int64_t max_part = max_part_weight(g, p.imbalance);
  SeparatorTrace local;
  SeparatorTrace* tr = trace ? &local : nullptr;
  std::vector<std::uint8_t> where;
  if (g.n < kTwoStageSize) {
    where = bisect_once(g, rng, p, max_part, p.init_tries, tr);
    if (timing) timing->small_s += stage.lap();
  } else {
    // A few contractions without a weight bound (METIS: CoarsenTo = max(100, n/30)).
    Lap timer;
    Hierarchy h;
    const std::int32_t target = std::max<std::int32_t>(100, g.n / 30);
    h.build(g, rng, g.total_vwgt, target, kTwoStageLevels);
    const double outer_coarsen = timer.lap();
    if (timing) timing->coarsen_s += stage.lap();
    const Graph& mid = h.top(g);
    const int tries = std::max(1, static_cast<int>(0.7 * p.init_tries));
    // Independent runs, each with its own seed: the same result whichever thread runs which.
    const int nruns = std::max(1, p.runs);
    std::vector<std::vector<std::uint8_t>> runs(static_cast<std::size_t>(nruns));
    std::vector<SeparatorTrace> run_traces(static_cast<std::size_t>(nruns));
    std::vector<std::uint64_t> run_seed(static_cast<std::size_t>(nruns));
    for (int r = 0; r < nruns; ++r) run_seed[r] = rng.next();
    const bool parallel = mid.n >= kParallelRunSize;
    // A taskgroup, not a taskwait: the caller (the dissection) may already have spawned other
    // tasks from the task it runs in - the other half of a separator - and a taskwait would wait
    // for those whole subtrees too (P9: at n = 1e6 the 16-thread dissection took 7.3 s for 18 s
    // of work, the trial runs' wall time 16 s instead of 3 s summed over the separators).
#pragma omp taskgroup
    {
      for (int r = 0; r < nruns; ++r) {
#pragma omp task if (parallel) default(shared) firstprivate(r)
        {
          Random run_rng(run_seed[r]);
          runs[r] = bisect_once(mid, run_rng, p, max_part, tries, tr ? &run_traces[r] : nullptr);
        }
      }
    }
    // Runs ranked by their separator on the mid graph (weight, then balance, then run number).
    std::vector<std::pair<std::int64_t, std::int64_t>> score(static_cast<std::size_t>(nruns));
    std::vector<int> rank(static_cast<std::size_t>(nruns));
    for (int r = 0; r < nruns; ++r) {
      score[r].first = separator_score(mid, runs[r], score[r].second);
      rank[r] = r;
    }
    std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) { return score[a] < score[b]; });
    const int best = rank[0];
    const int finalists = std::clamp(p.finalists, 1, nruns);
    if (finalists > 1) {
      // Carry the best `finalists` separators of the mid graph down to g, each refined with its
      // own random stream (concurrent tasks), and keep the best one there: the mid graph's
      // ranking is a poor predictor of the final separator (projection and refinement through
      // the four finest levels change it most).
      std::vector<std::uint64_t> fin_seed(static_cast<std::size_t>(finalists));
      for (int f = 0; f < finalists; ++f) fin_seed[f] = rng.next();
      std::vector<std::vector<std::uint8_t>> fin(static_cast<std::size_t>(finalists));
      std::vector<std::pair<std::int64_t, std::int64_t>> fscore(static_cast<std::size_t>(finalists));
      if (timing) timing->runs_s += stage.lap();
#pragma omp taskgroup
      {
        for (int f = 0; f < finalists; ++f) {
#pragma omp task if (parallel) default(shared) firstprivate(f)
          {
            Random frng(fin_seed[f]);
            NodeFm ffm(g.n);
            fin[f] = std::move(runs[rank[f]]);
            h.refine_down(g, frng, ffm, fin[f], max_part, p);
            fscore[f].first = separator_score(g, fin[f], fscore[f].second);
          }
        }
      }
      int pick = 0;
      for (int f = 1; f < finalists; ++f)
        if (fscore[f] < fscore[pick]) pick = f;
      where = std::move(fin[pick]);
      if (timing) timing->uncoarsen_s += stage.lap();
      if (trace) {
        local.vertices.push_back(g.n);
        local.edges.push_back(g.edges());
        local.projected.push_back(-1);
        local.refined.push_back(fscore[pick].first);
        local.coarsen_s = outer_coarsen;
        *trace = std::move(local);
      }
      return where;
    }
    where = std::move(runs[best]);
    runs.clear();
    timer.lap();
    if (timing) timing->runs_s += stage.lap();
    if (tr) {
      local = std::move(run_traces[best]);
      local.coarsen_s = outer_coarsen;
      local.init_s = local.refine_s = 0.0;
      for (const SeparatorTrace& t : run_traces) {
        local.coarsen_s += t.coarsen_s;
        local.init_s += t.init_s;
        local.refine_s += t.refine_s;
      }
    }
    NodeFm fm(g.n);
    h.uncoarsen(g, rng, fm, where, max_part, p, tr);
    if (tr) local.refine_s += timer.lap();
    if (timing) timing->uncoarsen_s += stage.lap();
  }
  if (trace) {
    // Collected coarsest first; report finest first.
    std::reverse(local.vertices.begin(), local.vertices.end());
    std::reverse(local.edges.begin(), local.edges.end());
    std::reverse(local.projected.begin(), local.projected.end());
    std::reverse(local.refined.begin(), local.refined.end());
    *trace = std::move(local);
  }
  return where;
}

}  // namespace hs::ordering
