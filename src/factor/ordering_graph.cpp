// Graphs of the ordering module: construction from an upper-triangle pattern, induced
// subgraphs, separator checks and the counter-based random numbers (ordering_detail.hpp).
#include <algorithm>
#include <cstdint>
#include <vector>

#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"

namespace hs::ordering {

std::uint64_t Random::next() noexcept {
  std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::uint32_t Random::below(std::uint32_t bound) noexcept {
  return static_cast<std::uint32_t>(((next() >> 32) * static_cast<std::uint64_t>(bound)) >> 32);
}

std::uint64_t child_seed(std::uint64_t parent, int side) noexcept {
  Random r(parent ^ (side == 0 ? 0x243F6A8885A308D3ULL : 0x13198A2E03707344ULL));
  return r.next();
}

Graph graph_from_upper(std::int32_t n, const std::int64_t* rowptr, const std::int32_t* colind) {
  HS_ASSERT(n >= 0, "graph_from_upper: negative order " << n);
  Graph g;
  g.n = n;
  g.xadj.assign(static_cast<std::size_t>(n) + 1, 0);
  // Degrees: every strictly upper entry (i, j) adds j to row i and i to row j.
  for (std::int32_t i = 0; i < n; ++i) {
    for (std::int64_t p = rowptr[i]; p < rowptr[i + 1]; ++p) {
      const std::int32_t j = colind[p];
      if (j == i) continue;
      HS_ASSERT(j > i && j < n, "graph_from_upper: entry (" << i << ", " << j << ") is not upper");
      ++g.xadj[i + 1];
      ++g.xadj[j + 1];
    }
  }
  for (std::int32_t i = 0; i < n; ++i) g.xadj[i + 1] += g.xadj[i];
  g.adj.resize(static_cast<std::size_t>(g.xadj[n]));
  // Row i receives its lower neighbours (from earlier rows, in increasing order) before its
  // upper ones (increasing): every row comes out sorted without a sort.
  std::vector<std::int64_t> fill(g.xadj.begin(), g.xadj.end() - 1);
  for (std::int32_t i = 0; i < n; ++i) {
    for (std::int64_t p = rowptr[i]; p < rowptr[i + 1]; ++p) {
      const std::int32_t j = colind[p];
      if (j == i) continue;
      g.adj[static_cast<std::size_t>(fill[j]++)] = i;
    }
    for (std::int64_t p = rowptr[i]; p < rowptr[i + 1]; ++p) {
      const std::int32_t j = colind[p];
      if (j == i) continue;
      g.adj[static_cast<std::size_t>(fill[i]++)] = j;
    }
  }
  g.ewgt.assign(g.adj.size(), 1);
  g.vwgt.assign(static_cast<std::size_t>(n), 1);
  g.total_vwgt = n;
  return g;
}

namespace {

// Order-independent hash of a vertex set (sum of mixed members).
inline std::uint64_t member_hash(std::int32_t v) noexcept {
  std::uint64_t z = 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(v) + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Whether adjacent vertices u and v have the same closed neighbourhood: adj(u) - {v} equals
// adj(v) - {u} (rows sorted).
bool same_closed(const Graph& g, std::int32_t u, std::int32_t v) {
  std::int64_t a = g.xadj[u], ae = g.xadj[u + 1], b = g.xadj[v], be = g.xadj[v + 1];
  for (;;) {
    while (a < ae && g.adj[a] == v) ++a;
    while (b < be && g.adj[b] == u) ++b;
    if (a == ae || b == be) return a == ae && b == be;
    if (g.adj[a] != g.adj[b]) return false;
    ++a;
    ++b;
  }
}

}  // namespace

bool compress_graph(const Graph& g, double max_ratio, int threads, Graph& c,
                    std::vector<std::int32_t>& first, std::vector<std::int32_t>& members) {
  const std::int32_t n = g.n;
  c = Graph{};
  first.clear();
  members.clear();
  if (n == 0) return false;
  std::vector<std::uint64_t> h(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
  for (std::int32_t v = 0; v < n; ++v) {
    std::uint64_t s = member_hash(v);
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) s += member_hash(g.adj[e]);
    h[v] = s;
  }
  // rep[v]: the smallest vertex of v's class - a neighbour of v (or v itself).
  std::vector<std::int32_t> rep(static_cast<std::size_t>(n));
  std::int64_t classes = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : classes) \
    num_threads(std::max(1, threads))
  for (std::int32_t v = 0; v < n; ++v) {
    std::int32_t r = v;
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const std::int32_t u = g.adj[e];
      if (u >= r) continue;
      if (h[u] == h[v] && g.degree(u) == g.degree(v) && same_closed(g, u, v)) r = u;
    }
    rep[v] = r;
    classes += r == v;
  }
  if (static_cast<double>(classes) > max_ratio * static_cast<double>(n)) return false;
  // Class numbers in the order of the representatives, members by class (increasing).
  const auto nc = static_cast<std::int32_t>(classes);
  std::vector<std::int32_t> id(static_cast<std::size_t>(n), -1);
  first.assign(static_cast<std::size_t>(nc) + 1, 0);
  std::int32_t k = 0;
  for (std::int32_t v = 0; v < n; ++v)
    if (rep[v] == v) id[v] = k++;
  for (std::int32_t v = 0; v < n; ++v) ++first[id[rep[v]] + 1];
  for (std::int32_t q = 0; q < nc; ++q) first[q + 1] += first[q];
  members.resize(static_cast<std::size_t>(n));
  {
    std::vector<std::int32_t> slot(first.begin(), first.end() - 1);
    for (std::int32_t v = 0; v < n; ++v) members[slot[id[rep[v]]]++] = v;
  }
  c.n = nc;
  c.xadj.assign(static_cast<std::size_t>(nc) + 1, 0);
  c.vwgt.assign(static_cast<std::size_t>(nc), 0);
  for (std::int32_t v = 0; v < n; ++v) c.vwgt[id[rep[v]]] += g.vwgt[v];
  c.total_vwgt = g.total_vwgt;
  // Rows: the classes adjacent to the representative's class, deduplicated with a stamp.
  std::vector<std::int32_t> stamp(static_cast<std::size_t>(nc), -1);
  for (std::int32_t q = 0; q < nc; ++q) {
    const std::int32_t r = members[first[q]];
    stamp[q] = q;
    for (std::int64_t e = g.xadj[r]; e < g.xadj[r + 1]; ++e) {
      const std::int32_t t = id[rep[g.adj[e]]];
      if (stamp[t] == q) continue;
      stamp[t] = q;
      c.adj.push_back(t);
    }
    c.xadj[q + 1] = static_cast<std::int64_t>(c.adj.size());
    std::sort(c.adj.begin() + c.xadj[q], c.adj.end());
  }
  c.ewgt.assign(c.adj.size(), 1);
  return true;
}

Graph induced_subgraph(const Graph& g, const std::vector<std::uint8_t>& where, std::uint8_t part,
                       std::vector<std::int32_t>& to_parent) {
  std::vector<std::int32_t> local(static_cast<std::size_t>(g.n), -1);
  to_parent.clear();
  for (std::int32_t v = 0; v < g.n; ++v) {
    if (where[v] == part) {
      local[v] = static_cast<std::int32_t>(to_parent.size());
      to_parent.push_back(v);
    }
  }
  Graph s;
  s.n = static_cast<std::int32_t>(to_parent.size());
  if (g.n >= kParallelGraphSize) {
    // The same rows, counted and copied by concurrent tasks (taskloop: the caller's team).
    s.xadj.assign(static_cast<std::size_t>(s.n) + 1, 0);
    s.vwgt.resize(static_cast<std::size_t>(s.n));
    constexpr std::int32_t kChunk = 4096;
#pragma omp taskloop grainsize(1) default(shared)
    for (std::int32_t b = 0; b < s.n; b += kChunk)
      for (std::int32_t k = b; k < std::min(s.n, b + kChunk); ++k) {
        const std::int32_t v = to_parent[k];
        std::int64_t d = 0;
        for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) d += (where[g.adj[e]] == part);
        s.xadj[k + 1] = d;
        s.vwgt[k] = g.vwgt[v];
      }
    for (std::int32_t k = 0; k < s.n; ++k) {
      s.xadj[k + 1] += s.xadj[k];
      s.total_vwgt += s.vwgt[k];
    }
    s.adj.resize(static_cast<std::size_t>(s.xadj[s.n]));
    s.ewgt.resize(static_cast<std::size_t>(s.xadj[s.n]));
#pragma omp taskloop grainsize(1) default(shared)
    for (std::int32_t b = 0; b < s.n; b += kChunk)
      for (std::int32_t k = b; k < std::min(s.n, b + kChunk); ++k) {
        const std::int32_t v = to_parent[k];
        std::int64_t q = s.xadj[k];
        for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
          const std::int32_t u = g.adj[e];
          if (where[u] != part) continue;
          s.adj[q] = local[u];
          s.ewgt[q] = g.ewgt[e];
          ++q;
        }
      }
    return s;
  }
  s.xadj.assign(static_cast<std::size_t>(s.n) + 1, 0);
  s.vwgt.resize(static_cast<std::size_t>(s.n));
  std::int64_t edges = 0;
  for (std::int32_t k = 0; k < s.n; ++k) {
    const std::int32_t v = to_parent[k];
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) edges += (where[g.adj[e]] == part);
    s.xadj[k + 1] = edges;
    s.vwgt[k] = g.vwgt[v];
    s.total_vwgt += g.vwgt[v];
  }
  s.adj.resize(static_cast<std::size_t>(edges));
  s.ewgt.resize(static_cast<std::size_t>(edges));
  std::int64_t q = 0;
  for (std::int32_t k = 0; k < s.n; ++k) {
    const std::int32_t v = to_parent[k];
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const std::int32_t u = g.adj[e];
      if (where[u] != part) continue;
      s.adj[q] = local[u];  // increasing: the parent row is sorted and `local` is monotone
      s.ewgt[q] = g.ewgt[e];
      ++q;
    }
  }
  return s;
}

SeparatorCheck check_separator(const Graph& g, const std::vector<std::uint8_t>& where) {
  SeparatorCheck c;
  c.valid = static_cast<std::int32_t>(where.size()) == g.n;
  if (!c.valid) return c;
  for (std::int32_t v = 0; v < g.n; ++v) {
    const std::uint8_t p = where[v];
    if (p > kSeparator) {
      c.valid = false;
      return c;
    }
    c.weight[p] += g.vwgt[v];
    if (p == kSeparator) continue;
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const std::uint8_t q = where[g.adj[e]];
      if (q != kSeparator && q != p) c.valid = false;
    }
  }
  return c;
}

}  // namespace hs::ordering
