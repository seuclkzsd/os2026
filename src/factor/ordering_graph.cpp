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
