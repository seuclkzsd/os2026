// Minimum-degree ordering of the small leaf subgraphs of the nested dissection
// (ordering_detail.hpp): exact degrees on the explicit elimination graph, stored as bitsets.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"

namespace hs::ordering {

namespace {

inline int popcount64(std::uint64_t x) noexcept {
  return __builtin_popcountll(x);
}

}  // namespace

std::vector<std::int32_t> minimum_degree(const Graph& g) {
  const std::int32_t n = g.n;
  const std::size_t words = (static_cast<std::size_t>(n) + 63) / 64;
  // Row v of `adj` holds the current neighbours of v in the elimination graph. Eliminating v
  // turns its neighbourhood into a clique and removes v; eliminated vertices never reappear,
  // because a vertex is only removed from the rows of its neighbours, which are exactly the rows
  // that contain it.
  std::vector<std::uint64_t> adj(static_cast<std::size_t>(n) * words, 0);
  auto row = [&](std::int32_t v) { return adj.data() + static_cast<std::size_t>(v) * words; };
  std::vector<std::int32_t> degree(static_cast<std::size_t>(n), 0);
  bool unit = true;
  for (std::int32_t v = 0; v < n; ++v) unit = unit && g.vwgt[v] == 1;
  // Degree of a row: popcount for unit weights, else the summed weights of its members.
  auto row_degree = [&](const std::uint64_t* r) {
    std::int64_t d = 0;
    if (unit) {
      for (std::size_t w = 0; w < words; ++w) d += popcount64(r[w]);
    } else {
      for (std::size_t w = 0; w < words; ++w)
        for (std::uint64_t bits = r[w]; bits; bits &= bits - 1)
          d += g.vwgt[static_cast<std::size_t>(w * 64 + __builtin_ctzll(bits))];
    }
    return static_cast<std::int32_t>(
        std::min<std::int64_t>(d, std::numeric_limits<std::int32_t>::max() - 1));
  };
  for (std::int32_t v = 0; v < n; ++v) {
    std::uint64_t* r = row(v);
    for (std::int64_t e = g.xadj[v]; e < g.xadj[v + 1]; ++e) {
      const std::int32_t u = g.adj[e];
      r[u >> 6] |= std::uint64_t{1} << (u & 63);
    }
    degree[v] = row_degree(r);
  }
  constexpr std::int32_t kDone = std::numeric_limits<std::int32_t>::max();
  std::vector<std::int32_t> order;
  order.reserve(static_cast<std::size_t>(n));
  std::vector<std::int32_t> nbrs;
  for (std::int32_t k = 0; k < n; ++k) {
    std::int32_t v = 0;
    for (std::int32_t u = 1; u < n; ++u)
      if (degree[u] < degree[v]) v = u;
    HS_ASSERT(degree[v] != kDone, "minimum_degree: no vertex left");
    order.push_back(v);
    degree[v] = kDone;
    const std::uint64_t* rv = row(v);
    nbrs.clear();
    for (std::size_t w = 0; w < words; ++w)
      for (std::uint64_t bits = rv[w]; bits; bits &= bits - 1)
        nbrs.push_back(static_cast<std::int32_t>(w * 64 + __builtin_ctzll(bits)));
    for (const std::int32_t u : nbrs) {
      std::uint64_t* ru = row(u);
      for (std::size_t w = 0; w < words; ++w) ru[w] |= rv[w];
      ru[u >> 6] &= ~(std::uint64_t{1} << (u & 63));
      ru[v >> 6] &= ~(std::uint64_t{1} << (v & 63));
      degree[u] = row_degree(ru);
    }
  }
  return order;
}

}  // namespace hs::ordering
