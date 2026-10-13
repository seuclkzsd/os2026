// hs::resolve_memory_budget (include/hsolve/solver.hpp): the memory budget of a solve, configured
// or derived from the memory limits of the process (docs/DESIGN.md §3.6-3).
#include <algorithm>
#include <cmath>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace hs {

MemoryBudget resolve_memory_budget(double configured_gb, const MemoryLimits& limits) {
  if (!std::isfinite(configured_gb) || configured_gb < 0)
    HS_THROW(InvalidInput, "memory_budget_gb must be > 0 (or 0 = auto), got " << configured_gb);
  MemoryBudget b;
  b.limits = limits.sources;
  b.limit_gb = limits.gib > 0 ? limits.gib : -1.0;
  b.limit_source = limits.gib > 0 ? limits.source : std::string();
  if (configured_gb > 0) {
    b.gb = configured_gb;
    b.source = "config";
    return b;
  }
  if (!(limits.gib > 0)) {
    b.gb = kAutoBudgetFallbackGb;
    b.source = "default";
    return b;
  }
  const double l = limits.gib;
  b.gb = std::max(kAutoBudgetFloorGb, std::min(kAutoBudgetFraction * l, l - kAutoBudgetHeadroomGb));
  b.source = limits.source;
  return b;
}

MemoryBudget resolve_memory_budget(double configured_gb) {
  return resolve_memory_budget(configured_gb, detect_memory_limits());
}

}  // namespace hs
