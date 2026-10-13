// hs::resolve_memory_budget (include/hsolve/solver.hpp): the memory budget of a solve, configured
// or derived from the memory limits of the process (docs/DESIGN.md §3.6-3).
#include <algorithm>
#include <cmath>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace hs {

MemoryBudget resolve_memory_budget(double configured_gb, const MemoryLimits& limits, double reserved_gib) {
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
  b.source = limits.source;
  b.measure = measure_of(limits.source);
  if (b.measure != MemoryMeasure::Resident && std::isfinite(reserved_gib) && reserved_gib > 0)
    b.reserved_gb = reserved_gib;
  const double l = limits.gib - b.reserved_gb;  // what the problem can use of the limit
  b.gb = b.reserved_gb +
         std::max(kAutoBudgetFloorGb, std::min(kAutoBudgetFraction * l, l - kAutoBudgetHeadroomGb));
  return b;
}

MemoryBudget resolve_memory_budget(double configured_gb) {
  const MemoryLimits limits = detect_memory_limits();
  double reserved = 0.0;
  const MemoryMeasure m = measure_of(limits.source);
  if (configured_gb == kMemoryBudgetAuto && limits.gib > 0 && m != MemoryMeasure::Resident) {
    const auto in_use = memory_in_use_bytes(m), rss = current_rss_bytes();
    if (in_use > 0 && rss >= 0 && in_use > rss) reserved = static_cast<double>(in_use - rss) / 1073741824.0;
  }
  return resolve_memory_budget(configured_gb, limits, reserved);
}

}  // namespace hs
