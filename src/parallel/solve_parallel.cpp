// hs::solve_parallel (include/hsolve/parallel.hpp; DESIGN §3.9): hs::solve on the leader with the
// helpers of the session behind driver::HelpersOf (RemoteHelpers).
#include <chrono>
#include <cstdint>

#include "driver/helpers.hpp"
#include "hsolve/error.hpp"
#include "hsolve/parallel.hpp"
#include "parallel/remote_helpers.hpp"

namespace hs {

namespace {
// Tags the problem of one solve in the session's messages (wire.hpp: generation): a sanity check
// - the leader collects every reply before the next solve - not a counter (no process state).
std::uint32_t generation_tag() {
  const auto t = std::chrono::steady_clock::now().time_since_epoch().count();
  return static_cast<std::uint32_t>(t ^ (t >> 32)) | 1u;
}
}  // namespace

template <class T>
Solution<T> solve_parallel(const CsrMatrix<T>& a, const non_deduced_t<CsrMatrix<T>>* b, const SolverConfig& cfg,
                           Logger* log, Transport& peers, const ParallelOptions& options) {
  if (peers.rank() != 0) HS_THROW(InvalidInput, "solve_parallel runs on rank 0, this is rank " << peers.rank());
  if (peers.size() == 1) return driver::solve_with_helpers<T>(a, b, cfg, log, nullptr);
  parallel::RemoteHelpers<T> helpers(peers, generation_tag(), options);
  return driver::solve_with_helpers<T>(a, b, cfg, log, &helpers);
}

#define HS_INSTANTIATE_SOLVE_PARALLEL(T)                                                                   \
  template Solution<T> solve_parallel<T>(const CsrMatrix<T>&, const non_deduced_t<CsrMatrix<T>>*,        \
                                         const SolverConfig&, Logger*, Transport&, const ParallelOptions&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_PARALLEL)
#undef HS_INSTANTIATE_SOLVE_PARALLEL

}  // namespace hs
