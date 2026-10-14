// Encoding of the driver's helper tasks and replies (driver/helpers.hpp) in session messages
// (wire.hpp). Module src/parallel, internal. Scalar set of the templates: T in {double, cplx}.
#pragma once

#include <type_traits>
#include <vector>

#include "driver/helpers.hpp"
#include "hsolve/factor.hpp"
#include "parallel/wire.hpp"

namespace hs::parallel {

static_assert(std::is_trivially_copyable_v<driver::ProblemInfo>, "ProblemInfo is sent as bytes");
static_assert(std::is_trivially_copyable_v<driver::CountTask>, "CountTask is sent as bytes");
static_assert(std::is_trivially_copyable_v<driver::SliceTask>, "SliceTask is sent as bytes");
static_assert(std::is_trivially_copyable_v<driver::PairMetrics>, "PairMetrics is sent as bytes");

inline void put(Writer& w, const driver::HelperError& e) {
  w.pod(e.ok);
  w.pod(static_cast<std::int32_t>(e.code));
  w.pod(e.memory);
  w.str(e.what);
}
inline driver::HelperError get_error(Reader& r) {
  driver::HelperError e;
  e.ok = r.pod<bool>();
  e.code = static_cast<Error::Code>(r.pod<std::int32_t>());
  e.memory = r.pod<bool>();
  e.what = r.str();
  return e;
}

// Every field of FactorOptions but the nested-dissection options (helpers analyse on the leader's
// pivot order, PERM_IN, and never order).
inline void put(Writer& w, const FactorOptions& o) {
  w.pod(o.precision);
  w.pod(o.ordering);
  w.pod(o.ordering_threads);
  w.pod(o.block_analysis);
  w.pod(o.scaling);
  w.pod(o.threads);
  w.pod(o.mem_relax_pct);
  w.pod(o.max_memory_mb);
  w.pod(o.blr);
  w.pod(o.blr_tol);
  w.pod(o.blr_estimates);
  w.pod(o.out_of_core);
  w.pod(o.ooc_on_memory_error);
  w.str(o.ooc_dir);
  w.pod(o.null_pivot_tol);
  w.pod(o.single_null_pivot_tol);
  w.pod(o.deterministic);
  w.pod(o.verbosity);
}
inline FactorOptions get_factor_options(Reader& r) {
  FactorOptions o;
  o.precision = r.pod<FactorPrecision>();
  o.ordering = r.pod<int>();
  o.ordering_threads = r.pod<int>();
  o.block_analysis = r.pod<int>();
  o.scaling = r.pod<int>();
  o.threads = r.pod<int>();
  o.mem_relax_pct = r.pod<int>();
  o.max_memory_mb = r.pod<double>();
  o.blr = r.pod<bool>();
  o.blr_tol = r.pod<double>();
  o.blr_estimates = r.pod<bool>();
  o.out_of_core = r.pod<bool>();
  o.ooc_on_memory_error = r.pod<bool>();
  o.ooc_dir = r.str();
  o.null_pivot_tol = r.pod<double>();
  o.single_null_pivot_tol = r.pod<double>();
  o.deterministic = r.pod<bool>();
  o.verbosity = r.pod<int>();
  return o;
}

inline void put(Writer& w, const driver::CountReply& c) {
  put(w, c.error);
  w.pod(c.nu);
  w.pod(c.s);
  w.pod(c.endpoint_moves);
  w.pod(c.detection_off);
  w.pod(c.seconds);
  w.pod(c.rss_bytes);
  w.pod(c.count_mb);
}
inline driver::CountReply get_count_reply(Reader& r) {
  driver::CountReply c;
  c.error = get_error(r);
  c.nu = r.pod<index_t>();
  c.s = r.pod<double>();
  c.endpoint_moves = r.pod<index_t>();
  c.detection_off = r.pod<bool>();
  c.seconds = r.pod<double>();
  c.rss_bytes = r.pod<double>();
  c.count_mb = r.pod<double>();
  return c;
}

inline void put(Writer& w, const SliceReport& s) {
  w.pod(s.lower);
  w.pod(s.upper);
  w.pod(s.sigma);
  w.str(s.which);
  w.pod(s.count_inertia);
  w.pod(s.count_found);
  w.pod(s.nev);
  w.pod(s.reruns);
  w.pod(s.restarts);
  w.pod(s.op_applies);
  w.pod(s.passed);
}
inline SliceReport get_slice_report(Reader& r) {
  SliceReport s;
  s.lower = r.pod<double>();
  s.upper = r.pod<double>();
  s.sigma = r.pod<double>();
  s.which = r.str();
  s.count_inertia = r.pod<index_t>();
  s.count_found = r.pod<index_t>();
  s.nev = r.pod<index_t>();
  s.reruns = r.pod<index_t>();
  s.restarts = r.pod<index_t>();
  s.op_applies = r.pod<index_t>();
  s.passed = r.pod<bool>();
  return s;
}

template <class T>
void put(Writer& w, const driver::SliceReply<T>& s) {
  put(w, s.error);
  w.pod(s.completed);
  put(w, s.report);
  w.vec(s.lambda);
  w.pod(s.x.rows());
  w.pod(s.x.cols());
  w.raw(s.x.data(), static_cast<std::uint64_t>(s.x.rows() * s.x.cols()));  // column-major, ld == rows
  w.vec(s.metrics);
  w.pod(s.restarts);
  w.pod(s.applies);
  w.pod(s.block_size);
  w.strs(s.warnings);
  w.pod(s.seconds);
  w.pod(s.rss_bytes);
}
template <class T>
driver::SliceReply<T> get_slice_reply(Reader& r) {
  driver::SliceReply<T> s;
  s.error = get_error(r);
  s.completed = r.pod<bool>();
  s.report = get_slice_report(r);
  s.lambda = r.vec<double>();
  const auto rows = r.pod<index_t>();
  const auto cols = r.pod<index_t>();
  if (rows < 0 || cols < 0) HS_THROW(Internal, "parallel message: bad slice vector shape");
  s.x.resize(rows, cols);
  r.raw(s.x.data(), static_cast<std::uint64_t>(rows * cols));
  s.metrics = r.vec<driver::PairMetrics>();
  s.restarts = r.pod<index_t>();
  s.applies = r.pod<index_t>();
  s.block_size = r.pod<index_t>();
  s.warnings = r.strs();
  s.seconds = r.pod<double>();
  s.rss_bytes = r.pod<double>();
  return s;
}

}  // namespace hs::parallel
