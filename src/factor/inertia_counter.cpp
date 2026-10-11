// InertiaCounter (include/hsolve/factor.hpp): inertia-only MUMPS instances (ICNTL(31)=1) that
// reuse the pivot order of an analysed SparseFactorization (PERM_IN, ICNTL(7)=1). The design
// choices and their measurements are in the class comment; the measurement program is
// bench/certpar_inertia.cpp.
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "factor/mumps_factorization.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/util.hpp"

namespace hs {

template <class T>
struct InertiaCounter<T>::Impl {
  const ShiftedPencil<T>* pencil = nullptr;
  FactorOptions options;          ///< resolved: fixed thread count, in-core, no BLR
  std::vector<int> pivot_order;   ///< the source's SYM_PERM (empty: options.ordering)
  std::unique_ptr<mumps::MumpsFactorization<T, double>> dbl;
  std::unique_ptr<mumps::MumpsFactorization<T, float>> sgl;  ///< created at the first single count
  InertiaCounterStats stats;

  // An inertia-only instance analysed with the source's pivot order at (alpha, beta).
  template <class U>
  std::unique_ptr<mumps::MumpsFactorization<T, U>> make_instance(double alpha, double beta) {
    auto f = std::make_unique<mumps::MumpsFactorization<T, U>>(options);
    f->set_discard_factors(true);
    f->set_pivot_order(pivot_order);
    f->analyze(*pencil, alpha, beta);
    ++stats.analyses;
    stats.analyze_s += f->stats().analyze_s;
    return f;
  }

  template <class U>
  Inertia count_with(mumps::MumpsFactorization<T, U>& f, double alpha, double beta,
                     bool detection) {
    const double before = f.stats().factorize_s;
    f.set_null_pivot_detection(detection);
    // A singular K is reported through inertia().valid == false, never thrown.
    static_cast<void>(f.factorize(alpha, beta));
    ++stats.counts;
    stats.count_s += f.stats().factorize_s - before;
    stats.max_mb_used = std::max(stats.max_mb_used, f.stats().factor_mb_used);
    stats.workspace_retries = dbl->stats().workspace_retries + (sgl ? sgl->stats().workspace_retries : 0);
    return f.inertia();
  }
};

template <class T>
InertiaCounter<T>::InertiaCounter(const SparseFactorization<T>& analysed_source,
                                  const FactorOptions& options)
    : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  if (const auto* d = dynamic_cast<const mumps::MumpsFactorization<T, double>*>(&analysed_source)) {
    m.pencil = d->pencil();
    m.pivot_order = d->pivot_order();
  } else if (const auto* s =
                 dynamic_cast<const mumps::MumpsFactorization<T, float>*>(&analysed_source)) {
    m.pencil = s->pencil();
    m.pivot_order = s->pivot_order();
  } else {
    HS_THROW(InvalidInput, "InertiaCounter: the source must come from make_mumps_factorization");
  }
  if (m.pencil == nullptr)
    HS_THROW(InvalidInput, "InertiaCounter: the source factorization has not been analysed");
  m.options = options;
  m.options.precision = FactorPrecision::Double;  // per count, see count()
  m.options.blr = false;
  m.options.blr_estimates = false;
  m.options.out_of_core = false;
  // L0-OpenMP factorizations must use the thread count of their analysis (INFO -58 otherwise).
  if (m.options.threads == 0) m.options.threads = max_threads();
  m.stats.reused_pivot_order = !m.pivot_order.empty();
  // Representative values: with a given pivot order MUMPS needs them for nothing that changes
  // the counts (no compressed 2x2 ordering; scaling ICNTL(8)=7 is recomputed per factorization).
  m.dbl = m.template make_instance<double>(1.0, 0.0);
}

template <class T>
InertiaCounter<T>::~InertiaCounter() = default;

template <class T>
Inertia InertiaCounter<T>::count(double alpha, double beta, bool null_pivot_detection,
                                 FactorPrecision precision) {
  if (!std::isfinite(alpha) || !std::isfinite(beta))
    HS_THROW(InvalidInput, "InertiaCounter::count: non-finite alpha=" << alpha << ", beta=" << beta);
  Impl& m = *impl_;
  if (precision == FactorPrecision::Single) {
    if (!m.sgl) m.sgl = m.template make_instance<float>(alpha, beta);
    const Inertia in = m.count_with(*m.sgl, alpha, beta, null_pivot_detection);
    ++m.stats.single_counts;
    return in;
  }
  return m.count_with(*m.dbl, alpha, beta, null_pivot_detection);
}

template <class T>
std::pair<Inertia, Inertia> InertiaCounter<T>::count_pair(double alpha1, double beta1, double alpha2,
                                                          double beta2, bool null_pivot_detection,
                                                          FactorPrecision precision) {
  Inertia first = count(alpha1, beta1, null_pivot_detection, precision);
  Inertia second = count(alpha2, beta2, null_pivot_detection, precision);
  return {first, second};
}

template <class T>
FactorEstimate InertiaCounter<T>::estimate() const {
  return impl_->dbl->estimate();
}

template <class T>
const InertiaCounterStats& InertiaCounter<T>::stats() const {
  return impl_->stats;
}

template <class T>
index_t InertiaCounter<T>::order() const {
  return impl_->pencil->order();
}

#define HS_INSTANTIATE_INERTIA_COUNTER(T) template class InertiaCounter<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_INERTIA_COUNTER)
#undef HS_INSTANTIATE_INERTIA_COUNTER

}  // namespace hs
