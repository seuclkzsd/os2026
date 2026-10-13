// Single-precision Sylvester counts that certify a window (include/hsolve/verify.hpp, "(2)").
//
// eta(s) = backward_factor * u_s * ||K(s)||_1 / min b_ii bounds how far the eigenvalues of the
// pencil move under the backward error of a single-precision count at s, ||K(s)||_1 <= ||A||_1 +
// |s| ||B||_1 (a rigorous upper bound of the 2-norm of the Hermitian K). Complex T is counted
// through the real embedding, whose 1-norm is at most sqrt(2) ||K||_1 and whose eigenvalues are
// those of K twice: the same argument counts each eigenvalue twice (a valid complex count has an
// even embedding count, see SparseFactorization::factorize()).
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "hsolve/error.hpp"
#include "hsolve/util.hpp"
#include "hsolve/verify.hpp"

namespace hs {

namespace {

/// Unit roundoff of IEEE single precision, 2^-24.
constexpr double kUnitSingle = 5.9604644775390625e-8;

/// min and max diagonal entry of a diagonal B; false when B has a nonzero off-diagonal entry.
template <class T>
bool diagonal_range(const CsrMatrix<T>& b, double& dmin, double& dmax) {
  dmin = std::numeric_limits<double>::infinity();
  dmax = 0.0;
  for (index_t i = 0; i < b.nrows; ++i) {
    double d = 0.0;
    for (index_t q = b.rowptr[i]; q < b.rowptr[i + 1]; ++q) {
      if (b.colind[q] == i) {
        d = std::real(b.values[q]);
      } else if (b.values[q] != T(0)) {
        return false;
      }
    }
    dmin = std::min(dmin, d);
    dmax = std::max(dmax, d);
  }
  return true;
}

}  // namespace

template <class T>
SingleCountValidity single_count_validity(const ShiftedPencil<T>& pencil, const SingleWindow& w,
                                          const SingleCountRule& rule) {
  if (!(w.lower < w.upper) || !std::isfinite(w.lower) || !std::isfinite(w.upper))
    HS_THROW(InvalidInput, "single_count_validity: need finite lower < upper, got [" << w.lower << ", " << w.upper << "]");
  if (!(w.lower < w.wanted_lower && w.wanted_lower <= w.wanted_upper && w.wanted_upper < w.upper))
    HS_THROW(InvalidInput, "single_count_validity: the wanted region [" << w.wanted_lower << ", " << w.wanted_upper
                                                                       << "] must lie inside (" << w.lower << ", "
                                                                       << w.upper << ")");
  if (!w.found_error.empty() && w.found_error.size() != w.found.size())
    HS_THROW(InvalidInput, "single_count_validity: found_error must be empty or match found");
  if (!(rule.backward_factor > 0.0) || !std::isfinite(rule.backward_factor))
    HS_THROW(InvalidInput, "single_count_validity: backward_factor must be positive and finite");
  if (!std::isfinite(rule.a_norm_1))
    HS_THROW(InvalidInput, "single_count_validity: a_norm_1 must be finite (< 0: computed)");

  if (!(rule.b_min_proven >= 0.0) || !std::isfinite(rule.b_min_proven))
    HS_THROW(InvalidInput, "single_count_validity: b_min_proven must be finite and >= 0");
  SingleCountValidity v;
  double bmin = 1.0, bmax = 1.0;
  if (const CsrMatrix<T>* b = pencil.b()) {
    if (!diagonal_range(*b, bmin, bmax)) {
      if (!(rule.b_min_proven > 0.0)) {
        v.reason = "B is not diagonal and no lower bound on lambda_min(B) is proven: single counts certify nothing";
        return v;
      }
      // A general HPD B with a proven lambda_min(B) >= b_min_proven (SingleCountRule).
      bmin = rule.b_min_proven;
      bmax = norm_one(*b);
    } else if (!(bmin > 0.0)) {
      v.reason = "diagonal B is not positive definite";
      return v;
    }
  }
  v.applicable = true;
  v.b_min = bmin;
  const double norm_a = rule.a_norm_1 >= 0.0 ? rule.a_norm_1 : norm_one(pencil.a());
  const double emb = is_complex_v<T> ? std::sqrt(2.0) : 1.0;
  auto eta = [&](double s) {
    return rule.backward_factor * kUnitSingle * emb * (norm_a + std::abs(s) * bmax) / bmin;
  };
  v.eta_lower = eta(w.lower);
  v.eta_upper = eta(w.upper);
  const double lo = w.lower + v.eta_lower, hi = w.upper - v.eta_upper;
  std::ostringstream why;
  why.precision(6);
  if (!(lo <= w.wanted_lower && w.wanted_upper < hi)) {
    why << "window [" << w.lower << ", " << w.upper << "] shrunk by eta = (" << v.eta_lower << ", " << v.eta_upper
        << ") no longer holds the wanted region [" << w.wanted_lower << ", " << w.wanted_upper << "]";
    v.reason = why.str();
    return v;
  }
  for (std::size_t i = 0; i < w.found.size(); ++i) {
    const double err = w.found_error.empty() ? 0.0 : w.found_error[i];
    if (!(err >= 0.0) || !std::isfinite(w.found[i]))
      HS_THROW(InvalidInput, "single_count_validity: found eigenvalues must be finite with non-negative error bars");
    if (!(w.found[i] - err >= lo && w.found[i] + err < hi)) {
      why << "found eigenvalue " << w.found[i] << " (+- " << err << ") lies within eta of a window end";
      v.reason = why.str();
      return v;
    }
  }
  v.margins_ok = true;
  return v;
}

template <class T>
SingleCertificate single_precision_certificate(InertiaCounter<T>& counter, const ShiftedPencil<T>& pencil,
                                               const SingleWindow& w, const SingleCountRule& rule) {
  if (counter.order() != pencil.order())
    HS_THROW(InvalidInput, "single_precision_certificate: counter of order " << counter.order() << ", pencil "
                                                                            << pencil.order());
  SingleCertificate c;
  c.validity = single_count_validity(pencil, w, rule);
  if (!c.validity.valid()) return c;
  Timer t;
  const auto counts = counter.count_pair(1.0, -w.lower, 1.0, -w.upper, true, FactorPrecision::Single);
  c.seconds = t.seconds();
  c.lower = counts.first;
  c.upper = counts.second;
  if (!c.lower.valid || !c.upper.valid) return c;
  c.count = c.upper.neg - c.lower.neg;
  c.certified = c.count == static_cast<index_t>(w.found.size());
  return c;
}

#define HS_INSTANTIATE_SINGLE_COUNT(T)                                                                        \
  template SingleCountValidity single_count_validity<T>(const ShiftedPencil<T>&, const SingleWindow&,         \
                                                        const SingleCountRule&);                              \
  template SingleCertificate single_precision_certificate<T>(InertiaCounter<T>&, const ShiftedPencil<T>&,     \
                                                             const SingleWindow&, const SingleCountRule&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SINGLE_COUNT)
#undef HS_INSTANTIATE_SINGLE_COUNT

}  // namespace hs
