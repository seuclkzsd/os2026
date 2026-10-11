// The randomized exclusion check and its start scaling (include/hsolve/verify.hpp, "(1)"; the
// header derives the probability statement). Implementation outline:
//   1. B-orthonormalize X (column-equilibrated Cholesky-QR, repeated until max |X^H B X - I| <=
//      1e-12; skipped for an X that is orthonormal already) - lanczos_krylov_schur demands an
//      orthonormal deflation block, and the Weyl argument needs the B-orthogonal splitting [X Y].
//   2. Coupling e = ||Op X - X H11||_B and the spectrum of H11 = X^H B Op X (one block
//      application of Op to the m columns), unless the caller supplies both.
//   3. eps from the thresholds, s from delta, one cycle of (block) Lanczos with deflation = X from
//      a Gaussian start block, without restart and with a convergence tolerance that never stops
//      it early, so that the Ritz values are those of the whole Krylov space K_s.
//   4. rho = max |Ritz value|: pass iff rho < tau = 1/r'.
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/util.hpp"
#include "hsolve/verify.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"
#include "verify/gaussian.hpp"

namespace hs {

namespace {

using lapack::fint;

constexpr double kEps = std::numeric_limits<double>::epsilon();
/// Krylov tolerance of the check's run: a pair counts as converged only with an estimate below
/// 1e-300 |theta|, i.e. never before the planned steps unless the space is exactly invariant.
constexpr double kNeverConverged = 1e-300;
/// X counts as B-orthonormal when max |X^H B X - I| <= kOrthoTol (lanczos_krylov_schur accepts
/// 1e-8; the two CGS2 passes against X then leave O(kOrthoTol^2) components in span X).
constexpr double kOrthoTol = 1e-12;
/// Cholesky-QR passes before giving up on orthonormalizing X.
constexpr int kMaxOrthoPasses = 3;
/// X is rejected as numerically rank deficient when the Cholesky factor of the equilibrated Gram
/// matrix D^{-1/2} X^H B X D^{-1/2} (D = its diagonal, unit B-norm columns) has min |R_ii| <
/// kRankTol max |R_ii| (cond_B(X D^{-1/2}) above ~1e6: span X is then not well defined). The
/// equilibration keeps the test about the conditioning of the span, not the column scaling.
constexpr double kRankTol = 1e-6;
/// The seed of the Gaussian start block is decorrelated from the solver's own random streams
/// (lanczos_krylov_schur draws replacement vectors from options.seed).
constexpr std::uint64_t kStartSeedMix = 0x9E3779B97F4A7C15ULL;

template <class T>
void apply_m(LinearOperator<T>* b, const DenseMatrix<T>& x, DenseMatrix<T>& bx) {
  bx.resize(x.rows(), x.cols());
  if (x.cols() == 0) return;
  if (b) {
    b->apply(x.data(), x.rows(), bx.data(), bx.rows(), x.cols());
  } else {
    std::copy(x.data(), x.data() + x.rows() * x.cols(), bx.data());
  }
}

/// V^H W for two n x k blocks, made exactly Hermitian (both are M-self-adjoint products here).
template <class T>
DenseMatrix<T> hermitian_gram(const DenseMatrix<T>& v, const DenseMatrix<T>& w) {
  const index_t k = v.cols();
  DenseMatrix<T> g(k, k);
  if (k == 0) return g;
  linalg::gram(v.rows(), k, v.data(), v.rows(), w.data(), w.rows(), g.data(), k);
  for (index_t j = 0; j < k; ++j) {
    g(j, j) = T(std::real(g(j, j)));
    for (index_t i = 0; i < j; ++i) {
      const T avg = T(0.5) * (g(i, j) + conj_if(g(j, i)));
      g(i, j) = avg;
      g(j, i) = conj_if(avg);
    }
  }
  return g;
}

/// Eigenvalues (ascending) of a Hermitian k x k matrix (copied).
template <class T>
std::vector<double> eigenvalues(DenseMatrix<T> h) {
  const index_t k = h.rows();
  std::vector<double> w(static_cast<std::size_t>(k));
  if (k > 0) lapack::heev('N', 'L', static_cast<fint>(k), h.data(), static_cast<fint>(k), w.data());
  return w;
}

template <class T>
bool all_finite(const DenseMatrix<T>& a) {
  for (index_t i = 0; i < a.rows() * a.cols(); ++i)
    if (!std::isfinite(std::abs(a.data()[i]))) return false;
  return true;
}

/// x <- x D^{-1/2} R^{-1} (column-equilibrated Cholesky-QR in the M inner product, D = diag(x^H M
/// x)) until max |x^H M x - I| <= kOrthoTol; mx = M x on exit. Throws InvalidInput for a
/// non-finite or numerically rank-deficient x.
template <class T>
void m_orthonormalize(LinearOperator<T>* m, DenseMatrix<T>& x, DenseMatrix<T>& mx) {
  const index_t n = x.rows(), k = x.cols();
  std::vector<double> scale(static_cast<std::size_t>(k));
  for (int pass = 0;; ++pass) {
    apply_m(m, x, mx);
    DenseMatrix<T> g = hermitian_gram(x, mx);
    double dev = 0.0;
    for (index_t j = 0; j < k; ++j)
      for (index_t i = 0; i < k; ++i) {
        const double e = std::abs(g(i, j) - T(i == j ? 1.0 : 0.0));
        if (std::isnan(e) || e > dev) dev = e;
      }
    if (!std::isfinite(dev))
      HS_THROW(InvalidInput, "exclusion_check: x (or b_inner applied to it) is not finite");
    if (dev <= kOrthoTol) return;
    if (pass == kMaxOrthoPasses)
      HS_THROW(InvalidInput, "exclusion_check: x could not be B-orthonormalized (max |X^H B X - I| = "
                                 << dev << " after " << pass << " Cholesky-QR passes)");
    // Equilibrate: unit M-norm columns, so that the rank test below measures the conditioning of
    // the span and not the relative column scaling (Cholesky-QR itself is scaling invariant).
    for (index_t j = 0; j < k; ++j) {
      const double gjj = std::real(g(j, j));
      if (!(gjj > 0.0))
        HS_THROW(InvalidInput, "exclusion_check: x is numerically rank deficient in the B inner product (column "
                                   << j << " has B-norm^2 " << gjj << ")");
      scale[j] = 1.0 / std::sqrt(gjj);
    }
    for (index_t j = 0; j < k; ++j)
      for (index_t i = 0; i < k; ++i) g(i, j) *= T(scale[i] * scale[j]);
    const fint info = lapack::potrf('U', static_cast<fint>(k), g.data(), static_cast<fint>(k));
    double dmin = std::numeric_limits<double>::infinity(), dmax = 0.0;
    if (info == 0)
      for (index_t i = 0; i < k; ++i) {
        dmin = std::min(dmin, std::abs(g(i, i)));
        dmax = std::max(dmax, std::abs(g(i, i)));
      }
    if (info != 0 || !(dmin >= kRankTol * dmax))
      HS_THROW(InvalidInput, "exclusion_check: x is numerically rank deficient in the B inner product "
                             "(Cholesky of the equilibrated X^H B X: info "
                                 << info << ", diagonal ratio " << (info == 0 ? dmin / dmax : 0.0) << ")");
    for (index_t j = 0; j < k; ++j) {
      T* col = x.col(j);
      for (index_t i = 0; i < n; ++i) col[i] *= T(scale[j]);
    }
    lapack::trsm('R', 'U', 'N', 'N', static_cast<fint>(n), static_cast<fint>(k), T(1), g.data(),
                 static_cast<fint>(k), x.data(), static_cast<fint>(n));
  }
}

void validate(index_t n, const ExclusionOptions& opt, index_t x_rows, index_t m, bool has_b,
              index_t b_size, double sigma, double radius) {
  if (n < 1) HS_THROW(InvalidInput, "exclusion_check: empty operator");
  if (has_b && b_size != n)
    HS_THROW(InvalidInput, "exclusion_check: b_inner has order " << b_size << ", operator " << n);
  if (m > 0 && x_rows != n)
    HS_THROW(InvalidInput, "exclusion_check: x has " << x_rows << " rows, operator order " << n);
  if (m >= n) HS_THROW(InvalidInput, "exclusion_check: x has " << m << " columns, nothing left of order " << n);
  if (!std::isfinite(sigma)) HS_THROW(InvalidInput, "exclusion_check: sigma must be finite");
  if (!(radius > 0.0) || !std::isfinite(radius))
    HS_THROW(InvalidInput, "exclusion_check: radius must be positive and finite, got " << radius);
  if (!(opt.pass_radius > radius) || !std::isfinite(opt.pass_radius))
    HS_THROW(InvalidInput, "exclusion_check: pass_radius must be finite and exceed radius = " << radius << ", got "
                                                                                               << opt.pass_radius);
  if (!(opt.delta > 0.0 && opt.delta < 1.0))
    HS_THROW(InvalidInput, "exclusion_check: delta must lie in (0, 1), got " << opt.delta);
  if (opt.block_size < 1) HS_THROW(InvalidInput, "exclusion_check: block_size must be >= 1");
  if (has_b && (!(opt.b_condition >= 1.0) || !std::isfinite(opt.b_condition)))
    HS_THROW(InvalidInput, "exclusion_check: with b_inner, b_condition must be a finite bound >= 1 on cond(S B S) "
                           "(see start_scaling_for), got "
                               << opt.b_condition);
  if (!has_b && opt.b_condition != 0.0 && (!(opt.b_condition >= 1.0) || !std::isfinite(opt.b_condition)))
    HS_THROW(InvalidInput, "exclusion_check: without b_inner, b_condition must be 0 (unset) or a finite bound >= 1, got "
                               << opt.b_condition);
  if (!(opt.rounding_margin >= 0.0 && opt.rounding_margin < 0.5))
    HS_THROW(InvalidInput, "exclusion_check: rounding_margin must lie in [0, 0.5), got " << opt.rounding_margin);
  if (!std::isfinite(opt.coupling_bound))
    HS_THROW(InvalidInput, "exclusion_check: coupling_bound must be finite (negative = compute)");
  if (opt.coupling_bound >= 0.0 && m > 0 &&
      (!opt.deflated_theta || static_cast<index_t>(opt.deflated_theta->size()) != m))
    HS_THROW(InvalidInput, "exclusion_check: coupling_bound >= 0 needs deflated_theta with one value per column of x");
  if (opt.deflated_theta && opt.coupling_bound >= 0.0)
    for (double t : *opt.deflated_theta)
      if (!std::isfinite(t)) HS_THROW(InvalidInput, "exclusion_check: deflated_theta must be finite");
  if (opt.max_block_steps < 0 || opt.max_new_pairs < 1)
    HS_THROW(InvalidInput, "exclusion_check: need max_block_steps >= 0 and max_new_pairs >= 1");
  if (opt.max_krylov_vectors < 0 || !(opt.max_memory_bytes >= 0.0) || !std::isfinite(opt.max_memory_bytes))
    HS_THROW(InvalidInput, "exclusion_check: need max_krylov_vectors >= 0 and a finite max_memory_bytes >= 0");
  if (!(opt.tol > 0.0) || !std::isfinite(opt.tol))
    HS_THROW(InvalidInput, "exclusion_check: tol must be positive and finite");
  if (opt.start_scaling) {
    if (static_cast<index_t>(opt.start_scaling->size()) != n)
      HS_THROW(InvalidInput, "exclusion_check: start_scaling has " << opt.start_scaling->size() << " entries, need "
                                                                   << n);
    for (double s : *opt.start_scaling)
      if (!(s > 0.0) || !std::isfinite(s))
        HS_THROW(InvalidInput, "exclusion_check: start_scaling entries must be positive and finite");
  }
}

std::string fmt(double x) {
  std::ostringstream os;
  os.precision(6);
  os << x;
  return os.str();
}

}  // namespace

const char* exclusion_status_name(ExclusionStatus s) {
  switch (s) {
    case ExclusionStatus::Passed: return "passed";
    case ExclusionStatus::Detected: return "detected";
    case ExclusionStatus::Inconclusive: return "inconclusive";
  }
  return "unknown";
}

template <class T>
ExclusionResult<T> exclusion_check(LinearOperator<T>& op, LinearOperator<T>* b_inner, const DenseMatrix<T>& x,
                                   double sigma, double radius, const ExclusionOptions& opt) {
  const index_t n = op.size();
  const index_t m = x.cols();
  validate(n, opt, x.rows(), m, b_inner != nullptr, b_inner ? b_inner->size() : 0, sigma, radius);
  const index_t p = opt.block_size;

  ExclusionResult<T> r;
  r.sigma = sigma;
  r.radius = radius;
  r.pass_radius = opt.pass_radius;
  // kappa >= cond(S B S) of the statement: the caller's bound with b_inner; for B = I computed
  // exactly, cond(S^2) = (max s / min s)^2 (inf when it overflows: no statement).
  if (b_inner) {
    r.kappa = opt.b_condition;
    r.rigorous = opt.b_condition_rigorous;
  } else {
    r.kappa = 1.0;
    if (opt.start_scaling) {
      const auto [lo, hi] = std::minmax_element(opt.start_scaling->begin(), opt.start_scaling->end());
      const double ratio = *hi / *lo;
      r.kappa = ratio * ratio;
    }
    r.kappa = std::max(r.kappa, opt.b_condition);
    r.rigorous = true;
  }
  r.deflated = m;
  r.dimension = n - m;
  r.block_size = p;
  const index_t op0 = op.applications();
  const index_t b0 = b_inner ? b_inner->applications() : 0;
  auto count_applications = [&]() {
    r.op_applications = op.applications() - op0;
    r.b_applications = b_inner ? b_inner->applications() - b0 : 0;
  };

  // ---- 1-2. B-orthonormal X, coupling e and the spectrum of H11.
  Timer setup;
  DenseMatrix<T> xq, bxq;
  std::vector<double> mu;
  double e = 0.0;
  if (m > 0) {
    xq = x;
    m_orthonormalize(b_inner, xq, bxq);
    if (opt.coupling_bound >= 0.0) {
      e = opt.coupling_bound;
      mu = *opt.deflated_theta;
    } else {
      DenseMatrix<T> y(n, m);
      op.apply(xq.data(), n, y.data(), n, m);
      if (!all_finite(y)) HS_THROW(InvalidInput, "exclusion_check: the operator returned Inf/NaN on x");
      const DenseMatrix<T> h = hermitian_gram(bxq, y);  // X^H B Op X
      mu = eigenvalues(h);
      DenseMatrix<T> xh(n, m);
      linalg::multiply(n, m, m, xq.data(), n, h.data(), m, xh.data(), n);
      for (index_t i = 0; i < n * m; ++i) y.data()[i] -= xh.data()[i];  // R = Op X - X H11
      DenseMatrix<T> by;
      apply_m(b_inner, y, by);
      const std::vector<double> g = eigenvalues(hermitian_gram(y, by));  // spectrum of R^H B R
      double mu_max = 0.0;
      for (double v : mu) mu_max = std::max(mu_max, std::abs(v));
      // Rounding guard: R is a difference of O(||H11||) quantities, each with relative error O(u).
      e = std::sqrt(std::max(g.empty() ? 0.0 : g.back(), 0.0)) + 16.0 * static_cast<double>(m + 1) * kEps * mu_max;
    }
  }
  std::sort(mu.begin(), mu.end(), [](double a, double b) { return std::abs(a) > std::abs(b); });
  r.deflated_theta = mu;
  r.coupling = e;
  const double theta_c = 1.0 / radius - e;
  r.claim_threshold = theta_c * (1.0 - opt.rounding_margin);
  r.pass_threshold = 1.0 / opt.pass_radius;
  r.epsilon = r.claim_threshold > 0.0 ? 1.0 - r.pass_threshold / r.claim_threshold : -1.0;
  for (double v : mu) {
    if (std::abs(v) >= theta_c) ++r.found_inside;
    if (std::abs(v) >= 1.0 / radius + e) ++r.certified_inside;
  }
  r.setup_s = setup.seconds();
  count_applications();

  if (!(r.epsilon > 0.0)) {
    r.status = ExclusionStatus::Inconclusive;
    r.failure_probability = 1.0;
    r.statement = "no statement: pass threshold 1/r' = " + fmt(r.pass_threshold) +
                  " is not below the claim threshold (1/r - e)(1 - margin) = " + fmt(r.claim_threshold) +
                  " (coupling e = " + fmt(e) + "); choose a larger pass_radius or a more accurate x";
    return r;
  }
  if (!std::isfinite(r.kappa)) {
    r.status = ExclusionStatus::Inconclusive;
    r.failure_probability = 1.0;
    r.statement = "no statement: the start scaling's condition (max s / min s)^2 overflows (kappa = inf)";
    return r;
  }

  // ---- 3. Plan s from delta, check it against the resource budget, and run one Lanczos cycle on
  // the deflated operator.
  const bool cplx_scalars = is_complex_v<T>;
  index_t s = exclusion_block_steps(r.epsilon, opt.delta, r.dimension, p, r.kappa, cplx_scalars);
  if (opt.max_block_steps > 0) s = std::min(s, opt.max_block_steps);
  r.block_steps_planned = s;
  const ExclusionRunSize plan = exclusion_run_size(n, m, s, p, opt.max_new_pairs, cplx_scalars);
  r.krylov_vectors_planned = plan.krylov_vectors;
  r.memory_bytes_planned = plan.bytes;
  if ((opt.max_krylov_vectors > 0 && plan.krylov_vectors > opt.max_krylov_vectors) ||
      (opt.max_memory_bytes > 0.0 && plan.bytes > opt.max_memory_bytes)) {
    r.over_budget = true;
    r.status = ExclusionStatus::Inconclusive;
    r.failure_probability = 1.0;
    std::ostringstream st;
    st.precision(6);
    st << "no statement: the plan (s = " << s << " block steps of p = " << p << ", " << plan.krylov_vectors
       << " Krylov vectors" << (plan.covers ? " covering the complement" : "") << ", " << plan.bytes
       << " bytes; eps = " << r.epsilon << ", kappa = " << r.kappa << ") exceeds the resource budget ("
       << opt.max_krylov_vectors << " vectors, " << opt.max_memory_bytes
       << " bytes; 0 = no limit): nothing was run, fall back to exact counts";
    r.statement = st.str();
    return r;
  }
  const index_t n_free = r.dimension;
  // Basis size: s blocks; the single-vector solver needs ncv > nev (or the whole free space).
  index_t ncv = std::max<index_t>(s * p, p == 1 ? 2 : 1);
  const index_t nev_req = opt.max_new_pairs;
  index_t nev = std::min(nev_req, n_free);
  if (ncv < n_free) nev = std::min(nev, std::max<index_t>(1, ncv - 1));

  DenseMatrix<T> start(n, p);
  for (index_t c = 0; c < p; ++c) {
    verify_detail::fill_gaussian(start.col(c), n, opt.seed ^ kStartSeedMix, static_cast<std::uint64_t>(c));
    if (opt.start_scaling)
      for (index_t i = 0; i < n; ++i) start(i, c) *= (*opt.start_scaling)[static_cast<std::size_t>(i)];
  }
  KrylovSchurOptions<T> ko;
  ko.nev = nev;
  ko.ncv = ncv;
  ko.max_restarts = 0;
  ko.tol = kNeverConverged;
  ko.seed = opt.seed;
  ko.initial = &start;
  ko.deflation = m > 0 ? &xq : nullptr;
  ko.block_size = p;
  Timer krylov;
  HermitianEigResult<T> res = lanczos_krylov_schur<T>(op, b_inner, Which::LargestMagnitude, ko);
  r.krylov_s = krylov.seconds();
  count_applications();

  r.ritz_all = res.ritz_all;
  for (double t : res.ritz_all) r.max_ritz = std::max(r.max_ritz, std::abs(t));
  r.exact = static_cast<index_t>(res.ritz_all.size()) >= n_free;
  r.block_steps = (p == 1 || r.exact) ? res.op_applies : res.op_applies / p;
  if (r.exact) {
    r.failure_probability = 0.0;
    r.rigorous = true;  // no probability, so no kappa involved
  } else {
    // Block mode (no fallback): op_applies / p block steps, the Krylov space K_s of the start block.
    r.failure_probability =
        exclusion_failure_bound(r.epsilon, std::max<index_t>(r.block_steps, 1), n_free, p, r.kappa, cplx_scalars);
  }

  // ---- 4. Verdict and candidate pairs.
  for (double t : res.ritz_all) {
    if (std::abs(t) >= r.pass_threshold) ++r.new_count;
    if (std::abs(t) >= 1.0 / radius) ++r.new_inside;
  }
  std::vector<index_t> keep;
  for (index_t i = 0; i < static_cast<index_t>(res.theta.size()); ++i)
    if (std::abs(res.theta[i]) >= r.pass_threshold) keep.push_back(i);
  r.new_vectors.resize(n, static_cast<index_t>(keep.size()));
  for (std::size_t c = 0; c < keep.size(); ++c) {
    const index_t i = keep[c];
    const double th = res.theta[i];
    r.new_theta.push_back(th);
    r.new_lambda.push_back(sigma + 1.0 / th);
    r.new_estimate.push_back(res.estimate[i]);
    r.new_converged.push_back(res.estimate[i] <= opt.tol * std::abs(th) ? 1 : 0);
    std::copy(res.vectors.col(i), res.vectors.col(i) + n, r.new_vectors.col(static_cast<index_t>(c)));
  }

  std::ostringstream st;
  st.precision(6);
  if (r.max_ritz >= r.pass_threshold) {
    r.status = ExclusionStatus::Detected;
    st << r.new_count << " Ritz value(s) with |theta| >= 1/r' = " << r.pass_threshold << " on the complement of x ("
       << r.new_inside << " within r): eigenvalue(s) missing within r' = " << opt.pass_radius << " of sigma = " << sigma;
  } else if (r.exact || r.failure_probability <= opt.delta) {
    r.status = ExclusionStatus::Passed;
    st << "P(pass and more than " << r.found_inside << " eigenvalues of the pencil within distance r = " << radius
       << " of sigma = " << sigma << ") <= " << r.failure_probability;
    if (r.exact)
      st << " (exact: the Krylov space spans the whole complement)";
    else
      st << " (Gaussian start block, p = " << p << ", s = " << r.block_steps << " block steps, N = " << n_free
         << ", kappa = " << r.kappa << ", eps = " << r.epsilon << ", coupling e = " << e << ", relative margin "
         << opt.rounding_margin << "; ideal Gaussian start, exact arithmetic on the computed operator)";
    if (!r.exact && !r.rigorous)
      st << "; NOT RIGOROUS: kappa is an estimate assumed to bound cond(S B S), not proven - a screening "
            "result, not a certificate";
  } else {
    r.status = ExclusionStatus::Inconclusive;
    st << "no Ritz value reached 1/r', but the " << r.block_steps << " block steps only give failure probability "
       << r.failure_probability << " > delta = " << opt.delta << " (step cap)";
  }
  r.statement = st.str();
  return r;
}

void set_start_scaling(ExclusionOptions& opt, const StartScaling& sc) {
  opt.start_scaling = sc.scaling.empty() ? nullptr : &sc.scaling;
  opt.b_condition = sc.kappa;
  opt.b_condition_rigorous = sc.rigorous;
}

template <class T>
StartScaling start_scaling_for(const CsrMatrix<T>* b, double safety, bool estimate) {
  StartScaling s;
  if (!b) {
    s.note = "B = I: kappa = 1";
    return s;
  }
  if (!b->square()) HS_THROW(InvalidInput, "start_scaling_for: B must be square");
  if (!(safety >= 1.0) || !std::isfinite(safety))
    HS_THROW(InvalidInput, "start_scaling_for: safety must be finite and >= 1");
  const index_t n = b->nrows;
  std::vector<double> diag(static_cast<std::size_t>(n), 0.0);
  bool diagonal = true;
  for (index_t i = 0; i < n; ++i)
    for (index_t q = b->rowptr[i]; q < b->rowptr[i + 1]; ++q) {
      if (b->colind[q] == i) {
        diag[i] = std::real(b->values[q]);
      } else if (b->values[q] != T(0)) {
        diagonal = false;
      }
    }
  for (index_t i = 0; i < n; ++i)
    if (!(diag[i] > 0.0) || !std::isfinite(diag[i]))
      HS_THROW(InvalidInput, "start_scaling_for: B has a non-positive diagonal entry at row " << i);
  s.scaling.resize(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) s.scaling[i] = 1.0 / std::sqrt(diag[i]);
  if (diagonal) {
    s.note = "diagonal B, S = B^{-1/2}: kappa = 1";
    return s;
  }
  // Gershgorin discs of S B S and of the similar D^{-1} B (both have unit diagonal and the
  // spectrum of S B S); each gives an enclosing interval, so their intersection does too. The
  // second keeps a diagonally dominant B's dominance, which the symmetric scaling may lose.
  double lo_sym = std::numeric_limits<double>::infinity(), hi_sym = 0.0;
  double lo_row = std::numeric_limits<double>::infinity(), hi_row = 0.0;
  for (index_t i = 0; i < n; ++i) {
    double rad_sym = 0.0, rad_row = 0.0;
    for (index_t q = b->rowptr[i]; q < b->rowptr[i + 1]; ++q)
      if (b->colind[q] != i) {
        rad_sym += std::abs(b->values[q]) * s.scaling[i] * s.scaling[b->colind[q]];
        rad_row += std::abs(b->values[q]) / diag[i];
      }
    lo_sym = std::min(lo_sym, 1.0 - rad_sym);
    hi_sym = std::max(hi_sym, 1.0 + rad_sym);
    lo_row = std::min(lo_row, 1.0 - rad_row);
    hi_row = std::max(hi_row, 1.0 + rad_row);
  }
  const double lo = std::max(lo_sym, lo_row), hi = std::min(hi_sym, hi_row);
  if (lo > 0.0) {
    s.kappa = hi / lo;
    s.note = "Jacobi scaling, diagonally dominant: Gershgorin kappa bound";
    return s;
  }
  if (!estimate) {
    s.kappa = std::numeric_limits<double>::infinity();
    s.rigorous = false;
    s.note = "Jacobi scaling, not diagonally dominant: no proven kappa (not estimated)";
    return s;
  }
  // Estimate: smallest Ritz value of a short Lanczos run on S B S (an upper bound on lambda_min).
  class Scaled final : public LinearOperator<T> {
   public:
    Scaled(const CsrMatrix<T>& b, const std::vector<double>& d) : b_(b), d_(d) {}
    index_t size() const override { return b_.nrows; }

   protected:
    void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override {
      const index_t nn = b_.nrows;
      tmp_.resize(static_cast<std::size_t>(nn * p));
      for (index_t c = 0; c < p; ++c)
        for (index_t i = 0; i < nn; ++i) tmp_[i + c * nn] = x[i + c * ldx] * d_[i];
      spmm(b_, tmp_.data(), nn, y, ldy, p);
      for (index_t c = 0; c < p; ++c)
        for (index_t i = 0; i < nn; ++i) y[i + c * ldy] *= d_[i];
    }

   private:
    const CsrMatrix<T>& b_;
    const std::vector<double>& d_;
    std::vector<T> tmp_;
  };
  Scaled sbs(*b, s.scaling);
  KrylovSchurOptions<T> ko;
  ko.nev = 1;
  ko.ncv = std::min<index_t>(n, 40);
  ko.max_restarts = 0;
  ko.tol = kNeverConverged;
  const HermitianEigResult<T> res = lanczos_krylov_schur<T>(sbs, nullptr, Which::SmallestAlgebraic, ko);
  const double lmin = res.ritz_all.empty() ? 0.0 : res.ritz_all.front();
  if (!(lmin > 0.0)) HS_THROW(InvalidInput, "start_scaling_for: B does not look positive definite (Ritz value " << lmin << ")");
  s.kappa = safety * hi / lmin;
  s.rigorous = false;
  s.note = "Jacobi scaling, not diagonally dominant: kappa = safety x Gershgorin upper / Lanczos lower estimate "
           "(an assumption, not a bound)";
  return s;
}

#define HS_INSTANTIATE_EXCLUSION(T)                                                                          \
  template ExclusionResult<T> exclusion_check<T>(LinearOperator<T>&, LinearOperator<T>*, const DenseMatrix<T>&, \
                                                 double, double, const ExclusionOptions&);                    \
  template StartScaling start_scaling_for<T>(const CsrMatrix<T>*, double, bool);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_EXCLUSION)
#undef HS_INSTANTIATE_EXCLUSION

}  // namespace hs
