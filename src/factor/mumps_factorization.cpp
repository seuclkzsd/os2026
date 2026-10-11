// MUMPS-backed SparseFactorization (docs/DESIGN.md §3.4). See include/hsolve/factor.hpp for the
// contract and the MUMPS facts it relies on; the class is declared in mumps_factorization.hpp.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "factor/mumps_factorization.hpp"

#include "factor/mumps_api.hpp"
#include "factor/mumps_input.hpp"
#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/util.hpp"

namespace hs {

namespace {

constexpr int kMaxWorkspaceRetries = 3;
// Each workspace retry doubles ICNTL(14), but grows it by at least this many percentage points
// so that a tiny (or zero) initial relaxation still makes progress.
constexpr int kMinRelaxIncrease = 10;

// Returns `o` after checking every field MUMPS would otherwise misinterpret (InvalidInput).
const FactorOptions& validated(const FactorOptions& o) {
  if ((o.ordering < 0 || o.ordering > 7 || o.ordering == 1) && o.ordering != kOrderingNestedDissection)
    HS_THROW(InvalidInput, "FactorOptions.ordering must be an ICNTL(7) value 0,2,3,4,5,6,7 or "
                           "kOrderingNestedDissection ("
                               << kOrderingNestedDissection << "), got " << o.ordering
                               << " (1 = user permutation is not supported)");
  if (o.ordering_threads < 0)
    HS_THROW(InvalidInput, "FactorOptions.ordering_threads must be >= 0, got " << o.ordering_threads);
  static constexpr int kScalings[] = {-2, 0, 1, 3, 4, 7, 8, 77};
  if (std::find(std::begin(kScalings), std::end(kScalings), o.scaling) == std::end(kScalings))
    HS_THROW(InvalidInput, "FactorOptions.scaling (ICNTL(8)) must be one of -2,0,1,3,4,7,8,77, got "
                               << o.scaling);
  if (o.threads < 0)
    HS_THROW(InvalidInput, "FactorOptions.threads must be >= 0, got " << o.threads);
  if (o.mem_relax_pct < 0)
    HS_THROW(InvalidInput, "FactorOptions.mem_relax_pct must be >= 0, got " << o.mem_relax_pct);
  if (!(o.max_memory_mb >= 0.0) || !std::isfinite(o.max_memory_mb))
    HS_THROW(InvalidInput,
             "FactorOptions.max_memory_mb must be finite and >= 0, got " << o.max_memory_mb);
  if (!(o.null_pivot_tol > 0.0) || !(o.null_pivot_tol < 1.0))
    HS_THROW(InvalidInput,
             "FactorOptions.null_pivot_tol must lie in (0, 1), got " << o.null_pivot_tol);
  if (!(o.single_null_pivot_tol > 0.0) || !(o.single_null_pivot_tol < 1.0))
    HS_THROW(InvalidInput, "FactorOptions.single_null_pivot_tol must lie in (0, 1), got "
                               << o.single_null_pivot_tol);
  if ((o.blr || o.blr_estimates) && !(o.blr_tol > 0.0 && o.blr_tol < 1.0))
    HS_THROW(InvalidInput, "FactorOptions.blr_tol must lie in (0, 1), got " << o.blr_tol);
  if (o.ooc_dir.size() >= sizeof(DMUMPS_STRUC_C::ooc_tmpdir))
    HS_THROW(InvalidInput, "FactorOptions.ooc_dir is longer than MUMPS allows ("
                               << sizeof(DMUMPS_STRUC_C::ooc_tmpdir) - 1 << " characters)");
  if (o.block_analysis < -1 || o.block_analysis > 1)
    HS_THROW(InvalidInput, "FactorOptions.block_analysis must be -1, 0 or 1, got " << o.block_analysis);
  if (o.verbosity < 0)
    HS_THROW(InvalidInput, "FactorOptions.verbosity must be >= 0, got " << o.verbosity);
  return o;
}

// ICNTL(14) for the next workspace retry: doubled, at least kMinRelaxIncrease points more,
// saturating at INT_MAX.
int grown_relaxation(int pct) {
  const long long next = std::max(2LL * pct, static_cast<long long>(pct) + kMinRelaxIncrease);
  return static_cast<int>(std::min<long long>(next, INT_MAX));
}

// Throws the hs::Error that the contract assigns to a negative INFOG(1) of `phase`.
[[noreturn]] void throw_mumps_error(const char* phase, int info1, int info2, int retries = 0) {
  if (mumps::is_memory_error(info1))
    HS_THROW(OutOfMemoryBudget,
             "MUMPS " << phase << " ran out of memory: INFO(1)=" << info1 << ", INFO(2)=" << info2
                      << (info1 == -19 ? " (working memory above the ICNTL(23) cap)" : ""));
  if (retries > 0)
    HS_THROW(FactorizationFailed, "MUMPS " << phase << " failed: INFO(1)=" << info1
                                           << ", INFO(2)=" << info2 << " after " << retries
                                           << " workspace retries (ICNTL(14) doubled)");
  HS_THROW(FactorizationFailed,
           "MUMPS " << phase << " failed: INFO(1)=" << info1 << ", INFO(2)=" << info2);
}

}  // namespace

namespace mumps {

template <class T, class U>
MumpsFactorization<T, U>::MumpsFactorization(const FactorOptions& options)
    // options_ is declared (and validated) before the MUMPS instance is created.
    : options_(validated(options)), relax_pct_(options.mem_relax_pct) {
  configure_static_controls();
}

// Controls that never change during the object's life. ICNTL indices follow the MUMPS guide.
template <class T, class U>
void MumpsFactorization<T, U>::configure_static_controls() {
  // Output: MUMPS treats a stream <= 0 as "off", so verbosity > 0 can only use Fortran unit 6
  // (stdout); unit 0 (stderr) is indistinguishable from "suppressed".
  const int stream = options_.verbosity > 0 ? 6 : -1;
  mumps_.icntl(1) = stream;                                // errors
  mumps_.icntl(2) = options_.verbosity > 1 ? stream : -1;  // diagnostics and warnings
  mumps_.icntl(3) = stream;                                // global information
  mumps_.icntl(4) = std::min(options_.verbosity, 4);       // print level
  mumps_.icntl(5) = 0;                                     // assembled input
  mumps_.icntl(7) = mumps_ordering();
  mumps_.icntl(8) = options_.scaling;
  // INFOG(12) counts the negative pivots of fronts MUMPS factorizes itself; keep the root front
  // out of ScaLAPACK (a no-op for the sequential library, but it keeps the inertia exact).
  mumps_.icntl(13) = 1;
  mumps_.icntl(16) = options_.threads;  // 0 is resolved at analyze()
  mumps_.icntl(18) = 0;                 // centralized matrix on the host
  mumps_.icntl(20) = 0;                 // dense right-hand sides
  mumps_.icntl(21) = 0;                 // centralized solution, overwrites the right-hand sides
  mumps_.icntl(23) =
      options_.max_memory_mb > 0.0
          ? static_cast<int>(std::clamp(std::floor(options_.max_memory_mb), 1.0, double(INT_MAX)))
          : 0;
  // Null-pivot detection: needs ICNTL(24)=1 AND CNTL(3) > 0 (threshold relative to the norm of
  // the scaled/permuted matrix); with CNTL(3)=0 exactly singular matrices "factorize". The
  // threshold depends on the arithmetic: float rounding alone moves eigenvalues by ~1e-7*||K||,
  // so smumps needs a much larger one (calibration in factor.hpp).
  mumps_.icntl(24) = 1;
  mumps_.cntl(3) = static_cast<U>(std::is_same_v<U, float> ? options_.single_null_pivot_tol
                                                           : options_.null_pivot_tol);
  mumps_.cntl(7) = static_cast<U>(options_.blr_tol);
  // L0-OpenMP assembles subtrees in a thread-timing-dependent order, so results then differ in
  // the last bits from run to run; ICNTL(48)=0 keeps only node-level parallelism (reproducible).
  mumps_.icntl(48) = options_.deterministic ? 0 : 1;
  if (!options_.ooc_dir.empty()) {
    auto& dir = mumps_.raw().ooc_tmpdir;
    std::memset(dir, 0, sizeof(dir));
    std::memcpy(dir, options_.ooc_dir.data(), options_.ooc_dir.size());
  }
}

template <class T, class U>
void MumpsFactorization<T, U>::analyze(const ShiftedPencil<T>& pencil, double alpha, double beta) {
  if (pencil.order() <= 0) HS_THROW(InvalidInput, "SparseFactorization::analyze: empty pencil");
  if (!std::isfinite(alpha) || !std::isfinite(beta))
    HS_THROW(InvalidInput,
             "SparseFactorization::analyze: non-finite alpha=" << alpha << ", beta=" << beta);
  Timer timer;
  factorized_ = false;
  inertia_ = Inertia{};
  pencil_ = nullptr;
  input_ = std::make_unique<mumps::Input<T, U>>(pencil);
  input_->assemble(alpha, beta);  // representative values for matching-based 2x2 pivot detection

  auto& s = mumps_.raw();
  s.n = input_->order();
  s.nz = 0;  // the 64-bit NNZ field is authoritative
  s.nnz = input_->nnz();
  s.irn = input_->irn();
  s.jcn = input_->jcn();
  s.a = input_->values();
  // BLR must be enabled at analysis to be usable at factorization (and to obtain the BLR
  // memory estimates INFOG(36..39)); it can still be switched off for the factorization itself.
  blr_at_analysis_ = options_.blr || options_.blr_estimates;
  mumps_.icntl(35) = blr_at_analysis_ ? 2 : 0;
  mumps_.icntl(14) = relax_pct_;
  // With L0-OpenMP (ICNTL(48)=1) a factorization must run with the thread count of its analysis
  // (else INFO -58): threads = 0 is resolved now, so a caller that changes its OpenMP setting
  // later does not break the factorizations. MUMPS restores the caller's setting after each call.
  mumps_.icntl(16) = options_.threads > 0 ? options_.threads : max_threads();
  holds_factors_ = false;
  // ICNTL(31) is read at analysis only (inertia-only objects, see set_discard_factors()).
  discard_at_analysis_ = discard_next_;
  mumps_.icntl(31) = discard_at_analysis_ ? 1 : 0;
  // A given pivot order replaces the fill-reducing ordering (PERM_IN, ICNTL(7)=1); so does
  // hsolve's own ordering, computed here.
  pivot_order_out_.clear();
  std::vector<int> computed_order;
  if (pivot_order_in_.empty() && options_.ordering == kOrderingNestedDissection) {
    OrderingOptions oo;
    oo.threads = options_.ordering_threads > 0 ? options_.ordering_threads : mumps_.icntl(16);
    Timer ordering_timer;
    computed_order = compute_ordering(pencil, oo);
    stats_.ordering_s += ordering_timer.seconds();
  }
  std::vector<int>& given = pivot_order_in_.empty() ? computed_order : pivot_order_in_;
  if (!given.empty()) {
    if (static_cast<index_t>(given.size()) != input_->order())
      HS_THROW(InvalidInput, "SparseFactorization::analyze: pivot order of size "
                                 << given.size() << " for a MUMPS matrix of order " << input_->order());
    mumps_.icntl(7) = 1;
    s.perm_in = given.data();
  } else {
    mumps_.icntl(7) = mumps_ordering();
    s.perm_in = nullptr;
  }
  // Complex T with a MUMPS ordering: the embedding's pairs {i, n+i} as blocks (ICNTL(15)=1), so
  // that MUMPS orders the n-vertex graph of K (FactorOptions::block_analysis).
  bool blocks = false;
  if constexpr (kEmbeddingFactor == 2) {
    if (given.empty() && options_.block_analysis != 0)
      blocks = options_.block_analysis > 0 ||
               ordering::complex_coupling_fraction(pencil) >= ordering::kCompressedFraction;
  }
  if (blocks) {
    const int n = static_cast<int>(pencil.order());
    blkptr_.resize(static_cast<std::size_t>(n) + 1);
    blkvar_.resize(static_cast<std::size_t>(2 * n));
    for (int b = 0; b <= n; ++b) blkptr_[b] = 2 * b + 1;
    for (int b = 0; b < n; ++b) {
      blkvar_[2 * b] = b + 1;          // Re part of vertex b
      blkvar_[2 * b + 1] = n + b + 1;  // Im part
    }
    mumps_.icntl(15) = 1;
    s.nblk = n;
    s.blkptr = blkptr_.data();
    s.blkvar = blkvar_.data();
  } else {
    mumps_.icntl(15) = 0;
    s.nblk = 0;
    s.blkptr = nullptr;
    s.blkvar = nullptr;
  }
  stats_.block_analysis = blocks;

  const int info1 = mumps_.run(kAnalyze);
  s.perm_in = nullptr;
  stats_.analyze_s += timer.seconds();
  ++stats_.analyses;
  stats_.last_info1 = info1;
  stats_.last_info2 = mumps_.infog(2);
  if (info1 < 0) throw_mumps_error("analysis", info1, mumps_.infog(2));
  if (s.sym_perm) pivot_order_out_.assign(s.sym_perm, s.sym_perm + s.n);
  pencil_ = &pencil;
  stats_.estimate = estimate();
}

template <class T, class U>
FactorEstimate MumpsFactorization<T, U>::estimate() const {
  HS_ASSERT(pencil_ != nullptr, "SparseFactorization::estimate() before a successful analyze()");
  // Sequential MUMPS: the "most loaded process" statistics equal the totals.
  FactorEstimate e;
  e.factor_entries = mumps::decode_count(mumps_.infog(20));
  e.incore_mb = mumps_.infog(16);
  e.ooc_mb = mumps_.infog(26);
  e.blr_incore_mb = blr_at_analysis_ ? mumps_.infog(36) : -1.0;
  e.blr_ooc_mb = blr_at_analysis_ ? mumps_.infog(38) : -1.0;
  e.flops = mumps_.rinfog(1);
  return e;
}

template <class T, class U>
bool MumpsFactorization<T, U>::factorize(double alpha, double beta) {
  HS_ASSERT(pencil_ != nullptr, "SparseFactorization::factorize() before a successful analyze()");
  if (!std::isfinite(alpha) || !std::isfinite(beta))
    HS_THROW(InvalidInput,
             "SparseFactorization::factorize: non-finite alpha=" << alpha << ", beta=" << beta);
  Timer timer;
  ++stats_.factorizations;
  factorized_ = false;
  inertia_ = Inertia{};
  input_->assemble(alpha, beta);

  mumps_.icntl(22) = options_.out_of_core ? 1 : 0;
  mumps_.icntl(35) = options_.blr ? 2 : 0;
  mumps_.icntl(24) = null_pivot_detection_ ? 1 : 0;
  int info1 = 0, retries = 0;
  for (;;) {
    mumps_.icntl(14) = relax_pct_;
    info1 = mumps_.run(mumps::kFactorize);
    if (!mumps::is_workspace_error(info1) || retries == kMaxWorkspaceRetries) break;
    relax_pct_ = grown_relaxation(relax_pct_);
    ++retries;
    ++stats_.workspace_retries;
  }
  stats_.factorize_s += timer.seconds();
  stats_.last_info1 = info1;
  stats_.last_info2 = mumps_.infog(2);
  // INFO(1) = -10 is MUMPS' own "numerically singular" verdict (not expected with ICNTL(24)=1).
  if (info1 < 0 && info1 != -10)
    throw_mumps_error("factorization", info1, mumps_.infog(2), retries);

  holds_factors_ = !discard_at_analysis_;
  factorized_ = !read_factorization_info(/*mumps_singular=*/info1 == -10);
  return factorized_;
}

// Records the statistics and inertia after JOB=2; returns whether K is (numerically) singular.
template <class T, class U>
bool MumpsFactorization<T, U>::read_factorization_info(bool mumps_singular) {
  stats_.factor_entries = mumps::decode_count(mumps_.infog(9));
  stats_.factor_mb_used = mumps_.infog(21);
  stats_.factor_mb_allocated = mumps_.infog(18);
  stats_.null_pivots = mumps_.infog(28);
  stats_.delayed_pivots = mumps_.infog(13);

  const index_t n = pencil_->order();
  const index_t neg_m = mumps_.infog(12);   // negative pivots of the MUMPS matrix (K or E)
  const index_t null_m = mumps_.infog(28);  // null pivots of the MUMPS matrix
  // Every eigenvalue of K appears twice in E. An odd count means the two copies of an
  // eigenvalue at the shift were given different signs, i.e. K is singular to working
  // precision even though no pivot fell below CNTL(3): treated like a null pivot.
  const bool split_pair = (kEmbeddingFactor == 2) && (neg_m % 2 != 0);
  const bool singular = mumps_singular || null_m > 0 || split_pair;

  inertia_.neg = neg_m / kEmbeddingFactor;
  inertia_.zero = (null_m + kEmbeddingFactor - 1) / kEmbeddingFactor;
  inertia_.pos = std::max<index_t>(0, n - inertia_.neg - inertia_.zero);
  inertia_.valid = !singular;
  return singular;
}

template <class T, class U>
bool MumpsFactorization<T, U>::release_factors() {
  if (pencil_ == nullptr || !holds_factors_) return false;
  const int info1 = mumps_.run(kFreeFactors);
  stats_.last_info1 = info1;
  stats_.last_info2 = mumps_.infog(2);
  if (info1 < 0) throw_mumps_error("release of the factors (JOB=-4)", info1, mumps_.infog(2));
  holds_factors_ = false;
  factorized_ = false;
  // inertia().valid is what other modules test for "solve() works" (ShiftInvertOperator's
  // constructor, the driver's rescue path): it must not stay true without factors. The counts
  // keep describing the released factorization.
  inertia_.valid = false;
  return true;
}

template <class T, class U>
void MumpsFactorization<T, U>::solve(T* b, index_t nrhs, index_t ldb) {
  if (!factorized_)
    HS_THROW(Internal, "SparseFactorization::solve() without a valid factorization");
  if (discard_at_analysis_)
    HS_THROW(Internal, "SparseFactorization::solve(): the factors were discarded (inertia-only object)");
  const index_t n = pencil_->order();
  if (nrhs < 0 || ldb < n)
    HS_THROW(InvalidInput,
             "SparseFactorization::solve: nrhs=" << nrhs << ", ldb=" << ldb << ", n=" << n);
  if (nrhs == 0) return;
  if (nrhs > INT_MAX)
    HS_THROW(Unsupported,
             "SparseFactorization::solve: nrhs=" << nrhs << " exceeds the MUMPS_INT range");
  Timer timer;

  auto& s = mumps_.raw();
  s.nrhs = static_cast<int>(nrhs);
  // Double-precision real right-hand sides are solved in place; everything else goes through a
  // packed buffer of the MUMPS arithmetic: [Re b; Im b] per column for complex T (the
  // embedding's right-hand side), rounded to float for smumps.
  constexpr bool kInPlace = std::is_same_v<T, double> && std::is_same_v<U, double>;
  std::vector<U> packed;
  if constexpr (kInPlace) {
    if (ldb > INT_MAX)
      HS_THROW(Unsupported,
               "SparseFactorization::solve: ldb=" << ldb << " exceeds the MUMPS_INT range");
    s.rhs = b;
    s.lrhs = static_cast<int>(ldb);
  } else {
    const index_t m = kEmbeddingFactor * n;
    packed.resize(static_cast<std::size_t>(m * nrhs));
#pragma omp parallel for collapse(2) schedule(static)
    for (index_t k = 0; k < nrhs; ++k)
      for (index_t i = 0; i < n; ++i) {
        const T v = b[i + k * ldb];
        if constexpr (is_complex_v<T>) {
          packed[i + k * m] = static_cast<U>(v.real());
          packed[n + i + k * m] = static_cast<U>(v.imag());
        } else {
          packed[i + k * m] = static_cast<U>(v);
        }
      }
    s.rhs = packed.data();
    s.lrhs = static_cast<int>(m);
  }

  const int info1 = mumps_.run(mumps::kSolve);
  s.rhs = nullptr;
  stats_.last_info1 = info1;
  stats_.last_info2 = mumps_.infog(2);
  if (info1 < 0) throw_mumps_error("solve", info1, mumps_.infog(2));

  if constexpr (!kInPlace) {
    const index_t m = kEmbeddingFactor * n;
#pragma omp parallel for collapse(2) schedule(static)
    for (index_t k = 0; k < nrhs; ++k)
      for (index_t i = 0; i < n; ++i) {
        if constexpr (is_complex_v<T>) {
          b[i + k * ldb] = T(packed[i + k * m], packed[n + i + k * m]);
        } else {
          b[i + k * ldb] = static_cast<T>(packed[i + k * m]);
        }
      }
  }
  stats_.solve_s += timer.seconds();
  ++stats_.solves;
  stats_.rhs_solved += nrhs;
}

}  // namespace mumps

template <class T>
std::unique_ptr<SparseFactorization<T>> make_mumps_factorization(const FactorOptions& options) {
  if (options.precision == FactorPrecision::Single)
    return std::make_unique<mumps::MumpsFactorization<T, float>>(options);
  return std::make_unique<mumps::MumpsFactorization<T, double>>(options);
}

template <class T>
Inertia compute_inertia(const ShiftedPencil<T>& pencil, double alpha, double beta,
                        const FactorOptions& options) {
  // A rounded (single) or compressed (BLR) factor has the inertia of a perturbed matrix, which
  // is wrong whenever an eigenvalue lies within the perturbation of the shift.
  FactorOptions exact = options;
  exact.precision = FactorPrecision::Double;
  exact.blr = false;
  exact.blr_estimates = false;
  mumps::MumpsFactorization<T, double> f(exact);
  f.set_discard_factors(true);  // only the pivot signs are needed (working memory only)
  f.analyze(pencil, alpha, beta);
  // A singular K is reported through inertia().valid == false, never thrown.
  static_cast<void>(f.factorize(alpha, beta));
  return f.inertia();
}

#define HS_INSTANTIATE_FACTOR(T)                                                \
  template class SparseFactorization<T>;                                        \
  template class mumps::MumpsFactorization<T, double>;                          \
  template class mumps::MumpsFactorization<T, float>;                           \
  template std::unique_ptr<SparseFactorization<T>> make_mumps_factorization<T>( \
      const FactorOptions&);                                                    \
  template Inertia compute_inertia<T>(const ShiftedPencil<T>&, double, double,  \
                                      const FactorOptions&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_FACTOR)
#undef HS_INSTANTIATE_FACTOR

}  // namespace hs
