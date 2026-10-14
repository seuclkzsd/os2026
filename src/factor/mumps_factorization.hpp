// MUMPS-backed SparseFactorization (module-internal to src/factor). Declared here so that
// InertiaCounter (inertia_counter.cpp) can build inertia-only instances and reuse the pivot order
// of an analysed object; everything outside src/factor uses include/hsolve/factor.hpp.
// T in {double, cplx}: pencil scalar; U in {double, float}: MUMPS arithmetic (dmumps / smumps).
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "factor/mumps_api.hpp"
#include "factor/mumps_input.hpp"
#include "hsolve/factor.hpp"

namespace hs::mumps {

template <class T, class U>
class MumpsFactorization final : public SparseFactorization<T> {
 public:
  explicit MumpsFactorization(const FactorOptions& options);
  MumpsFactorization(const MumpsFactorization&) = delete;
  MumpsFactorization& operator=(const MumpsFactorization&) = delete;

  void analyze(const ShiftedPencil<T>& pencil, double alpha, double beta) override;
  FactorEstimate estimate() const override;
  bool factorize(double alpha, double beta) override;
  void solve(T* b, index_t nrhs, index_t ldb) override;
  Inertia inertia() const override { return inertia_; }
  void set_out_of_core(bool on) override { options_.out_of_core = on; }
  void set_max_memory_mb(double mb) override;
  bool release_factors() override;
  void set_null_pivot_detection(bool on) override { null_pivot_detection_ = on; }
  const FactorStats& stats() const override { return stats_; }
  const FactorOptions& options() const override { return options_; }
  index_t order() const override { return pencil_ ? pencil_->order() : 0; }

  // ---- module-internal extensions (InertiaCounter, compute_inertia) ----

  /// Inertia-only mode: MUMPS ICNTL(31)=1 ("discard factors"), which MUMPS reads at ANALYSIS
  /// (set before a factorization it has no effect: measured, factors and memory unchanged).
  /// Factorizations then keep only the active working memory (INFOG(21) -41..-61% on the 1e5
  /// cases) and run up to 23% faster (factor.hpp); solve() throws Internal (MUMPS would return
  /// INFO -44).
  /// Takes effect at the next analyze().
  void set_discard_factors(bool on) { discard_next_ = on; }
  bool factors_discarded() const { return discard_at_analysis_; }

  /// Pivot order for the next analyze() (SparseFactorization::set_pivot_order).
  void set_pivot_order(std::vector<int> order) override { pivot_order_in_ = std::move(order); }

  /// SYM_PERM of the last successful analysis (empty before one). Copied out of MUMPS right
  /// after JOB=1 (O(n)); reusing it skips the fill-reducing ordering (SCOTCH/METIS: 0.9-1.3 s
  /// on the 1e5 cases vs 0.03-0.05 s for an analysis given the order) at the price of a little
  /// more fill (factor.hpp: +0.4..3.4% entries on the 1e5 cases).
  const std::vector<int>& pivot_order() const override { return pivot_order_out_; }

  /// The analysed pencil (nullptr before a successful analyze()).
  const ShiftedPencil<T>* pencil() const { return pencil_; }

  // ---- measurement hooks (bench/factspeed_*.cpp; not used by the library) ----

  /// Controls written after hsolve's own before every later analyze() and factorize().
  void set_control_overrides(MumpsControlOverrides o) { overrides_ = std::move(o); }
  /// The MUMPS instance, for its INFOG / RINFOG / KEEP statistics after a job.
  const Instance<U>& instance() const { return mumps_; }

 private:
  // Order of K and of the matrix MUMPS factorizes (2n for the embedding of a complex K).
  static constexpr int kEmbeddingFactor = is_complex_v<T> ? 2 : 1;

  void configure_static_controls();
  /// ICNTL(7) for options().ordering (hsolve's own orderings reach MUMPS as PERM_IN: ICNTL(7)=1).
  int mumps_ordering() const {
    return options_.ordering == kOrderingNestedDissection ? 1 : options_.ordering;
  }
  bool read_factorization_info(bool mumps_singular);
  void apply_overrides();

  FactorOptions options_;
  Instance<U> mumps_;
  const ShiftedPencil<T>* pencil_ = nullptr;
  std::unique_ptr<Input<T, U>> input_;
  FactorStats stats_;
  Inertia inertia_;
  bool blr_at_analysis_ = false;
  bool factorized_ = false;
  bool holds_factors_ = false;  ///< a JOB=2 since the last analysis / release left factors behind
  int relax_pct_;  ///< current ICNTL(14): grows with workspace retries and is kept afterwards
  bool null_pivot_detection_ = true;  ///< ICNTL(24) for the next factorization
  bool discard_next_ = false;         ///< ICNTL(31) for the next analysis
  bool discard_at_analysis_ = false;  ///< ICNTL(31) of the current analysis
  std::vector<int> pivot_order_in_;   ///< PERM_IN for the next analysis (empty: ICNTL(7) ordering)
  std::vector<int> pivot_order_out_;  ///< SYM_PERM of the current analysis
  std::vector<int> blkptr_, blkvar_;  ///< ICNTL(15) blocks of the current analysis (complex T)
  MumpsControlOverrides overrides_;   ///< measurement hook (empty in the library)
};

}  // namespace hs::mumps
