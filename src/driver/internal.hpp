// Driver-internal building blocks shared by the solve paths (src/driver/*.cpp only).
// Scalar set of the templates: T in {double, cplx}, instantiated in the .cpp that defines them.
#pragma once

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "driver/norms.hpp"
#include "hsolve/dense.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::driver {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ------------------------------------------------------------------------------ context

struct Ctx {
  const SolverConfig& cfg;
  Logger* log;
  SolveReport& rep;
};

void warn(Ctx& c, const std::string& msg);
void event(Ctx& c, const std::string& name, std::initializer_list<LogField> fields);
std::string fmt(double x);

// ------------------------------------------------------------------------------ pair blocks

/// Pairs being refined: eigenvalues with their vectors and A*X, B*X kept in sync.
template <class T>
struct Block {
  std::vector<double> lambda;
  DenseMatrix<T> x, ax, bx;

  index_t size() const { return static_cast<index_t>(lambda.size()); }
};

template <class T>
void recompute_products(const CsrMatrix<T>& a, const CsrMatrix<T>* b, Block<T>& blk);
template <class T>
Block<T> take_columns(const Block<T>& src, const std::vector<index_t>& cols);
/// dst <- [dst src] (dst may be empty).
template <class T>
void append_columns(DenseMatrix<T>& dst, const DenseMatrix<T>& src);

// ------------------------------------------------------------------------------ factorization

FactorOptions factor_options_from(const SolverConfig& cfg, double cap_mb);

/// Factorize K(sigma0) (DESIGN §3.6-4): perturb the shift on a null pivot, and factorize at sigma0
/// without null-pivot detection when every nearby shift looks singular (dense cluster). Returns
/// the shift used; detection_off tells later refactorizations at that shift to do the same.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off);

/// Refactorize K(sigma) the way factorize_near() did (needed after certificate factorizations).
template <class T>
void refactorize_at(SparseFactorization<T>& fac, double sigma, bool detection_off);

/// nu_-(A - s B): number of eigenvalues below s (DESIGN §4.2), moving s outward (direction
/// `outward`, steps step * 4^i) while K(s) is singular. Returns the final s via `s`.
template <class T>
index_t count_below(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double& s, double outward,
                    double step);

/// Symbolic analysis at sigma0 and the memory-budget decision (DESIGN §3.6-3).
template <class T>
void analyze_within_budget(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma0,
                           bool already_analyzed = false);

/// Adaptive fill-reducing ordering (DESIGN §3.6-3b): AMD probe + cost model, SCOTCH if it pays.
template <class T>
std::unique_ptr<SparseFactorization<T>> adaptive_analysis(Ctx& c, const FactorOptions& fopts,
                                                          const ShiftedPencil<T>& pencil, double sigma0,
                                                          index_t expected_factorizations, index_t expected_solves);

// ------------------------------------------------------------------------------ solve paths

/// HEP / GHEP (B HPD): shift-invert Lanczos, Rayleigh-Ritz, rescue, inertia certificate.
template <class T>
Solution<T> solve_definite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                           double n2b, double scale, double sigma0);

/// GHIEP (B indefinite): shift-invert general Krylov-Schur, projected refinement, real/complex
/// classification.
template <class T>
Solution<T> solve_indefinite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                             double n2b, double scale);

}  // namespace hs::driver
