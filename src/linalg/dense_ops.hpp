// Dense vector/block helpers shared by the eigensolvers and the driver (module-internal API).
// Single owner of: deterministic random vectors, inner products/norms, CGS2 orthogonalization.
// T in {double, cplx}. Vectors are contiguous; blocks are column-major with explicit ld.
#pragma once

#include <cstdint>
#include <functional>

#include "hsolve/dense.hpp"
#include "hsolve/types.hpp"

namespace hs::linalg {

/// Deterministic pseudo-random entries, uniform in [-1, 1] (real and imaginary parts
/// independently for complex T). (seed, stream) select an independent sequence; the result
/// does not depend on the number of threads. O(n).
template <class T>
void fill_random(T* x, index_t n, std::uint64_t seed, std::uint64_t stream = 0);

/// x^H y (conjugate-linear in x). OpenMP-parallel reduction; deterministic for a fixed
/// thread count.
template <class T>
T dot(index_t n, const T* x, const T* y);

/// ||x||_2 (overflow-safe two-pass scaling is not needed for our magnitudes; plain sum).
template <class T>
double norm2(index_t n, const T* x);

/// y += alpha * x.
template <class T>
void axpy(index_t n, T alpha, const T* x, T* y);

/// x *= alpha.
template <class T>
void scal(index_t n, T alpha, T* x);

/// y = M * x for one vector (used to recompute M*w instead of storing M*V).
template <class T>
using ApplyFn = std::function<void(const T* x, T* y)>;

/// Orthogonalize w against the k columns of V (n x k, ld ldv) by two passes of block classical
/// Gram-Schmidt (CGS2, BLAS2), in the inner product <x, y>_M = y^H M x. Three modes:
///   * Euclidean (M = I): bv == nullptr, bw == nullptr, apply_m == nullptr.
///   * Stored M*V:        bv = M*V (n x k, ld ldbv) and bw = M*w on entry; bw is updated
///                        consistently (bw -= BV*c), no application of M.
///   * Recompute:         bv == nullptr, apply_m != nullptr, bw = workspace (n); M*w is
///                        recomputed before each pass and at the end (3 applications of M).
/// On exit bw == M*w (stored/recompute modes). h (length k, may be nullptr) receives the
/// accumulated coefficients V^H M w. Returns the M-norm sqrt(Re(w^H M w)) (0 if not positive)
/// or ||w||_2. Precondition: the columns of V are M-orthonormal. Cost: 4nk flops per pass.
template <class T>
double orthogonalize_cgs2(index_t n, index_t k, const T* v, index_t ldv, const T* bv, index_t ldbv,
                          T* w, T* bw, T* h, const ApplyFn<T>* apply_m = nullptr);

/// C (k x k) = V^H * W for V (n x k, ld ldv), W (n x k, ld ldw). Used for Gram matrices and
/// Rayleigh quotients. BLAS3.
template <class T>
void gram(index_t n, index_t k, const T* v, index_t ldv, const T* w, index_t ldw, T* c, index_t ldc);

/// Y (n x m) = V (n x k) * S (k x m), Y must not alias V. BLAS3.
template <class T>
void multiply(index_t n, index_t k, index_t m, const T* v, index_t ldv, const T* s, index_t lds, T* y,
              index_t ldy);

}  // namespace hs::linalg
