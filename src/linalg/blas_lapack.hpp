// Thin, typed C++ wrappers over the Fortran BLAS/LAPACK routines used by hsolve.
// LP64 interface (32-bit Fortran INTEGER), as provided by OpenBLAS. Fortran CHARACTER
// arguments carry hidden length arguments (size_t) which are passed explicitly.
// All wrappers throw hs::Error(Internal) on info != 0, except where a positive info is a
// legitimate numerical outcome (documented per function).
#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/types.hpp"

namespace hs::lapack {

using fint = int;  // Fortran INTEGER (LP64)
using fstrlen = std::size_t;
using zcplx = std::complex<double>;

extern "C" {
// ---- BLAS level 1/2/3 ----
void dgemm_(const char* ta, const char* tb, const fint* m, const fint* n, const fint* k,
            const double* alpha, const double* a, const fint* lda, const double* b,
            const fint* ldb, const double* beta, double* c, const fint* ldc, fstrlen, fstrlen);
void zgemm_(const char* ta, const char* tb, const fint* m, const fint* n, const fint* k,
            const zcplx* alpha, const zcplx* a, const fint* lda, const zcplx* b,
            const fint* ldb, const zcplx* beta, zcplx* c, const fint* ldc, fstrlen, fstrlen);
void dgemv_(const char* t, const fint* m, const fint* n, const double* alpha, const double* a,
            const fint* lda, const double* x, const fint* incx, const double* beta, double* y,
            const fint* incy, fstrlen);
void zgemv_(const char* t, const fint* m, const fint* n, const zcplx* alpha, const zcplx* a,
            const fint* lda, const zcplx* x, const fint* incx, const zcplx* beta, zcplx* y,
            const fint* incy, fstrlen);
void dtrsm_(const char* side, const char* uplo, const char* ta, const char* diag, const fint* m,
            const fint* n, const double* alpha, const double* a, const fint* lda, double* b,
            const fint* ldb, fstrlen, fstrlen, fstrlen, fstrlen);
void ztrsm_(const char* side, const char* uplo, const char* ta, const char* diag, const fint* m,
            const fint* n, const zcplx* alpha, const zcplx* a, const fint* lda, zcplx* b,
            const fint* ldb, fstrlen, fstrlen, fstrlen, fstrlen);

// ---- LAPACK: Hermitian eigenproblems ----
void dsyev_(const char* jobz, const char* uplo, const fint* n, double* a, const fint* lda,
            double* w, double* work, const fint* lwork, fint* info, fstrlen, fstrlen);
void zheev_(const char* jobz, const char* uplo, const fint* n, zcplx* a, const fint* lda,
            double* w, zcplx* work, const fint* lwork, double* rwork, fint* info, fstrlen,
            fstrlen);
void dsygv_(const fint* itype, const char* jobz, const char* uplo, const fint* n, double* a,
            const fint* lda, double* b, const fint* ldb, double* w, double* work,
            const fint* lwork, fint* info, fstrlen, fstrlen);
void zhegv_(const fint* itype, const char* jobz, const char* uplo, const fint* n, zcplx* a,
            const fint* lda, zcplx* b, const fint* ldb, double* w, zcplx* work,
            const fint* lwork, double* rwork, fint* info, fstrlen, fstrlen);

// ---- LAPACK: Cholesky ----
void dpotrf_(const char* uplo, const fint* n, double* a, const fint* lda, fint* info, fstrlen);
void zpotrf_(const char* uplo, const fint* n, zcplx* a, const fint* lda, fint* info, fstrlen);

// ---- LAPACK: nonsymmetric Schur ----
void dgees_(const char* jobvs, const char* sort, void* select, const fint* n, double* a,
            const fint* lda, fint* sdim, double* wr, double* wi, double* vs, const fint* ldvs,
            double* work, const fint* lwork, fint* bwork, fint* info, fstrlen, fstrlen);
void zgees_(const char* jobvs, const char* sort, void* select, const fint* n, zcplx* a,
            const fint* lda, fint* sdim, zcplx* w, zcplx* vs, const fint* ldvs, zcplx* work,
            const fint* lwork, double* rwork, fint* bwork, fint* info, fstrlen, fstrlen);
void dtrsen_(const char* job, const char* compq, const fint* select, const fint* n, double* t,
             const fint* ldt, double* q, const fint* ldq, double* wr, double* wi, fint* m,
             double* s, double* sep, double* work, const fint* lwork, fint* iwork,
             const fint* liwork, fint* info, fstrlen, fstrlen);
void ztrsen_(const char* job, const char* compq, const fint* select, const fint* n, zcplx* t,
             const fint* ldt, zcplx* q, const fint* ldq, zcplx* w, fint* m, double* s,
             double* sep, zcplx* work, const fint* lwork, fint* info, fstrlen, fstrlen);
void dtrevc_(const char* side, const char* howmny, fint* select, const fint* n, const double* t,
             const fint* ldt, double* vl, const fint* ldvl, double* vr, const fint* ldvr,
             const fint* mm, fint* m, double* work, fint* info, fstrlen, fstrlen);
void ztrevc_(const char* side, const char* howmny, fint* select, const fint* n, zcplx* t,
             const fint* ldt, zcplx* vl, const fint* ldvl, zcplx* vr, const fint* ldvr,
             const fint* mm, fint* m, zcplx* work, double* rwork, fint* info, fstrlen, fstrlen);
}  // extern "C"

// ------------------------------------------------------------------------------------------
// Typed wrappers. Matrices are column-major. Op chars follow BLAS ('N', 'T', 'C').
// For real types 'C' is treated as 'T'.
// ------------------------------------------------------------------------------------------

inline void gemm(char ta, char tb, fint m, fint n, fint k, double alpha, const double* a, fint lda,
                 const double* b, fint ldb, double beta, double* c, fint ldc) {
  if (m == 0 || n == 0) return;
  if (ta == 'C') ta = 'T';
  if (tb == 'C') tb = 'T';
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
}
inline void gemm(char ta, char tb, fint m, fint n, fint k, zcplx alpha, const zcplx* a, fint lda,
                 const zcplx* b, fint ldb, zcplx beta, zcplx* c, fint ldc) {
  if (m == 0 || n == 0) return;
  zgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
}

inline void gemv(char t, fint m, fint n, double alpha, const double* a, fint lda, const double* x,
                 double beta, double* y) {
  if (m == 0 || n == 0) return;
  if (t == 'C') t = 'T';
  const fint one = 1;
  dgemv_(&t, &m, &n, &alpha, a, &lda, x, &one, &beta, y, &one, 1);
}
inline void gemv(char t, fint m, fint n, zcplx alpha, const zcplx* a, fint lda, const zcplx* x,
                 zcplx beta, zcplx* y) {
  if (m == 0 || n == 0) return;
  const fint one = 1;
  zgemv_(&t, &m, &n, &alpha, a, &lda, x, &one, &beta, y, &one, 1);
}

/// Solve op(A) X = alpha B (side 'L') or X op(A) = alpha B (side 'R') with triangular A.
inline void trsm(char side, char uplo, char ta, char diag, fint m, fint n, double alpha,
                 const double* a, fint lda, double* b, fint ldb) {
  if (m == 0 || n == 0) return;
  if (ta == 'C') ta = 'T';
  dtrsm_(&side, &uplo, &ta, &diag, &m, &n, &alpha, a, &lda, b, &ldb, 1, 1, 1, 1);
}
inline void trsm(char side, char uplo, char ta, char diag, fint m, fint n, zcplx alpha,
                 const zcplx* a, fint lda, zcplx* b, fint ldb) {
  if (m == 0 || n == 0) return;
  ztrsm_(&side, &uplo, &ta, &diag, &m, &n, &alpha, a, &lda, b, &ldb, 1, 1, 1, 1);
}

/// Hermitian eigenproblem: eigenvalues ascending in w; eigenvectors overwrite a if jobz == 'V'.
inline void heev(char jobz, char uplo, fint n, double* a, fint lda, double* w) {
  if (n == 0) return;
  fint info = 0, lwork = -1;
  double wq = 0;
  dsyev_(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, &info, 1, 1);
  lwork = static_cast<fint>(wq);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dsyev_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, &info, 1, 1);
  if (info != 0) HS_THROW(Internal, "dsyev failed, info=" << info);
}
inline void heev(char jobz, char uplo, fint n, zcplx* a, fint lda, double* w) {
  if (n == 0) return;
  fint info = 0, lwork = -1;
  zcplx wq = 0;
  std::vector<double> rwork(static_cast<std::size_t>(std::max<fint>(1, 3 * n - 2)));
  zheev_(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, rwork.data(), &info, 1, 1);
  lwork = static_cast<fint>(wq.real());
  std::vector<zcplx> work(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(), &info, 1, 1);
  if (info != 0) HS_THROW(Internal, "zheev failed, info=" << info);
}

/// Generalized Hermitian-definite problem A x = lambda B x (itype 1). Returns info:
/// 0 on success, info > n means B is not positive definite (the caller decides what to do).
inline fint hegv(char jobz, char uplo, fint n, double* a, fint lda, double* b, fint ldb, double* w) {
  if (n == 0) return 0;
  const fint itype = 1;
  fint info = 0, lwork = -1;
  double wq = 0;
  dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &wq, &lwork, &info, 1, 1);
  lwork = static_cast<fint>(wq);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, &info, 1, 1);
  if (info < 0 || (info > 0 && info <= n)) HS_THROW(Internal, "dsygv failed, info=" << info);
  return info;
}
inline fint hegv(char jobz, char uplo, fint n, zcplx* a, fint lda, zcplx* b, fint ldb, double* w) {
  if (n == 0) return 0;
  const fint itype = 1;
  fint info = 0, lwork = -1;
  zcplx wq = 0;
  std::vector<double> rwork(static_cast<std::size_t>(std::max<fint>(1, 3 * n - 2)));
  zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &wq, &lwork, rwork.data(), &info, 1, 1);
  lwork = static_cast<fint>(wq.real());
  std::vector<zcplx> work(static_cast<std::size_t>(lwork));
  zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, rwork.data(), &info,
         1, 1);
  if (info < 0 || (info > 0 && info <= n)) HS_THROW(Internal, "zhegv failed, info=" << info);
  return info;
}

/// Cholesky factorization. Returns info (> 0: leading minor not positive definite).
inline fint potrf(char uplo, fint n, double* a, fint lda) {
  fint info = 0;
  if (n == 0) return 0;
  dpotrf_(&uplo, &n, a, &lda, &info, 1);
  if (info < 0) HS_THROW(Internal, "dpotrf illegal argument " << -info);
  return info;
}
inline fint potrf(char uplo, fint n, zcplx* a, fint lda) {
  fint info = 0;
  if (n == 0) return 0;
  zpotrf_(&uplo, &n, a, &lda, &info, 1);
  if (info < 0) HS_THROW(Internal, "zpotrf illegal argument " << -info);
  return info;
}

/// Real Schur decomposition A = Q T Q^T (no sorting). Eigenvalues in w (complex).
inline void gees(fint n, double* a, fint lda, std::complex<double>* w, double* q, fint ldq) {
  if (n == 0) return;
  const char jobvs = 'V', sort = 'N';
  fint sdim = 0, info = 0, lwork = -1;
  std::vector<double> wr(n), wi(n);
  double wq = 0;
  dgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, wr.data(), wi.data(), q, &ldq, &wq, &lwork,
         nullptr, &info, 1, 1);
  lwork = static_cast<fint>(wq);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, wr.data(), wi.data(), q, &ldq, work.data(),
         &lwork, nullptr, &info, 1, 1);
  if (info != 0) HS_THROW(Internal, "dgees failed, info=" << info);
  for (fint i = 0; i < n; ++i) w[i] = {wr[i], wi[i]};
}
/// Complex Schur decomposition A = Q T Q^H (no sorting).
inline void gees(fint n, zcplx* a, fint lda, zcplx* w, zcplx* q, fint ldq) {
  if (n == 0) return;
  const char jobvs = 'V', sort = 'N';
  fint sdim = 0, info = 0, lwork = -1;
  std::vector<double> rwork(n);
  zcplx wq = 0;
  zgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, w, q, &ldq, &wq, &lwork, rwork.data(), nullptr,
         &info, 1, 1);
  lwork = static_cast<fint>(wq.real());
  std::vector<zcplx> work(static_cast<std::size_t>(lwork));
  zgees_(&jobvs, &sort, nullptr, &n, a, &lda, &sdim, w, q, &ldq, work.data(), &lwork, rwork.data(),
         nullptr, &info, 1, 1);
  if (info != 0) HS_THROW(Internal, "zgees failed, info=" << info);
}

/// Reorder a Schur form so that the eigenvalues with select[i] != 0 lead; updates Q.
/// For real T, select must flag both members of a complex-conjugate pair (LAPACK will
/// otherwise move the pair as a whole and set both flags). Returns the number of selected
/// eigenvalues (m). info == 1 (reordering failed: too close eigenvalues) is returned, not thrown.
inline fint trsen(fint n, double* t, fint ldt, double* q, fint ldq, std::vector<fint>& select,
                  std::complex<double>* w, fint* info_out = nullptr) {
  const char job = 'N', compq = 'V';
  fint m = 0, info = 0, lwork = std::max<fint>(1, n), liwork = 1;
  std::vector<double> wr(n), wi(n), work(static_cast<std::size_t>(lwork));
  std::vector<fint> iwork(1);
  double s = 0, sep = 0;
  dtrsen_(&job, &compq, select.data(), &n, t, &ldt, q, &ldq, wr.data(), wi.data(), &m, &s, &sep,
          work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1);
  if (info < 0) HS_THROW(Internal, "dtrsen illegal argument " << -info);
  for (fint i = 0; i < n; ++i) w[i] = {wr[i], wi[i]};
  if (info_out) *info_out = info;
  return m;
}
inline fint trsen(fint n, zcplx* t, fint ldt, zcplx* q, fint ldq, std::vector<fint>& select,
                  zcplx* w, fint* info_out = nullptr) {
  const char job = 'N', compq = 'V';
  fint m = 0, info = 0, lwork = std::max<fint>(1, n);
  std::vector<zcplx> work(static_cast<std::size_t>(lwork));
  double s = 0, sep = 0;
  ztrsen_(&job, &compq, select.data(), &n, t, &ldt, q, &ldq, w, &m, &s, &sep, work.data(), &lwork,
          &info, 1, 1);
  if (info < 0) HS_THROW(Internal, "ztrsen illegal argument " << -info);
  if (info_out) *info_out = info;
  return m;
}

/// Right eigenvectors of a (quasi-)upper-triangular Schur form T, back-transformed by X on
/// input (howmny = 'B'): on entry vr holds Q (n x n), on exit the eigenvectors of Q T Q^H.
/// Real T: complex pairs are returned as (real part, imag part) column pairs (LAPACK layout).
inline void trevc_backtransform(fint n, const double* t, fint ldt, double* vr, fint ldvr) {
  if (n == 0) return;
  const char side = 'R', howmny = 'B';
  fint m = 0, info = 0, mm = n;
  std::vector<double> work(static_cast<std::size_t>(3 * n));
  dtrevc_(&side, &howmny, nullptr, &n, t, &ldt, nullptr, &n, vr, &ldvr, &mm, &m, work.data(), &info,
          1, 1);
  if (info != 0) HS_THROW(Internal, "dtrevc failed, info=" << info);
}
inline void trevc_backtransform(fint n, zcplx* t, fint ldt, zcplx* vr, fint ldvr) {
  if (n == 0) return;
  const char side = 'R', howmny = 'B';
  fint m = 0, info = 0, mm = n;
  std::vector<zcplx> work(static_cast<std::size_t>(2 * n));
  std::vector<double> rwork(n);
  ztrevc_(&side, &howmny, nullptr, &n, t, &ldt, nullptr, &n, vr, &ldvr, &mm, &m, work.data(),
          rwork.data(), &info, 1, 1);
  if (info != 0) HS_THROW(Internal, "ztrevc failed, info=" << info);
}

}  // namespace hs::lapack
