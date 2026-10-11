#include "support/fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::test {

namespace {

template <class T>
T random_scalar(std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  if constexpr (is_complex_v<T>) {
    return T(u(g), u(g));
  } else {
    return u(g);
  }
}

}  // namespace

template <class T>
DenseMatrix<T> to_dense(const CsrMatrix<T>& a) {
  if (a.nrows > 4000 || a.ncols > 4000) HS_THROW(InvalidInput, "to_dense: test helper limited to n <= 4000");
  DenseMatrix<T> d(a.nrows, a.ncols);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) d(i, a.colind[p]) += a.values[p];
  return d;
}

template <class T>
CsrMatrix<T> from_dense(const DenseMatrix<T>& d, double drop_tol) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < d.rows(); ++i)
    for (index_t j = 0; j < d.cols(); ++j)
      if (std::abs(d(i, j)) > drop_tol) {
        r.push_back(i);
        c.push_back(j);
        v.push_back(d(i, j));
      }
  return csr_from_coo<T>(d.rows(), d.cols(), r, c, v, /*drop_zeros=*/false);
}

template <class T>
void DenseOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  const auto n = static_cast<lapack::fint>(a_.rows());
  lapack::gemm('N', 'N', n, static_cast<lapack::fint>(p), n, T(1), a_.data(), n, x,
               static_cast<lapack::fint>(ldx), T(0), y, static_cast<lapack::fint>(ldy));
}

template <class T>
void DiagonalOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  const index_t n = size();
  for (index_t k = 0; k < p; ++k)
    for (index_t i = 0; i < n; ++i) y[i + k * ldy] = d_[i] * x[i + k * ldx];
}

template <class T>
DenseShiftInvertOperator<T>::DenseShiftInvertOperator(const DenseMatrix<T>& a, const DenseMatrix<T>* b,
                                                      double sigma)
    : lu_(a) {
  const index_t n = a.rows();
  if (n > 2000) HS_THROW(InvalidInput, "DenseShiftInvertOperator: n <= 2000");
  if (b) b_ = *b;
  for (index_t j = 0; j < n; ++j)
    for (index_t i = 0; i < n; ++i) lu_(i, j) -= T(sigma) * (b ? (*b)(i, j) : (i == j ? T(1) : T(0)));
  ipiv_.resize(static_cast<std::size_t>(n));
  if (lapack::getrf(static_cast<lapack::fint>(n), lu_.data(), static_cast<lapack::fint>(n), ipiv_.data()) != 0)
    HS_THROW(Singular, "DenseShiftInvertOperator: A - sigma B is singular");
}

template <class T>
void DenseShiftInvertOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  const auto n = static_cast<lapack::fint>(lu_.rows());
  if (b_.empty()) {
    for (index_t k = 0; k < p; ++k)
      for (index_t i = 0; i < n; ++i) y[i + k * ldy] = x[i + k * ldx];
  } else {
    lapack::gemm('N', 'N', n, static_cast<lapack::fint>(p), n, T(1), b_.data(), n, x,
                 static_cast<lapack::fint>(ldx), T(0), y, static_cast<lapack::fint>(ldy));
  }
  lapack::getrs(n, static_cast<lapack::fint>(p), lu_.data(), n, ipiv_.data(), y, static_cast<lapack::fint>(ldy));
}

template <class T>
CsrMatrix<T> laplacian_1d(index_t n) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < n; ++i) {
    r.push_back(i), c.push_back(i), v.push_back(T(2));
    if (i + 1 < n) {
      r.push_back(i), c.push_back(i + 1), v.push_back(T(-1));
      r.push_back(i + 1), c.push_back(i), v.push_back(T(-1));
    }
  }
  return csr_from_coo<T>(n, n, r, c, v);
}

std::vector<double> laplacian_1d_eigenvalues(index_t n) {
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t k = 1; k <= n; ++k) w[k - 1] = 2.0 - 2.0 * std::cos(k * M_PI / static_cast<double>(n + 1));
  return w;
}

template <class T>
CsrMatrix<T> laplacian_2d(index_t nx, index_t ny) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  auto id = [nx](index_t i, index_t j) { return i + j * nx; };
  for (index_t j = 0; j < ny; ++j)
    for (index_t i = 0; i < nx; ++i) {
      const index_t p = id(i, j);
      r.push_back(p), c.push_back(p), v.push_back(T(4));
      if (i + 1 < nx) r.push_back(p), c.push_back(id(i + 1, j)), v.push_back(T(-1));
      if (i > 0) r.push_back(p), c.push_back(id(i - 1, j)), v.push_back(T(-1));
      if (j + 1 < ny) r.push_back(p), c.push_back(id(i, j + 1)), v.push_back(T(-1));
      if (j > 0) r.push_back(p), c.push_back(id(i, j - 1)), v.push_back(T(-1));
    }
  return csr_from_coo<T>(nx * ny, nx * ny, r, c, v);
}

std::vector<double> laplacian_2d_eigenvalues(index_t nx, index_t ny) {
  const auto wx = laplacian_1d_eigenvalues(nx), wy = laplacian_1d_eigenvalues(ny);
  std::vector<double> w;
  w.reserve(static_cast<std::size_t>(nx * ny));
  for (double a : wx)
    for (double b : wy) w.push_back(a + b);
  std::sort(w.begin(), w.end());
  return w;
}

template <class T>
CsrMatrix<T> prescribed_spectrum(const std::vector<double>& eigenvalues, index_t block_size, std::uint64_t seed) {
  const index_t n = static_cast<index_t>(eigenvalues.size());
  if (block_size < 1) HS_THROW(InvalidInput, "prescribed_spectrum: block_size >= 1");
  std::vector<index_t> r, c;
  std::vector<T> v;
  std::uint64_t stream = 0;
  for (index_t start = 0; start < n; start += block_size) {
    const index_t s = std::min(block_size, n - start);
    // Random unitary Q by modified Gram-Schmidt on a random matrix.
    DenseMatrix<T> q(s, s);
    for (index_t j = 0; j < s; ++j) {
      linalg::fill_random(q.col(j), s, seed, stream++);
      for (index_t k = 0; k < j; ++k) {
        T dotp = T(0);
        for (index_t i = 0; i < s; ++i) dotp += conj_if(q(i, k)) * q(i, j);
        for (index_t i = 0; i < s; ++i) q(i, j) -= dotp * q(i, k);
      }
      double nrm = 0;
      for (index_t i = 0; i < s; ++i) nrm += abs2(q(i, j));
      nrm = std::sqrt(nrm);
      for (index_t i = 0; i < s; ++i) q(i, j) /= nrm;
    }
    // M = Q diag(lambda) Q^H, symmetrized to be exactly Hermitian.
    DenseMatrix<T> m(s, s);
    for (index_t i = 0; i < s; ++i)
      for (index_t j = 0; j < s; ++j) {
        T acc = T(0);
        for (index_t k = 0; k < s; ++k) acc += q(i, k) * T(eigenvalues[start + k]) * conj_if(q(j, k));
        m(i, j) = acc;
      }
    for (index_t i = 0; i < s; ++i)
      for (index_t j = 0; j <= i; ++j) {
        const T h = 0.5 * (m(i, j) + conj_if(m(j, i)));
        r.push_back(start + i), c.push_back(start + j), v.push_back(i == j ? T(std::real(h)) : h);
        if (i != j) r.push_back(start + j), c.push_back(start + i), v.push_back(conj_if(h));
      }
  }
  // Random symmetric permutation.
  std::vector<index_t> perm(static_cast<std::size_t>(n));
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 g(seed ^ 0x9E3779B97F4A7C15ULL);
  std::shuffle(perm.begin(), perm.end(), g);
  for (auto& x : r) x = perm[x];
  for (auto& x : c) x = perm[x];
  return csr_from_coo<T>(n, n, r, c, v, /*drop_zeros=*/false);
}

template <class T>
CsrMatrix<T> random_hermitian(index_t n, double density, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < n; ++i) {
    r.push_back(i), c.push_back(i), v.push_back(T(std::real(random_scalar<T>(g))));
    for (index_t j = i + 1; j < n; ++j) {
      if (u(g) > density) continue;
      const T x = random_scalar<T>(g);
      r.push_back(i), c.push_back(j), v.push_back(x);
      r.push_back(j), c.push_back(i), v.push_back(conj_if(x));
    }
  }
  return csr_from_coo<T>(n, n, r, c, v, /*drop_zeros=*/false);
}

template <class T>
CsrMatrix<T> random_hpd(index_t n, double density, std::uint64_t seed) {
  auto h = random_hermitian<T>(n, density, seed);
  // Strict diagonal dominance with positive diagonal => HPD.
  for (index_t i = 0; i < n; ++i) {
    double off = 0.0;
    index_t diag = -1;
    for (index_t p = h.rowptr[i]; p < h.rowptr[i + 1]; ++p) {
      if (h.colind[p] == i) {
        diag = p;
      } else {
        off += std::abs(h.values[p]);
      }
    }
    HS_ASSERT(diag >= 0, "random_hermitian always stores the diagonal");
    h.values[diag] = T(1.0 + off + 0.5 * (1.0 + std::real(h.values[diag])));
  }
  return h;
}

template <class T>
std::vector<double> dense_eigenvalues(const CsrMatrix<T>& a, const CsrMatrix<T>* b) {
  auto da = to_dense(a);
  const auto n = static_cast<lapack::fint>(a.nrows);
  std::vector<double> w(static_cast<std::size_t>(n));
  if (b) {
    auto db = to_dense(*b);
    if (lapack::hegv('N', 'U', n, da.data(), n, db.data(), n, w.data()) != 0)
      HS_THROW(InvalidInput, "dense_eigenvalues: B is not positive definite");
  } else {
    lapack::heev('N', 'U', n, da.data(), n, w.data());
  }
  return w;
}

#define HS_INSTANTIATE_FIXTURES(T)                                                              \
  template DenseMatrix<T> to_dense<T>(const CsrMatrix<T>&);                                     \
  template CsrMatrix<T> from_dense<T>(const DenseMatrix<T>&, double);                           \
  template class DenseOperator<T>;                                                              \
  template class DiagonalOperator<T>;                                                           \
  template class DenseShiftInvertOperator<T>;                                                   \
  template CsrMatrix<T> laplacian_1d<T>(index_t);                                               \
  template CsrMatrix<T> laplacian_2d<T>(index_t, index_t);                                      \
  template CsrMatrix<T> prescribed_spectrum<T>(const std::vector<double>&, index_t, std::uint64_t); \
  template CsrMatrix<T> random_hermitian<T>(index_t, double, std::uint64_t);                    \
  template CsrMatrix<T> random_hpd<T>(index_t, double, std::uint64_t);                          \
  template std::vector<double> dense_eigenvalues<T>(const CsrMatrix<T>&, const CsrMatrix<T>*);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_FIXTURES)
#undef HS_INSTANTIATE_FIXTURES

}  // namespace hs::test
