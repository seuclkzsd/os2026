// Basic scalar/index types and compile-time helpers shared by all modules.
#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

namespace hs {

/// Global index type (rows, columns, nnz counts). CSR column indices use int32_t.
using index_t = std::int64_t;
using cplx = std::complex<double>;

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>> : std::true_type {};
template <class T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <class T>
struct real_type {
  using type = T;
};
template <class T>
struct real_type<std::complex<T>> {
  using type = T;
};
/// Underlying real type: real_t<double> = double, real_t<complex<float>> = float.
template <class T>
using real_t = typename real_type<T>::type;

/// Same "shape" of scalar with a different precision: rebind_t<cplx, float> = complex<float>.
template <class T, class R>
struct rebind_scalar {
  using type = R;
};
template <class T, class R>
struct rebind_scalar<std::complex<T>, R> {
  using type = std::complex<R>;
};
template <class T, class R>
using rebind_t = typename rebind_scalar<T, R>::type;

/// Complex conjugate that is the identity for real scalars (avoids promoting double to complex).
template <class T>
constexpr T conj_if(const T& x) {
  if constexpr (is_complex_v<T>) {
    return std::conj(x);
  } else {
    return x;
  }
}

/// |x|^2 without a square root.
template <class T>
constexpr real_t<T> abs2(const T& x) {
  if constexpr (is_complex_v<T>) {
    return x.real() * x.real() + x.imag() * x.imag();
  } else {
    return x * x;
  }
}

/// Blocks template argument deduction for a parameter (C++20 std::type_identity_t), so that e.g.
/// solve(a, nullptr, cfg) deduces T from `a` alone.
template <class T>
struct type_identity {
  using type = T;
};
template <class T>
using non_deduced_t = typename type_identity<T>::type;

enum class ScalarKind { Real64, Complex128 };

/// Explicit-instantiation policy (docs/DESIGN.md §5): every template declared in a public header
/// is instantiated exactly once, at the bottom of the .cpp of the owning module, via
///   HS_FOR_EACH_SCALAR(MY_INSTANTIATION_MACRO)
/// Headers state their scalar set in their top comment; the default set is {double, cplx}.
#define HS_FOR_EACH_SCALAR(X) X(double) X(::hs::cplx)

template <class T>
constexpr ScalarKind scalar_kind() {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, cplx>,
                "public scalar types are double and std::complex<double>");
  return std::is_same_v<T, double> ? ScalarKind::Real64 : ScalarKind::Complex128;
}

}  // namespace hs
