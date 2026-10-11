#include "verify/gaussian.hpp"

#include <cmath>
#include <vector>

#include "linalg/dense_ops.hpp"

namespace hs::verify_detail {

namespace {

/// Pairs of standard normal samples from two arrays of uniform [-1, 1] values (Box-Muller).
inline void box_muller(double v1, double v2, double& z1, double& z2) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  constexpr double kTiny = 1.1102230246251565e-16;  // 2^-53: keeps log() finite
  double u1 = 0.5 * (1.0 - v1);                      // [0, 1]
  if (u1 < kTiny) u1 = kTiny;
  const double u2 = 0.5 * (1.0 - v2);
  const double rad = std::sqrt(-2.0 * std::log(u1));
  z1 = rad * std::cos(kTwoPi * u2);
  z2 = rad * std::sin(kTwoPi * u2);
}

}  // namespace

template <class T>
void fill_gaussian(T* x, index_t n, std::uint64_t seed, std::uint64_t stream) {
  if (n <= 0) return;
  // m pairs of uniforms give 2m normals: n of them (real) or 2n (complex).
  const index_t m = is_complex_v<T> ? n : (n + 1) / 2;
  std::vector<double> u1(static_cast<std::size_t>(m)), u2(static_cast<std::size_t>(m));
  linalg::fill_random(u1.data(), m, seed, 2 * stream);
  linalg::fill_random(u2.data(), m, seed, 2 * stream + 1);
  for (index_t i = 0; i < m; ++i) {
    double z1 = 0.0, z2 = 0.0;
    box_muller(u1[i], u2[i], z1, z2);
    if constexpr (is_complex_v<T>) {
      constexpr double kHalfSqrt2 = 0.70710678118654752440084436210485;
      x[i] = T(kHalfSqrt2 * z1, kHalfSqrt2 * z2);
    } else {
      x[2 * i] = z1;
      if (2 * i + 1 < n) x[2 * i + 1] = z2;
    }
  }
}

#define HS_INSTANTIATE_GAUSSIAN(T) template void fill_gaussian<T>(T*, index_t, std::uint64_t, std::uint64_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_GAUSSIAN)
#undef HS_INSTANTIATE_GAUSSIAN

}  // namespace hs::verify_detail
