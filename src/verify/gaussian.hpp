// Deterministic standard Gaussian vectors for the randomized checks of src/verify (module-internal).
// Scalar set: T in {double, cplx}.
//
// The probability statement of hs::exclusion_check needs a rotation-invariant start distribution;
// linalg::fill_random is uniform in [-1, 1], so pairs of its values are mapped by Box-Muller
// (u -> (1 - u) / 2 in [0, 1], 0 replaced by 2^-53 before the logarithm). Complex entries have
// independent N(0, 1/2) real and imaginary parts (circular complex Gaussian, E|z|^2 = 1).
// Like fill_random, the result depends only on (seed, stream, n), not on the thread count.
#pragma once

#include <cstdint>

#include "hsolve/types.hpp"

namespace hs::verify_detail {

/// x[0..n) <- independent standard (real or circular complex) Gaussian samples. Uses
/// fill_random streams 2*stream and 2*stream + 1 of `seed`. O(n), 2n doubles of scratch.
template <class T>
void fill_gaussian(T* x, index_t n, std::uint64_t seed, std::uint64_t stream);

}  // namespace hs::verify_detail
