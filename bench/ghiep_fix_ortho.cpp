// Orthogonalization kernel of a block step of krylov_schur_general's block mode (module ghiep, P4
// fixes): the fast path eigen_detail::block_cgs_cholqr2 (block CGS with the DGKS test + CholQR2)
// against the two-pass kernel eigen_detail::BlockOrthonormalizer it falls back to (two passes of
// block CGS, each followed by Cholesky-QR), on the same inputs.
//
// Input per (k, p): V = k orthonormal columns (random, orthonormalized), and the new block W:
//   --input krylov (default): W = D U with D = diag(1 + i / n) and U = V(:, k-p:k), an operator
//     image of the last basis block as in a block Arnoldi step: about 80% of each column cancels
//     against V, so the DGKS test asks for its second pass (as in Krylov steps once the space
//     holds good approximations);
//   --input random: W random (nothing cancels: the fast path makes one projection pass).
// Both kernels orthonormalize W against V.
// The methods run interleaved, one call each per round with the order alternating (load drift on
// the shared machine and cache state affect both alike), after one untimed warm-up round; the
// medians are reported, with the result's loss of orthogonality ||[V Q]^H [V Q] - I||_F and
// whether the fast path accepted the block (otherwise its time is that of a refused attempt).
// Output: one JSON line per (k, p) on stdout.
//
// Usage (via the resource guard, 2 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 28,29 --mem-gb 4 --timeout 900 --
//     build/release/bench_ghiep_fix_ortho [--n 100000] [--k 24,48,72] [--p 2,4,8,16,64]
//     [--rounds 15] [--input krylov|random] [--complex]
// (the guard sets OMP_NUM_THREADS / OPENBLAS_NUM_THREADS to the number of cores).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "eigen/block_general.hpp"
#include "eigen/block_ortho.hpp"
#include "hsolve/dense.hpp"
#include "hsolve/types.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::index_t;
using json = nlohmann::json;

struct Args {
  index_t n = 100000;
  std::vector<index_t> k = {24, 48, 72};
  std::vector<index_t> p = {2, 4, 8, 16, 64};
  int rounds = 15;
  std::string input = "krylov";
  bool complex_scalars = false;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--n")
      a.n = std::stoll(value());
    else if (s == "--k")
      a.k = parse_list(value());
    else if (s == "--p")
      a.p = parse_list(value());
    else if (s == "--rounds")
      a.rounds = std::stoi(value());
    else if (s == "--input")
      a.input = value();
    else if (s == "--complex")
      a.complex_scalars = true;
    else
      throw std::runtime_error("unknown argument " + s);
  }
  if (a.input != "krylov" && a.input != "random")
    throw std::runtime_error("--input must be krylov or random");
  return a;
}

/// ||X^H X - I||_F for the n x c columns of x.
template <class T>
double orthogonality_loss(index_t n, index_t c, const T* x) {
  DenseMatrix<T> g(c, c);
  hs::linalg::gram(n, c, x, n, x, n, g.data(), c);
  double s = 0.0;
  for (index_t j = 0; j < c; ++j)
    for (index_t i = 0; i < c; ++i) s += std::norm(g(i, j) - T(i == j ? 1.0 : 0.0));
  return std::sqrt(s);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class T>
void run(const Args& args) {
  const index_t n = args.n;
  constexpr double kRatio = hs::eigen_detail::BlockKrylovSchurBasis<T>::kBreakdownRatio;
  for (const index_t k : args.k) {
    for (const index_t p : args.p) {
      if (p > k || k + p > n) continue;
      // basis(:, 0:k) = V orthonormal; the block slot basis(:, k:k+p) is refilled from w_in.
      DenseMatrix<T> basis(n, k + p), w_in(n, p), coef, r;
      for (index_t j = 0; j < k; ++j) hs::linalg::fill_random(basis.col(j), n, 2026, static_cast<std::uint64_t>(j));
      {
        hs::eigen_detail::BlockOrthonormalizer<T> init(n, k, nullptr, false);
        std::uint64_t stream = static_cast<std::uint64_t>(k);
        init.run(basis.data(), nullptr, 0, 1e-12, 2026, stream, coef, r);
      }
      for (index_t c = 0; c < p; ++c) {
        if (args.input == "random") {
          hs::linalg::fill_random(w_in.col(c), n, 7, static_cast<std::uint64_t>(c));
        } else {
          for (index_t i = 0; i < n; ++i)
            w_in(i, c) = T(1.0 + static_cast<double>(i) / static_cast<double>(n)) * basis(i, k - p + c);
        }
      }
      hs::eigen_detail::BlockOrthonormalizer<T> two_pass(n, p, nullptr, false);
      std::vector<double> t_fast, t_two;
      bool accepted = true;
      double loss_fast = 0.0, loss_two = 0.0;
      for (int round = 0; round <= args.rounds; ++round) {
        for (int q = 0; q < 2; ++q) {
          const bool fast = (q + round) % 2 == 0;
          std::copy(w_in.data(), w_in.data() + n * p, basis.col(k));
          hs::Timer t;
          if (fast) {
            const bool ok = hs::eigen_detail::block_cgs_cholqr2<T>(n, k, p, basis.data(), kRatio, coef, r);
            const double sec = t.seconds();
            accepted = accepted && ok;
            if (round > 0) t_fast.push_back(sec);
            if (round == args.rounds) loss_fast = orthogonality_loss(n, k + p, basis.data());
          } else {
            std::uint64_t stream = 1000;
            two_pass.run(basis.data(), nullptr, k, kRatio, 2026, stream, coef, r);
            const double sec = t.seconds();
            if (round > 0) t_two.push_back(sec);
            if (round == args.rounds) loss_two = orthogonality_loss(n, k + p, basis.data());
          }
        }
      }
      const double mf = median(t_fast), mt = median(t_two);
      std::cout << json{{"record", "ortho"},
                        {"scalar", hs::is_complex_v<T> ? "complex" : "real"},
                        {"n", n},
                        {"k", k},
                        {"p", p},
                        {"input", args.input},
                        {"threads", hs::max_threads()},
                        {"rounds", args.rounds},
                        {"fast_accepted", accepted},
                        {"fast_median_s", mf},
                        {"two_pass_median_s", mt},
                        {"speedup", mt / mf},
                        {"orth_loss_fast", loss_fast},
                        {"orth_loss_two_pass", loss_two}}
                       .dump()
                << std::endl;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.complex_scalars)
      run<cplx>(args);
    else
      run<double>(args);
  } catch (const std::exception& e) {
    std::cerr << "bench_ghiep_fix_ortho: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
