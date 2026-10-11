// Separator balance vs fill of hsolve's nested dissection (module analysis, P4 review fix): the
// MUMPS flops of the order hs::ordering::nested_dissection computes on grid Laplacians for a range
// of SeparatorParams::imbalance values - the measurement behind kMaxImbalance (the largest
// OrderingOptions::imbalance compute_ordering accepts, src/factor/ordering_detail.hpp).
//
//   bench_analysis_fix_imbalance [--grids 60x60,300x300,20x20x20]
//       [--imbalance 1,1.1,1.2,1.3,1.5,1.7,1.8,1.9,1.95,1.99,1.999] [--threads t]
//
// Grids: NXxNY (5-point) or NXxNYxNZ (7-point) Dirichlet Laplacians. Seed: OrderingOptions'
// default (as compute_ordering). One JSON line per (grid, imbalance): separators, leaves,
// dissection depth (separator levels, saturating at 64), time, and the MUMPS analysis estimate
// of the order (RINFOG(1) flops, INFOG(20) factor entries). Deterministic: the order does not
// depend on the thread count, the estimates not on the machine.
// Reproduction of the kMaxImbalance comment (cores 38,39):
//   python3 tools/run/guard.py --skip-busy-check --cores 38,39 --mem-gb 2 --timeout 600 --
//     build/release/bench_analysis_fix_imbalance --threads 2
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "factor/mumps_factorization.hpp"
#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;

std::vector<std::string> split_list(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string e; std::getline(ss, e, sep);)
    if (!e.empty()) out.push_back(e);
  return out;
}

// Dirichlet Laplacian of an nx x ny x nz grid (nz = 1: 5-point stencil, else 7-point).
hs::CsrMatrix<double> grid_laplacian(int nx, int ny, int nz) {
  const index_t n = static_cast<index_t>(nx) * ny * nz;
  std::vector<index_t> r, c;
  std::vector<double> v;
  auto id = [&](int i, int j, int k) { return static_cast<index_t>(i + nx * (j + ny * k)); };
  auto edge = [&](index_t p, index_t q) {
    r.push_back(p), c.push_back(q), v.push_back(-1.0);
    r.push_back(q), c.push_back(p), v.push_back(-1.0);
  };
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const index_t p = id(i, j, k);
        r.push_back(p), c.push_back(p), v.push_back(nz == 1 ? 4.0 : 6.0);
        if (i + 1 < nx) edge(p, id(i + 1, j, k));
        if (j + 1 < ny) edge(p, id(i, j + 1, k));
        if (k + 1 < nz) edge(p, id(i, j, k + 1));
      }
  return hs::csr_from_coo<double>(n, n, r, c, v);
}

}  // namespace

int main(int argc, char** argv) {
  std::string grids = "60x60,300x300,20x20x20";
  std::string imbalances = "1,1.1,1.2,1.3,1.5,1.7,1.8,1.9,1.95,1.99,1.999";
  int threads = 1;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        return 2;
      }
      if (k == "--grids")
        grids = argv[++i];
      else if (k == "--imbalance")
        imbalances = argv[++i];
      else if (k == "--threads")
        threads = std::stoi(argv[++i]);
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    for (const std::string& spec : split_list(grids, ',')) {
      const std::vector<std::string> dims = split_list(spec, 'x');
      if (dims.size() < 2 || dims.size() > 3) HS_THROW(InvalidInput, "grid " << spec);
      const int nx = std::stoi(dims[0]), ny = std::stoi(dims[1]);
      const int nz = dims.size() == 3 ? std::stoi(dims[2]) : 1;
      const auto a = grid_laplacian(nx, ny, nz);
      const hs::ShiftedPencil<double> pencil(a, nullptr);
      for (const std::string& imb : split_list(imbalances, ',')) {
        hs::ordering::DissectionParams p;
        p.separator.imbalance = std::stod(imb);
        p.seed = hs::OrderingOptions{}.seed;
        hs::ordering::Graph g =
            hs::ordering::graph_from_upper(static_cast<std::int32_t>(pencil.order()),
                                           pencil.rowptr().data(), pencil.colind().data());
        hs::ordering::DissectionStats st;
        hs::Timer t;
        const std::vector<std::int32_t> order =
            hs::ordering::nested_dissection(std::move(g), p, threads, &st);
        const double seconds = t.seconds();
        std::vector<int> perm(order.size());
        for (std::size_t k = 0; k < order.size(); ++k) perm[order[k]] = static_cast<int>(k + 1);
        hs::mumps::MumpsFactorization<double, double> f{hs::FactorOptions{}};
        f.set_pivot_order(perm);
        f.analyze(pencil, 1.0, 0.0);
        const hs::FactorEstimate e = f.estimate();
        std::cout << json{{"grid", spec},
                          {"n", pencil.order()},
                          {"imbalance", p.separator.imbalance},
                          {"threads", threads},
                          {"seconds", seconds},
                          {"separators", st.separators},
                          {"leaves", st.leaves},
                          {"depth", st.depth_separator.size()},
                          {"est_flops", e.flops},
                          {"est_entries", e.factor_entries}}
                         .dump()
                  << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
