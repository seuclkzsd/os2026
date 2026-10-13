// The fit of the ordering cost model (src/driver/steps.cpp, OrderingModel) and the replay of the
// adaptive ordering on the measured components (module anaint, P5c fix: the fit used to live in an
// uncommitted script, and its quoted replay error was in-sample only).
//
//   bench_anaint_fix_fit --ord16 FILE --ord4 FILE --data DIR [--loo 1] [--verbose 1]
//
// FILE: JSON lines of bench_anaint_orderings (bench/anaint_orderings.cpp), one per case and
// ordering and run - either its own output line or an object with "returncode" (0) and its output
// under "bench". --ord16: the 16-thread runs (P5c; gpu02 measurement slot); --ord4: the 4-thread
// runs with --nd-serial 1 (P5; 3 interleaved repetitions). The commands are in
// bench/anaint_orderings.cpp. DIR: data/cases (each case's A.npz, B.npz, config.json: the inputs
// the driver prices - mumps_input_size / ordering_input of the pencil, the planned work of
// plan_work after the cheap B tests).
// The fit, as described in steps.cpp: cases of MUMPS order >= 5000; medians over repetitions;
// nonnegative least squares on the relative error (Lawson-Hanson) for the floor, hsolve's nested
// dissection work W (4 threads, its 1-thread ordering time), AMD's ordering, the factorization and
// the 1-RHS solve; medians for nd_serial (16-thread ordering time over W, cases with W > 0.05 s),
// rho of METIS / PORD / SCOTCH, refactorization and count over t_f, the extra-RHS cost, AMD's
// low-fill factors (cases of fill <= 4, as ratios to hsolve's nested dissection; fill < 12 shown)
// and the 16 / 4-thread exponents; the shape fits on the 4-thread estimates (ln r piecewise linear
// in ln AMD fill, ridge 0.05). Output: the constants next to the committed ordering_model(), the
// ranges of predicted / measured, then the replay: for every case with all five orderings measured
// at 16 threads, the driver's choice (OrderingPricer, run_ordering_choice) priced with the
// committed model and with the refit one - analysed orderings' analyses plus the planned numeric
// work of the chosen one, from the measured components - against the best ordering; with --loo 1
// also each case replayed with the model refit without it (leave-one-case-out). Deterministic; no
// timing: run it in the build lane:
//   python3 tools/run/guard.py --skip-busy-check --mem-gb 10 --timeout 1800 --
//     build/release/bench_anaint_fix_fit --ord16 ord16.jsonl --ord4 ord4.jsonl --data data/cases --loo 1
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"

namespace {

using hs::index_t;
using hs::driver::OrderingModel;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kNd = hs::kOrderingNestedDissection;
constexpr int kOrderings[] = {0, kNd, 5, 4, 3};
constexpr double kMinOrder = 5000.0;

// ------------------------------------------------------------------------------ numerics

double median(std::vector<double> v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/// numpy's default (linear) percentile.
double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  if (v.empty()) return kNaN;
  const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
  const auto i = static_cast<std::size_t>(std::floor(pos));
  const double f = pos - static_cast<double>(i);
  return i + 1 < v.size() ? v[i] + f * (v[i + 1] - v[i]) : v[i];
}

/// Solves the k x k system a x = b (Gaussian elimination with partial pivoting; k <= 6).
std::vector<double> solve_dense(std::vector<std::vector<double>> a, std::vector<double> b) {
  const std::size_t k = b.size();
  for (std::size_t c = 0; c < k; ++c) {
    std::size_t p = c;
    for (std::size_t r = c + 1; r < k; ++r)
      if (std::abs(a[r][c]) > std::abs(a[p][c])) p = r;
    std::swap(a[c], a[p]);
    std::swap(b[c], b[p]);
    for (std::size_t r = c + 1; r < k; ++r) {
      const double f = a[r][c] / a[c][c];
      for (std::size_t j = c; j < k; ++j) a[r][j] -= f * a[c][j];
      b[r] -= f * b[c];
    }
  }
  std::vector<double> x(k);
  for (std::size_t c = k; c-- > 0;) {
    double s = b[c];
    for (std::size_t j = c + 1; j < k; ++j) s -= a[c][j] * x[j];
    x[c] = s / a[c][c];
  }
  return x;
}

/// Least squares on the columns `cols` of x (rows): min || x[:, cols] z - y ||.
std::vector<double> least_squares(const std::vector<std::vector<double>>& x, const std::vector<double>& y,
                                  const std::vector<std::size_t>& cols, double ridge = 0.0) {
  const std::size_t k = cols.size();
  std::vector<std::vector<double>> g(k, std::vector<double>(k, 0.0));
  std::vector<double> r(k, 0.0);
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t a = 0; a < k; ++a) {
      r[a] += x[i][cols[a]] * y[i];
      for (std::size_t b = 0; b < k; ++b) g[a][b] += x[i][cols[a]] * x[i][cols[b]];
    }
  for (std::size_t a = 0; a < k; ++a) g[a][a] += ridge * ridge;
  return solve_dense(g, r);
}

/// Nonnegative least squares min || x z - y ||, z >= 0 (Lawson-Hanson, as scipy.optimize.nnls).
std::vector<double> nnls(const std::vector<std::vector<double>>& x, const std::vector<double>& y) {
  const std::size_t n = x.empty() ? 0 : x[0].size();
  std::vector<double> z(n, 0.0);
  std::vector<bool> passive(n, false);
  auto gradient = [&]() {
    std::vector<double> w(n, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
      double res = y[i];
      for (std::size_t j = 0; j < n; ++j) res -= x[i][j] * z[j];
      for (std::size_t j = 0; j < n; ++j) w[j] += x[i][j] * res;
    }
    return w;
  };
  for (int outer = 0; outer < 3 * static_cast<int>(n) + 3; ++outer) {
    const std::vector<double> w = gradient();
    std::size_t t = n;
    double best = 1e-12;
    for (std::size_t j = 0; j < n; ++j)
      if (!passive[j] && w[j] > best) best = w[j], t = j;
    if (t == n) break;
    passive[t] = true;
    for (int inner = 0; inner < 3 * static_cast<int>(n) + 3; ++inner) {
      std::vector<std::size_t> cols;
      for (std::size_t j = 0; j < n; ++j)
        if (passive[j]) cols.push_back(j);
      const std::vector<double> s_p = least_squares(x, y, cols);
      std::vector<double> s(n, 0.0);
      for (std::size_t a = 0; a < cols.size(); ++a) s[cols[a]] = s_p[a];
      bool feasible = true;
      for (const std::size_t j : cols) feasible = feasible && s[j] > 0.0;
      if (feasible) {
        z = s;
        break;
      }
      double alpha = 1.0;
      for (const std::size_t j : cols)
        if (s[j] <= 0.0) alpha = std::min(alpha, z[j] / (z[j] - s[j]));
      for (std::size_t j = 0; j < n; ++j) z[j] += alpha * (s[j] - z[j]);
      for (const std::size_t j : cols)
        if (z[j] <= 1e-15) passive[j] = false, z[j] = 0.0;
    }
  }
  return z;
}

/// NNLS on the relative error: min || diag(1 / y) (x z) - 1 ||, z >= 0; also the ratios x z / y.
std::vector<double> nnls_rel(const std::vector<std::vector<double>>& x, const std::vector<double>& y,
                             std::vector<double>* ratio = nullptr) {
  std::vector<std::vector<double>> xs = x;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (double& v : xs[i]) v /= y[i];
  const std::vector<double> z = nnls(xs, std::vector<double>(y.size(), 1.0));
  if (ratio) {
    ratio->clear();
    for (std::size_t i = 0; i < x.size(); ++i) {
      double p = 0.0;
      for (std::size_t j = 0; j < z.size(); ++j) p += x[i][j] * z[j];
      ratio->push_back(p / y[i]);
    }
  }
  return z;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

/// Piecewise-linear basis values at ln fill (flat beyond the end knots).
std::vector<double> hat(double ln_fill) {
  constexpr int k = OrderingModel::kShapeKnots;
  double knots[k];
  for (int i = 0; i < k; ++i) knots[i] = std::log(hs::driver::kShapeKnotFill[i]);
  const double l = std::clamp(ln_fill, knots[0], knots[k - 1]);
  std::vector<double> v(k, 0.0);
  for (int i = 0; i + 1 < k; ++i)
    if (knots[i] <= l && l <= knots[i + 1]) {
      const double t = (l - knots[i]) / (knots[i + 1] - knots[i]);
      v[i] = 1.0 - t;
      v[i + 1] = t;
      break;
    }
  return v;
}

// ------------------------------------------------------------------------------ data

/// Medians of one case and ordering (bench_anaint_orderings fields) and the case's inputs.
struct Row {
  std::string c;
  int o = 0;
  double n = 0, order = 0, entries = 0, nnz = 0;  // nnz: upper entries of the pencil
  double gv = 0, ge = 0;                          // the graph hsolve's nested dissection orders
  double flops = kNaN, fe = kNaN, incore = kNaN;
  double an = kNaN, ords = kNaN, fac = kNaN, ref = kNaN, ca = kNaN, cc = kNaN, s1 = kNaN, s2 = kNaN, s4 = kNaN,
         nd1 = kNaN;
};
using Table = std::map<std::pair<std::string, int>, Row>;

/// What the driver knows about a case before its analysis.
struct CaseInput {
  hs::SolverConfig cfg;
  hs::driver::OrderingInput in;
  hs::driver::PlannedWork work;
  double n = 0;
};

double field(const json& b, const char* key, const char* sub = nullptr) {
  if (!b.contains(key) || b[key].is_null()) return kNaN;
  const json& v = b[key];
  if (sub) return v.contains(sub) && v[sub].is_number() ? v[sub].get<double>() : kNaN;
  return v.is_number() ? v.get<double>() : kNaN;
}

std::map<std::pair<std::string, int>, std::vector<json>> load_runs(const std::string& path) {
  std::map<std::pair<std::string, int>, std::vector<json>> runs;
  std::ifstream in(path);
  if (!in) HS_THROW(InvalidInput, "cannot read " << path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] != '{') continue;
    json r = json::parse(line);
    json b = r;
    if (r.contains("bench")) {
      if (r.value("returncode", 0) != 0) continue;
      b = r["bench"];
    }
    if (!b.contains("case") || !b.contains("ordering")) continue;
    runs[{b["case"].get<std::string>(), b["ordering"].get<int>()}].push_back(b);
  }
  return runs;
}

Table table(const std::map<std::pair<std::string, int>, std::vector<json>>& runs,
            const std::map<std::string, CaseInput>& inputs) {
  Table t;
  for (const auto& [key, bs] : runs) {
    const auto it = inputs.find(key.first);
    if (it == inputs.end()) continue;
    Row r;
    r.c = key.first;
    r.o = key.second;
    const json& b0 = bs.front();
    r.n = field(b0, "n");
    r.order = field(b0, "order");
    r.entries = field(b0, "entries");
    r.nnz = field(b0, "pencil_nnz");
    r.gv = it->second.in.nd.order;
    r.ge = it->second.in.nd.entries;
    auto med = [&](const char* k, const char* sub = nullptr) {
      std::vector<double> v;
      for (const json& b : bs) v.push_back(field(b, k, sub));
      return median(v);
    };
    r.flops = med("est_flops");
    r.fe = med("est_entries");
    r.incore = med("est_incore_mb");
    r.an = med("analyze_s");
    r.ords = med("ordering_s");
    r.fac = med("factorize_s");
    r.ref = med("refactorize_s");
    r.ca = med("counter_setup_s");
    r.cc = med("counter_count_s");
    r.s1 = med("solve1_s");
    r.s2 = med("solve_p_s", "2");
    r.s4 = med("solve_p_s", "4");
    r.nd1 = med("nd_ordering_s_1");
    t[key] = r;
  }
  return t;
}

template <class T>
CaseInput case_input(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::SolverConfig& cfg) {
  CaseInput ci;
  ci.cfg = cfg;
  ci.cfg.threads = 16;
  ci.n = static_cast<double>(a.nrows);
  const hs::ShiftedPencil<T> pencil(a, b);
  ci.in = hs::driver::ordering_input(pencil);
  // The planned type and scale as hs::solve has them before its analysis (solver.cpp).
  const auto na = hs::driver::estimate_norm2(a, 30, cfg.seed);
  hs::ProblemType type = cfg.problem;
  double scale = na.norm2;
  if (!b) {
    type = hs::ProblemType::Standard;
  } else {
    const auto nb = hs::driver::estimate_norm2(*b, 30, cfg.seed);
    scale = nb.norm2 > 0 ? na.norm2 / nb.norm2 : na.norm2;
    if (type == hs::ProblemType::Auto) {
      const auto d = hs::driver::test_b_definiteness(*b, nb);
      if (d.verdict != hs::driver::Definiteness::Unknown)
        type = d.verdict == hs::driver::Definiteness::Positive ? hs::ProblemType::GeneralizedDefinite
                                                               : hs::ProblemType::GeneralizedIndefinite;
    }
  }
  ci.work = hs::driver::plan_work(ci.cfg, type, a.nrows, std::max(scale, 1e-300));
  return ci;
}

CaseInput load_case(const fs::path& dir) {
  std::ifstream cf(dir / "config.json");
  std::stringstream ss;
  ss << cf.rdbuf();
  const hs::SolverConfig cfg = hs::parse_config(ss.str());
  auto la = hs::load_matrix(dir / "A.npz");
  const bool has_b = fs::exists(dir / "B.npz");
  hs::LoadedMatrix lb;
  if (has_b) lb = hs::load_matrix(dir / "B.npz");
  if (!la.is_complex() && (!has_b || !lb.is_complex())) {
    const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
    return case_input<double>(a, has_b ? &std::get<hs::CsrMatrix<double>>(lb.matrix) : nullptr, cfg);
  }
  auto to_c = [](hs::LoadedMatrix& m) {
    return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(std::move(m.matrix))
                          : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(m.matrix)));
  };
  const auto a = to_c(la);
  if (!has_b) return case_input<hs::cplx>(a, nullptr, cfg);
  const auto b = to_c(lb);
  return case_input<hs::cplx>(a, &b, cfg);
}

// ------------------------------------------------------------------------------ the fit

struct Range {
  explicit Range(std::string n) : name(std::move(n)) {}
  std::string name;
  std::vector<double> ratio;
  std::vector<std::string> labels;
};

void summarize(const Range& r, bool verbose) {
  if (r.ratio.empty()) return;
  const auto [lo, hi] = std::minmax_element(r.ratio.begin(), r.ratio.end());
  std::printf("  %-22s pred/meas min %.2f med %.2f max %.2f (%zu)", r.name.c_str(), *lo, median(r.ratio), *hi,
              r.ratio.size());
  if (verbose) {  // the three furthest from 1 (either way)
    std::vector<std::pair<double, std::size_t>> w;
    for (std::size_t i = 0; i < r.ratio.size(); ++i) w.emplace_back(std::abs(std::log(r.ratio[i])), i);
    std::sort(w.begin(), w.end());
    std::printf("; worst");
    for (std::size_t i = w.size() > 3 ? w.size() - 3 : 0; i < w.size(); ++i)
      std::printf(" %s %.2f", r.labels[w[i].second].c_str(), r.ratio[w[i].second]);
  }
  std::printf("\n");
}

struct Fit {
  OrderingModel m{};
  std::vector<Range> ranges;
  std::vector<double> serial;  ///< ND serial fractions of the cases
};

double floor_of(const OrderingModel& m, const Row& d, double fe) {
  return m.floor_order * d.order + m.floor_entry * d.entries + m.floor_fe * fe;
}
double w_of(const OrderingModel& m, double v, double e, double fe) {
  return m.nd_work_vertex * v + m.nd_work_edge * e + m.nd_work_fe * fe;
}
double tf_of(const OrderingModel& m, const Row& d) {
  return m.fac_order * d.order + m.fac_entry * d.entries + m.fac_fe * d.fe + m.fac_flops * d.flops;
}
double ts_of(const OrderingModel& m, const Row& d) { return m.solve_order * d.order + m.solve_fe * d.fe; }

Fit fit(const Table& d16, const Table& d4) {
  Fit f;
  OrderingModel& m = f.m;
  auto big = [](const Table& t) {
    std::vector<const Row*> v;
    for (const auto& [k, r] : t)
      if (r.order >= kMinOrder) v.push_back(&r);
    return v;
  };
  const auto b16 = big(d16), b4 = big(d4);
  auto label = [](const Row* r) { return r->c + "/" + std::to_string(r->o); };
  // floor: the InertiaCounter's analysis (given order), every ordering
  {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    Range rg{"floor"};
    for (const Row* r : b16) {
      if (std::isnan(r->ca)) continue;
      x.push_back({r->order, r->entries, r->fe});
      y.push_back(r->ca);
      rg.labels.push_back(label(r));
    }
    const auto z = nnls_rel(x, y, &rg.ratio);
    m.floor_order = z[0], m.floor_entry = z[1], m.floor_fe = z[2];
    f.ranges.push_back(rg);
  }
  // W: hsolve's nested dissection on one thread (4-thread campaign, --nd-serial 1)
  {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    Range rg{"ND work W"};
    for (const Row* r : b4) {
      if (r->o != kNd || std::isnan(r->nd1)) continue;
      x.push_back({r->gv, r->ge, r->fe});
      y.push_back(r->nd1);
      rg.labels.push_back(r->c);
    }
    const auto z = nnls_rel(x, y, &rg.ratio);
    m.nd_work_vertex = z[0], m.nd_work_edge = z[1], m.nd_work_fe = z[2];
    f.ranges.push_back(rg);
  }
  // serial fraction at 16 threads: ords = W (s + (1 - s) / 16)
  for (const Row* r : b16) {
    if (r->o != kNd || std::isnan(r->ords)) continue;
    const double w = w_of(m, r->gv, r->ge, r->fe);
    if (w > 0.05) f.serial.push_back((r->ords / w - 1.0 / 16.0) / (1.0 - 1.0 / 16.0));
  }
  m.nd_serial = median(f.serial);
  {
    Range rg{"ND analysis"};
    for (const Row* r : b16)
      if (r->o == kNd) {
        rg.ratio.push_back((floor_of(m, *r, r->fe) +
                            w_of(m, r->gv, r->ge, r->fe) * (m.nd_serial + (1.0 - m.nd_serial) / 16.0)) / r->an);
        rg.labels.push_back(r->c);
      }
    f.ranges.push_back(rg);
  }
  // MUMPS orderings: rho * W of the full input with their own fe
  for (const int o : {5, 4, 3}) {
    std::vector<double> q;
    Range rg{o == 5 ? "METIS analysis" : o == 4 ? "PORD analysis" : "SCOTCH analysis"};
    for (const Row* r : b16) {
      if (r->o != o) continue;
      const double y = r->an - floor_of(m, *r, r->fe), w = w_of(m, r->order, r->entries, r->fe);
      if (y > 0) q.push_back(y / w);
    }
    const double rho = median(q);
    (o == 5 ? m.rho_metis : o == 4 ? m.rho_pord : m.rho_scotch) = rho;
    for (const Row* r : b16)
      if (r->o == o) {
        rg.ratio.push_back((floor_of(m, *r, r->fe) + rho * w_of(m, r->order, r->entries, r->fe)) / r->an);
        rg.labels.push_back(r->c);
      }
    f.ranges.push_back(rg);
  }
  // AMD's ordering
  {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    Range rg{"AMD analysis"};
    for (const Row* r : b16)
      if (r->o == 0) {
        x.push_back({r->order, r->entries});
        y.push_back(std::max(r->an - floor_of(m, *r, r->fe), 0.1 * r->an));
      }
    const auto z = nnls_rel(x, y);
    m.amd_order = z[0], m.amd_entry = z[1];
    for (const Row* r : b16)
      if (r->o == 0) {
        rg.ratio.push_back((floor_of(m, *r, r->fe) + m.amd_order * r->order + m.amd_entry * r->entries) / r->an);
        rg.labels.push_back(r->c);
      }
    f.ranges.push_back(rg);
  }
  // shapes, on the 4-thread estimates (the committed constants' source; SCOTCH's estimates vary
  // between runs, the other orderings' do not)
  {
    int s = 0;
    for (const int o : {kNd, 5, 4, 3}) {
      std::vector<std::vector<double>> x;
      std::vector<double> yf, ye;
      std::set<std::string> cases;
      for (const Row* r : b4) cases.insert(r->c);
      std::vector<std::string> labels;
      for (const std::string& c : cases) {
        const auto a = d4.find({c, 0}), r = d4.find({c, o});
        if (a == d4.end() || r == d4.end()) continue;
        x.push_back(hat(std::log(std::max(a->second.fe / a->second.nnz, 1.0))));
        yf.push_back(std::log(r->second.flops / a->second.flops));
        ye.push_back(std::log(r->second.fe / a->second.fe));
        labels.push_back(c);
      }
      std::vector<std::size_t> cols(OrderingModel::kShapeKnots);
      for (std::size_t i = 0; i < cols.size(); ++i) cols[i] = i;
      const auto cf = least_squares(x, yf, cols, 0.05), ce = least_squares(x, ye, cols, 0.05);
      OrderingModel::Shape& sh = m.shapes[s++];
      sh.ordering = o;
      Range rf{"shape flops " + std::to_string(o)}, re{"shape entries " + std::to_string(o)};
      for (int i = 0; i < OrderingModel::kShapeKnots; ++i) sh.flops[i] = cf[i], sh.entries[i] = ce[i];
      for (std::size_t i = 0; i < x.size(); ++i) {
        rf.ratio.push_back(std::exp(dot(x[i], cf) - yf[i]));
        re.ratio.push_back(std::exp(dot(x[i], ce) - ye[i]));
      }
      rf.labels = re.labels = labels;
      f.ranges.push_back(rf);
      f.ranges.push_back(re);
    }
  }
  // numeric phases at 16 threads
  std::vector<const Row*> num;
  for (const Row* r : b16) {
    const double v[] = {r->fac, r->ref, r->cc, r->ca, r->s1, r->s2, r->s4};
    if (std::none_of(std::begin(v), std::end(v), [](double x) { return std::isnan(x); })) num.push_back(r);
  }
  {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    Range rg{"factorization"};
    for (const Row* r : num) {
      x.push_back({r->order, r->entries, r->fe, r->flops});
      y.push_back(r->fac);
      rg.labels.push_back(label(r));
    }
    const auto z = nnls_rel(x, y, &rg.ratio);
    m.fac_order = z[0], m.fac_entry = z[1], m.fac_fe = z[2], m.fac_flops = z[3];
    f.ranges.push_back(rg);
    std::vector<double> qr, qc;
    for (const Row* r : num) qr.push_back(r->ref / tf_of(m, *r)), qc.push_back(r->cc / tf_of(m, *r));
    m.refactor = median(qr);
    m.count = median(qc);
  }
  {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    Range rg{"1-RHS solve"};
    for (const Row* r : num) {
      x.push_back({r->order, r->fe});
      y.push_back(r->s1);
      rg.labels.push_back(label(r));
    }
    const auto z = nnls_rel(x, y, &rg.ratio);
    m.solve_order = z[0], m.solve_fe = z[1];
    f.ranges.push_back(rg);
  }
  // AMD at low fill: its measured / predicted over hsolve's nested dissection's, fill <= 4
  for (const bool fac : {true, false}) {
    std::vector<double> lo;
    std::set<std::string> cases;
    for (const Row* r : b16) cases.insert(r->c);
    for (const std::string& c : cases) {
      const auto a = d16.find({c, 0}), nd = d16.find({c, kNd});
      if (a == d16.end() || nd == d16.end()) continue;
      const Row &ra = a->second, &rn = nd->second;
      const double ma = fac ? ra.fac : ra.s4, mn = fac ? rn.fac : rn.s4;
      if (std::isnan(ma) || std::isnan(mn)) continue;
      const double fill = ra.fe / ra.nnz;
      if (fill > 4.0) continue;
      const double pa = fac ? tf_of(m, ra) : ts_of(m, ra), pn = fac ? tf_of(m, rn) : ts_of(m, rn);
      lo.push_back(std::log((ma / pa) / (mn / pn)));
    }
    (fac ? m.amd_low_fill_factorization : m.amd_low_fill_solve) = lo.empty() ? 1.0 : std::exp(median(lo));
  }
  {
    std::vector<double> q4;
    for (const Row* r : num) q4.push_back(r->s4 / r->s1);
    m.solve_per_rhs = (median(q4) - 1.0) / 3.0;
  }
  // thread exponents between 4 and 16 threads (same case and ordering)
  for (const bool fac : {true, false}) {
    std::vector<double> v;
    for (const auto& [k, r] : d16) {
      const auto it = d4.find(k);
      if (it == d4.end()) continue;
      const double a = fac ? r.fac : r.s1, b = fac ? it->second.fac : it->second.s1;
      if (!std::isnan(a) && !std::isnan(b) && b > 0.02) v.push_back(a / b);
    }
    (fac ? m.fac_thread_gamma : m.solve_thread_gamma) = -std::log(median(v)) / std::log(4.0);
  }
  return f;
}

void print_model(const OrderingModel& f, const OrderingModel& c) {
  auto row = [](const char* name, double fit, double committed) {
    const double rel = committed != 0.0 ? fit / committed - 1.0 : (fit == 0.0 ? 0.0 : kNaN);
    std::printf("  %-28s fit %12.5g  committed %12.5g  (%+.1f%%)\n", name, fit, committed, 100.0 * rel);
  };
  row("floor_order", f.floor_order, c.floor_order);
  row("floor_entry", f.floor_entry, c.floor_entry);
  row("floor_fe", f.floor_fe, c.floor_fe);
  row("nd_work_vertex", f.nd_work_vertex, c.nd_work_vertex);
  row("nd_work_edge", f.nd_work_edge, c.nd_work_edge);
  row("nd_work_fe", f.nd_work_fe, c.nd_work_fe);
  row("nd_serial", f.nd_serial, c.nd_serial);
  row("rho_metis", f.rho_metis, c.rho_metis);
  row("rho_pord", f.rho_pord, c.rho_pord);
  row("rho_scotch", f.rho_scotch, c.rho_scotch);
  row("amd_order", f.amd_order, c.amd_order);
  row("amd_entry", f.amd_entry, c.amd_entry);
  row("fac_order", f.fac_order, c.fac_order);
  row("fac_entry", f.fac_entry, c.fac_entry);
  row("fac_fe", f.fac_fe, c.fac_fe);
  row("fac_flops", f.fac_flops, c.fac_flops);
  row("refactor", f.refactor, c.refactor);
  row("count", f.count, c.count);
  row("solve_order", f.solve_order, c.solve_order);
  row("solve_fe", f.solve_fe, c.solve_fe);
  row("solve_per_rhs", f.solve_per_rhs, c.solve_per_rhs);
  row("fac_thread_gamma", f.fac_thread_gamma, c.fac_thread_gamma);
  row("solve_thread_gamma", f.solve_thread_gamma, c.solve_thread_gamma);
  row("amd_low_fill_factorization", f.amd_low_fill_factorization, c.amd_low_fill_factorization);
  row("amd_low_fill_solve", f.amd_low_fill_solve, c.amd_low_fill_solve);
  for (int s = 0; s < 4; ++s) {
    std::printf("  shape %2d flops  ", f.shapes[s].ordering);
    for (int i = 0; i < OrderingModel::kShapeKnots; ++i) std::printf(" %6.3f", f.shapes[s].flops[i]);
    std::printf("  committed");
    for (int i = 0; i < OrderingModel::kShapeKnots; ++i) std::printf(" %5.2f", c.shapes[s].flops[i]);
    std::printf("\n  shape %2d entries", f.shapes[s].ordering);
    for (int i = 0; i < OrderingModel::kShapeKnots; ++i) std::printf(" %6.3f", f.shapes[s].entries[i]);
    std::printf("  committed");
    for (int i = 0; i < OrderingModel::kShapeKnots; ++i) std::printf(" %5.2f", c.shapes[s].entries[i]);
    std::printf("\n");
  }
}

// ------------------------------------------------------------------------------ the replay

/// What the planned work costs with ordering `d`'s measured components (the driver's count rule).
double measured_numeric(const CaseInput& ci, const Row& d) {
  const hs::driver::PlannedWork& w = ci.work;
  if (std::isnan(d.fac)) return kNaN;
  double s = d.fac + std::max(0.0, w.factorizations - 1.0) * d.ref;
  if (w.counts > 0) {
    hs::FactorEstimate e;
    e.flops = d.flops;
    e.factor_entries = static_cast<std::int64_t>(d.fe);
    const auto counts = static_cast<index_t>(std::llround(w.counts));
    s += hs::driver::counter_pays(ci.cfg, e, ci.in.mumps, 16, counts) ? d.ca + w.counts * d.cc : w.counts * d.ref;
  }
  const double sp = w.block >= 4 ? d.s4 : w.block >= 2 ? d.s2 : d.s1;
  return s + w.solve_calls * sp;
}

struct Replay {
  int choice = -1;
  std::vector<int> analysed;
  double total = kNaN, best = kNaN;
  int best_ordering = -1;
  double excess() const { return total / best - 1.0; }
};

/// The driver's choice on the recorded estimates (OrderingPricer + run_ordering_choice), and what
/// it and the best ordering cost with the measured components; false: the case lacks some.
bool replay(const OrderingModel& model, const Table& d16, const std::string& c, const CaseInput& ci, Replay& out) {
  std::map<int, const Row*> rows;
  for (const int o : kOrderings) {
    const auto it = d16.find({c, o});
    if (it == d16.end()) return false;
    rows[o] = &it->second;
  }
  const Row& amd = *rows[0];
  out.best = std::numeric_limits<double>::infinity();
  for (const int o : kOrderings) {  // over the orderings with numeric measurements
    const double v = rows[o]->an + measured_numeric(ci, *rows[o]) + (o != 0 ? amd.an : 0.0);
    if (!std::isnan(v) && v < out.best) out.best = v, out.best_ordering = o;
  }
  if (out.best_ordering < 0) return false;
  const std::vector<int> orderings = hs::driver::candidate_orderings(ci.cfg);
  std::vector<hs::driver::OrderingCandidate> cand(orderings.size());
  for (std::size_t i = 0; i < orderings.size(); ++i) cand[i].ordering = orderings[i];
  hs::driver::OrderingPricer pricer{model, ci.cfg, ci.work, ci.in, amd.nnz, 16, 0.0, {}};
  out.analysed.clear();
  auto analyse = [&](std::size_t i) {
    const Row& r = *rows[orderings[i]];
    hs::FactorEstimate e;
    e.flops = r.flops;
    e.factor_entries = static_cast<std::int64_t>(r.fe);
    e.incore_mb = r.incore;
    pricer.analysed(cand, i, e);
    out.analysed.push_back(orderings[i]);
  };
  auto predict = [&](std::size_t anchor) { pricer.predict(cand, anchor); };
  const std::size_t pick = hs::driver::run_ordering_choice(cand, analyse, predict, [](std::size_t) {});
  out.choice = orderings[pick];
  out.total = measured_numeric(ci, *rows[out.choice]);
  for (const int o : out.analysed) out.total += rows[o]->an;
  return !std::isnan(out.total);
}

const char* name(int o) {
  switch (o) {
    case 0: return "AMD";
    case 3: return "SCOTCH";
    case 4: return "PORD";
    case 5: return "METIS";
    case kNd: return "ND";
    default: return "?";
  }
}

std::string analysed_list(const std::vector<int>& v) {
  std::string s;
  for (const int o : v) s += (s.empty() ? "" : ",") + std::string(name(o));
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ord16, ord4, data;
  bool loo = false, verbose = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        return 2;
      }
      const std::string v = argv[++i];
      if (k == "--ord16")
        ord16 = v;
      else if (k == "--ord4")
        ord4 = v;
      else if (k == "--data")
        data = v;
      else if (k == "--loo")
        loo = std::stoi(v) != 0;
      else if (k == "--verbose")
        verbose = std::stoi(v) != 0;
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (ord16.empty() || ord4.empty() || data.empty()) {
      std::cerr << "usage: bench_anaint_fix_fit --ord16 FILE --ord4 FILE --data DIR [--loo 1] [--verbose 1]\n";
      return 2;
    }
    omp_set_num_threads(std::max(1, omp_get_max_threads()));
    const auto runs16 = load_runs(ord16), runs4 = load_runs(ord4);
    std::set<std::string> cases;
    for (const auto& [k, v] : runs16) cases.insert(k.first);
    for (const auto& [k, v] : runs4) cases.insert(k.first);
    std::map<std::string, CaseInput> inputs;
    for (const std::string& c : cases) {
      if (!fs::exists(fs::path(data) / c / "A.npz")) continue;
      inputs[c] = load_case(fs::path(data) / c);
    }
    const Table d16 = table(runs16, inputs), d4 = table(runs4, inputs);
    std::printf("cases: %zu (16 threads: %zu rows, 4 threads: %zu rows)\n", inputs.size(), d16.size(), d4.size());
    const Fit f = fit(d16, d4);
    std::printf("\nconstants (refit from the raw data vs committed ordering_model()):\n");
    print_model(f.m, hs::driver::ordering_model());
    const auto [smin, smax] = std::minmax_element(f.serial.begin(), f.serial.end());
    std::printf("ND serial fraction at 16 threads: median %.3f, quartiles %.3f %.3f, range %.2f..%.2f (%zu cases)\n",
                f.m.nd_serial, percentile(f.serial, 25), percentile(f.serial, 75), *smin, *smax, f.serial.size());
    std::printf("\nranges of predicted / measured (refit):\n");
    for (const Range& r : f.ranges) summarize(r, verbose);
    // The replay: committed model, refit model, leave-one-case-out.
    std::printf("\nreplay at 16 threads (s; measured components; AMD probe always paid):\n");
    std::printf("  %-34s %8s %-7s  %-24s %8s %8s  %-24s %8s %8s\n", "case", "best", "", "committed: choice", "total",
                "excess", "refit: choice", "total", "excess");
    std::vector<double> ex_c, ex_f, ex_loo;
    double sum_best = 0, sum_c = 0, sum_f = 0;
    std::vector<std::string> loo_lines;
    for (const auto& [c, ci] : inputs) {
      Replay rc, rf;
      if (!replay(hs::driver::ordering_model(), d16, c, ci, rc) || !replay(f.m, d16, c, ci, rf)) continue;
      ex_c.push_back(rc.excess());
      ex_f.push_back(rf.excess());
      sum_best += rc.best, sum_c += rc.total, sum_f += rf.total;
      std::printf("  %-34s %8.3f %-7s  %-6s(%-16s) %8.3f %7.1f%%  %-6s(%-16s) %8.3f %7.1f%%\n", c.c_str(), rc.best,
                  name(rc.best_ordering), name(rc.choice), analysed_list(rc.analysed).c_str(), rc.total,
                  100.0 * rc.excess(), name(rf.choice), analysed_list(rf.analysed).c_str(), rf.total,
                  100.0 * rf.excess());
      if (loo) {
        Table l16, l4;
        for (const auto& [k, r] : d16)
          if (k.first != c) l16[k] = r;
        for (const auto& [k, r] : d4)
          if (k.first != c) l4[k] = r;
        const Fit lf = fit(l16, l4);
        Replay rl;
        if (replay(lf.m, d16, c, ci, rl)) {
          ex_loo.push_back(rl.excess());
          char buf[256];
          std::snprintf(buf, sizeof buf, "  %-34s %-6s %8.3f %7.1f%%", c.c_str(), name(rl.choice), rl.total,
                        100.0 * rl.excess());
          loo_lines.emplace_back(buf);
        }
      }
    }
    auto mean = [](const std::vector<double>& v) {
      double s = 0;
      for (const double x : v) s += x;
      return v.empty() ? kNaN : s / static_cast<double>(v.size());
    };
    std::printf("  sums over %zu cases: best %.2f s, committed %.2f s, refit %.2f s\n", ex_c.size(), sum_best, sum_c,
                sum_f);
    std::printf("mean excess over the best ordering: committed %.1f%%, refit (in-sample) %.1f%%\n", 100.0 * mean(ex_c),
                100.0 * mean(ex_f));
    if (loo) {
      std::printf("\nleave-one-case-out (each case replayed with the model refit without it):\n");
      for (const auto& l : loo_lines) std::printf("%s\n", l.c_str());
      std::printf("mean excess, leave-one-case-out: %.1f%% over %zu cases\n", 100.0 * mean(ex_loo), ex_loo.size());
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
