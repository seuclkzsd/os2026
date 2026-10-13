// Measurements behind include/hsolve/verify.hpp (module certfast): seconds of the exact Sylvester
// certificate (two counts) against the randomized exclusion check on the factorization held at
// sigma, and detection of artificially removed eigenvectors, with hsolve's own solution as X.
//
//   bench_certfast_measure --case DIR --solution DIR [--threads t] [--reps r]
//                          [--deltas 1e-6,1e-12] [--blocks 1,4] [--removal 1]
//
// DIR (case): A.npz (B.npz) and config.json; DIR (solution): the output directory of
//   hsolve --A .. [--B ..] --config DIR/config.json --out SOL --threads t
// (eigenvalues.npy, eigenvectors.npy, report.json: sigma_used, the certificate window and the
// ordering the driver chose). The main factorization is rebuilt here at sigma_used with that
// ordering and the driver's MUMPS settings (deterministic = false, L0-OpenMP).
// Per repetition (interleaved, the order rotating between repetitions), each variant once:
//   exact_driver    refactorize the main object at both window ends (today's driver path); the
//                   refactorization at sigma that a rerun would need is NOT charged
//   exact_counter   InertiaCounter double pair (its one-off analysis reported separately)
//   exact_single    InertiaCounter single pair, with the verdict of single_count_validity
//   randomized      exclusion_check for every delta x block size x pass-radius policy:
//                   window  r' = the driver's certificate window (mid-gap, min over both sides)
//                   hint    r' = r + 0.9 (d_hint - r), d_hint the next Ritz hint the window
//                           ends were placed against (reconstructed: end = (edge + hint) / 2)
//   removal         (--removal 1) the check with one copy of the largest found multiplet removed
//                   from X, and with the found eigenvalue farthest from sigma removed, for every
//                   delta x block size, policy window (radius r of the complete set)
// Output: one JSON object per line; medians over the repetitions in the "summary" lines.
// Timed runs take a node's measurement slot (docs/DESIGN.md §6; P4 measured 2 threads on the
// development server; the judging configuration is 16 threads):
//   python3 tools/run/guard.py --exclusive --timeout 1800 --
//     build/release/bench_certfast_measure --case data/cases/<c> --solution <sol>/<c> --threads 16
//     --reps 3   (one command line)
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "hsolve/verify.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Args {
  fs::path case_dir, solution;
  int threads = 2;
  int reps = 3;
  std::vector<double> deltas{1e-6, 1e-12};
  std::vector<index_t> blocks{1, 4};
  bool removal = true;
};

void emit(const json& j) { std::cout << j.dump() << std::endl; }

double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const std::size_t h = v.size() / 2;
  return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

template <class T>
hs::DenseMatrix<T> load_vectors(const fs::path& p) {
  const hs::NpyArray a = hs::read_npy(p);
  if (a.shape.size() != 2) HS_THROW(InvalidInput, p << ": expected a 2-D array");
  const auto n = static_cast<index_t>(a.shape[0]), m = static_cast<index_t>(a.shape[1]);
  const std::vector<T> v = hs::npy_to_vector<T>(a);
  hs::DenseMatrix<T> x(n, m);
  for (index_t j = 0; j < m; ++j)
    for (index_t i = 0; i < n; ++i) x(i, j) = a.fortran_order ? v[i + j * n] : v[i * m + j];
  return x;
}

template <class T>
hs::DenseMatrix<T> drop_column(const hs::DenseMatrix<T>& x, index_t c) {
  hs::DenseMatrix<T> y(x.rows(), x.cols() - 1);
  for (index_t j = 0, k = 0; j < x.cols(); ++j)
    if (j != c) {
      std::copy(x.col(j), x.col(j) + x.rows(), y.col(k));
      ++k;
    }
  return y;
}

/// A named timed variant; the repetitions call run() in rotating order.
struct Variant {
  std::string name;
  std::function<json()> run;  ///< returns the fields of one repetition (must contain "s")
  std::vector<json> records;
};

template <class T>
int run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  // ---- solution and report
  json rep;
  {
    std::ifstream f(args.solution / "report.json");
    f >> rep;
  }
  const double sigma = rep.at("sigma_used").get<double>();
  const double lower = rep.at("certificate").at("lower").get<double>();
  const double upper = rep.at("certificate").at("upper").get<double>();
  const int ordering = rep.at("factor_options").at("ordering").get<int>();
  const std::vector<double> lambda = hs::npy_to_vector<double>(hs::read_npy(args.solution / "eigenvalues.npy"));
  const hs::DenseMatrix<T> x = load_vectors<T>(args.solution / "eigenvectors.npy");
  const index_t n = a.nrows, m = x.cols();
  if (x.rows() != n || static_cast<index_t>(lambda.size()) != m) HS_THROW(InvalidInput, "solution does not match the case");
  double r = 0.0;
  for (double l : lambda) r = std::max(r, std::abs(l - sigma));
  r *= 1.0 + 1e-9;  // the found eigenvalues lie inside the claim radius
  // Next hints behind the window ends (driver: end = (edge + nearest hint beyond edge) / 2).
  const double hint_up = 2.0 * upper - (sigma + r), hint_lo = 2.0 * lower - (sigma - r);
  const double d_hint = std::min(hint_up - sigma, sigma - hint_lo);
  const double r_window = std::min(sigma - lower, upper - sigma);
  const double r_hint = r + 0.9 * (d_hint - r);
  std::vector<double> found_in_window;
  for (double l : lambda)
    if (l >= lower && l <= upper) found_in_window.push_back(l);

  const hs::ShiftedPencil<T> pencil(a, b);
  hs::FactorOptions fo;
  fo.ordering = ordering;
  fo.threads = args.threads;
  fo.deterministic = false;  // the driver's default (SolverConfig::deterministic = false)
  auto fac = hs::make_mumps_factorization<T>(fo);
  hs::Timer t_an;
  fac->analyze(pencil, 1.0, -sigma);
  const double analyze_s = t_an.seconds();
  if (!fac->factorize(1.0, -sigma)) HS_THROW(Singular, "K(sigma) singular");
  const double factor_s = fac->stats().factorize_s;
  std::unique_ptr<hs::CsrOperator<T>> b_op;
  if (b) b_op = std::make_unique<hs::CsrOperator<T>>(*b);
  const hs::StartScaling scaling = hs::start_scaling_for<T>(b);

  hs::Timer t_ca;
  hs::InertiaCounter<T> counter(*fac, fac->options());
  const double counter_analysis_s = t_ca.seconds();

  // Single-count rule on the driver's window (wanted region = the found eigenvalues' hull).
  hs::SingleWindow sw;
  sw.lower = lower;
  sw.upper = upper;
  sw.wanted_lower = sigma - r;
  sw.wanted_upper = sigma + r;
  sw.found = found_in_window;
  const hs::SingleCountValidity single_rule = hs::single_count_validity(pencil, sw);

  emit({{"exp", "setup"},
        {"case", args.case_dir.filename().string()},
        {"n", n},
        {"complex", hs::is_complex_v<T>},
        {"generalized", b != nullptr},
        {"threads", args.threads},
        {"ordering", ordering},
        {"sigma", sigma},
        {"k", m},
        {"radius", r},
        {"window", {lower, upper}},
        {"r_window", r_window},
        {"d_hint", d_hint},
        {"r_hint", r_hint},
        {"analyze_s", analyze_s},
        {"factorize_s", factor_s},
        {"factor_mb", fac->stats().factor_mb_used},
        {"counter_analysis_s", counter_analysis_s},
        {"kappa", scaling.kappa},
        {"kappa_rigorous", scaling.rigorous},
        {"single_rule_valid", single_rule.valid()},
        {"single_rule_reason", single_rule.reason},
        {"single_eta", {single_rule.eta_lower, single_rule.eta_upper}}});

  // ---- operator application times (calibration of the cost model), interleaved
  auto op = std::make_unique<hs::ShiftInvertOperator<T>>(*fac, pencil, sigma);
  std::map<index_t, std::vector<double>> apply_s;
  const std::vector<index_t> widths{1, 2, 4, 8, m};
  for (int rep_i = 0; rep_i < args.reps; ++rep_i)
    for (index_t q : widths) {
      hs::DenseMatrix<T> in(n, q), out(n, q);
      for (index_t c = 0; c < q; ++c) hs::linalg::fill_random(in.col(c), n, 99, static_cast<std::uint64_t>(c));
      hs::Timer t;
      op->apply(in.data(), n, out.data(), n, q);
      apply_s[q].push_back(t.seconds());
    }
  json cal = {{"exp", "apply"}};
  for (auto& [q, v] : apply_s) cal["apply_s_" + std::to_string(q)] = median(v);
  emit(cal);
  const double t1 = median(apply_s[1]);
  const double rhs_ratio = (median(apply_s[4]) / t1 - 1.0) / 3.0;

  // ---- variants
  std::vector<Variant> variants;
  variants.push_back({"exact_driver", [&]() {
                        hs::Timer t;
                        double ta = lower, tb = upper;
                        if (!fac->factorize(1.0, -ta)) return json{{"s", t.seconds()}, {"valid", false}};
                        const hs::Inertia ia = fac->inertia();
                        if (!fac->factorize(1.0, -tb)) return json{{"s", t.seconds()}, {"valid", false}};
                        const hs::Inertia ib = fac->inertia();
                        const double s = t.seconds();
                        // Restore K(sigma) for the operator (not charged).
                        if (!fac->factorize(1.0, -sigma)) HS_THROW(Singular, "K(sigma) singular");
                        op = std::make_unique<hs::ShiftInvertOperator<T>>(*fac, pencil, sigma);
                        return json{{"s", s}, {"count", ib.neg - ia.neg}, {"found", found_in_window.size()},
                                    {"valid", ia.valid && ib.valid}};
                      }, {}});
  for (hs::FactorPrecision prec : {hs::FactorPrecision::Double, hs::FactorPrecision::Single}) {
    const bool single = prec == hs::FactorPrecision::Single;
    variants.push_back({single ? "exact_single" : "exact_counter", [&, prec, single]() {
                          hs::Timer t;
                          const auto [ia, ib] = counter.count_pair(1.0, -lower, 1.0, -upper, true, prec);
                          const double s = t.seconds();
                          const index_t count = ib.neg - ia.neg;
                          json j{{"s", s}, {"count", count}, {"found", found_in_window.size()},
                                 {"valid", ia.valid && ib.valid}};
                          if (single)
                            j["certified"] = single_rule.valid() && ia.valid && ib.valid &&
                                             count == static_cast<index_t>(found_in_window.size());
                          return j;
                        }, {}});
  }
  // Every exclusion_check call gets the next seed (a fresh, independent start block).
  std::uint64_t next_seed = 1;
  auto check = [&](const hs::DenseMatrix<T>& xx, double r_pass, double delta, index_t p) {
    const std::uint64_t seed = next_seed++;
    hs::ExclusionOptions o;
    o.pass_radius = r_pass;
    o.delta = delta;
    o.block_size = p;
    o.seed = seed;
    hs::set_start_scaling(o, scaling);  // S, kappa and whether kappa is proven
    hs::Timer t;
    const auto res = hs::exclusion_check<T>(*op, b_op.get(), xx, sigma, r, o);
    const double s = t.seconds();
    json j{{"s", s},
           {"setup_s", res.setup_s},
           {"krylov_s", res.krylov_s},
           {"status", hs::exclusion_status_name(res.status)},
           {"eps", res.epsilon},
           {"rigorous", res.rigorous},
           {"kappa", res.kappa},
           {"steps", res.block_steps},
           {"op_applications", res.op_applications},
           {"failure_probability", res.failure_probability},
           {"exact", res.exact},
           {"max_ritz", res.max_ritz},
           {"pass_threshold", res.pass_threshold},
           {"coupling", res.coupling},
           {"found_inside", res.found_inside},
           {"new_count", res.new_count},
           {"new_inside", res.new_inside},
           {"new_lambda", res.new_lambda}};
    return std::make_pair(j, res);
  };
  for (double delta : args.deltas)
    for (index_t p : args.blocks)
      for (const char* policy : {"window", "hint"}) {
        const double r_pass = std::string(policy) == "window" ? r_window : r_hint;
        std::ostringstream name;
        name << "randomized_d" << delta << "_p" << p << "_" << policy;
        variants.push_back({name.str(), [&, delta, p, r_pass]() { return check(x, r_pass, delta, p).first; }, {}});
      }
  if (args.removal) {
    // Largest multiplet among the found eigenvalues (relative tolerance 1e-8) and the farthest one.
    index_t best = -1, best_mult = 1;
    for (index_t i = 0; i < m; ++i) {
      index_t mult = 0;
      for (index_t j = 0; j < m; ++j)
        if (std::abs(lambda[j] - lambda[i]) <= 1e-8 * std::max(std::abs(lambda[i]), r)) ++mult;
      if (mult > best_mult) best_mult = mult, best = i;
    }
    index_t far = 0;
    for (index_t i = 0; i < m; ++i)
      if (std::abs(lambda[i] - sigma) > std::abs(lambda[far] - sigma)) far = i;
    std::vector<std::pair<std::string, index_t>> removals;
    if (best >= 0) removals.push_back({"copy_of_" + std::to_string(best_mult) + "fold", best});
    removals.push_back({"farthest", far});
    for (const auto& [what, col] : removals)
      for (double delta : args.deltas)
        for (index_t p : args.blocks) {
          std::ostringstream name;
          name << "removed_" << what << "_d" << delta << "_p" << p;
          const double removed = lambda[col];
          const std::string label = what;
          variants.push_back({name.str(), [&, col, delta, p, removed, label]() {
                                const hs::DenseMatrix<T> xr = drop_column(x, col);
                                auto [j, res] = check(xr, r_window, delta, p);
                                double err = std::numeric_limits<double>::infinity();
                                for (double l : res.new_lambda) err = std::min(err, std::abs(l - removed));
                                j["removed_lambda"] = removed;
                                j["detected"] = res.status == hs::ExclusionStatus::Detected;
                                j["new_lambda_error"] = err;
                                return j;
                              }, {}});
        }
  }

  for (int rep_i = 0; rep_i < args.reps; ++rep_i) {
    const std::size_t nv = variants.size();
    for (std::size_t k = 0; k < nv; ++k) {
      Variant& v = variants[(k + static_cast<std::size_t>(rep_i)) % nv];
      json j = v.run();
      j["exp"] = "rep";
      j["variant"] = v.name;
      j["rep"] = rep_i;
      emit(j);
      v.records.push_back(j);
    }
  }
  // ---- summaries
  for (const Variant& v : variants) {
    std::vector<double> s;
    for (const json& j : v.records) s.push_back(j.at("s").get<double>());
    json sum{{"exp", "summary"}, {"variant", v.name}, {"median_s", median(s)}, {"reps", v.records.size()}};
    const json& last = v.records.back();
    for (const char* key : {"status", "steps", "eps", "rigorous", "failure_probability", "op_applications", "count",
                            "found",
                            "valid", "certified", "detected", "new_lambda_error", "exact", "max_ritz",
                            "pass_threshold", "new_count", "new_inside"})
      if (last.contains(key)) sum[key] = last[key];
    if (last.contains("setup_s")) {
      std::vector<double> su, kr;
      for (const json& j : v.records) su.push_back(j.at("setup_s").get<double>()), kr.push_back(j.at("krylov_s").get<double>());
      sum["median_setup_s"] = median(su);
      sum["median_krylov_s"] = median(kr);
      bool all_same = true;
      for (const json& j : v.records) all_same = all_same && j.at("status") == last.at("status");
      sum["status_all_reps"] = all_same ? last.at("status").get<std::string>() : std::string("mixed");
    }
    emit(sum);
  }
  // ---- cost model prediction with the measured inputs (window policy)
  for (const char* policy : {"window", "hint"}) {
    hs::CertificateCostInputs in;
    in.n = n;
    in.deflated = m;
    in.complex_scalars = hs::is_complex_v<T>;
    in.factorize_s = factor_s;
    in.counter_analysis_s = counter_analysis_s;
    in.single_valid = single_rule.valid();
    in.apply1_s = t1;
    in.rhs_ratio = std::max(0.0, rhs_ratio);
    in.kappa = scaling.kappa;
    const double rp = std::string(policy) == "window" ? r_window : r_hint;
    // eps as exclusion_check computes it for a negligible coupling.
    in.epsilon = 1.0 - (1.0 / rp) / ((1.0 / r) * (1.0 - 1e-6));
    for (double delta : args.deltas) {
      in.delta = delta;
      const auto est = hs::certificate_cost(in);
      emit({{"exp", "cost_model"},
            {"policy", policy},
            {"delta", delta},
            {"eps", in.epsilon},
            {"rhs_ratio", in.rhs_ratio},
            {"exact_s", est.exact_s},
            {"exact_single_s", std::isinf(est.exact_single_s) ? -1.0 : est.exact_single_s},
            {"randomized_s", std::isinf(est.randomized_s) ? -1.0 : est.randomized_s},
            {"block_size", est.block_size},
            {"block_steps", est.block_steps},
            {"recommended", hs::certificate_method_name(est.recommended)}});
    }
  }
  return 0;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](int& j) -> std::string {
      if (j + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++j];
    };
    if (k == "--case") args.case_dir = need(i);
    else if (k == "--solution") args.solution = need(i);
    else if (k == "--threads") args.threads = std::stoi(need(i));
    else if (k == "--reps") args.reps = std::stoi(need(i));
    else if (k == "--removal") args.removal = std::stoi(need(i)) != 0;
    else if (k == "--deltas") {
      args.deltas.clear();
      for (const auto& s : split(need(i))) args.deltas.push_back(std::stod(s));
    } else if (k == "--blocks") {
      args.blocks.clear();
      for (const auto& s : split(need(i))) args.blocks.push_back(std::stoll(s));
    } else {
      std::cerr << "unknown argument " << k << "\n";
      return 2;
    }
  }
  if (args.case_dir.empty() || args.solution.empty()) {
    std::cerr << "usage: bench_certfast_measure --case DIR --solution DIR [--threads t] [--reps r] "
                 "[--deltas d1,d2] [--blocks p1,p2] [--removal 0|1]\n";
    return 2;
  }
  omp_set_num_threads(args.threads);
  try {
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, a, b);
    }
    auto to_c = [](hs::LoadedMatrix& mm) {
      return mm.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(mm.matrix)
                             : hs::to_complex(std::get<hs::CsrMatrix<double>>(mm.matrix));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(args, a, b ? &*b : nullptr);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
