// Measurements behind src/factor/inertia_counter.cpp (module certpar): what the inertia
// certificate's endpoint factorizations cost, and which MUMPS features make them cheaper.
//
//   bench_certpar_inertia --case DIR --lower a --upper b [--sigma s] [--ordering o]
//                         [--threads t] [--reps r] [--l0 0|1] [--eigs eigenvalues.npy] --exp E
//
// DIR holds A.npz (and B.npz) and config.json (sigma); [a, b] is the certificate window (from a
// hsolve report: certificate.lower/upper), o the ordering the driver chose (factor_options).
// Experiments (--exp, comma-separated):
//   solve       hs::solve on the case with its config.json: the driver's window, ordering and
//               phase times (the "before" reference of the certificate cost)
//   discard     ICNTL(31)=1 (discard factors) set before analysis vs only before factorization vs
//               off: factorization time, INFOG(16/17/18/21) and sampled RSS growth
//   permin      analysis of a second instance given the main analysis' SYM_PERM as PERM_IN
//               (ICNTL(7)=1) vs a fresh analysis (same ordering, and AMD); entries, flops and
//               factorization time of the resulting factors
//   concurrent  the two endpoint factorizations on two instances from two std::threads (each with
//               threads/2 OpenMP threads) vs one after the other with all threads, repeated: the
//               inertia must be identical in every round. --mumps-lock 0 calls MUMPS without
//               hs::mumps::call_mutex() (the library serializes all MUMPS calls): this is what
//               aborts in MUMPS_LOAD_INIT (sequential MUMPS is not thread-safe across instances)
//   single      smumps (float) vs dmumps endpoint counts (time, agreement), and shifts at
//               lambda +- d*||K||_1 (d = 1e-3 .. 1e-9) around the eigenvalues next to the window
//               ends: where do single-precision counts go wrong? (--eigs: reference spectrum)
//   blockpair   both endpoints in one factorization of diag(K(b), -K(a)) vs two counts
//   counter     hs::InertiaCounter (public API) vs the driver's path (refactorize the main object
//               at both endpoints), double and single precision, with the main factors resident
// Output: one JSON object per line on stdout. Run through tools/run/guard.py.
//
// Reproduction of the numbers quoted in include/hsolve/factor.hpp (2 threads, cores 36,37):
//   python3 tools/run/guard.py --skip-busy-check --cores 36,37 --mem-gb 8 -- \
//     build/release/bench_certpar_inertia --case data/cases/<case> --threads 2 --reps 3 \
//     --eigs data/refs/<case>/eigenvalues.npy --exp discard,permin,blockpair,single <window>
//   (--exp counter --reps 5; --exp concurrent [--mumps-lock 0]), with the windows and orderings
//   of the driver's certificates (hsolve reports, 2 threads):
//   syn-mem27-46     --lower -0.0009953284018943528 --upper 0.0012144683426627602 --ordering 3
//   yee-pec-cube-32  --lower 38.4130644402438 --upper 68.88929837661638 --ordering 3
//   ss-qa8fk-qa8fm   --lower 11.797166768925827 --upper 13.833100136337173 --ordering 3
//   wg-square-1e5    --lower -1.02 --upper -0.05 --ordering 0   (GHIEP: timing of K only; the
//                    discard/permin/blockpair/single runs used --lower -0.9994466288878361
//                    --upper -0.05501589913473426, which lie on eigenvalues)
//   syn-near-singular-complex-1e5  --sigma 0.29999367606763133 --lower 0.29986915072174825
//                    --upper 0.30012121734172936 --ordering 0
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "factor/mumps_api.hpp"
#include "factor/mumps_input.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Args {
  fs::path case_dir;
  std::vector<std::string> exps;
  double sigma = kNaN, lower = kNaN, upper = kNaN;
  int ordering = 3;
  int threads = 0;
  int reps = 3;
  bool l0 = true;  // ICNTL(48)=1, the driver default (SolverConfig::deterministic = false)
  fs::path eigs;
  bool mumps_lock = true;
};

// --mumps-lock 0: the concurrent experiment bypasses hs::mumps::call_mutex().
bool g_mumps_lock = true;

void emit(json j) { std::cout << j.dump() << std::endl; }

// Peak RSS growth (MB = 1e6 B) over the lifetime of the object, sampled every 2 ms.
class RssPeak {
 public:
  RssPeak() : base_(hs::current_rss_bytes()), peak_(base_), th_([this] { loop(); }) {}
  ~RssPeak() { finish(); }
  double finish() {
    if (th_.joinable()) {
      stop_ = true;
      th_.join();
    }
    return static_cast<double>(peak_ - base_) / 1e6;
  }

 private:
  void loop() {
    while (!stop_) {
      peak_ = std::max<std::int64_t>(peak_, hs::current_rss_bytes());
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    peak_ = std::max<std::int64_t>(peak_, hs::current_rss_bytes());
  }
  std::int64_t base_;
  std::int64_t peak_;  // written by th_ only, read after join()
  std::atomic<bool> stop_{false};
  std::thread th_;
};

struct Fact {
  double seconds = 0.0;
  int info1 = 0;
  index_t neg = -1, nulls = -1;
  double mb_used = 0.0, mb_alloc = 0.0, rss_mb = 0.0, flops = 0.0;
  std::int64_t entries = 0;
  json to_json() const {
    return {{"s", seconds}, {"info1", info1}, {"neg", neg},          {"null", nulls},
            {"infog21_mb", mb_used}, {"infog18_mb", mb_alloc}, {"rss_mb", rss_mb},
            {"entries", entries}, {"flops", flops}};
  }
};

struct RawConfig {
  int ordering = 3;
  int threads = 2;
  bool l0 = true;
  bool discard_at_analysis = false;
  bool discard_at_factorization = false;
};

// One raw MUMPS instance on the pencil (same controls as src/factor/mumps_factorization.cpp).
template <class T, class U>
class Raw {
 public:
  static constexpr int kEmb = hs::is_complex_v<T> ? 2 : 1;

  Raw(const hs::ShiftedPencil<T>& pencil, const RawConfig& c) : in_(pencil), cfg_(c) {
    m_.icntl(1) = -1;
    m_.icntl(2) = -1;
    m_.icntl(3) = -1;
    m_.icntl(4) = 0;
    m_.icntl(5) = 0;
    m_.icntl(7) = c.ordering;
    m_.icntl(8) = 7;
    m_.icntl(13) = 1;
    m_.icntl(14) = 35;
    m_.icntl(16) = c.threads;
    m_.icntl(18) = 0;
    m_.icntl(20) = 0;
    m_.icntl(21) = 0;
    m_.icntl(24) = 1;
    m_.cntl(3) = static_cast<U>(std::is_same_v<U, float> ? 1e-4 : 1e-8);
    m_.icntl(31) = c.discard_at_analysis ? 1 : 0;
    m_.icntl(48) = c.l0 ? 1 : 0;
    auto& s = m_.raw();
    s.n = in_.order();
    s.nz = 0;
    s.nnz = in_.nnz();
    s.irn = in_.irn();
    s.jcn = in_.jcn();
    s.a = in_.values();
  }

  double analyze(double alpha, double beta, const std::vector<int>* perm = nullptr) {
    in_.assemble(alpha, beta);
    if (perm) {
      perm_ = *perm;
      m_.icntl(7) = 1;
      m_.raw().perm_in = perm_.data();
    }
    hs::Timer t;
    const int info = m_.run(hs::mumps::kAnalyze);
    const double sec = t.seconds();
    if (info < 0) HS_THROW(FactorizationFailed, "analysis INFO(1)=" << info << " INFO(2)=" << m_.infog(2));
    return sec;
  }

  std::vector<int> sym_perm() {
    const auto& s = m_.raw();
    return std::vector<int>(s.sym_perm, s.sym_perm + s.n);
  }

  void set_threads(int t) { m_.icntl(16) = t; }

  Fact factorize(double alpha, double beta, bool sample = true) {
    in_.assemble(alpha, beta);
    if (cfg_.discard_at_factorization) m_.icntl(31) = 1;
    std::unique_ptr<RssPeak> rss;
    if (sample) rss = std::make_unique<RssPeak>();
    Fact f;
    hs::Timer t;
    if (g_mumps_lock) {
      f.info1 = m_.run(hs::mumps::kFactorize);
    } else {
      m_.raw().job = hs::mumps::kFactorize;
      hs::mumps::Api<U>::call(&m_.raw());
      f.info1 = m_.infog(1);
    }
    f.seconds = t.seconds();
    if (rss) f.rss_mb = rss->finish();
    f.neg = m_.infog(12) / kEmb;
    f.nulls = (m_.infog(28) + kEmb - 1) / kEmb;
    f.mb_used = m_.infog(21);
    f.mb_alloc = m_.infog(18);
    f.entries = hs::mumps::decode_count(m_.infog(9));
    f.flops = m_.rinfog(3);
    return f;
  }

  json estimates() {
    return {{"infog16_incore_mb", m_.infog(16)}, {"infog17_incore_total_mb", m_.infog(17)},
            {"infog26_ooc_mb", m_.infog(26)},   {"infog20_entries", hs::mumps::decode_count(m_.infog(20))},
            {"rinfog1_flops", m_.rinfog(1)}};
  }

  hs::mumps::Instance<U>& mumps() { return m_; }

 private:
  hs::mumps::Instance<U> m_;
  hs::mumps::Input<T, U> in_;
  RawConfig cfg_;
  std::vector<int> perm_;
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? kNaN : v[v.size() / 2];
}

// ------------------------------------------------------------------------------- experiments

template <class T>
struct Ctx {
  const Args& args;
  const hs::CsrMatrix<T>& a;
  const hs::CsrMatrix<T>* b;
  const hs::ShiftedPencil<T>& pencil;
  int threads;
  RawConfig base() const {
    RawConfig c;
    c.ordering = args.ordering;
    c.threads = threads;
    c.l0 = args.l0;
    return c;
  }
};

template <class T>
void exp_solve(const Ctx<T>& c, const hs::SolverConfig& cfg0) {
  hs::SolverConfig cfg = cfg0;
  cfg.threads = c.threads;
  hs::Timer t;
  const auto sol = hs::solve<T>(c.a, c.b, cfg);
  const auto& r = sol.report;
  emit({{"exp", "solve"},
        {"wall_s", t.seconds()},
        {"time_total_s", r.time_total_s},
        {"analyze_s", r.factor_stats.analyze_s},
        {"factorize_s", r.factor_stats.factorize_s},
        {"factorizations", r.factor_stats.factorizations},
        {"solve_s", r.factor_stats.solve_s},
        {"ordering", r.factor_options.ordering},
        {"sigma_used", r.sigma_used},
        {"lower", r.certificate.lower},
        {"upper", r.certificate.upper},
        {"count_inertia", r.certificate.count_inertia},
        {"passed", r.certificate.passed},
        {"success", r.success}});
}

// (1) ICNTL(31)=1: when must it be set, and what does it save? Every variant is analysed with
// the same pivot order (PERM_IN from a first analysis): two SCOTCH analyses in one process give
// different orderings (SCOTCH's random state is not reset between calls).
template <class T>
void exp_discard(const Ctx<T>& c) {
  const double sigma = c.args.sigma, lo = c.args.lower, hi = c.args.upper;
  std::vector<int> perm;
  {
    Raw<T, double> main(c.pencil, c.base());
    main.analyze(1.0, -sigma);
    perm = main.sym_perm();
  }
  struct Variant {
    const char* name;
    bool at_analysis, at_factorization;
  };
  for (const Variant v : {Variant{"keep", false, false}, Variant{"discard_at_analysis", true, false},
                          Variant{"discard_at_factorization", false, true}}) {
    RawConfig rc = c.base();
    rc.discard_at_analysis = v.at_analysis;
    rc.discard_at_factorization = v.at_factorization;
    RssPeak life;  // RSS growth over the instance's whole life (construction .. last count)
    auto r = std::make_unique<Raw<T, double>>(c.pencil, rc);
    const double ta = r->analyze(1.0, -sigma, &perm);
    json est = r->estimates();
    std::vector<double> tl, th;
    Fact fl, fh;
    for (int rep = 0; rep < c.args.reps; ++rep) {
      fl = r->factorize(1.0, -lo, false);
      fh = r->factorize(1.0, -hi, false);
      tl.push_back(fl.seconds);
      th.push_back(fh.seconds);
    }
    const double life_mb = life.finish();
    // Does a solve after a discarding factorization fail cleanly?
    int solve_info = 0;
    {
      const index_t n = c.pencil.order() * Raw<T, double>::kEmb;
      std::vector<double> rhs(static_cast<std::size_t>(n), 1.0);
      auto& s = r->mumps().raw();
      s.nrhs = 1;
      s.lrhs = static_cast<int>(n);
      s.rhs = rhs.data();
      solve_info = r->mumps().run(hs::mumps::kSolve);
      s.rhs = nullptr;
    }
    emit({{"exp", "discard"}, {"variant", v.name}, {"analyze_s", ta}, {"estimates", est},
          {"lower", fl.to_json()}, {"upper", fh.to_json()}, {"lower_median_s", median(tl)},
          {"upper_median_s", median(th)}, {"pair_median_s", median(tl) + median(th)},
          {"rss_life_mb", life_mb}, {"solve_after_info1", solve_info}});
  }
}

// (2) Second instance analysed with the main instance's pivot order.
template <class T>
void exp_permin(const Ctx<T>& c) {
  const double sigma = c.args.sigma, lo = c.args.lower;
  RawConfig rc = c.base();
  Raw<T, double> main(c.pencil, rc);
  const double t_main = main.analyze(1.0, -sigma);
  const json est_main = main.estimates();
  hs::Timer tp;
  const std::vector<int> perm = main.sym_perm();
  const double t_copy = tp.seconds();
  const Fact f_main = main.factorize(1.0, -lo);

  std::vector<double> t_perm, t_fresh, t_amd;
  json est_perm, est_amd;
  Fact f_perm, f_amd;
  for (int rep = 0; rep < c.args.reps; ++rep) {
    {
      RawConfig pc = rc;
      pc.discard_at_analysis = true;
      Raw<T, double> p(c.pencil, pc);
      t_perm.push_back(p.analyze(1.0, -lo, &perm));
      est_perm = p.estimates();
      if (rep == 0) f_perm = p.factorize(1.0, -lo);
    }
    {
      Raw<T, double> f(c.pencil, rc);
      t_fresh.push_back(f.analyze(1.0, -lo));
    }
    {
      RawConfig ac = rc;
      ac.ordering = 0;
      ac.discard_at_analysis = true;
      Raw<T, double> f(c.pencil, ac);
      t_amd.push_back(f.analyze(1.0, -lo));
      est_amd = f.estimates();
      if (rep == 0) f_amd = f.factorize(1.0, -lo);
    }
  }
  emit({{"exp", "permin"},
        {"main_analyze_s", t_main},
        {"sym_perm_copy_s", t_copy},
        {"permin_analyze_median_s", median(t_perm)},
        {"fresh_analyze_median_s", median(t_fresh)},
        {"amd_analyze_median_s", median(t_amd)},
        {"estimates_main", est_main},
        {"estimates_permin", est_perm},
        {"estimates_amd", est_amd},
        {"factor_main_keep", f_main.to_json()},
        {"factor_permin_discard", f_perm.to_json()},
        {"factor_amd_discard", f_amd.to_json()},
        {"same_inertia", f_main.neg == f_perm.neg && f_main.neg == f_amd.neg}});
}

// (3) Two endpoint factorizations concurrently from two std::threads.
template <class T>
void exp_concurrent(const Ctx<T>& c) {
  const double sigma = c.args.sigma, lo = c.args.lower, hi = c.args.upper;
  const int t_all = c.threads, t_half = std::max(1, c.threads / 2);
  RawConfig rc = c.base();
  std::vector<int> perm;
  {
    Raw<T, double> main(c.pencil, rc);
    main.analyze(1.0, -sigma);
    perm = main.sym_perm();
  }
  for (const bool discard : {true, false}) {
    // With L0-OpenMP (ICNTL(48)=1) the factorization must use the thread count of the analysis
    // (else INFO(1) = -58), so the sequential and the concurrent instances are analysed apart.
    RawConfig pc = rc;
    pc.discard_at_analysis = discard;
    Raw<T, double> p1(c.pencil, pc), p2(c.pencil, pc);
    p1.analyze(1.0, -lo, &perm);
    p2.analyze(1.0, -hi, &perm);
    RawConfig hc = pc;
    hc.threads = t_half;
    Raw<T, double> q1(c.pencil, hc), q2(c.pencil, hc);
    q1.analyze(1.0, -lo, &perm);
    q2.analyze(1.0, -hi, &perm);
    std::vector<double> t_seq, t_conc, t_th1, t_th2;
    index_t ref_lo = -2, ref_hi = -2;
    bool consistent = true;
    json bad = json::array();
    double rss_seq = 0, rss_conc = 0;
    for (int rep = 0; rep < c.args.reps; ++rep) {
      {
        RssPeak rss;
        hs::Timer t;
        const Fact a1 = p1.factorize(1.0, -lo, false);
        const Fact a2 = p2.factorize(1.0, -hi, false);
        t_seq.push_back(t.seconds());
        rss_seq = std::max(rss_seq, rss.finish());
        if (rep == 0) ref_lo = a1.neg, ref_hi = a2.neg;
        consistent = consistent && a1.neg == ref_lo && a2.neg == ref_hi && a1.info1 >= 0 && a2.info1 >= 0;
      }
      {
        RssPeak rss;
        Fact b1, b2;
        hs::Timer t;
        std::thread th1([&] {
          omp_set_num_threads(t_half);
          b1 = q1.factorize(1.0, -lo, false);
        });
        std::thread th2([&] {
          omp_set_num_threads(t_half);
          b2 = q2.factorize(1.0, -hi, false);
        });
        th1.join();
        th2.join();
        t_conc.push_back(t.seconds());
        rss_conc = std::max(rss_conc, rss.finish());
        t_th1.push_back(b1.seconds);
        t_th2.push_back(b2.seconds);
        const bool ok = b1.neg == ref_lo && b2.neg == ref_hi && b1.info1 >= 0 && b2.info1 >= 0;
        if (!ok) bad.push_back({{"round", rep}, {"t1", b1.to_json()}, {"t2", b2.to_json()}});
        consistent = consistent && ok;
      }
    }
    emit({{"exp", "concurrent"}, {"mumps_lock", g_mumps_lock}, {"discard", discard}, {"threads", t_all}, {"threads_each", t_half},
          {"seq_median_s", median(t_seq)}, {"conc_median_s", median(t_conc)},
          {"conc_thread1_median_s", median(t_th1)}, {"conc_thread2_median_s", median(t_th2)},
          {"seq_all", t_seq}, {"conc_all", t_conc}, {"rss_seq_mb", rss_seq}, {"rss_conc_mb", rss_conc},
          {"neg_lower", ref_lo}, {"neg_upper", ref_hi}, {"consistent", consistent},
          {"inconsistent_rounds", bad.size()}, {"first_bad", bad.empty() ? json() : bad[0]}});
  }
}

std::vector<double> read_eigs(const fs::path& p) {
  if (p.empty()) return {};
  auto arr = hs::read_npy(p);
  std::vector<double> w;
  try {
    w = hs::npy_to_vector<double>(arr);
  } catch (const hs::Error&) {
    for (const auto& z : hs::npy_to_vector<hs::cplx>(arr)) w.push_back(z.real());
  }
  std::sort(w.begin(), w.end());
  return w;
}

// (4) Single precision counts.
template <class T>
void exp_single(const Ctx<T>& c) {
  const double sigma = c.args.sigma, lo = c.args.lower, hi = c.args.upper;
  RawConfig rc = c.base();
  rc.discard_at_analysis = true;
  std::vector<int> perm;
  {
    Raw<T, double> main(c.pencil, c.base());
    main.analyze(1.0, -sigma);
    perm = main.sym_perm();
  }
  Raw<T, double> d(c.pencil, rc);
  Raw<T, float> s(c.pencil, rc);
  d.analyze(1.0, -lo, &perm);
  s.analyze(1.0, -lo, &perm);
  std::vector<double> td, ts;
  Fact fd_lo, fd_hi, fs_lo, fs_hi;
  for (int rep = 0; rep < c.args.reps; ++rep) {
    fd_lo = d.factorize(1.0, -lo);
    fd_hi = d.factorize(1.0, -hi);
    fs_lo = s.factorize(1.0, -lo);
    fs_hi = s.factorize(1.0, -hi);
    td.push_back(fd_lo.seconds + fd_hi.seconds);
    ts.push_back(fs_lo.seconds + fs_hi.seconds);
  }
  emit({{"exp", "single"}, {"double_pair_median_s", median(td)}, {"single_pair_median_s", median(ts)},
        {"double_lower", fd_lo.to_json()}, {"double_upper", fd_hi.to_json()},
        {"single_lower", fs_lo.to_json()}, {"single_upper", fs_hi.to_json()},
        {"agree", fd_lo.neg == fs_lo.neg && fd_hi.neg == fs_hi.neg && fs_lo.nulls == 0 && fs_hi.nulls == 0}});

  // Accuracy sweep around the eigenvalues next to the window ends.
  const std::vector<double> w = read_eigs(c.args.eigs);
  if (w.empty()) return;
  const double knorm = hs::norm_one(c.pencil.assemble_full(1.0, -sigma));
  std::vector<double> lams;
  {
    auto first_ge = std::lower_bound(w.begin(), w.end(), lo);
    auto first_gt_hi = std::upper_bound(w.begin(), w.end(), hi);
    if (first_ge != w.end()) lams.push_back(*first_ge);                 // innermost at the lower end
    if (first_ge != w.begin()) lams.push_back(*(first_ge - 1));         // outside, below the window
    if (first_gt_hi != w.begin()) lams.push_back(*(first_gt_hi - 1));   // innermost at the upper end
    if (first_gt_hi != w.end()) lams.push_back(*first_gt_hi);           // outside, above the window
  }
  for (const double lam : lams) {
    for (const double dist : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9}) {
      for (const double sgn : {-1.0, 1.0}) {
        const double shift = lam + sgn * dist * knorm;
        const index_t ref = std::lower_bound(w.begin(), w.end(), shift) - w.begin();
        const Fact a = d.factorize(1.0, -shift, false);
        const Fact b = s.factorize(1.0, -shift, false);
        // the gap from the shift to the nearest reference eigenvalue, relative to ||K||_1
        auto it = std::lower_bound(w.begin(), w.end(), shift);
        double gap = std::numeric_limits<double>::infinity();
        if (it != w.end()) gap = std::min(gap, *it - shift);
        if (it != w.begin()) gap = std::min(gap, shift - *(it - 1));
        emit({{"exp", "single_sweep"}, {"lambda", lam}, {"d", sgn * dist}, {"rel_gap", gap / knorm},
              {"knorm1", knorm}, {"ref_count", ref}, {"double_neg", a.neg}, {"double_null", a.nulls},
              {"single_neg", b.neg}, {"single_null", b.nulls}, {"double_ok", a.nulls == 0 && a.neg == ref},
              {"single_ok", b.nulls == 0 && b.neg == ref}, {"single_s", b.seconds}, {"double_s", a.seconds}});
      }
    }
  }
}

// (3b) Both endpoints in ONE factorization of the block-diagonal D = diag(K(b), -K(a)):
// neg(D) = neg(K(b)) + pos(K(a)) = nu(b) + m - nu(a) - zero(a), so the window count is
// neg(D) - m when D is nonsingular. The two blocks form two independent elimination trees that
// L0-OpenMP can factorize in parallel, inside one (thread-safe) MUMPS instance. Pivot order: the
// main analysis' order for block 1, the same order shifted by m for block 2.
template <class T>
void exp_blockpair(const Ctx<T>& c) {
  const double sigma = c.args.sigma, lo = c.args.lower, hi = c.args.upper;
  std::vector<int> perm;
  {
    Raw<T, double> main(c.pencil, c.base());
    main.analyze(1.0, -sigma);
    perm = main.sym_perm();
  }
  // Reference: the sequential pair on one discarding instance with all threads.
  RawConfig dc = c.base();
  dc.discard_at_analysis = true;
  Raw<T, double> seq(c.pencil, dc);
  seq.analyze(1.0, -lo, &perm);
  std::vector<double> t_seq;
  Fact f_lo, f_hi;
  for (int rep = 0; rep < c.args.reps; ++rep) {
    hs::Timer t;
    f_lo = seq.factorize(1.0, -lo, false);
    f_hi = seq.factorize(1.0, -hi, false);
    t_seq.push_back(t.seconds());
  }

  hs::mumps::Input<T, double> in(c.pencil);
  const int m = in.order();
  const std::int64_t nz = in.nnz();
  std::vector<int> irn(2 * nz), jcn(2 * nz), perm2(2 * static_cast<std::size_t>(m));
  std::vector<double> val(2 * nz);
  for (std::int64_t p = 0; p < nz; ++p) {
    irn[p] = in.irn()[p];
    jcn[p] = in.jcn()[p];
    irn[nz + p] = in.irn()[p] + m;
    jcn[nz + p] = in.jcn()[p] + m;
  }
  for (int i = 0; i < m; ++i) {
    perm2[i] = perm[i];
    perm2[m + i] = perm[i] + m;
  }
  auto fill = [&](double a_hi, double a_lo) {
    in.assemble(1.0, -a_hi);
    std::copy(in.values(), in.values() + nz, val.begin());
    in.assemble(1.0, -a_lo);
    for (std::int64_t p = 0; p < nz; ++p) val[nz + p] = -in.values()[p];
  };
  hs::mumps::Instance<double> mm;
  mm.icntl(1) = -1;
  mm.icntl(2) = -1;
  mm.icntl(3) = -1;
  mm.icntl(4) = 0;
  mm.icntl(5) = 0;
  mm.icntl(7) = 1;
  mm.icntl(8) = 7;
  mm.icntl(13) = 1;
  mm.icntl(14) = 35;
  mm.icntl(16) = c.threads;
  mm.icntl(24) = 1;
  mm.cntl(3) = 1e-8;
  mm.icntl(31) = 1;
  mm.icntl(48) = c.args.l0 ? 1 : 0;
  auto& st = mm.raw();
  st.n = 2 * m;
  st.nz = 0;
  st.nnz = 2 * nz;
  st.irn = irn.data();
  st.jcn = jcn.data();
  st.a = val.data();
  st.perm_in = perm2.data();
  fill(hi, lo);
  hs::Timer ta;
  const int ainfo = mm.run(hs::mumps::kAnalyze);
  const double t_ana = ta.seconds();
  std::vector<double> t_blk, t_fill;
  index_t count = -1, nulls = -1;
  int info = 0;
  double mb = 0;
  for (int rep = 0; rep < c.args.reps; ++rep) {
    hs::Timer tf;
    fill(hi, lo);
    t_fill.push_back(tf.seconds());
    hs::Timer t;
    info = mm.run(hs::mumps::kFactorize);
    t_blk.push_back(t.seconds() + t_fill.back());
    count = mm.infog(12) - m;
    nulls = mm.infog(28);
    mb = mm.infog(21);
  }
  const int emb = Raw<T, double>::kEmb;
  emit({{"exp", "blockpair"}, {"seq_pair_median_s", median(t_seq)}, {"block_pair_median_s", median(t_blk)},
        {"block_fill_median_s", median(t_fill)}, {"block_analyze_s", t_ana}, {"block_analyze_info", ainfo},
        {"block_info1", info}, {"block_count", count / emb}, {"block_nulls", nulls}, {"block_infog21_mb", mb},
        {"seq_count", f_hi.neg - f_lo.neg}, {"seq_infog21_mb", f_lo.mb_used},
        {"agree", count / emb == f_hi.neg - f_lo.neg && nulls == 0}});
}

template <class T>
void exp_counter(const Ctx<T>& c);

template <class T>
int run(Args args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  const int threads = args.threads > 0 ? args.threads : omp_get_max_threads();
  hs::SolverConfig cfg;
  {
    std::ifstream f(args.case_dir / "config.json");
    std::stringstream ss;
    ss << f.rdbuf();
    cfg = hs::parse_config(ss.str());
  }
  if (std::isnan(args.sigma)) args.sigma = cfg.target.sigma;
  const Ctx<T> c{args, a, b, pencil, threads};
  emit({{"exp", "setup"}, {"case", args.case_dir.filename().string()}, {"n", pencil.order()},
        {"nnz_upper", pencil.nnz()}, {"complex", hs::is_complex_v<T>}, {"threads", threads},
        {"ordering", args.ordering}, {"l0", args.l0}, {"sigma", args.sigma}, {"lower", args.lower},
        {"upper", args.upper}});
  for (const auto& e : args.exps) {
    if (e == "solve") {
      exp_solve(c, cfg);
      continue;
    }
    if (std::isnan(args.lower) || std::isnan(args.upper))
      HS_THROW(InvalidInput, "--lower and --upper are required for --exp " << e);
    if (e == "discard") exp_discard(c);
    else if (e == "permin") exp_permin(c);
    else if (e == "concurrent") exp_concurrent(c);
    else if (e == "single") exp_single(c);
    else if (e == "blockpair") exp_blockpair(c);
    else if (e == "counter") exp_counter(c);
    else HS_THROW(InvalidInput, "unknown experiment " << e);
  }
  return 0;
}

// (5) The facility through the public API, with the main factors resident as in the driver.
template <class T>
void exp_counter(const Ctx<T>& c) {
  const double sigma = c.args.sigma, lo = c.args.lower, hi = c.args.upper;
  hs::FactorOptions o;
  o.ordering = c.args.ordering;
  o.threads = c.threads;
  o.deterministic = !c.args.l0;
  auto fac = hs::make_mumps_factorization<T>(o);
  fac->analyze(c.pencil, 1.0, -sigma);
  if (!fac->factorize(1.0, -sigma)) HS_THROW(Singular, "main factorization at sigma failed");
  const double t_main = fac->stats().factorize_s;
  const double main_mb = fac->stats().factor_mb_used;

  // Before: the driver's path refactorizes the main object at both endpoints (and again at
  // sigma when a deflated rerun follows). After: an InertiaCounter on the analysed main object
  // (double, then single). The three are interleaved per repetition (A B C A B C ...) so that
  // drifting machine load biases none of them.
  hs::Timer tc;
  RssPeak rss_construct;
  hs::InertiaCounter<T> counter(*fac, o);
  const double t_construct = tc.seconds();
  double rss_after = rss_construct.finish();
  std::vector<double> t_before, t_after, t_single;
  hs::Inertia b_lo, b_hi;
  std::pair<hs::Inertia, hs::Inertia> a, s;
  double rss_before = 0;
  for (int rep = 0; rep < c.args.reps; ++rep) {
    {
      RssPeak rss;
      hs::Timer t;
      static_cast<void>(fac->factorize(1.0, -lo));
      b_lo = fac->inertia();
      static_cast<void>(fac->factorize(1.0, -hi));
      b_hi = fac->inertia();
      t_before.push_back(t.seconds());
      rss_before = std::max(rss_before, rss.finish());
    }
    {
      RssPeak rss;
      hs::Timer t;
      a = counter.count_pair(1.0, -lo, 1.0, -hi);
      t_after.push_back(t.seconds());
      rss_after = std::max(rss_after, rss.finish());
    }
    {
      hs::Timer t;
      s = counter.count_pair(1.0, -lo, 1.0, -hi, true, hs::FactorPrecision::Single);
      t_single.push_back(t.seconds());
    }
  }
  // Release variant: free the main factors first (JOB=-4), so that the counter's working memory
  // replaces them instead of adding to them; RSS is then measured against the factored state.
  double rss_release = 0, t_release_pair = 0;
  bool released = false;
  {
    static_cast<void>(fac->factorize(1.0, -sigma));  // main factors resident again
    RssPeak rss;
    hs::Timer t;
    released = fac->release_factors();
    const auto r = counter.count_pair(1.0, -lo, 1.0, -hi);
    t_release_pair = t.seconds();
    rss_release = rss.finish();
    if (r.first.neg != a.first.neg || r.second.neg != a.second.neg) HS_THROW(Internal, "release variant disagrees");
  }
  const auto& st = counter.stats();
  emit({{"exp", "counter"},
        {"main_factorize_s", t_main},
        {"main_infog21_mb", main_mb},
        {"before_pair_median_s", median(t_before)},
        {"after_construct_s", t_construct},
        {"after_pair_median_s", median(t_after)},
        {"after_total_s", t_construct + median(t_after)},
        {"single_pair_median_s", median(t_single)},
        {"before_all", t_before},
        {"after_all", t_after},
        {"single_all", t_single},
        {"rss_before_mb", rss_before},
        {"rss_after_mb", rss_after},
        {"released", released},
        {"release_pair_s", t_release_pair},
        {"rss_release_mb", rss_release},
        {"counter_estimate_mb", counter.estimate().incore_mb},
        {"counter_max_mb_used", st.max_mb_used},
        {"counter_analyze_s", st.analyze_s},
        {"reused_pivot_order", st.reused_pivot_order},
        {"count_before", b_hi.neg - b_lo.neg},
        {"count_after", a.second.neg - a.first.neg},
        {"count_single", s.second.neg - s.first.neg},
        {"valid", b_lo.valid && b_hi.valid && a.first.valid && a.second.valid},
        {"agree", b_lo.neg == a.first.neg && b_hi.neg == a.second.neg},
        {"single_agree", s.first.neg == a.first.neg && s.second.neg == a.second.neg && s.first.valid &&
                             s.second.valid}});
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << argv[i] << "\n";
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--case") args.case_dir = need(i);
    else if (k == "--exp") {
      std::stringstream ss(need(i));
      for (std::string e; std::getline(ss, e, ',');) args.exps.push_back(e);
    } else if (k == "--sigma") args.sigma = std::stod(need(i));
    else if (k == "--lower") args.lower = std::stod(need(i));
    else if (k == "--upper") args.upper = std::stod(need(i));
    else if (k == "--ordering") args.ordering = std::stoi(need(i));
    else if (k == "--threads") args.threads = std::stoi(need(i));
    else if (k == "--reps") args.reps = std::stoi(need(i));
    else if (k == "--l0") args.l0 = std::stoi(need(i)) != 0;
    else if (k == "--eigs") args.eigs = need(i);
    else if (k == "--mumps-lock") args.mumps_lock = std::stoi(need(i)) != 0;
    else {
      std::cerr << "unknown argument " << k << "\n";
      return 2;
    }
  }
  g_mumps_lock = args.mumps_lock;
  if (args.case_dir.empty() || args.exps.empty()) {
    std::cerr << "usage: bench_certpar_inertia --case DIR --exp E[,E...] [--lower a --upper b] ...\n";
    return 2;
  }
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
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(m.matrix)
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(m.matrix));
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
