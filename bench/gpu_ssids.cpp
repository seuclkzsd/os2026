// GPU experiment G1 (results/gpu_g1.md): SPRAL SSIDS - multifrontal LDL^T with inertia, OpenMP CPU
// path and CUDA path - against hsolve's MUMPS backend on the shift-invert matrix K = A - sigma B of a
// case: analysis, factorizations, inertia, solves with 1 and p right-hand sides, accuracy, memory.
// A measurement prototype: SSIDS is not part of hsolve. Built only with -DHSOLVE_WITH_SPRAL=ON
// (SPRAL from tools/dev/build_spral.sh, CUDA toolkit for the driver API and NVML).
//
//   bench_gpu_ssids --case DIR --backend mumps|ssids-cpu|ssids-gpu [--ordering O] [--sigma s]
//       [--threads t] [--factorizations F] [--solves S] [--block p] [--block-solves P]
//       [--u 0.01] [--scaling 4] [--pivot-method 2] [--nemin 32] [--min-gpu-work 5e9]
//       [--gpu-perf-coeff 1] [--max-load-inbalance 1.2] [--memory-budget-gb 14] [--analyze-only 1]
//
// DIR holds A.npz (and B.npz) and config.json (sigma = target.sigma unless --sigma; k = target.k).
// K is assembled exactly as the driver's MUMPS input: ShiftedPencil::assemble(1, -sigma) on the
// pencil's upper pattern (row i holds columns >= i), which is at the same time the lower-triangle
// CSC that SSIDS takes (handed over 1-based, converted once outside the timings). Real cases only.
// One backend per process, so that the host peak RSS belongs to it:
//   mumps      MumpsFactorization with the driver's FactorOptions (factor_options_from: t threads,
//              L0-OpenMP, ICNTL(8)=7 scaling, null-pivot detection CNTL(3)=1e-8, the driver's
//              ICNTL(23) cap for SolverConfig::memory_budget_gb = --memory-budget-gb, default 14
//              GB like the driver; the 1e6 3-D case needs 30). --ordering driver (default): the
//              driver's adaptive choice (src/driver/steps.cpp adaptive_analysis: AMD probe,
//              decide_ordering with the work the driver plans - 1 + 2 certificate (+1 B test)
//              factorizations, max(40, 3k) solves - and SCOTCH when nested dissection pays);
//              analyse_s includes the probe. amd | scotch | metis: ICNTL(7) = 0 / 3 / 5.
//   ssids-cpu  SSIDS with use_gpu = false on one region of t threads (explicit topology, no CUDA
//              call at all; run with CUDA_VISIBLE_DEVICES="").
//   ssids-gpu  SSIDS with use_gpu = true on one region of t threads + CUDA device 0 (gpu02: run with
//              CUDA_VISIBLE_DEVICES=7). cuda_init_s = cuInit + creation of the device's primary
//              context (which SSIDS' CUDA runtime then uses), timed apart from analyse_s; it is part
//              of an end-to-end run all the same.
//   SSIDS orderings: metis (default; SSIDS' own METIS_NodeND call) | mumps (the pivot order SYM_PERM
//   of the mumps backend's driver analysis, given as SSIDS user order: the same elimination order as
//   the reference; that MUMPS analysis runs untimed and is freed before SSIDS starts).
//   SSIDS options: threshold u = 0.01 (MUMPS' CNTL(1) default for symmetric matrices is 0.01 too),
//   scaling 4 = Ruiz equilibration (MUMPS' ICNTL(8)=7 is also iterative row/column scaling),
//   pivot_method 2 (block a posteriori, SSIDS default), action = true (zero pivots are counted,
//   zero = n - matrix_rank), nemin 32, min_gpu_work 5e9 flops, gpu_perf_coeff 1 (SSIDS defaults).
//   SSIDS needs OMP_CANCELLATION=TRUE (else flag -53) and OMP_PROC_BIND=TRUE (else warning +50) in
//   the environment of the process.
// Steps: analysis; F factorizations, all at sigma (the first is cold: first touch, CUDA module
// loading, cuBLAS handles; the later ones warm) with their inertia (neg, zero); S solves with one
// right-hand side and P with p (median times; solve1_first_s is the first one-column solve) of
// random right-hand sides (fixed seed) on the last factor. Accuracy: the normwise backward error
//   resid = ||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)
// of the first one-column solve and the worst column of the first p-column solve. Memory: host
// peak RSS of the backend phase (VmHWM reset through /proc/self/clear_refs after loading, before
// the CUDA context; it includes the resident matrices, rss_load_gib) and, for ssids-gpu, the GPU's
// used memory sampled every 5 ms through NVML (device-wide, so the GPU must be otherwise idle):
// after the context, peak, and after the factorizations. --analyze-only 1 stops after the analysis
// (estimates only).
// Output: one JSON line on stdout. Run through tools/run/guard.py; the G1 numbers in
// results/gpu_g1.md come from tools/bench/gpu_g1.sh (one process per backend variant, variants and
// repetitions interleaved, every process in gpu02's measurement slot), e.g.
//   tools/dev/cluster_run.sh gpu02 -- tools/bench/gpu_g1.sh --reps 3 syn-mem27-46 yee-pec-box-1e5
//   (the exact invocations: results/gpu_g1.md, "Environment and commands")
#include <cuda.h>
#include <nvml.h>
#include <omp.h>
#include <spral_ssids.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "driver/internal.hpp"
#include "factor/mumps_factorization.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;
constexpr double kGiB = 1073741824.0;

struct Args {
  fs::path case_dir;
  std::string backend;
  std::string ordering;  // empty: driver (mumps) / metis (ssids)
  double sigma = std::numeric_limits<double>::quiet_NaN();
  int threads = 0;
  int factorizations = 2;
  int solves = 10;
  int block = 8;
  int block_solves = 5;
  bool analyze_only = false;
  double memory_budget_gb = 0.0;  // 0: SolverConfig's default (14 GB)
  // SSIDS options
  double u = 0.01;
  int scaling = 4;
  int pivot_method = 2;
  int nemin = 32;
  double min_gpu_work = 5e9;
  double gpu_perf_coeff = 1.0;
  double max_load_inbalance = 1.2;
};

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// ------------------------------------------------------------------------------ host memory

/// Resets the kernel's peak RSS mark (VmHWM) of this process to the current RSS.
void reset_peak_rss() { std::ofstream("/proc/self/clear_refs") << "5"; }

/// VmHWM of this process (GiB): the peak RSS since the last reset_peak_rss().
double peak_rss_gib() {
  std::ifstream f("/proc/self/status");
  for (std::string line; std::getline(f, line);)
    if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) * 1024.0 / kGiB;
  return -1.0;
}

// ------------------------------------------------------------------------------ GPU

void cu_check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(what) + ": " + (s ? s : "CUDA error"));
  }
}
void nvml_check(nvmlReturn_t r, const char* what) {
  if (r != NVML_SUCCESS) throw std::runtime_error(std::string(what) + ": " + nvmlErrorString(r));
}

/// CUDA device 0 of this process: its primary context, and its used memory through NVML (device-wide)
/// sampled from a thread.
class Gpu {
 public:
  ~Gpu() {
    stop_sampling();
    if (ctx_) cuDevicePrimaryCtxRelease(dev_);
    if (nvml_) nvmlShutdown();
  }
  /// cuInit and the primary context of device 0 - what the first CUDA call of a process pays.
  /// Returns its wall time (s).
  double create_context() {
    hs::Timer t;
    cu_check(cuInit(0), "cuInit");
    cu_check(cuDeviceGet(&dev_, 0), "cuDeviceGet");
    cu_check(cuDevicePrimaryCtxRetain(&ctx_, dev_), "cuDevicePrimaryCtxRetain");
    cu_check(cuCtxSetCurrent(ctx_), "cuCtxSetCurrent");
    CUdeviceptr p = 0;  // the first allocation completes the context's initialization
    cu_check(cuMemAlloc(&p, 256), "cuMemAlloc");
    cu_check(cuMemFree(p), "cuMemFree");
    const double s = t.seconds();
    char bus[32] = {};
    cu_check(cuDeviceGetPCIBusId(bus, sizeof(bus), dev_), "cuDeviceGetPCIBusId");
    cu_check(cuDeviceGetName(name_, sizeof(name_), dev_), "cuDeviceGetName");
    bus_ = bus;
    nvml_check(nvmlInit_v2(), "nvmlInit");
    nvml_ = true;
    nvml_check(nvmlDeviceGetHandleByPciBusId_v2(bus, &handle_), "nvmlDeviceGetHandleByPciBusId");
    return s;
  }
  double used_mib() const {
    nvmlMemory_t m{};
    nvml_check(nvmlDeviceGetMemoryInfo(handle_, &m), "nvmlDeviceGetMemoryInfo");
    return static_cast<double>(m.used) / 1048576.0;
  }
  void start_sampling() {
    peak_mib_ = used_mib();
    running_ = true;
    sampler_ = std::thread([this] {
      while (running_) {
        const double u = used_mib();
        if (u > peak_mib_) peak_mib_ = u;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }
  void stop_sampling() {
    running_ = false;
    if (sampler_.joinable()) sampler_.join();
  }
  double peak_mib() const { return std::max(peak_mib_.load(), used_mib()); }
  json describe() const { return {{"name", name_}, {"pci_bus", bus_}}; }

 private:
  CUdevice dev_ = 0;
  CUcontext ctx_ = nullptr;
  bool nvml_ = false;
  nvmlDevice_t handle_{};
  char name_[128] = {};
  std::string bus_;
  std::thread sampler_;
  std::atomic<bool> running_{false};
  std::atomic<double> peak_mib_{0.0};
};

// ------------------------------------------------------------------------------ K and accuracy

/// K = A - sigma B on the pencil's upper pattern (the MUMPS input values of the driver).
struct ShiftedMatrix {
  const hs::ShiftedPencil<double>& pencil;
  std::vector<double> vals;
  double norm_inf = 0.0;  ///< ||K||_inf = ||K||_1 (symmetric)

  ShiftedMatrix(const hs::ShiftedPencil<double>& p, double sigma)
      : pencil(p), vals(static_cast<std::size_t>(p.nnz())) {
    p.assemble(1.0, -sigma, vals.data());
    const index_t n = p.order();
    std::vector<double> rows(static_cast<std::size_t>(n), 0.0);
    for (index_t i = 0; i < n; ++i)
      for (index_t q = p.rowptr()[i]; q < p.rowptr()[i + 1]; ++q) {
        const index_t j = p.colind()[q];
        rows[i] += std::abs(vals[q]);
        if (j != i) rows[j] += std::abs(vals[q]);
      }
    norm_inf = *std::max_element(rows.begin(), rows.end());
  }

  /// ||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf). Serial, untimed.
  double backward_error(const double* x, const double* b) const {
    const index_t n = pencil.order();
    std::vector<double> r(b, b + n);
    for (index_t i = 0; i < n; ++i)
      for (index_t q = pencil.rowptr()[i]; q < pencil.rowptr()[i + 1]; ++q) {
        const index_t j = pencil.colind()[q];
        r[i] -= vals[q] * x[j];
        if (j != i) r[j] -= vals[q] * x[i];
      }
    double rn = 0, xn = 0, bn = 0;
    for (index_t i = 0; i < n; ++i) {
      rn = std::max(rn, std::abs(r[i]));
      xn = std::max(xn, std::abs(x[i]));
      bn = std::max(bn, std::abs(b[i]));
    }
    return rn / (norm_inf * xn + bn);
  }
};

// ------------------------------------------------------------------------------ backends

struct Inertia {
  index_t neg = -1, zero = -1;
  bool ok = false;  ///< the backend reports a usable (nonsingular) factorization
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual void analyse() = 0;
  virtual Inertia factorize() = 0;
  /// Solves K X = B in place for nrhs columns of length n (leading dimension n).
  virtual void solve(double* x, int nrhs) = 0;
  /// Backend statistics after the last call (estimates, factor size, pivots).
  virtual json stats() const = 0;
};

/// The driver's adaptive analysis (src/driver/steps.cpp adaptive_analysis, without its Ctx/log).
std::unique_ptr<hs::SparseFactorization<double>> driver_analysis(const hs::ShiftedPencil<double>& pencil,
                                                                 const hs::FactorOptions& fo, double sigma,
                                                                 index_t k, bool has_b, json& out) {
  hs::FactorOptions amd = fo;
  amd.ordering = 0;
  auto fac = hs::make_mumps_factorization<double>(amd);
  fac->analyze(pencil, 1.0, -sigma);
  const auto d = hs::driver::decide_ordering(fac->estimate(), hs::driver::mumps_input_size(pencil),
                                             static_cast<double>(pencil.nnz()), fo.threads,
                                             1 + 2 + (has_b ? 1 : 0), std::max<index_t>(40, 3 * k));
  out["driver_fill_ratio"] = d.fill_ratio;
  out["driver_saving_s"] = d.saving_s;
  out["driver_nd_cost_s"] = d.nd_cost_s;
  if (!d.use_nd) {
    out["mumps_ordering"] = "amd";
    return fac;
  }
  fac.reset();
  hs::FactorOptions nd = fo;
  nd.ordering = fo.deterministic ? 5 : 3;  // nd_ordering(cfg)
  out["mumps_ordering"] = nd.ordering == 5 ? "metis" : "scotch";
  fac = hs::make_mumps_factorization<double>(nd);
  fac->analyze(pencil, 1.0, -sigma);
  return fac;
}

/// hsolve's MUMPS backend as the driver configures it.
class Mumps final : public Backend {
 public:
  Mumps(const hs::ShiftedPencil<double>& pencil, double sigma, index_t k, bool has_b, int threads,
        double budget_gb, std::string ordering, json& out)
      : pencil_(pencil), sigma_(sigma), k_(k), has_b_(has_b), ordering_(std::move(ordering)), out_(out) {
    hs::SolverConfig cfg;
    cfg.threads = threads;
    if (budget_gb > 0) cfg.memory_budget_gb = budget_gb;
    const double cap_mb =
        (cfg.memory_budget_gb * kGiB - static_cast<double>(hs::current_rss_bytes())) * 0.95 / 1e6;
    fo_ = hs::driver::factor_options_from(cfg, cap_mb);
  }
  void analyse() override {
    if (ordering_ == "driver") {
      fac_ = driver_analysis(pencil_, fo_, sigma_, k_, has_b_, out_);
      return;
    }
    hs::FactorOptions o = fo_;
    o.ordering = ordering_ == "amd" ? 0 : ordering_ == "scotch" ? 3 : ordering_ == "metis" ? 5 : -1;
    if (o.ordering < 0) throw std::invalid_argument("mumps --ordering: driver|amd|scotch|metis");
    out_["mumps_ordering"] = ordering_;
    fac_ = hs::make_mumps_factorization<double>(o);
    fac_->analyze(pencil_, 1.0, -sigma_);
  }
  Inertia factorize() override {
    const bool ok = fac_->factorize(1.0, -sigma_);
    const auto in = fac_->inertia();
    return {in.neg, in.zero, ok};
  }
  void solve(double* x, int nrhs) override { fac_->solve(x, nrhs, pencil_.order()); }
  json stats() const override {
    const auto& s = fac_->stats();
    const auto e = fac_->estimate();
    return {{"est_flops", e.flops},
            {"est_entries", e.factor_entries},
            {"est_incore_mb", e.incore_mb},
            {"factor_entries", s.factor_entries},
            {"factor_mb_used", s.factor_mb_used},
            {"delayed_pivots", s.delayed_pivots},
            {"null_pivots", s.null_pivots},
            {"threads", fo_.threads},
            {"l0_openmp", !fo_.deterministic},
            {"max_memory_mb", fo_.max_memory_mb}};
  }
  /// SYM_PERM of the analysis (1-based pivot position of every variable).
  std::vector<int> pivot_order() const {
    return dynamic_cast<const hs::mumps::MumpsFactorization<double, double>&>(*fac_).pivot_order();
  }

 private:
  const hs::ShiftedPencil<double>& pencil_;
  double sigma_;
  index_t k_;
  bool has_b_;
  std::string ordering_;
  json& out_;
  hs::FactorOptions fo_;
  std::unique_ptr<hs::SparseFactorization<double>> fac_;
};

/// SPRAL SSIDS on one region of `threads` CPU threads, with or without CUDA device 0.
class Ssids final : public Backend {
 public:
  Ssids(const ShiftedMatrix& k, bool gpu, int threads, const Args& a, std::vector<int> user_order)
      : k_(k), n_(static_cast<int>(k.pencil.order())), order_(std::move(user_order)) {
    // 1-based CSC of the lower triangle = the pencil's upper CSR (array_base = 1: SSIDS then uses
    // the arrays as they are instead of copying them at every call).
    ptr_.resize(static_cast<std::size_t>(n_) + 1);
    for (int i = 0; i <= n_; ++i) ptr_[i] = k.pencil.rowptr()[i] + 1;
    row_.resize(k.pencil.colind().size());
    for (std::size_t q = 0; q < row_.size(); ++q) row_[q] = k.pencil.colind()[q] + 1;
    spral_ssids_default_options(&opt_);
    opt_.array_base = 1;
    opt_.print_level = -1;  // flags are reported in the JSON line; stdout stays machine-readable
    opt_.unit_diagnostics = opt_.unit_error = opt_.unit_warning = -1;
    opt_.ordering = order_.empty() ? 1 : 0;
    opt_.use_gpu = gpu;
    opt_.u = a.u;
    opt_.scaling = a.scaling;
    opt_.pivot_method = a.pivot_method;
    opt_.nemin = a.nemin;
    opt_.min_gpu_work = static_cast<std::int64_t>(a.min_gpu_work);
    opt_.gpu_perf_coeff = static_cast<float>(a.gpu_perf_coeff);
    opt_.max_load_inbalance = static_cast<float>(a.max_load_inbalance);
    opt_.action = true;
    region_.nproc = threads;
    region_.ngpu = gpu ? 1 : 0;
    region_.gpus = gpu ? &device_ : nullptr;
    if (order_.empty()) order_.assign(static_cast<std::size_t>(n_), 0);  // output of METIS
  }
  ~Ssids() override { spral_ssids_free(&akeep_, &fkeep_); }
  Ssids(const Ssids&) = delete;
  Ssids& operator=(const Ssids&) = delete;

  void analyse() override {
    spral_ssids_analyse_topology(false, n_, order_.data(), ptr_.data(), row_.data(), k_.vals.data(),
                                 &akeep_, &opt_, &info_, 1, &region_);
    check("analyse");
    analyse_info_ = info_;
  }
  Inertia factorize() override {
    spral_ssids_factor(false, ptr_.data(), row_.data(), k_.vals.data(), nullptr, akeep_, &fkeep_, &opt_, &info_);
    check("factor");
    factor_info_ = info_;
    return {info_.num_neg, static_cast<index_t>(n_) - info_.matrix_rank, info_.flag >= 0};
  }
  void solve(double* x, int nrhs) override {
    if (nrhs == 1)
      spral_ssids_solve1(0, x, akeep_, fkeep_, &opt_, &info_);
    else
      spral_ssids_solve(0, nrhs, x, n_, akeep_, fkeep_, &opt_, &info_);
    check("solve");
  }
  json stats() const override {
    const auto& a = analyse_info_;
    const auto& f = factor_info_;
    return {{"analyse_flag", a.flag},
            {"est_flops", static_cast<double>(a.num_flops)},
            {"est_entries", a.num_factor},
            {"est_maxfront", a.maxfront},
            {"num_sup", a.num_sup},
            {"maxdepth", a.maxdepth},
            {"factor_flag", f.flag},
            {"factor_entries", f.num_factor},
            {"factor_flops", static_cast<double>(f.num_flops)},
            {"maxfront", f.maxfront},
            {"maxsupernode", f.maxsupernode},
            {"delayed_pivots", f.num_delay},
            {"two_by_two", f.num_two},
            {"matrix_rank", f.matrix_rank},
            {"cuda_error", f.cuda_error},
            {"cublas_error", f.cublas_error},
            {"options",
             {{"u", opt_.u},
              {"scaling", opt_.scaling},
              {"pivot_method", opt_.pivot_method},
              {"nemin", opt_.nemin},
              {"ordering", opt_.ordering == 0 ? "user" : "metis"},
              {"use_gpu", opt_.use_gpu},
              {"min_gpu_work", static_cast<double>(opt_.min_gpu_work)},
              {"gpu_perf_coeff", opt_.gpu_perf_coeff},
              {"max_load_inbalance", opt_.max_load_inbalance},
              {"region_threads", region_.nproc},
              {"region_gpus", region_.ngpu}}}};
  }

 private:
  /// Errors (flag < 0) throw; warnings (flag > 0: +7 singular, +50 OMP_PROC_BIND off ...) are kept.
  void check(const char* phase) const {
    if (info_.flag < 0)
      throw std::runtime_error(std::string("SSIDS ") + phase + " failed: flag " + std::to_string(info_.flag) +
                               " stat " + std::to_string(info_.stat) + " cuda_error " +
                               std::to_string(info_.cuda_error) + " cublas_error " +
                               std::to_string(info_.cublas_error));
  }

  const ShiftedMatrix& k_;
  int n_;
  std::vector<int> order_;
  std::vector<std::int64_t> ptr_;
  std::vector<int> row_;
  spral_ssids_options opt_{};
  spral_ssids_inform info_{}, analyse_info_{}, factor_info_{};
  int device_ = 0;
  spral_numa_region region_{};
  void* akeep_ = nullptr;
  void* fkeep_ = nullptr;
};

// ------------------------------------------------------------------------------ the measurement

int run(const Args& a0, const hs::CsrMatrix<double>& am, const hs::CsrMatrix<double>* bm) {
  Args a = a0;
  const hs::ShiftedPencil<double> pencil(am, bm);
  if (a.threads <= 0) a.threads = omp_get_max_threads();
  std::ifstream cf(a.case_dir / "config.json");
  std::stringstream ss;
  ss << cf.rdbuf();
  const hs::SolverConfig cfg = hs::parse_config(ss.str());
  if (std::isnan(a.sigma)) a.sigma = cfg.target.sigma;
  const index_t k = cfg.target.k;
  const bool ssids = a.backend == "ssids-cpu" || a.backend == "ssids-gpu";
  if (a.backend != "mumps" && !ssids) throw std::invalid_argument("--backend mumps|ssids-cpu|ssids-gpu");
  if (a.ordering.empty()) a.ordering = ssids ? "metis" : "driver";
  if (ssids && a.ordering != "metis" && a.ordering != "mumps")
    throw std::invalid_argument("ssids --ordering: metis|mumps");

  const ShiftedMatrix km(pencil, a.sigma);
  const index_t n = pencil.order();
  const char* env_cancel = std::getenv("OMP_CANCELLATION");
  const char* env_bind = std::getenv("OMP_PROC_BIND");
  const char* env_cvd = std::getenv("CUDA_VISIBLE_DEVICES");
  json out = {{"case", a.case_dir.filename().string()},
              {"n", n},
              {"nnz_upper", pencil.nnz()},
              {"sigma", a.sigma},
              {"backend", a.backend},
              {"ordering", a.ordering},
              {"threads", a.threads},
              {"omp_cancellation", env_cancel ? env_cancel : ""},
              {"omp_proc_bind", env_bind ? env_bind : ""},
              {"cuda_visible_devices", env_cvd ? env_cvd : "(unset)"},
              {"rss_load_gib", static_cast<double>(hs::current_rss_bytes()) / kGiB}};

  // SSIDS with the reference's elimination order: the driver's MUMPS analysis, untimed, then freed.
  std::vector<int> user_order;
  if (ssids && a.ordering == "mumps") {
    json mo;
    Mumps ref(pencil, a.sigma, k, bm != nullptr, a.threads, a.memory_budget_gb, "driver", mo);
    hs::Timer t;
    ref.analyse();
    out["order_source"] = {{"mumps_ordering", mo["mumps_ordering"]}, {"analyse_s", t.seconds()}};
    user_order = ref.pivot_order();
  }

  reset_peak_rss();  // the backend phase: CUDA context, analysis, factorizations, solves
  std::unique_ptr<Gpu> gpu;
  if (a.backend == "ssids-gpu") {
    gpu = std::make_unique<Gpu>();
    out["cuda_init_s"] = gpu->create_context();
    out["gpu"] = gpu->describe();
    out["gpu_mem_ctx_mib"] = gpu->used_mib();
    gpu->start_sampling();
  }

  std::unique_ptr<Backend> be;
  if (ssids)
    be = std::make_unique<Ssids>(km, a.backend == "ssids-gpu", a.threads, a, std::move(user_order));
  else
    be = std::make_unique<Mumps>(pencil, a.sigma, k, bm != nullptr, a.threads, a.memory_budget_gb, a.ordering, out);
  hs::Timer t;
  be->analyse();
  out["analyse_s"] = t.seconds();
  if (a.analyze_only) {
    out["stats"] = be->stats();
    out["rss_peak_gib"] = peak_rss_gib();
    std::cout << out.dump() << std::endl;
    return 0;
  }

  json fs_ = json::array(), negs = json::array(), zeros = json::array(), oks = json::array();
  for (int f = 0; f < a.factorizations; ++f) {
    t.reset();
    const Inertia in = be->factorize();
    fs_.push_back(t.seconds());
    negs.push_back(in.neg);
    zeros.push_back(in.zero);
    oks.push_back(in.ok);
  }
  out["factor_s"] = fs_;
  out["neg"] = negs;
  out["zero"] = zeros;
  out["ok"] = oks;
  if (gpu) out["gpu_mem_factor_mib"] = gpu->used_mib();

  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  auto random = [&](std::size_t len) {
    std::vector<double> v(len);
    for (auto& x : v) x = unif(rng);
    return v;
  };
  std::vector<double> s1, sp;
  std::vector<double> b1, x1, bp, xp;
  for (int s = 0; s < a.solves; ++s) {
    std::vector<double> b = random(static_cast<std::size_t>(n)), x = b;
    t.reset();
    be->solve(x.data(), 1);
    s1.push_back(t.seconds());
    if (s == 0) b1 = std::move(b), x1 = std::move(x);
  }
  for (int s = 0; s < a.block_solves; ++s) {
    std::vector<double> b = random(static_cast<std::size_t>(n) * a.block), x = b;
    t.reset();
    be->solve(x.data(), a.block);
    sp.push_back(t.seconds());
    if (s == 0) bp = std::move(b), xp = std::move(x);
  }
  out["solve1_first_s"] = s1.empty() ? 0.0 : s1.front();
  out["solve1_s"] = median(s1);
  out["block"] = a.block;
  out["solveP_s"] = median(sp);
  out["rss_peak_gib"] = peak_rss_gib();
  if (gpu) {
    gpu->stop_sampling();
    out["gpu_mem_peak_mib"] = gpu->peak_mib();
  }
  out["stats"] = be->stats();

  // Accuracy (untimed).
  if (!x1.empty()) out["resid1"] = km.backward_error(x1.data(), b1.data());
  if (!xp.empty()) {
    double worst = 0.0;
    for (int c = 0; c < a.block; ++c)
      worst = std::max(worst, km.backward_error(xp.data() + static_cast<std::size_t>(c) * n,
                                                bp.data() + static_cast<std::size_t>(c) * n));
    out["residP"] = worst;
  }
  std::cout << out.dump() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << key << "\n";
        return 2;
      }
      const std::string v = argv[++i];
      if (key == "--case") args.case_dir = v;
      else if (key == "--backend") args.backend = v;
      else if (key == "--ordering") args.ordering = v;
      else if (key == "--sigma") args.sigma = std::stod(v);
      else if (key == "--threads") args.threads = std::stoi(v);
      else if (key == "--factorizations") args.factorizations = std::stoi(v);
      else if (key == "--solves") args.solves = std::stoi(v);
      else if (key == "--block") args.block = std::stoi(v);
      else if (key == "--block-solves") args.block_solves = std::stoi(v);
      else if (key == "--analyze-only") args.analyze_only = std::stoi(v) != 0;
      else if (key == "--memory-budget-gb") args.memory_budget_gb = std::stod(v);
      else if (key == "--u") args.u = std::stod(v);
      else if (key == "--scaling") args.scaling = std::stoi(v);
      else if (key == "--pivot-method") args.pivot_method = std::stoi(v);
      else if (key == "--nemin") args.nemin = std::stoi(v);
      else if (key == "--min-gpu-work") args.min_gpu_work = std::stod(v);
      else if (key == "--gpu-perf-coeff") args.gpu_perf_coeff = std::stod(v);
      else if (key == "--max-load-inbalance") args.max_load_inbalance = std::stod(v);
      else {
        std::cerr << "unknown argument " << key << "\n";
        return 2;
      }
    }
    if (args.case_dir.empty() || args.backend.empty()) {
      std::cerr << "usage: bench_gpu_ssids --case DIR --backend mumps|ssids-cpu|ssids-gpu [...]\n";
      return 2;
    }
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    if (la.is_complex() || (lb && lb->is_complex())) {
      std::cerr << "bench_gpu_ssids: complex cases are not supported (SSIDS is real symmetric)\n";
      return 2;
    }
    const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
    const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
    return run(args, a, b);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
