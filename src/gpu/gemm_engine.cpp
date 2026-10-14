// GPU module: hybrid CPU + GPU dgemm (gemm_engine.hpp).
#include "gpu/gemm_engine.hpp"

#include <dlfcn.h>
#include <omp.h>
#include <time.h>

#if defined(__x86_64__) || defined(__SSE2__)
#include <emmintrin.h>
#define HS_GPU_NT_STORES 1
#else
#define HS_GPU_NT_STORES 0
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "gpu/cuda_lib.hpp"

namespace hs::gpu {
namespace {

double now_s() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

bool is_trans(char c) {
  return c == 'T' || c == 't' || c == 'C' || c == 'c';
}

// A column-major block: `cols` runs of `rows` contiguous doubles, `ld` apart.
struct View {
  const double* p;
  std::int64_t rows, cols, ld;
  std::int64_t bytes() const { return 8 * rows * cols; }
};

// dst[0, n) = src[0, n) with non-temporal stores: the staging buffer is read by the copy engine,
// not by the CPU, so its lines need not be read for ownership nor kept in the caches. The caller
// fences (_mm_sfence) before the copy engine may read the buffer.
void copy_run(double* dst, const double* src, std::int64_t n, bool nt) {
#if HS_GPU_NT_STORES
  if (nt) {
    std::int64_t i = 0;
    if ((reinterpret_cast<std::uintptr_t>(dst) & 15) != 0 && n > 0) {
      dst[0] = src[0];
      i = 1;
    }
    for (; i + 8 <= n; i += 8) {
      _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
      _mm_stream_pd(dst + i + 2, _mm_loadu_pd(src + i + 2));
      _mm_stream_pd(dst + i + 4, _mm_loadu_pd(src + i + 4));
      _mm_stream_pd(dst + i + 6, _mm_loadu_pd(src + i + 6));
    }
    for (; i < n; ++i) dst[i] = src[i];
    return;
  }
#endif
  std::memcpy(dst, src, 8 * static_cast<std::size_t>(n));
}

// Copies a block into contiguous storage (leading dimension = rows).
void pack(const View& v, double* dst, bool nt) {
  if (v.ld == v.rows) {
    copy_run(dst, v.p, v.rows * v.cols, nt);
  } else {
    for (std::int64_t j = 0; j < v.cols; ++j) copy_run(dst + j * v.rows, v.p + j * v.ld, v.rows, nt);
  }
#if HS_GPU_NT_STORES
  if (nt) _mm_sfence();
#endif
}

// C (rows x cols, ldc) = beta C + T (contiguous); beta = 0 never reads C.
void accumulate(double* c, std::int64_t ldc, const double* t, std::int64_t rows, std::int64_t cols,
                double beta) {
  for (std::int64_t j = 0; j < cols; ++j) {
    double* cj = c + j * ldc;
    const double* tj = t + j * rows;
    if (beta == 0.0) {
      std::memcpy(cj, tj, 8 * static_cast<std::size_t>(rows));
    } else if (beta == 1.0) {
      for (std::int64_t i = 0; i < rows; ++i) cj[i] += tj[i];
    } else {
      for (std::int64_t i = 0; i < rows; ++i) cj[i] = beta * cj[i] + tj[i];
    }
  }
}

// One dgemm call in flight. The range [lo, hi) of the split dimension is still unassigned: the
// GPU takes tiles from lo, the CPU chunks from hi.
struct Call {
  bool ta = false, tb = false;
  int m = 0, n = 0, k = 0;
  double alpha = 0.0, beta = 0.0;
  const double* a = nullptr;
  int lda = 0;
  const double* b = nullptr;
  int ldb = 0;
  double* c = nullptr;
  int ldc = 0;
  bool split_n = true;  // tiles are columns of C (shared operand op(A)); else rows (op(B))
  int len = 0;          // length of the split dimension
  int tile = 0;         // GPU tile width

  std::mutex range_mu;
  int lo = 0, hi = 0;

  // Written by the pipeline threads under Impl::mu.
  bool shared_ready = false;  // the shared operand's upload is issued (shared_ev recorded)
  bool error = false;
  int issuers_left = 0;
  std::vector<std::pair<int, int>> failed;
  std::int64_t tiles = 0;
  std::int64_t gpu_units = 0;
  double h2d = 0.0, d2h = 0.0;
  double shared_s = 0.0, pack_s = 0.0, slot_wait_s = 0.0, sync_s = 0.0, add_s = 0.0;

  // The shared operand, as stored.
  View shared() const {
    if (split_n) return ta ? View{a, k, m, lda} : View{a, m, k, lda};
    return tb ? View{b, n, k, ldb} : View{b, k, n, ldb};
  }
  // The tile [t0, t1) of the split operand, as stored.
  View tile_view(int t0, int t1) const {
    const std::int64_t w = t1 - t0;
    if (split_n)
      return tb ? View{b + t0, w, k, ldb} : View{b + static_cast<std::int64_t>(t0) * ldb, k, w, ldb};
    return ta ? View{a + static_cast<std::int64_t>(t0) * lda, k, w, lda} : View{a + t0, w, k, lda};
  }
  // Extent of the result per unit (column / row of C) of the split dimension, in doubles.
  std::int64_t result_per_unit() const { return split_n ? m : n; }
};

// Events of the shared operand's staging ring (the most chunks in flight at once).
constexpr int kStageRing = 64;

// The model's CPU dgemm rate, flop/s.
double cpu_rate(const EngineOptions& o) {
  const double g = o.cpu_gflops > 0.0 ? o.cpu_gflops
                                      : o.cpu_gflops_per_thread * std::max(1, omp_get_max_threads());
  return std::max(1.0, g) * 1e9;
}

// Item of the queue from the issuing threads to the completing thread.
struct Item {
  int slot, t0, t1;
};

}  // namespace

struct GemmEngine::Impl {
  const CudaLib* L = nullptr;
  EngineOptions opt;
  EngineInfo info;
  int slots = 0;

  std::vector<cublasHandle_t> handles;  // one per issuing thread
  std::vector<cudaStream_t> streams;    // one per slot
  std::vector<cudaEvent_t> done_ev;     // per slot: the tile's result is in pin_out
  std::vector<cudaEvent_t> stage_ev;  // per staging-ring chunk: its upload is done
  cudaEvent_t shared_ev = nullptr;               // the shared operand is on the device
  double* dev_shared = nullptr;
  std::size_t shared_cap = 0;  // bytes
  std::vector<double*> dev_in, dev_out, pin_in, pin_out;
  double* pin_stage = nullptr;  // staging ring of the shared operand: 2 slot_bytes

  std::mutex call_mu;  // one call at a time (try_lock)
  std::atomic<std::int64_t> tiles_issued{0};  // lifetime count (fault injection)

  // Pipeline coordination.
  std::mutex mu;
  std::condition_variable cv;
  Call* cur = nullptr;
  std::uint64_t gen = 0;
  bool quit = false;
  int finished = 0;  // pipeline threads done with the current call
  std::vector<bool> slot_free;
  std::deque<Item> queue;
  bool queue_closed = false;
  std::vector<std::thread> threads;

  mutable std::mutex stats_mu;
  EngineStats st;
  std::atomic<bool> is_broken{false};
  std::string err;

  ~Impl();
  bool check(cudaError_t e, const char* what);
  bool check_blas(cublasStatus_t s, const char* what);
  void fail(const std::string& msg);
  void thread_loop(int role);  // role < pack_threads: issuing thread `role`; else completing
  bool upload_shared(Call& call);
  void run_issue(Call& call, int id);
  void run_complete(Call& call);
  int take_free_slot(Call& call);
  void cpu_range(const Call& call, CpuDgemm cpu, int u0, int u1) const;
};

void GemmEngine::Impl::fail(const std::string& msg) {
  std::lock_guard<std::mutex> lk(stats_mu);
  ++st.errors;
  if (!is_broken.exchange(true)) err = msg;
}

bool GemmEngine::Impl::check(cudaError_t e, const char* what) {
  if (e == kCudaSuccess) return true;
  fail(std::string(what) + ": " + error_text(L, e));
  return false;
}

bool GemmEngine::Impl::check_blas(cublasStatus_t s, const char* what) {
  if (s == kCublasStatusSuccess) return true;
  fail(std::string(what) + ": cublas status " + std::to_string(s));
  return false;
}

GemmEngine::Impl::~Impl() {
  {
    std::lock_guard<std::mutex> lk(mu);
    quit = true;
  }
  cv.notify_all();
  for (std::thread& t : threads)
    if (t.joinable()) t.join();
  if (L == nullptr) return;
  if (!streams.empty()) L->DeviceSynchronize();
  for (double* p : dev_in)
    if (p != nullptr) L->Free(p);
  for (double* p : dev_out)
    if (p != nullptr) L->Free(p);
  if (dev_shared != nullptr) L->Free(dev_shared);
  for (double* p : pin_in)
    if (p != nullptr) L->FreeHost(p);
  for (double* p : pin_out)
    if (p != nullptr) L->FreeHost(p);
  if (pin_stage != nullptr) L->FreeHost(pin_stage);
  for (cudaEvent_t e : done_ev)
    if (e != nullptr) L->EventDestroy(e);
  for (cudaEvent_t e : stage_ev)
    if (e != nullptr) L->EventDestroy(e);
  if (shared_ev != nullptr) L->EventDestroy(shared_ev);
  for (cudaStream_t s : streams)
    if (s != nullptr) L->StreamDestroy(s);
  for (cublasHandle_t h : handles)
    if (h != nullptr) L->BlasDestroy(h);
}

// ---- pipeline threads ----

void GemmEngine::Impl::thread_loop(int role) {
  L->SetDevice(opt.device);
  std::uint64_t seen = 0;
  for (;;) {
    Call* call = nullptr;
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return quit || gen != seen; });
      if (quit) return;
      seen = gen;
      call = cur;
    }
    if (role < opt.pack_threads)
      run_issue(*call, role);
    else
      run_complete(*call);
    {
      std::lock_guard<std::mutex> lk(mu);
      ++finished;
    }
    cv.notify_all();
  }
}

// A free slot (marked busy), or -1 when the call has failed.
int GemmEngine::Impl::take_free_slot(Call& call) {
  const double t0 = now_s();
  std::unique_lock<std::mutex> lk(mu);
  int s = -1;
  cv.wait(lk, [&] {
    if (call.error) return true;
    for (int i = 0; i < slots; ++i)
      if (slot_free[i]) {
        s = i;
        return true;
      }
    return false;
  });
  call.slot_wait_s += now_s() - t0;
  if (call.error) return -1;
  slot_free[s] = false;
  return s;
}

// Packs the shared operand through two pinned staging buffers onto the device (stream 0) and
// records shared_ev; false on a CUDA error (recorded in the call).
bool GemmEngine::Impl::upload_shared(Call& call) {
  const double t0 = now_s();
  const View sv = call.shared();
  const std::int64_t col_bytes = 8 * sv.rows;
  // Chunks of whole columns, at least chunk_bytes (or one slot without chunking), staged through
  // a ring of 2 slot_bytes: as many chunks in flight as fit, each reused when its upload is done.
  const std::int64_t want = opt.chunk_bytes > 0 ? static_cast<std::int64_t>(opt.chunk_bytes)
                                                : static_cast<std::int64_t>(opt.slot_bytes);
  const std::int64_t chunk_cols = std::max<std::int64_t>(1, want / col_bytes);
  const std::int64_t ring = std::clamp<std::int64_t>(
      2 * static_cast<std::int64_t>(opt.slot_bytes) / (chunk_cols * col_bytes), 1,
      static_cast<std::int64_t>(stage_ev.size()));
  bool ok = true;
  double bytes_done = 0.0;
  std::int64_t idx = 0;
  for (std::int64_t j0 = 0; ok && j0 < sv.cols; j0 += chunk_cols, ++idx) {
    const std::int64_t w = std::min(chunk_cols, sv.cols - j0);
    const std::int64_t bi = idx % ring;
    if (idx >= ring && !(ok = check(L->EventSynchronize(stage_ev[bi]), "staging wait"))) break;
    double* buf = pin_stage + bi * chunk_cols * sv.rows;
    pack(View{sv.p + j0 * sv.ld, sv.rows, w, sv.ld}, buf, opt.stream_stores);
    const std::size_t bytes = 8 * static_cast<std::size_t>(sv.rows * w);
    ok = check(L->MemcpyAsync(dev_shared + j0 * sv.rows, buf, bytes, kCudaMemcpyHostToDevice,
                              streams[0]),
               "upload of the shared operand") &&
         check(L->EventRecord(stage_ev[bi], streams[0]), "event record");
    if (ok) bytes_done += static_cast<double>(bytes);
  }
  if (ok) ok = check(L->EventRecord(shared_ev, streams[0]), "event record");
  {
    std::lock_guard<std::mutex> lk(mu);
    call.h2d += bytes_done;
    call.shared_s += now_s() - t0;
    if (ok)
      call.shared_ready = true;
    else
      call.error = true;
  }
  cv.notify_all();
  return ok;
}

void GemmEngine::Impl::run_issue(Call& call, int id) {
  bool ok = id != 0 || upload_shared(call);
  const double zero = 0.0;
  const View sv = call.shared();
  const int shared_ld = static_cast<int>(sv.rows);
  bool waited_shared = id == 0;
  while (ok) {
    const int s = take_free_slot(call);
    if (s < 0) break;
    int t0 = -1, t1 = -1;
    {
      std::lock_guard<std::mutex> lk(call.range_mu);
      if (call.lo < call.hi) {  // guided: a share of what is left, at most a full slot
        const int left = call.hi - call.lo;
        const int w = std::min(call.tile, std::max(std::min(kMinTile, left), left / (slots + 1)));
        t0 = call.lo;
        t1 = call.lo + w;
        call.lo = t1;
      }
    }
    if (t0 < 0) {
      std::lock_guard<std::mutex> lk(mu);
      slot_free[s] = true;
      break;
    }
    const double tp = now_s();
    const View tv = call.tile_view(t0, t1);
    const std::int64_t serial = tiles_issued.fetch_add(1);
    const bool injected =
        opt.inject_fault_after_tiles >= 0 && serial >= opt.inject_fault_after_tiles;
    if (injected) fail("injected fault (EngineOptions::inject_fault_after_tiles)");
    // Pack and upload in chunks of whole columns, each copied as soon as it is packed (with
    // chunks small enough to stay in the last-level cache the copy engine reads them from there).
    bool tok = !injected;
    {
      const std::int64_t col = 8 * tv.rows;
      const std::int64_t cc =
          opt.chunk_bytes > 0
              ? std::max<std::int64_t>(1, static_cast<std::int64_t>(opt.chunk_bytes) / col)
              : tv.cols;
      for (std::int64_t j0 = 0; tok && j0 < tv.cols; j0 += cc) {
        const std::int64_t w = std::min(cc, tv.cols - j0);
        pack(View{tv.p + j0 * tv.ld, tv.rows, w, tv.ld}, pin_in[s] + j0 * tv.rows,
             opt.stream_stores);
        tok = check(L->MemcpyAsync(dev_in[s] + j0 * tv.rows, pin_in[s] + j0 * tv.rows,
                                   8 * static_cast<std::size_t>(tv.rows * w),
                                   kCudaMemcpyHostToDevice, streams[s]),
                    "tile upload");
      }
    }
    const double pack_time = now_s() - tp;
    if (!waited_shared) {  // the shared operand's event must be recorded before streams wait on it
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return call.shared_ready || call.error; });
      waited_shared = true;
      if (call.error) {
        call.failed.emplace_back(t0, t1);
        slot_free[s] = true;
        break;
      }
    }
    const int w = t1 - t0;
    const int tile_ld = static_cast<int>(tv.rows);
    const int rows_t = call.split_n ? call.m : w;  // result tile rows
    const int cols_t = call.split_n ? w : call.n;
    const double* da = call.split_n ? dev_shared : dev_in[s];
    const int dlda = call.split_n ? shared_ld : tile_ld;
    const double* db = call.split_n ? dev_in[s] : dev_shared;
    const int dldb = call.split_n ? tile_ld : shared_ld;
    const std::size_t in_bytes = static_cast<std::size_t>(tv.bytes());
    const std::size_t out_bytes = 8 * static_cast<std::size_t>(rows_t) * cols_t;
    tok = tok &&
        check(L->StreamWaitEvent(streams[s], shared_ev, 0), "stream wait") &&
        check_blas(L->BlasSetStream(handles[id], streams[s]), "cublasSetStream") &&
        check_blas(L->Dgemm(handles[id], call.ta ? kCublasOpT : kCublasOpN,
                            call.tb ? kCublasOpT : kCublasOpN, rows_t, cols_t, call.k, &call.alpha,
                            da, dlda, db, dldb, &zero, dev_out[s], rows_t),
                   "cublasDgemm") &&
        check(L->MemcpyAsync(pin_out[s], dev_out[s], out_bytes, kCudaMemcpyDeviceToHost,
                             streams[s]),
              "tile download") &&
        check(L->EventRecord(done_ev[s], streams[s]), "event record");
    {
      std::lock_guard<std::mutex> lk(mu);
      call.pack_s += pack_time;
      if (tok) {
        call.h2d += static_cast<double>(in_bytes);
        queue.push_back(Item{s, t0, t1});
      } else {
        call.failed.emplace_back(t0, t1);
        call.error = true;
        slot_free[s] = true;
      }
    }
    cv.notify_all();
    if (!tok) break;
  }
  // The staging buffers must be free when the next call packs into them (when the CPU took every
  // unit, no tile waited for the shared operand's upload).
  if (id == 0)
    for (cudaEvent_t e : stage_ev) check(L->EventSynchronize(e), "staging wait");
  {
    std::lock_guard<std::mutex> lk(mu);
    if (--call.issuers_left == 0) queue_closed = true;
  }
  cv.notify_all();
}

void GemmEngine::Impl::run_complete(Call& call) {
  for (;;) {
    Item it{};
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return !queue.empty() || queue_closed; });
      if (queue.empty()) break;
      it = queue.front();
      queue.pop_front();
    }
    const double t0 = now_s();
    const bool ok = check(L->EventSynchronize(done_ev[it.slot]), "tile completion");
    const double t1 = now_s();
    const int w = it.t1 - it.t0;
    if (ok) {
      if (call.split_n)
        accumulate(call.c + static_cast<std::int64_t>(it.t0) * call.ldc, call.ldc, pin_out[it.slot],
                   call.m, w, call.beta);
      else
        accumulate(call.c + it.t0, call.ldc, pin_out[it.slot], w, call.n, call.beta);
    }
    const double t2 = now_s();
    {
      std::lock_guard<std::mutex> lk(mu);
      call.sync_s += t1 - t0;
      call.add_s += t2 - t1;
      if (ok) {
        ++call.tiles;
        call.gpu_units += w;
        call.d2h += 8.0 * static_cast<double>(w) * static_cast<double>(call.result_per_unit());
      } else {
        call.failed.emplace_back(it.t0, it.t1);
        call.error = true;
      }
      slot_free[it.slot] = true;
    }
    cv.notify_all();
  }
}

void GemmEngine::Impl::cpu_range(const Call& call, CpuDgemm cpu, int u0, int u1) const {
  const char ta = call.ta ? 'T' : 'N', tb = call.tb ? 'T' : 'N';
  const int w = u1 - u0;
  if (w <= 0) return;
  if (call.split_n) {
    const double* bp = call.tb ? call.b + u0 : call.b + static_cast<std::int64_t>(u0) * call.ldb;
    cpu(&ta, &tb, &call.m, &w, &call.k, &call.alpha, call.a, &call.lda, bp, &call.ldb, &call.beta,
        call.c + static_cast<std::int64_t>(u0) * call.ldc, &call.ldc, 1, 1);
  } else {
    const double* ap = call.ta ? call.a + static_cast<std::int64_t>(u0) * call.lda : call.a + u0;
    cpu(&ta, &tb, &w, &call.n, &call.k, &call.alpha, ap, &call.lda, call.b, &call.ldb, &call.beta,
        call.c + u0, &call.ldc, 1, 1);
  }
}

// ---- GemmEngine ----

GemmEngine::GemmEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GemmEngine::~GemmEngine() = default;

std::unique_ptr<GemmEngine> GemmEngine::create(const EngineOptions& opt_in, std::string* why) {
  const double t0 = now_s();
  auto impl = std::make_unique<Impl>();
  EngineOptions opt = opt_in;
  opt.pack_threads = std::clamp(opt.pack_threads, 1, 4);
  opt.slots = std::clamp(opt.slots, opt.pack_threads + 1, 8);
  opt.slot_bytes = std::max<std::size_t>(opt.slot_bytes, std::size_t{1} << 20);
  impl->opt = opt;
  impl->slots = opt.slots;
  const int slots = impl->slots;
  std::string reason;
  auto failed = [&](const std::string& msg) -> std::unique_ptr<GemmEngine> {
    if (why != nullptr) *why = msg;
    return nullptr;
  };
  const CudaLib* L = load(&reason);
  if (L == nullptr) return failed(reason);
  impl->L = L;
  Impl& I = *impl;

  int count = 0;
  cudaError_t e = L->GetDeviceCount(&count);
  if (e != kCudaSuccess) return failed("no usable CUDA device: " + error_text(L, e));
  if (count <= opt.device)
    return failed("no CUDA device " + std::to_string(opt.device) + " (" + std::to_string(count) +
                  " visible)");
  if ((e = L->SetDevice(opt.device)) != kCudaSuccess)
    return failed("cudaSetDevice: " + error_text(L, e));
  // Blocking waits (the pipeline threads sleep instead of spinning next to the BLAS threads); an
  // already active context keeps its flags (the events below block regardless).
  if (L->SetDeviceFlags(kCudaDeviceScheduleBlockingSync) != kCudaSuccess) L->GetLastError();
  if ((e = L->Free(nullptr)) != kCudaSuccess)  // creates the context
    return failed("CUDA context: " + error_text(L, e));

  EngineInfo& inf = I.info;
  inf.options = opt;
  inf.cudart_path = L->cudart_path;
  inf.cublas_path = L->cublas_path;
  L->DeviceGetAttribute(&inf.compute_major, kCudaDevAttrComputeCapabilityMajor, opt.device);
  L->DeviceGetAttribute(&inf.compute_minor, kCudaDevAttrComputeCapabilityMinor, opt.device);
  L->DeviceGetAttribute(&inf.multiprocessors, kCudaDevAttrMultiProcessorCount, opt.device);
  L->DriverGetVersion(&inf.driver_version);
  L->RuntimeGetVersion(&inf.runtime_version);
  char bus[64] = {0};
  if (L->DeviceGetPCIBusId(bus, sizeof bus, opt.device) == kCudaSuccess) inf.pci_bus_id = bus;
  // The device name from the driver API (libcuda is loaded by the runtime by now).
  if (void* cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD)) {
    using GetFn = int (*)(int*, int);
    using NameFn = int (*)(char*, int, int);
    auto get = reinterpret_cast<GetFn>(dlsym(cu, "cuDeviceGet"));
    auto name = reinterpret_cast<NameFn>(dlsym(cu, "cuDeviceGetName"));
    int dev = 0;
    char buf[256] = {0};
    if (get != nullptr && name != nullptr && get(&dev, opt.device) == 0 &&
        name(buf, sizeof buf - 1, dev) == 0)
      inf.device_name = buf;
    dlclose(cu);
  }
  L->GetLastError();
  if ((e = L->MemGetInfo(&inf.device_free_bytes, &inf.device_total_bytes)) != kCudaSuccess)
    return failed("cudaMemGetInfo: " + error_text(L, e));
  const std::size_t tiles_bytes = 2 * static_cast<std::size_t>(slots) * opt.slot_bytes;
  if (opt.mem_bytes < tiles_bytes + opt.slot_bytes)
    return failed("device memory option too small for " + std::to_string(slots) + " slots");
  const std::size_t margin = std::size_t{256} << 20;  // cuBLAS workspaces, context growth
  if (inf.device_free_bytes < opt.mem_bytes + margin)
    return failed("too little free device memory: " +
                  std::to_string(inf.device_free_bytes >> 20) + " MiB free, need " +
                  std::to_string((opt.mem_bytes + margin) >> 20) + " MiB");

  I.handles.assign(opt.pack_threads, nullptr);
  for (cublasHandle_t& h : I.handles) {
    if (L->BlasCreate(&h) != kCublasStatusSuccess) return failed("cublasCreate failed");
    L->BlasSetMathMode(h, kCublasDefaultMath);
  }
  L->BlasGetVersion(I.handles[0], &inf.cublas_version);

  I.streams.assign(slots, nullptr);
  I.done_ev.assign(slots, nullptr);
  I.dev_in.assign(slots, nullptr);
  I.dev_out.assign(slots, nullptr);
  I.pin_in.assign(slots, nullptr);
  I.pin_out.assign(slots, nullptr);
  const unsigned evf = kCudaEventBlockingSync | kCudaEventDisableTiming;
  auto alloc_dev = [&](double** p, std::size_t bytes) {
    return L->Malloc(reinterpret_cast<void**>(p), bytes);
  };
  auto alloc_pin = [&](double** p, std::size_t bytes) {
    return L->HostAlloc(reinterpret_cast<void**>(p), bytes, kCudaHostAllocDefault);
  };
  for (int s = 0; s < slots; ++s) {
    if ((e = L->StreamCreateWithFlags(&I.streams[s], kCudaStreamNonBlocking)) != kCudaSuccess ||
        (e = L->EventCreateWithFlags(&I.done_ev[s], evf)) != kCudaSuccess ||
        (e = alloc_dev(&I.dev_in[s], opt.slot_bytes)) != kCudaSuccess ||
        (e = alloc_dev(&I.dev_out[s], opt.slot_bytes)) != kCudaSuccess ||
        (e = alloc_pin(&I.pin_in[s], opt.slot_bytes)) != kCudaSuccess ||
        (e = alloc_pin(&I.pin_out[s], opt.slot_bytes)) != kCudaSuccess)
      return failed("CUDA set-up: " + error_text(L, e));
  }
  I.stage_ev.assign(kStageRing, nullptr);
  for (cudaEvent_t& ev : I.stage_ev)
    if ((e = L->EventCreateWithFlags(&ev, evf)) != kCudaSuccess)
      return failed("CUDA set-up: " + error_text(L, e));
  if ((e = alloc_pin(&I.pin_stage, 2 * opt.slot_bytes)) != kCudaSuccess)
    return failed("CUDA set-up: " + error_text(L, e));
  if ((e = L->EventCreateWithFlags(&I.shared_ev, kCudaEventDisableTiming)) != kCudaSuccess)
    return failed("CUDA set-up: " + error_text(L, e));
  I.shared_cap = opt.mem_bytes - tiles_bytes;
  if ((e = alloc_dev(&I.dev_shared, I.shared_cap)) != kCudaSuccess)
    return failed("CUDA set-up: " + error_text(L, e));
  inf.pinned_bytes = (2 * static_cast<std::size_t>(slots) + 2) * opt.slot_bytes;
  I.slot_free.assign(slots, true);
  for (int role = 0; role <= opt.pack_threads; ++role)
    I.threads.emplace_back([p = &I, role] { p->thread_loop(role); });
  inf.init_seconds = now_s() - t0;
  return std::unique_ptr<GemmEngine>(new GemmEngine(std::move(impl)));
}

double GemmEngine::predicted_speedup(int m, int n, int k) const noexcept {
  const EngineOptions& o = impl_->opt;
  const double dm = m, dn = n, dk = k;
  const double t_cpu = 2.0 * dm * dn * dk / cpu_rate(o);
  const double traffic = 24.0 * (dm + dn) * dk + 32.0 * dm * dn;  // host bytes, see the header
  const double t_gpu = traffic / (o.host_gbs * 1e9) + o.latency_s;
  return t_cpu / t_gpu;
}

bool GemmEngine::eligible(char ta_c, char tb_c, int m, int n, int k) const noexcept {
  const EngineOptions& o = impl_->opt;
  if (m <= 0 || n <= 0 || k <= 0) return false;
  if (2.0 * m * n * static_cast<double>(k) < o.min_flops) return false;
  if (predicted_speedup(m, n, k) < o.min_speedup) return false;
  const bool split_n = m <= n;
  const std::int64_t shared_bytes = 8 * static_cast<std::int64_t>(split_n ? m : n) * k;
  if (shared_bytes > static_cast<std::int64_t>(impl_->shared_cap)) return false;
  // One unit of the split dimension: its operand (k) and its result (the other dimension).
  const std::int64_t per_in = 8 * static_cast<std::int64_t>(k);
  const std::int64_t per_out = 8 * static_cast<std::int64_t>(split_n ? m : n);
  const std::int64_t tile = static_cast<std::int64_t>(o.slot_bytes) / std::max(per_in, per_out);
  // The shared operand is staged column by column: one column must fit a staging buffer.
  const std::int64_t shared_rows = split_n ? (is_trans(ta_c) ? k : m) : (is_trans(tb_c) ? n : k);
  return tile >= kMinTile && 8 * shared_rows <= static_cast<std::int64_t>(o.slot_bytes);
}

bool GemmEngine::dgemm(char ta_c, char tb_c, int m, int n, int k, double alpha, const double* a,
                       int lda, const double* b, int ldb, double beta, double* c, int ldc,
                       CpuDgemm cpu) {
  Impl& I = *impl_;
  if (I.is_broken.load() || !eligible(ta_c, tb_c, m, n, k)) return false;
  std::unique_lock<std::mutex> call_lock(I.call_mu, std::try_to_lock);
  if (!call_lock.owns_lock()) {
    std::lock_guard<std::mutex> lk(I.stats_mu);
    ++I.st.declined_busy;
    return false;
  }
  const double t_start = now_s();
  Call call;
  call.ta = is_trans(ta_c);
  call.tb = is_trans(tb_c);
  call.m = m;
  call.n = n;
  call.k = k;
  call.alpha = alpha;
  call.beta = beta;
  call.a = a;
  call.lda = lda;
  call.b = b;
  call.ldb = ldb;
  call.c = c;
  call.ldc = ldc;
  call.split_n = m <= n;
  call.len = call.split_n ? n : m;
  const std::int64_t per = 8 * std::max<std::int64_t>(k, call.result_per_unit());
  call.tile = static_cast<int>(
      std::min<std::int64_t>(call.len, static_cast<std::int64_t>(I.opt.slot_bytes) / per));
  call.lo = 0;
  call.hi = call.len;
  call.issuers_left = I.opt.pack_threads;
  {
    std::lock_guard<std::mutex> lk(I.mu);
    I.cur = &call;
    I.finished = 0;
    I.queue.clear();
    I.queue_closed = false;
    std::fill(I.slot_free.begin(), I.slot_free.end(), true);
    ++I.gen;
  }
  I.cv.notify_all();

  // CPU share: chunks of ~4 GFlop from the high end, threaded by the BLAS.
  const double unit_flops = 2.0 * static_cast<double>(call.result_per_unit()) * k;
  const int chunk = static_cast<int>(
      std::clamp<double>(std::ceil(4e9 / unit_flops), kMinTile, static_cast<double>(call.len)));
  std::int64_t cpu_units = 0;
  const double t_cpu0 = now_s();
  if (I.opt.hybrid) {
    const int saved = omp_get_max_threads();
    const int cpu_threads = I.opt.cpu_threads > 0
                                ? I.opt.cpu_threads
                                : std::max(1, saved - I.opt.pack_threads - 1);
    omp_set_num_threads(cpu_threads);
    for (;;) {
      int u0 = 0, u1 = 0;
      {
        std::lock_guard<std::mutex> lk(call.range_mu);
        if (call.lo >= call.hi) break;
        const int left = call.hi - call.lo;  // guided, as the GPU's tiles
        const int w = std::min(chunk, std::max(std::min(kMinTile, left), left / 4));
        u1 = call.hi;
        u0 = call.hi - w;
        call.hi = u0;
      }
      I.cpu_range(call, cpu, u0, u1);
      cpu_units += u1 - u0;
    }
    omp_set_num_threads(saved);
  }
  const double t_cpu1 = now_s();
  {
    std::unique_lock<std::mutex> lk(I.mu);
    I.cv.wait(lk, [&] { return I.finished == I.opt.pack_threads + 1; });
    I.cur = nullptr;
  }
  const double t_tail = now_s();
  // Whatever the GPU did not finish (an error, or GPU-only mode stopping early) on the CPU.
  for (const auto& f : call.failed) {
    I.cpu_range(call, cpu, f.first, f.second);
    cpu_units += f.second - f.first;
  }
  if (call.lo < call.hi) {
    I.cpu_range(call, cpu, call.lo, call.hi);
    cpu_units += call.hi - call.lo;
  }
  const double t_end = now_s();
  {
    std::lock_guard<std::mutex> lk(I.stats_mu);
    EngineStats& s = I.st;
    ++s.calls;
    s.tiles += call.tiles;
    s.gpu_flops += unit_flops * static_cast<double>(call.gpu_units);
    s.cpu_flops += unit_flops * static_cast<double>(cpu_units);
    s.h2d_bytes += call.h2d;
    s.d2h_bytes += call.d2h;
    s.seconds += t_end - t_start;
    s.predicted_cpu_s += 2.0 * m * n * static_cast<double>(k) / cpu_rate(I.opt);
    s.shared_s += call.shared_s;
    s.pack_s += call.pack_s;
    s.slot_wait_s += call.slot_wait_s;
    s.sync_s += call.sync_s;
    s.add_s += call.add_s;
    s.cpu_s += t_cpu1 - t_cpu0;
    s.tail_s += t_tail - t_cpu1;
  }
  return true;
}

EngineStats GemmEngine::stats() const {
  std::lock_guard<std::mutex> lk(impl_->stats_mu);
  return impl_->st;
}

const EngineInfo& GemmEngine::info() const noexcept {
  return impl_->info;
}

bool GemmEngine::broken() const noexcept {
  return impl_->is_broken.load();
}

std::string GemmEngine::error() const {
  std::lock_guard<std::mutex> lk(impl_->stats_mu);
  return impl_->err;
}

}  // namespace hs::gpu
