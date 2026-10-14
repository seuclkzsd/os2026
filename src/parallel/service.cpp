// The helpers' side of a multi-process session (hs::serve, include/hsolve/parallel.hpp;
// DESIGN §3.9): per solve the problem the leader shares (its matrices, configuration, pivot order),
// an inertia-only MUMPS instance for certificate counts and a main object for slices, both on the
// leader's pivot order and with this process's OpenMP threads; every task is answered, a failure
// as an error reply (the leader then does the work itself). Drop / End free the problem and all
// built on it at once (the leader takes the work back or the solve ended), and return the freed
// heap to the system, so that the resident memory the helper reports falls to its idle level.
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "driver/helpers.hpp"
#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/parallel.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "parallel/protocol.hpp"
#include "parallel/wire.hpp"

#if defined(__GLIBC__)  // (defined by the C++ headers above)
#include <malloc.h>
#endif

namespace hs {

namespace {

using parallel::Msg;
using parallel::Reader;
using parallel::Writer;

driver::HelperError error_of(const std::exception& e) {
  driver::HelperError err;
  err.ok = false;
  err.what = e.what();
  if (const auto* he = dynamic_cast<const Error*>(&e)) {
    err.code = he->code();
    err.memory = he->code() == Error::Code::OutOfMemoryBudget;
  } else if (dynamic_cast<const std::bad_alloc*>(&e)) {
    err.code = Error::Code::OutOfMemoryBudget;
    err.memory = true;
    err.what = "allocation failure (std::bad_alloc)";
  }
  return err;
}

/// One solve's problem on a helper.
class Problem {
 public:
  virtual ~Problem() = default;
  virtual driver::CountReply count(const driver::CountTask& t) = 0;
  virtual void slice(Writer& w, const driver::SliceTask& t) = 0;
  /// Estimated working memory of one count (MB; analyses the count instance).
  virtual double prepare_count() = 0;
  /// The leader will hand out a certificate count (ProblemInfo::prepare_count).
  virtual bool wants_count() const = 0;
};

template <class T>
class ProblemOf final : public Problem {
 public:
  ProblemOf(Reader& r, Logger* log) : log_(log) {
    cfg_ = parse_config(r.str());
    info_ = r.pod<driver::ProblemInfo>();
    fopts_ = parallel::get_factor_options(r);
    order_ = r.vec<int>();
    a_ = r.csr<T>();
    if (r.pod<std::uint8_t>()) {
      b_ = std::make_unique<CsrMatrix<T>>(r.csr<T>());
    }
    pencil_ = std::make_unique<ShiftedPencil<T>>(a_, b_.get());
    // This process's threads (the launcher's OMP_NUM_THREADS), not the leader's.
    fopts_.threads = max_threads();
    cfg_.threads = 0;
    rep_.memory.measure = MemoryMeasure::Resident;
    rep_.parallel.processes = 1;
  }

  double prepare_count() override { return count_instance().estimate().incore_mb; }
  bool wants_count() const override { return info_.prepare_count; }

  driver::CountReply count(const driver::CountTask& t) override {
    cfg_.memory_budget_gb = static_cast<double>(current_rss_bytes()) / 1073741824.0 + t.budget_gb;
    driver::Ctx c{cfg_, log_, rep_, clock_};
    c.held = held_;
    driver::CountReply r = driver::count_on_helper<T>(c, count_instance(), t);
    held_ = c.held;
    return r;
  }

  void slice(Writer& w, const driver::SliceTask& t) override {
    cfg_.memory_budget_gb = static_cast<double>(current_rss_bytes()) / 1073741824.0 + t.budget_gb;
    driver::Ctx c{cfg_, log_, rep_, clock_};
    c.held = held_;
    rep_.warnings.clear();
    driver::SliceReply<T> r;
    try {
      r = driver::slice_on_helper<T>(c, *pencil_, main_object(), info_, t);
    } catch (const std::exception& e) {
      r = driver::SliceReply<T>{};
      r.error = error_of(e);
    }
    held_ = c.held;
    r.rss_bytes = static_cast<double>(current_rss_bytes());
    parallel::put(w, r);
  }

 private:
  // An InertiaCounter on the main object (its pivot order: the leader's, rebuilt by MUMPS like the
  // leader's own counter's). The counter keeps the pencil, not the source: when no slice is
  // expected (nearest targets) the main object's analysis is dropped again.
  InertiaCounter<T>& count_instance() {
    if (!counter_) {
      const bool keep_main = main_ != nullptr;
      counter_ = std::make_unique<InertiaCounter<T>>(main_object(), fopts_);
      if (!keep_main && info_.prepare_count) main_.reset();
    }
    return *counter_;
  }
  // The main object of slices: the leader's options on its pivot order, analysed at its shift.
  SparseFactorization<T>& main_object() {
    if (!main_) {
      auto f = make_mumps_factorization<T>(fopts_);
      f->set_pivot_order(order_);
      f->analyze(*pencil_, 1.0, -info_.sigma0);
      main_ = std::move(f);
    }
    return *main_;
  }

  Logger* log_;
  SolverConfig cfg_;
  driver::ProblemInfo info_;
  FactorOptions fopts_;
  std::vector<int> order_;
  CsrMatrix<T> a_;
  std::unique_ptr<CsrMatrix<T>> b_;
  std::unique_ptr<ShiftedPencil<T>> pencil_;
  std::unique_ptr<SparseFactorization<T>> main_;
  std::unique_ptr<InertiaCounter<T>> counter_;
  SolveReport rep_;
  driver::PhaseClock clock_;
  driver::FactorHeld held_{};
};

/// Frees `problem` and gives the freed heap back to the system (glibc keeps freed blocks below its
/// mmap threshold resident otherwise); returns the resident memory then: the helper's idle level.
double release(std::unique_ptr<Problem>& problem) {
  problem.reset();
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
  return static_cast<double>(current_rss_bytes());
}

void send_ready(Transport& peers, std::uint32_t gen, const driver::HelperError& e, double count_mb, double idle) {
  Writer w(Msg::Ready);
  w.pod(gen);
  parallel::put(w, e);
  w.pod(count_mb);
  w.pod(static_cast<double>(current_rss_bytes()));
  w.pod(max_threads());
  w.pod(idle);
  peers.send(0, w.bytes());
}

}  // namespace

void serve(Transport& peers, Logger* log) {
  HS_ASSERT(peers.rank() > 0, "serve() on the leader");
  std::unique_ptr<Problem> problem;
  std::uint32_t generation = 0;
  for (;;) {
    const std::vector<std::uint8_t> msg = peers.receive(0);
    Reader r(msg);
    const Msg type = r.type();
    if (type == Msg::Quit) return;
    const auto gen = r.pod<std::uint32_t>();
    switch (type) {
      case Msg::Setup: {
        const double idle = release(problem);  // the previous problem first: it is not needed any more
        generation = gen;
        driver::HelperError err;
        double count_mb = 0.0;
        try {
          const bool complex = r.pod<std::uint8_t>() != 0;
          if (complex)
            problem = std::make_unique<ProblemOf<cplx>>(r, log);
          else
            problem = std::make_unique<ProblemOf<double>>(r, log);
        } catch (const std::exception& e) {
          problem.reset();
          err = error_of(e);
        }
        if (problem && err.ok && problem->wants_count()) {
          // A nearest certificate will count: analyse now, while the leader factorizes.
          try {
            count_mb = problem->prepare_count();
          } catch (const std::exception& e) {
            err = error_of(e);
          }
        }
        send_ready(peers, gen, err, count_mb, idle);
        break;
      }
      case Msg::Count: {
        const auto t = r.pod<driver::CountTask>();
        driver::CountReply reply;
        if (!problem || gen != generation) {
          reply.error.ok = false;
          reply.error.what = "no problem shared for this solve";
        } else {
          try {
            reply = problem->count(t);
          } catch (const std::exception& e) {
            reply = driver::CountReply{};
            reply.error = error_of(e);
          }
        }
        reply.rss_bytes = static_cast<double>(current_rss_bytes());
        Writer w(Msg::CountDone);
        w.pod(gen);
        parallel::put(w, reply);
        peers.send(0, w.bytes());
        break;
      }
      case Msg::Slice: {
        const auto t = r.pod<driver::SliceTask>();
        Writer w(Msg::SliceDone);
        w.pod(gen);
        if (!problem || gen != generation) {
          driver::SliceReply<double> none;
          none.error.ok = false;
          none.error.what = "no problem shared for this solve";
          parallel::put(w, none);
        } else {
          problem->slice(w, t);
        }
        peers.send(0, w.bytes());
        break;
      }
      case Msg::Drop: {
        // The leader takes the work back (or this batch of tasks is done): free everything now and
        // say how much is left - its budget is charged with this helper's memory until then.
        const double idle = release(problem);
        Writer w(Msg::Dropped);
        w.pod(gen);
        w.pod(idle);
        peers.send(0, w.bytes());
        break;
      }
      case Msg::End:
        if (gen == generation) release(problem);
        break;
      default:
        HS_THROW(Internal, "parallel session: unexpected message " << static_cast<int>(type) << " on a helper");
    }
  }
}

void release_helpers(Transport& peers) {
  HS_ASSERT(peers.rank() == 0, "release_helpers() on a helper");
  for (int h = 1; h < peers.size(); ++h) {
    Writer w(Msg::Quit);
    peers.send(h, w.bytes());
  }
}

}  // namespace hs
