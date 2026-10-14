// The leader's side of a multi-process session (remote_helpers.hpp; DESIGN §3.9).
#include "parallel/remote_helpers.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "parallel/protocol.hpp"
#include "parallel/wire.hpp"

namespace hs::parallel {

template <class T>
RemoteHelpers<T>::RemoteHelpers(Transport& peers, std::uint32_t generation)
    : peers_(peers), generation_(generation), helpers_(static_cast<std::size_t>(peers.size())) {
  HS_ASSERT(peers.rank() == 0, "RemoteHelpers on rank " << peers.rank());
}

template <class T>
RemoteHelpers<T>::~RemoteHelpers() {
  ctx_ = nullptr;  // the solve's context is gone (no report notes from here on)
  try {
    for (int h = 1; h <= size(); ++h) {
      Helper& x = helpers_[static_cast<std::size_t>(h)];
      // Collect what is still owed (a solve that threw while a helper worked) before End.
      while (x.owes_ready || (x.task != 0 && x.reply.empty())) receive_one(h);
      x.task = 0;
      x.reply.clear();
      if (sent_) {
        Writer w(Msg::End);
        w.pod(generation_);
        peers_.send(h, w.bytes());
      }
    }
  } catch (...) {
    // A destructor must not throw; a broken transport ends the session anyway (MPI aborts).
  }
}

template <class T>
void RemoteHelpers<T>::abandon(int h, const std::string& why) {
  Helper& x = helpers_.at(static_cast<std::size_t>(h));
  if (!x.usable) return;
  x.usable = false;
  if (ctx_) ctx_->rep.parallel.notes.push_back("helper " + std::to_string(h) + " not used further: " + why);
}

template <class T>
double RemoteHelpers<T>::memory_bytes() const {
  double sum = 0.0;
  for (int h = 1; h <= size(); ++h) {
    const Helper& x = helpers_[static_cast<std::size_t>(h)];
    sum += std::max(x.rss_bytes, x.task != 0 ? x.granted_bytes : 0.0);
  }
  return sum;
}

template <class T>
void RemoteHelpers<T>::await_ready(int h) const {
  while (helpers_[static_cast<std::size_t>(h)].owes_ready) receive_one(h);
}

template <class T>
double RemoteHelpers<T>::count_estimate_mb() const {
  double mb = 0.0;
  for (int h = 1; h <= size(); ++h) {
    if (!helpers_[static_cast<std::size_t>(h)].usable) continue;
    await_ready(h);
    mb = std::max(mb, helpers_[static_cast<std::size_t>(h)].count_mb);
  }
  return mb;
}

template <class T>
int RemoteHelpers<T>::helper_threads() const {
  int t = 0;
  for (int h = 1; h <= size(); ++h) t = std::max(t, helpers_[static_cast<std::size_t>(h)].threads);
  return t;
}

template <class T>
int RemoteHelpers<T>::receive_one(int h) const {
  int from = 0;
  std::vector<std::uint8_t> msg = peers_.receive(h == 0 ? Transport::kAnySource : h, &from);
  if (from < 1 || from > size()) HS_THROW(Internal, "parallel session: a message from rank " << from);
  Helper& x = helpers_[static_cast<std::size_t>(from)];
  Reader r(msg);
  const Msg type = r.type();
  const auto gen = r.pod<std::uint32_t>();
  if (gen != generation_) return from;  // a stale reply of an earlier solve (not expected)
  if (type == Msg::Ready) {
    const driver::HelperError e = get_error(r);
    x.count_mb = r.pod<double>();
    x.rss_bytes = r.pod<double>();
    x.threads = r.pod<int>();
    x.owes_ready = false;
    x.ready = e.ok;
    if (!e.ok) {
      x.usable = false;
      if (ctx_)
        ctx_->rep.parallel.notes.push_back("helper " + std::to_string(from) + " could not take the problem: " +
                                           to_string(e.code) + ": " + e.what);
    }
    return from;
  }
  if (type != Msg::CountDone && type != Msg::SliceDone)
    HS_THROW(Internal, "parallel session: unexpected message " << static_cast<int>(type) << " from rank " << from);
  if (x.task == 0) HS_THROW(Internal, "parallel session: a reply from idle rank " << from);
  x.reply = std::move(msg);
  return from;
}

template <class T>
void RemoteHelpers<T>::ensure_idle(int h) const {
  const Helper& x = helpers_.at(static_cast<std::size_t>(h));
  HS_ASSERT(h >= 1 && h <= size() && x.task == 0 && x.usable && sent_, "helper " << h << " cannot take a task");
}

template <class T>
void RemoteHelpers<T>::share(driver::Ctx& c, const ShiftedPencil<T>& pencil, const SparseFactorization<T>& main,
                             const driver::ProblemInfo& info) {
  ctx_ = &c;
  pencil_ = &pencil;
  main_ = &main;
  info_ = info;
  shared_ = true;
  sent_ = false;  // (again, after a fallback to double factors: the helpers set up anew)
}

// The problem in one message: the pencil's matrices as the solve factorizes them (after its
// Hermitian check), the working configuration, the norms and shift, the factor options and the
// main analysis' pivot order. Sending blocks until each helper has the data (shared memory between
// ranks on one node); the helpers then build their pencil and, for a certificate, analyse their
// count instance, and answer Ready - all while the leader waits here.
template <class T>
void RemoteHelpers<T>::prepare(driver::Ctx& c) {
  ctx_ = &c;
  HS_ASSERT(shared_, "prepare() before share()");
  if (!sent_) {
    Timer timer;
    for (int h = 1; h <= size(); ++h) {
      while (helpers_[static_cast<std::size_t>(h)].owes_ready) receive_one(h);  // an earlier attempt's
      HS_ASSERT(helpers_[static_cast<std::size_t>(h)].task == 0, "prepare() while helper " << h << " works");
    }
    Writer w(Msg::Setup);
    w.pod(generation_);
    w.pod(static_cast<std::uint8_t>(is_complex_v<T> ? 1 : 0));
    w.str(config_to_json(c.cfg, -1));
    w.pod(info_);
    put(w, main_->options());
    w.vec(main_->pivot_order());
    w.csr(pencil_->a());
    w.pod(static_cast<std::uint8_t>(pencil_->b() ? 1 : 0));
    if (pencil_->b()) w.csr(*pencil_->b());
    for (int h = 1; h <= size(); ++h) {
      Helper& x = helpers_[static_cast<std::size_t>(h)];
      if (!x.usable) continue;
      peers_.send(h, w.bytes());
      x.owes_ready = true;
      x.ready = false;
    }
    sent_ = true;
    c.rep.parallel.share_s += timer.seconds();
    driver::event(c, "helpers_share", {{"helpers", static_cast<std::int64_t>(size())},
                                       {"bytes", static_cast<std::int64_t>(w.bytes().size())},
                                       {"seconds", timer.seconds()}});
  }
  Timer wait;
  bool waited = false;
  for (int h = 1; h <= size(); ++h) {
    waited = waited || helpers_[static_cast<std::size_t>(h)].owes_ready;
    await_ready(h);
  }
  if (waited) {
    c.rep.parallel.wait_s += wait.seconds();
    driver::event(c, "helpers_ready", {{"seconds", wait.seconds()}, {"count_mb", count_estimate_mb()}});
  }
}

template <class T>
void RemoteHelpers<T>::post_count(driver::Ctx& c, int h, const driver::CountTask& t) {
  ctx_ = &c;
  await_ready(h);
  ensure_idle(h);
  Helper& x = helpers_[static_cast<std::size_t>(h)];
  Writer w(Msg::Count);
  w.pod(generation_);
  w.pod(t);
  peers_.send(h, w.bytes());
  x.task = static_cast<int>(Msg::Count);
  x.granted_bytes = x.rss_bytes + t.cap_mb * 1e6;
}

template <class T>
driver::CountReply RemoteHelpers<T>::wait_count(driver::Ctx& c, int h) {
  Helper& x = helpers_.at(static_cast<std::size_t>(h));
  HS_ASSERT(x.task == static_cast<int>(Msg::Count), "wait_count without a count on helper " << h);
  Timer timer;
  while (x.reply.empty()) receive_one(h);
  c.rep.parallel.wait_s += timer.seconds();
  Reader r(x.reply);
  r.type();
  r.pod<std::uint32_t>();
  driver::CountReply reply = get_count_reply(r);
  x.reply.clear();
  x.task = 0;
  x.rss_bytes = reply.rss_bytes;
  c.rep.parallel.helper_busy_s += reply.seconds;
  return reply;
}

template <class T>
void RemoteHelpers<T>::post_slice(driver::Ctx& c, int h, const driver::SliceTask& t) {
  ctx_ = &c;
  await_ready(h);
  ensure_idle(h);
  Helper& x = helpers_[static_cast<std::size_t>(h)];
  Writer w(Msg::Slice);
  w.pod(generation_);
  w.pod(t);
  peers_.send(h, w.bytes());
  x.task = static_cast<int>(Msg::Slice);
  x.granted_bytes = x.rss_bytes + t.budget_gb * 1073741824.0;
}

template <class T>
int RemoteHelpers<T>::wait_slice(driver::Ctx& c, driver::SliceReply<T>& out) {
  Timer timer;
  int h = 0;
  for (;;) {
    for (int i = 1; i <= size() && h == 0; ++i) {
      const Helper& x = helpers_[static_cast<std::size_t>(i)];
      if (x.task == static_cast<int>(Msg::Slice) && !x.reply.empty()) h = i;
    }
    if (h != 0) break;
    HS_ASSERT(busy() > 0, "wait_slice without a slice in flight");
    receive_one(0);
  }
  c.rep.parallel.wait_s += timer.seconds();
  Helper& x = helpers_[static_cast<std::size_t>(h)];
  Reader r(x.reply);
  r.type();
  r.pod<std::uint32_t>();
  out = get_slice_reply<T>(r);
  x.reply.clear();
  x.reply.shrink_to_fit();
  x.task = 0;
  x.rss_bytes = out.rss_bytes;
  c.rep.parallel.helper_busy_s += out.seconds;
  return h;
}

template <class T>
int RemoteHelpers<T>::busy() const {
  int k = 0;
  for (int h = 1; h <= size(); ++h) k += helpers_[static_cast<std::size_t>(h)].task != 0 ? 1 : 0;
  return k;
}

template class RemoteHelpers<double>;
template class RemoteHelpers<cplx>;

}  // namespace hs::parallel
