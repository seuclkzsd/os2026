// The leader's side of a multi-process session (module src/parallel, internal; DESIGN §3.9):
// driver::HelpersOf<T> over an hs::Transport. Scalar set: T in {double, cplx}.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "driver/helpers.hpp"
#include "hsolve/parallel.hpp"

namespace hs::parallel {

template <class T>
class RemoteHelpers final : public driver::HelpersOf<T> {
 public:
  /// peers.rank() must be 0. generation: this solve's number in the session (fresh per solve).
  RemoteHelpers(Transport& peers, std::uint32_t generation, const ParallelOptions& options = {});
  /// Ends the solve on the helpers (End: they drop the problem). Replies still owed are collected
  /// first, so that the next solve starts in step.
  ~RemoteHelpers() override;
  RemoteHelpers(const RemoteHelpers&) = delete;
  RemoteHelpers& operator=(const RemoteHelpers&) = delete;

  int size() const override { return static_cast<int>(helpers_.size()) - 1; }
  const ParallelOptions& options() const override { return options_; }
  bool usable(int h) const override { return helpers_.at(static_cast<std::size_t>(h)).usable; }
  void abandon(int h, const std::string& why) override;
  bool shared() const override { return shared_; }
  void prepare(driver::Ctx& c) override;
  void drop(driver::Ctx& c, const std::string& why) override;
  double memory_bytes() const override;
  double count_estimate_mb() const override;
  int helper_threads() const override;
  void post_count(driver::Ctx& c, int h, const driver::CountTask& t) override;
  driver::CountReply wait_count(driver::Ctx& c, int h) override;

  void share(driver::Ctx& c, const ShiftedPencil<T>& pencil, const SparseFactorization<T>& main,
             const driver::ProblemInfo& info) override;
  void post_slice(driver::Ctx& c, int h, const driver::SliceTask& t) override;
  int wait_slice(driver::Ctx& c, driver::SliceReply<T>& out) override;
  int busy() const override;

 private:
  struct Helper {
    bool usable = true;
    bool ready = false;          ///< its Ready (after share) has arrived
    bool owes_ready = false;     ///< a Setup was sent, its Ready not received yet
    bool holds = false;          ///< it got the problem (Setup) and has not dropped it since
    bool owes_dropped = false;   ///< a Drop was sent, its Dropped not received yet
    int task = 0;                ///< 0 idle, else the Msg of the task in flight (Count, Slice)
    double rss_bytes = 0.0;      ///< resident memory as last reported
    double idle_bytes = 0.0;     ///< its resident memory without a problem (Ready, Dropped)
    double granted_bytes = 0.0;  ///< memory (above idle) the task in flight may reach
    double count_mb = 0.0;       ///< estimate of one count (Ready)
    int threads = 0;
    std::vector<std::uint8_t> reply;  ///< a task reply that arrived while waiting for another helper
    /// What it holds for this solve: its resident memory above its idle level.
    double held() const { return std::max(0.0, rss_bytes - idle_bytes); }
  };
  /// Receives one message (from h, or any helper when h == 0) and files it: Ready updates the
  /// helper, a task reply is kept in Helper::reply. Returns the sender. Charged to the caller.
  int receive_one(int h) const;
  /// Blocks until helper h's Ready (after share) arrived (no-op if none is owed).
  void await_ready(int h) const;
  void ensure_idle(int h) const;

  Transport& peers_;
  std::uint32_t generation_;
  ParallelOptions options_;
  mutable std::vector<Helper> helpers_;  ///< index 0 unused (the leader)
  bool shared_ = false;  ///< share() was called: the problem below is available
  bool sent_ = false;    ///< ... and was sent to the helpers (prepare())
  const ShiftedPencil<T>* pencil_ = nullptr;
  const SparseFactorization<T>* main_ = nullptr;
  driver::ProblemInfo info_;
  driver::Ctx* ctx_ = nullptr;  ///< the solve's context, for the report (set at share/post)
};

}  // namespace hs::parallel
