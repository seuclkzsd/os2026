// make_thread_transports (include/hsolve/parallel.hpp): a session between threads of one process,
// for tests of the protocol without MPI. One mailbox per rank; FIFO per (sender, receiver) pair.
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/parallel.hpp"

namespace hs {

namespace {

struct Hub {
  struct Letter {
    int from;
    std::vector<std::uint8_t> bytes;
  };
  explicit Hub(int n) : boxes(static_cast<std::size_t>(n)) {}
  std::mutex mutex;
  std::condition_variable arrived;
  std::vector<std::deque<Letter>> boxes;
};

class ThreadTransport final : public Transport {
 public:
  ThreadTransport(std::shared_ptr<Hub> hub, int rank) : hub_(std::move(hub)), rank_(rank) {}
  int rank() const override { return rank_; }
  int size() const override { return static_cast<int>(hub_->boxes.size()); }
  void send(int to, const std::vector<std::uint8_t>& bytes) override {
    if (to < 0 || to >= size() || to == rank_) HS_THROW(InvalidInput, "Transport::send to rank " << to);
    {
      std::lock_guard<std::mutex> lock(hub_->mutex);
      hub_->boxes[static_cast<std::size_t>(to)].push_back({rank_, bytes});
    }
    hub_->arrived.notify_all();
  }
  std::vector<std::uint8_t> receive(int from, int* source) override {
    std::unique_lock<std::mutex> lock(hub_->mutex);
    auto& box = hub_->boxes[static_cast<std::size_t>(rank_)];
    for (;;) {
      for (auto it = box.begin(); it != box.end(); ++it) {
        if (from != kAnySource && it->from != from) continue;
        std::vector<std::uint8_t> bytes = std::move(it->bytes);
        if (source) *source = it->from;
        box.erase(it);
        return bytes;
      }
      hub_->arrived.wait(lock);
    }
  }
  bool ready(int from) override {
    std::lock_guard<std::mutex> lock(hub_->mutex);
    for (const auto& l : hub_->boxes[static_cast<std::size_t>(rank_)])
      if (from == kAnySource || l.from == from) return true;
    return false;
  }

 private:
  std::shared_ptr<Hub> hub_;
  int rank_;
};

}  // namespace

std::vector<std::unique_ptr<Transport>> make_thread_transports(int n) {
  if (n < 1) HS_THROW(InvalidInput, "make_thread_transports: n = " << n);
  auto hub = std::make_shared<Hub>(n);
  std::vector<std::unique_ptr<Transport>> out;
  for (int r = 0; r < n; ++r) out.push_back(std::make_unique<ThreadTransport>(hub, r));
  return out;
}

}  // namespace hs
