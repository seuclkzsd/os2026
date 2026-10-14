// hs::mpi::MpiTransport (mpi_transport.hpp).
#include "mpi/mpi_transport.hpp"

#include <dlfcn.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <thread>

#include "hsolve/error.hpp"

namespace hs::mpi {

namespace {

constexpr int kTagHeader = 7001;
constexpr int kTagPart = 7002;

void check(int rc, const char* what) {
  if (rc != MPI_SUCCESS) HS_THROW(Internal, "MPI " << what << " failed (" << rc << ")");
}

}  // namespace

MpiTransport::MpiTransport() {
  int init = 0;
  check(MPI_Initialized(&init), "Initialized");
  if (!init) HS_THROW(InvalidInput, "MpiTransport: MPI is not initialized");
  check(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "Comm_rank");
  check(MPI_Comm_size(MPI_COMM_WORLD, &size_), "Comm_size");
}

void MpiTransport::send(int to, const std::vector<std::uint8_t>& bytes) {
  if (to < 0 || to >= size_ || to == rank_) HS_THROW(InvalidInput, "MpiTransport::send to rank " << to);
  const auto n = static_cast<std::uint64_t>(bytes.size());
  check(MPI_Send(&n, 1, MPI_UINT64_T, to, kTagHeader, MPI_COMM_WORLD), "Send");
  for (std::uint64_t off = 0; off < n; off += kChunk) {
    const auto part = static_cast<int>(std::min<std::uint64_t>(kChunk, n - off));
    check(MPI_Send(bytes.data() + off, part, MPI_BYTE, to, kTagPart, MPI_COMM_WORLD), "Send");
  }
}

bool MpiTransport::ready(int from) {
  int flag = 0;
  MPI_Status st;
  check(MPI_Iprobe(from == kAnySource ? MPI_ANY_SOURCE : from, kTagHeader, MPI_COMM_WORLD, &flag, &st), "Iprobe");
  return flag != 0;
}

std::vector<std::uint8_t> MpiTransport::receive(int from, int* source) {
  // Poll with sleeps: a blocking MPI_Recv busy-waits on a core (Open MPI) through whole solves.
  MPI_Status st;
  auto pause = std::chrono::microseconds(50);
  for (;;) {
    int flag = 0;
    check(MPI_Iprobe(from == kAnySource ? MPI_ANY_SOURCE : from, kTagHeader, MPI_COMM_WORLD, &flag, &st), "Iprobe");
    if (flag) break;
    std::this_thread::sleep_for(pause);
    pause = std::min(pause * 2, std::chrono::microseconds(2000));
  }
  const int src = st.MPI_SOURCE;
  std::uint64_t n = 0;
  check(MPI_Recv(&n, 1, MPI_UINT64_T, src, kTagHeader, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "Recv");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(n));
  // The parts follow the header from the same sender in order (MPI's non-overtaking rule).
  for (std::uint64_t off = 0; off < n; off += kChunk) {
    const auto part = static_cast<int>(std::min<std::uint64_t>(kChunk, n - off));
    check(MPI_Recv(bytes.data() + off, part, MPI_BYTE, src, kTagPart, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "Recv");
  }
  if (source) *source = src;
  return bytes;
}

bool init(int* argc, char*** argv) {
  int provided = 0;
  return MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS;
}

void finalize() { MPI_Finalize(); }

int world_size() {
  int size = 0;
  check(MPI_Comm_size(MPI_COMM_WORLD, &size), "Comm_size");
  return size;
}

const char* mpi_library() {
  // What the dynamic linker's global lookup gives every caller (not the executable's PLT slot).
  void* sym = dlsym(RTLD_DEFAULT, "MPI_Init");
  Dl_info info;
  if (sym && dladdr(sym, &info) && info.dli_fname) return info.dli_fname;
  return "";
}

}  // namespace hs::mpi
