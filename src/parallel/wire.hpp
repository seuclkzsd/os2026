// Binary messages of multi-process solves (module src/parallel, internal; DESIGN §3.9): the
// session protocol between the leader (rank 0) and its helpers and its encoding. Both sides run the
// same build of hsolve on one machine, so values are copied in native byte order and layout; every
// read is bounds-checked (a truncated or foreign message throws Internal, never reads past the end).
//
// Protocol (one outstanding task per helper; replies in order):
//   leader -> helper   Setup   generation, scalar kind, configuration (JSON), ProblemInfo, factor
//                              options, pivot order, A and B (CSR)          -> Ready
//                      Count   generation, CountTask                         -> CountDone
//                      Slice   generation, SliceTask                         -> SliceDone
//                      Drop    generation: drop the problem now              -> Dropped
//                      End     generation: drop the problem (no reply)
//                      Quit    leave serve() (no reply)
// Ready, CountDone, SliceDone and Dropped carry the helper's resident memory; Ready and Dropped
// also its idle level (its resident memory without a problem), so that the leader charges the
// budget with what a helper holds for the solve (remote_helpers.hpp).
// The generation numbers a solve's problem; a reply carries it, and the leader ignores replies of
// an older generation (none are expected: it collects every reply before it moves on).
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"

namespace hs::parallel {

enum class Msg : std::uint8_t {
  Setup = 1,
  Ready = 2,
  Count = 3,
  CountDone = 4,
  Slice = 5,
  SliceDone = 6,
  End = 7,
  Quit = 8,
  Drop = 9,
  Dropped = 10,
};

class Writer {
 public:
  explicit Writer(Msg type) { pod(static_cast<std::uint8_t>(type)); }
  template <class P>
  void pod(const P& v) {
    static_assert(std::is_trivially_copyable_v<P>, "Writer::pod needs a trivially copyable type");
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(P));
  }
  template <class P>
  void vec(const std::vector<P>& v) {
    static_assert(std::is_trivially_copyable_v<P>, "Writer::vec needs a trivially copyable type");
    pod(static_cast<std::uint64_t>(v.size()));
    const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
    buf_.insert(buf_.end(), p, p + v.size() * sizeof(P));
  }
  /// n elements at p (their count first), e.g. a dense matrix's columns.
  template <class P>
  void raw(const P* p, std::uint64_t n) {
    static_assert(std::is_trivially_copyable_v<P>, "Writer::raw needs a trivially copyable type");
    pod(n);
    const auto* b = reinterpret_cast<const std::uint8_t*>(p);
    buf_.insert(buf_.end(), b, b + n * sizeof(P));
  }
  void str(const std::string& s) {
    pod(static_cast<std::uint64_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
  void strs(const std::vector<std::string>& v) {
    pod(static_cast<std::uint64_t>(v.size()));
    for (const auto& s : v) str(s);
  }
  template <class T>
  void csr(const CsrMatrix<T>& m) {
    pod(m.nrows);
    pod(m.ncols);
    vec(m.rowptr);
    vec(m.colind);
    vec(m.values);
  }
  std::vector<std::uint8_t>& bytes() { return buf_; }

 private:
  std::vector<std::uint8_t> buf_;
};

class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& bytes) : buf_(bytes) {}
  Msg type() {
    pos_ = 0;
    return static_cast<Msg>(pod<std::uint8_t>());
  }
  template <class P>
  P pod() {
    static_assert(std::is_trivially_copyable_v<P>, "Reader::pod needs a trivially copyable type");
    need(sizeof(P));
    P v;
    std::memcpy(&v, buf_.data() + pos_, sizeof(P));
    pos_ += sizeof(P);
    return v;
  }
  template <class P>
  std::vector<P> vec() {
    const auto n = pod<std::uint64_t>();
    if (n > (buf_.size() - pos_) / sizeof(P)) HS_THROW(Internal, "parallel message: truncated array");
    std::vector<P> v(static_cast<std::size_t>(n));
    if (n) std::memcpy(v.data(), buf_.data() + pos_, n * sizeof(P));
    pos_ += n * sizeof(P);
    return v;
  }
  /// Exactly n elements (as Writer::raw wrote them) into p.
  template <class P>
  void raw(P* p, std::uint64_t n) {
    if (pod<std::uint64_t>() != n) HS_THROW(Internal, "parallel message: array of an unexpected size");
    if (n > (buf_.size() - pos_) / sizeof(P)) HS_THROW(Internal, "parallel message: truncated array");
    if (n) std::memcpy(p, buf_.data() + pos_, n * sizeof(P));
    pos_ += n * sizeof(P);
  }
  std::string str() {
    const auto n = pod<std::uint64_t>();
    need(n);
    std::string s(reinterpret_cast<const char*>(buf_.data() + pos_), static_cast<std::size_t>(n));
    pos_ += n;
    return s;
  }
  std::vector<std::string> strs() {
    const auto n = pod<std::uint64_t>();
    if (n > buf_.size()) HS_THROW(Internal, "parallel message: truncated string list");
    std::vector<std::string> v;
    for (std::uint64_t i = 0; i < n; ++i) v.push_back(str());
    return v;
  }
  template <class T>
  CsrMatrix<T> csr() {
    CsrMatrix<T> m;
    m.nrows = pod<index_t>();
    m.ncols = pod<index_t>();
    m.rowptr = vec<index_t>();
    m.colind = vec<std::int32_t>();
    m.values = vec<T>();
    return m;
  }
  bool done() const { return pos_ == buf_.size(); }

 private:
  void need(std::uint64_t n) const {
    if (n > buf_.size() - pos_) HS_THROW(Internal, "parallel message: truncated");
  }
  const std::vector<std::uint8_t>& buf_;
  std::size_t pos_ = 0;
};

}  // namespace hs::parallel
