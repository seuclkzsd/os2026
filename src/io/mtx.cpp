// MatrixMarket reader (see mtx.hpp).
#include "io/mtx.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "hsolve/error.hpp"
#include "io/byte_source.hpp"

namespace hs::io {
namespace {

constexpr std::size_t kChunk = std::size_t{4} << 20;
constexpr std::size_t kMaxLine = std::size_t{1} << 20;  // < kChunk: a partial line always fits
constexpr index_t kMaxOrder = index_t{1} << 31;          // CSR column indices are int32

enum class Field { Real, Integer, Pattern, Complex };

struct Banner {
  bool coordinate = true;
  Field field = Field::Real;
  MtxSymmetry symmetry = MtxSymmetry::General;
};

// Splits a file into lines ("\n" or "\r\n"), reading it in large chunks.
class LineReader {
 public:
  explicit LineReader(const std::filesystem::path& path)
      : in_(open_binary(path)), name_(path.string()), buf_(kChunk) {}

  /// Next line without its terminator; false at the end of the file. The view stays valid until
  /// the next call.
  bool next(std::string_view& line) {
    for (;;) {
      const char* begin = buf_.data() + pos_;
      const auto* nl = static_cast<const char*>(std::memchr(begin, '\n', end_ - pos_));
      if (nl != nullptr || (eof_ && pos_ < end_)) {
        const char* stop = nl != nullptr ? nl : buf_.data() + end_;
        line = std::string_view(begin, static_cast<std::size_t>(stop - begin));
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        pos_ = static_cast<std::size_t>(stop - buf_.data()) + (nl != nullptr ? 1 : 0);
        ++line_no_;
        return true;
      }
      if (eof_) return false;
      const std::size_t rest = end_ - pos_;  // partial line: keep it and read behind it
      if (rest >= kMaxLine)
        HS_THROW(Io, name_ << ":" << line_no_ + 1 << ": line longer than " << kMaxLine << " bytes");
      std::memmove(buf_.data(), buf_.data() + pos_, rest);
      pos_ = 0;
      end_ = rest;
      in_.read(buf_.data() + end_, static_cast<std::streamsize>(buf_.size() - end_));
      if (in_.bad()) HS_THROW(Io, name_ << ": read error");
      const auto got = static_cast<std::size_t>(in_.gcount());
      eof_ = got < buf_.size() - end_;
      end_ += got;
    }
  }

  std::uint64_t line_number() const { return line_no_; }
  const std::string& name() const { return name_; }

 private:
  std::ifstream in_;
  std::string name_;
  std::vector<char> buf_;
  std::size_t pos_ = 0;
  std::size_t end_ = 0;
  bool eof_ = false;
  std::uint64_t line_no_ = 0;
};

class Tokens {
 public:
  explicit Tokens(std::string_view s) : s_(s) {}
  /// Next whitespace-separated token; empty at the end of the line.
  std::string_view next() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t')) ++pos_;
    const std::size_t begin = pos_;
    while (pos_ < s_.size() && s_[pos_] != ' ' && s_[pos_] != '\t') ++pos_;
    return s_.substr(begin, pos_ - begin);
  }

 private:
  std::string_view s_;
  std::size_t pos_ = 0;
};

bool blank_or_comment(std::string_view line) {
  for (const char c : line) {
    if (c != ' ' && c != '\t') return c == '%';
  }
  return true;
}

std::string lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

[[noreturn]] void fail_at(const LineReader& lr, const std::string& what) {
  HS_THROW(Io, lr.name() << ":" << lr.line_number() << ": " << what);
}

// from_chars rejects a leading '+', which some writers emit.
template <class V>
V parse_number(std::string_view tok, const LineReader& lr, const char* what) {
  if (!tok.empty() && tok.front() == '+') tok.remove_prefix(1);
  V v{};
  const auto [end, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
  if (tok.empty() || ec != std::errc() || end != tok.data() + tok.size())
    fail_at(lr, std::string("expected ") + what + ", got '" + std::string(tok) + "'");
  return v;
}

Banner parse_banner(std::string_view line, const LineReader& lr) {
  Tokens t(line);
  if (lower(t.next()) != "%%matrixmarket") fail_at(lr, "missing '%%MatrixMarket' banner");
  const std::string object = lower(t.next()), format = lower(t.next());
  const std::string field = lower(t.next()), symmetry = lower(t.next());
  if (symmetry.empty())
    fail_at(lr, "incomplete banner, expected '%%MatrixMarket matrix <format> <field> <symmetry>'");
  if (object != "matrix")
    HS_THROW(InvalidInput, lr.name() << ": MatrixMarket object '" << object << "' is not a matrix");
  Banner b;
  if (format == "coordinate") {
    b.coordinate = true;
  } else if (format == "array") {
    b.coordinate = false;
  } else {
    fail_at(lr, "unknown MatrixMarket format '" + format + "'");
  }
  if (field == "real" || field == "double") {
    b.field = Field::Real;
  } else if (field == "integer") {
    b.field = Field::Integer;
  } else if (field == "pattern") {
    b.field = Field::Pattern;
  } else if (field == "complex") {
    b.field = Field::Complex;
  } else {
    fail_at(lr, "unknown MatrixMarket field '" + field + "'");
  }
  if (symmetry == "general") {
    b.symmetry = MtxSymmetry::General;
  } else if (symmetry == "symmetric") {
    b.symmetry = MtxSymmetry::Symmetric;
  } else if (symmetry == "hermitian") {
    b.symmetry = MtxSymmetry::Hermitian;
  } else if (symmetry == "skew-symmetric") {
    HS_THROW(InvalidInput, lr.name() << ": skew-symmetric matrices are not Hermitian");
  } else {
    fail_at(lr, "unknown MatrixMarket symmetry '" + symmetry + "'");
  }
  if (!b.coordinate && b.field == Field::Pattern)
    fail_at(lr, "the array format cannot have the pattern field");
  return b;
}

// Reads the value lines. Coordinate lines are "i j [value]" (1-based); array lines hold one
// value each, column-major, lower triangle only for symmetric/hermitian matrices.
template <class T>
CsrMatrix<T> read_entries(LineReader& lr, const Banner& b, index_t n, std::uint64_t declared_nnz,
                          bool drop_zeros, std::uint64_t file_size, AssemblyStats& stats) {
  const bool one_triangle = b.symmetry != MtxSymmetry::General;
  const auto un = static_cast<std::uint64_t>(n);  // n <= 2^31: n*n fits
  const std::uint64_t expected =
      b.coordinate ? declared_nnz : (one_triangle ? un * (un + 1) / 2 : un * un);
  // Every value line takes at least two bytes ("0\n"); a larger count means a corrupt header,
  // caught before reserving memory for it.
  if (expected > file_size / 2)
    fail_at(lr, "declares " + std::to_string(expected) + " entries, more than the file can hold");

  std::vector<std::int32_t> rows, cols;
  std::vector<T> vals;
  if (b.coordinate) {
    rows.reserve(expected);
    cols.reserve(expected);
    vals.reserve(expected);
  }
  std::uint64_t count = 0;
  index_t next_i = 0, next_j = 0;  // array format: position of the next value
  std::string_view line;
  while (lr.next(line)) {
    if (blank_or_comment(line)) continue;
    if (count == expected)
      fail_at(lr, "more entries than the " + std::to_string(expected) + " declared");
    Tokens t(line);
    index_t i = next_i, j = next_j;
    if (b.coordinate) {
      i = parse_number<std::int64_t>(t.next(), lr, "a row index") - 1;
      j = parse_number<std::int64_t>(t.next(), lr, "a column index") - 1;
      if (i < 0 || i >= n || j < 0 || j >= n)
        fail_at(lr, "entry (" + std::to_string(i + 1) + ", " + std::to_string(j + 1) +
                        ") lies outside the " + std::to_string(n) + " x " + std::to_string(n) +
                        " matrix");
    } else if (++next_i == n) {
      ++next_j;
      next_i = one_triangle ? next_j : 0;
    }
    T v;
    if constexpr (is_complex_v<T>) {
      const double re = parse_number<double>(t.next(), lr, "a real part");
      const double im = parse_number<double>(t.next(), lr, "an imaginary part");
      v = T(re, im);
    } else if (b.field == Field::Pattern) {
      v = 1.0;
    } else if (b.field == Field::Integer) {
      v = static_cast<double>(parse_number<std::int64_t>(t.next(), lr, "an integer value"));
    } else {
      v = parse_number<double>(t.next(), lr, "a value");
    }
    if (!t.next().empty()) fail_at(lr, "unexpected extra token");
    ++count;
    if (!b.coordinate && v == T(0)) continue;  // dense format: zeros are structural absence
    rows.push_back(static_cast<std::int32_t>(i));
    cols.push_back(static_cast<std::int32_t>(j));
    vals.push_back(v);
  }
  if (count != expected)
    HS_THROW(Io, lr.name() << ": file ends after " << count << " of " << expected << " entries");
  return csr_from_triplets<T>(n, n, rows.data(), cols.data(), vals.data(), vals.size(),
                              drop_zeros, stats);
}

}  // namespace

MtxMatrix read_matrix_market(const std::filesystem::path& path, bool drop_zeros) {
  const std::uint64_t file_size = regular_file_size(path);
  LineReader lr(path);
  std::string_view line;
  if (!lr.next(line)) HS_THROW(Io, path.string() << ": empty MatrixMarket file");
  const Banner b = parse_banner(line, lr);
  do {
    if (!lr.next(line)) HS_THROW(Io, path.string() << ": MatrixMarket size line missing");
  } while (blank_or_comment(line));
  Tokens t(line);
  const auto m = parse_number<std::int64_t>(t.next(), lr, "the row count");
  const auto n = parse_number<std::int64_t>(t.next(), lr, "the column count");
  const auto nnz = b.coordinate ? parse_number<std::int64_t>(t.next(), lr, "the entry count") : 0;
  if (!t.next().empty()) fail_at(lr, "unexpected extra token on the size line");
  if (m < 0 || n < 0 || nnz < 0) fail_at(lr, "negative size");
  if (m != n)
    HS_THROW(InvalidInput, path.string() << ": matrix is " << m << " x " << n
                                         << "; only square matrices can be loaded");
  if (n > kMaxOrder)
    HS_THROW(InvalidInput, path.string() << ": order " << n << " exceeds the int32 index range");

  MtxMatrix out;
  out.symmetry = b.symmetry;
  if (b.field == Field::Complex) {
    out.matrix = read_entries<cplx>(lr, b, n, static_cast<std::uint64_t>(nnz), drop_zeros,
                                    file_size, out.stats);
  } else {
    out.matrix = read_entries<double>(lr, b, n, static_cast<std::uint64_t>(nnz), drop_zeros,
                                      file_size, out.stats);
  }
  return out;
}

}  // namespace hs::io
