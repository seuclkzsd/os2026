// MatrixMarket reader (see mtx.hpp).
#include "io/mtx.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
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

  /// Back to the first line (the buffer is reused).
  void rewind() {
    in_.clear();
    in_.seekg(0);
    if (!in_) HS_THROW(Io, name_ << ": cannot rewind");
    pos_ = end_ = 0;
    eof_ = false;
    line_no_ = 0;
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

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// True when `tok`, a decimal that std::from_chars matched completely but reported as
// result_out_of_range, is a nonzero value below the smallest subnormal double (~4.9e-324) rather
// than one above DBL_MAX: from_chars reports both cases with the same code. They are told apart
// by the decimal exponent of the leading nonzero digit, which is <= -324 for underflow and >= 308
// for overflow.
bool underflows_double(std::string_view tok) {
  constexpr std::int64_t kExponentCap = std::int64_t{1} << 40;  // any larger |exponent| is equivalent
  std::size_t k = (!tok.empty() && tok[0] == '-') ? 1 : 0;
  std::int64_t lead = 0;  // decimal exponent of the leading nonzero digit of the mantissa
  bool nonzero = false;
  for (; k < tok.size() && is_digit(tok[k]); ++k) {
    if (nonzero) {
      ++lead;
    } else if (tok[k] != '0') {
      nonzero = true;
    }
  }
  if (k < tok.size() && tok[k] == '.') {
    std::int64_t zeros = 0;  // fraction zeros before the leading nonzero digit
    for (++k; k < tok.size() && is_digit(tok[k]); ++k) {
      if (nonzero) continue;
      if (tok[k] == '0') {
        ++zeros;
      } else {
        nonzero = true;
        lead = -(zeros + 1);
      }
    }
  }
  if (!nonzero) return false;
  std::int64_t exponent = 0;
  if (k < tok.size() && (tok[k] == 'e' || tok[k] == 'E')) {
    ++k;
    const bool negative = k < tok.size() && tok[k] == '-';
    if (k < tok.size() && (tok[k] == '-' || tok[k] == '+')) ++k;
    for (; k < tok.size() && is_digit(tok[k]); ++k)
      exponent = std::min(kExponentCap, exponent * 10 + (tok[k] - '0'));
    if (negative) exponent = -exponent;
  }
  return lead + exponent < 0;
}

// from_chars rejects a leading '+', which some writers emit. A real value below the double range
// (e.g. round-off written as 1e-400) reads as a signed zero, as with strtod and scipy.io.mmread;
// coordinate zeros are then dropped per drop_zeros and array zeros are structural absence.
// Overflow stays an error.
template <class V>
V parse_number(std::string_view tok, const LineReader& lr, const char* what) {
  if (!tok.empty() && tok.front() == '+') tok.remove_prefix(1);
  const char* const last = tok.data() + tok.size();
  V v{};
  const auto [end, ec] = std::from_chars(tok.data(), last, v);
  if constexpr (std::is_floating_point_v<V>) {
    if (!tok.empty() && ec == std::errc::result_out_of_range && end == last &&
        underflows_double(tok))
      return tok.front() == '-' ? -V(0) : V(0);
  }
  if (tok.empty() || ec != std::errc() || end != last)
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

// Reads the value lines, checking each in file order, and hands every stored entry to
// put(i, j, v) (0-based). Coordinate lines are "i j [value]" (1-based); array lines hold one
// value each, column-major, lower triangle only for symmetric/hermitian matrices, and their zeros
// are not handed over (structural absence in a dense format).
template <class T, class Put>
void read_entries(LineReader& lr, const Banner& b, index_t n, std::uint64_t expected, Put&& put) {
  const bool one_triangle = b.symmetry != MtxSymmetry::General;
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
    put(i, j, v);
  }
  if (count != expected)
    HS_THROW(Io, lr.name() << ": file ends after " << count << " of " << expected << " entries");
}

// Pass 1 over a coordinate file: row pointers from the row tokens of the first `expected` entry
// lines alone. A line whose row token cannot be read is not counted; pass 2 meets it in file order
// and reports it exactly as a single pass would.
std::vector<index_t> count_rows(LineReader& lr, index_t n, std::uint64_t expected) {
  std::vector<index_t> rowptr(static_cast<std::size_t>(n) + 1, 0);
  std::uint64_t count = 0;
  std::string_view line;
  while (count < expected && lr.next(line)) {
    if (blank_or_comment(line)) continue;
    ++count;
    std::string_view tok = Tokens(line).next();
    if (!tok.empty() && tok.front() == '+') tok.remove_prefix(1);
    std::int64_t i = 0;
    const char* last = tok.data() + tok.size();
    const auto [end, ec] = std::from_chars(tok.data(), last, i);
    if (ec == std::errc() && end == last && i >= 1 && i <= n) ++rowptr[i];  // row i - 1
  }
  std::partial_sum(rowptr.begin(), rowptr.end(), rowptr.begin());
  return rowptr;
}

template <class T>
CsrMatrix<T> read_values(LineReader& lr, const Banner& b, index_t n, std::uint64_t declared_nnz,
                         bool drop_zeros, std::uint64_t file_size, AssemblyStats& stats) {
  const bool one_triangle = b.symmetry != MtxSymmetry::General;
  const auto un = static_cast<std::uint64_t>(n);  // n <= 2^31: n*n fits
  const std::uint64_t expected =
      b.coordinate ? declared_nnz : (one_triangle ? un * (un + 1) / 2 : un * un);
  // Every value line takes at least two bytes ("0\n"); a larger count means a corrupt header,
  // caught before reserving memory for it.
  if (expected > file_size / 2)
    fail_at(lr, "declares " + std::to_string(expected) + " entries, more than the file can hold");

  if (!b.coordinate) {  // dense array format (small matrices): triplets, then CSR
    std::vector<std::int32_t> rows, cols;
    std::vector<T> vals;
    read_entries<T>(lr, b, n, expected, [&](index_t i, index_t j, const T& v) {
      rows.push_back(static_cast<std::int32_t>(i));
      cols.push_back(static_cast<std::int32_t>(j));
      vals.push_back(v);
    });
    return csr_from_triplets<T>(n, n, rows.data(), cols.data(), vals.data(), vals.size(),
                                drop_zeros, stats);
  }

  // Coordinate format: two passes over the file instead of a triplet copy of the entries (which
  // cost 16-24 bytes per entry on top of the result). Pass 1 counts the entries of every row,
  // pass 2 parses and checks every line in file order and places each entry into the next slot
  // of its row; rows keep the file order, so canonicalize_csr sums duplicates exactly as
  // csr_from_triplets would. Peak: the result plus the line buffer.
  const std::uint64_t header_lines = lr.line_number();
  CsrMatrix<T> m;
  m.nrows = m.ncols = n;
  m.rowptr = count_rows(lr, n, expected);
  m.colind.resize(static_cast<std::size_t>(m.rowptr[n]));
  m.values.resize(static_cast<std::size_t>(m.rowptr[n]));
  lr.rewind();
  std::string_view skipped;
  for (std::uint64_t k = 0; k < header_lines; ++k) lr.next(skipped);
  std::vector<index_t> filled(static_cast<std::size_t>(n), 0);
  const auto changed = [&lr] { HS_THROW(Io, lr.name() << ": the file changed while it was read"); };
  read_entries<T>(lr, b, n, expected, [&](index_t i, index_t j, const T& v) {
    const index_t dst = m.rowptr[i] + filled[i]++;
    if (dst >= m.rowptr[i + 1]) changed();
    m.colind[dst] = static_cast<std::int32_t>(j);
    m.values[dst] = v;
  });
  for (index_t i = 0; i < n; ++i)
    if (filled[i] != m.rowptr[i + 1] - m.rowptr[i]) changed();
  stats = canonicalize_csr(m, drop_zeros);
  return m;
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
    out.matrix = read_values<cplx>(lr, b, n, static_cast<std::uint64_t>(nnz), drop_zeros,
                                   file_size, out.stats);
  } else {
    out.matrix = read_values<double>(lr, b, n, static_cast<std::uint64_t>(nnz), drop_zeros,
                                     file_size, out.stats);
  }
  return out;
}

}  // namespace hs::io
