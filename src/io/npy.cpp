// .npy reading, writing and element conversion (include/hsolve/io.hpp, src/io/npy_format.hpp).
// Format reference: numpy/lib/_format_impl.py (NEP 1). A file is
//   "\x93NUMPY" | major | minor | header length (u16 for 1.0, u32 for 2.0/3.0) | header | payload
// where the header is the repr of a Python dict {'descr', 'fortran_order', 'shape'}.
#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "io/byte_source.hpp"
#include "io/npy_format.hpp"

namespace hs {
namespace io {
namespace {

constexpr std::string_view kMagic{"\x93NUMPY", 6};
// numpy rejects headers above 10000 bytes by default (max_header_size, a guard against malicious
// files); allow far more than any real header but still bound the allocation.
constexpr std::size_t kMaxHeaderBytes = std::size_t{1} << 20;
constexpr std::size_t kArrayAlign = 64;           // numpy ARRAY_ALIGN
constexpr std::size_t kGrowthAxisMaxDigits = 21;  // numpy GROWTH_AXIS_MAX_DIGITS

bool is_one_of(char c, std::string_view set) { return set.find(c) != std::string_view::npos; }

bool is_py_space(char c) { return is_one_of(c, " \t\n\r\f\v"); }

std::size_t checked_mul(std::size_t a, std::size_t b, const std::string& what) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    HS_THROW(Io, what << ": array size overflows");
  return a * b;
}

std::string shape_repr(const std::vector<std::size_t>& shape) {
  std::string s = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";  // Python's 1-tuple
  return s + ")";
}

// Recursive-descent parser for the header, a Python dict literal. numpy itself evaluates it with
// ast.literal_eval, so any spacing, key order, quote style and trailing commas are legal; the keys
// must be exactly descr / fortran_order / shape, and shape must be a tuple of non-negative ints.
class HeaderParser {
 public:
  HeaderParser(std::string_view text, const std::string& source) : s_(text), source_(source) {}

  NpyHeader parse() {
    NpyHeader h;
    bool have_descr = false, have_order = false, have_shape = false;
    skip_ws();
    expect('{');
    for (;;) {
      skip_ws();
      if (consume('}')) break;  // empty dict or trailing comma
      const std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      if (key == "descr") {
        if (have_descr) fail("duplicate key 'descr'");
        if (peek() == '[' || peek() == '{')
          HS_THROW(Unsupported, source_ << ": structured numpy dtypes are not supported");
        h.dtype = parse_string();
        have_descr = true;
      } else if (key == "fortran_order") {
        if (have_order) fail("duplicate key 'fortran_order'");
        h.fortran_order = parse_bool();
        have_order = true;
      } else if (key == "shape") {
        if (have_shape) fail("duplicate key 'shape'");
        h.shape = parse_shape();
        have_shape = true;
      } else {
        fail("unexpected key '" + key + "'");
      }
      skip_ws();
      if (consume(',')) continue;
      expect('}');
      break;
    }
    skip_ws();  // numpy pads with spaces and a final '\n'
    if (pos_ != s_.size()) fail("unexpected text after the dict");
    if (!(have_descr && have_order && have_shape))
      fail("the keys 'descr', 'fortran_order' and 'shape' are all required");
    h.info = parse_dtype(h.dtype);
    h.count = 1;
    for (std::size_t d : h.shape) h.count = checked_mul(h.count, d, source_);
    h.data_bytes = checked_mul(h.count, h.info.item_size, source_);
    return h;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    constexpr std::size_t kExcerpt = 200;
    HS_THROW(Io, source_ << ": malformed npy header (" << what << ") at offset " << pos_ << " in \""
                         << s_.substr(0, kExcerpt) << (s_.size() > kExcerpt ? "..." : "") << "\"");
  }
  char peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  void skip_ws() {
    while (pos_ < s_.size() && is_py_space(s_[pos_])) ++pos_;
  }
  bool consume(char c) {
    if (pos_ >= s_.size() || s_[pos_] != c) return false;
    ++pos_;
    return true;
  }
  void expect(char c) {
    if (!consume(c)) fail(std::string("expected '") + c + "'");
  }

  std::string parse_string() {
    const char quote = peek();
    if (quote != '\'' && quote != '"') fail("expected a string");
    ++pos_;
    std::string out;
    for (;;) {
      if (pos_ >= s_.size()) fail("unterminated string");
      char c = s_[pos_++];
      if (c == quote) break;
      if (c == '\\') {  // descr strings never need escapes; keep the escaped character literally
        if (pos_ >= s_.size()) fail("unterminated string");
        c = s_[pos_++];
      }
      out += c;
    }
    return out;
  }

  bool parse_bool() {
    const std::size_t begin = pos_;
    while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_'))
      ++pos_;
    const std::string_view word = s_.substr(begin, pos_ - begin);
    if (word == "True") return true;
    if (word == "False") return false;
    pos_ = begin;
    fail("fortran_order must be True or False");
  }

  std::size_t parse_uint() {
    const std::size_t begin = pos_;
    std::size_t v = 0;
    while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') {
      const auto digit = static_cast<std::size_t>(s_[pos_] - '0');
      if (v > (std::numeric_limits<std::size_t>::max() - digit) / 10) fail("dimension overflows");
      v = v * 10 + digit;
      ++pos_;
    }
    if (pos_ == begin) fail("expected a non-negative integer dimension");
    if (peek() == 'L' || peek() == 'l') ++pos_;  // Python 2 long literal (numpy strips it too)
    return v;
  }

  std::vector<std::size_t> parse_shape() {
    expect('(');
    std::vector<std::size_t> dims;
    bool comma_after_last = false;
    for (;;) {
      skip_ws();
      if (consume(')')) break;
      if (!dims.empty() && !comma_after_last) fail("expected ',' between shape entries");
      dims.push_back(parse_uint());
      skip_ws();
      comma_after_last = consume(',');
    }
    // "(n)" is a parenthesised int in Python, not a tuple; numpy rejects it, and so do we.
    if (dims.size() == 1 && !comma_after_last) fail("shape '(n)' is not a tuple, expected '(n,)'");
    return dims;
  }

  std::string_view s_;
  const std::string& source_;
  std::size_t pos_ = 0;
};

void check_payload_length(const ByteSource& src, const NpyHeader& h) {
  if (src.remaining() != h.data_bytes)
    HS_THROW(Io, src.name() << ": npy payload is " << src.remaining() << " bytes but dtype '"
                            << h.dtype << "' and shape " << shape_repr(h.shape) << " imply "
                            << h.data_bytes << (src.remaining() < h.data_bytes ? " (truncated)"
                                                                                : " (trailing data)"));
}

// ----------------------------------------------------------------------- element conversion

template <class S>
struct Tag {
  using type = S;
};

// Calls f(Tag<S>{}) with the C++ type S that holds one element of `info`; returns false for
// non-numeric kinds and for sizes without a C++ counterpart (float16, long double, ...).
template <class F>
bool visit_numeric(const DtypeInfo& info, F&& f) {
  switch (info.kind) {
    case 'b':
      if (info.item_size == 1) return f(Tag<bool>{});
      break;
    case 'i':
      switch (info.item_size) {
        case 1: return f(Tag<std::int8_t>{});
        case 2: return f(Tag<std::int16_t>{});
        case 4: return f(Tag<std::int32_t>{});
        case 8: return f(Tag<std::int64_t>{});
      }
      break;
    case 'u':
      switch (info.item_size) {
        case 1: return f(Tag<std::uint8_t>{});
        case 2: return f(Tag<std::uint16_t>{});
        case 4: return f(Tag<std::uint32_t>{});
        case 8: return f(Tag<std::uint64_t>{});
      }
      break;
    case 'f':
      if (info.item_size == 4) return f(Tag<float>{});
      if (info.item_size == 8) return f(Tag<double>{});
      break;
    case 'c':
      if (info.item_size == 8) return f(Tag<std::complex<float>>{});
      if (info.item_size == 16) return f(Tag<std::complex<double>>{});
      break;
  }
  return false;
}

template <class S>
S load_element(const std::byte* p, bool big_endian) {
  if constexpr (std::is_same_v<S, bool>) {
    return std::to_integer<unsigned>(*p) != 0;  // numpy bool: any nonzero byte is True
  } else if constexpr (is_complex_v<S>) {
    using R = typename S::value_type;  // real and imaginary parts are swapped separately
    return S(load_element<R>(p, big_endian), load_element<R>(p + sizeof(R), big_endian));
  } else {
    std::array<std::byte, sizeof(S)> b;
    std::memcpy(b.data(), p, sizeof(S));
    if (big_endian) std::reverse(b.begin(), b.end());
    S v;
    std::memcpy(&v, b.data(), sizeof(S));
    return v;
  }
}

enum class Cast { Safe, Checked, Rejected };

// numpy's 'safe' casting table restricted to our targets, plus the documented value-checked
// narrowings (Checked: allowed when every element converts exactly).
template <class S, class T>
constexpr Cast cast_rule() {
  if constexpr (std::is_same_v<S, bool>) {
    return Cast::Safe;
  } else if constexpr (std::is_integral_v<T>) {
    if constexpr (!std::is_integral_v<S>) {
      return Cast::Rejected;  // float/complex -> int would silently truncate
    } else if constexpr (std::is_signed_v<S>) {
      return sizeof(S) <= sizeof(T) ? Cast::Safe : Cast::Checked;
    } else {
      return sizeof(S) < sizeof(T) ? Cast::Safe : Cast::Checked;  // unsigned needs a wider signed
    }
  } else if constexpr (std::is_integral_v<S>) {
    // numpy: every integer is safe to float64; only integers of <= 16 bits are safe to float32.
    return sizeof(real_t<T>) >= 8 || sizeof(S) <= 2 ? Cast::Safe : Cast::Rejected;
  } else if constexpr (sizeof(real_t<S>) > sizeof(real_t<T>)) {
    return Cast::Rejected;  // f8 -> f4, c16 -> float: loses precision
  } else if constexpr (is_complex_v<S> && !is_complex_v<T>) {
    return Cast::Checked;  // complex -> real: every imaginary part must be exactly 0
  } else {
    return Cast::Safe;
  }
}

template <class T, class S>
bool fits_exactly(const S& s) {
  if constexpr (is_complex_v<S>) {
    return s.imag() == 0;
  } else if constexpr (std::is_signed_v<S>) {
    return s >= std::numeric_limits<T>::min() && s <= std::numeric_limits<T>::max();
  } else {
    return s <= static_cast<std::make_unsigned_t<T>>(std::numeric_limits<T>::max());
  }
}

template <class T, class S>
T convert_value(const S& s) {
  if constexpr (std::is_same_v<S, bool>) {
    return T(s ? 1 : 0);
  } else if constexpr (is_complex_v<T>) {
    using TR = real_t<T>;
    if constexpr (is_complex_v<S>) {
      return T(static_cast<TR>(s.real()), static_cast<TR>(s.imag()));
    } else {
      return T(static_cast<TR>(s), TR(0));
    }
  } else if constexpr (is_complex_v<S>) {
    return static_cast<T>(s.real());
  } else {
    return static_cast<T>(s);
  }
}

template <class T>
inline constexpr bool kIsElementType =
    std::is_same_v<T, double> || std::is_same_v<T, cplx> || std::is_same_v<T, float> ||
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>;

template <class T>
const char* type_name() {
  static_assert(kIsElementType<T>);
  if constexpr (std::is_same_v<T, double>) {
    return "float64";
  } else if constexpr (std::is_same_v<T, cplx>) {
    return "complex128";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float32";
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return "int32";
  } else {
    return "int64";
  }
}

// descr that numpy writes for T on a little-endian host.
template <class T>
const char* native_descr() {
  static_assert(kIsElementType<T>);
  if constexpr (std::is_same_v<T, double>) {
    return "<f8";
  } else if constexpr (std::is_same_v<T, cplx>) {
    return "<c16";
  } else if constexpr (std::is_same_v<T, float>) {
    return "<f4";
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return "<i4";
  } else {
    return "<i8";
  }
}

template <class T>
bool stores_native(const DtypeInfo& info) {
  const char kind = is_complex_v<T> ? 'c' : std::is_integral_v<T> ? 'i' : 'f';
  return info.kind == kind && info.item_size == sizeof(T) && !info.big_endian();
}

// ------------------------------------------------------------------------------- writing

// Byte-for-byte numpy's _write_array_header: sorted-key dict repr, growth-axis padding, then
// spaces so that magic + version + length + header (incl. '\n') is a multiple of ARRAY_ALIGN.
std::string npy_header(const std::string& descr, bool fortran_order,
                       const std::vector<std::size_t>& shape) {
  std::string dict = "{'descr': '" + descr + "', 'fortran_order': " +
                     (fortran_order ? "True" : "False") + ", 'shape': " + shape_repr(shape) + ", }";
  // Room for the growth axis (last for Fortran order, first for C order) to reach 21 digits, so
  // numpy can rewrite the header in place when appending; absent for 0-d arrays.
  if (!shape.empty()) {
    const std::size_t digits = std::to_string(shape[fortran_order ? shape.size() - 1 : 0]).size();
    dict.append(kGrowthAxisMaxDigits > digits ? kGrowthAxisMaxDigits - digits : 0, ' ');
  }
  const std::size_t hlen = dict.size() + 1;  // + '\n'
  for (const unsigned major : {1u, 2u}) {    // 1.0: u16 length; 2.0: u32 length
    const std::size_t len_bytes = major == 1 ? 2 : 4;
    const std::size_t pad = kArrayAlign - (kMagic.size() + 2 + len_bytes + hlen) % kArrayAlign;
    const std::uint64_t total = hlen + pad;
    if (major == 1 && total > 0xFFFF) continue;
    if (total > 0xFFFFFFFFu) break;
    std::string out(kMagic);
    out += static_cast<char>(major);
    out += '\0';
    for (std::size_t b = 0; b < len_bytes; ++b) out += static_cast<char>((total >> (8 * b)) & 0xFF);
    out += dict;
    out.append(pad, ' ');
    out += '\n';
    return out;
  }
  HS_THROW(InvalidInput, "npy header of " << hlen << " bytes is too long");
}

void write_npy_bytes(const std::filesystem::path& path, const std::string& descr,
                     bool fortran_order, const std::vector<std::size_t>& shape,
                     const std::byte* data, std::size_t nbytes) {
  const std::string header = npy_header(descr, fortran_order, shape);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) HS_THROW(Io, "cannot open '" << path.string() << "' for writing");
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  constexpr std::size_t kMaxChunk = std::size_t{1} << 30;  // ostream::write takes a signed count
  for (std::size_t done = 0; done < nbytes && out; done += kMaxChunk) {
    const std::size_t len = std::min(kMaxChunk, nbytes - done);
    out.write(reinterpret_cast<const char*>(data + done), static_cast<std::streamsize>(len));
  }
  out.close();
  if (!out) HS_THROW(Io, "writing '" << path.string() << "' failed");
}

}  // namespace

// ---------------------------------------------------------------------- npy_format.hpp API

DtypeInfo parse_dtype(const std::string& descr) {
  DtypeInfo info;
  std::size_t i = 0;
  if (!descr.empty() && is_one_of(descr[0], "<>|=")) {
    info.byte_order = descr[0];
    i = 1;
  } else {
    info.byte_order = '=';  // np.dtype('f8'): native order
  }
  if (i >= descr.size()) HS_THROW(Io, "malformed numpy dtype '" << descr << "'");
  char kind = descr[i++];
  if (kind == 'O') HS_THROW(Unsupported, "numpy object dtype '" << descr << "' is not supported");
  if (kind == 'a') kind = 'S';  // deprecated alias of bytes
  if (!is_one_of(kind, "biufcSUVMm")) HS_THROW(Io, "unknown numpy dtype '" << descr << "'");
  std::size_t n = 0;
  const std::size_t digits_begin = i;
  constexpr std::size_t kMaxItem = std::size_t{1} << 40;
  for (; i < descr.size() && descr[i] >= '0' && descr[i] <= '9'; ++i) {
    n = n * 10 + static_cast<std::size_t>(descr[i] - '0');
    if (n > kMaxItem) HS_THROW(Io, "numpy dtype '" << descr << "' has an absurd item size");
  }
  if (i == digits_begin) HS_THROW(Io, "numpy dtype '" << descr << "' has no item size");
  if ((kind == 'M' || kind == 'm') && i < descr.size()) {  // datetime unit, e.g. "<M8[ns]"
    const bool ok = descr[i] == '[' && descr.back() == ']' &&
                    std::all_of(descr.begin() + static_cast<std::ptrdiff_t>(i) + 1, descr.end() - 1,
                                [](char c) { return std::isalnum(static_cast<unsigned char>(c)); });
    if (!ok) HS_THROW(Io, "malformed numpy datetime dtype '" << descr << "'");
    i = descr.size();
  }
  if (i != descr.size()) HS_THROW(Io, "malformed numpy dtype '" << descr << "'");
  if (n == 0) HS_THROW(Unsupported, "zero-size numpy dtype '" << descr << "' is not supported");
  info.kind = kind;
  info.item_size = kind == 'U' ? 4 * n : n;  // 'U' counts UCS-4 characters
  return info;
}

NpyHeader read_npy_header(ByteSource& src) {
  std::array<std::byte, 8> pre{};
  if (src.remaining() < pre.size()) HS_THROW(Io, src.name() << ": not an npy file (too short)");
  src.read(pre.data(), pre.size());
  if (std::memcmp(pre.data(), kMagic.data(), kMagic.size()) != 0)
    HS_THROW(Io, src.name() << ": not an npy file (bad magic)");
  const auto major = std::to_integer<unsigned>(pre[6]);
  const auto minor = std::to_integer<unsigned>(pre[7]);
  if (!(major >= 1 && major <= 3 && minor == 0))
    HS_THROW(Io, src.name() << ": unsupported npy format version " << major << "." << minor);
  std::size_t hlen = 0;
  if (major == 1) {
    std::array<std::byte, 2> len{};
    src.read(len.data(), len.size());
    hlen = load_le<std::uint16_t>(len.data());
  } else {
    std::array<std::byte, 4> len{};
    src.read(len.data(), len.size());
    hlen = load_le<std::uint32_t>(len.data());
  }
  if (hlen > kMaxHeaderBytes)
    HS_THROW(Io, src.name() << ": npy header of " << hlen << " bytes exceeds the limit");
  std::string text(hlen, '\0');
  src.read(reinterpret_cast<std::byte*>(text.data()), hlen);
  // Versions 1.0/2.0 are latin-1 and 3.0 is UTF-8; the structure is ASCII either way and descr
  // bytes are kept verbatim.
  return HeaderParser(text, src.name()).parse();
}

NpyArray read_npy_payload(ByteSource& src, NpyHeader header) {
  check_payload_length(src, header);
  NpyArray a;
  a.shape = std::move(header.shape);
  a.dtype = std::move(header.dtype);
  a.fortran_order = header.fortran_order;
  a.data.resize(header.data_bytes);
  src.read(a.data.data(), a.data.size());
  src.finish();
  return a;
}

namespace {

constexpr std::size_t kChunkBytes = std::size_t{1} << 20;  // payload bytes per streamed chunk

std::size_t chunk_elements(const NpyHeader& h) {
  return std::max<std::size_t>(1, kChunkBytes / h.info.item_size);
}

// Reads the payload in chunks of chunk_elements(h) elements converted to T: dst(first, count)
// returns where elements [first, first + count) go, done(first, count) runs once they are there.
// An hs::Error thrown by the conversion or by `done` is deferred until the rest of the payload has
// been read and src.finish() has checked its integrity, so a corrupt member still reports Io, as
// when the payload was read in one piece; after such an error the remaining chunks are only read.
template <class T, class Dst, class Done>
void stream_payload(ByteSource& src, const NpyHeader& h, Dst&& dst, Done&& done) {
  check_payload_length(src, h);
  const bool native = stores_native<T>(h.info);
  const std::size_t per_chunk = chunk_elements(h);
  std::vector<std::byte> raw(native ? 0 : std::min(per_chunk, h.count) * h.info.item_size);
  std::exception_ptr deferred;
  const auto convert = [&](std::size_t first, std::size_t count, T* out) {
    try {
      convert_into<T>(raw.data(), count, h.info, h.dtype, out, first);
    } catch (const Error& e) {
      deferred = std::make_exception_ptr(Error(e.code(), src.name() + ": " + e.what()));
    }
  };
  if (!native && h.count == 0) convert(0, 0, nullptr);  // a rejected cast fails for empty arrays too
  for (std::size_t first = 0; first < h.count; first += per_chunk) {
    const std::size_t count = std::min(per_chunk, h.count - first);
    T* out = dst(first, count);
    if (native) {
      src.read(reinterpret_cast<std::byte*>(out), count * sizeof(T));
    } else {
      src.read(raw.data(), count * h.info.item_size);
      if (!deferred) convert(first, count, out);
    }
    if (deferred) continue;
    try {
      done(first, count);
    } catch (const Error&) {
      deferred = std::current_exception();
    }
  }
  src.finish();
  if (deferred) std::rethrow_exception(deferred);
}

}  // namespace

template <class T>
std::vector<T> read_npy_payload_as(ByteSource& src, const NpyHeader& header) {
  // Before allocating: a corrupt header may claim any size, and only the member's length says
  // whether the payload backs it (Io instead of an allocation of the claimed size).
  check_payload_length(src, header);
  std::vector<T> v(header.count);
  if (stores_native<T>(header.info)) {
    src.read(reinterpret_cast<std::byte*>(v.data()), header.data_bytes);
    src.finish();
    return v;
  }
  // Converted chunk by chunk straight into the result: no raw copy of the whole payload.
  stream_payload<T>(
      src, header, [&](std::size_t first, std::size_t) { return v.data() + first; },
      [](std::size_t, std::size_t) {});
  return v;
}

template <class T>
void read_npy_payload_chunks(ByteSource& src, const NpyHeader& header, const ChunkSink<T>& sink) {
  check_payload_length(src, header);  // also in stream_payload; here before the buffer exists
  std::vector<T> buf(std::min(chunk_elements(header), header.count));
  stream_payload<T>(
      src, header, [&](std::size_t, std::size_t) { return buf.data(); },
      [&](std::size_t first, std::size_t count) { sink(buf.data(), first, count); });
}

template <class T>
void convert_into(const std::byte* data, std::size_t count, const DtypeInfo& info,
                  const std::string& dtype, T* out, std::size_t first_index) {
  const bool handled = visit_numeric(info, [&](auto tag) {
    using S = typename decltype(tag)::type;
    constexpr Cast rule = cast_rule<S, T>();
    if constexpr (rule == Cast::Rejected) {
      return false;
    } else {
      const bool big_endian = info.big_endian();
      for (std::size_t k = 0; k < count; ++k) {
        const S s = load_element<S>(data + k * sizeof(S), big_endian);
        if constexpr (rule == Cast::Checked) {
          if (!fits_exactly<T>(s))
            HS_THROW(InvalidInput, "cannot convert numpy dtype '"
                                       << dtype << "' to " << type_name<T>() << ": element "
                                       << first_index + k << " does not fit exactly");
        }
        out[k] = convert_value<T>(s);
      }
      return true;
    }
  });
  if (!handled)
    HS_THROW(InvalidInput, "cannot convert numpy dtype '" << dtype << "' to " << type_name<T>()
                                                           << " (not a safe numpy cast)");
}

template <class T>
std::vector<T> convert_elements(const std::byte* data, std::size_t count, const DtypeInfo& info,
                                const std::string& dtype) {
  std::vector<T> out(count);
  convert_into<T>(data, count, info, dtype, out.data(), 0);
  return out;
}

}  // namespace io

// --------------------------------------------------------------------------- public API

std::size_t NpyArray::size() const {
  std::size_t n = 1;
  for (std::size_t d : shape) n = io::checked_mul(n, d, "NpyArray");
  return n;
}

std::size_t NpyArray::item_size() const { return io::parse_dtype(dtype).item_size; }

NpyArray read_npy(const std::filesystem::path& path) {
  io::FileSource src(path);
  io::NpyHeader header = io::read_npy_header(src);
  return io::read_npy_payload(src, std::move(header));
}

NpyArray parse_npy(const std::byte* bytes, std::size_t nbytes) {
  io::MemorySource src(bytes, nbytes, "<npy bytes>");
  io::NpyHeader header = io::read_npy_header(src);
  return io::read_npy_payload(src, std::move(header));
}

void write_npy(const std::filesystem::path& path, const NpyArray& array) {
  const std::size_t expected = io::checked_mul(array.size(), array.item_size(), "write_npy");
  if (array.data.size() != expected)
    HS_THROW(InvalidInput, "write_npy: payload has " << array.data.size() << " bytes, dtype '"
                                                     << array.dtype << "' and shape "
                                                     << io::shape_repr(array.shape) << " need "
                                                     << expected);
  io::write_npy_bytes(path, array.dtype, array.fortran_order, array.shape, array.data.data(),
                      array.data.size());
}

template <class T>
void write_npy_vector(const std::filesystem::path& path, const std::vector<T>& v) {
  io::write_npy_bytes(path, io::native_descr<T>(), false, {v.size()},
                      reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(T));
}

template <class T>
void write_npy_matrix(const std::filesystem::path& path, const DenseMatrix<T>& m) {
  const auto rows = static_cast<std::size_t>(m.rows());
  const auto cols = static_cast<std::size_t>(m.cols());
  // numpy records fortran_order only for arrays that are F- but not C-contiguous; with a single
  // row or column (or no elements) both layouts are the same byte sequence.
  const bool fortran_order = rows > 1 && cols > 1;
  io::write_npy_bytes(path, io::native_descr<T>(), fortran_order, {rows, cols},
                      reinterpret_cast<const std::byte*>(m.data()), rows * cols * sizeof(T));
}

template <class T>
std::vector<T> npy_to_vector(const NpyArray& a) {
  const io::DtypeInfo info = io::parse_dtype(a.dtype);
  const std::size_t count = a.size();
  if (a.data.size() != io::checked_mul(count, info.item_size, "npy_to_vector"))
    HS_THROW(InvalidInput, "npy_to_vector: payload has " << a.data.size() << " bytes, dtype '"
                                                         << a.dtype << "' and shape "
                                                         << io::shape_repr(a.shape) << " need "
                                                         << count * info.item_size);
  return io::convert_elements<T>(a.data.data(), count, info, a.dtype);
}

// Element templates cover the documented wider set {double, cplx, float, int32_t, int64_t}: the
// public scalars via HS_FOR_EACH_SCALAR plus the three extra element types.
#define HS_IO_INSTANTIATE_ELEMENT(T)                                                             \
  template void write_npy_vector<T>(const std::filesystem::path&, const std::vector<T>&);       \
  template std::vector<T> npy_to_vector<T>(const NpyArray&);                                     \
  template std::vector<T> io::read_npy_payload_as<T>(io::ByteSource&, const io::NpyHeader&);    \
  template void io::read_npy_payload_chunks<T>(io::ByteSource&, const io::NpyHeader&,            \
                                               const io::ChunkSink<T>&);                         \
  template std::vector<T> io::convert_elements<T>(const std::byte*, std::size_t,                 \
                                                  const io::DtypeInfo&, const std::string&);     \
  template void io::convert_into<T>(const std::byte*, std::size_t, const io::DtypeInfo&,         \
                                    const std::string&, T*, std::size_t);
#define HS_IO_INSTANTIATE_MATRIX(T) \
  template void write_npy_matrix<T>(const std::filesystem::path&, const DenseMatrix<T>&);

HS_FOR_EACH_SCALAR(HS_IO_INSTANTIATE_ELEMENT)
HS_IO_INSTANTIATE_ELEMENT(float)
HS_IO_INSTANTIATE_ELEMENT(std::int32_t)
HS_IO_INSTANTIATE_ELEMENT(std::int64_t)
HS_FOR_EACH_SCALAR(HS_IO_INSTANTIATE_MATRIX)
#undef HS_IO_INSTANTIATE_ELEMENT
#undef HS_IO_INSTANTIATE_MATRIX

}  // namespace hs
