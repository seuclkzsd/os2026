// Unit tests for src/io, part 3: the low-memory assembly paths of load_matrix (module memio).
// scipy csc / coo files are scattered straight into CSR slots with the values streamed from the
// archive (io::NpzFile::read_chunks), and MatrixMarket coordinate files are read in two passes.
// Every path must give exactly what the previous assembly (io::csr_from_triplets on the stored
// entries in file order, i.e. duplicates summed in file order) gives. The archives are written
// here, stored or deflated, with members larger than one streamed chunk (1 MiB).
#include <gtest/gtest.h>
#include <zlib.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"
#include "io/csr_assembly.hpp"
#include "io/npz.hpp"
#include "support/fixtures.hpp"
#include "test_io_util.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::io_test::error_of;
using hs::io_test::read_bytes;
using hs::io_test::write_bytes;
using hs::io_test::write_text;

// ------------------------------------------------------------------------ npz writing

template <class V>
const char* descr_of() {
  if constexpr (std::is_same_v<V, double>) {
    return "<f8";
  } else if constexpr (std::is_same_v<V, cplx>) {
    return "<c16";
  } else if constexpr (std::is_same_v<V, float>) {
    return "<f4";
  } else if constexpr (std::is_same_v<V, std::complex<float>>) {
    return "<c8";
  } else if constexpr (std::is_same_v<V, std::int32_t>) {
    return "<i4";
  } else {
    static_assert(std::is_same_v<V, std::int64_t>);
    return "<i8";
  }
}

template <class V>
hs::NpyArray npy_of(const std::vector<V>& v, std::vector<std::size_t> shape = {},
                    bool fortran_order = false) {
  hs::NpyArray a;
  a.shape = shape.empty() ? std::vector<std::size_t>{v.size()} : std::move(shape);
  a.dtype = descr_of<V>();
  a.fortran_order = fortran_order;
  a.data.resize(v.size() * sizeof(V));
  if (!v.empty()) std::memcpy(a.data.data(), v.data(), a.data.size());
  return a;
}

hs::NpyArray format_array(const std::string& fmt) {
  hs::NpyArray a;
  a.dtype = "|S3";
  for (const char c : fmt) a.data.push_back(static_cast<std::byte>(c));
  return a;
}

template <class U>
void put_le(std::vector<char>& out, U v) {
  for (std::size_t b = 0; b < sizeof(U); ++b) out.push_back(static_cast<char>((v >> (8 * b)) & 0xFF));
}

using Members = std::vector<std::pair<std::string, hs::NpyArray>>;
using RawMembers = std::vector<std::pair<std::string, std::vector<char>>>;  ///< key, .npy bytes

// The .npy bytes hs::write_npy writes for `array`.
std::vector<char> npy_bytes(const hs::NpyArray& array, const std::filesystem::path& scratch) {
  hs::write_npy(scratch, array);
  std::vector<char> content = read_bytes(scratch);
  std::filesystem::remove(scratch);
  return content;
}

// np.savez(_compressed) layout: classic zip records, one ".npy" member per array.
void write_npz_raw(const std::filesystem::path& path, const RawMembers& members, bool compressed) {
  std::vector<char> zip, central;
  for (const auto& [key, content] : members) {
    std::vector<char> packed = content;
    if (compressed) {
      z_stream z{};
      EXPECT_EQ(deflateInit2(&z, Z_BEST_SPEED, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY), Z_OK);
      packed.resize(deflateBound(&z, static_cast<uLong>(content.size())));
      z.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(content.data()));
      z.avail_in = static_cast<uInt>(content.size());
      z.next_out = reinterpret_cast<Bytef*>(packed.data());
      z.avail_out = static_cast<uInt>(packed.size());
      EXPECT_EQ(deflate(&z, Z_FINISH), Z_STREAM_END);
      packed.resize(z.total_out);
      deflateEnd(&z);
    }
    const auto crc = static_cast<std::uint32_t>(
        crc32(0, reinterpret_cast<const Bytef*>(content.data()), static_cast<uInt>(content.size())));
    const std::string name = key + ".npy";
    const auto common = [&](std::vector<char>& out) {
      put_le<std::uint16_t>(out, 0);                      // flags
      put_le<std::uint16_t>(out, compressed ? 8 : 0);     // method
      put_le<std::uint32_t>(out, 0);                      // time, date
      put_le(out, crc);
      put_le(out, static_cast<std::uint32_t>(packed.size()));
      put_le(out, static_cast<std::uint32_t>(content.size()));
      put_le(out, static_cast<std::uint16_t>(name.size()));
      put_le<std::uint16_t>(out, 0);  // extra field length
    };
    const auto offset = static_cast<std::uint32_t>(zip.size());
    put_le<std::uint32_t>(zip, 0x04034b50);
    put_le<std::uint16_t>(zip, 20);
    common(zip);
    zip.insert(zip.end(), name.begin(), name.end());
    zip.insert(zip.end(), packed.begin(), packed.end());
    put_le<std::uint32_t>(central, 0x02014b50);
    put_le<std::uint16_t>(central, 20);
    put_le<std::uint16_t>(central, 20);
    common(central);
    put_le<std::uint16_t>(central, 0);  // comment length
    put_le<std::uint16_t>(central, 0);  // disk
    put_le<std::uint16_t>(central, 0);  // internal attributes
    put_le<std::uint32_t>(central, 0);  // external attributes
    put_le(central, offset);
    central.insert(central.end(), name.begin(), name.end());
  }
  const auto cd_offset = static_cast<std::uint32_t>(zip.size());
  zip.insert(zip.end(), central.begin(), central.end());
  put_le<std::uint32_t>(zip, 0x06054b50);
  put_le<std::uint32_t>(zip, 0);
  put_le(zip, static_cast<std::uint16_t>(members.size()));
  put_le(zip, static_cast<std::uint16_t>(members.size()));
  put_le(zip, static_cast<std::uint32_t>(central.size()));
  put_le(zip, cd_offset);
  put_le<std::uint16_t>(zip, 0);
  write_bytes(path, zip);
}

void write_npz(const std::filesystem::path& path, const Members& members, bool compressed) {
  RawMembers raw;
  for (const auto& [key, array] : members) raw.emplace_back(key, npy_bytes(array, path.string() + ".member.npy"));
  write_npz_raw(path, raw, compressed);
}

// The .npy bytes of `array` with the header's shape replaced by `shape` (e.g. "(1099511627776,)")
// and the payload kept: a header that claims more (or other) elements than the payload holds.
// The header keeps its length (numpy's growth-axis padding absorbs the longer shape).
std::vector<char> with_claimed_shape(const hs::NpyArray& array, const std::string& shape,
                                     const std::filesystem::path& scratch) {
  std::vector<char> bytes = npy_bytes(array, scratch);
  std::string text(bytes.begin(), bytes.end());
  const std::size_t begin = text.find("'shape': ") + 9;
  const std::size_t end = text.find(')', begin) + 1;
  const std::size_t newline = text.find('\n');
  std::size_t pad = 0;  // trailing spaces before the '\n'
  while (text[newline - 1 - pad] == ' ') ++pad;
  EXPECT_LE(shape.size(), end - begin + pad) << "no room for the claimed shape";
  std::string header = text.substr(0, newline);
  header.replace(begin, end - begin, shape);
  header.resize(newline, ' ');  // shorter or longer shape: re-pad to the same length
  std::copy(header.begin(), header.end(), bytes.begin());
  return bytes;
}

// ---------------------------------------------------------------- matrices and entries

template <class T>
T scalar(double re, double im) {
  if constexpr (hs::is_complex_v<T>) {
    return T(re, im);
  } else {
    return re;
  }
}

// Coordinate entries in file order.
template <class T>
struct Entries {
  index_t n = 0;
  std::vector<std::int32_t> rows, cols;
  std::vector<T> vals;
  void add(index_t i, index_t j, T v) {
    rows.push_back(static_cast<std::int32_t>(i));
    cols.push_back(static_cast<std::int32_t>(j));
    vals.push_back(v);
  }
};

// The entries of a matrix in row-major or column-major order.
template <class T>
Entries<T> entries_of(const CsrMatrix<T>& a, bool column_major) {
  Entries<T> e;
  e.n = a.nrows;
  if (!column_major) {
    for (index_t i = 0; i < a.nrows; ++i)
      for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) e.add(i, a.colind[p], a.values[p]);
  } else {
    const CsrMatrix<T> t = hs::conj_transpose(a);  // row j of t: column j of a, conjugated
    for (index_t j = 0; j < t.nrows; ++j)
      for (index_t p = t.rowptr[j]; p < t.rowptr[j + 1]; ++p) e.add(t.colind[p], j, hs::conj_if(t.values[p]));
  }
  return e;
}

// Shuffles the entries deterministically, splits every 11th into the duplicates v, 1e16, -1e16
// (whose sum is v rounded to a multiple of 2 in this order but v in others, so the loader must sum
// in file order) and adds explicit zeros.
template <class T>
Entries<T> scramble(const Entries<T>& in, std::uint64_t seed) {
  std::vector<std::size_t> perm(in.vals.size());
  for (std::size_t k = 0; k < perm.size(); ++k) perm[k] = k;
  std::mt19937_64 g(seed);
  std::shuffle(perm.begin(), perm.end(), g);
  Entries<T> out;
  out.n = in.n;
  for (std::size_t k = 0; k < perm.size(); ++k) {
    const std::size_t p = perm[k];
    if (k % 11 == 0) {  // v = (v + 1e16) - 1e16 in file order only
      out.add(in.rows[p], in.cols[p], in.vals[p]);
      out.add(in.rows[p], in.cols[p], T(1e16));
      out.add(in.rows[p], in.cols[p], T(-1e16));
    } else {
      out.add(in.rows[p], in.cols[p], in.vals[p]);
    }
    if (k % 17 == 0) out.add(in.rows[p], (in.cols[p] + 1) % in.n, T(0));  // explicit zero
  }
  return out;
}

// The previous assembly: counting sort in file order, then canonicalize_csr.
template <class T>
CsrMatrix<T> reference_csr(const Entries<T>& e, bool drop_zeros, hs::io::AssemblyStats* stats = nullptr) {
  hs::io::AssemblyStats s;
  CsrMatrix<T> m = hs::io::csr_from_triplets<T>(e.n, e.n, e.rows.data(), e.cols.data(), e.vals.data(),
                                                e.vals.size(), drop_zeros, s);
  if (stats) *stats = s;
  return m;
}

template <class T>
Members coo_members(const Entries<T>& e) {
  return {{"format", format_array("coo")},
          {"shape", npy_of(std::vector<std::int64_t>{e.n, e.n})},
          {"row", npy_of(e.rows)},
          {"col", npy_of(e.cols)},
          {"data", npy_of(e.vals)}};
}

// CSC arrays of the entries: stable counting sort by column (file order kept within a column).
template <class T>
Members csc_members(const Entries<T>& e, std::size_t unused_tail = 0) {
  std::vector<std::int64_t> colptr(static_cast<std::size_t>(e.n) + 1, 0);
  for (const std::int32_t c : e.cols) ++colptr[c + 1];
  for (index_t j = 0; j < e.n; ++j) colptr[j + 1] += colptr[j];
  std::vector<std::int64_t> next(colptr.begin(), colptr.end() - 1);
  std::vector<std::int32_t> rows(e.vals.size() + unused_tail, -5);  // garbage in the unused tail
  std::vector<T> vals(e.vals.size() + unused_tail, T(7));
  for (std::size_t k = 0; k < e.vals.size(); ++k) {
    const auto dst = next[e.cols[k]]++;
    rows[dst] = e.rows[k];
    vals[dst] = e.vals[k];
  }
  std::vector<std::int32_t> colptr32(colptr.begin(), colptr.end());
  return {{"format", format_array("csc")},
          {"shape", npy_of(std::vector<std::int64_t>{e.n, e.n})},
          {"indptr", npy_of(colptr32)},
          {"indices", npy_of(rows)},
          {"data", npy_of(vals)}};
}

// The CSC entries in column-major file order (what the csc file means, for reference_csr).
template <class T>
Entries<T> column_order(const Entries<T>& e) {
  std::vector<std::size_t> order(e.vals.size());
  for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return e.cols[x] < e.cols[y]; });
  Entries<T> out;
  out.n = e.n;
  for (const std::size_t k : order) out.add(e.rows[k], e.cols[k], e.vals[k]);
  return out;
}

// For loads that may come back real although the file is complex (all imaginary parts 0).
template <class T>
void expect_same_values(const hs::LoadedMatrix& got, const CsrMatrix<T>& want) {
  std::visit(
      [&](const auto& m) {
        EXPECT_EQ(m.rowptr, want.rowptr);
        EXPECT_EQ(m.colind, want.colind);
        ASSERT_EQ(m.values.size(), want.values.size());
        for (std::size_t k = 0; k < m.values.size(); ++k) {
          const auto& v = m.values[k];
          if constexpr (hs::is_complex_v<std::decay_t<decltype(v)>> && !hs::is_complex_v<T>) {
            ADD_FAILURE() << "complex result, real expected";
          } else {
            EXPECT_EQ(T(v), want.values[k]);
          }
        }
      },
      got.matrix);
}

template <class T>
void expect_same(const hs::LoadedMatrix& got, const CsrMatrix<T>& want) {
  ASSERT_TRUE(std::holds_alternative<CsrMatrix<T>>(got.matrix));
  const auto& m = std::get<CsrMatrix<T>>(got.matrix);
  EXPECT_NO_THROW(m.validate());
  EXPECT_EQ(m.rowptr, want.rowptr);
  EXPECT_EQ(m.colind, want.colind);
  ASSERT_EQ(m.values.size(), want.values.size());
  if (!m.values.empty()) {
    EXPECT_EQ(std::memcmp(m.values.data(), want.values.data(), m.values.size() * sizeof(T)), 0);
  }
}

std::string note_with_tag(const hs::LoadedMatrix& m, const std::string& tag) {
  for (const auto& n : m.notes)
    if (n.rfind(tag + ":", 0) == 0) return n;
  return "";
}

hs::LoadOptions unchecked(bool drop_zeros = true) {
  hs::LoadOptions o;
  o.check_hermitian = false;  // scrambled entries need not be Hermitian; compare assembly only
  o.storage = hs::Storage::Full;
  o.drop_zeros = drop_zeros;
  return o;
}

// ------------------------------------------------------------------------------- tests

template <class T>
class IoLayouts : public hs::io_test::TempDirTest {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(IoLayouts, Scalars);

TYPED_TEST(IoLayouts, CooAndCscMatchTheTripletAssemblyBitwise) {
  using T = TypeParam;
  // ~230k entries: the data members span several streamed chunks.
  const CsrMatrix<T> a = hs::test::random_hermitian<T>(2200, 0.04, 2026);
  const Entries<T> e = scramble(entries_of(a, false), 7);
  ASSERT_GT(e.vals.size() * sizeof(T), std::size_t{1} << 20);
  for (const bool compressed : {false, true})
    for (const bool drop_zeros : {true, false}) {
      SCOPED_TRACE(std::string(compressed ? "deflate" : "stored") + (drop_zeros ? " drop" : " keep"));
      hs::io::AssemblyStats stats;
      const CsrMatrix<T> want = reference_csr(e, drop_zeros, &stats);
      write_npz(this->tmp("coo.npz"), coo_members(e), compressed);
      const hs::LoadedMatrix coo = hs::load_matrix(this->tmp("coo.npz"), unchecked(drop_zeros));
      expect_same(coo, want);
      EXPECT_EQ(note_with_tag(coo, "duplicates"),
                "duplicates: summed " + std::to_string(stats.duplicates_summed) + " duplicate entries");

      const Entries<T> by_column = column_order(e);
      write_npz(this->tmp("csc.npz"), csc_members(e, /*unused_tail=*/5), compressed);
      const hs::LoadedMatrix csc = hs::load_matrix(this->tmp("csc.npz"), unchecked(drop_zeros));
      expect_same(csc, reference_csr(by_column, drop_zeros, &stats));
      EXPECT_EQ(note_with_tag(csc, "zeros"),
                stats.zeros_dropped > 0
                    ? "zeros: dropped " + std::to_string(stats.zeros_dropped) + " explicit zero entries"
                    : "");
    }
}

TYPED_TEST(IoLayouts, HermitianFilesGiveTheSameMatrixInEveryLayout) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::random_hermitian<T>(1500, 0.02, 11);
  const CsrMatrix<T> up = hs::triangle(a, true), lo = hs::triangle(a, false);
  for (const auto& [name, members, storage] :
       std::vector<std::tuple<std::string, Members, hs::Storage>>{
           {"coo full", coo_members(entries_of(a, false)), hs::Storage::Full},
           {"coo upper", coo_members(entries_of(up, false)), hs::Storage::Upper},
           {"csc full", csc_members(entries_of(a, true)), hs::Storage::Full},
           {"csc lower", csc_members(entries_of(lo, false)), hs::Storage::Lower}}) {
    SCOPED_TRACE(name);
    write_npz(this->tmp("m.npz"), members, true);
    const hs::LoadedMatrix m = hs::load_matrix(this->tmp("m.npz"));
    EXPECT_EQ(m.detected_storage, storage);
    EXPECT_EQ(m.hermitian_defect, 0.0);
    expect_same(m, a);
  }
}

TYPED_TEST(IoLayouts, NonNativeDtypesAreConvertedInChunks) {
  using T = TypeParam;
  using F = hs::rebind_t<T, float>;
  const CsrMatrix<T> a = hs::test::random_hermitian<T>(3000, 0.04, 5);
  // Single-precision values (exact in T) with int64 indices, in csr, csc and coo files.
  CsrMatrix<T> af = a;
  std::vector<F> data(a.values.size());
  for (std::size_t k = 0; k < data.size(); ++k) {
    data[k] = static_cast<F>(a.values[k]);
    af.values[k] = static_cast<T>(data[k]);
  }
  ASSERT_GT(data.size() * sizeof(F), std::size_t{1} << 20);
  std::vector<std::int64_t> indices(a.colind.begin(), a.colind.end()), indptr(a.rowptr.begin(), a.rowptr.end());
  std::vector<std::int64_t> rows;
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) rows.push_back(i);
  const auto shape = npy_of(std::vector<std::int64_t>{a.nrows, a.ncols});
  // A Hermitian matrix's CSC arrays are the CSR arrays of its conjugate (the real diagonal kept
  // as stored: conjugation would turn +0 imaginary parts into -0).
  std::vector<F> conj_data = data;
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] != i) conj_data[p] = hs::conj_if(conj_data[p]);
  for (const auto& [fmt, members] : std::vector<std::pair<std::string, Members>>{
           {"csr", {{"format", format_array("csr")}, {"shape", shape}, {"indptr", npy_of(indptr)},
                    {"indices", npy_of(indices)}, {"data", npy_of(data)}}},
           {"csc", {{"format", format_array("csc")}, {"shape", shape}, {"indptr", npy_of(indptr)},
                    {"indices", npy_of(indices)}, {"data", npy_of(conj_data)}}},
           {"coo", {{"format", format_array("coo")}, {"shape", shape}, {"row", npy_of(rows)},
                    {"col", npy_of(indices)}, {"data", npy_of(data)}}}}) {
    SCOPED_TRACE(fmt);
    write_npz(this->tmp("f.npz"), members, true);
    expect_same(hs::load_matrix(this->tmp("f.npz")), af);
  }
}

TYPED_TEST(IoLayouts, CooCoordsInBothMemoryOrders) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::random_hermitian<T>(800, 0.05, 3);
  const Entries<T> e = scramble(entries_of(a, false), 9);
  const std::size_t nnz = e.vals.size();
  for (const bool fortran : {false, true}) {
    SCOPED_TRACE(fortran ? "fortran" : "c");
    std::vector<std::int64_t> coords(2 * nnz);  // int64: converted while streamed
    for (std::size_t k = 0; k < nnz; ++k) {
      coords[fortran ? 2 * k : k] = e.rows[k];
      coords[fortran ? 2 * k + 1 : nnz + k] = e.cols[k];
    }
    const Members members{{"format", format_array("coo")},
                          {"shape", npy_of(std::vector<std::int64_t>{e.n, e.n})},
                          {"coords", npy_of(coords, {2, nnz}, fortran)},
                          {"data", npy_of(e.vals)}};
    write_npz(this->tmp("c.npz"), members, true);
    expect_same(hs::load_matrix(this->tmp("c.npz"), unchecked()), reference_csr(e, true));
  }
}

TYPED_TEST(IoLayouts, EmptyAndTinyMatrices) {
  using T = TypeParam;
  for (const index_t n : {0, 1, 4}) {
    SCOPED_TRACE(n);
    Entries<T> e;
    e.n = n;
    if (n == 1) e.add(0, 0, T(-1));
    if (n == 4) {  // rows 1 and 2 empty
      e.add(3, 3, T(2));
      e.add(0, 3, scalar<T>(1, 1));
      e.add(3, 0, scalar<T>(1, -1));
    }
    // (A complex file without imaginary parts, e.g. n <= 1, comes back real.)
    write_npz(this->tmp("coo.npz"), coo_members(e), false);
    expect_same_values(hs::load_matrix(this->tmp("coo.npz")), reference_csr(e, true));
    write_npz(this->tmp("csc.npz"), csc_members(e), true);
    expect_same_values(hs::load_matrix(this->tmp("csc.npz")), reference_csr(e, true));
  }
}

TYPED_TEST(IoLayouts, BrokenCooAndCscFilesAreRejected) {
  using T = TypeParam;
  Entries<T> e;
  e.n = 3;
  e.add(0, 0, T(1));
  e.add(2, 1, T(2));
  e.add(1, 2, T(2));
  const auto code_and_message = [&](const Members& members) {
    write_npz(this->tmp("bad.npz"), members, true);
    const auto err = error_of([&] { hs::load_matrix(this->tmp("bad.npz"), unchecked()); });
    return err.code + " | " + err.message;
  };
  const auto contains = [](const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
  };
  {  // coo: index outside the matrix, lengths that differ
    Members m = coo_members(e);
    m[3].second = npy_of(std::vector<std::int32_t>{0, 1, 3});
    const std::string s = code_and_message(m);
    EXPECT_TRUE(contains(s, "invalid_input") && contains(s, "entry 2 at (1, 3) lies outside")) << s;
    m = coo_members(e);
    m[2].second = npy_of(std::vector<std::int32_t>{0, 2});
    EXPECT_TRUE(contains(code_and_message(m), "coo arrays differ in length (row 2, col 3, data 3)"));
  }
  {  // csc: row index out of range, decreasing column pointer, index/value counts
    Members m = csc_members(e);
    m[3].second = npy_of(std::vector<std::int32_t>{0, 7, 1});
    EXPECT_TRUE(contains(code_and_message(m), "CSC row index 7 at position 1 is outside [0, 3)"));
    m = csc_members(e);
    m[2].second = npy_of(std::vector<std::int32_t>{0, 2, 1, 3});
    EXPECT_TRUE(contains(code_and_message(m), "CSC index pointer decreases at column 1"));
    m = csc_members(e);
    m[4].second = npy_of(std::vector<T>{T(1), T(2)});
    EXPECT_TRUE(contains(code_and_message(m), "CSC has 3 indices but 2 values"));
  }
}

// "<code> | <message>" of the error a load of `path` throws.
std::string load_error(const std::filesystem::path& path) {
  const auto err = error_of([&] { hs::load_matrix(path, unchecked()); });
  return err.code + " | " + err.message;
}

bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

class IoPrecedence : public hs::io_test::TempDirTest {
 protected:
  // 'data' that load_matrix cannot convert: float16 (no safe cast to float64).
  static hs::NpyArray half_data(std::size_t count) {
    hs::NpyArray a;
    a.dtype = "<f2";
    a.shape = {count};
    a.data.assign(2 * count, std::byte{0x3c});
    return a;
  }
  // A stored archive of `members` whose 'data' payload no longer matches its CRC-32.
  void write_with_corrupt_data(const std::filesystem::path& path, const Members& members) {
    write_npz(path, members, false);
    std::vector<char> bytes = read_bytes(path);
    const std::string text(bytes.begin(), bytes.end());
    const std::size_t name = text.find(std::string("data.npy\x93NUMPY"));  // the local record
    ASSERT_NE(name, std::string::npos);
    const std::size_t npy = name + 8;
    const std::size_t hlen = static_cast<unsigned char>(bytes[npy + 8]) |
                             static_cast<std::size_t>(static_cast<unsigned char>(bytes[npy + 9])) << 8;
    bytes[npy + 10 + hlen] ^= 0x41;  // first payload byte
    write_bytes(path, bytes);
  }
};

TEST_F(IoPrecedence, DataMemberErrorsComeBeforeIndexChecks) {
  // csc/coo files are checked entry by entry before 'data' is streamed. The error reported must
  // still be the one of reading every member whole first (as before the streaming readers): an
  // unconvertible or corrupt 'data' wins over bad indices and lengths.
  Entries<double> e;
  e.n = 3;
  e.add(0, 0, 1.0);
  e.add(2, 1, 2.0);
  e.add(1, 2, 2.0);
  const std::string f2 = "invalid_input | ";
  const std::string f2_message = "data.npy: cannot convert numpy dtype '<f2' to float64 (not a safe numpy cast)";
  const std::string crc_message = "data.npy: CRC-32 mismatch";
  {  // coo: row index outside the matrix
    Members m = coo_members(e);
    m[2].second = npy_of(std::vector<std::int32_t>{0, 7, 1});
    write_npz(tmp("a.npz"), m, true);
    std::string s = load_error(tmp("a.npz"));
    EXPECT_TRUE(contains(s, "entry 1 at (7, 1) lies outside the 3 x 3 matrix")) << s;
    m[4].second = half_data(3);
    write_npz(tmp("a.npz"), m, true);
    s = load_error(tmp("a.npz"));
    EXPECT_TRUE(contains(s, f2) && contains(s, f2_message)) << s;
    m[4].second = npy_of(e.vals);
    write_with_corrupt_data(tmp("a.npz"), m);
    s = load_error(tmp("a.npz"));
    EXPECT_TRUE(contains(s, "io | ") && contains(s, crc_message)) << s;
  }
  {  // coo: arrays of different lengths
    Members m = coo_members(e);
    m[3].second = npy_of(std::vector<std::int32_t>{0, 2});
    m[4].second = half_data(3);
    write_npz(tmp("b.npz"), m, true);
    const std::string s = load_error(tmp("b.npz"));
    EXPECT_TRUE(contains(s, f2_message)) << s;
  }
  {  // csc: row index outside the matrix, or index and value counts that differ
    Members m = csc_members(e);
    m[3].second = npy_of(std::vector<std::int32_t>{0, 9, 1});
    m[4].second = half_data(3);
    write_npz(tmp("c.npz"), m, true);
    std::string s = load_error(tmp("c.npz"));
    EXPECT_TRUE(contains(s, f2_message)) << s;
    m[4].second = npy_of(e.vals);
    write_with_corrupt_data(tmp("c.npz"), m);
    s = load_error(tmp("c.npz"));
    EXPECT_TRUE(contains(s, "io | ") && contains(s, crc_message)) << s;
    m = csc_members(e);
    m[4].second = half_data(2);
    write_npz(tmp("c.npz"), m, true);
    s = load_error(tmp("c.npz"));
    EXPECT_TRUE(contains(s, f2_message)) << s;
  }
  {  // coo 'coords': a member that cannot be converted wins over its wrong shape
    Members m{{"format", format_array("coo")},
              {"shape", npy_of(std::vector<std::int64_t>{3, 3})},
              {"coords", npy_of(std::vector<double>{0, 1, 2, 0, 1, 2}, {3, 2})},
              {"data", npy_of(e.vals)}};
    write_npz(tmp("d.npz"), m, true);
    std::string s = load_error(tmp("d.npz"));
    EXPECT_TRUE(contains(s, "coords.npy: cannot convert numpy dtype '<f8' to int32")) << s;
    m[2].second = npy_of(std::vector<std::int32_t>{0, 1, 2, 0, 1, 2}, {3, 2});
    write_npz(tmp("d.npz"), m, true);
    s = load_error(tmp("d.npz"));
    EXPECT_TRUE(contains(s, "'coords' must have shape (2, nnz)")) << s;
  }
}

class IoChunks : public hs::io_test::TempDirTest {};

TEST_F(IoChunks, HeaderClaimsBeyondThePayloadFailBeforeAnyAllocation) {
  // A corrupt npy header may claim any number of elements; only the member's length tells whether
  // the payload backs the claim, and it must be checked before anything is sized from the claim.
  // 2^40 elements: an attempt to allocate them throws std::bad_alloc, which is not an hs::Error and
  // fails the test, instead of the expected Io error.
  const auto claim = [&](const hs::NpyArray& a, const std::string& shape = "(1099511627776,)") {
    return with_claimed_shape(a, shape, tmp("member.npy"));
  };
  const auto raw = [&](const hs::NpyArray& a) { return npy_bytes(a, tmp("member.npy")); };
  const std::vector<double> d{1.0, 2.0, 3.0};
  const std::vector<float> f{1.0f, 2.0f, 3.0f};
  for (const bool compressed : {false, true}) {
    SCOPED_TRACE(compressed ? "deflate" : "stored");
    write_npz_raw(tmp("claim.npz"), {{"d", claim(npy_of(d))}, {"f", claim(npy_of(f))}}, compressed);
    const hs::io::NpzFile npz(tmp("claim.npz"));
    for (const char* key : {"d", "f"}) {  // read in place, and converted float -> double
      SCOPED_TRACE(key);
      const auto whole = error_of([&] { npz.read_as<double>(key); });
      EXPECT_EQ(whole.code, "io");
      EXPECT_TRUE(contains(whole.message, "(truncated)")) << whole.message;
      EXPECT_EQ(error_of([&] { npz.read_chunks<double>(key, [](const double*, std::size_t, std::size_t) {}); }).code,
                "io");
    }
  }
  // Every member load_matrix reads: csr/csc arrays, coo row/col/data and coords, shape.
  const CsrMatrix<double> a = hs::test::random_hermitian<double>(6, 0.5, 1);
  std::vector<std::int32_t> rows;
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) rows.push_back(static_cast<std::int32_t>(i));
  std::vector<std::int32_t> coords(rows);
  coords.insert(coords.end(), a.colind.begin(), a.colind.end());
  const std::vector<std::int32_t> indptr(a.rowptr.begin(), a.rowptr.end());
  const hs::NpyArray shape = npy_of(std::vector<std::int64_t>{a.nrows, a.ncols});
  const auto csr_like = [&](const char* fmt) {
    return RawMembers{{"format", raw(format_array(fmt))}, {"shape", raw(shape)}, {"indptr", raw(npy_of(indptr))},
                      {"indices", raw(npy_of(a.colind))}, {"data", raw(npy_of(a.values))}};
  };
  const RawMembers coo{{"format", raw(format_array("coo"))}, {"shape", raw(shape)}, {"row", raw(npy_of(rows))},
                       {"col", raw(npy_of(a.colind))}, {"data", raw(npy_of(a.values))}};
  std::vector<std::pair<std::string, RawMembers>> cases;
  const auto with = [&](const std::string& name, RawMembers members, std::size_t k, std::vector<char> bytes) {
    members[k].second = std::move(bytes);
    cases.emplace_back(name, std::move(members));
  };
  with("csr shape", csr_like("csr"), 1, claim(shape));
  with("csr indptr", csr_like("csr"), 2, claim(npy_of(indptr)));
  with("csr indices", csr_like("csr"), 3, claim(npy_of(a.colind)));
  with("csr data", csr_like("csr"), 4, claim(npy_of(a.values)));
  with("csc indices", csr_like("csc"), 3, claim(npy_of(a.colind)));
  with("csc data", csr_like("csc"), 4, claim(npy_of(a.values)));
  with("coo row", coo, 2, claim(npy_of(rows)));
  with("coo data", coo, 4, claim(npy_of(a.values)));
  RawMembers by_coords{{"format", raw(format_array("coo"))}, {"shape", raw(shape)},
                       {"coords", raw(npy_of(coords, {2, rows.size()}))}, {"data", raw(npy_of(a.values))}};
  with("coo coords", by_coords, 2, claim(npy_of(coords, {2, rows.size()}), "(2, 1099511627776)"));
  for (const auto& [name, members] : cases) {
    SCOPED_TRACE(name);
    write_npz_raw(tmp("m.npz"), members, true);
    const auto err = error_of([&] { hs::load_matrix(tmp("m.npz")); });
    EXPECT_EQ(err.code, "io") << err.message;
    EXPECT_TRUE(contains(err.message, "(truncated)")) << err.message;
  }
}

TEST_F(IoChunks, ConversionErrorsNameTheElementButYieldToCorruption) {
  // A complex member whose element big-2 is not real, read as double: the error comes from a
  // late chunk and numbers the element globally. In a stored archive, a flipped payload byte is
  // only detected by the CRC at the end; that Io error must win, as when members were read whole.
  const std::size_t big = (std::size_t{1} << 18) + 5;
  std::vector<cplx> v(big, cplx(1.0, 0.0));
  v[big - 2] = cplx(1.0, 0.5);
  write_npz(tmp("c.npz"), {{"data", npy_of(v)}}, false);
  const auto read = [&] {
    const hs::io::NpzFile npz(tmp("c.npz"));
    return error_of([&] { npz.read_chunks<double>("data", [](const double*, std::size_t, std::size_t) {}); });
  };
  const auto err = read();
  EXPECT_EQ(err.code, "invalid_input");
  EXPECT_NE(err.message.find("data.npy: cannot convert numpy dtype '<c16' to float64: element " +
                             std::to_string(big - 2) + " does not fit exactly"),
            std::string::npos)
      << err.message;
  EXPECT_EQ(error_of([&] { hs::io::NpzFile(tmp("c.npz")).read_as<double>("data"); }).message, err.message);
  std::vector<char> bytes = read_bytes(tmp("c.npz"));
  bytes[bytes.size() / 2] ^= 0x5A;  // inside the stored payload
  write_bytes(tmp("c.npz"), bytes);
  EXPECT_EQ(read().code, "io");
}

TEST_F(IoChunks, StreamsEveryElementOnceInOrder) {
  std::vector<std::int64_t> v(400000);
  for (std::size_t k = 0; k < v.size(); ++k) v[k] = static_cast<std::int64_t>(k * 7 + 3);
  write_npz(tmp("v.npz"), {{"v", npy_of(v)}, {"empty", npy_of(std::vector<std::int64_t>{})}}, true);
  const hs::io::NpzFile npz(tmp("v.npz"));
  for (const bool as_int32 : {false, true}) {  // native, and converted (checked narrowing)
    std::size_t expected_first = 0, calls = 0;
    bool in_order = true;
    std::vector<std::int64_t> got;
    const auto collect = [&](const auto* values, std::size_t first, std::size_t count) {
      in_order = in_order && first == expected_first && count > 0;
      expected_first = first + count;
      ++calls;
      got.insert(got.end(), values, values + count);
    };
    if (as_int32) {
      npz.read_chunks<std::int32_t>("v", [&](const std::int32_t* p, std::size_t f, std::size_t c) { collect(p, f, c); });
    } else {
      npz.read_chunks<std::int64_t>("v", [&](const std::int64_t* p, std::size_t f, std::size_t c) { collect(p, f, c); });
    }
    EXPECT_TRUE(in_order);
    EXPECT_GT(calls, 1u);  // 3.2 MB of int64: several chunks
    EXPECT_EQ(got, v);
  }
  std::size_t empty_calls = 0;
  npz.read_chunks<double>("empty", [&](const double*, std::size_t, std::size_t) { ++empty_calls; });
  EXPECT_EQ(empty_calls, 0u);
  // A rejected cast fails even for an empty array; an error thrown by the sink surfaces after the
  // member has been read completely.
  EXPECT_EQ(error_of([&] { npz.read_chunks<float>("empty", [](const float*, std::size_t, std::size_t) {}); }).code,
            "invalid_input");
  std::size_t after_error = 0;
  const auto err = error_of([&] {
    npz.read_chunks<std::int64_t>("v", [&](const std::int64_t*, std::size_t first, std::size_t) {
      if (first == 0) HS_THROW(InvalidInput, "sink says no");
      ++after_error;
    });
  });
  EXPECT_EQ(err.message, "sink says no");
  EXPECT_EQ(after_error, 0u);  // later chunks are read but not delivered
}

// --------------------------------------------------------------- MatrixMarket two-pass

template <class T>
std::string mtx_text(const Entries<T>& e, const char* symmetry) {
  std::string s = std::string("%%MatrixMarket matrix coordinate ") + (hs::is_complex_v<T> ? "complex " : "real ") +
                  symmetry + "\n% a comment line\n\n" + std::to_string(e.n) + " " + std::to_string(e.n) + " " +
                  std::to_string(e.vals.size()) + "\n";
  char buf[64];
  for (std::size_t k = 0; k < e.vals.size(); ++k) {
    s += std::to_string(e.rows[k] + 1) + " " + std::to_string(e.cols[k] + 1);
    if constexpr (hs::is_complex_v<T>) {
      std::snprintf(buf, sizeof(buf), " %.17g %.17g\n", e.vals[k].real(), e.vals[k].imag());
    } else {
      std::snprintf(buf, sizeof(buf), " %.17g\n", e.vals[k]);
    }
    s += buf;
    if (k % 1000 == 0) s += "% interleaved comment\n";
  }
  return s;
}

TYPED_TEST(IoLayouts, MatrixMarketCoordinateMatchesTheTripletAssembly) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::random_hermitian<T>(1200, 0.03, 17);
  const Entries<T> e = scramble(entries_of(a, true), 23);  // column-major, scrambled, duplicates
  for (const bool drop_zeros : {true, false}) {
    write_text(this->tmp("m.mtx"), mtx_text(e, "general"));
    hs::io::AssemblyStats stats;
    const CsrMatrix<T> want = reference_csr(e, drop_zeros, &stats);
    const hs::LoadedMatrix m = hs::load_matrix(this->tmp("m.mtx"), unchecked(drop_zeros));
    expect_same(m, want);
    EXPECT_EQ(note_with_tag(m, "duplicates"),
              "duplicates: summed " + std::to_string(stats.duplicates_summed) + " duplicate entries");
  }
  // One stored triangle (symmetric / hermitian banner), restored on load.
  const Entries<T> lower = entries_of(hs::triangle(a, false), true);
  write_text(this->tmp("s.mtx"), mtx_text(lower, hs::is_complex_v<T> ? "hermitian" : "symmetric"));
  const hs::LoadedMatrix s = hs::load_matrix(this->tmp("s.mtx"));
  EXPECT_EQ(s.detected_storage, hs::Storage::Lower);
  expect_same(s, a);
}

class IoMtxTwoPass : public hs::io_test::TempDirTest {};

TEST_F(IoMtxTwoPass, ErrorsAreReportedInFileOrder) {
  const auto error_for = [&](const std::string& body, std::size_t entries) {
    write_text(tmp("e.mtx"), "%%MatrixMarket matrix coordinate real general\n3 3 " + std::to_string(entries) +
                                 "\n" + body);
    const auto e = error_of([&] { hs::load_matrix(tmp("e.mtx")); });
    return e.code + " | " + e.message;
  };
  // A malformed value (line 4) before an unreadable row index (line 5): line 4 is reported, as
  // with a single pass, although the counting pass skips line 5 silently.
  std::string s = error_for("1 1 1\n2 2 x\nq 3 1\n", 3);
  EXPECT_NE(s.find("e.mtx:4: expected a value, got 'x'"), std::string::npos) << s;
  s = error_for("1 1 1\n4 2 1\n", 2);
  EXPECT_NE(s.find("e.mtx:4: entry (4, 2) lies outside the 3 x 3 matrix"), std::string::npos) << s;
  s = error_for("1 1 1\n+2 2 1\n3 3 1\n1 2 1\n", 3);  // a '+' row index counts; one entry too many
  EXPECT_NE(s.find("e.mtx:6: more entries than the 3 declared"), std::string::npos) << s;
  s = error_for("1 1 1\n2 2 1\n", 3);
  EXPECT_NE(s.find("file ends after 2 of 3 entries"), std::string::npos) << s;
}

TEST_F(IoMtxTwoPass, AnOverlongLineDoesNotHideAnEarlierError) {
  // The counting pass reads the lines too. A line the reader rejects (5 MiB: longer than its
  // whole buffer, so rejected wherever it starts) behind a malformed value must not be reported
  // first; on its own it is reported with its line number.
  const std::string banner = "%%MatrixMarket matrix coordinate real general\n3 3 3\n";
  const std::string long_line = "%" + std::string(std::size_t{5} << 20, 'x') + "\n";
  const auto error_for = [&](const std::string& body) {
    write_text(tmp("l.mtx"), banner + body);
    const auto e = error_of([&] { hs::load_matrix(tmp("l.mtx")); });
    return e.code + " | " + e.message;
  };
  std::string s = error_for("1 1 1\n2 2 x\n" + long_line + "3 3 1\n");
  EXPECT_NE(s.find("io | "), std::string::npos) << s;
  EXPECT_NE(s.find("l.mtx:4: expected a value, got 'x'"), std::string::npos) << s;
  s = error_for("1 1 1\n2 2 1\n" + long_line + "3 3 1\n");
  EXPECT_NE(s.find("l.mtx:5: line longer than 1048576 bytes"), std::string::npos) << s;
}

}  // namespace
