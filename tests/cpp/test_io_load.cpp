// Unit tests for src/io, part 2: load_matrix (scipy npz, dense npy, MatrixMarket) against the
// fixtures of tests/data/make_io_fixtures.py, MatrixMarket parsing, and the CSR assembly helpers
// behind the loader (src/io/csr_assembly.hpp).
#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"
#include "io/csr_assembly.hpp"
#include "support/fixtures.hpp"
#include "test_io_util.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::io_test::error_code_of;
using hs::io_test::expected;
using hs::io_test::fixture;
using hs::io_test::write_text;
using json = nlohmann::json;

std::string storage_name(hs::Storage s) {
  switch (s) {
    case hs::Storage::Auto: return "auto";
    case hs::Storage::Full: return "full";
    case hs::Storage::Upper: return "upper";
    case hs::Storage::Lower: return "lower";
  }
  return "?";
}

hs::LoadOptions options_from(const json& o) {
  hs::LoadOptions opt;
  if (o.contains("storage")) {
    for (const auto s : {hs::Storage::Auto, hs::Storage::Full, hs::Storage::Upper, hs::Storage::Lower})
      if (storage_name(s) == o["storage"]) opt.storage = s;
  }
  if (o.contains("drop_zeros")) opt.drop_zeros = o["drop_zeros"];
  if (o.contains("check_hermitian")) opt.check_hermitian = o["check_hermitian"];
  if (o.contains("hermitian_tol")) opt.hermitian_tol = o["hermitian_tol"];
  return opt;
}

std::vector<std::string> note_tags(const std::vector<std::string>& notes) {
  std::vector<std::string> tags;
  for (const auto& n : notes) tags.push_back(n.substr(0, n.find(':')));
  std::sort(tags.begin(), tags.end());
  return tags;
}

template <class T>
void expect_dense(const CsrMatrix<T>& a, const json& c) {
  EXPECT_NO_THROW(a.validate());
  const hs::DenseMatrix<T> d = hs::test::to_dense(a);
  const auto& re = c["re"];
  ASSERT_EQ(static_cast<std::size_t>(d.rows()), re.size());
  for (index_t i = 0; i < d.rows(); ++i) {
    ASSERT_EQ(static_cast<std::size_t>(d.cols()), re[i].size());
    for (index_t j = 0; j < d.cols(); ++j) {
      T want;
      if constexpr (hs::is_complex_v<T>) {
        want = T(re[i][j].get<double>(), c["im"][i][j].get<double>());
      } else {
        want = re[i][j].get<double>();
      }
      EXPECT_EQ(d(i, j), want) << "entry (" << i << ", " << j << ")";
    }
  }
}

void check_loaded(const hs::LoadedMatrix& m, const json& c) {
  EXPECT_EQ(m.source_format, c["source_format"].get<std::string>());
  EXPECT_EQ(storage_name(m.detected_storage), c["storage"].get<std::string>());
  EXPECT_EQ(note_tags(m.notes), c["notes"].get<std::vector<std::string>>());
  EXPECT_EQ(m.order(), c["n"].get<index_t>());
  EXPECT_EQ(m.is_complex(), c["dtype"] == "complex128");
  const bool checked = !c["defect"].is_null();
  if (!checked) {
    EXPECT_EQ(m.hermitian_defect, -1.0);
  } else if (c["defect"].get<double>() == 0.0) {
    EXPECT_EQ(m.hermitian_defect, 0.0);
  } else {
    const double want = c["defect"];
    EXPECT_NEAR(m.hermitian_defect, want, 1e-6 * want);
  }
  std::visit(
      [&](const auto& a) {
        EXPECT_EQ(a.nnz(), c["nnz"].get<index_t>());
        expect_dense(a, c);
        if (checked) {
          EXPECT_EQ(hs::hermitian_defect(a), 0.0);  // exactly Hermitian after loading
        }
      },
      m.matrix);
}

// Runs every expected.json load case whose file has the given extension.
void run_fixture_cases(const std::string& ext) {
  int ran = 0;
  for (const auto& c : expected()["load"]) {
    const std::string file = c["file"];
    if (file.size() < ext.size() || file.compare(file.size() - ext.size(), ext.size(), ext) != 0)
      continue;
    SCOPED_TRACE(c["name"].get<std::string>());
    ++ran;
    const auto path = fixture(file);
    const hs::LoadOptions opt = options_from(c["options"]);
    if (c.contains("error")) {
      EXPECT_EQ(error_code_of([&] { hs::load_matrix(path, opt); }), c["error"].get<std::string>());
    } else {
      check_loaded(hs::load_matrix(path, opt), c);
    }
  }
  EXPECT_GT(ran, 0) << "no fixtures with extension " << ext;
}

TEST(IoLoad, ScipyNpzFixtures) { run_fixture_cases(".npz"); }
TEST(IoLoad, DenseNpyFixtures) { run_fixture_cases(".npy"); }
TEST(IoLoad, MatrixMarketFixtures) { run_fixture_cases(".mtx"); }

class IoLoadFiles : public hs::io_test::TempDirTest {};

TEST_F(IoLoadFiles, DispatchesOnCaseInsensitiveExtension) {
  std::filesystem::copy_file(fixture("csr_matrix_f8.npz"), tmp("A.NPZ"));
  const hs::LoadedMatrix m = hs::load_matrix(tmp("A.NPZ"));
  EXPECT_EQ(m.source_format, "npz-csr");
  EXPECT_EQ(m.order(), 5);
  std::filesystem::copy_file(fixture("csr_matrix_f8.npz"), tmp("a.txt"));
  EXPECT_EQ(error_code_of([&] { hs::load_matrix(tmp("a.txt")); }), "invalid_input");
  EXPECT_EQ(error_code_of([&] { hs::load_matrix(tmp("missing.npz")); }), "io");
  EXPECT_EQ(error_code_of([&] { hs::load_matrix(tmp("missing.mtx")); }), "io");
  hs::LoadOptions bad;
  bad.hermitian_tol = -1.0;
  EXPECT_EQ(error_code_of([&] { hs::load_matrix(tmp("A.NPZ"), bad); }), "invalid_input");
  bad.hermitian_tol = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(error_code_of([&] { hs::load_matrix(tmp("A.NPZ"), bad); }), "invalid_input");
}

// ------------------------------------------------------------------------- MatrixMarket

class IoMtx : public hs::io_test::TempDirTest {
 protected:
  std::string load_error(const std::string& text) {
    write_text(tmp("m.mtx"), text);
    return error_code_of([&] { hs::load_matrix(tmp("m.mtx")); });
  }
};

TEST_F(IoMtx, RejectsMalformedFiles) {
  const std::string real = "%%MatrixMarket matrix coordinate real general\n";
  EXPECT_EQ(load_error(""), "io");
  EXPECT_EQ(load_error(real), "io");                                       // no size line
  EXPECT_EQ(load_error("2 2 1\n1 1 1.0\n"), "io");                         // no banner
  EXPECT_EQ(load_error("%%MatrixMarket matrix coordinate real\n"), "io");  // incomplete
  EXPECT_EQ(load_error("%%MatrixMarket matrix coordinate quaternion general\n1 1 0\n"), "io");
  EXPECT_EQ(load_error("%%MatrixMarket matrix sparse real general\n1 1 0\n"), "io");
  EXPECT_EQ(load_error("%%MatrixMarket matrix coordinate real upper\n1 1 0\n"), "io");
  EXPECT_EQ(load_error("%%MatrixMarket matrix array pattern general\n1 1\n"), "io");
  EXPECT_EQ(load_error("%%MatrixMarket vector coordinate real general\n1 1 0\n"), "invalid_input");
  EXPECT_EQ(load_error(real + "2 2 1 7\n1 1 1.0\n"), "io");    // extra size token
  EXPECT_EQ(load_error(real + "2 -2 1\n1 1 1.0\n"), "io");     // negative size
  EXPECT_EQ(load_error(real + "2 2 1\n1 1 1.0 2.0\n"), "io");  // extra value token
  EXPECT_EQ(load_error(real + "2 2 1\n1 1 abc\n"), "io");
  EXPECT_EQ(load_error(real + "2 2 1\n1 1\n"), "io");          // missing value
  EXPECT_EQ(load_error(real + "2 2 1\n1.5 1 1.0\n"), "io");    // non-integer index
  EXPECT_EQ(load_error(real + "2 2 1\n0 1 1.0\n"), "io");      // 1-based
  EXPECT_EQ(load_error(real + "2 2 1\n1 1 1e400\n"), "io");    // overflows double
  EXPECT_EQ(load_error(real + "2 2 1\n1 1 1.0\n2 2 1.0\n"), "io");
  EXPECT_EQ(load_error(real + "2 2 100000\n1 1 1.0\n"), "io");  // more than the file can hold
  EXPECT_EQ(load_error(real + "2 2 1\n1 1 inf\n"), "invalid_input");
  EXPECT_EQ(load_error("%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 2.5\n"), "io");
  EXPECT_EQ(load_error("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n"), "io");
  EXPECT_EQ(load_error("%%MatrixMarket matrix array real general\n1 1\n1\n2\n"), "io");
}

TEST_F(IoMtx, ToleratesCaseCommentsAndMissingFinalNewline) {
  write_text(tmp("m.mtx"),
             "%%matrixmarket MATRIX Coordinate REAL General\n%\n% comment\n\n  2 2 4\n"
             "1 1 +2.5\n% interleaved comment\n2 2 -1e-3\n\t1 2 0.5\r\n2 1 0.5");
  const hs::LoadedMatrix m = hs::load_matrix(tmp("m.mtx"));
  const auto& a = std::get<CsrMatrix<double>>(m.matrix);
  EXPECT_EQ(a.rowptr, (std::vector<index_t>{0, 2, 4}));
  EXPECT_EQ(a.values, (std::vector<double>{2.5, 0.5, 0.5, -1e-3}));
}

template <class T>
std::string mtx_number(T v) {
  // Shortest text that parses back to exactly v.
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, res.ptr);
}

template <class T>
class IoMtxTyped : public hs::io_test::TempDirTest {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(IoMtxTyped, Scalars);

// A matrix written as MatrixMarket text (upper triangle only, entries in reverse order to force
// sorting) comes back exactly; the file exceeds the reader's 4 MiB chunk, so lines straddle
// chunk boundaries.
TYPED_TEST(IoMtxTyped, RoundTripsLargeFilesExactly) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::random_hermitian<T>(3000, 0.04, 2026);
  std::string text = std::string("%%MatrixMarket matrix coordinate ") +
                     (hs::is_complex_v<T> ? "complex" : "real") + " general\n";
  std::vector<std::string> lines;
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      if (a.colind[p] < i) continue;
      std::string ln = std::to_string(i + 1) + " " + std::to_string(a.colind[p] + 1) + " ";
      if constexpr (hs::is_complex_v<T>) {
        ln += mtx_number(a.values[p].real()) + " " + mtx_number(a.values[p].imag());
      } else {
        ln += mtx_number(a.values[p]);
      }
      lines.push_back(ln);
    }
  text += std::to_string(a.nrows) + " " + std::to_string(a.ncols) + " " + std::to_string(lines.size()) + "\n";
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) text += *it + "\n";
  ASSERT_GT(text.size(), std::size_t{5} << 20);
  write_text(this->tmp("big.mtx"), text);

  const hs::LoadedMatrix m = hs::load_matrix(this->tmp("big.mtx"));
  EXPECT_EQ(m.detected_storage, hs::Storage::Upper);
  const auto& b = std::get<CsrMatrix<T>>(m.matrix);
  EXPECT_EQ(b.rowptr, a.rowptr);
  EXPECT_EQ(b.colind, a.colind);
  EXPECT_EQ(b.values, a.values);
}

// ------------------------------------------------------------------------ CSR assembly

template <class T>
class IoAssembly : public ::testing::Test {};
TYPED_TEST_SUITE(IoAssembly, Scalars);

template <class T>
T scalar(double re, double im) {
  if constexpr (hs::is_complex_v<T>) {
    return T(re, im);
  } else {
    return re;
  }
}

TYPED_TEST(IoAssembly, CanonicalizeSortsSumsAndDropsZeros) {
  using T = TypeParam;
  CsrMatrix<T> raw;
  raw.nrows = 3;
  raw.ncols = 4;
  // row 0: (0,3)=1, (0,1)=2, (0,3)=3 ; row 1: (1,2)=0 ; row 2: (2,0)=5, (2,0)=-5, (2,1)=1
  // plus two trailing unused entries (allowed by scipy).
  raw.rowptr = {0, 3, 4, 7};
  raw.colind = {3, 1, 3, 2, 0, 0, 1, 9, 9};
  raw.values = {scalar<T>(1, 1), scalar<T>(2, 0), scalar<T>(3, -1), T(0), scalar<T>(5, 0),
                scalar<T>(-5, 0), scalar<T>(1, 0), T(7), T(7)};
  CsrMatrix<T> keep = raw;
  const hs::io::AssemblyStats s = hs::io::canonicalize_csr(raw, /*drop_zeros=*/true);
  EXPECT_EQ(s.duplicates_summed, 2);
  EXPECT_EQ(s.zeros_dropped, 2);
  EXPECT_NO_THROW(raw.validate());
  EXPECT_EQ(raw.rowptr, (std::vector<index_t>{0, 2, 2, 3}));
  EXPECT_EQ(raw.colind, (std::vector<std::int32_t>{1, 3, 1}));
  EXPECT_EQ(raw.values, (std::vector<T>{scalar<T>(2, 0), scalar<T>(4, 0), scalar<T>(1, 0)}));

  const hs::io::AssemblyStats k = hs::io::canonicalize_csr(keep, /*drop_zeros=*/false);
  EXPECT_EQ(k.duplicates_summed, 2);
  EXPECT_EQ(k.zeros_dropped, 0);
  EXPECT_EQ(keep.rowptr, (std::vector<index_t>{0, 2, 3, 5}));
  EXPECT_EQ(keep.colind, (std::vector<std::int32_t>{1, 3, 2, 0, 1}));
}

TYPED_TEST(IoAssembly, CanonicalizeRejectsBrokenStructure) {
  using T = TypeParam;
  const auto check = [](std::vector<index_t> rowptr, std::vector<std::int32_t> colind,
                        std::size_t nvals) {
    CsrMatrix<T> m;
    m.nrows = 2;
    m.ncols = 2;
    m.rowptr = std::move(rowptr);
    m.colind = std::move(colind);
    m.values.assign(nvals, T(1));
    return error_code_of([&] { hs::io::canonicalize_csr(m, true); });
  };
  EXPECT_EQ(check({0, 1}, {0}, 1), "invalid_input");              // rowptr too short
  EXPECT_EQ(check({1, 1, 2}, {0, 1}, 2), "invalid_input");        // does not start at 0
  EXPECT_EQ(check({0, 2, 1}, {0, 1}, 2), "invalid_input");        // decreasing
  EXPECT_EQ(check({0, 1, 3}, {0, 1}, 2), "invalid_input");        // beyond the stored entries
  EXPECT_EQ(check({0, 1, 2}, {0, 1}, 3), "invalid_input");        // indices vs values
  EXPECT_EQ(check({0, 1, 2}, {0, 2}, 2), "invalid_input");        // column out of range
  EXPECT_EQ(check({0, 1, 2}, {-1, 0}, 2), "invalid_input");       // negative column
  EXPECT_EQ(check({0, 1, 2}, {1, 0}, 2), "none");
}

TYPED_TEST(IoAssembly, TripletsAgreeWithCsrFromCoo) {
  using T = TypeParam;
  // Integer-valued entries make sums exact regardless of the summation order.
  std::vector<std::int32_t> r, c;
  std::vector<index_t> r64, c64;
  std::vector<T> v;
  for (int k = 0; k < 4000; ++k) {
    const int i = (k * 7919) % 97, j = (k * 104729) % 89;
    r.push_back(i);
    c.push_back(j);
    r64.push_back(i);
    c64.push_back(j);
    v.push_back(scalar<T>(k % 5 - 2, k % 3 - 1));
  }
  hs::io::AssemblyStats s;
  const CsrMatrix<T> got = hs::io::csr_from_triplets<T>(97, 89, r.data(), c.data(), v.data(), v.size(),
                                                        true, s);
  const CsrMatrix<T> want = hs::csr_from_coo<T>(97, 89, r64, c64, v, true);
  EXPECT_EQ(got.rowptr, want.rowptr);
  EXPECT_EQ(got.colind, want.colind);
  EXPECT_EQ(got.values, want.values);
  EXPECT_EQ(s.duplicates_summed + s.zeros_dropped + got.nnz(), static_cast<index_t>(v.size()));

  r[17] = 97;
  EXPECT_EQ(error_code_of([&] {
              hs::io::csr_from_triplets<T>(97, 89, r.data(), c.data(), v.data(), v.size(), true, s);
            }),
            "invalid_input");
}

TYPED_TEST(IoAssembly, TransposeDoesNotConjugate) {
  using T = TypeParam;
  const std::vector<index_t> r{0, 0, 1, 2}, c{1, 2, 0, 2};
  const std::vector<T> v{scalar<T>(1, 2), scalar<T>(3, -4), scalar<T>(5, 6), scalar<T>(7, 0)};
  const CsrMatrix<T> a = hs::csr_from_coo<T>(3, 3, r, c, v);
  const CsrMatrix<T> t = hs::io::transpose(a);
  const auto da = hs::test::to_dense(a), dt = hs::test::to_dense(t);
  for (index_t i = 0; i < 3; ++i)
    for (index_t j = 0; j < 3; ++j) EXPECT_EQ(dt(j, i), da(i, j));
}

TYPED_TEST(IoAssembly, HermitianPartInPlaceMatchesSparseModuleBitwise) {
  using T = TypeParam;
  CsrMatrix<T> a = hs::test::random_hermitian<T>(400, 0.05, 7);
  // Rounding-level asymmetry on an unchanged (symmetric) pattern.
  for (index_t p = 0; p < a.nnz(); ++p) a.values[p] *= 1.0 + 1e-13 * static_cast<double>(p % 7 - 3);
  ASSERT_GT(hs::hermitian_defect(a), 0.0);
  const CsrMatrix<T> want = hs::hermitian_part(a);
  ASSERT_TRUE(hs::io::hermitian_part_in_place(a));
  EXPECT_EQ(a.rowptr, want.rowptr);
  EXPECT_EQ(a.colind, want.colind);
  EXPECT_EQ(a.values, want.values);
  EXPECT_EQ(hs::hermitian_defect(a), 0.0);
}

TYPED_TEST(IoAssembly, HermitianPartInPlaceDeclinesAsymmetricPatterns) {
  using T = TypeParam;
  const std::vector<index_t> r{0, 1, 1, 2}, c{0, 1, 2, 2};  // (1,2) without (2,1)
  const std::vector<T> v{T(1), T(2), scalar<T>(1e-14, 1e-14), T(3)};
  CsrMatrix<T> a = hs::csr_from_coo<T>(3, 3, r, c, v);
  const CsrMatrix<T> before = a;
  EXPECT_FALSE(hs::io::hermitian_part_in_place(a));
  EXPECT_EQ(a.values, before.values);
}

}  // namespace
