// Unit tests for src/io, part 1: .npy parsing / conversion / writing and zip / .npz reading.
// File fixtures come from tests/data/make_io_fixtures.py (expected values in expected.json).
#include <gtest/gtest.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "io/npz.hpp"
#include "linalg/dense_ops.hpp"
#include "test_io_util.hpp"

namespace {

using hs::cplx;
using hs::NpyArray;
using hs::io_test::error_code_of;
using hs::io_test::expected;
using hs::io_test::fixture;
using hs::io_test::read_bytes;
using hs::io_test::write_bytes;
using json = nlohmann::json;

// .npy bytes: magic, version, header length (u16 for 1.x, u32 otherwise), header + '\n', payload.
std::vector<std::byte> make_npy(const std::string& header, const std::vector<std::byte>& payload,
                                int major = 1, int minor = 0) {
  const std::string h = header + "\n";
  std::vector<std::byte> out;
  for (const char c : std::string("\x93NUMPY")) out.push_back(static_cast<std::byte>(c));
  out.push_back(static_cast<std::byte>(major));
  out.push_back(static_cast<std::byte>(minor));
  const std::size_t len_bytes = major == 1 ? 2 : 4;
  for (std::size_t b = 0; b < len_bytes; ++b)
    out.push_back(static_cast<std::byte>((h.size() >> (8 * b)) & 0xFF));
  for (const char c : h) out.push_back(static_cast<std::byte>(c));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

template <class T>
std::vector<std::byte> bytes_of(const std::vector<T>& v) {
  std::vector<std::byte> b(v.size() * sizeof(T));
  if (!b.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}

NpyArray parse(const std::string& header, const std::vector<std::byte>& payload, int major = 1) {
  const auto bytes = make_npy(header, payload, major);
  return hs::parse_npy(bytes.data(), bytes.size());
}

std::string parse_error(const std::string& header, const std::vector<std::byte>& payload) {
  return error_code_of([&] { parse(header, payload); });
}

NpyArray array_of(const std::string& dtype, std::vector<std::size_t> shape,
                  std::vector<std::byte> data) {
  NpyArray a;
  a.dtype = dtype;
  a.shape = std::move(shape);
  a.data = std::move(data);
  return a;
}

template <class T>
NpyArray vector_array(const std::string& dtype, const std::vector<T>& v) {
  return array_of(dtype, {v.size()}, bytes_of(v));
}

// Big-endian copy of the elements of v (each component of a complex swapped separately).
template <class T>
std::vector<std::byte> big_endian_bytes(const std::vector<T>& v, std::size_t component) {
  std::vector<std::byte> b = bytes_of(v);
  for (std::size_t k = 0; k < b.size(); k += component)
    std::reverse(b.begin() + static_cast<std::ptrdiff_t>(k),
                 b.begin() + static_cast<std::ptrdiff_t>(k + component));
  return b;
}

const std::vector<double> kTwo{1.5, -2.0};

// --------------------------------------------------------------------------- npy header

TEST(IoNpyHeader, AcceptsAnyPythonDictSpelling) {
  const std::vector<std::string> headers{
      "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }",
      "{\"descr\":\"<f8\",\"shape\":(2,),\"fortran_order\":False}",
      "{ 'shape' : ( 2 , ) ,\t'fortran_order' : False , 'descr' : '<f8' , }   ",
      "{'descr': '<f8', 'fortran_order': False, 'shape': (2L,)}",
      "{'descr':'f8','fortran_order':False,'shape':(2,),}",
  };
  for (const auto& h : headers) {
    SCOPED_TRACE(h);
    const NpyArray a = parse(h, bytes_of(kTwo));
    EXPECT_EQ(a.shape, std::vector<std::size_t>{2});
    EXPECT_FALSE(a.fortran_order);
    EXPECT_EQ(hs::npy_to_vector<double>(a), kTwo);
  }
  const NpyArray v2 = parse("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }",
                            bytes_of(kTwo), /*major=*/2);
  EXPECT_EQ(hs::npy_to_vector<double>(v2), kTwo);
  const NpyArray v3 = parse("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }",
                            bytes_of(kTwo), /*major=*/3);
  EXPECT_EQ(hs::npy_to_vector<double>(v3), kTwo);
}

TEST(IoNpyHeader, ParsesShapesAndOrder) {
  const NpyArray scalar = parse("{'descr': '<f8', 'fortran_order': False, 'shape': (), }",
                                bytes_of(std::vector<double>{4.0}));
  EXPECT_TRUE(scalar.shape.empty());
  EXPECT_EQ(scalar.size(), 1u);
  const NpyArray empty = parse("{'descr': '<i4', 'fortran_order': False, 'shape': (3, 0), }", {});
  EXPECT_EQ(empty.shape, (std::vector<std::size_t>{3, 0}));
  EXPECT_EQ(empty.size(), 0u);
  const NpyArray f = parse("{'descr': '<f8', 'fortran_order': True, 'shape': (1, 2,), }",
                           bytes_of(kTwo));
  EXPECT_TRUE(f.fortran_order);
  EXPECT_EQ(f.shape, (std::vector<std::size_t>{1, 2}));
}

TEST(IoNpyHeader, RejectsMalformedHeaders) {
  const auto two = bytes_of(kTwo);
  const std::string tail = "'fortran_order': False, 'shape': (2,), }";
  EXPECT_EQ(parse_error("{'descr': '<f8', 'fortran_order': False, 'shape': (2), }", two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'fortran_order': False, 'shape': [2], }", two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'fortran_order': False, 'shape': (-2,), }", two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'fortran_order': False, 'shape': (2 1), }", two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'fortran_order': 0, 'shape': (2,), }", two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'shape': (2,), }", two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'descr': '<f8', " + tail, two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', 'extra': 1, " + tail, two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8, " + tail, two), "io");
  EXPECT_EQ(parse_error("{'descr': '<f8', " + tail + " junk", two), "io");
  EXPECT_EQ(parse_error("'descr': '<f8', " + tail, two), "io");
  EXPECT_EQ(parse_error("{'descr': '<q8', " + tail, two), "io");
  EXPECT_EQ(parse_error("{'descr': [('a', '<f8')], " + tail, two), "unsupported");
  EXPECT_EQ(parse_error("{'descr': '|O', " + tail, two), "unsupported");
}

TEST(IoNpyHeader, RejectsBadMagicVersionAndPayloadSize) {
  const std::string h = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
  auto bytes = make_npy(h, bytes_of(kTwo));
  bytes[1] = std::byte{'X'};
  EXPECT_EQ(error_code_of([&] { hs::parse_npy(bytes.data(), bytes.size()); }), "io");
  for (const auto& [major, minor] : {std::pair{4, 0}, std::pair{1, 1}, std::pair{0, 0}}) {
    const auto b = make_npy(h, bytes_of(kTwo), major, minor);
    EXPECT_EQ(error_code_of([&] { hs::parse_npy(b.data(), b.size()); }), "io") << major << "." << minor;
  }
  EXPECT_EQ(parse_error(h, bytes_of(std::vector<double>{1.0})), "io");            // truncated
  EXPECT_EQ(parse_error(h, bytes_of(std::vector<double>{1.0, 2.0, 3.0})), "io");  // trailing
  const auto full = make_npy(h, bytes_of(kTwo));
  EXPECT_EQ(error_code_of([&] { hs::parse_npy(full.data(), 5); }), "io");
  EXPECT_EQ(error_code_of([&] { hs::parse_npy(full.data(), 12); }), "io");
}

TEST(IoNpyArray, ItemSizeFollowsDescr) {
  const std::vector<std::pair<std::string, std::size_t>> cases{
      {"<f8", 8}, {">c16", 16}, {"|b1", 1}, {"|u1", 1}, {"<i2", 2}, {"|S3", 3}, {"<U3", 12},
      {"<M8[ns]", 8}, {"<m8", 8}, {"f4", 4}, {"=i8", 8}, {"|V5", 5}, {"|a2", 2}};
  for (const auto& [descr, size] : cases) {
    NpyArray a;
    a.dtype = descr;
    EXPECT_EQ(a.item_size(), size) << descr;
  }
  for (const std::string bad : {"", "<", "<f", "<x4", "<f8x", "<M8[ns", "<M8[n-s]"}) {
    NpyArray a;
    a.dtype = bad;
    EXPECT_EQ(error_code_of([&] { a.item_size(); }), "io") << bad;
  }
  NpyArray zero;
  zero.dtype = "|S0";
  EXPECT_EQ(error_code_of([&] { zero.item_size(); }), "unsupported");
}

// ------------------------------------------------------------------------- npy fixtures

template <class T>
void expect_converted(const NpyArray& a, const json& c) {
  const std::vector<T> got = hs::npy_to_vector<T>(a);
  const auto& re = c["re"];
  ASSERT_EQ(got.size(), re.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    if constexpr (hs::is_complex_v<T>) {
      const double im = c["im"].is_null() ? 0.0 : c["im"][k].get<double>();  // real source
      EXPECT_EQ(got[k], cplx(re[k].get<double>(), im)) << k;
    } else if constexpr (std::is_integral_v<T>) {
      EXPECT_EQ(got[k], re[k].get<T>()) << k;
    } else {
      EXPECT_EQ(got[k], re[k].get<double>()) << k;
    }
  }
}

void check_array(const NpyArray& a, const json& c) {
  EXPECT_EQ(a.dtype, c["dtype"].get<std::string>());
  EXPECT_EQ(a.shape, c["shape"].get<std::vector<std::size_t>>());
  EXPECT_EQ(a.fortran_order, c["fortran_order"].get<bool>());
  EXPECT_EQ(a.data.size(), a.size() * a.item_size());
  if (c["convert"].is_null()) {
    EXPECT_EQ(error_code_of([&] { hs::npy_to_vector<double>(a); }), "invalid_input");
    return;
  }
  const std::string to = c["convert"];
  if (to == "float64") expect_converted<double>(a, c);
  if (to == "complex128") expect_converted<cplx>(a, c);
  if (to == "int32") expect_converted<std::int32_t>(a, c);
  if (to == "int64") expect_converted<std::int64_t>(a, c);
}

TEST(IoNpy, ReadsNumpyWrittenFiles) {
  ASSERT_FALSE(expected()["npy"].empty());
  for (const auto& c : expected()["npy"]) {
    const std::string file = c["file"];
    SCOPED_TRACE(file);
    check_array(hs::read_npy(fixture(file)), c);
    const auto raw = read_bytes(fixture(file));
    check_array(hs::parse_npy(reinterpret_cast<const std::byte*>(raw.data()), raw.size()), c);
  }
}

TEST(IoNpy, ReadsStringPayloadsVerbatim) {
  const NpyArray u = hs::read_npy(fixture("npy_u3.npy"));
  ASSERT_EQ(u.data.size(), 24u);  // 2 x 3 UCS-4 characters
  EXPECT_EQ(std::to_integer<char>(u.data[12]), 'x');
  const NpyArray s = hs::read_npy(fixture("npy_s3.npy"));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(s.data.data()), 3), "csr");
  EXPECT_EQ(error_code_of([] { hs::read_npy(fixture("does_not_exist.npy")); }), "io");
}

// ----------------------------------------------------------------------- npy_to_vector

TEST(IoNpyConvert, ToDoubleAcceptsSafeCastsAndRealComplex) {
  EXPECT_EQ(hs::npy_to_vector<double>(vector_array("<f4", std::vector<float>{0.5f, -2.0f})),
            (std::vector<double>{0.5, -2.0}));
  EXPECT_EQ(hs::npy_to_vector<double>(vector_array("<i8", std::vector<std::int64_t>{-3, 1LL << 60})),
            (std::vector<double>{-3.0, std::ldexp(1.0, 60)}));
  EXPECT_EQ(hs::npy_to_vector<double>(vector_array("<u8", std::vector<std::uint64_t>{7})),
            (std::vector<double>{7.0}));
  EXPECT_EQ(hs::npy_to_vector<double>(vector_array("|b1", std::vector<std::uint8_t>{0, 2})),
            (std::vector<double>{0.0, 1.0}));
  EXPECT_EQ(hs::npy_to_vector<double>(vector_array("<c16", std::vector<cplx>{{2.0, 0.0}, {-1.0, -0.0}})),
            (std::vector<double>{2.0, -1.0}));
  EXPECT_EQ(hs::npy_to_vector<double>(
                vector_array("<c8", std::vector<std::complex<float>>{{2.0f, 0.0f}})),
            (std::vector<double>{2.0}));
  EXPECT_EQ(error_code_of([] {
              hs::npy_to_vector<double>(vector_array("<c16", std::vector<cplx>{{2.0, 0.0}, {1.0, 1e-300}}));
            }),
            "invalid_input");
  EXPECT_EQ(error_code_of([] { hs::npy_to_vector<double>(array_of("|S3", {1}, bytes_of(std::vector<char>{'c', 's', 'r'}))); }),
            "invalid_input");
  EXPECT_EQ(error_code_of([] { hs::npy_to_vector<double>(vector_array("<f2", std::vector<std::uint16_t>{0})); }),
            "invalid_input");
}

TEST(IoNpyConvert, ToComplexWidensEverything) {
  EXPECT_EQ(hs::npy_to_vector<cplx>(vector_array("<c8", std::vector<std::complex<float>>{{0.5f, -0.25f}})),
            (std::vector<cplx>{{0.5, -0.25}}));
  EXPECT_EQ(hs::npy_to_vector<cplx>(vector_array("<f8", kTwo)), (std::vector<cplx>{{1.5, 0.0}, {-2.0, 0.0}}));
  EXPECT_EQ(hs::npy_to_vector<cplx>(vector_array("<i4", std::vector<std::int32_t>{-9})),
            (std::vector<cplx>{{-9.0, 0.0}}));
  const std::vector<cplx> z{{1.0, -2.0}, {3e-300, 1e300}};
  EXPECT_EQ(hs::npy_to_vector<cplx>(array_of(">c16", {2}, big_endian_bytes(z, 8))), z);
}

TEST(IoNpyConvert, ToFloatOnlyWithoutPrecisionLoss) {
  EXPECT_EQ(hs::npy_to_vector<float>(vector_array("<f4", std::vector<float>{1.25f})),
            (std::vector<float>{1.25f}));
  EXPECT_EQ(hs::npy_to_vector<float>(vector_array("<i2", std::vector<std::int16_t>{-300})),
            (std::vector<float>{-300.0f}));
  EXPECT_EQ(hs::npy_to_vector<float>(vector_array("<u2", std::vector<std::uint16_t>{65535})),
            (std::vector<float>{65535.0f}));
  EXPECT_EQ(hs::npy_to_vector<float>(vector_array("<c8", std::vector<std::complex<float>>{{3.0f, 0.0f}})),
            (std::vector<float>{3.0f}));
  EXPECT_EQ(hs::npy_to_vector<float>(array_of(">f4", {1}, big_endian_bytes(std::vector<float>{-0.5f}, 4))),
            (std::vector<float>{-0.5f}));
  for (const auto& a : {vector_array("<f8", kTwo), vector_array("<i4", std::vector<std::int32_t>{1}),
                        vector_array("<i8", std::vector<std::int64_t>{1}),
                        vector_array("<c16", std::vector<cplx>{{1.0, 0.0}}),
                        vector_array("<c8", std::vector<std::complex<float>>{{1.0f, 1.0f}})}) {
    EXPECT_EQ(error_code_of([&] { hs::npy_to_vector<float>(a); }), "invalid_input") << a.dtype;
  }
}

TEST(IoNpyConvert, ToInt32ChecksRanges) {
  using V32 = std::vector<std::int32_t>;
  EXPECT_EQ(hs::npy_to_vector<std::int32_t>(vector_array("<i8", std::vector<std::int64_t>{-5, 1LL << 30})),
            (V32{-5, 1 << 30}));
  EXPECT_EQ(hs::npy_to_vector<std::int32_t>(vector_array("|u1", std::vector<std::uint8_t>{255})), (V32{255}));
  EXPECT_EQ(hs::npy_to_vector<std::int32_t>(vector_array("<u4", std::vector<std::uint32_t>{7})), (V32{7}));
  EXPECT_EQ(hs::npy_to_vector<std::int32_t>(vector_array("|b1", std::vector<std::uint8_t>{1})), (V32{1}));
  EXPECT_EQ(hs::npy_to_vector<std::int32_t>(
                array_of(">i4", {1}, big_endian_bytes(std::vector<std::int32_t>{-123456}, 4))),
            (V32{-123456}));
  for (const auto& a : {vector_array("<i8", std::vector<std::int64_t>{1LL << 31}),
                        vector_array("<i8", std::vector<std::int64_t>{-(1LL << 31) - 1}),
                        vector_array("<u4", std::vector<std::uint32_t>{1u << 31}),
                        vector_array("<u8", std::vector<std::uint64_t>{~0ULL}),
                        vector_array("<f8", std::vector<double>{1.0}),
                        vector_array("<c16", std::vector<cplx>{{1.0, 0.0}})}) {
    EXPECT_EQ(error_code_of([&] { hs::npy_to_vector<std::int32_t>(a); }), "invalid_input") << a.dtype;
  }
}

TEST(IoNpyConvert, ToInt64ChecksRanges) {
  using V64 = std::vector<std::int64_t>;
  EXPECT_EQ(hs::npy_to_vector<std::int64_t>(vector_array("<i4", std::vector<std::int32_t>{-7})), (V64{-7}));
  EXPECT_EQ(hs::npy_to_vector<std::int64_t>(vector_array("<u4", std::vector<std::uint32_t>{~0u})),
            (V64{0xFFFFFFFFLL}));
  EXPECT_EQ(hs::npy_to_vector<std::int64_t>(vector_array("<u8", std::vector<std::uint64_t>{1ULL << 62})),
            (V64{1LL << 62}));
  EXPECT_EQ(hs::npy_to_vector<std::int64_t>(
                array_of(">i8", {1}, big_endian_bytes(std::vector<std::int64_t>{-(1LL << 40)}, 8))),
            (V64{-(1LL << 40)}));
  EXPECT_EQ(error_code_of([] { hs::npy_to_vector<std::int64_t>(vector_array("<u8", std::vector<std::uint64_t>{1ULL << 63})); }),
            "invalid_input");
  EXPECT_EQ(error_code_of([] { hs::npy_to_vector<std::int64_t>(vector_array("<f4", std::vector<float>{1.0f})); }),
            "invalid_input");
}

TEST(IoNpyConvert, RejectsPayloadSizeMismatch) {
  NpyArray a = vector_array("<f8", kTwo);
  a.shape = {3};
  EXPECT_EQ(error_code_of([&] { hs::npy_to_vector<double>(a); }), "invalid_input");
}

// --------------------------------------------------------------------------------- writers

class IoNpyWrite : public hs::io_test::TempDirTest {};

template <class T>
T json_scalar(const json& c, std::size_t k) {
  if constexpr (hs::is_complex_v<T>) {
    return T(c["re"][k].get<double>(), c["im"][k].get<double>());
  } else {
    return c["re"][k].get<T>();
  }
}

template <class T>
void write_golden_vector(const std::filesystem::path& out, const json& c) {
  std::vector<T> v(c["re"].size());
  for (std::size_t k = 0; k < v.size(); ++k) v[k] = json_scalar<T>(c, k);
  hs::write_npy_vector(out, v);
}

template <class T>
void write_golden_matrix(const std::filesystem::path& out, const json& c) {
  hs::DenseMatrix<T> m(c["rows"].get<hs::index_t>(), c["cols"].get<hs::index_t>());
  for (std::size_t k = 0; k < c["re"].size(); ++k) m.data()[k] = json_scalar<T>(c, k);
  hs::write_npy_matrix(out, m);
}

TEST_F(IoNpyWrite, MatchesNumpyByteForByte) {
  ASSERT_FALSE(expected()["golden"].empty());
  for (const auto& c : expected()["golden"]) {
    const std::string file = c["file"], type = c["type"];
    SCOPED_TRACE(file);
    const auto out = tmp(file);
    if (c["kind"] == "vector") {
      if (type == "float64") write_golden_vector<double>(out, c);
      if (type == "complex128") write_golden_vector<cplx>(out, c);
      if (type == "float32") write_golden_vector<float>(out, c);
      if (type == "int32") write_golden_vector<std::int32_t>(out, c);
      if (type == "int64") write_golden_vector<std::int64_t>(out, c);
    } else {
      if (type == "float64") write_golden_matrix<double>(out, c);
      if (type == "complex128") write_golden_matrix<cplx>(out, c);
    }
    EXPECT_EQ(read_bytes(out), read_bytes(fixture(file)));
  }
}

TEST_F(IoNpyWrite, RoundTripsRawArrays) {
  NpyArray a = array_of(">i2", {2, 2}, {});
  a.fortran_order = true;
  for (int k = 0; k < 8; ++k) a.data.push_back(static_cast<std::byte>(k * 17));
  hs::write_npy(tmp("raw.npy"), a);
  const NpyArray b = hs::read_npy(tmp("raw.npy"));
  EXPECT_EQ(b.dtype, a.dtype);
  EXPECT_EQ(b.shape, a.shape);
  EXPECT_TRUE(b.fortran_order);
  EXPECT_EQ(b.data, a.data);
  EXPECT_EQ(read_bytes(tmp("raw.npy")).size() % 64, 8u);  // header padded to 64, then 8 bytes
}

TEST_F(IoNpyWrite, SwitchesToVersion2ForHugeHeaders) {
  // 30000 unit dimensions make a ~90 KB header: too long for the 16-bit length of format 1.0.
  NpyArray a = array_of("<f8", std::vector<std::size_t>(30000, 1), bytes_of(std::vector<double>{2.5}));
  hs::write_npy(tmp("v2.npy"), a);
  const auto raw = read_bytes(tmp("v2.npy"));
  EXPECT_EQ(raw[6], 2);
  EXPECT_EQ((raw.size() - 8) % 64, 0u);
  const NpyArray b = hs::read_npy(tmp("v2.npy"));
  EXPECT_EQ(b.shape, a.shape);
  EXPECT_EQ(hs::npy_to_vector<double>(b), std::vector<double>{2.5});
}

TEST_F(IoNpyWrite, RejectsInconsistentArraysAndUnwritablePaths) {
  NpyArray a = vector_array("<f8", kTwo);
  a.shape = {3};
  EXPECT_EQ(error_code_of([&] { hs::write_npy(tmp("bad.npy"), a); }), "invalid_input");
  a.shape = {2};
  a.dtype = "<q8";
  EXPECT_EQ(error_code_of([&] { hs::write_npy(tmp("bad.npy"), a); }), "io");
  EXPECT_EQ(error_code_of([&] { hs::write_npy_vector(tmp("missing_dir") / "x.npy", kTwo); }), "io");
}

// ------------------------------------------------------------------------------- npz / zip

TEST(IoNpz, ReadsNumpyArchivesAndSkipsNonArrayMembers) {
  ASSERT_FALSE(expected()["npz"].empty());
  for (const auto& c : expected()["npz"]) {
    const std::string file = c["file"];
    SCOPED_TRACE(file);
    const auto arrays = hs::read_npz(fixture(file));
    std::vector<std::string> keys;
    for (const auto& kv : arrays) keys.push_back(kv.first);
    EXPECT_EQ(keys, c["keys"].get<std::vector<std::string>>());
    for (const auto& [key, want] : c["arrays"].items()) {
      SCOPED_TRACE(key);
      check_array(arrays.at(key), want);
    }
  }
}

TEST(IoNpz, ArchiveVariantsHoldTheSameArrays) {
  // zip64 records, comments, bytes appended after the archive (also after a comment, and holding
  // a stray end-record signature) and the hand-built writer of make_io_fixtures.py.
  const auto ref = hs::read_npz(fixture("csr_matrix_f8.npz"));
  for (const char* file : {"zip64_deflate.npz", "zip64_saturated_eocd.npz", "comment_with_signature.npz",
                           "trailing_bytes.npz", "comment_then_bytes.npz", "trailing_signature.npz",
                           "crafted_ok.npz"}) {
    SCOPED_TRACE(file);
    const auto got = hs::read_npz(fixture(file));
    ASSERT_EQ(got.size(), ref.size());
    for (const auto& [key, a] : ref) {
      const NpyArray& b = got.at(key);
      EXPECT_EQ(b.dtype, a.dtype) << key;
      EXPECT_EQ(b.shape, a.shape) << key;
      EXPECT_EQ(b.data, a.data) << key;
    }
  }
}

TEST(IoNpz, DetectsCorruptMembers) {
  // CRC, deflate streams longer / shorter than declared or cut, duplicate members, multi-volume
  // archives, saturated sizes without a zip64 extra field: each error pins its message.
  ASSERT_FALSE(expected()["errors"].empty());
  for (const auto& c : expected()["errors"]) {
    const std::string file = c["file"];
    SCOPED_TRACE(file);
    hs::io_test::expect_error(c, [&] { hs::read_npz(fixture(file)); });
  }
}

TEST(IoNpzFile, ReadAsCoversEveryElementType) {
  // read_as<T> for every instantiated T: straight into typed storage when the dtype matches,
  // converted otherwise.
  const hs::io::NpzFile f8(fixture("csr_matrix_f8.npz"));
  const hs::io::NpzFile f4(fixture("csr_f4.npz"));
  const hs::io::NpzFile i64(fixture("csr_int64_indices_f8.npz"));
  const std::vector<double> data = f8.read_as<double>("data");
  ASSERT_EQ(data.size(), 14u);
  hs::io::NpyHeader h;
  const std::vector<float> data_f4 = f4.read_as<float>("data", &h);  // values exact in float
  EXPECT_EQ(h.dtype, "<f4");
  EXPECT_EQ(h.shape, std::vector<std::size_t>{14});
  EXPECT_EQ(std::vector<double>(data_f4.begin(), data_f4.end()), data);
  EXPECT_EQ(f4.read_as<double>("data"), data);
  EXPECT_EQ(f8.read_as<cplx>("data"), std::vector<cplx>(data.begin(), data.end()));
  EXPECT_EQ(error_code_of([&] { f8.read_as<float>("data"); }), "invalid_input");  // f8 -> f4 unsafe

  const std::vector<std::int32_t> indices = f8.read_as<std::int32_t>("indices");
  EXPECT_EQ(f8.read_as<std::int64_t>("indices"),
            std::vector<std::int64_t>(indices.begin(), indices.end()));
  EXPECT_EQ(i64.read_as<std::int32_t>("indices"), indices);
  EXPECT_EQ(i64.read_as<std::int64_t>("indptr"), f8.read_as<std::int64_t>("indptr"));
  EXPECT_EQ(error_code_of([&] { f8.read_as<double>("missing"); }), "io");
}

class IoZip : public hs::io_test::TempDirTest {
 protected:
  // Copy of a fixture with the first central-directory record patched at `offset` (bytes after
  // the record signature) by the little-endian value v of width `width`.
  std::filesystem::path patch_central(const std::string& file, std::size_t offset, std::uint32_t v,
                                      std::size_t width) {
    std::vector<char> raw = read_bytes(fixture(file));
    const std::string sig("PK\x01\x02", 4);
    const auto pos = std::string(raw.begin(), raw.end()).find(sig);
    EXPECT_NE(pos, std::string::npos);
    for (std::size_t b = 0; b < width; ++b)
      raw[pos + sig.size() + offset + b] = static_cast<char>((v >> (8 * b)) & 0xFF);
    const auto out = tmp("patched.npz");
    write_bytes(out, raw);
    return out;
  }
};

TEST_F(IoZip, RejectsTruncatedEmptyAndNonZipFiles) {
  const std::vector<char> raw = read_bytes(fixture("csr_matrix_f8.npz"));
  write_bytes(tmp("half.npz"), std::vector<char>(raw.begin(), raw.begin() + raw.size() / 2));
  EXPECT_EQ(error_code_of([&] { hs::read_npz(tmp("half.npz")); }), "io");
  write_bytes(tmp("cut_tail.npz"), std::vector<char>(raw.begin(), raw.end() - 3));
  EXPECT_EQ(error_code_of([&] { hs::read_npz(tmp("cut_tail.npz")); }), "io");
  write_bytes(tmp("empty.npz"), {});
  EXPECT_EQ(error_code_of([&] { hs::read_npz(tmp("empty.npz")); }), "io");
  EXPECT_EQ(error_code_of([] { hs::read_npz(fixture("dense_c_f8.npy")); }), "io");
  EXPECT_EQ(error_code_of([] { hs::read_npz(fixture("missing.npz")); }), "io");
  EXPECT_EQ(error_code_of([] { hs::read_npz(hs::io_test::fixture_dir()); }), "io");
}

TEST_F(IoZip, RejectsEncryptionAndUnknownCompression) {
  // Central record layout after the signature: versions (4), flags (2), method (2), ...
  EXPECT_EQ(error_code_of([&] { hs::read_npz(patch_central("csr_matrix_f8.npz", 4, 0x0001, 2)); }),
            "unsupported");
  EXPECT_EQ(error_code_of([&] { hs::read_npz(patch_central("csr_matrix_f8.npz", 6, 12, 2)); }),
            "unsupported");  // bzip2
}

TEST_F(IoZip, ValidatesSizesAndOffsetsFromTheCentralDirectory) {
  // Local header offset of the first member moved onto the second member's local header.
  const std::vector<char> raw = read_bytes(fixture("csr_matrix_f8_stored.npz"));
  const auto second = std::string(raw.begin(), raw.end()).find(std::string("PK\x03\x04", 4), 1);
  ASSERT_NE(second, std::string::npos);
  const auto moved =
      patch_central("csr_matrix_f8_stored.npz", 38, static_cast<std::uint32_t>(second), 4);
  EXPECT_EQ(error_code_of([&] { hs::read_npz(moved); }), "io");
  // Uncompressed size of a stored member no longer equal to its compressed size.
  EXPECT_EQ(error_code_of([&] { hs::read_npz(patch_central("csr_matrix_f8_stored.npz", 20, 7, 4)); }),
            "io");
  // Deflated member declared one byte shorter / longer than its content: the npy layer rejects it
  // (payload length vs header) before the zip layer's own stream-length checks run; those are
  // pinned by IoNpz.DetectsCorruptMembers (deflate_longer / deflate_shorter / deflate_cut).
  const auto info = hs::read_npz(fixture("csr_matrix_f8.npz"));
  const std::uint32_t real = 128 + static_cast<std::uint32_t>(info.at("indices").data.size());
  EXPECT_EQ(error_code_of([&] { hs::read_npz(patch_central("csr_matrix_f8.npz", 20, real - 1, 4)); }),
            "io");
  EXPECT_EQ(error_code_of([&] { hs::read_npz(patch_central("csr_matrix_f8.npz", 20, real + 1, 4)); }),
            "io");
  // An uncompressed size beyond what deflate can expand the member to (1032:1) is rejected when
  // the archive is opened, before any reader sizes a buffer from it.
  const auto huge = hs::io_test::error_of(
      [&] { hs::io::NpzFile(patch_central("csr_matrix_f8.npz", 20, 0x7FFFFFFF, 4)); });
  EXPECT_EQ(huge.code, "io");
  EXPECT_NE(huge.message.find("declares 2147483647 uncompressed bytes"), std::string::npos)
      << huge.message;
}

// Little-endian field of a hand-built zip record.
template <class U>
void put_le(std::vector<char>& out, U v) {
  for (std::size_t b = 0; b < sizeof(U); ++b) out.push_back(static_cast<char>((v >> (8 * b)) & 0xFF));
}

// One-member archive with a raw-deflated member (classic records), for archives too large to
// commit; make_io_fixtures.py builds the small hand-made ones.
std::vector<char> deflated_zip(const std::string& name, const std::vector<char>& content) {
  z_stream z{};
  EXPECT_EQ(deflateInit2(&z, Z_BEST_SPEED, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY), Z_OK);
  std::vector<char> packed(deflateBound(&z, static_cast<uLong>(content.size())));
  z.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(content.data()));
  z.avail_in = static_cast<uInt>(content.size());
  z.next_out = reinterpret_cast<Bytef*>(packed.data());
  z.avail_out = static_cast<uInt>(packed.size());
  EXPECT_EQ(deflate(&z, Z_FINISH), Z_STREAM_END);
  packed.resize(z.total_out);
  deflateEnd(&z);
  const auto crc = static_cast<std::uint32_t>(
      crc32(0, reinterpret_cast<const Bytef*>(content.data()), static_cast<uInt>(content.size())));
  const auto name_len = static_cast<std::uint16_t>(name.size());
  // Fields from the flags to the extra length, identical in the local and the central header.
  const auto common = [&](std::vector<char>& out) {
    put_le<std::uint16_t>(out, 0);  // flags
    put_le<std::uint16_t>(out, 8);  // deflate
    put_le<std::uint32_t>(out, 0);  // time, date
    put_le(out, crc);
    put_le(out, static_cast<std::uint32_t>(packed.size()));
    put_le(out, static_cast<std::uint32_t>(content.size()));
    put_le(out, name_len);
    put_le<std::uint16_t>(out, 0);  // extra field length
  };
  std::vector<char> zip;
  put_le<std::uint32_t>(zip, 0x04034b50);
  put_le<std::uint16_t>(zip, 20);  // version needed
  common(zip);
  zip.insert(zip.end(), name.begin(), name.end());
  zip.insert(zip.end(), packed.begin(), packed.end());
  const auto cd_offset = static_cast<std::uint32_t>(zip.size());
  put_le<std::uint32_t>(zip, 0x02014b50);
  put_le<std::uint16_t>(zip, 20);  // version made by
  put_le<std::uint16_t>(zip, 20);  // version needed
  common(zip);
  put_le<std::uint16_t>(zip, 0);  // comment length
  put_le<std::uint16_t>(zip, 0);  // disk
  put_le<std::uint16_t>(zip, 0);  // internal attributes
  put_le<std::uint32_t>(zip, 0);  // external attributes
  put_le<std::uint32_t>(zip, 0);  // local header offset
  zip.insert(zip.end(), name.begin(), name.end());
  const auto cd_size = static_cast<std::uint32_t>(zip.size()) - cd_offset;
  put_le<std::uint32_t>(zip, 0x06054b50);
  put_le<std::uint32_t>(zip, 0);  // this disk, central directory disk
  put_le<std::uint16_t>(zip, 1);  // entries on this disk
  put_le<std::uint16_t>(zip, 1);  // entries
  put_le(zip, cd_size);
  put_le(zip, cd_offset);
  put_le<std::uint16_t>(zip, 0);  // comment length
  return zip;
}

TEST_F(IoZip, InflatesMembersLargerThanTheInputChunk) {
  // Random doubles barely compress: the member's deflate data exceeds the reader's 1 MiB input
  // buffer, which is refilled mid-member (and mid-array for the typed read).
  std::vector<double> v(200000);
  hs::linalg::fill_random(v.data(), static_cast<hs::index_t>(v.size()), 2026);
  hs::write_npy_vector(tmp("v.npy"), v);
  std::vector<char> zip = deflated_zip("v.npy", read_bytes(tmp("v.npy")));
  ASSERT_GT(zip.size(), (std::size_t{1} << 20) + 4096);
  write_bytes(tmp("big.npz"), zip);
  EXPECT_EQ(hs::npy_to_vector<double>(hs::read_npz(tmp("big.npz")).at("v")), v);
  EXPECT_EQ(hs::io::NpzFile(tmp("big.npz")).read_as<double>("v"), v);
  // Corruption behind the first input chunk is still detected.
  zip[(std::size_t{1} << 20) + 1000] ^= 0x55;
  write_bytes(tmp("bad.npz"), zip);
  EXPECT_EQ(error_code_of([&] { hs::read_npz(tmp("bad.npz")); }), "io");
}

}  // namespace
