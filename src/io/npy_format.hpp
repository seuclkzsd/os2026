// Internal to src/io: the .npy format in streaming form, shared by read_npy, read_npz and the
// scipy-npz loader. Splitting header and payload lets callers inspect the header first (e.g. the
// dtype of scipy's `data` member) and then read a payload whose layout already matches the target
// type straight into typed storage, without holding raw bytes and a converted copy at once.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "hsolve/io.hpp"
#include "io/byte_source.hpp"

namespace hs::io {

/// A parsed numpy descr such as "<f8", "|b1", ">c16", "<U3" or "<M8[ns]".
struct DtypeInfo {
  char byte_order = '|';      ///< '<', '>', '|' or '=' as written ('|' and '=' mean host order)
  char kind = 0;              ///< numpy kind: b i u f c S U V M m ('a' is normalised to 'S')
  std::size_t item_size = 0;  ///< bytes per element
  bool big_endian() const noexcept { return byte_order == '>'; }
};

/// Parses a numpy descr. Throws Io when malformed, Unsupported for object/zero-size dtypes.
DtypeInfo parse_dtype(const std::string& descr);

struct NpyHeader {
  std::vector<std::size_t> shape;
  std::string dtype;
  bool fortran_order = false;
  DtypeInfo info;
  std::size_t count = 0;       ///< number of elements (1 for a 0-d array)
  std::size_t data_bytes = 0;  ///< count * info.item_size
};

/// Reads magic, version and the header dict (versions 1.0/2.0/3.0). Throws Io / Unsupported.
NpyHeader read_npy_header(ByteSource& src);

/// Reads the payload described by `header` as raw bytes. The source must hold exactly
/// header.data_bytes more bytes (otherwise Io: truncated or trailing data); calls src.finish().
NpyArray read_npy_payload(ByteSource& src, NpyHeader header);

/// Reads the payload converted to T (npy_to_vector rules). When the stored layout already is T's
/// little-endian layout the bytes go straight into the result (peak memory 1x instead of 2x).
/// T in {double, cplx, float, int32_t, int64_t}. Calls src.finish().
template <class T>
std::vector<T> read_npy_payload_as(ByteSource& src, const NpyHeader& header);

/// Converts `count` elements stored with layout `info` (descr text `dtype`, for messages) to T
/// following the npy_to_vector rules. T in {double, cplx, float, int32_t, int64_t}.
template <class T>
std::vector<T> convert_elements(const std::byte* data, std::size_t count, const DtypeInfo& info,
                                const std::string& dtype);

}  // namespace hs::io
