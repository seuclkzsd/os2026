// Reading/writing numpy (.npy/.npz, incl. scipy.sparse.save_npz) and MatrixMarket files
// (module src/io). Scalar sets are stated per function. File formats: docs/FORMATS.md.
#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Raw numpy array: shape, type descriptor and the payload as stored on disk (payload order given
/// by fortran_order). parse_npy accepts ANY fixed-size scalar descr ("<f8", "<c16", "<i4", "|b1",
/// "|u1", "<u8", "|S3", "<U3", ...; big-endian '>' too) and keeps raw bytes; only conversions
/// (npy_to_vector) reject unsupported types. Structured/object dtypes are rejected.
struct NpyArray {
  std::vector<std::size_t> shape;
  std::string dtype;  ///< numpy descr exactly as in the header
  bool fortran_order = false;
  std::vector<std::byte> data;

  std::size_t size() const;       ///< product of shape (1 for a 0-d array)
  std::size_t item_size() const;  ///< bytes per element derived from dtype
};

/// Parse a .npy file. Supports format versions 1.0/2.0/3.0. Throws Error::Code::Io.
NpyArray read_npy(const std::filesystem::path& path);
/// Parse .npy bytes held in memory (used for npz members).
NpyArray parse_npy(const std::byte* bytes, std::size_t nbytes);
/// Read every member of an .npz archive (stored or deflated). Member sizes/offsets are taken from
/// the central directory and its zip64 extra field (numpy writes members with force_zip64, so
/// the local headers carry 0xFFFFFFFF). CRC-32 is verified. Keys drop ".npy".
std::map<std::string, NpyArray> read_npz(const std::filesystem::path& path);

/// Write a .npy (format 1.0, or 2.0 if the header needs it).
void write_npy(const std::filesystem::path& path, const NpyArray& array);
/// Typed convenience writers. write_npy_vector: T in {double, cplx, float, int32_t, int64_t}.
template <class T>
void write_npy_vector(const std::filesystem::path& path, const std::vector<T>& v);
/// Writes rows x cols with fortran_order = True (no transpose; np.load gives shape (rows, cols)).
/// T in {double, cplx}.
template <class T>
void write_npy_matrix(const std::filesystem::path& path, const DenseMatrix<T>& m);

/// Convert a numeric NpyArray to a typed vector, T in {double, cplx, float, int32_t, int64_t}:
/// widening conversions (f4->f8, c8->c16, i4->i8, bool/u1/u4->int, ints->float) and byte swaps.
/// Complex -> real throws unless every imaginary part is exactly 0; int64 -> int32 throws on
/// overflow; float -> int throws.
template <class T>
std::vector<T> npy_to_vector(const NpyArray& a);

enum class Storage { Auto, Full, Upper, Lower };

struct LoadOptions {
  Storage storage = Storage::Auto;
  double hermitian_tol = 1e-10;  ///< relative ||A - A^H||_F / ||A||_F accepted as Hermitian
  bool check_hermitian = true;
  bool drop_zeros = true;
};

/// A matrix loaded from disk, already restored to full Hermitian storage.
struct LoadedMatrix {
  std::variant<CsrMatrix<double>, CsrMatrix<cplx>> matrix;
  std::string source_format;  ///< "npz-csr", "npz-csc", "npz-coo", "npy-dense", "mtx"
  Storage detected_storage = Storage::Full;
  double hermitian_defect = 0.0;
  std::vector<std::string> notes;  ///< human-readable remarks (e.g. storage auto-detection)

  bool is_complex() const noexcept { return matrix.index() == 1; }
  index_t order() const;
};

/// Load .npz (scipy sparse csr/csc/coo, matrix or array API), .npy (dense 2-D) or .mtx by
/// extension. npz keys not used by the format (e.g. scipy's `_is_array`) are ignored; bsr/dia
/// throw Unsupported. Steps: build CSR (sum duplicates) -> restore triangular storage
/// (symmetrize_from_triangle) -> Hermitian check (NotHermitian above hermitian_tol) -> if the
/// defect is > 0 replace A by hermitian_part(A) and record it in notes, so every module sees the
/// same exactly-Hermitian matrix. Complex input whose imaginary parts are all exactly 0 is
/// returned as real (noted). Throws hs::Error.
LoadedMatrix load_matrix(const std::filesystem::path& path, const LoadOptions& options = {});

}  // namespace hs
