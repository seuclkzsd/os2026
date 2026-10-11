// Reading/writing numpy (.npy/.npz, incl. scipy.sparse.save_npz) and MatrixMarket files
// (module src/io). Scalar sets are stated per function. File formats: docs/FORMATS.md.
//
// Error codes used throughout this header:
//   Io           missing/unreadable file, malformed npy/zip/mtx content, CRC mismatch, truncation;
//   Unsupported  well-formed input this module cannot represent (structured/object dtypes,
//                encrypted or non-deflate zip members, multi-volume zip, scipy bsr/dia);
//   InvalidInput a readable file whose content is not an acceptable matrix (non-square, wrong
//                array kinds, inconsistent storage declaration, unsupported value conversion);
//   NotHermitian the loaded matrix fails the Hermitian check.
// Thread safety: every function is reentrant (no shared state); concurrent calls are safe.
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
/// "|u1", "<u8", "|S3", "<U3", "<M8[ns]", ...; big-endian '>' too) and keeps raw bytes; only
/// conversions (npy_to_vector) reject unsupported types. Structured (list descr) and object ('O')
/// dtypes are rejected with Unsupported.
struct NpyArray {
  std::vector<std::size_t> shape;
  std::string dtype;  ///< numpy descr exactly as in the header
  bool fortran_order = false;
  std::vector<std::byte> data;

  std::size_t size() const;       ///< product of shape (1 for a 0-d array)
  /// Bytes per element derived from dtype ("<U3" holds 3 UCS-4 characters: 12 bytes).
  /// Throws Io for a malformed descr, Unsupported for object/structured dtypes.
  std::size_t item_size() const;
};

/// Parse a .npy file. Supports format versions 1.0/2.0/3.0; the header is parsed as the Python
/// dict literal it is (any spacing, key order, quote style, trailing commas; shape (), (n,),
/// (m, n), ...), with exactly the keys descr/fortran_order/shape as numpy requires. The payload
/// must have exactly the size implied by the header (truncation or trailing bytes throw Io).
/// O(file size); reads the payload directly into the result.
NpyArray read_npy(const std::filesystem::path& path);
/// Parse .npy bytes held in memory (same rules as read_npy). Copies the payload.
NpyArray parse_npy(const std::byte* bytes, std::size_t nbytes);
/// Read every member of an .npz archive (stored or deflated). Member sizes/offsets are taken from
/// the central directory and its zip64 extra field (numpy writes members with force_zip64, so
/// the local headers carry 0xFFFFFFFF); the zip64 end-of-central-directory record is honoured and
/// an archive comment may follow the end record. CRC-32 is verified. Keys drop ".npy"; members
/// whose names do not end in ".npy" are not numpy arrays (np.load returns them as raw bytes) and
/// are skipped. Members are inflated straight into their arrays (no whole-archive buffer).
std::map<std::string, NpyArray> read_npz(const std::filesystem::path& path);

/// Write a .npy byte-identical to np.save of the same array: format 1.0 (2.0 if the header does
/// not fit 65535 bytes), header "{'descr': ..., 'fortran_order': ..., 'shape': (...), }" plus
/// numpy's growth padding, space-padded so that magic + length + header is a multiple of 64 bytes
/// and terminated by '\n'. Throws InvalidInput if data.size() != size() * item_size(), Io on
/// write failure.
void write_npy(const std::filesystem::path& path, const NpyArray& array);
/// Typed convenience writer: shape (n,), little-endian descr ("<f8", "<c16", "<f4", "<i4", "<i8").
/// T in {double, cplx, float, int32_t, int64_t}.
template <class T>
void write_npy_vector(const std::filesystem::path& path, const std::vector<T>& v);
/// Writes the rows x cols column-major buffer as is (no transpose; np.load gives shape
/// (rows, cols)). Byte-identical to np.save(np.asfortranarray(M)): fortran_order is True when
/// rows > 1 and cols > 1; otherwise column- and row-major layouts coincide and numpy (hence this
/// writer) records fortran_order False. T in {double, cplx}.
template <class T>
void write_npy_matrix(const std::filesystem::path& path, const DenseMatrix<T>& m);

/// Convert a numeric NpyArray to a typed vector in payload order (Fortran order when
/// fortran_order), T in {double, cplx, float, int32_t, int64_t}. Byte order ('<', '>', '|', '=')
/// is honoured. Allowed conversions are numpy's "safe" casts (np.can_cast(src, T, 'safe'); e.g.
/// f4->f8, c8->c16, i4->i8, b1/u1/u2/u4->int64, any integer->double, i1/i2/u1/u2->float), plus
/// value-checked narrowings that throw InvalidInput when a value does not fit exactly: complex ->
/// real (every imaginary part must be exactly 0), i8/u4/u8 -> int32 and u8 -> int64 (range).
/// Everything else (float -> int, f8 -> float, i4/i8 -> float, float16, strings, dates, ...)
/// throws InvalidInput. O(size).
template <class T>
std::vector<T> npy_to_vector(const NpyArray& a);

/// How the file stores the matrix. Upper/Lower: only that triangle (plus the diagonal) is stored
/// and the other is implied by Hermitian symmetry.
enum class Storage { Auto, Full, Upper, Lower };

struct LoadOptions {
  /// Auto: a matrix with an empty strict lower triangle and a non-empty strict upper triangle is
  /// taken as upper storage, and vice versa; anything else is Full. Upper/Lower: entries in the
  /// other strict triangle throw InvalidInput. Full: no restoration.
  Storage storage = Storage::Auto;
  double hermitian_tol = 1e-10;  ///< relative ||A - A^H||_F / ||A||_F accepted as Hermitian
  /// false: skip the Hermitian check and the Hermitian-part replacement (the matrix is returned as
  /// stored, hermitian_defect = -1). Intended for tests/diagnostics only.
  bool check_hermitian = true;
  /// Drop entries that are exactly zero after summing duplicates (sparse inputs). Dense inputs
  /// (.npy, MatrixMarket array) always drop exact zeros: there they mark structural absence.
  bool drop_zeros = true;
};

/// A matrix loaded from disk, already restored to full Hermitian storage.
struct LoadedMatrix {
  std::variant<CsrMatrix<double>, CsrMatrix<cplx>> matrix;
  std::string source_format;  ///< "npz-csr", "npz-csc", "npz-coo", "npy-dense", "mtx"
  /// Storage the matrix was restored from: Upper/Lower when one triangle was mirrored (detected,
  /// declared by options.storage or by a MatrixMarket symmetric/hermitian banner), else Full.
  Storage detected_storage = Storage::Full;
  /// ||A - A^H||_F / ||A||_F of the matrix as read (before taking its Hermitian part);
  /// -1 when options.check_hermitian is false.
  double hermitian_defect = 0.0;
  /// Human-readable remarks, each "<tag>: <text>" with tag in {storage, hermitian, real,
  /// duplicates, zeros, ignored, unchecked} (restored triangle, Hermitian part taken, complex
  /// demoted to real, duplicates summed, explicit zeros dropped, unknown npz members ignored,
  /// Hermitian check skipped).
  std::vector<std::string> notes;

  bool is_complex() const noexcept { return matrix.index() == 1; }
  index_t order() const;
};

/// Load .npz (scipy sparse csr/csc/coo, matrix or array API, row/col or coords keys), .npy
/// (dense 2-D, exact zeros dropped) or .mtx (MatrixMarket coordinate real/double/complex/integer/
/// pattern x general/symmetric/hermitian, or array format; skew-symmetric throws InvalidInput) by
/// extension (case-insensitive). npz keys not used by the format (e.g. scipy's `_is_array`) are
/// ignored; bsr/dia throw Unsupported. The matrix must be square with order <= 2^31.
/// Steps: build CSR (sort rows, sum duplicates, drop exact zeros per options) -> reject
/// non-finite values (InvalidInput) -> restore triangular storage (symmetrize_from_triangle; a
/// MatrixMarket symmetric/hermitian banner declares one stored triangle) -> Hermitian check
/// (NotHermitian above hermitian_tol) -> if the defect is > 0 replace A by hermitian_part(A) and
/// record it in notes, so every module sees the same exactly-Hermitian matrix. Complex input
/// whose imaginary parts are all exactly 0 is returned as real (noted). Throws hs::Error.
/// Memory: npz payloads whose dtype matches the result type are inflated straight into the CSR
/// arrays; peak ~1x the result for full storage with zero defect (see src/io/load.cpp).
LoadedMatrix load_matrix(const std::filesystem::path& path, const LoadOptions& options = {});

}  // namespace hs
