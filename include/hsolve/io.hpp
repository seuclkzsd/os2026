// Reading/writing numpy (.npy/.npz, incl. scipy.sparse.save_npz) and MatrixMarket files.
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

/// Raw numpy array: shape, byte order/type descriptor and the payload as stored on disk
/// (little-endian; payload order given by fortran_order).
struct NpyArray {
  std::vector<std::size_t> shape;
  std::string dtype;  ///< numpy descr, e.g. "<f8", "<c16", "<i4", "|S3"
  bool fortran_order = false;
  std::vector<std::byte> data;

  std::size_t size() const;       ///< product of shape (1 for a 0-d array)
  std::size_t item_size() const;  ///< bytes per element derived from dtype
};

/// Parse a .npy file. Supports format versions 1.0/2.0/3.0. Throws Error::Code::Io.
NpyArray read_npy(const std::filesystem::path& path);
/// Parse .npy bytes held in memory (used for npz members).
NpyArray parse_npy(const std::byte* bytes, std::size_t nbytes);
/// Read every member of an .npz archive (stored or deflated, zip64 aware). Keys drop ".npy".
std::map<std::string, NpyArray> read_npz(const std::filesystem::path& path);

/// Write a .npy (format 1.0, or 2.0 if the header needs it).
void write_npy(const std::filesystem::path& path, const NpyArray& array);
/// Typed convenience writers (T in {double, cplx, float, int32_t, int64_t}).
template <class T>
void write_npy_vector(const std::filesystem::path& path, const std::vector<T>& v);
/// Writes rows x cols with fortran_order = True (no transpose; np.load gives shape (rows, cols)).
template <class T>
void write_npy_matrix(const std::filesystem::path& path, const DenseMatrix<T>& m);

/// Convert a numeric NpyArray to a typed vector (dtype conversion f4->f8, c8->c16, i4->i8 ...).
/// Converting complex to real throws unless every imaginary part is exactly zero.
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

/// Load .npz (scipy sparse), .npy (dense 2-D) or .mtx by extension. Restores triangular
/// storage, validates the CSR invariants and the Hermitian property. Throws hs::Error.
LoadedMatrix load_matrix(const std::filesystem::path& path, const LoadOptions& options = {});

/// Restore full storage from a triangular half: U + U^H - diag(U) (diag kept once).
template <class T>
CsrMatrix<T> symmetrize_from_triangle(const CsrMatrix<T>& tri);

}  // namespace hs
