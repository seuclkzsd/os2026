// load_matrix: format readers (scipy npz, dense npy, MatrixMarket) feeding one post-processing
// pipeline (include/hsolve/io.hpp).
//
// Memory, as peak / final CSR (8(n+1) + 12 nnz bytes real, + 20 nnz complex), measured by
// bench/memio_load.cpp (n = 343000, nnz = 9.0e6, real / complex):
//   * npz csr: every array is inflated straight into the CSR vectors, one member at a time;
//     non-native dtypes (f4/c8 values, int64 indices) are converted in 1 MiB chunks -> 1.02x.
//   * npz csc / coo: the row indices are held, the column indices and then the values (streamed
//     from the archive) are placed into their CSR slots -> 1.36x / 1.22x (was 2.0x / 2.35x).
//   * MatrixMarket coordinate: two passes over the file (row counts, then placement) -> ~1.0x
//     plus the 4 MiB line buffer (was 2.5x); about 30% slower than one pass.
//   * One stored triangle: symmetrize_from_triangle(&&) -> 1.22x / 1.34x (was 2.5x).
//   * Hermitian defect > 0: hermitian_part(&&), in place for a structurally symmetric pattern
//     (no extra memory), otherwise 1.70x / 1.82x (was 3.0x).
//   * Complex input with zero imaginary parts: + the real values while converting (<= 1.4x).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"
#include "io/csr_assembly.hpp"
#include "io/mtx.hpp"
#include "io/npy_format.hpp"
#include "io/npz.hpp"

namespace hs {
namespace {

constexpr std::uint64_t kMaxOrder = std::uint64_t{1} << 31;  // CSR column indices are int32

using AnyCsr = std::variant<CsrMatrix<double>, CsrMatrix<cplx>>;

// What a format reader hands to the common pipeline.
struct RawMatrix {
  AnyCsr matrix;
  std::string format;
  io::MtxSymmetry declared = io::MtxSymmetry::General;  ///< MatrixMarket banner (else General)
  std::vector<std::string> notes;
};

void note_assembly(const io::AssemblyStats& s, std::vector<std::string>& notes) {
  if (s.duplicates_summed > 0)
    notes.push_back("duplicates: summed " + std::to_string(s.duplicates_summed) +
                    " duplicate entries");
  if (s.zeros_dropped > 0)
    notes.push_back("zeros: dropped " + std::to_string(s.zeros_dropped) + " explicit zero entries");
}

// Runs f; an hs::Error escaping it gets the file name prepended (for helpers that do not know it).
template <class F>
auto naming_source(const std::string& source, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const Error& e) {
    throw Error(e.code(), source + ": " + e.what());
  }
}

index_t checked_order(std::uint64_t rows, std::uint64_t cols, const std::string& source) {
  if (rows != cols)
    HS_THROW(InvalidInput, source << ": matrix is " << rows << " x " << cols
                                  << "; only square matrices can be loaded");
  if (rows > kMaxOrder)
    HS_THROW(InvalidInput, source << ": order " << rows << " exceeds the int32 index range");
  return static_cast<index_t>(rows);
}

// ------------------------------------------------------------------------ scipy.sparse npz

// scipy writes format.encode('ascii') (|S3); np.savez of a Python str gives <U3.
std::string decode_format(const NpyArray& a, const std::string& source) {
  const io::DtypeInfo info = io::parse_dtype(a.dtype);
  if (a.size() != 1)
    HS_THROW(InvalidInput, source << ": 'format' must hold a single string");
  std::string s;
  if (info.kind == 'S') {
    s.assign(reinterpret_cast<const char*>(a.data.data()), a.data.size());
  } else if (info.kind == 'U') {
    for (std::size_t k = 0; k + 4 <= a.data.size(); k += 4) {
      std::uint32_t cp = io::load_le<std::uint32_t>(a.data.data() + k);
      if (info.big_endian()) io::byte_swap_elements(reinterpret_cast<std::byte*>(&cp), 1, 4);
      if (cp > 0x7F) HS_THROW(InvalidInput, source << ": 'format' is not an ASCII string");
      s += static_cast<char>(cp);
    }
  } else {
    HS_THROW(InvalidInput, source << ": 'format' has dtype '" << a.dtype
                                  << "', expected bytes or str");
  }
  s.resize(std::strlen(s.c_str()));  // numpy pads fixed-width strings with NULs
  return s;
}

template <class V>
void release(std::vector<V>& v) {
  std::vector<V>().swap(v);
}

// Coordinate (coo) and compressed-column (csc) files are assembled by placing entries straight
// into their CSR slots (io::RowSlots): first the column indices, then the values, which are
// streamed from the archive and never held as a whole. Peak: row indices + result (+ the column
// indices while the CSR pattern is built), ~1.33x the result for real and ~1.2x for complex
// values, instead of input + result (coo ~2.3x) or CSC + transpose (csc ~2x).

// Streams `key` into m.values at the slots handed out for `rows` (in stream order); entries at or
// behind `count` (unused csc storage) are skipped.
template <class T>
void scatter_values(const io::NpzFile& npz, const std::string& key, const std::vector<std::int32_t>& rows,
                    std::size_t count, io::RowSlots& slots, CsrMatrix<T>& m) {
  m.values.resize(count);
  slots.rewind();
  npz.read_chunks<T>(key, [&](const T* v, std::size_t first, std::size_t len) {
    const std::size_t last = std::min(first + len, count);
    for (std::size_t k = first; k < last; ++k) m.values[slots.next(rows[k])] = v[k - first];
  });
}

template <class T>
CsrMatrix<T> read_scipy_coo(const io::NpzFile& npz, index_t n, bool drop_zeros,
                            io::AssemblyStats& stats) {
  const std::string source = npz.path().string();
  std::vector<std::int32_t> rows, cols;
  if (npz.contains("row") && npz.contains("col")) {
    rows = npz.read_as<std::int32_t>("row");
    cols = npz.read_as<std::int32_t>("col");
  } else if (npz.contains("coords")) {  // n-D coo arrays (and future 2-D ones): shape (ndim, nnz)
    const io::NpyHeader h = npz.header("coords");
    if (h.shape.size() != 2 || h.shape[0] != 2)
      HS_THROW(InvalidInput, source << ": 'coords' must have shape (2, nnz)");
    const std::size_t nnz = h.shape[1];
    rows.resize(nnz);
    cols.resize(nnz);
    // Element (d, k) is at d*nnz + k in C order and at 2k + d in Fortran order.
    npz.read_chunks<std::int32_t>("coords", [&](const std::int32_t* v, std::size_t first, std::size_t len) {
      for (std::size_t e = first; e < first + len; ++e) {
        const std::int32_t x = v[e - first];
        if (h.fortran_order) {
          (e % 2 == 0 ? rows : cols)[e / 2] = x;
        } else {
          (e < nnz ? rows[e] : cols[e - nnz]) = x;
        }
      }
    });
  } else {
    HS_THROW(InvalidInput, source << ": coo npz has neither 'row'/'col' nor 'coords'");
  }
  const std::size_t nnz = npz.header("data").count;
  if (rows.size() != nnz || cols.size() != nnz)
    HS_THROW(InvalidInput, source << ": coo arrays differ in length (row " << rows.size()
                                  << ", col " << cols.size() << ", data " << nnz << ")");
  naming_source(source, [&] { io::check_coordinates(n, n, rows.data(), cols.data(), nnz); });

  CsrMatrix<T> m;
  m.nrows = m.ncols = n;
  io::RowSlots slots(n, rows.data(), nnz);
  m.colind.resize(nnz);
  for (std::size_t k = 0; k < nnz; ++k) m.colind[slots.next(rows[k])] = cols[k];
  release(cols);
  scatter_values(npz, "data", rows, nnz, slots, m);
  release(rows);
  m.rowptr = slots.take_rowptr();
  stats = naming_source(source, [&] { return io::canonicalize_csr(m, drop_zeros); });
  return m;
}

// csc: column pointers, row indices and values of A in column order. Visiting the columns in
// order (which is stream order) appends each column index to its row, so every CSR row comes out
// sorted, with duplicates adjacent in file order; canonicalize_csr then sums them as it would
// have summed them in the file's columns.
template <class T>
CsrMatrix<T> read_scipy_csc(const io::NpzFile& npz, index_t n, bool drop_zeros,
                            io::AssemblyStats& stats) {
  const std::string source = npz.path().string();
  std::vector<index_t> colptr = npz.read_as<index_t>("indptr");
  std::vector<std::int32_t> rows = npz.read_as<std::int32_t>("indices");
  const std::size_t nvals = npz.header("data").count;
  const auto nz = static_cast<std::size_t>(naming_source(
      source, [&] { return io::check_compressed(colptr, n, rows, nvals, n, /*csc=*/true); }));

  CsrMatrix<T> m;
  m.nrows = m.ncols = n;
  io::RowSlots slots(n, rows.data(), nz);
  m.colind.resize(nz);
  for (index_t j = 0; j < n; ++j)
    for (index_t k = colptr[j]; k < colptr[j + 1]; ++k)
      m.colind[slots.next(rows[k])] = static_cast<std::int32_t>(j);
  release(colptr);
  scatter_values(npz, "data", rows, nz, slots, m);
  release(rows);
  m.rowptr = slots.take_rowptr();
  stats = naming_source(source, [&] { return io::canonicalize_csr(m, drop_zeros); });
  return m;
}

template <class T>
CsrMatrix<T> read_scipy_csr(const io::NpzFile& npz, index_t n, bool drop_zeros,
                            io::AssemblyStats& stats) {
  // Every array is read straight into the CSR vectors (converted in chunks if its dtype differs).
  CsrMatrix<T> m;
  m.nrows = m.ncols = n;
  m.rowptr = npz.read_as<index_t>("indptr");
  m.colind = npz.read_as<std::int32_t>("indices");
  m.values = npz.read_as<T>("data");
  stats = naming_source(npz.path().string(), [&] { return io::canonicalize_csr(m, drop_zeros); });
  return m;
}

RawMatrix read_scipy_npz(const std::filesystem::path& path, bool drop_zeros) {
  const io::NpzFile npz(path);
  const std::string source = path.string();
  for (const char* key : {"format", "shape", "data"})
    if (!npz.contains(key))
      HS_THROW(InvalidInput, source << ": not a scipy.sparse npz (no '" << key << "' array)");
  const std::string fmt = decode_format(npz.read("format"), source);
  if (fmt == "bsr" || fmt == "dia")
    HS_THROW(Unsupported, source << ": scipy '" << fmt
                                 << "' matrices are not supported; save as csr, csc or coo");
  if (fmt != "csr" && fmt != "csc" && fmt != "coo")
    HS_THROW(InvalidInput, source << ": unknown scipy sparse format '" << fmt << "'");

  const std::vector<std::int64_t> shape = npz.read_as<std::int64_t>("shape");
  if (shape.size() != 2)
    HS_THROW(InvalidInput, source << ": sparse shape has " << shape.size()
                                  << " dimensions; only 2-D matrices can be loaded");
  if (shape[0] < 0 || shape[1] < 0) HS_THROW(InvalidInput, source << ": negative sparse shape");
  const index_t n = checked_order(static_cast<std::uint64_t>(shape[0]),
                                  static_cast<std::uint64_t>(shape[1]), source);

  const char kind = npz.header("data").info.kind;
  const bool is_cplx = kind == 'c';
  if (!is_cplx && kind != 'f' && kind != 'i' && kind != 'u' && kind != 'b')
    HS_THROW(InvalidInput, source << ": 'data' is not numeric");

  RawMatrix raw;
  raw.format = "npz-" + fmt;
  io::AssemblyStats stats;
  if (fmt == "coo") {
    raw.matrix = is_cplx ? AnyCsr(read_scipy_coo<cplx>(npz, n, drop_zeros, stats))
                         : AnyCsr(read_scipy_coo<double>(npz, n, drop_zeros, stats));
  } else if (fmt == "csc") {
    raw.matrix = is_cplx ? AnyCsr(read_scipy_csc<cplx>(npz, n, drop_zeros, stats))
                         : AnyCsr(read_scipy_csc<double>(npz, n, drop_zeros, stats));
  } else {
    raw.matrix = is_cplx ? AnyCsr(read_scipy_csr<cplx>(npz, n, drop_zeros, stats))
                         : AnyCsr(read_scipy_csr<double>(npz, n, drop_zeros, stats));
  }
  note_assembly(stats, raw.notes);

  std::set<std::string> used{"format", "shape", "data", "_is_array"};  // _is_array: array API
  if (fmt == "coo" && npz.contains("row") && npz.contains("col")) {
    used.insert({"row", "col"});
  } else if (fmt == "coo") {
    used.insert("coords");
  } else {
    used.insert({"indices", "indptr"});
  }
  std::string ignored;
  for (const std::string& key : npz.keys())
    if (used.count(key) == 0) ignored += (ignored.empty() ? "" : ", ") + key;
  if (!ignored.empty())
    raw.notes.push_back("ignored: npz arrays not used by scipy '" + fmt + "' format: " + ignored);
  return raw;
}

// ------------------------------------------------------------------------------ dense npy

template <class T>
CsrMatrix<T> dense_to_csr(const std::vector<T>& v, index_t n, bool fortran_order) {
  // Element (i, j) is at i*n + j in C order and at i + j*n in Fortran order. Exact zeros are
  // structural absence in a dense array.
  const auto at = [&](index_t i, index_t j) -> const T& {
    return v[static_cast<std::size_t>(fortran_order ? i + j * n : i * n + j)];
  };
  CsrMatrix<T> m;
  m.nrows = m.ncols = n;
  m.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (index_t i = 0; i < n; ++i) {
    index_t cnt = 0;
    for (index_t j = 0; j < n; ++j) cnt += at(i, j) != T(0);
    m.rowptr[i + 1] = m.rowptr[i] + cnt;
  }
  m.colind.reserve(static_cast<std::size_t>(m.rowptr[n]));
  m.values.reserve(static_cast<std::size_t>(m.rowptr[n]));
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j)
      if (at(i, j) != T(0)) {
        m.colind.push_back(static_cast<std::int32_t>(j));
        m.values.push_back(at(i, j));
      }
  return m;
}

RawMatrix read_dense_npy(const std::filesystem::path& path) {
  NpyArray a = read_npy(path);
  const std::string source = path.string();
  if (a.shape.size() != 2)
    HS_THROW(InvalidInput, source << ": a dense .npy matrix must be 2-D, got " << a.shape.size()
                                  << " dimensions");
  const index_t n = checked_order(a.shape[0], a.shape[1], source);
  const bool fortran_order = a.fortran_order;
  RawMatrix raw;
  raw.format = "npy-dense";
  if (io::parse_dtype(a.dtype).kind == 'c') {
    const auto v = naming_source(source, [&] { return npy_to_vector<cplx>(a); });
    a = NpyArray{};  // release the raw payload before building the CSR
    raw.matrix = dense_to_csr(v, n, fortran_order);
  } else {
    const auto v = naming_source(source, [&] { return npy_to_vector<double>(a); });
    a = NpyArray{};
    raw.matrix = dense_to_csr(v, n, fortran_order);
  }
  return raw;
}

// ---------------------------------------------------------------------------- MatrixMarket

RawMatrix read_mtx(const std::filesystem::path& path, bool drop_zeros) {
  io::MtxMatrix m = io::read_matrix_market(path, drop_zeros);
  RawMatrix raw{std::move(m.matrix), "mtx", m.symmetry, {}};
  note_assembly(m.stats, raw.notes);
  return raw;
}

// ------------------------------------------------------------------------ common pipeline

const char* storage_name(Storage s) {
  switch (s) {
    case Storage::Auto: return "auto";
    case Storage::Full: return "full";
    case Storage::Upper: return "upper";
    case Storage::Lower: return "lower";
  }
  return "?";
}

const char* symmetry_name(io::MtxSymmetry s) {
  switch (s) {
    case io::MtxSymmetry::General: return "general";
    case io::MtxSymmetry::Symmetric: return "symmetric";
    case io::MtxSymmetry::Hermitian: return "hermitian";
  }
  return "?";
}

struct Triangles {
  bool lower = false;  ///< some entry strictly below the diagonal
  bool upper = false;  ///< some entry strictly above the diagonal
};

template <class T>
Triangles strict_triangles(const CsrMatrix<T>& a) {
  bool lower = false, upper = false;
#pragma omp parallel for reduction(|| : lower, upper) schedule(static)
  for (index_t i = 0; i < a.nrows; ++i) {
    const index_t b = a.rowptr[i], e = a.rowptr[i + 1];
    if (b == e) continue;
    lower = lower || a.colind[b] < i;      // rows are sorted: first entry is the leftmost
    upper = upper || a.colind[e - 1] > i;  // and the last the rightmost
  }
  return {lower, upper};
}

// Which stored triangle is mirrored (Full: none). Throws InvalidInput on contradictions.
Storage resolve_storage(Triangles t, Storage requested, io::MtxSymmetry declared,
                        const std::string& source) {
  if (declared != io::MtxSymmetry::General) {
    // A symmetric/hermitian MatrixMarket file stores one triangle by definition.
    if (requested == Storage::Full)
      HS_THROW(InvalidInput, source << ": options.storage = full contradicts the MatrixMarket '"
                                    << symmetry_name(declared)
                                    << "' banner, which declares one stored triangle");
    if (t.lower && t.upper)
      HS_THROW(InvalidInput, source << ": MatrixMarket '" << symmetry_name(declared)
                                    << "' file stores entries in both triangles");
  }
  switch (requested) {
    case Storage::Full:
      return Storage::Full;
    case Storage::Upper:
      if (t.lower)
        HS_THROW(InvalidInput, source << ": options.storage = upper but the strict lower "
                                         "triangle holds entries");
      return Storage::Upper;
    case Storage::Lower:
      if (t.upper)
        HS_THROW(InvalidInput, source << ": options.storage = lower but the strict upper "
                                         "triangle holds entries");
      return Storage::Lower;
    case Storage::Auto:
      if (t.upper && !t.lower) return Storage::Upper;
      if (t.lower && !t.upper) return Storage::Lower;
      // Both triangles: full storage. Diagonal only: nothing to mirror; a MatrixMarket
      // symmetric file still declares its conventional lower storage.
      return declared != io::MtxSymmetry::General && !t.lower ? Storage::Lower : Storage::Full;
  }
  return Storage::Full;
}

template <class T>
bool is_finite(const T& v) {
  if constexpr (is_complex_v<T>) {
    return std::isfinite(v.real()) && std::isfinite(v.imag());
  } else {
    return std::isfinite(v);
  }
}

template <class T>
void check_finite(const CsrMatrix<T>& a, const std::string& source) {
  const index_t nz = a.nnz();
  index_t bad = nz;
#pragma omp parallel for reduction(min : bad) schedule(static)
  for (index_t p = 0; p < nz; ++p)
    if (!is_finite(a.values[p])) bad = std::min(bad, p);
  if (bad == nz) return;
  const auto row = std::upper_bound(a.rowptr.begin(), a.rowptr.end(), bad) - a.rowptr.begin() - 1;
  HS_THROW(InvalidInput, source << ": non-finite value " << a.values[bad] << " at (" << row << ", "
                                << a.colind[bad] << ")");
}

std::string format_double(double v) {
  std::ostringstream os;
  os.precision(3);
  os << v;
  return os.str();
}

template <class T>
void finish(CsrMatrix<T>&& a, io::MtxSymmetry declared, const LoadOptions& opt,
            const std::string& source, LoadedMatrix& out) {
  HS_ASSERT(a.square(), source << ": readers must deliver square matrices");
  check_finite(a, source);

  // 1. Triangular storage.
  const Triangles tri = strict_triangles(a);
  const Storage stored = resolve_storage(tri, opt.storage, declared, source);
  out.detected_storage = stored;
  if (stored != Storage::Full && (tri.lower || tri.upper)) {
    a = symmetrize_from_triangle(std::move(a));
    std::string why;
    if (declared != io::MtxSymmetry::General) {
      why = std::string("declared by the MatrixMarket '") + symmetry_name(declared) + "' banner";
    } else if (opt.storage == Storage::Auto) {
      why = "auto-detected: the other strict triangle is empty";
    } else {
      why = "declared by options.storage";
    }
    if constexpr (is_complex_v<T>) {
      // MatrixMarket complex *symmetric* means A = A^T: undo the conjugation of the mirrored
      // triangle. The Hermitian check below then rejects it unless every entry is real.
      if (declared == io::MtxSymmetry::Symmetric) {
        const bool mirrored_upper = stored == Storage::Lower;
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < a.nrows; ++i)
          for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
            if (mirrored_upper ? a.colind[p] > i : a.colind[p] < i)
              a.values[p] = std::conj(a.values[p]);
        why += ", complex symmetric: mirrored without conjugation";
      }
    }
    out.notes.push_back(std::string("storage: ") + storage_name(stored) +
                        " triangle mirrored to full Hermitian storage (" + why + ")");
  }

  // 2. Hermitian check; a nonzero defect within tolerance is removed so that every module (one
  // triangle for the factorization, full matrix for operators/residuals) sees the same matrix.
  // Not hs::hermitian_defect: its unscaled sums of squares report 0 for any matrix with entries
  // below ~1e-154 and NaN above ~1e154 (see io::hermitian_defect_scaled). Whether the Hermitian
  // part is taken is decided by the exact value comparison, never by the computed norm.
  if (!opt.check_hermitian) {
    out.hermitian_defect = -1.0;
    out.notes.push_back("unchecked: Hermitian check skipped (check_hermitian = false)");
  } else {
    const io::HermitianDefect defect = io::hermitian_defect_scaled(a);
    out.hermitian_defect = defect.relative;
    if (!(defect.relative <= opt.hermitian_tol))
      HS_THROW(NotHermitian, source << ": relative Hermitian defect ||A - A^H||_F / ||A||_F = "
                                    << defect.relative
                                    << " exceeds hermitian_tol = " << opt.hermitian_tol);
    if (!defect.exact) {
      a = hermitian_part(std::move(a));  // in place when the pattern is structurally symmetric
      out.notes.push_back("hermitian: relative defect " + format_double(defect.relative) +
                          " <= hermitian_tol; A replaced by its Hermitian part (A + A^H)/2");
    }
  }

  // 3. Complex input that is really real.
  if constexpr (is_complex_v<T>) {
    bool all_real = true;
#pragma omp parallel for reduction(&& : all_real) schedule(static)
    for (index_t p = 0; p < a.nnz(); ++p) all_real = all_real && a.values[p].imag() == 0.0;
    if (all_real) {
      CsrMatrix<double> r;
      r.nrows = a.nrows;
      r.ncols = a.ncols;
      r.rowptr = std::move(a.rowptr);
      r.colind = std::move(a.colind);
      r.values.resize(a.values.size());
#pragma omp parallel for schedule(static)
      for (index_t p = 0; p < r.nnz(); ++p) r.values[p] = a.values[p].real();
      std::vector<cplx>().swap(a.values);
      out.matrix = std::move(r);
      out.notes.push_back("real: complex input has all-zero imaginary parts; returned as real");
      return;
    }
  }
  out.matrix = std::move(a);
}

std::string lowercase_extension(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

}  // namespace

index_t LoadedMatrix::order() const {
  return std::visit([](const auto& m) { return m.nrows; }, matrix);
}

LoadedMatrix load_matrix(const std::filesystem::path& path, const LoadOptions& options) {
  if (!(options.hermitian_tol >= 0.0))
    HS_THROW(InvalidInput, "load_matrix: hermitian_tol must be >= 0, got " << options.hermitian_tol);
  const std::string ext = lowercase_extension(path);
  RawMatrix raw;
  if (ext == ".npz") {
    raw = read_scipy_npz(path, options.drop_zeros);
  } else if (ext == ".npy") {
    raw = read_dense_npy(path);
  } else if (ext == ".mtx") {
    raw = read_mtx(path, options.drop_zeros);
  } else {
    HS_THROW(InvalidInput, path.string() << ": unsupported matrix file extension '" << ext
                                         << "' (expected .npz, .npy or .mtx)");
  }
  LoadedMatrix out;
  out.source_format = std::move(raw.format);
  out.notes = std::move(raw.notes);
  std::visit([&](auto& a) { finish(std::move(a), raw.declared, options, path.string(), out); },
             raw.matrix);
  return out;
}

}  // namespace hs
