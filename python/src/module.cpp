// pybind11 extension hsolve._hsolve: the in-process Python API (docs/PYTHON.md).
//
// The pure-Python package (python/hsolve/__init__.py) normalises the arguments - scipy formats,
// dtypes, configuration overrides - and calls the private functions below, which
//   1. parse the configuration with hs::parse_config (the single validator, as for the CLI),
//   2. release the GIL, convert the borrowed numpy buffers into hs::CsrMatrix (csr_convert.hpp:
//      one pass per matrix, no densification) or load matrix files with hs::load_matrix,
//   3. run hs::solve, and
//   4. hand the eigenvectors to numpy WITHOUT copying (the array owns the C++ buffer).
// hs::Error is translated into the hsolve.HsolveError subclass registered for its code.
//
// Concurrency: solves are serialised by a process-wide mutex (taken after the GIL is released,
// so other Python threads keep running). One solve already uses every configured core, and
// MUMPS' sequential (libseq) build is not documented to be safe for concurrent instances.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "csr_convert.hpp"
#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace py = pybind11;

namespace hs {
namespace {

// code -> exception class, filled by hsolve/__init__.py. Owned by the module attribute
// `_error_classes` plus one intentionally leaked reference (no destructor at interpreter exit).
PyObject* g_error_classes = nullptr;

std::mutex& solve_mutex() {
  static std::mutex m;
  return m;
}

/// hs::solve sets the calling thread's OpenMP thread count when cfg.threads > 0; a Python
/// process must not be left with a changed default, so the previous value is restored.
class OmpThreadsScope {
 public:
  explicit OmpThreadsScope(int threads) {
#ifdef _OPENMP
    previous_ = omp_get_max_threads();
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif
  }
  ~OmpThreadsScope() {
#ifdef _OPENMP
    omp_set_num_threads(previous_);
#endif
  }
  OmpThreadsScope(const OmpThreadsScope&) = delete;
  OmpThreadsScope& operator=(const OmpThreadsScope&) = delete;

 private:
  int previous_ = 1;
};

Storage parse_storage(const std::string& s) {
  if (s == "auto") return Storage::Auto;
  if (s == "full") return Storage::Full;
  if (s == "upper") return Storage::Upper;
  if (s == "lower") return Storage::Lower;
  HS_THROW(InvalidInput, "storage = '" << s << "' is not one of {auto, full, upper, lower}");
}

const char* storage_name(Storage s) {
  switch (s) {
    case Storage::Upper: return "upper";
    case Storage::Lower: return "lower";
    case Storage::Full: return "full";
    case Storage::Auto: return "auto";
  }
  return "full";
}

pyconv::ValueType value_type_of(const py::array& a, const std::string& what) {
  const py::dtype dt = a.dtype();
  const char kind = dt.kind();
  const auto size = dt.itemsize();
  const char bo = dt.byteorder();
  if (bo == '>') HS_THROW(InvalidInput, what << ": big-endian data is not supported (convert with astype)");
  if (kind == 'f' && size == 4) return pyconv::ValueType::F32;
  if (kind == 'f' && size == 8) return pyconv::ValueType::F64;
  if (kind == 'c' && size == 8) return pyconv::ValueType::C64;
  if (kind == 'c' && size == 16) return pyconv::ValueType::C128;
  HS_THROW(InvalidInput, what << ": unsupported value dtype '" << std::string(py::str(dt)) << "'");
}

pyconv::IndexType index_type_of(const py::array& a, const std::string& what) {
  const py::dtype dt = a.dtype();
  if (dt.byteorder() == '>') HS_THROW(InvalidInput, what << ": big-endian indices are not supported");
  if (dt.kind() == 'i' && dt.itemsize() == 4) return pyconv::IndexType::I32;
  if (dt.kind() == 'i' && dt.itemsize() == 8) return pyconv::IndexType::I64;
  HS_THROW(InvalidInput, what << ": unsupported index dtype '" << std::string(py::str(dt)) << "'");
}

void require_vector(const py::array& a, const std::string& what) {
  if (a.ndim() != 1) HS_THROW(InvalidInput, what << " must be 1-D, got " << a.ndim() << "-D");
  if (!(a.flags() & py::array::c_style)) HS_THROW(InvalidInput, what << " must be contiguous");
}

/// A matrix argument as built by hsolve/__init__.py (_matrix_arg):
///   ("csr" | "csc", (nrows, ncols), indptr, indices, data)
///   ("coo", (nrows, ncols), row, col, data)
///   ("dense", (nrows, ncols), array2d)
///   ("path", path)
/// The py::array references keep the buffers alive; they are released with the GIL held (the
/// MatrixArg outlives the GIL-free region).
struct MatrixArg {
  std::string name;
  std::string path;  ///< non-empty: load the matrix from this file (hs::load_matrix)
  pyconv::MatrixView view;
  std::vector<py::array> keep;
  std::optional<LoadedMatrix> loaded;  ///< filled inside the GIL-free region for "path"
};

MatrixArg matrix_arg(const py::tuple& t, const std::string& name) {
  MatrixArg m;
  m.name = name;
  m.view.name = name;
  if (t.size() < 2) HS_THROW(InvalidInput, name << ": malformed matrix argument");
  const std::string kind = t[0].cast<std::string>();
  if (kind == "path") {
    m.path = t[1].cast<std::string>();
    return m;
  }
  const auto shape = t[1].cast<std::pair<index_t, index_t>>();
  // Checked here, in the caller's orientation (csc input is converted to A^H, ncols x nrows),
  // before any conversion or triangle restoration.
  if (shape.first != shape.second)
    HS_THROW(InvalidInput, name << " is not square (" << shape.first << "x" << shape.second << ")");
  m.view.nrows = shape.first;
  m.view.ncols = shape.second;
  if (kind == "dense") {
    if (t.size() != 3) HS_THROW(InvalidInput, name << ": malformed dense argument");
    py::array a = t[2].cast<py::array>();
    if (a.ndim() != 2) HS_THROW(InvalidInput, name << " must be 2-D, got " << a.ndim() << "-D");
    if (a.shape(0) != shape.first || a.shape(1) != shape.second)
      HS_THROW(InvalidInput, name << ": shape mismatch in dense argument");
    m.view.layout = pyconv::Layout::Dense;
    m.view.value_type = value_type_of(a, name);
    m.view.values = a.data();
    m.view.nvalues = static_cast<std::size_t>(a.size());
    m.view.row_stride = a.strides(0);
    m.view.col_stride = a.strides(1);
    m.keep.push_back(std::move(a));
    return m;
  }
  if (t.size() != 5) HS_THROW(InvalidInput, name << ": malformed sparse argument");
  if (kind == "csr") {
    m.view.layout = pyconv::Layout::Csr;
  } else if (kind == "csc") {
    m.view.layout = pyconv::Layout::Csc;
  } else if (kind == "coo") {
    m.view.layout = pyconv::Layout::Coo;
  } else {
    HS_THROW(InvalidInput, name << ": unknown matrix layout '" << kind << "'");
  }
  py::array i0 = t[2].cast<py::array>(), i1 = t[3].cast<py::array>(), d = t[4].cast<py::array>();
  const bool coo = m.view.layout == pyconv::Layout::Coo;
  require_vector(i0, name + (coo ? ".row" : ".indptr"));
  require_vector(i1, name + (coo ? ".col" : ".indices"));
  require_vector(d, name + ".data");
  m.view.index_type = index_type_of(i0, name + (coo ? ".row" : ".indptr"));
  if (index_type_of(i1, name + (coo ? ".col" : ".indices")) != m.view.index_type)
    HS_THROW(InvalidInput, name << ": index arrays must share one dtype");
  m.view.value_type = value_type_of(d, name + ".data");
  m.view.index0 = i0.data();
  m.view.nindex0 = static_cast<std::size_t>(i0.size());
  m.view.index1 = i1.data();
  m.view.nindex1 = static_cast<std::size_t>(i1.size());
  m.view.values = d.data();
  m.view.nvalues = static_cast<std::size_t>(d.size());
  m.keep.push_back(std::move(i0));
  m.keep.push_back(std::move(i1));
  m.keep.push_back(std::move(d));
  return m;
}

bool complex_input(const MatrixArg& m) {
  return m.loaded ? m.loaded->is_complex() : pyconv::has_nonzero_imag(m.view);
}

/// The matrix was given with a complex dtype - in memory, or stored as complex in the file - even
/// if every imaginary part is zero and it is solved as real. For "path" call after loading.
bool complex_dtype(const MatrixArg& m) {
  if (!m.loaded) return pyconv::is_complex_type(m.view.value_type);
  if (m.loaded->is_complex()) return true;
  // hs::load_matrix records demoting complex input to real as its note with tag "real" (io.hpp).
  return std::any_of(m.loaded->notes.begin(), m.loaded->notes.end(),
                     [](const std::string& note) { return note.rfind("real:", 0) == 0; });
}

/// The matrix as CsrMatrix<T> plus notes ("A: ..." in hs::load_matrix wording). GIL-free.
template <class T>
CsrMatrix<T> obtain(MatrixArg& m, Storage storage, std::vector<std::string>& notes) {
  if (m.loaded) {
    for (const auto& note : m.loaded->notes) notes.push_back(m.name + ": " + note);
    if (m.loaded->is_complex()) {
      if constexpr (is_complex_v<T>) return std::get<1>(std::move(m.loaded->matrix));
      HS_THROW(Internal, m.name << ": complex matrix in a real solve");
    } else {
      if constexpr (is_complex_v<T>) {
        return to_complex(std::get<0>(m.loaded->matrix));
      } else {
        return std::get<0>(std::move(m.loaded->matrix));
      }
    }
  }
  pyconv::ConvertStats st;
  CsrMatrix<T> a = pyconv::to_csr<T>(m.view, st);
  for (const auto& note : pyconv::stats_notes(st)) notes.push_back(m.name + ": " + note);
  const bool transposed = m.view.layout == pyconv::Layout::Csc;
  const std::string note = pyconv::restore_storage(a, storage, transposed, m.name);
  if (!note.empty()) notes.push_back(m.name + ": " + note);
  if (!is_complex_v<T> && pyconv::is_complex_type(m.view.value_type))
    notes.push_back(m.name + ": real: complex input has all-zero imaginary parts; solved as real");
  return a;
}

using AnySolution = std::variant<Solution<double>, Solution<cplx>>;

struct SolveOutcome {
  AnySolution solution;
  bool complex_dtype = false;  ///< some input had a complex dtype: eigenvectors go back complex
};

template <class T>
Solution<T> solve_as(MatrixArg& a, MatrixArg* b, const SolverConfig& cfg, int verbose, Storage storage) {
  std::vector<std::string> notes;
  Solution<T> sol;
  {
    CsrMatrix<T> am = obtain<T>(a, storage, notes);
    a.loaded.reset();
    CsrMatrix<T> bm;
    if (b) {
      bm = obtain<T>(*b, storage, notes);
      b->loaded.reset();
    }
    std::unique_ptr<Logger> log;
    if (verbose > 0) log = std::make_unique<Logger>(verbose);
    sol = solve(am, b ? &bm : nullptr, cfg, log.get());
  }  // the converted matrices are freed before the results go back to Python
  for (const auto& note : notes) sol.report.warnings.push_back("input: " + note);
  return sol;
}

/// GIL-free part of _solve: load/convert, dispatch real/complex, solve.
SolveOutcome run_solve(MatrixArg& a, MatrixArg* b, const SolverConfig& cfg, int verbose, Storage storage) {
  LoadOptions lo;
  lo.storage = storage;
  lo.hermitian_tol = cfg.hermitian_tol;
  if (!a.path.empty()) a.loaded = load_matrix(a.path, lo);
  if (b && !b->path.empty()) b->loaded = load_matrix(b->path, lo);
  SolveOutcome out;
  out.complex_dtype = complex_dtype(a) || (b && complex_dtype(*b));
  // Complex A with real B promotes B (and vice versa); complex dtypes whose imaginary parts are
  // all zero are solved as real, exactly like hs::load_matrix does for files.
  const bool complex_problem = complex_input(a) || (b && complex_input(*b));
  if (complex_problem) {
    out.solution = solve_as<cplx>(a, b, cfg, verbose, storage);
  } else {
    out.solution = solve_as<double>(a, b, cfg, verbose, storage);
  }
  return out;
}

/// n x m column-major DenseMatrix -> numpy (n, m) Fortran-ordered array owning the buffer.
/// No columns -> an (n, 0) array (the driver leaves unused blocks default-constructed, 0 x 0).
template <class T>
py::array take_matrix(DenseMatrix<T>&& m, index_t n) {
  const py::ssize_t rows = m.cols() == 0 ? static_cast<py::ssize_t>(n) : m.rows(), cols = m.cols();
  if (rows == 0 || cols == 0) return py::array_t<T, py::array::f_style>(std::vector<py::ssize_t>{rows, cols});
  auto owner = std::make_unique<DenseMatrix<T>>(std::move(m));
  T* data = owner->data();
  py::capsule base(owner.get(), [](void* p) { delete static_cast<DenseMatrix<T>*>(p); });
  owner.release();  // now owned by the capsule
  return py::array_t<T>(std::vector<py::ssize_t>{rows, cols},
                        std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(T)),
                                                 static_cast<py::ssize_t>(sizeof(T)) * rows},
                        data, base);
}

template <class T>
py::array copy_vector(const std::vector<T>& v) {
  py::array_t<T> out(static_cast<py::ssize_t>(v.size()));
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

template <class T>
py::tuple to_python(Solution<T>&& sol, const SolverConfig& cfg, bool complex_dtype) {
  const std::string report = sol.report.to_json(-1);
  py::array w = copy_vector(sol.eigenvalues);
  const index_t n = sol.report.n;
  py::array v = take_matrix(std::move(sol.eigenvectors), n);
  py::array cw = copy_vector(sol.complex_eigenvalues);
  py::array cv = take_matrix(std::move(sol.complex_eigenvectors), n);
  return py::make_tuple(w, v, cw, cv, report, config_to_json(cfg, -1), complex_dtype);
}

py::tuple py_solve(const py::tuple& a_arg, const py::object& b_arg, const std::string& config_json, int verbose,
                   const std::string& storage_text) {
  const SolverConfig cfg = parse_config(config_json);
  const Storage storage = parse_storage(storage_text);
  MatrixArg a = matrix_arg(a_arg, "A");
  std::optional<MatrixArg> b;
  if (!b_arg.is_none()) b = matrix_arg(b_arg.cast<py::tuple>(), "B");
  // Both orders are known before any conversion (a file's only after loading; hs::solve checks).
  if (b && a.path.empty() && b->path.empty() && b->view.nrows != a.view.nrows)
    HS_THROW(InvalidInput, "B has order " << b->view.nrows << ", A has " << a.view.nrows);
  std::optional<SolveOutcome> out;
  {
    py::gil_scoped_release nogil;
    std::lock_guard<std::mutex> lock(solve_mutex());
    OmpThreadsScope threads(cfg.threads);
    out = run_solve(a, b ? &*b : nullptr, cfg, verbose, storage);
  }
  const bool complex_dtype = out->complex_dtype;
  return std::visit([&](auto& s) { return to_python(std::move(s), cfg, complex_dtype); }, out->solution);
}

/// A loaded CSR matrix as the arrays of a scipy csr_array (indptr, indices, data), with the
/// fewest copies. scipy keeps ONE index dtype for indptr and indices - int64 as soon as either
/// is int64 (sparray never downcasts by content) - while CsrMatrix has an int64 rowptr and int32
/// colind. Handing both over as they are made scipy copy colind to int64 while the capsule kept
/// the int32 original alive (12 instead of 4 bytes per nonzero). Therefore:
///  * n, nnz <= INT32_MAX (every practical case): indptr is an int32 copy of rowptr (n + 1
///    entries; rowptr is then freed), indices and data are views of colind and values;
///  * otherwise, or wide_indices (tests): indptr is a view of rowptr and indices an int64 copy of
///    colind (the copy scipy would make; colind is then freed); data is a view of values.
/// The views own the CsrMatrix through one capsule. The copies run without the GIL.
template <class T>
py::tuple csr_to_python(CsrMatrix<T>&& m, bool wide_indices) {
  auto owner = std::make_unique<CsrMatrix<T>>(std::move(m));
  CsrMatrix<T>* p = owner.get();
  py::capsule base(owner.get(), [](void* q) { delete static_cast<CsrMatrix<T>*>(q); });
  owner.release();
  auto view = [&](auto& vec) -> py::array {
    using V = typename std::remove_reference_t<decltype(vec)>::value_type;
    if (vec.empty()) return py::array_t<V>(0);  // numpy would treat a null data pointer as "allocate"
    return py::array_t<V>(static_cast<py::ssize_t>(vec.size()), vec.data(), base);
  };
  constexpr index_t kInt32Max = std::numeric_limits<std::int32_t>::max();
  const index_t n = p->nrows, nnz = p->nnz();
  py::array indptr, indices;
  if (!wide_indices && n <= kInt32Max && nnz <= kInt32Max) {
    py::array_t<std::int32_t> narrow(static_cast<py::ssize_t>(n) + 1);
    std::int32_t* dst = narrow.mutable_data();
    {
      py::gil_scoped_release nogil;
      std::transform(p->rowptr.begin(), p->rowptr.end(), dst, [](index_t x) { return static_cast<std::int32_t>(x); });
      std::vector<index_t>().swap(p->rowptr);
    }
    indptr = std::move(narrow);
    indices = view(p->colind);
  } else {
    py::array_t<std::int64_t> wide(static_cast<py::ssize_t>(nnz));
    std::int64_t* dst = wide.mutable_data();
    {
      py::gil_scoped_release nogil;
      std::copy(p->colind.begin(), p->colind.end(), dst);
      std::vector<std::int32_t>().swap(p->colind);
    }
    indptr = view(p->rowptr);
    indices = std::move(wide);
  }
  py::array data = view(p->values);
  return py::make_tuple(n, indptr, indices, data);
}

py::tuple py_load_matrix(const std::string& path, const std::string& storage_text, double hermitian_tol,
                         bool wide_indices) {
  LoadOptions lo;
  lo.storage = parse_storage(storage_text);
  lo.hermitian_tol = hermitian_tol;
  std::optional<LoadedMatrix> lm;
  {
    py::gil_scoped_release nogil;
    lm = load_matrix(path, lo);
  }
  py::dict info;
  info["source_format"] = lm->source_format;
  info["storage"] = storage_name(lm->detected_storage);
  info["hermitian_defect"] = lm->hermitian_defect;
  info["notes"] = lm->notes;
  py::tuple arrays = std::visit([&](auto& m) { return csr_to_python(std::move(m), wide_indices); }, lm->matrix);
  return py::make_tuple(arrays[0], arrays[1], arrays[2], arrays[3], info);
}

void translate_error(const Error& e) {
  PyObject* cls = nullptr;
  if (g_error_classes) cls = PyDict_GetItemString(g_error_classes, to_string(e.code()));  // borrowed
  PyErr_SetString(cls ? cls : PyExc_RuntimeError, e.what());
}

}  // namespace
}  // namespace hs

PYBIND11_MODULE(_hsolve, m) {
  m.doc() = "hsolve C++ core (private; use the hsolve package, see docs/PYTHON.md)";

  py::dict classes;
  m.attr("_error_classes") = classes;
  hs::g_error_classes = classes.inc_ref().ptr();
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const hs::Error& e) {
      hs::translate_error(e);
    }
  });

  m.def("solve", &hs::py_solve, py::arg("a"), py::arg("b"), py::arg("config_json"), py::arg("verbose"),
        py::arg("storage"),
        "solve(a, b, config_json, verbose, storage) -> (w, v, complex_w, complex_v, report_json, config_json, "
        "complex_dtype)");
  m.def("load_matrix", &hs::py_load_matrix, py::arg("path"), py::arg("storage"), py::arg("hermitian_tol"),
        py::arg("wide_indices") = false,
        "load_matrix(path, storage, hermitian_tol, wide_indices=False) -> (n, indptr, indices, data, info); "
        "int64 indices only when n or nnz exceeds int32 (or wide_indices), else int32 indptr and indices");
  m.def("parse_config", [](const std::string& text) { return hs::config_to_json(hs::parse_config(text), -1); },
        py::arg("config_json"), "Validate a config JSON document; returns the complete normalised document.");
  m.def("version", [] { return std::string(hs::version()); });
  m.def("build_info", [] { return hs::build_info(); });
  m.def("max_threads", [] { return hs::max_threads(); });
}
