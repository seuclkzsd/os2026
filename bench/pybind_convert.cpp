// Measures the input conversion of the Python module (python/src/csr_convert.hpp): the cost of
// turning scipy/numpy buffers into the owning hs::CsrMatrix, per input layout and dtype, against
// the floor for an owning copy (memcpy of indices and values + widening indptr to int64) and
// against the library's generic hs::csr_from_coo. Backs the numbers in docs/PYTHON.md.
//
//   bench_pybind_convert [N=100] [repeats=5]
//
// Test matrix: 3-D 7-point Laplacian on an N^3 grid (n = N^3, ~7n entries), in the layout scipy
// holds it (int32 indptr/indices, float64 data); the variants are derived from it. Prints one JSON
// line per case: median seconds, effective GB/s (bytes read + written once), ratio to the floor.
// Run through tools/run/guard.py (e.g. --cores 38,39 --mem-gb 6).
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../python/src/csr_convert.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using hs::pyconv::IndexType;
using hs::pyconv::Layout;
using hs::pyconv::MatrixView;
using hs::pyconv::ValueType;

struct Scipy {  // CSR arrays as scipy holds them
  index_t n = 0;
  std::vector<std::int32_t> indptr, indices;
  std::vector<double> data;
};

Scipy laplacian3d(index_t g) {
  Scipy s;
  s.n = g * g * g;
  s.indptr.reserve(static_cast<std::size_t>(s.n) + 1);
  s.indptr.push_back(0);
  for (index_t z = 0; z < g; ++z)
    for (index_t y = 0; y < g; ++y)
      for (index_t x = 0; x < g; ++x) {
        const index_t i = x + g * (y + g * z);
        auto add = [&](index_t j, double v) {
          s.indices.push_back(static_cast<std::int32_t>(j));
          s.data.push_back(v);
        };
        if (z > 0) add(i - g * g, -1.0);
        if (y > 0) add(i - g, -1.0);
        if (x > 0) add(i - 1, -1.0);
        add(i, 6.0);
        if (x + 1 < g) add(i + 1, -1.0);
        if (y + 1 < g) add(i + g, -1.0);
        if (z + 1 < g) add(i + g * g, -1.0);
        s.indptr.push_back(static_cast<std::int32_t>(s.indices.size()));
      }
  return s;
}

double median_seconds(int repeats, const std::function<void()>& f) {
  std::vector<double> t;
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

MatrixView view_of(Layout layout, index_t n, IndexType it, const void* i0, std::size_t n0, const void* i1,
                   std::size_t n1, ValueType vt, const void* vals, std::size_t nv) {
  MatrixView v;
  v.layout = layout;
  v.nrows = v.ncols = n;
  v.index_type = it;
  v.index0 = i0;
  v.nindex0 = n0;
  v.index1 = i1;
  v.nindex1 = n1;
  v.value_type = vt;
  v.values = vals;
  v.nvalues = nv;
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const index_t g = argc > 1 ? std::atoll(argv[1]) : 100;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  const Scipy s = laplacian3d(g);
  const index_t n = s.n;
  const std::size_t nnz = s.data.size();
  std::printf("{\"bench\": \"pybind_convert\", \"n\": %lld, \"nnz\": %zu, \"threads\": %d, \"repeats\": %d}\n",
              static_cast<long long>(n), nnz, hs::max_threads(), repeats);

  // Bytes of the owning result (written once) + bytes of the input (read once), per case.
  const double out_bytes = 8.0 * static_cast<double>(n + 1) + 12.0 * static_cast<double>(nnz);
  double floor_s = 0.0;
  auto report = [&](const char* name, double sec, double in_bytes, const char* note) {
    const double gbs = (in_bytes + out_bytes) / sec / 1e9;
    std::printf("{\"case\": \"%s\", \"median_s\": %.5f, \"gb_per_s\": %.2f, \"vs_floor\": %.2f, \"note\": \"%s\"}\n",
                name, sec, gbs, floor_s > 0 ? sec / floor_s : 1.0, note);
    std::fflush(stdout);
  };
  const double csr_in = 4.0 * static_cast<double>(n + 1) + 12.0 * static_cast<double>(nnz);

  // Floor: what any owning CsrMatrix<double> copy costs (allocation + zero fill + copy).
  floor_s = median_seconds(repeats, [&] {
    hs::CsrMatrix<double> m;
    m.nrows = m.ncols = n;
    m.rowptr.assign(s.indptr.begin(), s.indptr.end());
    m.colind = s.indices;
    m.values = s.data;
  });
  report("floor_memcpy", floor_s, csr_in, "std::vector copies of indices/data, indptr widened");

  auto run = [&](const char* name, const MatrixView& v, double in_bytes, const char* note) {
    const double sec = median_seconds(repeats, [&] {
      hs::pyconv::ConvertStats local;
      hs::CsrMatrix<double> m = hs::pyconv::to_csr<double>(v, local);
      if (m.nnz() != static_cast<index_t>(nnz)) std::abort();
    });
    report(name, sec, in_bytes, note);
  };

  run("csr_i32_f64", view_of(Layout::Csr, n, IndexType::I32, s.indptr.data(), s.indptr.size(), s.indices.data(),
                             nnz, ValueType::F64, s.data.data(), nnz),
      csr_in, "canonical scipy csr");
  run("csc_i32_f64", view_of(Layout::Csc, n, IndexType::I32, s.indptr.data(), s.indptr.size(), s.indices.data(),
                             nnz, ValueType::F64, s.data.data(), nnz),
      csr_in, "csc: no transpose (A^H)");
  {
    std::vector<std::int64_t> p64(s.indptr.begin(), s.indptr.end()), i64(s.indices.begin(), s.indices.end());
    run("csr_i64_f64", view_of(Layout::Csr, n, IndexType::I64, p64.data(), p64.size(), i64.data(), nnz,
                               ValueType::F64, s.data.data(), nnz),
        8.0 * static_cast<double>(n + 1) + 16.0 * static_cast<double>(nnz), "int64 indices narrowed in the copy");
  }
  {
    std::vector<float> f32(s.data.begin(), s.data.end());
    run("csr_i32_f32", view_of(Layout::Csr, n, IndexType::I32, s.indptr.data(), s.indptr.size(), s.indices.data(),
                               nnz, ValueType::F32, f32.data(), nnz),
        4.0 * static_cast<double>(n + 1) + 8.0 * static_cast<double>(nnz), "float32 widened in the copy");
  }
  {
    std::vector<std::complex<double>> c128(s.data.begin(), s.data.end());
    const MatrixView v = view_of(Layout::Csr, n, IndexType::I32, s.indptr.data(), s.indptr.size(), s.indices.data(),
                                 nnz, ValueType::C128, c128.data(), nnz);
    const double sec = median_seconds(repeats, [&] {
      if (hs::pyconv::has_nonzero_imag(v)) std::abort();  // worst case: scans everything
      hs::pyconv::ConvertStats local;
      hs::CsrMatrix<double> m = hs::pyconv::to_csr<double>(v, local);
      if (m.nnz() != static_cast<index_t>(nnz)) std::abort();
    });
    report("csr_c128_zero_imag", sec, 4.0 * static_cast<double>(n + 1) + 36.0 * static_cast<double>(nnz),
           "imaginary-part scan + conversion to real");
  }
  {
    std::vector<std::int32_t> rev(s.indices);
    std::vector<double> revd(s.data);
    for (index_t i = 0; i < n; ++i) {
      std::reverse(rev.begin() + s.indptr[i], rev.begin() + s.indptr[i + 1]);
      std::reverse(revd.begin() + s.indptr[i], revd.begin() + s.indptr[i + 1]);
    }
    run("csr_unsorted_rows", view_of(Layout::Csr, n, IndexType::I32, s.indptr.data(), s.indptr.size(), rev.data(),
                                     nnz, ValueType::F64, revd.data(), nnz),
        csr_in, "every row reversed: per-row sort");
  }
  // COO: row-ordered (as tocoo() gives) and randomly shuffled.
  std::vector<std::int32_t> row(nnz), col(s.indices);
  std::vector<double> val(s.data);
  for (index_t i = 0; i < n; ++i)
    std::fill(row.begin() + s.indptr[i], row.begin() + s.indptr[i + 1], static_cast<std::int32_t>(i));
  const double coo_in = 16.0 * static_cast<double>(nnz);
  run("coo_row_ordered", view_of(Layout::Coo, n, IndexType::I32, row.data(), nnz, col.data(), nnz, ValueType::F64,
                                 val.data(), nnz),
      coo_in, "counting sort, rows already sorted");
  {
    std::vector<std::size_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    std::shuffle(perm.begin(), perm.end(), std::mt19937_64(2026));
    std::vector<std::int32_t> r2(nnz), c2(nnz);
    std::vector<double> v2(nnz);
    for (std::size_t p = 0; p < nnz; ++p) {
      r2[p] = row[perm[p]];
      c2[p] = col[perm[p]];
      v2[p] = val[perm[p]];
    }
    run("coo_shuffled", view_of(Layout::Coo, n, IndexType::I32, r2.data(), nnz, c2.data(), nnz, ValueType::F64,
                                v2.data(), nnz),
        coo_in, "counting sort + per-row sort");
    // The library's generic builder on the same triplets (index_t vectors it needs are prepared
    // outside the timing, so this favours it).
    std::vector<index_t> r64(r2.begin(), r2.end()), c64(c2.begin(), c2.end());
    const double sec = median_seconds(repeats, [&] {
      hs::CsrMatrix<double> m = hs::csr_from_coo<double>(n, n, r64, c64, v2);
      if (m.nnz() != static_cast<index_t>(nnz)) std::abort();
    });
    report("hs_csr_from_coo_shuffled", sec, 24.0 * static_cast<double>(nnz), "library reference (int64 triplets)");
  }
  {
    // Upper-triangle storage: conversion + mirroring (symmetrize_from_triangle).
    std::vector<std::int32_t> up_ptr{0}, up_idx;
    std::vector<double> up_val;
    for (index_t i = 0; i < n; ++i) {
      for (index_t p = s.indptr[i]; p < s.indptr[i + 1]; ++p)
        if (s.indices[p] >= i) {
          up_idx.push_back(s.indices[p]);
          up_val.push_back(s.data[p]);
        }
      up_ptr.push_back(static_cast<std::int32_t>(up_idx.size()));
    }
    const MatrixView v = view_of(Layout::Csr, n, IndexType::I32, up_ptr.data(), up_ptr.size(), up_idx.data(),
                                 up_idx.size(), ValueType::F64, up_val.data(), up_val.size());
    const double sec = median_seconds(repeats, [&] {
      hs::pyconv::ConvertStats local;
      hs::CsrMatrix<double> m = hs::pyconv::to_csr<double>(v, local);
      hs::pyconv::restore_storage(m, hs::Storage::Auto, false, "A");
      if (m.nnz() != static_cast<index_t>(nnz)) std::abort();
    });
    report("csr_upper_triangle", sec, 4.0 * static_cast<double>(n + 1) + 12.0 * static_cast<double>(up_idx.size()),
           "conversion + symmetrize_from_triangle");
  }
  {
    // Dense input (small: dense arrays are for small problems), 3-D Laplacian of order 2744.
    const Scipy d = laplacian3d(14);
    std::vector<double> dense(static_cast<std::size_t>(d.n * d.n), 0.0);
    for (index_t i = 0; i < d.n; ++i)
      for (index_t p = d.indptr[i]; p < d.indptr[i + 1]; ++p) dense[static_cast<std::size_t>(i * d.n + d.indices[p])] = d.data[p];
    MatrixView v;
    v.layout = Layout::Dense;
    v.nrows = v.ncols = d.n;
    v.value_type = ValueType::F64;
    v.values = dense.data();
    v.nvalues = dense.size();
    v.row_stride = static_cast<std::ptrdiff_t>(sizeof(double) * d.n);  // C order
    v.col_stride = sizeof(double);
    const double sec = median_seconds(repeats, [&] {
      hs::pyconv::ConvertStats local;
      hs::CsrMatrix<double> m = hs::pyconv::to_csr<double>(v, local);
      if (m.nnz() != static_cast<index_t>(d.data.size())) std::abort();
    });
    std::printf("{\"case\": \"dense_c_order_n2744\", \"median_s\": %.5f, \"gb_per_s\": %.2f, \"note\": "
                "\"scan of %lld dense entries\"}\n",
                sec, 8.0 * static_cast<double>(dense.size()) / sec / 1e9, static_cast<long long>(dense.size()));
  }
  return 0;
}
