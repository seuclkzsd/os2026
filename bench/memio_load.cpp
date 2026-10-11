// Peak memory of hs::load_matrix per input layout (module memio, docs/DESIGN.md §3.1).
//
//   bench_memio_load [--grid M] [--mtx-grid M] [--dir DIR] [--only SUBSTR] [--keep]
//   bench_memio_load --files FILE...
//
// Generates a Hermitian 27-point-stencil matrix on an M^3 grid (real and complex), writes it in
// every layout the loader accepts (scipy npz csr/csc/coo with full or one-triangle storage,
// near-Hermitian inputs that need their Hermitian part, non-native dtypes, MatrixMarket general and
// symmetric/hermitian), and loads each file once. Files are written one at a time into DIR (default:
// a fresh directory under $TMPDIR or /tmp) and removed after the measurement unless --keep.
// With --files, the given matrix files are measured instead.
//
// Method: before each load the allocator's free memory is returned to the kernel (malloc_trim) and
// the kernel's resident high-water mark is reset ("5" > /proc/self/clear_refs, Linux >= 4.0). Then
//   peak_delta = VmHWM after the load - VmRSS before it
// is the largest amount of memory the load held at any instant (kernel accounting, no sampling
// gaps), and ratio = peak_delta / csr_bytes, where csr_bytes is the capacity of the returned CSR
// arrays (8 (n+1) + (4 + sizeof(T)) nnz when tight). The loaded matrix is compared with the
// generator's matrix (bitwise for exactly Hermitian inputs, by value for Hermitian-part inputs)
// after the measurement. One JSON line per file on stdout. Run through tools/run/guard.py.
#include <malloc.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <variant>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"

namespace fs = std::filesystem;
using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;

namespace {

// ------------------------------------------------------------------------- memory probes

struct ProcMem {
  std::int64_t rss = 0;  ///< VmRSS, bytes
  std::int64_t hwm = 0;  ///< VmHWM, bytes
};

ProcMem proc_mem() {
  std::ifstream in("/proc/self/status");
  ProcMem m;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) m.rss = std::atoll(line.c_str() + 6) * 1024;
    if (line.rfind("VmHWM:", 0) == 0) m.hwm = std::atoll(line.c_str() + 6) * 1024;
  }
  return m;
}

bool reset_hwm() {
  std::ofstream out("/proc/self/clear_refs");
  out << "5";
  out.flush();
  return static_cast<bool>(out);
}

template <class T>
std::int64_t csr_bytes(const CsrMatrix<T>& a) {
  return static_cast<std::int64_t>(a.rowptr.capacity() * sizeof(index_t) +
                                   a.colind.capacity() * sizeof(std::int32_t) +
                                   a.values.capacity() * sizeof(T));
}

// ------------------------------------------------------------------------ test matrices

std::uint64_t splitmix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

double unit(std::uint64_t h) { return static_cast<double>(h >> 11) * 0x1.0p-53 * 2.0 - 1.0; }

// Hermitian 27-point stencil on an m^3 grid: a_ij = conj(a_ji) from a hash of (min, max).
template <class T>
CsrMatrix<T> grid_hermitian(index_t m) {
  const index_t n = m * m * m;
  CsrMatrix<T> a;
  a.nrows = a.ncols = n;
  a.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  auto neighbours = [m](index_t i, auto&& f) {
    const index_t x = i % m, y = (i / m) % m, z = i / (m * m);
    for (index_t dz = -1; dz <= 1; ++dz)
      for (index_t dy = -1; dy <= 1; ++dy)
        for (index_t dx = -1; dx <= 1; ++dx) {
          if (x + dx < 0 || x + dx >= m || y + dy < 0 || y + dy >= m || z + dz < 0 || z + dz >= m)
            continue;
          f(i + dz * m * m + dy * m + dx);  // ascending in (dz, dy, dx)
        }
  };
  for (index_t i = 0; i < n; ++i) {
    index_t c = 0;
    neighbours(i, [&](index_t) { ++c; });
    a.rowptr[i + 1] = a.rowptr[i] + c;
  }
  a.colind.resize(static_cast<std::size_t>(a.rowptr[n]));
  a.values.resize(static_cast<std::size_t>(a.rowptr[n]));
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) {
    index_t p = a.rowptr[i];
    neighbours(i, [&](index_t j) {
      a.colind[p] = static_cast<std::int32_t>(j);
      const index_t lo = std::min(i, j), hi = std::max(i, j);
      const std::uint64_t h = splitmix(static_cast<std::uint64_t>(lo * n + hi));
      if (i == j) {
        a.values[p] = T(27.0 + unit(h));
      } else if constexpr (hs::is_complex_v<T>) {
        const T v(unit(h), unit(splitmix(h)));
        a.values[p] = i < j ? v : std::conj(v);
      } else {
        a.values[p] = unit(h);
      }
      ++p;
    });
  }
  return a;
}

// Rounding-level asymmetry on the same pattern: Hermitian part taken in place.
template <class T>
CsrMatrix<T> perturbed_symmetric(const CsrMatrix<T>& a) {
  CsrMatrix<T> b = a;
  for (index_t p = 0; p < b.nnz(); ++p) b.values[p] *= 1.0 + 1e-13 * static_cast<double>(p % 7 - 3);
  return b;
}

// Plus tiny entries without a mirror ("orphans") in every 16th row: the Hermitian part needs the
// union pattern of A and A^H.
template <class T>
CsrMatrix<T> perturbed_asymmetric(const CsrMatrix<T>& a) {
  const CsrMatrix<T> s = perturbed_symmetric(a);
  const index_t n = a.nrows;
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < n; ++i)
    for (index_t p = s.rowptr[i]; p < s.rowptr[i + 1]; ++p) {
      r.push_back(i);
      c.push_back(s.colind[p]);
      v.push_back(s.values[p]);
    }
  for (index_t i = 0; i < n; i += 16) {
    r.push_back(i);
    c.push_back((i + n / 2) % n);  // far from the 27-point band: never stored
    v.push_back(T(1e-12));
  }
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

template <class T>
CsrMatrix<T> conj_values(CsrMatrix<T> a) {
  for (T& v : a.values) v = hs::conj_if(v);
  return a;
}

// --------------------------------------------------------------------------- npz writer

std::uint32_t crc32_update(std::uint32_t crc, const unsigned char* p, std::size_t n) {
  static const std::array<std::uint32_t, 256> table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  crc = ~crc;
  for (std::size_t k = 0; k < n; ++k) crc = table[(crc ^ p[k]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

std::string npy_header(const std::string& descr, const std::vector<std::size_t>& shape) {
  std::string s = "(";
  for (std::size_t d : shape) s += std::to_string(d) + ",";
  if (shape.size() > 1) s.pop_back();
  s += ")";
  std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + s + ", }";
  std::string out = std::string("\x93NUMPY\x01\x00", 8);
  const std::size_t total = 10 + dict.size() + 1;
  dict.append((64 - total % 64) % 64, ' ');
  dict += '\n';
  out += static_cast<char>(dict.size() & 0xFF);
  out += static_cast<char>(dict.size() >> 8);
  return out + dict;
}

// A member: header text plus a payload given as raw bytes.
struct Member {
  std::string name;
  std::string header;
  const void* data;
  std::size_t bytes;
};

template <class V>
Member array_member(const std::string& key, const char* descr, const std::vector<V>& v) {
  return {key + ".npy", npy_header(descr, {v.size()}), v.data(), v.size() * sizeof(V)};
}

void put16(std::string& s, std::uint32_t v) {
  s += static_cast<char>(v & 0xFF);
  s += static_cast<char>((v >> 8) & 0xFF);
}
void put32(std::string& s, std::uint32_t v) {
  put16(s, v & 0xFFFF);
  put16(s, v >> 16);
}

// Stored (uncompressed) zip, as np.savez writes it (scipy.sparse.save_npz(compressed=False)).
void write_zip(const fs::path& path, const std::vector<Member>& members) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  std::string central;
  std::uint64_t offset = 0;
  for (const Member& m : members) {
    std::uint32_t crc = crc32_update(0, reinterpret_cast<const unsigned char*>(m.header.data()), m.header.size());
    crc = crc32_update(crc, static_cast<const unsigned char*>(m.data), m.bytes);
    const std::uint64_t size = m.header.size() + m.bytes;
    if (size >= 0xFFFFFFFFull || offset >= 0xFFFFFFFFull) HS_THROW(Unsupported, "bench zip writer: member too large");
    std::string local;
    put32(local, 0x04034b50);
    put16(local, 20);
    put16(local, 0);
    put16(local, 0);  // stored
    put16(local, 0);
    put16(local, 0x21);
    put32(local, crc);
    put32(local, static_cast<std::uint32_t>(size));
    put32(local, static_cast<std::uint32_t>(size));
    put16(local, static_cast<std::uint32_t>(m.name.size()));
    put16(local, 0);
    local += m.name;
    put32(central, 0x02014b50);
    put16(central, 20);
    put16(central, 20);
    put16(central, 0);
    put16(central, 0);
    put16(central, 0);
    put16(central, 0x21);
    put32(central, crc);
    put32(central, static_cast<std::uint32_t>(size));
    put32(central, static_cast<std::uint32_t>(size));
    put16(central, static_cast<std::uint32_t>(m.name.size()));
    put16(central, 0);
    put16(central, 0);
    put16(central, 0);
    put16(central, 0);
    put32(central, 0);
    put32(central, static_cast<std::uint32_t>(offset));
    central += m.name;
    out.write(local.data(), static_cast<std::streamsize>(local.size()));
    out.write(m.header.data(), static_cast<std::streamsize>(m.header.size()));
    out.write(static_cast<const char*>(m.data), static_cast<std::streamsize>(m.bytes));
    offset += local.size() + size;
  }
  std::string end;
  put32(end, 0x06054b50);
  put16(end, 0);
  put16(end, 0);
  put16(end, static_cast<std::uint32_t>(members.size()));
  put16(end, static_cast<std::uint32_t>(members.size()));
  put32(end, static_cast<std::uint32_t>(central.size()));
  put32(end, static_cast<std::uint32_t>(offset));
  put16(end, 0);
  out.write(central.data(), static_cast<std::streamsize>(central.size()));
  out.write(end.data(), static_cast<std::streamsize>(end.size()));
  if (!out) HS_THROW(Io, "bench: cannot write " << path.string());
}

template <class T>
const char* descr_of() {
  if constexpr (std::is_same_v<T, double>) {
    return "<f8";
  } else if constexpr (std::is_same_v<T, cplx>) {
    return "<c16";
  } else if constexpr (std::is_same_v<T, float>) {
    return "<f4";
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return "<c8";
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return "<i4";
  } else {
    static_assert(std::is_same_v<T, std::int64_t>);
    return "<i8";
  }
}

// scipy.sparse.save_npz layout: format (|S3), shape (<i8), data + indices/indptr or row/col.
// I is the index dtype, V the value dtype written (converted from the matrix's T).
template <class V, class I, class T>
void write_scipy(const fs::path& path, const std::string& fmt, const CsrMatrix<T>& m) {
  std::vector<V> data(m.values.size());
  for (std::size_t k = 0; k < data.size(); ++k) data[k] = static_cast<V>(m.values[k]);
  const std::vector<std::int64_t> shape{m.nrows, m.ncols};
  const std::string fbytes = fmt;  // 3 bytes, |S3
  std::vector<Member> members;
  members.push_back({"format.npy", npy_header("|S3", {}), fbytes.data(), 3});
  members.push_back(array_member("shape", "<i8", shape));
  members.push_back(array_member("data", descr_of<V>(), data));
  std::vector<I> a1, a2;
  if (fmt == "coo") {
    a1.resize(m.colind.size());
    for (index_t i = 0; i < m.nrows; ++i)
      for (index_t p = m.rowptr[i]; p < m.rowptr[i + 1]; ++p) a1[p] = static_cast<I>(i);
    a2.assign(m.colind.begin(), m.colind.end());
    members.push_back(array_member("row", descr_of<I>(), a1));
    members.push_back(array_member("col", descr_of<I>(), a2));
  } else {  // csr, or csc given the CSR arrays of A^T
    a1.assign(m.colind.begin(), m.colind.end());
    a2.assign(m.rowptr.begin(), m.rowptr.end());
    members.push_back(array_member("indices", descr_of<I>(), a1));
    members.push_back(array_member("indptr", descr_of<I>(), a2));
  }
  write_zip(path, members);
}

// --------------------------------------------------------------------- MatrixMarket writer

void append_number(std::string& s, double v) {
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);
  s.append(buf, r.ptr);
}

// coordinate general (all entries) or symmetric/hermitian (lower triangle).
template <class T>
void write_mtx(const fs::path& path, const CsrMatrix<T>& a, bool lower_only) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const char* field = hs::is_complex_v<T> ? "complex" : "real";
  const char* sym = !lower_only ? "general" : (hs::is_complex_v<T> ? "hermitian" : "symmetric");
  index_t count = 0;
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) count += !lower_only || a.colind[p] <= i;
  out << "%%MatrixMarket matrix coordinate " << field << " " << sym << "\n"
      << a.nrows << " " << a.ncols << " " << count << "\n";
  std::string buf;
  for (index_t i = 0; i < a.nrows; ++i) {
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      if (lower_only && a.colind[p] > i) continue;
      buf += std::to_string(i + 1);
      buf += ' ';
      buf += std::to_string(a.colind[p] + 1);
      buf += ' ';
      if constexpr (hs::is_complex_v<T>) {
        append_number(buf, a.values[p].real());
        buf += ' ';
        append_number(buf, a.values[p].imag());
      } else {
        append_number(buf, a.values[p]);
      }
      buf += '\n';
    }
    if (buf.size() > (1u << 22)) {
      out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
      buf.clear();
    }
  }
  out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  if (!out) HS_THROW(Io, "bench: cannot write " << path.string());
}

// --------------------------------------------------------------------------- measuring

struct Sample {
  std::int64_t peak_delta = 0;
  std::int64_t retained = 0;
  std::int64_t csr = 0;
  double seconds = 0;
  hs::LoadedMatrix loaded;
};

Sample measure_load(const fs::path& path) {
  malloc_trim(0);
  if (!reset_hwm()) HS_THROW(Io, "cannot write /proc/self/clear_refs (needs Linux >= 4.0)");
  const ProcMem before = proc_mem();
  Sample s;
  const auto t0 = std::chrono::steady_clock::now();
  s.loaded = hs::load_matrix(path);
  s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const ProcMem after = proc_mem();
  s.peak_delta = after.hwm - before.rss;
  s.retained = after.rss - before.rss;
  s.csr = std::visit([](const auto& a) { return csr_bytes(a); }, s.loaded.matrix);
  return s;
}

std::string json_line(const std::string& layout, const std::string& file, const Sample& s,
                      const std::string& check) {
  index_t nnz = 0;
  std::visit([&](const auto& a) { nnz = a.nnz(); }, s.loaded.matrix);
  std::ostringstream os;
  os.precision(4);
  os << "{\"layout\": \"" << layout << "\", \"file\": \"" << file << "\", \"dtype\": \""
     << (s.loaded.is_complex() ? "complex128" : "float64") << "\", \"n\": " << s.loaded.order()
     << ", \"nnz\": " << nnz << ", \"file_mb\": " << static_cast<double>(fs::file_size(file)) / 1e6
     << ", \"csr_mb\": " << static_cast<double>(s.csr) / 1e6
     << ", \"peak_delta_mb\": " << static_cast<double>(s.peak_delta) / 1e6
     << ", \"ratio\": " << static_cast<double>(s.peak_delta) / static_cast<double>(s.csr)
     << ", \"retained_mb\": " << static_cast<double>(s.retained) / 1e6 << ", \"seconds\": " << s.seconds
     << ", \"check\": \"" << check << "\"}";
  return os.str();
}

template <class T>
std::string compare(const hs::LoadedMatrix& got, const CsrMatrix<T>& want) {
  if (!std::holds_alternative<CsrMatrix<T>>(got.matrix)) return "FAIL: scalar type";
  const auto& g = std::get<CsrMatrix<T>>(got.matrix);
  if (g.rowptr != want.rowptr || g.colind != want.colind) return "FAIL: pattern";
  if (g.values != want.values) return "FAIL: values";  // operator== on values: -0 == +0
  return "ok";
}

struct Options {
  index_t grid = 70;
  index_t mtx_grid = 45;
  fs::path dir;
  std::string only;
  bool keep = false;
};

template <class T>
void run_layouts(const Options& opt) {
  using F = hs::rebind_t<T, float>;
  const std::string tag = hs::is_complex_v<T> ? "c16" : "f8";
  const CsrMatrix<T> a = grid_hermitian<T>(opt.grid);
  const auto selected = [&](const std::string& layout) {
    return opt.only.empty() || (layout + "-" + tag).find(opt.only) != std::string::npos;
  };
  const auto run = [&](const std::string& layout, const std::string& ext,
                       const std::function<void(const fs::path&)>& write, const CsrMatrix<T>& want) {
    if (!selected(layout)) return;
    const fs::path file = opt.dir / (layout + "-" + tag + ext);
    write(file);
    std::string check;
    Sample s = measure_load(file);
    check = compare(s.loaded, want);
    std::cout << json_line(layout + "-" + tag, file.string(), s, check) << std::endl;
    if (!opt.keep) fs::remove(file);
  };

  const CsrMatrix<T> up = hs::triangle(a, true), lo = hs::triangle(a, false);
  run("npz-csr-full", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csr", a); }, a);
  run("npz-csr-upper", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csr", up); }, a);
  run("npz-csr-lower", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csr", lo); }, a);
  // CSC arrays of A are the CSR arrays of A^T = conj(A); of U: CSR of U^T = conj(L).
  run("npz-csc-full", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csc", conj_values(a)); }, a);
  run("npz-csc-upper", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csc", conj_values(lo)); }, a);
  run("npz-coo-full", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "coo", a); }, a);
  run("npz-coo-upper", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "coo", up); }, a);
  {
    // Non-native dtypes: single-precision values and int64 indices (values exactly representable
    // in float are not needed: the comparison is against the converted matrix).
    CsrMatrix<T> af = a;
    for (T& v : af.values) v = static_cast<T>(static_cast<F>(v));
    run("npz-csr-f4-i8", ".npz", [&](const fs::path& f) { write_scipy<F, std::int64_t>(f, "csr", af); }, af);
  }
  if (selected("npz-csr-nearherm-sym") || selected("npz-csr-nearherm-asym")) {
    const CsrMatrix<T> s = perturbed_symmetric(a);
    run("npz-csr-nearherm-sym", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csr", s); },
        hs::hermitian_part(s));
    const CsrMatrix<T> u = perturbed_asymmetric(a);
    run("npz-csr-nearherm-asym", ".npz", [&](const fs::path& f) { write_scipy<T, std::int32_t>(f, "csr", u); },
        hs::hermitian_part(u));
  }
  if (selected("mtx-general") || selected("mtx-symmetric")) {
    const CsrMatrix<T> small = grid_hermitian<T>(opt.mtx_grid);
    // Values are written in shortest round-trip form, so the file holds the matrix exactly.
    run("mtx-general", ".mtx", [&](const fs::path& f) { write_mtx(f, small, false); }, small);
    run("mtx-symmetric", ".mtx", [&](const fs::path& f) { write_mtx(f, small, true); }, small);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt;
    std::vector<fs::path> files;
    for (int k = 1; k < argc; ++k) {
      const std::string arg = argv[k];
      const auto next = [&]() -> std::string {
        if (k + 1 >= argc) HS_THROW(InvalidInput, arg << " needs a value");
        return argv[++k];
      };
      if (arg == "--grid") {
        opt.grid = std::stoll(next());
      } else if (arg == "--mtx-grid") {
        opt.mtx_grid = std::stoll(next());
      } else if (arg == "--dir") {
        opt.dir = next();
      } else if (arg == "--only") {
        opt.only = next();
      } else if (arg == "--keep") {
        opt.keep = true;
      } else if (arg == "--files") {
        for (++k; k < argc; ++k) files.emplace_back(argv[k]);
      } else {
        HS_THROW(InvalidInput, "unknown argument " << arg);
      }
    }
    // A fixed mmap threshold keeps large arrays out of the heap, so freed arrays leave the process
    // and every measurement starts from the same state.
    mallopt(M_MMAP_THRESHOLD, 1 << 20);
    mallopt(M_TRIM_THRESHOLD, 1 << 20);
    if (!files.empty()) {
      for (const fs::path& f : files) {
        Sample s = measure_load(f);
        std::cout << json_line("file", f.string(), s, "-") << std::endl;
      }
      return 0;
    }
    if (opt.dir.empty()) {
      const char* tmp = std::getenv("TMPDIR");
      opt.dir = fs::path(tmp ? tmp : "/tmp") / ("bench_memio_load." + std::to_string(::getpid()));
    }
    fs::create_directories(opt.dir);
    run_layouts<double>(opt);
    run_layouts<cplx>(opt);
    if (!opt.keep) fs::remove(opt.dir);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_memio_load: %s\n", e.what());
    return 2;
  }
}
