// Peak memory of hs::load_matrix on inputs with few entries per row (module memio, review fixes).
//
//   bench_memio_fix_rows [--n N] [--band B]... [--dir DIR] [--only SUBSTR] [--keep]
//
// bench/memio_load.cpp measures a 27-point stencil (~27 entries per row), where the per-row arrays
// of the loader (row pointers, placement cursors, per-row lengths: 8 bytes each per row) are small
// next to the 12-20 bytes per entry of the result. Mass matrices B are often diagonal or banded,
// and there those per-row arrays dominate. This bench generates a banded Hermitian matrix of order
// N (default 3e6) with half-bandwidth B (default 0 and 1: diagonal and tridiagonal; real and
// complex), writes it as scipy npz coo / csc / csr, full and upper-triangle storage (stored zip,
// int32 indices, as scipy.sparse.save_npz(compressed=False) writes it), and as MatrixMarket
// coordinate general / symmetric(hermitian), and loads every file once (complex only for B > 0:
// a diagonal Hermitian matrix is real).
//
// Method (as bench/memio_load.cpp): before each load free heap memory is returned to the kernel
// (malloc_trim) and the resident high-water mark is reset ("5" > /proc/self/clear_refs), then
//   peak_delta = VmHWM after the load - VmRSS before it,  ratio = peak_delta / csr_bytes,
// csr_bytes = capacity of the returned CSR arrays (8 (n+1) + (4 + sizeof(T)) nnz when tight).
// The loaded matrix is compared with the generated one. One JSON line per file on stdout; run it
// through tools/run/guard.py.
#include <malloc.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <charconv>
#include <chrono>
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

// ------------------------------------------------------------------------- test matrix

std::uint64_t splitmix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

double unit(std::uint64_t h) { return static_cast<double>(h >> 11) * 0x1.0p-53 * 2.0 - 1.0; }

// Hermitian, half-bandwidth `band`: a_ij = conj(a_ji) from a hash of (min, max).
template <class T>
CsrMatrix<T> banded_hermitian(index_t n, index_t band) {
  CsrMatrix<T> a;
  a.nrows = a.ncols = n;
  a.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (index_t i = 0; i < n; ++i)
    a.rowptr[i + 1] = a.rowptr[i] + (std::min(n - 1, i + band) - std::max(index_t{0}, i - band) + 1);
  a.colind.resize(static_cast<std::size_t>(a.rowptr[n]));
  a.values.resize(static_cast<std::size_t>(a.rowptr[n]));
  for (index_t i = 0; i < n; ++i) {
    index_t p = a.rowptr[i];
    for (index_t j = std::max(index_t{0}, i - band); j <= std::min(n - 1, i + band); ++j, ++p) {
      a.colind[p] = static_cast<std::int32_t>(j);
      const index_t lo = std::min(i, j), hi = std::max(i, j);
      const std::uint64_t h = splitmix(static_cast<std::uint64_t>(lo) * 0x10000000ull + hi);
      if (i == j) {
        a.values[p] = T(4.0 + unit(h));
      } else if constexpr (hs::is_complex_v<T>) {
        const T v(unit(h), unit(splitmix(h)));
        a.values[p] = i < j ? v : std::conj(v);
      } else {
        a.values[p] = unit(h);
      }
    }
  }
  return a;
}

template <class T>
CsrMatrix<T> conj_values(CsrMatrix<T> a) {
  for (T& v : a.values) v = hs::conj_if(v);
  return a;
}

// ------------------------------------------------------------------- npz (stored zip) writer

std::string npy_header(const char* descr, const std::vector<std::size_t>& shape) {
  std::string s = "(";
  for (const std::size_t d : shape) s += std::to_string(d) + ",";
  if (shape.size() > 1) s.pop_back();
  s += ")";
  std::string dict = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + s + ", }";
  const std::size_t total = 10 + dict.size() + 1;
  dict.append((64 - total % 64) % 64, ' ');
  dict += '\n';
  std::string out("\x93NUMPY\x01\x00", 8);
  out += static_cast<char>(dict.size() & 0xFF);
  out += static_cast<char>(dict.size() >> 8);
  return out + dict;
}

struct Member {
  std::string name;
  std::string header;
  const void* data;
  std::size_t bytes;
};

template <class V>
Member member(const std::string& key, const char* descr, const std::vector<V>& v) {
  return {key + ".npy", npy_header(descr, {v.size()}), v.data(), v.size() * sizeof(V)};
}

template <class U>
void put_le(std::string& s, U v) {
  for (std::size_t b = 0; b < sizeof(U); ++b) s += static_cast<char>((v >> (8 * b)) & 0xFF);
}

std::uint32_t crc_of(const Member& m) {
  uLong crc = crc32(0, reinterpret_cast<const Bytef*>(m.header.data()), static_cast<uInt>(m.header.size()));
  const auto* p = static_cast<const Bytef*>(m.data);
  for (std::size_t done = 0; done < m.bytes;) {  // crc32 takes a 32-bit length
    const std::size_t len = std::min<std::size_t>(m.bytes - done, std::size_t{1} << 30);
    crc = crc32(crc, p + done, static_cast<uInt>(len));
    done += len;
  }
  return static_cast<std::uint32_t>(crc);
}

void write_stored_zip(const fs::path& path, const std::vector<Member>& members) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  std::string central;
  std::uint64_t offset = 0;
  for (const Member& m : members) {
    const std::uint64_t size = m.header.size() + m.bytes;
    if (size >= 0xFFFFFFFFull || offset >= 0xFFFFFFFFull)
      HS_THROW(Unsupported, "bench zip writer: member too large");
    const std::uint32_t crc = crc_of(m);
    const auto common = [&](std::string& s) {
      put_le<std::uint16_t>(s, 0);  // flags
      put_le<std::uint16_t>(s, 0);  // stored
      put_le<std::uint32_t>(s, 0);  // time, date
      put_le(s, crc);
      put_le(s, static_cast<std::uint32_t>(size));
      put_le(s, static_cast<std::uint32_t>(size));
      put_le(s, static_cast<std::uint16_t>(m.name.size()));
      put_le<std::uint16_t>(s, 0);  // extra field length
    };
    std::string local;
    put_le<std::uint32_t>(local, 0x04034b50);
    put_le<std::uint16_t>(local, 20);
    common(local);
    local += m.name;
    put_le<std::uint32_t>(central, 0x02014b50);
    put_le<std::uint16_t>(central, 20);
    put_le<std::uint16_t>(central, 20);
    common(central);
    put_le<std::uint16_t>(central, 0);  // comment length
    put_le<std::uint16_t>(central, 0);  // disk
    put_le<std::uint16_t>(central, 0);  // internal attributes
    put_le<std::uint32_t>(central, 0);  // external attributes
    put_le(central, static_cast<std::uint32_t>(offset));
    central += m.name;
    out.write(local.data(), static_cast<std::streamsize>(local.size()));
    out.write(m.header.data(), static_cast<std::streamsize>(m.header.size()));
    out.write(static_cast<const char*>(m.data), static_cast<std::streamsize>(m.bytes));
    offset += local.size() + size;
  }
  std::string end;
  put_le<std::uint32_t>(end, 0x06054b50);
  put_le<std::uint32_t>(end, 0);  // this disk, central directory disk
  put_le(end, static_cast<std::uint16_t>(members.size()));
  put_le(end, static_cast<std::uint16_t>(members.size()));
  put_le(end, static_cast<std::uint32_t>(central.size()));
  put_le(end, static_cast<std::uint32_t>(offset));
  put_le<std::uint16_t>(end, 0);  // comment length
  out.write(central.data(), static_cast<std::streamsize>(central.size()));
  out.write(end.data(), static_cast<std::streamsize>(end.size()));
  if (!out) HS_THROW(Io, "bench: cannot write " << path.string());
}

// scipy.sparse.save_npz layout with int32 indices: coo from the CSR arrays of m, csr/csc from
// them directly (csc is given the CSR arrays of A^T).
template <class T>
void write_scipy(const fs::path& path, const std::string& fmt, const CsrMatrix<T>& m) {
  const std::vector<std::int64_t> shape{m.nrows, m.ncols};
  std::vector<Member> members;
  members.push_back({"format.npy", npy_header("|S3", {}), fmt.data(), 3});
  members.push_back(member("shape", "<i8", shape));
  members.push_back(member("data", hs::is_complex_v<T> ? "<c16" : "<f8", m.values));
  std::vector<std::int32_t> a1, a2;
  if (fmt == "coo") {
    a1.resize(m.colind.size());
    for (index_t i = 0; i < m.nrows; ++i)
      for (index_t p = m.rowptr[i]; p < m.rowptr[i + 1]; ++p) a1[p] = static_cast<std::int32_t>(i);
    members.push_back(member("row", "<i4", a1));
    members.push_back(member("col", "<i4", m.colind));
  } else {
    a2.assign(m.rowptr.begin(), m.rowptr.end());
    members.push_back(member("indices", "<i4", m.colind));
    members.push_back(member("indptr", "<i4", a2));
  }
  write_stored_zip(path, members);
}

void append_number(std::string& s, double v) {
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);
  s.append(buf, r.ptr);
}

// MatrixMarket coordinate: general (all entries) or symmetric/hermitian (lower triangle).
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

// ------------------------------------------------------------------------------ measuring

struct Sample {
  std::int64_t peak_delta = 0;
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
  s.csr = std::visit([](const auto& a) { return csr_bytes(a); }, s.loaded.matrix);
  return s;
}

template <class T>
std::string compare(const hs::LoadedMatrix& got, const CsrMatrix<T>& want) {
  if (!std::holds_alternative<CsrMatrix<T>>(got.matrix)) return "FAIL: scalar type";
  const auto& g = std::get<CsrMatrix<T>>(got.matrix);
  if (g.rowptr != want.rowptr || g.colind != want.colind) return "FAIL: pattern";
  if (g.values != want.values) return "FAIL: values";
  return "ok";
}

struct Options {
  index_t n = 3000000;
  std::vector<index_t> bands;
  fs::path dir;
  std::string only;
  bool keep = false;
};

template <class T>
void run_layouts(const Options& opt, index_t band) {
  const std::string tag = std::string(hs::is_complex_v<T> ? "c16" : "f8") + "-band" + std::to_string(band);
  const CsrMatrix<T> a = banded_hermitian<T>(opt.n, band);
  const CsrMatrix<T> up = hs::triangle(a, true);
  const auto run = [&](const std::string& layout, const std::string& ext,
                       const std::function<void(const fs::path&)>& write) {
    const std::string name = layout + "-" + tag;
    if (!opt.only.empty() && name.find(opt.only) == std::string::npos) return;
    const fs::path file = opt.dir / (name + ext);
    write(file);
    Sample s = measure_load(file);
    const std::string check = compare(s.loaded, a);
    index_t nnz = 0;
    std::visit([&](const auto& m) { nnz = m.nnz(); }, s.loaded.matrix);
    std::ostringstream os;
    os.precision(4);
    os << "{\"layout\": \"" << name << "\", \"n\": " << s.loaded.order() << ", \"nnz\": " << nnz
       << ", \"nnz_per_row\": " << static_cast<double>(nnz) / static_cast<double>(s.loaded.order())
       << ", \"file_mb\": " << static_cast<double>(fs::file_size(file)) / 1e6
       << ", \"csr_mb\": " << static_cast<double>(s.csr) / 1e6
       << ", \"peak_delta_mb\": " << static_cast<double>(s.peak_delta) / 1e6
       << ", \"ratio\": " << static_cast<double>(s.peak_delta) / static_cast<double>(s.csr)
       << ", \"seconds\": " << s.seconds << ", \"check\": \"" << check << "\"}";
    std::cout << os.str() << std::endl;
    if (!opt.keep) fs::remove(file);
  };
  run("npz-coo-full", ".npz", [&](const fs::path& f) { write_scipy(f, "coo", a); });
  run("npz-coo-upper", ".npz", [&](const fs::path& f) { write_scipy(f, "coo", up); });
  // CSC arrays of A are the CSR arrays of A^T = conj(A).
  run("npz-csc-full", ".npz", [&](const fs::path& f) { write_scipy(f, "csc", conj_values(a)); });
  run("npz-csr-full", ".npz", [&](const fs::path& f) { write_scipy(f, "csr", a); });
  run("npz-csr-upper", ".npz", [&](const fs::path& f) { write_scipy(f, "csr", up); });
  run("mtx-general", ".mtx", [&](const fs::path& f) { write_mtx(f, a, false); });
  run("mtx-symmetric", ".mtx", [&](const fs::path& f) { write_mtx(f, a, true); });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt;
    for (int k = 1; k < argc; ++k) {
      const std::string arg = argv[k];
      const auto next = [&]() -> std::string {
        if (k + 1 >= argc) HS_THROW(InvalidInput, arg << " needs a value");
        return argv[++k];
      };
      if (arg == "--n") {
        opt.n = std::stoll(next());
      } else if (arg == "--band") {
        opt.bands.push_back(std::stoll(next()));
      } else if (arg == "--dir") {
        opt.dir = next();
      } else if (arg == "--only") {
        opt.only = next();
      } else if (arg == "--keep") {
        opt.keep = true;
      } else {
        HS_THROW(InvalidInput, "unknown argument " << arg);
      }
    }
    if (opt.bands.empty()) opt.bands = {0, 1};
    // A fixed mmap threshold keeps large arrays out of the heap, so freed arrays leave the process
    // and every measurement starts from the same state.
    mallopt(M_MMAP_THRESHOLD, 1 << 20);
    mallopt(M_TRIM_THRESHOLD, 1 << 20);
    if (opt.dir.empty()) {
      const char* tmp = std::getenv("TMPDIR");
      opt.dir = fs::path(tmp ? tmp : "/tmp") / ("bench_memio_fix_rows." + std::to_string(::getpid()));
    }
    fs::create_directories(opt.dir);
    for (const index_t band : opt.bands) {
      run_layouts<double>(opt, band);
      // A diagonal Hermitian matrix is real: the loader would return it as such.
      if (band > 0) run_layouts<cplx>(opt, band);
    }
    if (!opt.keep) fs::remove(opt.dir);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_memio_fix_rows: %s\n", e.what());
    return 2;
  }
}
