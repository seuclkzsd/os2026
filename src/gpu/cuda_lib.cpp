// GPU module: run-time loading of the CUDA runtime and cuBLAS (cuda_lib.hpp).
#include "gpu/cuda_lib.hpp"

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace hs::gpu {
namespace {

struct Version {
  const char* cudart;
  const char* cublaslt;
  const char* cublas;
};
// CUDA 12, then CUDA 11 (same _v2 ABI for everything used here).
constexpr Version kVersions[] = {{"libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"},
                                 {"libcudart.so.11.0", "libcublasLt.so.11", "libcublas.so.11"}};

bool file_exists(const std::string& p) {
  struct stat st {};
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string dir_of_symbol(void* sym) {
  Dl_info info{};
  if (sym == nullptr || dladdr(sym, &info) == 0 || info.dli_fname == nullptr) return {};
  std::string f = info.dli_fname;
  const std::size_t slash = f.rfind('/');
  return slash == std::string::npos ? std::string() : f.substr(0, slash);
}

std::string file_of_symbol(void* sym) {
  Dl_info info{};
  if (sym == nullptr || dladdr(sym, &info) == 0 || info.dli_fname == nullptr) return {};
  return info.dli_fname;
}

void* open_lib(const std::string& name) {
  return dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
}

// Toolkit library directories under /usr/local, newest first: cuda-12*, then cuda-11*. (readdir,
// not glob(3): the toolchain's glob is a GLIBC_2.27 symbol, the cluster nodes have glibc 2.17.)
std::vector<std::string> default_dirs() {
  std::vector<std::string> dirs{"/usr/local/cuda/lib64"};
  std::vector<std::string> v12, v11;
  if (DIR* d = ::opendir("/usr/local")) {
    while (const dirent* e = ::readdir(d)) {
      const std::string name = e->d_name;
      if (name.rfind("cuda-12", 0) == 0) v12.push_back("/usr/local/" + name + "/lib64");
      if (name.rfind("cuda-11", 0) == 0) v11.push_back("/usr/local/" + name + "/lib64");
    }
    ::closedir(d);
  }
  for (auto* v : {&v12, &v11}) {
    std::sort(v->rbegin(), v->rend());
    dirs.insert(dirs.end(), v->begin(), v->end());
  }
  return dirs;
}

template <class F>
bool sym(void* h, const char* name, F* out, std::string* missing) {
  *out = reinterpret_cast<F>(dlsym(h, name));
  if (*out == nullptr) {
    if (!missing->empty()) *missing += ", ";
    *missing += name;
  }
  return *out != nullptr;
}

// Resolves every entry point; false (with the missing names) when one is absent.
bool resolve(void* rt, void* bl, CudaLib* L, std::string* why) {
  std::string missing;
  sym(rt, "cudaGetDeviceCount", &L->GetDeviceCount, &missing);
  sym(rt, "cudaSetDevice", &L->SetDevice, &missing);
  sym(rt, "cudaSetDeviceFlags", &L->SetDeviceFlags, &missing);
  sym(rt, "cudaDeviceGetAttribute", &L->DeviceGetAttribute, &missing);
  sym(rt, "cudaDeviceGetPCIBusId", &L->DeviceGetPCIBusId, &missing);
  sym(rt, "cudaMemGetInfo", &L->MemGetInfo, &missing);
  sym(rt, "cudaMalloc", &L->Malloc, &missing);
  sym(rt, "cudaFree", &L->Free, &missing);
  sym(rt, "cudaHostAlloc", &L->HostAlloc, &missing);
  sym(rt, "cudaFreeHost", &L->FreeHost, &missing);
  sym(rt, "cudaHostRegister", &L->HostRegister, &missing);
  sym(rt, "cudaHostUnregister", &L->HostUnregister, &missing);
  sym(rt, "cudaMemcpyAsync", &L->MemcpyAsync, &missing);
  sym(rt, "cudaMemcpy2DAsync", &L->Memcpy2DAsync, &missing);
  sym(rt, "cudaStreamCreateWithFlags", &L->StreamCreateWithFlags, &missing);
  sym(rt, "cudaStreamDestroy", &L->StreamDestroy, &missing);
  sym(rt, "cudaStreamSynchronize", &L->StreamSynchronize, &missing);
  sym(rt, "cudaStreamWaitEvent", &L->StreamWaitEvent, &missing);
  sym(rt, "cudaEventCreateWithFlags", &L->EventCreateWithFlags, &missing);
  sym(rt, "cudaEventDestroy", &L->EventDestroy, &missing);
  sym(rt, "cudaEventRecord", &L->EventRecord, &missing);
  sym(rt, "cudaEventSynchronize", &L->EventSynchronize, &missing);
  sym(rt, "cudaEventQuery", &L->EventQuery, &missing);
  sym(rt, "cudaDeviceSynchronize", &L->DeviceSynchronize, &missing);
  sym(rt, "cudaDriverGetVersion", &L->DriverGetVersion, &missing);
  sym(rt, "cudaRuntimeGetVersion", &L->RuntimeGetVersion, &missing);
  sym(rt, "cudaGetLastError", &L->GetLastError, &missing);
  sym(rt, "cudaGetErrorString", &L->GetErrorString, &missing);
  sym(bl, "cublasCreate_v2", &L->BlasCreate, &missing);
  sym(bl, "cublasDestroy_v2", &L->BlasDestroy, &missing);
  sym(bl, "cublasSetStream_v2", &L->BlasSetStream, &missing);
  sym(bl, "cublasSetMathMode", &L->BlasSetMathMode, &missing);
  sym(bl, "cublasGetVersion_v2", &L->BlasGetVersion, &missing);
  sym(bl, "cublasDgemm_v2", &L->Dgemm, &missing);
  if (!missing.empty()) {
    *why = "missing CUDA entry points: " + missing;
    return false;
  }
  L->cudart_path = file_of_symbol(reinterpret_cast<void*>(L->GetDeviceCount));
  L->cublas_path = file_of_symbol(reinterpret_cast<void*>(L->Dgemm));
  return true;
}

// cuBLAS (and cublasLt first) from `dir`, else by soname.
void* open_cublas(const std::string& dir, const Version& v) {
  if (!dir.empty()) {
    if (file_exists(dir + "/" + v.cublaslt)) open_lib(dir + "/" + v.cublaslt);
    if (file_exists(dir + "/" + v.cublas))
      if (void* h = open_lib(dir + "/" + v.cublas)) return h;
  }
  open_lib(v.cublaslt);
  return open_lib(v.cublas);
}

// Tries one directory ("" = the loader's search); true with both handles on success.
bool try_dir(const std::string& dir, void** rt, void** bl) {
  for (const Version& v : kVersions) {
    if (!dir.empty() && !(file_exists(dir + "/" + v.cudart) && file_exists(dir + "/" + v.cublas)))
      continue;
    void* r = open_lib(dir.empty() ? std::string(v.cudart) : dir + "/" + v.cudart);
    if (r == nullptr) continue;
    const std::string rdir = dir.empty() ? dir_of_symbol(dlsym(r, "cudaGetDeviceCount")) : dir;
    void* b = open_cublas(rdir, v);
    if (b == nullptr) continue;  // the runtime stays loaded (harmless); try the next version / place
    *rt = r;
    *bl = b;
    return true;
  }
  return false;
}

struct Loaded {
  CudaLib lib;
  bool ok = false;
  std::string why;
};

Loaded do_load() {
  Loaded out;
  std::vector<std::string> dirs;
  if (const char* d = std::getenv("HS_CUDA_LIBDIR"); d != nullptr && *d != '\0') dirs.emplace_back(d);
  dirs.emplace_back("");  // the dynamic loader's search
  for (const std::string& d : default_dirs()) dirs.push_back(d);
  void* rt = nullptr;
  void* bl = nullptr;
  for (const std::string& d : dirs)
    if (try_dir(d, &rt, &bl)) break;
  if (rt == nullptr || bl == nullptr) {
    out.why = "no CUDA runtime / cuBLAS library found (libcudart.so.12 + libcublas.so.12 or the CUDA 11 "
              "pair; searched HS_CUDA_LIBDIR, the loader's path, /usr/local/cuda*/lib64)";
    return out;
  }
  out.ok = resolve(rt, bl, &out.lib, &out.why);
  return out;
}

}  // namespace

const CudaLib* load(std::string* why) {
  static std::once_flag once;
  static Loaded* loaded = nullptr;  // never freed: the libraries stay loaded for the process
  std::call_once(once, [] { loaded = new Loaded(do_load()); });
  if (!loaded->ok) {
    if (why != nullptr) *why = loaded->why;
    return nullptr;
  }
  return &loaded->lib;
}

std::string error_text(const CudaLib* lib, cudaError_t e) {
  std::string s = "cuda error " + std::to_string(e);
  if (lib != nullptr && lib->GetErrorString != nullptr) {
    const char* t = lib->GetErrorString(e);
    if (t != nullptr) s += std::string(" (") + t + ")";
  }
  return s;
}

}  // namespace hs::gpu
