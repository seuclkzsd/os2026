// .npz reading (include/hsolve/io.hpp read_npz; internal NpzFile, see npz.hpp).
#include "io/npz.hpp"

#include <complex>
#include <cstdint>
#include <string_view>

#include "hsolve/error.hpp"

namespace hs {
namespace io {

namespace {
constexpr std::string_view kNpySuffix = ".npy";
}  // namespace

NpzFile::NpzFile(const std::filesystem::path& path) : zip_(path) {
  const auto& entries = zip_.entries();
  for (std::size_t k = 0; k < entries.size(); ++k) {
    const std::string& name = entries[k].name;
    if (name.size() < kNpySuffix.size() ||
        name.compare(name.size() - kNpySuffix.size(), kNpySuffix.size(), kNpySuffix) != 0)
      continue;
    std::string key = name.substr(0, name.size() - kNpySuffix.size());
    if (!index_.emplace(key, k).second)
      HS_THROW(Io, path.string() << ": duplicate npz member '" << name << "'");
    keys_.push_back(std::move(key));
  }
}

const ZipEntry& NpzFile::entry(const std::string& key) const {
  const auto it = index_.find(key);
  if (it == index_.end()) HS_THROW(Io, path().string() << ": npz has no array '" << key << "'");
  return zip_.entries()[it->second];
}

NpyHeader NpzFile::header(const std::string& key) const {
  const auto src = zip_.open(entry(key));
  return read_npy_header(*src);
}

NpyArray NpzFile::read(const std::string& key) const {
  const auto src = zip_.open(entry(key));
  NpyHeader h = read_npy_header(*src);
  return read_npy_payload(*src, std::move(h));
}

template <class T>
std::vector<T> NpzFile::read_as(const std::string& key, NpyHeader* header) const {
  const auto src = zip_.open(entry(key));
  const NpyHeader h = read_npy_header(*src);
  if (header != nullptr) *header = h;
  return read_npy_payload_as<T>(*src, h);
}

template std::vector<double> NpzFile::read_as<double>(const std::string&, NpyHeader*) const;
template std::vector<cplx> NpzFile::read_as<cplx>(const std::string&, NpyHeader*) const;
template std::vector<std::int32_t> NpzFile::read_as<std::int32_t>(const std::string&,
                                                                  NpyHeader*) const;
template std::vector<std::int64_t> NpzFile::read_as<std::int64_t>(const std::string&,
                                                                  NpyHeader*) const;

}  // namespace io

std::map<std::string, NpyArray> read_npz(const std::filesystem::path& path) {
  const io::NpzFile npz(path);
  std::map<std::string, NpyArray> out;
  for (const std::string& key : npz.keys()) out.emplace(key, npz.read(key));
  return out;
}

}  // namespace hs
