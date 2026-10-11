// Internal to src/io: lazily read .npz archives. Members are inflated one at a time on request,
// so a caller that needs a few arrays (the scipy loader) never holds the others, nor a raw copy
// of a payload whose dtype already matches the requested type.
#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "hsolve/io.hpp"
#include "io/npy_format.hpp"
#include "io/zip.hpp"

namespace hs::io {

class NpzFile {
 public:
  /// Reads the central directory. Members whose names do not end in ".npy" are not arrays and
  /// are not listed. Throws Io on a malformed archive or duplicate member names.
  explicit NpzFile(const std::filesystem::path& path);

  const std::filesystem::path& path() const noexcept { return zip_.path(); }
  /// Array keys (member names without ".npy") in archive order.
  const std::vector<std::string>& keys() const noexcept { return keys_; }
  bool contains(const std::string& key) const { return index_.count(key) != 0; }

  /// Header of one array; inflates only the first bytes of the member.
  NpyHeader header(const std::string& key) const;
  /// The whole array as raw bytes (CRC verified).
  NpyArray read(const std::string& key) const;
  /// The array converted to T (npy_to_vector rules), read in place when the dtype matches.
  /// If `header` is not null it receives the member's header. T in {double, cplx, float,
  /// int32_t, int64_t}.
  template <class T>
  std::vector<T> read_as(const std::string& key, NpyHeader* header = nullptr) const;

 private:
  const ZipEntry& entry(const std::string& key) const;

  ZipArchive zip_;
  std::vector<std::string> keys_;
  std::map<std::string, std::size_t> index_;  ///< key -> position in zip_.entries()
};

}  // namespace hs::io
