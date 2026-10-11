// Internal to src/io: read-only access to zip archives (the container of .npz files).
// Spec: PKWARE APPNOTE.TXT 6.3. Supported: stored and deflate members, zip64 (extra field 0x0001,
// zip64 end-of-central-directory record + locator), archive comments, data descriptors (ignored:
// sizes come from the central directory). Rejected: encryption, other compression methods,
// multi-volume archives.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io/byte_source.hpp"

namespace hs::io {

/// One central-directory record with zip64 values already resolved.
struct ZipEntry {
  std::string name;
  std::uint16_t flags = 0;
  std::uint16_t method = 0;  ///< 0 stored, 8 deflate
  std::uint32_t crc32 = 0;
  std::uint64_t compressed_size = 0;
  std::uint64_t uncompressed_size = 0;
  std::uint64_t local_header_offset = 0;
};

/// The central directory of an archive. Member sizes and offsets are taken from the central
/// directory, never from local headers (numpy writes 0xFFFFFFFF there and puts the real sizes
/// into a zip64 extra field; streamed archives put them in data descriptors).
class ZipArchive {
 public:
  /// Reads and validates the central directory. Throws Io / Unsupported.
  explicit ZipArchive(const std::filesystem::path& path);

  const std::filesystem::path& path() const noexcept { return path_; }
  const std::vector<ZipEntry>& entries() const noexcept { return entries_; }

  /// Sequential reader of a member's uncompressed bytes (deflate is inflated on the fly in 1 MiB
  /// input chunks). remaining() starts at uncompressed_size; finish() checks that the deflate
  /// stream ends exactly there and that the CRC-32 matches (Io otherwise).
  std::unique_ptr<ByteSource> open(const ZipEntry& entry) const;

 private:
  std::filesystem::path path_;
  std::uint64_t data_end_ = 0;  ///< start of the central directory: member data lies before it
  std::vector<ZipEntry> entries_;
};

}  // namespace hs::io
