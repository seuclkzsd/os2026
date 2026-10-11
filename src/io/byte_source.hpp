// Internal to src/io: sequential byte sources (memory, file; zip members live in zip.hpp) and
// little-endian field decoding shared by the npy, zip and MatrixMarket readers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace hs::io {

// numpy/zip fields are little-endian and so is every supported host (x86-64, aarch64). This lets
// payloads whose descr matches the host type be read straight into typed storage, and keeps byte
// swapping confined to explicitly big-endian ('>') numpy data.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "src/io assumes a little-endian host");

/// Unaligned little-endian load of an integer field.
template <class U>
U load_le(const std::byte* p) {
  U v;
  std::memcpy(&v, p, sizeof(U));
  return v;
}

/// Reverses the byte order of each of the `count` elements of `width` bytes starting at p.
void byte_swap_elements(std::byte* p, std::size_t count, std::size_t width);

/// Sequential reader with an exactly known length. Errors throw hs::Error(Io) naming the source.
class ByteSource {
 public:
  virtual ~ByteSource() = default;
  /// Reads exactly n bytes into dst; throws if fewer than n remain or reading fails.
  virtual void read(std::byte* dst, std::size_t n) = 0;
  /// Bytes not yet read.
  virtual std::uint64_t remaining() const = 0;
  /// Integrity checks once the caller has consumed everything (CRC-32 of zip members).
  virtual void finish() {}
  /// Description for error messages (path, or "path:member").
  virtual const std::string& name() const = 0;
};

class MemorySource final : public ByteSource {
 public:
  MemorySource(const std::byte* bytes, std::size_t nbytes, std::string name)
      : p_(bytes), n_(nbytes), name_(std::move(name)) {}
  void read(std::byte* dst, std::size_t n) override;
  std::uint64_t remaining() const override { return n_ - pos_; }
  const std::string& name() const override { return name_; }

 private:
  const std::byte* p_;
  std::size_t n_;
  std::size_t pos_ = 0;
  std::string name_;
};

/// A whole regular file. Large reads go straight from the OS into dst (no intermediate copy).
class FileSource final : public ByteSource {
 public:
  explicit FileSource(const std::filesystem::path& path);
  void read(std::byte* dst, std::size_t n) override;
  std::uint64_t remaining() const override { return size_ - pos_; }
  const std::string& name() const override { return name_; }

 private:
  std::ifstream in_;
  std::uint64_t size_ = 0;
  std::uint64_t pos_ = 0;
  std::string name_;
};

/// Opens `path` for binary reading; throws Io (with the path) if that fails.
std::ifstream open_binary(const std::filesystem::path& path);
/// Size of a regular file in bytes; throws Io if it does not exist or is not a regular file.
std::uint64_t regular_file_size(const std::filesystem::path& path);

}  // namespace hs::io
