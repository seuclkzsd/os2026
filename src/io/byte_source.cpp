// Byte sources and helpers shared by the src/io readers (see byte_source.hpp).
#include "io/byte_source.hpp"

#include <algorithm>
#include <system_error>
#include <utility>

#include "hsolve/error.hpp"

namespace hs::io {

void byte_swap_elements(std::byte* p, std::size_t count, std::size_t width) {
  if (width <= 1) return;
  for (std::size_t k = 0; k < count; ++k, p += width) std::reverse(p, p + width);
}

void MemorySource::read(std::byte* dst, std::size_t n) {
  if (n > n_ - pos_)
    HS_THROW(Io, name_ << ": unexpected end of data (need " << n << " bytes, " << n_ - pos_
                       << " left)");
  if (n > 0) std::memcpy(dst, p_ + pos_, n);
  pos_ += n;
}

std::ifstream open_binary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) HS_THROW(Io, "cannot open '" << path.string() << "' for reading");
  return in;
}

std::uint64_t regular_file_size(const std::filesystem::path& path) {
  std::error_code ec;
  const auto status = std::filesystem::status(path, ec);
  if (ec || !std::filesystem::exists(status))
    HS_THROW(Io, "file '" << path.string() << "' does not exist");
  if (!std::filesystem::is_regular_file(status))
    HS_THROW(Io, "'" << path.string() << "' is not a regular file");
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) HS_THROW(Io, "cannot stat '" << path.string() << "': " << ec.message());
  return size;
}

FileSource::FileSource(const std::filesystem::path& path)
    : size_(regular_file_size(path)), name_(path.string()) {
  in_ = open_binary(path);
}

void FileSource::read(std::byte* dst, std::size_t n) {
  if (n > size_ - pos_)
    HS_THROW(Io, name_ << ": unexpected end of file (need " << n << " bytes, " << size_ - pos_
                       << " left)");
  // istream::read takes a signed count; split huge reads to stay within std::streamsize.
  constexpr std::size_t kMaxChunk = std::size_t{1} << 30;
  for (std::size_t done = 0; done < n;) {
    const std::size_t len = std::min(kMaxChunk, n - done);
    in_.read(reinterpret_cast<char*>(dst + done), static_cast<std::streamsize>(len));
    if (static_cast<std::size_t>(in_.gcount()) != len)
      HS_THROW(Io, name_ << ": read failed at offset " << pos_ + done);
    done += len;
  }
  pos_ += n;
}

}  // namespace hs::io
