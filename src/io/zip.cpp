// Zip archive reader for .npz files (see zip.hpp).
#include "io/zip.hpp"

#include <zlib.h>

#include <algorithm>
#include <ios>
#include <limits>
#include <sstream>
#include <string_view>

#include "hsolve/error.hpp"

namespace hs::io {
namespace {

constexpr std::uint32_t kLocalHeaderSig = 0x04034b50;
constexpr std::uint32_t kCentralHeaderSig = 0x02014b50;
constexpr std::uint32_t kEndSig = 0x06054b50;
constexpr std::uint32_t kZip64EndSig = 0x06064b50;
constexpr std::uint32_t kZip64LocatorSig = 0x07064b50;
constexpr std::size_t kLocalHeaderSize = 30;
constexpr std::size_t kEndSize = 22;
constexpr std::size_t kZip64LocatorSize = 20;
constexpr std::size_t kZip64EndSize = 56;
constexpr std::size_t kMaxComment = 0xFFFF;
constexpr std::uint16_t kZip64ExtraId = 0x0001;
constexpr std::uint16_t kFlagEncrypted = 0x0001;
constexpr std::uint16_t kFlagStrongEncryption = 0x0040;
constexpr std::uint16_t kMethodStored = 0;
constexpr std::uint16_t kMethodDeflate = 8;
constexpr std::uint32_t kSat32 = 0xFFFFFFFF;  // "value is in the zip64 extra field"
constexpr std::uint16_t kSat16 = 0xFFFF;
constexpr std::size_t kInputChunk = std::size_t{1} << 20;
constexpr std::size_t kMaxIoChunk = std::size_t{1} << 30;  // stream/zlib counts are 32-bit

std::string hex32(std::uint32_t v) {
  std::ostringstream os;
  os << "0x" << std::hex << v;
  return os.str();
}

void read_at(std::ifstream& in, std::uint64_t offset, std::byte* dst, std::size_t n,
             const std::string& name) {
  in.clear();
  in.seekg(static_cast<std::streamoff>(offset));
  for (std::size_t done = 0; done < n;) {
    const std::size_t len = std::min(kMaxIoChunk, n - done);
    in.read(reinterpret_cast<char*>(dst + done), static_cast<std::streamsize>(len));
    if (static_cast<std::size_t>(in.gcount()) != len)
      HS_THROW(Io, name << ": cannot read " << n << " bytes at offset " << offset);
    done += len;
  }
}

std::vector<std::byte> read_range(std::ifstream& in, std::uint64_t offset, std::size_t n,
                                  const std::string& name) {
  std::vector<std::byte> buf(n);
  read_at(in, offset, buf.data(), n, name);
  return buf;
}

// Bounds-checked little-endian cursor over a record.
class Cursor {
 public:
  Cursor(const std::byte* p, std::size_t n, const std::string& what) : p_(p), n_(n), what_(what) {}
  template <class U>
  U get() {
    need(sizeof(U));
    const U v = load_le<U>(p_ + pos_);
    pos_ += sizeof(U);
    return v;
  }
  const std::byte* take(std::size_t n) {
    need(n);
    const std::byte* p = p_ + pos_;
    pos_ += n;
    return p;
  }
  std::size_t left() const { return n_ - pos_; }

 private:
  void need(std::size_t n) const {
    if (n > n_ - pos_) HS_THROW(Io, what_ << ": truncated zip record");
  }
  const std::byte* p_;
  std::size_t n_;
  std::size_t pos_ = 0;
  const std::string& what_;
};

// Streams one member. Not copyable or movable: zlib keeps a pointer back to z_.
class ZipMemberSource final : public ByteSource {
 public:
  ZipMemberSource(const std::filesystem::path& path, std::uint64_t data_end, const ZipEntry& e)
      : entry_(e), name_(path.string() + ":" + e.name) {
    in_ = open_binary(path);
    // The local header repeats name and extra field with possibly different lengths (numpy's
    // local zip64 extra is 20 bytes, the central one may be absent); only these lengths are
    // taken from it to locate the data.
    std::byte hdr[kLocalHeaderSize];
    read_at(in_, e.local_header_offset, hdr, kLocalHeaderSize, name_);
    Cursor c(hdr, kLocalHeaderSize, name_);
    if (c.get<std::uint32_t>() != kLocalHeaderSig)
      HS_THROW(Io, name_ << ": bad local header signature at offset " << e.local_header_offset);
    c.take(22);  // version, flags, method, time, date, crc, sizes: central directory is authoritative
    const auto name_len = c.get<std::uint16_t>();
    const auto extra_len = c.get<std::uint16_t>();
    std::vector<std::byte> local_name(name_len);
    read_at(in_, e.local_header_offset + kLocalHeaderSize, local_name.data(), name_len, name_);
    if (std::string_view(reinterpret_cast<const char*>(local_name.data()), name_len) != e.name)
      HS_THROW(Io, name_ << ": local header names a different member (corrupt offsets)");
    in_pos_ = e.local_header_offset + kLocalHeaderSize + name_len + extra_len;
    if (in_pos_ > data_end || e.compressed_size > data_end - in_pos_)
      HS_THROW(Io, name_ << ": member data extends into the central directory");
    in_left_ = e.compressed_size;
    if (e.method == kMethodDeflate) {
      if (inflateInit2(&z_, -MAX_WBITS) != Z_OK)  // raw deflate: zip members have no zlib header
        HS_THROW(Io, name_ << ": zlib initialisation failed");
      inflating_ = true;
      inbuf_.resize(static_cast<std::size_t>(std::min<std::uint64_t>(kInputChunk, in_left_)));
    }
  }
  ZipMemberSource(const ZipMemberSource&) = delete;
  ZipMemberSource& operator=(const ZipMemberSource&) = delete;
  ~ZipMemberSource() override {
    if (inflating_) inflateEnd(&z_);
  }

  void read(std::byte* dst, std::size_t n) override {
    if (n > remaining())
      HS_THROW(Io, name_ << ": read of " << n << " bytes past the end of the member ("
                         << remaining() << " left)");
    if (n == 0) return;
    if (entry_.method == kMethodStored) {
      read_at(in_, in_pos_, dst, n, name_);
      in_pos_ += n;
      in_left_ -= n;
    } else {
      inflate_into(dst, n);
    }
    for (std::size_t done = 0; done < n;) {
      const std::size_t len = std::min(kMaxIoChunk, n - done);
      crc_ = crc32_z(crc_, reinterpret_cast<const Bytef*>(dst + done), len);
      done += len;
    }
    produced_ += n;
  }

  std::uint64_t remaining() const override { return entry_.uncompressed_size - produced_; }

  void finish() override {
    HS_ASSERT(produced_ == entry_.uncompressed_size,
              name_ << ": finish() after " << produced_ << " of " << entry_.uncompressed_size);
    if (inflating_ && !stream_end_) {
      // The declared size has been produced; the deflate stream must end here.
      Bytef probe = 0;
      while (!stream_end_) {
        if (z_.avail_in == 0) refill();
        z_.next_out = &probe;
        z_.avail_out = 1;
        const int rc = inflate(&z_, Z_NO_FLUSH);
        if (z_.avail_out == 0)
          HS_THROW(Io, name_ << ": member holds more data than its declared size "
                             << entry_.uncompressed_size);
        check_inflate(rc);
      }
    }
    if (crc_ != entry_.crc32)
      HS_THROW(Io, name_ << ": CRC-32 mismatch (stored " << hex32(entry_.crc32) << ", computed "
                         << hex32(static_cast<std::uint32_t>(crc_)) << "): archive is corrupt");
  }

  const std::string& name() const override { return name_; }

 private:
  void refill() {
    const auto len = static_cast<std::size_t>(std::min<std::uint64_t>(inbuf_.size(), in_left_));
    read_at(in_, in_pos_, inbuf_.data(), len, name_);
    in_pos_ += len;
    in_left_ -= len;
    z_.next_in = reinterpret_cast<Bytef*>(inbuf_.data());
    z_.avail_in = static_cast<uInt>(len);
  }

  // Interprets an inflate() return code; sets stream_end_ or throws.
  void check_inflate(int rc) {
    if (rc == Z_STREAM_END) {
      stream_end_ = true;
    } else if (rc == Z_BUF_ERROR) {
      // No progress possible: with output space available this means the input ran out.
      if (z_.avail_in == 0 && in_left_ == 0)
        HS_THROW(Io, name_ << ": deflate data is truncated");
    } else if (rc != Z_OK) {
      HS_THROW(Io, name_ << ": corrupt deflate data (zlib: " << (z_.msg ? z_.msg : "error") << ")");
    }
  }

  void inflate_into(std::byte* dst, std::size_t n) {
    std::size_t done = 0;
    while (done < n) {
      if (stream_end_)
        HS_THROW(Io, name_ << ": deflate stream ends before the declared size "
                           << entry_.uncompressed_size);
      if (z_.avail_in == 0 && in_left_ > 0) refill();
      const std::size_t chunk = std::min(kMaxIoChunk, n - done);
      z_.next_out = reinterpret_cast<Bytef*>(dst + done);
      z_.avail_out = static_cast<uInt>(chunk);
      const int rc = inflate(&z_, Z_NO_FLUSH);
      done += chunk - z_.avail_out;
      check_inflate(rc);
    }
  }

  std::ifstream in_;
  ZipEntry entry_;
  std::string name_;
  std::uint64_t in_pos_ = 0;   ///< file offset of the next compressed byte
  std::uint64_t in_left_ = 0;  ///< compressed bytes not yet read from the file
  std::uint64_t produced_ = 0;
  uLong crc_ = 0;
  z_stream z_{};
  bool inflating_ = false;
  bool stream_end_ = false;
  std::vector<std::byte> inbuf_;
};

}  // namespace

ZipArchive::ZipArchive(const std::filesystem::path& path) : path_(path) {
  const std::string name = path.string();
  const std::uint64_t file_size = regular_file_size(path);
  std::ifstream in = open_binary(path);
  if (file_size < kEndSize) HS_THROW(Io, name << ": not a zip archive (too short)");

  // The end record sits at the end, followed by a comment of at most 64 KiB. Candidates (the
  // signature, with a central directory that lies before the record unless zip64 holds its
  // location) are scanned backwards from the end. Preferred: the last one whose comment length
  // accounts exactly for the bytes after it, so signature bytes inside a comment are not mistaken
  // for the record. Failing that, like Python's zipfile (hence np.load and sp.load_npz), the last
  // one whose comment fits: tools that pad files append bytes after the archive.
  const auto tail_len = static_cast<std::size_t>(std::min<std::uint64_t>(file_size, kEndSize + kMaxComment));
  const std::uint64_t tail_offset = file_size - tail_len;
  const std::vector<std::byte> tail = read_range(in, tail_offset, tail_len, name);
  constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
  std::size_t exact_pos = kNone, fitting_pos = kNone;
  for (std::size_t pos = tail_len - kEndSize + 1; pos-- > 0;) {
    const std::byte* rec = tail.data() + pos;
    if (load_le<std::uint32_t>(rec) != kEndSig) continue;
    const auto cd_size32 = load_le<std::uint32_t>(rec + 12);
    const auto cd_offset32 = load_le<std::uint32_t>(rec + 16);
    const std::uint64_t at = tail_offset + pos;
    if (cd_size32 != kSat32 && cd_offset32 != kSat32 &&
        (cd_offset32 > at || cd_size32 > at - cd_offset32))
      continue;
    const std::size_t after = tail_len - pos - kEndSize;
    const std::size_t comment_len = load_le<std::uint16_t>(rec + 20);
    if (comment_len == after) {
      exact_pos = pos;
      break;
    }
    if (comment_len < after && fitting_pos == kNone) fitting_pos = pos;
  }
  const std::size_t end_pos = exact_pos != kNone ? exact_pos : fitting_pos;
  if (end_pos == kNone)
    HS_THROW(Io, name << ": not a zip archive (no end-of-central-directory record)");
  const std::uint64_t end_offset = tail_offset + end_pos;

  Cursor end(tail.data() + end_pos, kEndSize, name);
  end.get<std::uint32_t>();  // signature
  const auto disk = end.get<std::uint16_t>();
  const auto cd_disk = end.get<std::uint16_t>();
  const auto entries_on_disk = end.get<std::uint16_t>();
  std::uint64_t n_entries = end.get<std::uint16_t>();
  std::uint64_t cd_size = end.get<std::uint32_t>();
  std::uint64_t cd_offset = end.get<std::uint32_t>();
  bool multi_volume = disk != 0 || cd_disk != 0 || entries_on_disk != n_entries;
  std::uint64_t cd_limit = end_offset;  // the central directory ends before this offset

  // zip64: a locator immediately before the end record points to the zip64 end record, whose
  // 64-bit counts/offsets replace the (then usually saturated) 16/32-bit ones.
  if (end_offset >= kZip64LocatorSize) {
    const std::vector<std::byte> loc =
        read_range(in, end_offset - kZip64LocatorSize, kZip64LocatorSize, name);
    if (load_le<std::uint32_t>(loc.data()) == kZip64LocatorSig) {
      Cursor lc(loc.data(), loc.size(), name);
      lc.get<std::uint32_t>();
      const auto z64_disk = lc.get<std::uint32_t>();
      const auto z64_offset = lc.get<std::uint64_t>();
      const auto total_disks = lc.get<std::uint32_t>();
      if (z64_offset > end_offset - kZip64LocatorSize ||
          end_offset - kZip64LocatorSize - z64_offset < kZip64EndSize)
        HS_THROW(Io, name << ": zip64 end record offset " << z64_offset << " is invalid");
      const std::vector<std::byte> rec = read_range(in, z64_offset, kZip64EndSize, name);
      Cursor rc(rec.data(), rec.size(), name);
      if (rc.get<std::uint32_t>() != kZip64EndSig)
        HS_THROW(Io, name << ": bad zip64 end-of-central-directory signature");
      rc.take(12);  // record size, version made by, version needed
      const auto disk64 = rc.get<std::uint32_t>();
      const auto cd_disk64 = rc.get<std::uint32_t>();
      const auto on_disk64 = rc.get<std::uint64_t>();
      n_entries = rc.get<std::uint64_t>();
      cd_size = rc.get<std::uint64_t>();
      cd_offset = rc.get<std::uint64_t>();
      multi_volume = z64_disk != 0 || total_disks > 1 || disk64 != 0 || cd_disk64 != 0 ||
                     on_disk64 != n_entries;
      cd_limit = z64_offset;
    }
  }
  if (multi_volume) HS_THROW(Unsupported, name << ": multi-volume zip archives are not supported");
  if (cd_offset > cd_limit || cd_size > cd_limit - cd_offset)
    HS_THROW(Io, name << ": central directory (offset " << cd_offset << ", " << cd_size
                      << " bytes) lies outside the archive");
  data_end_ = cd_offset;

  const std::vector<std::byte> cd = read_range(in, cd_offset, static_cast<std::size_t>(cd_size), name);
  Cursor c(cd.data(), cd.size(), name);
  entries_.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(n_entries, cd_size / 46)));
  for (std::uint64_t k = 0; k < n_entries; ++k) {
    if (c.get<std::uint32_t>() != kCentralHeaderSig)
      HS_THROW(Io, name << ": bad central directory record " << k);
    ZipEntry e;
    c.take(4);  // version made by, version needed
    e.flags = c.get<std::uint16_t>();
    e.method = c.get<std::uint16_t>();
    c.take(4);  // time, date
    e.crc32 = c.get<std::uint32_t>();
    const auto csize32 = c.get<std::uint32_t>();
    const auto usize32 = c.get<std::uint32_t>();
    const auto name_len = c.get<std::uint16_t>();
    const auto extra_len = c.get<std::uint16_t>();
    const auto comment_len = c.get<std::uint16_t>();
    const auto disk_start16 = c.get<std::uint16_t>();
    c.take(6);  // internal / external attributes
    const auto offset32 = c.get<std::uint32_t>();
    const std::byte* name_bytes = c.take(name_len);
    e.name.assign(reinterpret_cast<const char*>(name_bytes), name_len);
    const std::byte* extra = c.take(extra_len);
    c.take(comment_len);
    e.compressed_size = csize32;
    e.uncompressed_size = usize32;
    e.local_header_offset = offset32;
    std::uint64_t disk_start = disk_start16;

    // zip64 extra field: holds, in this fixed order, exactly those values whose fixed-size field
    // is saturated (APPNOTE 4.5.3).
    bool have_zip64 = false;
    Cursor x(extra, extra_len, name);
    while (x.left() >= 4) {
      const auto id = x.get<std::uint16_t>();
      const auto size = x.get<std::uint16_t>();
      const std::byte* data = x.take(size);
      if (id != kZip64ExtraId) continue;
      have_zip64 = true;
      Cursor z(data, size, name);
      if (usize32 == kSat32) e.uncompressed_size = z.get<std::uint64_t>();
      if (csize32 == kSat32) e.compressed_size = z.get<std::uint64_t>();
      if (offset32 == kSat32) e.local_header_offset = z.get<std::uint64_t>();
      if (disk_start16 == kSat16) disk_start = z.get<std::uint32_t>();
    }
    if (!have_zip64 && (usize32 == kSat32 || csize32 == kSat32 || offset32 == kSat32))
      HS_THROW(Io, name << ": member '" << e.name << "' lacks its zip64 extra field");
    if (disk_start != 0)
      HS_THROW(Unsupported, name << ": multi-volume zip archives are not supported");
    if (e.flags & (kFlagEncrypted | kFlagStrongEncryption))
      HS_THROW(Unsupported, name << ": member '" << e.name << "' is encrypted");
    if (e.method != kMethodStored && e.method != kMethodDeflate)
      HS_THROW(Unsupported, name << ": member '" << e.name << "' uses compression method "
                                 << e.method << " (only stored and deflate are supported)");
    if (e.method == kMethodStored && e.compressed_size != e.uncompressed_size)
      HS_THROW(Io, name << ": stored member '" << e.name << "' has different compressed ("
                        << e.compressed_size << ") and uncompressed (" << e.uncompressed_size
                        << ") sizes");
    if (e.local_header_offset >= data_end_)
      HS_THROW(Io, name << ": member '" << e.name << "' starts inside the central directory");
    entries_.push_back(std::move(e));
  }
}

std::unique_ptr<ByteSource> ZipArchive::open(const ZipEntry& entry) const {
  return std::make_unique<ZipMemberSource>(path_, data_end_, entry);
}

}  // namespace hs::io
