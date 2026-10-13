// Structured logging: JSON-lines events to a file + human-readable lines on stderr.
// Only the driver and the applications log; lower layers return statistics in structs.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>

namespace hs {

using LogValue = std::variant<std::int64_t, double, bool, std::string>;
using LogField = std::pair<std::string, LogValue>;

class Logger {
 public:
  /// verbosity: 0 = errors only, 1 = info, 2 = debug. jsonl_path empty = no file output.
  explicit Logger(int verbosity = 1, const std::filesystem::path& jsonl_path = {});
  ~Logger();
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  /// Machine-readable event; also printed to stderr when verbosity >= level. The JSON-lines
  /// record also carries the process's current and peak resident memory (rss_mb, peak_rss_mb).
  void event(const std::string& name, std::initializer_list<LogField> fields, int level = 1);
  void info(const std::string& message) { message_line("info", message, 1); }
  void debug(const std::string& message) { message_line("debug", message, 2); }
  void warn(const std::string& message) { message_line("warn", message, 0); }

  int verbosity() const noexcept { return verbosity_; }
  /// Seconds since the logger was created.
  double elapsed() const;

 private:
  void message_line(const char* kind, const std::string& message, int level);

  int verbosity_;
  std::unique_ptr<std::ofstream> file_;
  std::mutex mutex_;
  std::int64_t t0_ns_;
};

}  // namespace hs
