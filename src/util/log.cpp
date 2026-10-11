#include "hsolve/log.hpp"

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"

namespace hs {
namespace {

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

nlohmann::json to_json_value(const LogValue& v) {
  return std::visit([](const auto& x) { return nlohmann::json(x); }, v);
}

std::string to_text(const LogValue& v) {
  return std::visit(
      [](const auto& x) -> std::string {
        using X = std::decay_t<decltype(x)>;
        std::ostringstream os;
        if constexpr (std::is_same_v<X, double>) {
          os << std::setprecision(6) << x;
        } else if constexpr (std::is_same_v<X, bool>) {
          os << (x ? "true" : "false");
        } else {
          os << x;
        }
        return os.str();
      },
      v);
}

}  // namespace

Logger::Logger(int verbosity, const std::filesystem::path& jsonl_path)
    : verbosity_(verbosity), t0_ns_(now_ns()) {
  if (!jsonl_path.empty()) {
    file_ = std::make_unique<std::ofstream>(jsonl_path, std::ios::out | std::ios::trunc);
    if (!*file_) HS_THROW(Io, "cannot open log file " << jsonl_path.string());
  }
}

Logger::~Logger() = default;

double Logger::elapsed() const { return static_cast<double>(now_ns() - t0_ns_) * 1e-9; }

void Logger::event(const std::string& name, std::initializer_list<LogField> fields, int level) {
  const double t = elapsed();
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_) {
    nlohmann::json j;
    j["t"] = t;
    j["event"] = name;
    for (const auto& [k, v] : fields) j[k] = to_json_value(v);
    (*file_) << j.dump() << '\n';
    file_->flush();
  }
  if (verbosity_ >= level) {
    std::ostringstream os;
    os << "[" << std::fixed << std::setprecision(3) << t << "s] " << name;
    for (const auto& [k, v] : fields) os << ' ' << k << '=' << to_text(v);
    std::fprintf(stderr, "%s\n", os.str().c_str());
  }
}

void Logger::message_line(const char* kind, const std::string& message, int level) {
  const double t = elapsed();
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_) {
    nlohmann::json j;
    j["t"] = t;
    j["event"] = kind;
    j["message"] = message;
    (*file_) << j.dump() << '\n';
    file_->flush();
  }
  if (verbosity_ >= level) {
    std::fprintf(stderr, "[%.3fs] %s: %s\n", t, kind, message.c_str());
  }
}

}  // namespace hs
