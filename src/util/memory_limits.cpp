// detect_memory_limits (include/hsolve/util.hpp): the memory limits of this process from the
// memory cgroup, the caller's declaration (HSOLVE_MEMORY_LIMIT_GB), the resource limits and the
// machine's MemTotal. Input of the automatic memory budget (hs::resolve_memory_budget, solver.hpp).
#include <sys/resource.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "hsolve/util.hpp"

namespace hs {

namespace {

constexpr double kGiB = 1073741824.0;
// cgroup v1 reports "no limit" as PAGE_COUNTER_MAX pages (9223372036854771712 bytes on x86-64);
// anything from 2^62 bytes on is treated as unlimited.
constexpr double kUnlimitedBytes = 4611686018427387904.0;

std::optional<std::string> read_text(const std::string& path) {
  std::ifstream in(path);
  if (!in) return std::nullopt;
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Bytes of a cgroup limit file: "max" (v2) or a number; nullopt when unreadable or malformed,
// +inf when unlimited.
std::optional<double> read_limit_bytes(const std::string& path) {
  const auto text = read_text(path);
  if (!text) return std::nullopt;
  const std::string v = trim(*text);
  if (v == "max") return HUGE_VAL;
  char* end = nullptr;
  errno = 0;
  const double bytes = std::strtod(v.c_str(), &end);
  if (v.empty() || errno != 0 || end == nullptr || *end != '\0' || !(bytes >= 0)) return std::nullopt;
  return bytes >= kUnlimitedBytes ? HUGE_VAL : bytes;
}

// "/a/b/c" -> {"/a/b/c", "/a/b", "/a", ""}: the cgroup and its ancestors up to the mount's root.
std::vector<std::string> ancestors(std::string path) {
  while (path.size() > 1 && path.back() == '/') path.pop_back();
  std::vector<std::string> out;
  if (path == "/") path.clear();
  while (!path.empty()) {
    out.push_back(path);
    const auto slash = path.find_last_of('/');
    path = slash == std::string::npos ? std::string() : path.substr(0, slash);
  }
  out.push_back("");
  return out;
}

struct CgroupLimit {
  double bytes = HUGE_VAL;  // +inf: no limit found
  std::string detail;
};

// The smallest limit file along the cgroup's path below `mount`. A cgroup path that does not
// exist below the mount (a container with a cgroup namespace shows the host's path, or "/") is
// read at the mount's root, which is then the container's own cgroup.
void scan_hierarchy(const std::string& mount, const std::string& cgroup_path, const char* file, CgroupLimit& out) {
  bool found_dir = false;
  for (const std::string& dir : ancestors(cgroup_path)) {
    const std::string path = mount + dir + "/" + file;
    const auto bytes = read_limit_bytes(path);
    if (!bytes) continue;
    found_dir = true;
    if (*bytes < out.bytes) {
      out.bytes = *bytes;
      out.detail = path;
    }
  }
  if (!found_dir) {
    const std::string path = mount + "/" + file;
    const auto bytes = read_limit_bytes(path);
    if (bytes && *bytes < out.bytes) {
      out.bytes = *bytes;
      out.detail = path;
    }
  }
}

MemoryLimitSource cgroup_source(const std::string& root) {
  MemoryLimitSource s{"cgroup", -1.0, ""};
  const auto self = read_text(root + "/proc/self/cgroup");
  if (!self) {
    s.detail = "no /proc/self/cgroup";
    return s;
  }
  CgroupLimit lim;
  bool any = false;
  std::istringstream lines(*self);
  for (std::string line; std::getline(lines, line);) {
    // hierarchy-ID:controller-list:cgroup-path
    const auto c1 = line.find(':');
    const auto c2 = c1 == std::string::npos ? std::string::npos : line.find(':', c1 + 1);
    if (c2 == std::string::npos) continue;
    const std::string id = line.substr(0, c1), controllers = line.substr(c1 + 1, c2 - c1 - 1);
    const std::string path = trim(line.substr(c2 + 1));
    if (id == "0" && controllers.empty()) {  // cgroup v2 (unified)
      any = true;
      scan_hierarchy(root + "/sys/fs/cgroup", path, "memory.max", lim);
    } else {
      std::istringstream list(controllers);
      for (std::string ctl; std::getline(list, ctl, ',');) {
        if (ctl != "memory") continue;
        any = true;
        const std::string mount = root + "/sys/fs/cgroup/memory";
        scan_hierarchy(mount, path, "memory.limit_in_bytes", lim);
        // v1 also states the effective limit of the whole ancestry in memory.stat.
        for (const std::string& base : {mount + path, mount}) {
          const auto stat = read_text(base + "/memory.stat");
          if (!stat) continue;
          std::istringstream st(*stat);
          for (std::string key, value; st >> key >> value;) {
            if (key != "hierarchical_memory_limit") continue;
            char* end = nullptr;
            const double bytes = std::strtod(value.c_str(), &end);
            if (end && *end == '\0' && bytes > 0 && bytes < kUnlimitedBytes && bytes < lim.bytes) {
              lim.bytes = bytes;
              lim.detail = base + "/memory.stat hierarchical_memory_limit";
            }
          }
          break;
        }
      }
    }
  }
  if (!any) {
    s.detail = "no memory cgroup";
  } else if (std::isinf(lim.bytes)) {
    s.detail = "unlimited";
  } else {
    s.gib = lim.bytes / kGiB;
    s.detail = lim.detail;
  }
  return s;
}

MemoryLimitSource env_source(const MemoryProbe& probe) {
  MemoryLimitSource s{"env", -1.0, ""};
  std::string value;
  if (probe.use_environment) {
    const char* v = std::getenv(kMemoryLimitEnv);
    if (v) value = v;
  } else {
    value = probe.env_value;
  }
  if (value.empty()) {
    s.detail = std::string(kMemoryLimitEnv) + " not set";
    return s;
  }
  char* end = nullptr;
  errno = 0;
  const double gib = std::strtod(value.c_str(), &end);
  if (errno != 0 || end == nullptr || trim(end) != "" || !(gib > 0) || !std::isfinite(gib)) {
    s.detail = std::string(kMemoryLimitEnv) + "=\"" + value + "\" ignored: not a positive number of GiB";
    return s;
  }
  s.gib = gib;
  s.detail = std::string(kMemoryLimitEnv) + "=" + value;
  return s;
}

MemoryLimitSource rlimit_source(const char* name, int resource, bool query, std::uint64_t given) {
  MemoryLimitSource s{name, -1.0, ""};
  double bytes = HUGE_VAL;
  if (query) {
    struct rlimit rl {};
    if (getrlimit(resource, &rl) != 0) {
      s.detail = "getrlimit failed";
      return s;
    }
    if (rl.rlim_cur != RLIM_INFINITY) bytes = static_cast<double>(rl.rlim_cur);
  } else if (given != std::numeric_limits<std::uint64_t>::max()) {
    bytes = static_cast<double>(given);
  }
  if (std::isinf(bytes) || bytes >= kUnlimitedBytes) {
    s.detail = "unlimited";
    return s;
  }
  s.gib = bytes / kGiB;
  s.detail = "getrlimit";
  return s;
}

MemoryLimitSource mem_total_source(const std::string& root) {
  MemoryLimitSource s{"mem_total", -1.0, ""};
  const std::string path = root + "/proc/meminfo";
  const auto text = read_text(path);
  if (!text) {
    s.detail = "no " + path;
    return s;
  }
  std::istringstream lines(*text);
  for (std::string line; std::getline(lines, line);) {
    if (line.rfind("MemTotal:", 0) != 0) continue;
    std::istringstream f(line.substr(9));
    double kib = 0;
    std::string unit;
    if (f >> kib >> unit && unit == "kB" && kib > 0) {
      s.gib = kib * 1024.0 / kGiB;
      s.detail = path;
    } else {
      s.detail = "malformed MemTotal in " + path;
    }
    return s;
  }
  s.detail = "no MemTotal in " + path;
  return s;
}

}  // namespace

MemoryLimits detect_memory_limits(const MemoryProbe& probe) {
  MemoryLimits m;
  m.sources.push_back(cgroup_source(probe.root));
  m.sources.push_back(env_source(probe));
  m.sources.push_back(rlimit_source("rlimit_as", RLIMIT_AS, probe.use_rlimits, probe.rlimit_as));
  m.sources.push_back(rlimit_source("rlimit_data", RLIMIT_DATA, probe.use_rlimits, probe.rlimit_data));
  m.sources.push_back(mem_total_source(probe.root));
  for (const auto& s : m.sources) {
    if (s.gib > 0 && (m.gib < 0 || s.gib < m.gib)) {
      m.gib = s.gib;
      m.source = s.name;
    }
  }
  return m;
}

}  // namespace hs
