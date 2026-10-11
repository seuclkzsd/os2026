#include "hsolve/error.hpp"

#include <sstream>

namespace hs {

const char* to_string(Error::Code code) noexcept {
  switch (code) {
    case Error::Code::InvalidInput: return "invalid_input";
    case Error::Code::Io: return "io";
    case Error::Code::NotHermitian: return "not_hermitian";
    case Error::Code::Unsupported: return "unsupported";
    case Error::Code::FactorizationFailed: return "factorization_failed";
    case Error::Code::Singular: return "singular";
    case Error::Code::OutOfMemoryBudget: return "out_of_memory_budget";
    case Error::Code::NotConverged: return "not_converged";
    case Error::Code::Internal: return "internal";
  }
  return "unknown";
}

namespace detail {
void assert_fail(const char* expr, const char* file, int line, const std::string& msg) {
  std::ostringstream os;
  os << "internal invariant violated: " << expr << " at " << file << ":" << line;
  if (!msg.empty()) os << " (" << msg << ")";
  throw Error(Error::Code::Internal, os.str());
}
}  // namespace detail

}  // namespace hs
