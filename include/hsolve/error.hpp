// Error type used across the library. Input/runtime errors throw hs::Error; internal
// invariants use HS_ASSERT (active in all build types: a wrong eigenvalue is worse than a crash).
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace hs {

class Error : public std::runtime_error {
 public:
  enum class Code {
    InvalidInput,       ///< malformed or unsupported user input
    Io,                 ///< file system / format errors
    NotHermitian,       ///< matrix fails the Hermitian check
    Unsupported,        ///< valid request that this build/backend cannot serve
    FactorizationFailed,
    Singular,           ///< shifted matrix numerically singular (driver may perturb the shift)
    OutOfMemoryBudget,  ///< predicted memory exceeds the configured budget
    NotConverged,
    Internal,           ///< broken invariant (bug)
  };

  Error(Code code, const std::string& message) : std::runtime_error(message), code_(code) {}

  Code code() const noexcept { return code_; }

 private:
  Code code_;
};

const char* to_string(Error::Code code) noexcept;

namespace detail {
[[noreturn]] void assert_fail(const char* expr, const char* file, int line, const std::string& msg);
}  // namespace detail

}  // namespace hs

/// Internal invariant check (always enabled). Usage: HS_ASSERT(n > 0, "n=" << n);
#define HS_ASSERT(cond, msg)                                                  \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::ostringstream hs_assert_os_;                                       \
      hs_assert_os_ << msg;                                                   \
      ::hs::detail::assert_fail(#cond, __FILE__, __LINE__, hs_assert_os_.str()); \
    }                                                                         \
  } while (0)

/// Throw hs::Error with a streamed message. Usage: HS_THROW(InvalidInput, "bad n=" << n);
#define HS_THROW(code, msg)                                                   \
  do {                                                                        \
    std::ostringstream hs_throw_os_;                                          \
    hs_throw_os_ << msg;                                                      \
    throw ::hs::Error(::hs::Error::Code::code, hs_throw_os_.str());           \
  } while (0)
