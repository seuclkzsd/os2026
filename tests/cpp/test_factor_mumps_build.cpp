// The MUMPS this process runs (P9 "factspeed", CMakeLists.txt HSOLVE_MUMPS_ROOT): the libraries the
// dynamic loader mapped are the configured ones - those of tools/dev/build_mumps.sh's install when
// HSOLVE_MUMPS_ROOT is set (its lib/ must win over the environment's in the RPATH), else the
// environment's - and they are MUMPS 5.8.2 with the header's ABI (a factorization works).
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "factor/mumps_api.hpp"
#include "factor/mumps_factorization.hpp"
#include "hsolve/factor.hpp"
#include "support/fixtures.hpp"

#ifndef HSOLVE_MUMPS_ROOT
#define HSOLVE_MUMPS_ROOT ""
#endif

namespace {

// Paths of the mapped objects whose file name starts with `stem` ("libdmumps").
std::vector<std::string> mapped(const std::string& stem) {
  std::vector<std::string> out;
  std::ifstream f("/proc/self/maps");
  std::string line;
  while (std::getline(f, line)) {
    const auto slash = line.find('/');
    if (slash == std::string::npos) continue;
    const std::string path = line.substr(slash);
    const std::string name = path.substr(path.rfind('/') + 1);
    if (name.rfind(stem, 0) == 0 && (out.empty() || out.back() != path)) out.push_back(path);
  }
  return out;
}

TEST(FactorMumpsBuild, MappedLibrariesAreTheConfiguredOnes) {
  // Make sure the libraries are in use (the factorization below needs them anyway).
  const auto a = hs::test::laplacian_2d<double>(20, 21);
  hs::ShiftedPencil<double> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<double>();
  f->analyze(pencil, 1.0, -1.0);
  ASSERT_TRUE(f->factorize(1.0, -1.0));
  EXPECT_EQ(std::string(MUMPS_VERSION), "5.8.2");
  const std::string root = HSOLVE_MUMPS_ROOT;
  for (const std::string stem : {"libdmumps", "libsmumps", "libmumps_common"}) {
    const auto paths = mapped(stem);
    ASSERT_EQ(paths.size(), 1u) << stem << ": one copy mapped, got " << paths.size();
    if (!root.empty()) {
      EXPECT_EQ(paths[0].rfind(root, 0), 0u) << paths[0] << " is not under HSOLVE_MUMPS_ROOT " << root;
    } else {
      EXPECT_EQ(paths[0].find("/deps/mumps-"), std::string::npos) << paths[0];
    }
  }
}

}  // namespace
