// hsolve command-line front end. Placeholder until the driver lands (roadmap P2):
// it only reports the build configuration so that the build/CI pipeline is exercised.
#include <cstdio>

#include "hsolve/util.hpp"

int main() {
  std::printf("%s\n", hs::build_info().c_str());
  std::printf("driver not implemented yet (see docs/DESIGN.md roadmap P2)\n");
  return 0;
}
