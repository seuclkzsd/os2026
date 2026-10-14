// hsolve: the command line (apps/cli.cpp) in one process.
//
//   hsolve --A A.npz [--B B.npz] [--config config.json] --out DIR [overrides...]
#include "cli.hpp"

int main(int argc, char** argv) { return hs::cli::run(argc, argv, nullptr); }
