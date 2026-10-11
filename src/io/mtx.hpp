// Internal to src/io: MatrixMarket (.mtx) reader.
// Format: https://math.nist.gov/MatrixMarket/formats.html (banner "%%MatrixMarket matrix
// <coordinate|array> <real|double|integer|pattern|complex> <general|symmetric|hermitian|...>").
#pragma once

#include <filesystem>
#include <variant>

#include "hsolve/sparse.hpp"
#include "io/csr_assembly.hpp"

namespace hs::io {

enum class MtxSymmetry { General, Symmetric, Hermitian };

struct MtxMatrix {
  /// The stored entries, 0-based, duplicates summed. Symmetric/Hermitian files keep their single
  /// triangle: restoring the other one is the loader's storage step.
  std::variant<CsrMatrix<double>, CsrMatrix<cplx>> matrix;
  MtxSymmetry symmetry = MtxSymmetry::General;
  AssemblyStats stats;
};

/// Reads a square MatrixMarket matrix. Fields real/double/integer/pattern give double (pattern
/// entries are 1), complex gives cplx. Coordinate entries may repeat (summed) and exact zeros are
/// dropped if drop_zeros; array format (column-major, only the lower triangle for symmetric/
/// hermitian) always drops exact zeros. Comment ('%') and blank lines are skipped anywhere.
/// Throws Io for malformed text (with the line number), InvalidInput for skew-symmetric,
/// non-square or non-'matrix' objects. Streams the file in 4 MiB chunks.
MtxMatrix read_matrix_market(const std::filesystem::path& path, bool drop_zeros);

}  // namespace hs::io
