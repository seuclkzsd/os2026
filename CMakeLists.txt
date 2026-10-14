cmake_minimum_required(VERSION 3.20)
project(hsolve VERSION 0.1.0 DESCRIPTION "Sparse Hermitian (indefinite) eigensolver" LANGUAGES C CXX)

option(HSOLVE_NATIVE "Optimize for the build machine (-march=native)" ON)
option(HSOLVE_BUILD_TESTS "Build unit tests" ON)
option(HSOLVE_BUILD_APPS "Build command-line applications" ON)
option(HSOLVE_WERROR "Treat compiler warnings as errors" OFF)
option(HSOLVE_BUILD_BENCH "Build benchmark/measurement programs (bench/*.cpp)" ON)
set(HSOLVE_SANITIZE "" CACHE STRING "Sanitizers for debug builds, e.g. address,undefined")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- dependencies (all open source; see docs/DEPENDENCIES.md) ----
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(ZLIB REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# MUMPS: HSOLVE_MUMPS_ROOT names an install of tools/dev/build_mumps.sh (same library names and
# sonames as conda-forge's mumps-seq); empty = the environment's conda-forge mumps-seq. The default
# build (tools/dev/build.sh, HS_MUMPS=source) sets it to MUMPS 5.8.2 built from
# third_party/mumps/ with patches/mumps-5.8.2-l0-blocking.patch (tools/dev/mumps_cache.sh, P9
# integration); `HS_MUMPS=env tools/dev/build.sh` or a plain `cmake --preset` without
# -DHSOLVE_MUMPS_ROOT links the environment's. The install's lib/ goes first into the RPATH of
# executables, shared libraries and the Python module, ahead of the environment's (conda's LDFLAGS
# carry -rpath $CONDA_PREFIX/lib), so that the dynamic loader picks the rebuilt libraries
# (test_factor_mumps_build, tests/py/test_mumps_build.py check it).
# P9 "factspeed" (results/p9_factspeed.md §3.3-4.5): a rebuild alone gains nothing; built with
# the patch and run with hsolve's P9 block sizes a count is 1.07-1.08x faster on a fixed pivot
# order, end to end at 1e6 syn-mem27-100 pooled 1.084x, yee-pec-box-1e6 1.059x (1.040x on a fixed
# METIS order); made the default build by the user's decision (2026-10-14; results/p9_measurements.md
# measures the integrated build). HSOLVE_MUMPS_BLOCKING: the block sizes hsolve gives MUMPS when
# HS_MUMPS_BLOCKING is unset (src/factor/mumps_api.hpp Blocking) - "p9" by default with
# HSOLVE_MUMPS_ROOT, else "mumps".
set(HSOLVE_MUMPS_ROOT "" CACHE PATH "MUMPS install of tools/dev/build_mumps.sh (empty: the environment's)")
if(HSOLVE_MUMPS_ROOT)
  foreach(_kind EXE SHARED MODULE)
    string(PREPEND CMAKE_${_kind}_LINKER_FLAGS "-Wl,-rpath,${HSOLVE_MUMPS_ROOT}/lib ")
  endforeach()
endif()
set(HSOLVE_MUMPS_BLOCKING "" CACHE STRING
    "MUMPS block sizes when HS_MUMPS_BLOCKING is unset: mumps | p9 (empty: p9 with HSOLVE_MUMPS_ROOT)")
set_property(CACHE HSOLVE_MUMPS_BLOCKING PROPERTY STRINGS "" mumps p9)
if(HSOLVE_MUMPS_BLOCKING STREQUAL "")
  if(HSOLVE_MUMPS_ROOT)
    set(_hs_blocking p9)
  else()
    set(_hs_blocking mumps)
  endif()
elseif(HSOLVE_MUMPS_BLOCKING MATCHES "^(mumps|p9)$")
  set(_hs_blocking "${HSOLVE_MUMPS_BLOCKING}")
else()
  message(FATAL_ERROR "HSOLVE_MUMPS_BLOCKING must be empty, mumps or p9, got '${HSOLVE_MUMPS_BLOCKING}'")
endif()
if(HSOLVE_MUMPS_ROOT)
  message(STATUS "hsolve: MUMPS of ${HSOLVE_MUMPS_ROOT}, default block sizes ${_hs_blocking}")
  get_filename_component(_hs_mumps_label "${HSOLVE_MUMPS_ROOT}" NAME)
else()
  message(STATUS "hsolve: the environment's MUMPS, default block sizes ${_hs_blocking}")
  set(_hs_mumps_label env)
endif()
find_package(MUMPS 5.6 REQUIRED)

# ---- common compile options ----
add_library(hsolve_options INTERFACE)
target_compile_options(hsolve_options INTERFACE -Wall -Wextra -Wno-unused-parameter)
if(HSOLVE_WERROR)
  target_compile_options(hsolve_options INTERFACE -Werror)
endif()
if(HSOLVE_NATIVE)
  target_compile_options(hsolve_options INTERFACE -march=native)
endif()
if(HSOLVE_SANITIZE)
  target_compile_options(hsolve_options INTERFACE -fsanitize=${HSOLVE_SANITIZE} -fno-omit-frame-pointer)
  target_link_options(hsolve_options INTERFACE -fsanitize=${HSOLVE_SANITIZE})
endif()

# ---- library: every src/**/*.cpp belongs to libhsolve (module = directory) ----
file(GLOB_RECURSE HSOLVE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# src/blas is the optional BLAS hook (block at the end of this file), not part of libhsolve.
list(FILTER HSOLVE_SOURCES EXCLUDE REGEX "/src/blas/")
add_library(hsolve ${HSOLVE_SOURCES})
target_include_directories(hsolve
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(hsolve
  PUBLIC OpenMP::OpenMP_CXX hsolve_options
  PRIVATE MUMPS::MUMPS LAPACK::LAPACK BLAS::BLAS ZLIB::ZLIB nlohmann_json::nlohmann_json)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
target_compile_definitions(hsolve PRIVATE
  HSOLVE_VERSION="${PROJECT_VERSION}"
  HSOLVE_BUILD_FLAGS="${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} native=${HSOLVE_NATIVE} mumps=${_hs_mumps_label} mumps_blocking=${_hs_blocking}"
  HSOLVE_MUMPS_VERSION="${MUMPS_VERSION}")
if(_hs_blocking STREQUAL "p9")
  target_compile_definitions(hsolve PRIVATE HSOLVE_MUMPS_BLOCKING_P9=1)
endif()

# ---- applications ----
if(HSOLVE_BUILD_APPS)
  find_package(CLI11 2.0 REQUIRED)
  add_executable(hsolve_cli apps/hsolve_cli.cpp)
  set_target_properties(hsolve_cli PROPERTIES OUTPUT_NAME hsolve)
  target_link_libraries(hsolve_cli PRIVATE hsolve CLI11::CLI11 nlohmann_json::nlohmann_json)
endif()

# ---- benchmarks: one executable per bench/*.cpp (reproducible measurements, run via guard.py) ----
if(HSOLVE_BUILD_BENCH)
  file(GLOB HSOLVE_BENCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp")
  foreach(src IN LISTS HSOLVE_BENCH_SOURCES)
    get_filename_component(name "${src}" NAME_WE)
    add_executable(bench_${name} "${src}")
    target_include_directories(bench_${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
    target_link_libraries(bench_${name} PRIVATE hsolve LAPACK::LAPACK BLAS::BLAS MUMPS::MUMPS nlohmann_json::nlohmann_json)
  endforeach()
endif()

# ---- Python package (python/, docs/PYTHON.md): ON by default when pybind11 is found ----
find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
if(Python_FOUND)
  find_package(pybind11 2.10 CONFIG QUIET)
endif()
option(HSOLVE_BUILD_PYTHON "Build the Python module hsolve._hsolve (needs pybind11)" ${pybind11_FOUND})
if(HSOLVE_BUILD_PYTHON)
  if(NOT pybind11_FOUND)
    find_package(pybind11 2.10 CONFIG REQUIRED)
  endif()
  add_subdirectory(python)
endif()

# ---- tests: one executable per tests/cpp/test_*.cpp ----
if(HSOLVE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  include(GoogleTest)
  # Shared test fixtures (dense reference operators, known-spectrum generators).
  file(GLOB HSOLVE_TEST_SUPPORT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/support/*.cpp")
  add_library(hsolve_test_support STATIC ${HSOLVE_TEST_SUPPORT_SOURCES})
  target_include_directories(hsolve_test_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp"
                                                        "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(hsolve_test_support PUBLIC hsolve GTest::gtest LAPACK::LAPACK BLAS::BLAS)
  file(GLOB HSOLVE_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_*.cpp")
  foreach(src IN LISTS HSOLVE_TEST_SOURCES)
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(${name} PRIVATE HSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                                               HSOLVE_MUMPS_ROOT="${HSOLVE_MUMPS_ROOT}"
                                               HSOLVE_MUMPS_BLOCKING_DEFAULT="${_hs_blocking}")
    target_link_libraries(${name} PRIVATE hsolve hsolve_test_support GTest::gtest_main LAPACK::LAPACK
                                          BLAS::BLAS MUMPS::MUMPS nlohmann_json::nlohmann_json)
    # Test names are "<executable>.<Suite>.<Name>", so `ctest -R test_factor` selects one file.
    gtest_discover_tests(${name} TEST_PREFIX "${name}." DISCOVERY_MODE PRE_TEST
      PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
  endforeach()
endif()

# ---- BLAS hook (src/blas, docs/BLAS.md): profiler and shape routing of the dense BLAS under MUMPS ----
# Always built (small): hsolve_blas (router, kernels, profiler; compiled without -march=native, the
# AVX-512 kernels are selected at run time) and libhsolve_blashook.so for LD_PRELOAD. The option
# links the hook into the hsolve executable instead; HS_BLAS_ROUTE then configures it at run time.
# Default: linked, route "all" - 1.05-1.16x end to end on the six head-to-head configurations at
# 16 threads, equal accuracy (docs/BLAS.md sections 5-6); HS_BLAS_ROUTE=off turns the routes off at
# run time. The defaults apply to new build directories (cached values persist).
option(HSOLVE_BLAS_HOOK "Link the BLAS hook (src/blas) into the hsolve executable" ON)
set(HSOLVE_BLAS_ROUTE_DEFAULT "all" CACHE STRING "BLAS hook routes when HS_BLAS_ROUTE is unset (\"\" = off)")
add_library(hsolve_blas STATIC src/blas/route.cpp src/blas/narrow_kernels.cpp src/blas/blas_profile.cpp)
target_include_directories(hsolve_blas PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
# -ffp-contract=off: the narrow kernels promise the same NN result for every row split (simd_kernels.hpp);
# GCC contracts C++ by default (-ffp-contract=fast), which would fuse the register path's
# alpha * t + beta * c into an FMA while the chunked path (baseline ISA) rounds both products.
target_compile_options(hsolve_blas PRIVATE -Wall -Wextra -Wno-unused-parameter -ffp-contract=off
                                           $<$<BOOL:${HSOLVE_WERROR}>:-Werror>)
target_compile_definitions(hsolve_blas PRIVATE HSOLVE_BLAS_ROUTE_DEFAULT="${HSOLVE_BLAS_ROUTE_DEFAULT}")
target_link_libraries(hsolve_blas PUBLIC OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
add_library(hsolve_blashook SHARED src/blas/hook_symbols.cpp)
target_include_directories(hsolve_blashook PRIVATE $<TARGET_PROPERTY:MUMPS::MUMPS,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(hsolve_blashook PRIVATE -Wall -Wextra -Wno-unused-parameter $<$<BOOL:${HSOLVE_WERROR}>:-Werror>)
target_link_libraries(hsolve_blashook PRIVATE hsolve_blas)
# Links the hook's entry points into an executable that uses libhsolve. Its own dmumps_c / smumps_c
# then satisfy libhsolve's references, so a linker running with --as-needed (conda-forge's LDFLAGS)
# would drop the MUMPS libraries from DT_NEEDED and the wrappers would find no real routine:
# --no-as-needed (after the toolchain's flags) keeps every library of the link line. The dynamic
# list exports the executable's entry points, so the shared libraries' calls (MUMPS' BLAS calls)
# bind to them.
set(HSOLVE_BLASHOOK_DYNLIST "${CMAKE_CURRENT_BINARY_DIR}/hsolve_blashook.dynlist")
file(WRITE "${HSOLVE_BLASHOOK_DYNLIST}"
     "{ dgemm_; dtrsm_; dgemv_; daxpy_; dcopy_; dscal_; dswap_; dnrm2_; idamax_; dmumps_c; smumps_c; };\n")
function(hsolve_link_blas_hook target)
  target_sources(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src/blas/hook_symbols.cpp")
  target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:MUMPS::MUMPS,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(${target} PRIVATE hsolve_blas MUMPS::MUMPS)
  target_link_options(${target} PRIVATE "LINKER:--no-as-needed"
                                        "LINKER:--dynamic-list=${HSOLVE_BLASHOOK_DYNLIST}")
endfunction()
if(HSOLVE_BLAS_HOOK AND HSOLVE_BUILD_APPS)
  hsolve_link_blas_hook(hsolve_cli)
endif()
foreach(src IN LISTS HSOLVE_TEST_SOURCES HSOLVE_BENCH_SOURCES)
  get_filename_component(name "${src}" NAME_WE)
  if(name MATCHES "^test_asmkern" AND TARGET ${name})
    target_link_libraries(${name} PRIVATE hsolve_blas)
    target_compile_definitions(${name} PRIVATE HSOLVE_BLASHOOK_SO="$<TARGET_FILE:hsolve_blashook>")
    add_dependencies(${name} hsolve_blashook)
    if(name STREQUAL "test_asmkern_fix_exehook")
      # The in-executable hook exactly as hsolve gets it; with HSOLVE_BLAS_HOOK=ON also a run of
      # that hsolve executable.
      hsolve_link_blas_hook(${name})
      if(HSOLVE_BLAS_HOOK AND HSOLVE_BUILD_APPS)
        target_compile_definitions(${name} PRIVATE HSOLVE_HOOKED_CLI="$<TARGET_FILE:hsolve_cli>")
        add_dependencies(${name} hsolve_cli)
      endif()
    elseif(name STREQUAL "test_asmkern_fix")
      # The parallel routes under a runtime that grants fewer threads than they ask for.
      add_test(NAME test_asmkern_fix.ShortTeamUnderThreadLimit2
               COMMAND ${name} --gtest_filter=AsmkernFixShortTeam.*)
      set_tests_properties(test_asmkern_fix.ShortTeamUnderThreadLimit2 PROPERTIES
                           ENVIRONMENT "OMP_THREAD_LIMIT=2;OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    endif()
  elseif(name MATCHES "^asmkern_" AND TARGET bench_${name})
    target_link_libraries(bench_${name} PRIVATE hsolve_blas)
  endif()
endforeach()
