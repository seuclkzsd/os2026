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
  HSOLVE_BUILD_FLAGS="${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} native=${HSOLVE_NATIVE}"
  HSOLVE_MUMPS_VERSION="${MUMPS_VERSION}")

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
    target_compile_definitions(${name} PRIVATE HSOLVE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
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
option(HSOLVE_BLAS_HOOK "Link the BLAS hook (src/blas) into the hsolve executable" OFF)
add_library(hsolve_blas STATIC src/blas/route.cpp src/blas/narrow_kernels.cpp src/blas/blas_profile.cpp)
target_include_directories(hsolve_blas PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(hsolve_blas PRIVATE -Wall -Wextra -Wno-unused-parameter $<$<BOOL:${HSOLVE_WERROR}>:-Werror>)
target_link_libraries(hsolve_blas PUBLIC OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
add_library(hsolve_blashook SHARED src/blas/hook_symbols.cpp)
target_include_directories(hsolve_blashook PRIVATE $<TARGET_PROPERTY:MUMPS::MUMPS,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(hsolve_blashook PRIVATE -Wall -Wextra -Wno-unused-parameter $<$<BOOL:${HSOLVE_WERROR}>:-Werror>)
target_link_libraries(hsolve_blashook PRIVATE hsolve_blas)
if(HSOLVE_BLAS_HOOK AND HSOLVE_BUILD_APPS)
  target_sources(hsolve_cli PRIVATE src/blas/hook_symbols.cpp)
  target_include_directories(hsolve_cli PRIVATE $<TARGET_PROPERTY:MUMPS::MUMPS,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(hsolve_cli PRIVATE hsolve_blas)
endif()
foreach(src IN LISTS HSOLVE_TEST_SOURCES HSOLVE_BENCH_SOURCES)
  get_filename_component(name "${src}" NAME_WE)
  if(name MATCHES "^test_asmkern" AND TARGET ${name})
    target_link_libraries(${name} PRIVATE hsolve_blas)
  elseif(name MATCHES "^asmkern_" AND TARGET bench_${name})
    target_link_libraries(bench_${name} PRIVATE hsolve_blas)
  endif()
endforeach()
