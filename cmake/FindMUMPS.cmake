# FindMUMPS
# ---------
# Locates the sequential MUMPS libraries (conda-forge "mumps-seq" layout: lib*mumps_seq.so,
# or the upstream layout without the _seq suffix) and defines the imported target
#   MUMPS::MUMPS  (all four arithmetics: s, d, c, z + common + pord + mpiseq)
# Result variables: MUMPS_FOUND, MUMPS_INCLUDE_DIR, MUMPS_LIBRARIES, MUMPS_VERSION.
# HSOLVE_MUMPS_ROOT (CMakeLists.txt): search only there (an install of tools/dev/build_mumps.sh);
# otherwise $CONDA_PREFIX, MUMPS_ROOT and the default paths. The results are cached; when
# HSOLVE_MUMPS_ROOT changes in an existing build directory (tools/dev/build.sh HS_MUMPS=source|env)
# they are searched again, so the headers and libraries always come from the configured MUMPS.

if(NOT "${HSOLVE_MUMPS_ROOT}" STREQUAL "${_HSOLVE_MUMPS_ROOT_SEARCHED}")
  unset(MUMPS_INCLUDE_DIR CACHE)
  foreach(_c smumps dmumps cmumps zmumps mumps_common pord mpiseq)
    unset(MUMPS_${_c}_LIBRARY CACHE)
  endforeach()
endif()
set(_HSOLVE_MUMPS_ROOT_SEARCHED "${HSOLVE_MUMPS_ROOT}" CACHE INTERNAL "HSOLVE_MUMPS_ROOT of the cached MUMPS search")

if(HSOLVE_MUMPS_ROOT)
  set(_mumps_search PATHS "${HSOLVE_MUMPS_ROOT}" NO_DEFAULT_PATH)
else()
  set(_mumps_search HINTS ENV CONDA_PREFIX ${MUMPS_ROOT} ENV MUMPS_ROOT)
endif()

find_path(MUMPS_INCLUDE_DIR NAMES dmumps_c.h ${_mumps_search} PATH_SUFFIXES include)

set(_mumps_components smumps dmumps cmumps zmumps mumps_common pord mpiseq)
set(MUMPS_LIBRARIES "")
foreach(_c IN LISTS _mumps_components)
  find_library(MUMPS_${_c}_LIBRARY NAMES ${_c}_seq ${_c} ${_mumps_search} PATH_SUFFIXES lib lib64)
  if(MUMPS_${_c}_LIBRARY)
    list(APPEND MUMPS_LIBRARIES ${MUMPS_${_c}_LIBRARY})
  endif()
  mark_as_advanced(MUMPS_${_c}_LIBRARY)
endforeach()

if(MUMPS_INCLUDE_DIR AND EXISTS "${MUMPS_INCLUDE_DIR}/dmumps_c.h")
  file(STRINGS "${MUMPS_INCLUDE_DIR}/dmumps_c.h" _mumps_ver_line
       REGEX "^#define MUMPS_VERSION \"[0-9.]+\"")
  string(REGEX REPLACE ".*\"([0-9.]+)\".*" "\\1" MUMPS_VERSION "${_mumps_ver_line}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUMPS
  REQUIRED_VARS MUMPS_INCLUDE_DIR MUMPS_dmumps_LIBRARY MUMPS_zmumps_LIBRARY
                MUMPS_smumps_LIBRARY MUMPS_cmumps_LIBRARY MUMPS_mumps_common_LIBRARY
  VERSION_VAR MUMPS_VERSION)

if(MUMPS_FOUND AND NOT TARGET MUMPS::MUMPS)
  add_library(MUMPS::MUMPS INTERFACE IMPORTED)
  set_target_properties(MUMPS::MUMPS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${MUMPS_LIBRARIES}")
  # The sequential build ships its own fake-MPI headers.
  if(EXISTS "${MUMPS_INCLUDE_DIR}/mumps_seq")
    set_property(TARGET MUMPS::MUMPS APPEND PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIR}/mumps_seq")
  endif()
endif()
mark_as_advanced(MUMPS_INCLUDE_DIR)
