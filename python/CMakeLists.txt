# Python package `hsolve` (docs/PYTHON.md): the pybind11 extension hsolve._hsolve plus the
# pure-Python front end, laid out as ${CMAKE_CURRENT_BINARY_DIR}/hsolve/ so that
#   PYTHONPATH=build/<preset>/python python -c "import hsolve"
# works straight from the build tree. Included from the top-level CMakeLists.txt when
# HSOLVE_BUILD_PYTHON is ON (the default when pybind11 is found).

set(HSOLVE_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/hsolve")

pybind11_add_module(_hsolve MODULE src/module.cpp)
target_include_directories(_hsolve PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_hsolve PRIVATE hsolve)
# $<1:...> keeps multi-config generators from appending a per-configuration subdirectory.
set_target_properties(_hsolve PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${HSOLVE_PYTHON_PACKAGE_DIR}>")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Do not re-export the static libhsolve symbols from the extension (pybind11 already hides its own).
  target_link_options(_hsolve PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# Pure-Python sources: copied at configure time (CMake re-runs when they change).
file(GLOB HSOLVE_PYTHON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hsolve/*.py")
foreach(src IN LISTS HSOLVE_PYTHON_SOURCES)
  get_filename_component(name "${src}" NAME)
  configure_file("${src}" "${HSOLVE_PYTHON_PACKAGE_DIR}/${name}" COPYONLY)
endforeach()
