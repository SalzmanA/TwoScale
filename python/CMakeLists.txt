# ===============================================================================================
# Two-scale warper project
cmake_minimum_required(VERSION 3.21)

project(ts_nanobind)

include(FeatureSummary)
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# ===============================================================================================
# nanobind 
find_package(Python COMPONENTS Interpreter Development ${SKBUILD_SABI_COMPONENT} REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
#list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ===============================================================================================
# get target that will be binded
find_package(Twoscale REQUIRED CONFIG)

if(Twoscale_FOUND)
   message(STATUS "Found TwoScale at ${Twoscale_DIR}")
endif()
if (TARGET twoscale_dolfinx)
   get_target_property(tsdef twoscale_dolfinx INTERFACE_COMPILE_DEFINITIONS)
   if ( "${tsdef}" STREQUAL "tsdef-NOTFOUND" )
      message(FATAL_ERROR "target twoscale_dolfinx do not provides INTERFACE_COMPILE_DEFINITIONS ??")
   endif()
   get_target_property(tsinc twoscale_dolfinx INTERFACE_INCLUDE_DIRECTORIES)
endif()

# ===============================================================================================
# Create the binding library nanobind handles its own calls to
# target_link_libraries

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_SABI "STABLE_ABI")
endif()

nanobind_add_module(
  ts_cpp
  ${NANOBIND_SABI}
  twoscale/binding/ts.cc
)
if (TARGET twoscale_dolfinx)
   target_sources(ts_cpp PRIVATE 
      twoscale/binding/dolfinx/bind.cc
      twoscale/binding/dolfinx/bind_coarseManager.cc
      twoscale/binding/dolfinx/bind_patchManager.cc
      twoscale/binding/dolfinx/bind_MPC.cc
      twoscale/binding/dolfinx/bind_enrichFunc.cc
      twoscale/binding/dolfinx/bind_scaleJump.cc)
   target_link_libraries(ts_cpp PRIVATE twoscale_dolfinx)
   target_compile_definitions(ts_cpp PRIVATE ${tsdef})
   target_include_directories(ts_cpp PRIVATE ${tsinc})
endif()
#target_link_libraries(ts_bind PRIVATE ts_cpp)
#target_link_libraries(ts_cpp PRIVATE dolfinx)
get_target_property(tsdef ts_cpp COMPILE_DEFINITIONS)
message(STATUS "Inherited definition ${tsdef}")

# Check for dolfinx python api
execute_process(
   COMMAND ${Python_EXECUTABLE} -c
          "import dolfinx; print(dolfinx.get_include())"
   OUTPUT_VARIABLE SDOLFINX_INCLUDE_DIR
   RESULT_VARIABLE SDOLFINX_COMMAND_RESULT
   ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT SDOLFINX_COMMAND_RESULT)
  message(STATUS "Found dolfinx include directory at ${SDOLFINX_INCLUDE_DIR}")
  target_include_directories(ts_cpp PRIVATE ${SDOLFINX_INCLUDE_DIR})
  target_compile_definitions(ts_cpp PRIVATE HAS_DOLFINX_PY)
else()
  message(STATUS "dolfinx not found.")
endif()

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(ts_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
  target_compile_definitions(ts_cpp PRIVATE HAS_PETSC4PY)
else()
  message(STATUS "petsc4py not found.")
endif()

install(TARGETS ts_cpp DESTINATION twoscale)
