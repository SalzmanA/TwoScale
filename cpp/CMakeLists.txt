# ===============================================================================================
# Two-scale project
cmake_minimum_required(VERSION 3.21)
project(Twoscale VERSION "0.11.0")

# ===============================================================================================
# pyproject.toml generation from VERSION above
math( EXPR DOLFINX_NEXT_MINOR "${PROJECT_VERSION_MINOR}+1" )
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/pyproject.toml.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/pyproject.toml"
    @ONLY
)

# ===============================================================================================
# options
include(FeatureSummary)
option(BUILD_SHARED_LIBS "Build two-scale with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build two-scale with shared libraries.")
option(OUTPUT_MATVECT "Output coarse enriched matrix vector sytem and operator." OFF)
add_feature_info(OUTPUT_MATVECT OUTPUT_MATVECT "Output coarse enriched matrix vector sytem and operator.")
option(USE_PETSC_MATNEST "Use PETSc nested matrix format." ON)
add_feature_info(USE_PETSC_MATNEST USE_PETSC_MATNEST "Use PETSc nested matrix format.")
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")
option(ENABLE_TESTING       "Create executables for testing (ctest)." ON)
add_feature_info(ENABLE_TESTING ENABLE_TESTING "Create executables for testing (ctest).")



# ===============================================================================================
# twoscale dolfinx requires dolfinx
#find_package(DOLFINX REQUIRED)
find_package(DOLFINX REQUIRED CONFIG)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()


# ===============================================================================================
# display feature
feature_summary(WHAT ENABLED_FEATURES)

# ===============================================================================================
# dolfinx version add to build process if dolfinx found
if(DOLFINX_FOUND)
add_subdirectory(dolfinx)
endif()

# ===============================================================================================
# packaging
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/TwoscaleConfigVersion.cmake
  VERSION ${Twoscale_VERSION}
  COMPATIBILITY ExactVersion
)
configure_package_config_file(
   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TwoscaleConfig.cmake.in
   ${CMAKE_BINARY_DIR}/TwoscaleConfig.cmake
   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/twoscale
)
# ===============================================================================================
# instalation
install(
  FILES ${CMAKE_BINARY_DIR}/TwoscaleConfig.cmake
        ${CMAKE_BINARY_DIR}/TwoscaleConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/twoscale
  COMPONENT Development
)
install(EXPORT TwoscaleTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/twoscale)
