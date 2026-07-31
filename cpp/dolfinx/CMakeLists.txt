# ===============================================================================================
# target creation
add_library(twoscale_dolfinx)

# ===============================================================================================
# link
target_link_libraries(twoscale_dolfinx PUBLIC dolfinx)

# ===============================================================================================
# features
target_compile_features(twoscale_dolfinx PUBLIC cxx_std_20)

# ===============================================================================================
# definitions
target_compile_definitions(twoscale_dolfinx PUBLIC TWOSCALE_VERSION="${Twoscale_VERSION}")
target_compile_definitions(twoscale_dolfinx PUBLIC TWOSCALE_DOLFINX)
if (OUTPUT_MATVECT)
   target_compile_definitions(twoscale_dolfinx PUBLIC TWOSCALE_OUTPUT_MATVECT)
endif()
if (USE_PETSC_MATNEST )
   target_compile_definitions(twoscale_dolfinx PUBLIC TWOSCALE_PETSC_MATNEST)
endif()

# ===============================================================================================
# properties
set_target_properties(
  twoscale_dolfinx
  PROPERTIES VERSION ${Twoscale_VERSION}
  SOVERSION ${Twoscale_VERSION_MAJOR}.${Twoscale_VERSION_MINOR}
)

# ===============================================================================================
# sources
target_sources(
  twoscale_dolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/scaleJump.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/patchManager.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/extraPartitioners.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coarseManager.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/debug.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/util.cc
)

# ===============================================================================================
# Headers 
set(HEADERS_twoscale_dolfinx
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleJump.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleJump_imp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MPC.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MPC_imp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/coarseManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/coarseManager_imp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/patchManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/patchManager_imp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/enrichedFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/enrichedFunctions_imp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleJump_imp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataTsPerMacro.h
    ${CMAKE_CURRENT_SOURCE_DIR}/extraPartitioners.h
    ${CMAKE_CURRENT_SOURCE_DIR}/extraUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/extraRefine.h
    ${CMAKE_CURRENT_SOURCE_DIR}/util.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp23.h
    ${CMAKE_CURRENT_SOURCE_DIR}/debug.h
)
target_include_directories(
  twoscale_dolfinx
  PUBLIC
    $<INSTALL_INTERFACE:include>
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

# ===============================================================================================
# instalation
install(
   TARGETS twoscale_dolfinx
    EXPORT TwoscaleTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
)
install(
  FILES ${HEADERS_twoscale_dolfinx}
  DESTINATION include/twoscale/dolfinx
  COMPONENT Development
)



