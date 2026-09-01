# Add tests
add_executable(scatter scatter_test.cc)
target_link_libraries( scatter  PRIVATE twoscale_dolfinx)
add_custom_target(scatter_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            ${CMAKE_CURRENT_BINARY_DIR}/data
)
add_dependencies(scatter scatter_data)
add_test( NAME scatter  COMMAND mpirun -np 2 ./scatter )
