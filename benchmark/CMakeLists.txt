set(EXEC_NAME bench)

add_executable(${EXEC_NAME}
    ${PROJECT_SOURCE_DIR}/src/ransac.cu
    bench_utils.cc
)

set_target_properties(${EXEC_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Static
)

# Check if the project is using CUDA
if(CMAKE_CUDA_COMPILER)
    # Add the flag to NVCC flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

target_include_directories(${EXEC_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${EXEC_NAME} PUBLIC
    benchmark::benchmark
    ${CUDA_LIBRARIES}
    fmt::fmt-header-only
    nova
    ${PCL_LIBRARIES}
)
