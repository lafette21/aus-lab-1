set(EXEC_NAME bench)

add_executable(${EXEC_NAME}
    bench_utils.cc
)

target_include_directories(${EXEC_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${EXEC_NAME} PUBLIC
    benchmark::benchmark
    fmt::fmt-header-only
    nova
    ${PCL_LIBRARIES}
)
