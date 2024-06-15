add_executable(poc
    poc.cc
)

target_include_directories(poc PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(poc PUBLIC
    fmt::fmt-header-only
    nova
    ${PCL_LIBRARIES}
)

code_analysis(poc PRIVATE)

set(EXEC_NAME ph)

add_executable(${EXEC_NAME}
    main.cc
)

target_include_directories(${EXEC_NAME} PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${EXEC_NAME} PUBLIC
    fmt::fmt-header-only
    nova
    range-v3::range-v3
    ${PCL_LIBRARIES}
)

code_analysis(${EXEC_NAME} PRIVATE)

add_subdirectory(simulator)
