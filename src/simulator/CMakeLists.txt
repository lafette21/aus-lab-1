set(EXEC_NAME simulator)

add_executable(${EXEC_NAME}
    main.cc
)

target_link_libraries(${EXEC_NAME} PUBLIC
    fmt::fmt-header-only
    nova
)

code_analysis(${EXEC_NAME} PRIVATE)
