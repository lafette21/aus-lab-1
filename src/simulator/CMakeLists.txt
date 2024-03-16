set(EXEC_NAME simulator)

add_executable(${EXEC_NAME}
    main.cc
)

target_include_directories(${EXEC_NAME} PUBLIC
    # ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${EXEC_NAME} PUBLIC
    fmt::fmt-header-only
    nova
    # ${OpenCV_LIBS}
    # opencv_gapi
    # ${PCL_LIBRARIES}
)

code_analysis(${EXEC_NAME} PRIVATE)
