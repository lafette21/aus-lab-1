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
    # ${OpenCV_LIBS}
    # opencv_gapi
    range-v3::range-v3
    ${PCL_LIBRARIES}
)

code_analysis(${EXEC_NAME} PRIVATE)

add_executable(register
    test.cc
)

target_include_directories(register PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(register PUBLIC
    fmt::fmt-header-only
    nova
    # ${OpenCV_LIBS}
    # opencv_gapi
    ${PCL_LIBRARIES}
)

add_executable(circle
    test_circle.cc
)

target_include_directories(circle PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(circle PUBLIC
    fmt::fmt-header-only
    nova
    # ${OpenCV_LIBS}
    # opencv_gapi
    ${PCL_LIBRARIES}
)

add_executable(cluster
    test_cluster.cc
)

target_include_directories(cluster PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(cluster PUBLIC
    fmt::fmt-header-only
    nova
    # ${OpenCV_LIBS}
    # opencv_gapi
    ${PCL_LIBRARIES}
)

add_subdirectory(simulator)
