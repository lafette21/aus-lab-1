set(EXEC_NAME ph)

add_executable(${EXEC_NAME}
    # main.cc
    test.cc
)

target_include_directories(${EXEC_NAME} PUBLIC
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${EXEC_NAME} PUBLIC
    # Eigen3::Eigen
    # fmt::fmt-header-only
    # nova
    # ${OpenCV_LIBS}
    # opencv_gapi
    # PCL
    pcl::PCL_COMMON_LIBRARIES
    ${PCL_LIBRARIES}
)

code_analysis(${EXEC_NAME} PRIVATE)
