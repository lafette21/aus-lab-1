set(EXEC_NAME simulator)

add_executable(${EXEC_NAME}
    main.cc
)

target_link_libraries(${EXEC_NAME} PUBLIC
    fmt::fmt-header-only
    GSL::gsl
    nova
    range-v3::range-v3
    ${rclcpp_TARGETS}
    ${rosbag2_cpp_TARGETS}
    ${sensor_msgs_TARGETS}
)

code_analysis(${EXEC_NAME} PRIVATE)
