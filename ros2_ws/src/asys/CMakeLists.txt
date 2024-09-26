cmake_minimum_required(VERSION 3.8)
project(asys)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    msg/Point.msg
    msg/PointList.msg
    DEPENDENCIES std_msgs
)

add_executable(perception src/perception.cc)
rosidl_target_interfaces(perception ${PROJECT_NAME} "rosidl_typesupport_cpp")
ament_target_dependencies(perception PCL pcl_conversions rclcpp rosidl_default_runtime rosidl_typesupport_cpp sensor_msgs std_msgs)
target_include_directories(perception PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
    ${PROJECT_SOURCE_DIR}/../../../deps/nova/include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(perception PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_compile_definitions(perception PUBLIC
    FMT_HEADER_ONLY=1
)

install(TARGETS perception
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
