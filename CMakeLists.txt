cmake_minimum_required(VERSION 3.16)

project(aus-lab-1
    VERSION 0.0.1
    DESCRIPTION ""
    HOMEPAGE_URL ""
    LANGUAGES CXX C
)

option(ENABLE_ROS2BAG_OUTPUT "Enable output into ROS2 bag format" OFF)

if(ENABLE_ROS2BAG_OUTPUT)
    add_definitions(-DENABLE_ROS2BAG_OUTPUT=1)
else()
    add_definitions(-DENABLE_ROS2BAG_OUTPUT=0)
endif()

include(cmake/compiler-warnings.cmake)
include(cmake/dependencies.cmake)
include(cmake/settings.cmake)

add_library(project_warnings INTERFACE)
set_project_warnings(project_warnings)

enable_testing()

add_subdirectory(deps)
add_subdirectory(src)
add_subdirectory(unit-tests)
