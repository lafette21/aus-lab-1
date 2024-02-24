cmake_minimum_required(VERSION 3.16)

project(aus-lab-1
    VERSION 0.0.1
    DESCRIPTION ""
    HOMEPAGE_URL ""
    LANGUAGES CXX C
)

include(cmake/compiler-warnings.cmake)
include(cmake/dependencies.cmake)
include(cmake/settings.cmake)

add_library(project_warnings INTERFACE)
set_project_warnings(project_warnings)

enable_testing()

add_subdirectory(src)
add_subdirectory(unit-tests)
