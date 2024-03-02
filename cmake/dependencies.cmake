if(DEFINED NO_CPM AND NO_CPM EQUAL 1)
    find_package(fmt REQUIRED)
    # find_package(OpenCV REQUIRED)
    find_package(PCL REQUIRED COMPONENTS common filters io segmentation)
else()
    include(cmake/cpm.cmake)

    # set(CPM_USE_LOCAL_PACKAGES ON)

    # CPMAddPackage("gl:libeigen/eigen#3.4.0")
    # CPMAddPackage("gh:fmtlib/fmt#10.2.1")
    CPMAddPackage("gh:PointCloudLibrary/pcl#pcl-1.14.0")
    # CPMAddPackage("gh:gabime/spdlog#v1.13.0")

    message("assd ${pcl_SOURCE_DIR}")

    # CPMAddPackage(
        # NAME PCL
        # VERSION 1.14.0
        # URL https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.14.0.tar.gz
        # DOWNLOAD_ONLY YES
    # )

    # if(PCL_ADDED)
        # message("${PCL_INCLUDE_DIR} asd ${PCL_SOURCE_DIR}")
        # add_library(PCL INTERFACE IMPORTED)
        # target_include_directories(PCL INTERFACE ${PCL_SOURCE_DIR})
    # endif()
endif()
