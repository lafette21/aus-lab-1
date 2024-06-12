#ifndef TYPES_HH
#define TYPES_HH

#include <fmt/format.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sstream>
#include <string>


struct trafo {
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
};

struct lidar_data_parser {
    [[nodiscard]] pcl::PointCloud<pcl::PointXYZRGB> operator()(std::istream& iF) {
        pcl::PointCloud<pcl::PointXYZRGB> result;
        std::stringstream ss;
        for (std::string line; std::getline(iF, line); ) {
            ss << line << '\n';
            pcl::PointXYZRGB point;
            float dummy;
            ss >> point.x >> point.y >> point.z >> dummy >> dummy >> dummy;
            result.push_back(point);
            ss.clear();
        }
        return result;
    }
};

#endif // TYPES_HH
