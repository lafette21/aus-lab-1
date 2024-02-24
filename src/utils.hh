#ifndef UTILS_HH
#define UTILS_HH

#include <fmt/format.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <expected>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>


struct measurement {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
};

struct error {
    std::string msg;
    operator std::string() { return msg; }
};

struct def_parser {
    [[nodiscard]] std::string operator()(std::ifstream&& iF) {
        std::stringstream ss;
        for (std::string line; std::getline(iF, line); ) {
            ss << line << '\n';
        }
        return ss.str();
    }
};

struct lidar_data_parser {
    [[nodiscard]] pcl::PointCloud<pcl::PointXYZRGB> operator()(std::ifstream&& iF) {
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

template <typename Parser = def_parser>
[[nodiscard]] auto read_file(std::string_view path, Parser&& parser = {})
        -> std::expected<std::remove_cvref_t<std::invoke_result_t<Parser, std::ifstream>>, error>
{
    const auto fs = std::filesystem::path(path);
    if (not std::filesystem::is_regular_file(fs)) {
        return std::unexpected<error>(fmt::format("{} is not a regular file!", std::filesystem::absolute(fs).string()));
    }

    return parser(std::ifstream(fs));
}

[[nodiscard]] measurement read_measurement(
    std::size_t id,
    std::string_view points_path
) {
    const pcl::PointCloud<pcl::PointXYZRGB> cloud = read_file<lidar_data_parser>((std::filesystem::path(points_path) / fmt::format("test_fn{}.xyz", id)).string()).value();

    return {
        .cloud = cloud,
    };
}

#endif // UTILS_HH
