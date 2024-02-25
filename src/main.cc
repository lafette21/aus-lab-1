#include "cylinder.hh"
#include "utils.hh"

#include <nova/vec.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <pcl/io/ply_io.h>

#include <cstdlib>
#include <span>
#include <ranges>


int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view {arg}; });

    constexpr std::size_t From = 165;
    constexpr std::size_t To = 166;

    for (std::size_t i = From; i < To; ++i) {
        const auto cloud = read_file<lidar_data_parser>(std::filesystem::path(args[1]).string()).value();
        fmt::println("cloud size: {}", cloud.size());

        auto tmp = cloud
                 | std::views::transform([](const auto& elem) { return nova::Vec3f { elem.x, elem.y, elem.z }; })
                 | std::views::filter([](const auto& elem) { return elem != nova::Vec3f { 0, 0, 0 }; });

        const std::vector<nova::Vec3f> points(std::begin(tmp), std::end(tmp));

        [[maybe_unused]] const auto cylinder = estimate_cylinder_RANSAC(points, 0.07f, 1'000);

        const auto differences = calculate_RANSAC_diffs(points, cylinder, 0.07f);

        pcl::PointCloud<pcl::PointXYZRGB> out_cloud;

        for (std::size_t j = 0; j < points.size(); ++j) {
            if (differences.is_inliers.at(j)) {
                out_cloud.emplace_back(points[j].x(), points[j].y(), points[j].z(), 255, 0, 0);
            } else {
                out_cloud.emplace_back(points[j].x(), points[j].y(), points[j].z(), 0, 0, 0);
            }
        }

        pcl::io::savePLYFile("./test.ply", out_cloud);
    }

    return EXIT_SUCCESS;
}
