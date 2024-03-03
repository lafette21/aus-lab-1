#include "cylinder.hh"
#include "utils.hh"

#include <nova/vec.h>
#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <future>
#include <span>
#include <ranges>


inline auto& init(const std::string& name) {
    spdlog::set_default_logger(spdlog::create<spdlog::sinks::ansicolor_stdout_sink_mt>(name));

    auto& logger = *spdlog::get(name);
    logger.set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n @%t] %^[%l]%$ %v");

    return logger;
}

[[nodiscard]] auto extract_cylinder(const pcl::PointCloud<pcl::PointXYZRGB>& cloud)
        -> std::pair<pcl::PointCloud<pcl::PointXYZRGB>, std::vector<nova::Vec3f>>
{
    auto tmp = cloud
             | std::views::transform([](const auto& elem) { return nova::Vec3f { elem.x, elem.y, elem.z }; })
             | std::views::filter([](const auto& elem) { return elem != nova::Vec3f { 0, 0, 0 }; });

    std::vector<nova::Vec3f> points(std::begin(tmp), std::end(tmp));

    const auto cylinder_params = estimate_cylinder_RANSAC(points, 0.07f, 5'000);
    const auto differences = calculate_RANSAC_diffs(points, cylinder_params, 0.07f);

    pcl::PointCloud<pcl::PointXYZRGB> cylinder;
    std::vector<nova::Vec3f> rest;

    for (std::size_t i = 0; i < points.size(); ++i) {
        if (differences.is_inliers.at(i)) {
            cylinder.emplace_back(points[i].x(), points[i].y(), points[i].z(), 255, 0, 0);
        } else {
            rest.push_back(points[i]);
        }
    }

    return {
        cylinder,
        rest
    };
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    init("logger");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view {arg}; });

    constexpr std::size_t From = 165;
    constexpr std::size_t To = 166;

    for (std::size_t i = From; i < To; ++i) {
        const auto cloud = read_file<lidar_data_parser>(std::filesystem::path(args[1]).string()).value();
        spdlog::info("cloud size: {}", cloud.size());

        pcl::PointCloud<pcl::PointXYZRGB> tmp_cloud;
        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(cloud.makeShared());
        vg.setLeafSize(0.07f, 0.07f, 0.07f); // TODO: Need a good param - Set the leaf size (adjust as needed)
        vg.filter(tmp_cloud);

        pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
        ror.setInputCloud(tmp_cloud.makeShared());
        ror.setRadiusSearch(0.7); // TODO: Need a good param
        ror.setMinNeighborsInRadius(40); // TODO: Need a good param
        ror.filter(cloud_filtered);

        // TODO: This might not be needed
        // auto tmp = cloud_filtered
                 // | std::views::transform([](const auto& elem) { return nova::Vec3f { elem.x, elem.y, elem.z }; })
                 // | std::views::filter([](const auto& elem) { return elem != nova::Vec3f { 0, 0, 0 }; });

        // std::vector<nova::Vec3f> points(std::begin(tmp), std::end(tmp));

        spdlog::info("filtered cloud size: {}", cloud_filtered.size());

        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(cloud_filtered, min_pt, max_pt);

        // Calculate the midpoint along X and Y axes
        float mid_x = (min_pt.x + max_pt.x) / 2.0f;
        float mid_y = (min_pt.y + max_pt.y) / 2.0f;

        pcl::PointCloud<pcl::PointXYZRGB> cloud_quarter1;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_quarter2;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_quarter3;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_quarter4;

        // Split the point cloud into four quarters based on the midpoint
        for (const auto& point : cloud_filtered) {
            if (point.x < mid_x && point.y < mid_y) {
                cloud_quarter1.push_back(point);
            } else if (point.x >= mid_x && point.y < mid_y) {
                cloud_quarter2.push_back(point);
            } else if (point.x < mid_x && point.y >= mid_y) {
                cloud_quarter3.push_back(point);
            } else {
                cloud_quarter4.push_back(point);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGB> out_cloud;

        auto futures = std::array {
            std::async(extract_cylinder, cloud_quarter1),
            std::async(extract_cylinder, cloud_quarter2),
            std::async(extract_cylinder, cloud_quarter3),
            std::async(extract_cylinder, cloud_quarter4)
        };

        for (auto& f : futures) {
            const auto [cylinder, rest] = f.get();
            out_cloud += cylinder;
        }

        pcl::io::savePLYFile("./test.ply", out_cloud);
    }

    return EXIT_SUCCESS;
}
