#include "cylinder.hh"
#include "utils.hh"

#include <nova/vec.h>
#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstdlib>
#include <future>
#include <queue>
#include <ranges>
#include <span>


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

[[nodiscard]] auto quarter_cloud(const pcl::PointCloud<pcl::PointXYZRGB>& cloud)
        -> std::array<pcl::PointCloud<pcl::PointXYZRGB>, 4>
{
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);

    // Calculate the midpoint along X and Y axes
    float mid_x = (min_pt.x + max_pt.x) / 2.0f;
    float mid_y = (min_pt.y + max_pt.y) / 2.0f;

    pcl::PointCloud<pcl::PointXYZRGB> quarter1;
    pcl::PointCloud<pcl::PointXYZRGB> quarter2;
    pcl::PointCloud<pcl::PointXYZRGB> quarter3;
    pcl::PointCloud<pcl::PointXYZRGB> quarter4;

    // Split the point cloud into four quarters based on the midpoint
    for (const auto& point : cloud) {
        if (point.x < mid_x && point.y < mid_y) {
            quarter1.push_back(point);
        } else if (point.x >= mid_x && point.y < mid_y) {
            quarter2.push_back(point);
        } else if (point.x < mid_x && point.y >= mid_y) {
            quarter3.push_back(point);
        } else {
            quarter4.push_back(point);
        }
    }

    return {
        quarter1,
        quarter2,
        quarter3,
        quarter4
    };
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    [[maybe_unused]] auto& logger = init("logger");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view {arg}; });

    constexpr std::size_t From = 165;
    constexpr std::size_t To = 176;

    std::queue<pcl::PointCloud<pcl::PointXYZRGB>> clouds;

    spdlog::info("Reading cloud(s)");

    for (std::size_t i = From; i < To; ++i) {
        const auto cloud = read_file<lidar_data_parser>(
            (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", i)).string()
        ).value();
        clouds.push(cloud);
    }

    spdlog::info("Read {} cloud(s)", clouds.size());
    spdlog::info("Processing cloud(s)");

    pcl::PointCloud<pcl::PointXYZRGB> out_cloud;

    while (not clouds.empty()) {
        const auto cloud = clouds.front();
        clouds.pop();
        spdlog::info("Cloud size: {}", cloud.size());

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

        spdlog::info("Filtered cloud size: {}", cloud_filtered.size());

        const auto [quarter1, quarter2, quarter3, quarter4] = quarter_cloud(cloud_filtered);

        pcl::PointCloud<pcl::PointXYZRGB> cylinders;

        auto futures = std::array {
            std::async(extract_cylinder, quarter1),
            std::async(extract_cylinder, quarter2),
            std::async(extract_cylinder, quarter3),
            std::async(extract_cylinder, quarter4)
        };

        for (auto& f : futures) {
            const auto [cylinder, rest] = f.get();
            cylinders += cylinder;
        }

        Eigen::Affine3f transform = Eigen::Affine3f::Identity();

        // Define a translation of 10 meters on the x axis.
        transform.translation() << 10.0, 0.0, 0.0;

        // The same rotation matrix as before; theta radians around Z axis
        transform.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ()));

        pcl::transformPointCloud(cylinders, cylinders, transform);

        out_cloud += cylinders;
    }

    pcl::io::savePLYFile("./test.ply", out_cloud);

    return EXIT_SUCCESS;
}
