#include "simulator/logging.hh"
#include "types.hh"
#include "utils.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <charconv>
#include "nova/io.h"
#include <nova/vec.h>
#include <nova/utils.h>
#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <range/v3/range/conversion.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <concepts>
#include <cstdlib>
#include <future>
#include <numbers>
#include <ranges>
#include <span>
#include <utility>


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
                    | std::views::transform([](const auto& arg) { return std::string_view{ arg }; });

    std::size_t from = 0;
    std::size_t to = 0;

    std::from_chars(args[2].begin(), args[2].begin() + args[2].size(), from);
    std::from_chars(args[3].begin(), args[3].begin() + args[3].size(), to);

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> clouds;
    clouds.reserve(to - from);

    spdlog::info("Reading cloud(s)");

    for (std::size_t i = from; i < to; ++i) {
        const auto cloud = nova::read_file<lidar_data_parser>(
            (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", i)).string()
        ).value();
        clouds.push_back(cloud);
    }

    spdlog::info("Read {} cloud(s)", clouds.size());
    spdlog::info("Processing cloud(s)");

    pcl::PointCloud<pcl::PointXYZRGB> out_cloud;

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> cyl_clouds;
    std::vector<std::pair<std::size_t, nova::Vec4f>> prev_cyl_params;
    Eigen::Matrix4f trafo = Eigen::Matrix4f::Identity();

    std::ofstream oF("./out.xyz");

    for (const auto& [idx, cloud] : std::views::enumerate(clouds)) {
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

        auto futures = std::array {
            std::async(extract_cylinder, quarter1),
            std::async(extract_cylinder, quarter2),
            std::async(extract_cylinder, quarter3),
            std::async(extract_cylinder, quarter4)
        };

        std::vector<std::pair<std::size_t, nova::Vec4f>> cyl_params;
        cyl_params.reserve(4);

        for (auto [id, f] : std::views::enumerate(futures)) {
            const auto [params, cylinder, rest] = f.get();

            if (std::isnan(params.x()) or std::isnan(params.y()) or std::isnan(params.z()) or std::isnan(params.w())) {
                continue;
            }

            cyl_params.push_back(std::make_pair(id, params));
        }

        for (const auto& p : cyl_params) {
            fmt::print("{} {} {}\n", p.first, p.second.x(), p.second.y());
        }

        // pcl::PointCloud<pcl::PointXYZRGB> cylinders;

        // for (const auto& param : cyl_params) {
            // if (std::isnan(param.x()) or std::isnan(param.y()) or std::isnan(param.z()) or std::isnan(param.w())) {
                // continue;
            // }

            // const auto points = gen_circle(param);

            // for (const auto& p : points) {
                // cylinders.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
            // }
        // }

        // out_cloud += cylinders;

        if (prev_cyl_params.size() > 0) {
            std::vector<nova::Vec4f> prev_vec;
            std::vector<nova::Vec4f> curr_vec;

            for (const auto& elem : cyl_params) {
                const auto it = std::ranges::find(prev_cyl_params, elem.first, &std::pair<std::size_t, nova::Vec4f>::first);

                if (it != std::end(prev_cyl_params)) {
                    prev_vec.push_back(it->second);
                    curr_vec.push_back(elem.second);
                }
            }

            // for (const auto& p : prev_vec) {
                // fmt::print("{} {}\n", p.x(), p.y());
            // }

            // for (const auto& p : curr_vec) {
                // fmt::print("{} {}\n", p.x(), p.y());
            // }

            pcl::PointCloud<pcl::PointXYZRGB> cylinders;

            for (const auto& param : curr_vec) {
                const auto points = gen_circle(param);

                for (const auto& p : points) {
                    cylinders.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
                }
            }

            out_cloud += cylinders;

            const auto& prev_cloud = cyl_clouds.back();
            const auto min_size = std::min(std::size(prev_cloud), std::size(cylinders));

            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3, static_cast<int>(min_size));
            Eigen::MatrixXf B = Eigen::MatrixXf::Zero(3, static_cast<int>(min_size));

            for (std::size_t i = 0; i < min_size; ++i) {
                A(0, static_cast<int>(i)) = prev_cloud[i].x;
                A(1, static_cast<int>(i)) = prev_cloud[i].y;
                A(2, static_cast<int>(i)) = prev_cloud[i].z;

                B(0, static_cast<int>(i)) = cylinders[i].x;
                B(1, static_cast<int>(i)) = cylinders[i].y;
                B(2, static_cast<int>(i)) = cylinders[i].z;
            }

            const auto new_trafo_tmp = rigid_transform_3D(B, A);

            Eigen::Matrix4f new_trafo = Eigen::Matrix4f::Identity();
            new_trafo.block<3, 3>(0, 0) = new_trafo_tmp.R;
            new_trafo.block<3, 1>(0, 3) = new_trafo_tmp.t;

            trafo = trafo * new_trafo;

            pcl::PointCloud<pcl::PointXYZRGB> out;

            for (const auto& p : cylinders) {
                const Eigen::Vector4f pt = Eigen::Vector4f{ p.x, p.y, p.z, 1.0f };
                const Eigen::Vector4f ptt = trafo * pt;
                out.emplace_back(ptt.x(), ptt.y(), ptt.z(), 255, 0, 0);
            }

            for (const auto& p : prev_cloud) {
                out.emplace_back(p.x, p.y, p.z, 0, 255, 0);
            }

            pcl::io::savePLYFile(fmt::format("./registered-{}.ply", idx), out);

            cyl_clouds.push_back(cylinders);
        } else {
            pcl::PointCloud<pcl::PointXYZRGB> cylinders;

            for (const auto& [id, params] : cyl_params) {
                const auto points = gen_circle(params);

                for (const auto& p : points) {
                    cylinders.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
                }
            }

            out_cloud += cylinders;
            cyl_clouds.push_back(cylinders);
        }

        prev_cyl_params = cyl_params;
    }

    pcl::io::savePLYFile("./raw.ply", out_cloud);

    return EXIT_SUCCESS;
}
