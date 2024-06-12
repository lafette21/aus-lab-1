#include "nova/vec.h"
#include "simulator/logging.hh"
#include "utils.hh"
#include "types.hh"

#include <fmt/format.h>
#include <nova/io.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <future>
#include <numbers>
#include <ranges>


auto closest_to(const nova::Vec2f& point, const std::vector<nova::Vec2f>& points)
        -> nova::Vec2f
{
    nova::Vec2f closest;
    auto min_dist = std::numeric_limits<float>::max();

    for (const auto& p : points) {
        if (const auto dist = (point - p).length(); dist < min_dist) {
            min_dist = dist;
            closest = p;
        }
    }

    return closest;
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

    logging::info("Reading cloud(s)");

    for (std::size_t i = from; i < to; ++i) {
        const auto cloud = nova::read_file<lidar_data_parser>(
            (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", i)).string()
        ).value();
        clouds.push_back(cloud);
    }

    logging::info("Read {} cloud(s)", clouds.size());
    logging::info("Processing cloud(s)");

    pcl::PointCloud<pcl::PointXYZRGB> out;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> circle_clouds;
    std::vector<nova::Vec4f> prev_cyl_params;
    Eigen::Matrix4f trafo = Eigen::Matrix4f::Identity();

    for (const auto& [idx, cloud] : std::views::enumerate(clouds)) {
        logging::info("Cloud size: {}", cloud.size());

        const auto downsampled = downsample(cloud);
        const auto filtered = filter_planes(downsampled);
        const auto clusters = cluster(filtered);

        logging::info("Clusters found: {}", clusters.size());

        for (const auto& cl : clusters) {
            logging::info("Cluster size: {}", cl.indices.size());
        }

        const auto point_clouds = extract_clusters(filtered, clusters);

        logging::info("Point clouds extracted: {}", point_clouds.size());

        for (const auto& elem : point_clouds) {
            logging::info("Cloud size: {}", elem.size());
        }

        std::vector<std::future<std::tuple<nova::Vec4f, pcl::PointCloud<pcl::PointXYZRGB>, std::vector<nova::Vec3f>>>> futures;

        for (const auto& elem : point_clouds) {
            futures.push_back(std::async(extract_cylinder, elem));
        }

        std::vector<nova::Vec4f> cyl_params;

        for (auto& f : futures) {
            const auto [params, cylinder, rest] = f.get();

            if (std::isnan(params.x()) or std::isnan(params.y()) or std::isnan(params.z()) or std::isnan(params.w())) {
                continue;
            }

            cyl_params.push_back(params);
        }

        for (const auto& p : cyl_params) {
            fmt::print("{} {}\n", p.x(), p.y());
        }

        if (prev_cyl_params.size() > 0) {
            std::vector<nova::Vec4f> curr_cyl_params;

            for (const auto& elem : cyl_params) {
                const auto prev_cyl_centers = prev_cyl_params
                                            | std::views::transform([](const auto& elem) { return nova::Vec2f{ elem.x(), elem.y() }; })
                                            | ranges::to<std::vector>();
                const auto closest = closest_to({ elem.x(), elem.y() }, prev_cyl_centers);

                constexpr auto Threshold = 0.5f;

                if ((closest - nova::Vec2f{ elem.x(), elem.y() }).length() < Threshold) {
                    curr_cyl_params.push_back(elem);
                }
            }

            pcl::PointCloud<pcl::PointXYZRGB> circles;

            for (const auto& params : curr_cyl_params) {
                const auto points = gen_circle(params);

                for (const auto& p : points) {
                    circles.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
                }
            }

            out += circles;

            const auto& prev_cloud = circle_clouds.back();
            const auto min_size = std::min(std::size(prev_cloud), std::size(circles));

            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3, static_cast<int>(min_size));
            Eigen::MatrixXf B = Eigen::MatrixXf::Zero(3, static_cast<int>(min_size));

            for (std::size_t i = 0; i < min_size; ++i) {
                A(0, static_cast<int>(i)) = prev_cloud[i].x;
                A(1, static_cast<int>(i)) = prev_cloud[i].y;
                A(2, static_cast<int>(i)) = prev_cloud[i].z;

                B(0, static_cast<int>(i)) = circles[i].x;
                B(1, static_cast<int>(i)) = circles[i].y;
                B(2, static_cast<int>(i)) = circles[i].z;
            }

            const auto new_trafo_tmp = rigid_transform_3D(B, A);

            Eigen::Matrix4f new_trafo = Eigen::Matrix4f::Identity();
            new_trafo.block<3, 3>(0, 0) = new_trafo_tmp.R;
            new_trafo.block<3, 1>(0, 3) = new_trafo_tmp.t;

            trafo = trafo * new_trafo;

            pcl::PointCloud<pcl::PointXYZRGB> registered;

            for (const auto& p : circles) {
                const Eigen::Vector4f pt = Eigen::Vector4f{ p.x, p.y, p.z, 1.0f };
                const Eigen::Vector4f ptt = trafo * pt;
                registered.emplace_back(ptt.x(), ptt.y(), ptt.z(), 255, 0, 0);
            }

            for (const auto& p : prev_cloud) {
                registered.emplace_back(p.x, p.y, p.z, 0, 255, 0);
            }

            pcl::io::savePLYFile(fmt::format("./registered-{}.ply", idx), registered);

            circle_clouds.push_back(circles);
        } else {
            pcl::PointCloud<pcl::PointXYZRGB> circles;

            for (const auto& params : cyl_params) {
                const auto points = gen_circle(params);

                for (const auto& p : points) {
                    circles.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
                }
            }

            out += circles;
            circle_clouds.push_back(circles);
        }

        prev_cyl_params = cyl_params;
    }

    pcl::io::savePLYFile("./raw.ply", out);
}
