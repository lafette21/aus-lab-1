#include "nova/vec.h"
#include "simulator/logging.hh"
#include "utils.hh"
#include "types.hh"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <nova/io.h>
#include <range/v3/view/enumerate.hpp>
#include <spdlog/spdlog.h>

#include <charconv>
#include <chrono>
#include <filesystem>
#include <future>
#include <numbers>
#include <ranges>


std::pair<std::vector<nova::Vec4f>, std::vector<nova::Vec4f>> pairing(const std::vector<nova::Vec4f>& params_a, const std::vector<nova::Vec4f>& params_b, float threshold = 0.5f) {
    std::vector<nova::Vec4f> ret_a;
    std::vector<nova::Vec4f> ret_b;
    std::vector<std::vector<float>> dist_mx;

    for (const auto& a : params_a) {
        dist_mx.emplace_back(std::vector<float>{});
        auto& vec = dist_mx.back();
        const auto& c_a = nova::Vec2f{ a.x(), a.y() };

        for (const auto& b : params_b) {
            const auto& c_b = nova::Vec2f{ b.x(), b.y() };
            vec.push_back((c_a - c_b).length());
        }
    }

    // for (const auto& vec : dist_mx) {
        // for (const auto& elem : vec) {
            // std::cout << elem << ", ";
        // }
        // std::cout << std::endl;
    // }

    for (const auto& [idx, vec] : ranges::views::enumerate(dist_mx)) {
        const auto& min = std::ranges::min(vec);
        const auto idx_b = std::distance(vec.begin(), std::ranges::find(vec, min));

        if (min < threshold) {
            ret_a.push_back(params_a[idx]);
            ret_b.push_back(params_b[idx_b]);
            logging::debug("({}, {})\t({}, {})\tdist: {}", params_a[idx].x(), params_a[idx].y(), params_b[idx_b].x(), params_b[idx_b].y(), min);
        }
    }

    return { ret_a, ret_b };
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
    std::vector<nova::Vec4f> prev_cyl_params;
    Eigen::Matrix3f trafo = Eigen::Matrix3f::Identity();

    for (const auto& [idx, cloud] : ranges::views::enumerate(clouds)) {
        logging::debug("Cloud size: {}", cloud.size());

        const auto start = nova::now();

        const auto downsampled = downsample(cloud);
        const auto filtered = filter_planes(downsampled);
        const auto clusters = cluster(filtered);

        logging::info("Clusters found: {}", clusters.size());

        for (const auto& cl : clusters) {
            logging::debug("Cluster size: {}", cl.indices.size());
        }

        const auto point_clouds = extract_clusters(filtered, clusters);

        logging::debug("Point clouds extracted: {}", point_clouds.size());

        for (const auto& elem : point_clouds) {
            logging::debug("Cloud size: {}", elem.size());
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

        // for (const auto& p : cyl_params) {
            // fmt::print("{} {}\n", p.x(), p.y());
        // }

        if (prev_cyl_params.size() > 0) {
            const auto [curr_cyl_params, new_prev_cyl_params] = pairing(cyl_params, prev_cyl_params);

            pcl::PointCloud<pcl::PointXYZRGB> prev_points;

            for (const auto& params : new_prev_cyl_params) {
                prev_points.emplace_back(params.x(), params.y(), 0, 0, 255, 0);
            }

            pcl::PointCloud<pcl::PointXYZRGB> curr_points;

            for (const auto& params : curr_cyl_params) {
                curr_points.emplace_back(params.x(), params.y(), 0, 0, 255, 0);
            }

            out += curr_points;

            const auto min_size = std::min(std::size(prev_points), std::size(curr_points));

            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(2, static_cast<int>(min_size));
            Eigen::MatrixXf B = Eigen::MatrixXf::Zero(2, static_cast<int>(min_size));

            for (std::size_t i = 0; i < min_size; ++i) {
                A(0, static_cast<int>(i)) = prev_points[i].x;
                A(1, static_cast<int>(i)) = prev_points[i].y;

                B(0, static_cast<int>(i)) = curr_points[i].x;
                B(1, static_cast<int>(i)) = curr_points[i].y;
            }

            const auto new_trafo_tmp = rigid_transform_2D(B, A);

            Eigen::Matrix3f new_trafo = Eigen::Matrix3f::Identity();
            new_trafo.block<2, 2>(0, 0) = new_trafo_tmp.R;
            new_trafo.block<2, 1>(0, 2) = new_trafo_tmp.t;

            trafo = trafo * new_trafo;

            pcl::PointCloud<pcl::PointXYZRGB> registered;

            for (const auto& p : curr_points) {
                const Eigen::Vector3f pt = Eigen::Vector3f{ p.x, p.y, 1.0f };
                const Eigen::Vector3f ptt = trafo * pt;
                registered.emplace_back(ptt.x(), ptt.y(), 0, 255, 0, 0);
            }

            for (const auto& p : prev_points) {
                registered.emplace_back(p.x, p.y, 0, 0, 255, 0);
            }

            pcl::io::savePLYFile(fmt::format("./registered-{}.ply", idx), registered);
        } else {
            pcl::PointCloud<pcl::PointXYZRGB> points;

            for (const auto& params : cyl_params) {
                points.emplace_back(params.x(), params.y(), 0, 0, 255, 0);
            }

            out += points;
        }

        prev_cyl_params = cyl_params;

        logging::info("Processing took: {}", std::chrono::duration_cast<std::chrono::milliseconds>(nova::now() - start));
    }

    pcl::io::savePLYFile("./raw.ply", out);
}
