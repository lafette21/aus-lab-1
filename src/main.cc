#include "nova/vec.h"
#include "simulator/logging.hh"
#include "utils.hh"
#include "types.hh"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <nova/io.h>
#include <spdlog/spdlog.h>

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

    for (const auto& [idx, vec] : std::views::enumerate(dist_mx)) {
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
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> circle_clouds;
    std::vector<nova::Vec4f> prev_cyl_params;
    Eigen::Matrix4f trafo = Eigen::Matrix4f::Identity();

    for (const auto& [idx, cloud] : std::views::enumerate(clouds)) {
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

            pcl::PointCloud<pcl::PointXYZRGB> circles;

            for (const auto& params : curr_cyl_params) {
                const auto points = gen_circle(params);

                for (const auto& p : points) {
                    circles.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
                }
            }

            pcl::PointCloud<pcl::PointXYZRGB> prev_circles;

            for (const auto& params : new_prev_cyl_params) {
                const auto points = gen_circle(params);

                for (const auto& p : points) {
                    prev_circles.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
                }
            }

            out += circles;

            const auto min_size = std::min(std::size(prev_circles), std::size(circles));

            Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3, static_cast<int>(min_size));
            Eigen::MatrixXf B = Eigen::MatrixXf::Zero(3, static_cast<int>(min_size));

            for (std::size_t i = 0; i < min_size; ++i) {
                A(0, static_cast<int>(i)) = prev_circles[i].x;
                A(1, static_cast<int>(i)) = prev_circles[i].y;
                A(2, static_cast<int>(i)) = prev_circles[i].z;

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

            for (const auto& p : prev_circles) {
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

        logging::info("Processing took: {}", std::chrono::duration_cast<std::chrono::milliseconds>(nova::now() - start));
    }

    pcl::io::savePLYFile("./raw.ply", out);
}
