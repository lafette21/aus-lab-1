#include "cylinder.hh"
#include "utils.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
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


inline auto& init(const std::string& name) {
    spdlog::set_default_logger(spdlog::create<spdlog::sinks::ansicolor_stdout_sink_mt>(name));

    auto& logger = *spdlog::get(name);
    logger.set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n @%t] %^[%l]%$ %v");

    return logger;
}

template <std::floating_point T = float>
[[nodiscard]] auto gen_circle(T x, T y, T z, T r, std::size_t num_points = 100)
        -> std::vector<nova::vec3<T>>
{
    std::vector<nova::vec3<T>> ret;
    ret.reserve(num_points);

    const auto angles = nova::linspace<T>(nova::range<T>{ 0.f, 2 * std::numbers::pi_v<T> }, num_points);

    for (const auto& angle : angles) {
        const auto x_ = x + r * std::cos(angle);
        const auto y_ = y + r * std::sin(angle);

        ret.emplace_back(x_, y_, z);
    }

    return ret;
}

template <std::floating_point T = float>
[[nodiscard]] auto gen_circle(nova::vec4<T> params, std::size_t num_points = 100)
        -> std::vector<nova::vec3<T>>
{
    return gen_circle(params.x(), params.y(), params.z(), params.w(), num_points);
}

[[nodiscard]] auto extract_cylinder(const pcl::PointCloud<pcl::PointXYZRGB>& cloud)
        -> std::tuple<nova::Vec4f, pcl::PointCloud<pcl::PointXYZRGB>, std::vector<nova::Vec3f>>
{
    const auto points = cloud
                      | std::views::transform([](const auto& elem) { return nova::Vec3f { elem.x, elem.y, elem.z }; })
                      | std::views::filter([](const auto& elem) { return elem != nova::Vec3f { 0, 0, 0 }; })
                      | ranges::to<std::vector>();

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
        cylinder_params,
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

struct trafo {
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
};

/*
 * @brief   Finding optimal rotation and translation between corresponding 3D points
 *
 * https://nghiaho.com/?page_id=671
 * https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
 *
 */
auto rigid_transform_3D(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B)
        -> trafo
{
    assert(A.rows() == B.rows() and A.cols() == B.cols());

    if (A.rows() != 3) {
        throw std::runtime_error("Matrix A is not 3xN!");
    }

    if (B.rows() != 3) {
        throw std::runtime_error("Matrix B is not 3xN!");
    }

    trafo ret;

    const Eigen::Vector3f centroid_A = A.rowwise().mean();
    const Eigen::Vector3f centroid_B = B.rowwise().mean();

    const Eigen::MatrixXf Am = A.colwise() - centroid_A;
    const Eigen::MatrixXf Bm = B.colwise() - centroid_B;

    const Eigen::Matrix3f H = Am * Bm.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix3f Ut = svd.matrixU().transpose();
    Eigen::Matrix3f V = svd.matrixV();

    ret.R = V * Ut;

    // special reflection case
    if (ret.R.determinant() < 0) {
        V.col(2) *= -1;
        ret.R = V * Ut;
    }

    ret.t = -ret.R * centroid_A + centroid_B;

    return ret;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    [[maybe_unused]] auto& logger = init("logger");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view {arg}; });

    constexpr std::size_t From = 165;
    constexpr std::size_t To = 167;

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> clouds;
    clouds.reserve(To - From);

    spdlog::info("Reading cloud(s)");

    for (std::size_t i = From; i < To; ++i) {
        const auto cloud = read_file<lidar_data_parser>(
            (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", i)).string()
        ).value();
        clouds.push_back(cloud);
    }

    spdlog::info("Read {} cloud(s)", clouds.size());
    spdlog::info("Processing cloud(s)");

    pcl::PointCloud<pcl::PointXYZRGB> out_cloud;

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> cyl_clouds;

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

        std::vector<nova::Vec4f> cyl_params;
        cyl_params.reserve(4);

        for (auto& f : futures) {
            const auto [params, cylinder, rest] = f.get();
            cyl_params.push_back(params);
        }

        for (const auto& p : cyl_params) {
            fmt::print("{} {}\n", p.x(), p.y());
        }

        pcl::PointCloud<pcl::PointXYZRGB> cylinders;

        for (const auto& param : cyl_params) {
            if (std::isnan(param.x()) or std::isnan(param.y()) or std::isnan(param.z()) or std::isnan(param.w())) {
                continue;
            }

            const auto points = gen_circle(param);

            for (const auto& p : points) {
                cylinders.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
            }
        }

        out_cloud += cylinders;

        if (cyl_clouds.size() > 0) {
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

            const auto trafo = rigid_transform_3D(A, B);

            pcl::PointCloud<pcl::PointXYZRGB> out;

            for (const auto& p : prev_cloud) {
                const Eigen::Vector3f pt = Eigen::Vector3f{ p.x, p.y, p.z };
                const auto ptt = (trafo.R * pt) + trafo.t;
                out.emplace_back(ptt.x(), ptt.y(), ptt.z(), 255, 0, 0);
            }

            for (const auto& p : cylinders) {
                out.push_back(p);
            }

            pcl::io::savePLYFile("./registered.ply", out);

            break;
        }

        cyl_clouds.push_back(cylinders);
    }

    pcl::io::savePLYFile("./raw.ply", out_cloud);

    return EXIT_SUCCESS;
}
