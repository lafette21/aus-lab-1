#include <Eigen/Core>
#include <Eigen/Dense>
#include <nova/vec.h>
#include <fmt/format.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <charconv>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <span>
#include <sstream>
#include <string>


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

struct trafo {
    Eigen::Matrix3f R;
    nova::Vec3f offset1;
    nova::Vec3f offset2;
    float scale;
};

auto register_(const auto& pts1, const auto& pts2)
        -> trafo
{
    assert(pts1.size() <= pts2.size());

    trafo ret;

    const auto num_pts = pts1.size();

    nova::Vec3f offset1{ 0, 0, 0 };
    nova::Vec3f offset2{ 0, 0, 0 };

    for (std::size_t i = 0; i < num_pts; ++i) {
        const auto& p1 = pts1[i];
        const auto& p2 = pts2[i];

        offset1 += p1;
        offset2 += p2;
    }

    offset1 /= static_cast<float>(num_pts);
    offset2 /= static_cast<float>(num_pts);

    ret.offset1 = offset1;
    ret.offset2 = offset2;

    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

    for (std::size_t i = 0; i < num_pts; ++i) {
        const auto p1 = pts1[i] - offset1;
        const auto p2 = pts2[i] - offset2;

        H(0, 0) += p2.x() * p1.x();
        H(0, 1) += p2.x() * p1.y();
        H(0, 2) += p2.x() * p1.z();

        H(1, 0) += p2.y() * p1.x();
        H(1, 1) += p2.y() * p1.y();
        H(1, 2) += p2.y() * p1.z();

        H(2, 0) += p2.z() * p1.x();
        H(2, 1) += p2.z() * p1.y();
        H(2, 2) += p2.z() * p1.z();
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Matrix3f vt = svd.matrixV().transpose();
    // Eigen::Vector3f w = svd.singularValues();

    ret.R = vt.transpose() * u.transpose();

    float num = 0;
    float denom = 0;

    for (std::size_t i = 0; i < num_pts; ++i) {
        const auto p1 = pts1[i] - offset1;
        const auto p2 = pts2[i] - offset2;

        Eigen::Vector3f p2_;
        p2_(0) = p2.x();
        p2_(1) = p2.y();
        p2_(2) = p2.z();

        const auto p2_rot = ret.R * p2_;

        num += p1.x() * p2_rot(0) + p1.y() * p2_rot(1) + p1.z() * p2_rot(2);
        denom += p2_rot(0) * p2_rot(0) + p2_rot(1) * p2_rot(1) + p2_rot(2) * p2_rot(2);
    }

    ret.scale = num / denom;

    return ret;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view {arg}; });

    if (args.size() < 5) {
        fmt::println("Usage: {} file1.xyz file2.xyz COMMON_NUM res.xyz", args[0]);
        return EXIT_FAILURE;
    }

    const auto cloud1 = read_file<lidar_data_parser>(std::filesystem::path(args[1]).string()).value();
    const auto cloud2 = read_file<lidar_data_parser>(std::filesystem::path(args[2]).string()).value();

    std::size_t common_num = 0;
    std::from_chars(args[3].data(), args[3].data() + args[3].size(), common_num);

    std::vector<nova::Vec3f> pts1, pts2;

    for (std::size_t i = 0; i < common_num; ++i) {
        pts1.emplace_back(cloud1[i].x, cloud1[i].y, cloud1[i].z);
        pts2.emplace_back(cloud2[i].x, cloud2[i].y, cloud2[i].z);
    }

    const auto trafo = register_(pts1, pts2);

    pcl::PointCloud<pcl::PointXYZRGB> out;

    for (const auto& p : cloud1) {
        out.emplace_back(p.x - trafo.offset1.x(), p.y - trafo.offset1.y(), p.z - trafo.offset1.z(), 255, 0, 0);
    }

    for (const auto& p : cloud2) {
        Eigen::Vector3f pt;
        pt(0) = p.x - trafo.offset2.x();
        pt(1) = p.y - trafo.offset2.y();
        pt(2) = p.z - trafo.offset2.z();

        const auto pp = (trafo.R * pt) * trafo.scale;

        out.emplace_back(pp(0), pp(1), pp(2), 0, 255, 0);
    }

    pcl::io::savePLYFile(std::string{ args[4] }, out);

    return EXIT_SUCCESS;
}
