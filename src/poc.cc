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


int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    [[maybe_unused]] auto& logger = init("logger");

    const auto args = std::span<char*>(argv, static_cast<std::size_t>(argc))
                    | std::views::transform([](const auto& arg) { return std::string_view{ arg }; });

    std::size_t cloud_id;

    std::from_chars(args[2].begin(), args[2].begin() + args[2].size(), cloud_id);

    logging::info("Reading cloud");

    const auto cloud_tmp = nova::read_file<lidar_data_parser>(
        (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", cloud_id)).string()
    ).value();

    logging::info("Processing cloud(s)");

    const auto downsampled = downsample(cloud_tmp);
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

    pcl::io::savePLYFile("./downsampled.ply", downsampled);
    pcl::io::savePLYFile("./filtered.ply", filtered);

    pcl::PointCloud<pcl::PointXYZRGB> out;
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

    for (const auto& param : cyl_params) {
        pcl::PointCloud<pcl::PointXYZRGB> circle;

        const auto points = gen_circle(param);

        for (const auto& p : points) {
            circle.emplace_back(p.x(), p.y(), p.z(), 0, 255, 0);
        }

        out += circle;
    }

    pcl::io::savePLYFile("./circles.ply", out);
}
