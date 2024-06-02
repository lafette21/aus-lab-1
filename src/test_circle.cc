#include "utils.hh"

#include <nova/vec.h>
#include <nova/utils.h>
#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <spdlog/spdlog.h>

#include <charconv>
#include <cstdlib>
#include <ranges>
#include <span>


inline auto& init(const std::string& name) {
    spdlog::set_default_logger(spdlog::create<spdlog::sinks::ansicolor_stdout_sink_mt>(name));

    auto& logger = *spdlog::get(name);
    logger.set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n @%t] %^[%l]%$ %v");

    return logger;
}

[[nodiscard]] auto flatten(const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    pcl::PointCloud<pcl::PointXYZRGB> ret;
    ret.reserve(cloud.size());

    for (const auto& elem : cloud) {
        ret.emplace_back(elem.x, elem.y, 0);
    }

    return ret;
}

auto segment(const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud = cloud.makeShared();
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    // seg.setOptimizeCoefficients (false);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1);

    seg.setInputCloud(_cloud);
    seg.segment(*inliers, *coefficients);

    while (inliers->indices.size() > 500) {
        // extract inliers
        pcl::ExtractIndices<pcl::PointXYZRGB> extractor;
        extractor.setInputCloud(_cloud);
        extractor.setIndices(inliers);
        extractor.setNegative(true); // extract the inliers in consensus model (the part to be removed from point cloud)
        extractor.filter(*_cloud); // cloud_inliers contains the found plane

        seg.segment(*inliers, *coefficients);
    }

    return *_cloud;
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
        const auto cloud = read_file<lidar_data_parser>(
            (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", i)).string()
        ).value();
        clouds.push_back(cloud);
    }

    spdlog::info("Read {} cloud(s)", clouds.size());
    spdlog::info("Processing cloud(s)");

    const auto cloud = segment(clouds[0]);

    // pcl::PointCloud<pcl::PointXYZRGB> out;

    // for (std::size_t i = 0; i < clouds[0].size(); ++i) {
        // if (std::end(indices) == std::ranges::find(indices, i)) {
            // out.push_back(clouds[0][i]);
        // }
    // }

    // for (const auto& idx : inliers->indices) {
        // out.push_back(clouds[0][static_cast<std::size_t>(idx)]);
    // }

    const auto out = flatten(cloud);

    pcl::io::savePLYFile("./flat.ply", out);

    return EXIT_SUCCESS;
}
