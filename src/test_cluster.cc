#include "utils.hh"

#include <nova/utils.h>
#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/filters/extract_indices.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing.h>
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

    std::size_t cloud_id;

    std::from_chars(args[2].begin(), args[2].begin() + args[2].size(), cloud_id);

    spdlog::info("Reading cloud");

    const auto cloud_tmp = read_file<lidar_data_parser>(
        (std::filesystem::path(args[1]) / fmt::format("test_fn{}.xyz", cloud_id)).string()
    ).value();

    spdlog::info("Processing cloud(s)");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(segment(cloud_tmp));
    pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(1000000);

    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);

    reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size () << " points." << std::endl;
    std::cout << "These are the indices of the points of the initial" <<
    std::endl << "cloud that belong to the first cluster:" << std::endl;
    std::size_t counter = 0;
    while (counter < clusters[0].indices.size ())
    {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
    }
    std::cout << std::endl;

    pcl::io::savePLYFile("./cluster.ply", *reg.getColoredCloud());

    return EXIT_SUCCESS;
}
