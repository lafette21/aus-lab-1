#ifndef UTILS_HH
#define UTILS_HH

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>


[[nodiscard]] inline auto downsample(const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    pcl::PointCloud<pcl::PointXYZRGB> ret;

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud.makeShared());
    vg.setLeafSize(0.07f, 0.07f, 0.07f); // TODO: Need a good param - Set the leaf size (adjust as needed)
    vg.filter(ret);

    return ret;
}

[[nodiscard]] inline auto filter_planes(const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud = cloud.makeShared();
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    // seg.setOptimizeCoefficients(false);
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

[[nodiscard]] inline auto cluster(const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(cloud);
    pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(_cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*_cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(1000000);

    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(_cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);

    reg.setSmoothnessThreshold(18.0f / 180.0f * std::numbers::pi_v<float>);
    reg.setCurvatureThreshold(1.0);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    pcl::io::savePLYFile("./clusters.ply", *reg.getColoredCloud());

    return clusters;
}

#endif // UTILS_HH
