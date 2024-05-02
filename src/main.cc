#include "cylinder.hh"
#include "utils.hh"

#include <nova/vec.h>
#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstdlib>
#include <future>
#include <ranges>
#include <span>


inline auto& init(const std::string& name) {
    spdlog::set_default_logger(spdlog::create<spdlog::sinks::ansicolor_stdout_sink_mt>(name));

    auto& logger = *spdlog::get(name);
    logger.set_pattern("[%Y-%m-%d %H:%M:%S.%f %z] [%n @%t] %^[%l]%$ %v");

    return logger;
}

[[nodiscard]] auto extract_cylinder(const pcl::PointCloud<pcl::PointXYZRGB>& cloud)
        -> std::tuple<nova::Vec4f, pcl::PointCloud<pcl::PointXYZRGB>, std::vector<nova::Vec3f>>
{
    auto tmp = cloud
             | std::views::transform([](const auto& elem) { return nova::Vec3f { elem.x, elem.y, elem.z }; })
             | std::views::filter([](const auto& elem) { return elem != nova::Vec3f { 0, 0, 0 }; });

    std::vector<nova::Vec3f> points(std::begin(tmp), std::end(tmp));

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

// [[nodiscard]] auto calc_curr_pose(
    // float prev_velocity,
    // float prev_angular_velocity,
    // float prev_x,
    // float prev_y,
    // float prev_theta,
    // float prev_side_slipping = 0
// )
        // -> std::tuple<float, float, float>
// {
    // return {
        // prev_x + prev_velocity * std::cos(prev_theta + prev_angular_velocity / 2 + prev_side_slipping),
        // prev_y + prev_velocity * std::sin(prev_theta + prev_angular_velocity / 2 + prev_side_slipping),
        // prev_theta + prev_angular_velocity
    // };
// }

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
    // R;
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

    offset1 /= num_pts;
    offset2 /= num_pts;

    ret.offset1 = offset1;
    ret.offset2 = offset2;


}

// typedef struct Trafo{
// Mat rot;
// float scale;
// Point3f offset1,offset2;
// } Trafo;

// Trafo registration(vector<Point3f>& pts1, vector<Point3f>& pts2){

// Trafo ret;

	// int NUM=pts1.size();

	// //Subtract offsets

	// Point3d offset1(0.0,0.0,0.0);
	// Point3d offset2(0.0,0.0,0.0);

	// for (int i=0;i<NUM;i++){
		// Point3f v1=pts1[i];
		// Point3f v2=pts2[i];

		// offset1.x+=v1.x;
		// offset1.y+=v1.y;
		// offset1.z+=v1.z;

		// offset2.x+=v2.x;
		// offset2.y+=v2.y;
		// offset2.z+=v2.z;
	// }

	// offset1.x/=NUM;
	// offset1.y/=NUM;
	// offset1.z/=NUM;

	// offset2.x/=NUM;
	// offset2.y/=NUM;
	// offset2.z/=NUM;

	// ret.offset1.x=offset1.x;
	// ret.offset1.y=offset1.y;
	// ret.offset1.z=offset1.z;

	// ret.offset2.x=offset2.x;
	// ret.offset2.y=offset2.y;
	// ret.offset2.z=offset2.z;

	// Mat H=Mat::zeros(3,3,CV_32F);
	// for (int i=0;i<NUM;i++){
		// Point3f v1=pts1[i];
		// Point3f v2=pts2[i];

		// float x1=v1.x-offset1.x;
		// float y1=v1.y-offset1.y;
		// float z1=v1.z-offset1.z;

		// float x2=v2.x-offset2.x;
		// float y2=v2.y-offset2.y;
		// float z2=v2.z-offset2.z;

		// H.at<float>(0,0)+=x2*x1;
		// H.at<float>(0,1)+=x2*y1;
		// H.at<float>(0,2)+=x2*z1;

		// H.at<float>(1,0)+=y2*x1;
		// H.at<float>(1,1)+=y2*y1;
		// H.at<float>(1,2)+=y2*z1;

		// H.at<float>(2,0)+=z2*x1;
		// H.at<float>(2,1)+=z2*y1;
		// H.at<float>(2,2)+=z2*z1;
	// }


	// Mat w(3,3,CV_32F);
	// Mat u(3,3,CV_32F);
	// Mat vt(3,3,CV_32F);

	// SVD::compute(H,w,u, vt);

	// Mat rot=vt.t()*u.t();
	// ret.rot=rot;

	// float numerator=0.0;
	// float denominator=0.0;

	// for (int i=0;i<NUM;i++){
		// Point3f v1=pts1[i];
		// Point3f v2=pts2[i];

		// float x1=v1.x-offset1.x;
		// float y1=v1.y-offset1.y;
		// float z1=v1.z-offset1.z;


		// Mat p2(3,1,CV_32F);

		// p2.at<float>(0,0)=v2.x-offset2.x;
		// p2.at<float>(1,0)=v2.y-offset2.y;
		// p2.at<float>(2,0)=v2.z-offset2.z;

		// p2=rot*p2;

		// float x2=p2.at<float>(0,0);
		// float y2=p2.at<float>(1,0);
		// float z2=p2.at<float>(2,0);


		// numerator+=x1*x2+y1*y2+z1*z2;
		// denominator+=x2*x2+y2*y2+z2*z2;

	// }

	// ret.scale=numerator/denominator;


	// return ret;
// }//end register


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

        pcl::PointCloud<pcl::PointXYZRGB> cylinders;

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
            cylinders += cylinder;
        }

        out_cloud += cylinders;

        pcl::io::savePLYFile(fmt::format("./cyl-{}.ply", idx), cylinders);

        for (const auto& p : cyl_params) {
            fmt::print("{} {}\n", p.x(), p.y());
        }
    }

    pcl::io::savePLYFile("./test.ply", out_cloud);

    return EXIT_SUCCESS;
}
