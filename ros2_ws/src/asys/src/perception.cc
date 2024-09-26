#include "asys/utils.hh"
#include "asys/msg/point.hpp"
#include "asys/msg/point_list.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <memory>

#include <cstdlib>


class perception_node : public rclcpp::Node {
public:
    perception_node()
        : Node("perception")
        , m_last_time(this->now())
    {
        this->declare_parameter("inputTopic", rclcpp::PARAMETER_STRING);
        this->declare_parameter("outputTopic", rclcpp::PARAMETER_STRING);
        this->declare_parameter("qos", 10);

        const rclcpp::Parameter inputTopic = this->get_parameter("inputTopic");
        const rclcpp::Parameter outputTopic = this->get_parameter("outputTopic");
        const rclcpp::Parameter qos = this->get_parameter("qos");

        // m_subscription = this->create_subscription<sensor_msgs::msg::PointCloud>(
            // inputTopic.as_string(),
            // qos.as_int(),
            // std::bind(&perception_node::topic_callback, this, std::placeholders::_1)
        // );
        m_subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            inputTopic.as_string(),
            qos.as_int(),
            std::bind(&perception_node::topic_callback, this, std::placeholders::_1)
        );

        m_publisher = this->create_publisher<asys::msg::PointList>(outputTopic.as_string(), qos.as_int());
    }

private:
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr m_subscription;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_subscription;
    rclcpp::Publisher<asys::msg::PointList>::SharedPtr m_publisher;
    // int m_msg_count = 0;
    // static constexpr auto Divisor = 5;
    rclcpp::Time m_last_time;  // To store the last time we processed a message
    const double m_target_interval = 1.0 / 4.0;  // Target frequency (4 Hz), so we need 1/4 seconds between messages

    // void topic_callback(const sensor_msgs::msg::PointCloud& msg) {
        // if (m_msg_count % Divisor == 0) {
            // RCLCPP_INFO(this->get_logger(), "I heard: '%d.%d'", msg.header.stamp.sec, msg.header.stamp.nanosec);
        // }
        // m_msg_count++;
    // }

    void topic_callback(const sensor_msgs::msg::PointCloud2& msg) {
        const rclcpp::Time curr_time = this->now();
        const double elapsed = (curr_time - m_last_time).seconds();

        // Process the message only if more than 0.25 seconds (1/4 Hz) have passed
        if (elapsed >= m_target_interval) {
            RCLCPP_INFO(this->get_logger(), "Processing point cloud at: '%d.%d'", msg.header.stamp.sec, msg.header.stamp.nanosec);

            // Update the last processed time
            m_last_time = curr_time;

            pcl::PointCloud<pcl::PointXYZRGB> cloud;
            pcl::fromROSMsg(msg, cloud);

            const auto downsampled = downsample(cloud);
            const auto filtered = filter_planes(downsampled);
            const auto clusters = cluster(filtered);
            const auto point_clouds = extract_clusters(filtered, clusters);

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

            asys::msg::PointList out_msg;
            out_msg.header.stamp = this->now();
            out_msg.header.frame_id = "point_list";

            for (const auto& param : cyl_params) {
                asys::msg::Point pt;
                pt.x = param.x();
                pt.y = param.y();
                out_msg.points.push_back(pt);
            }

            m_publisher->publish(out_msg);
        }
    }
};


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<perception_node>());
    rclcpp::shutdown();

    return EXIT_SUCCESS;
}
