#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>

#include <memory>

#include <cstdlib>


class perception_node : public rclcpp::Node {
public:
    perception_node()
        : Node("Perception")
    {
        m_subscription = this->create_subscription<sensor_msgs::msg::PointCloud>(
            "/lidar",
            10,
            std::bind(&perception_node::topic_callback, this, std::placeholders::_1)
        );
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr m_subscription;

    void topic_callback(const sensor_msgs::msg::PointCloud& msg) const {
        RCLCPP_INFO(this->get_logger(), "I heard: '%d.%d'", msg.header.stamp.sec, msg.header.stamp.nanosec);
    }
};


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<perception_node>());
    rclcpp::shutdown();

    return EXIT_SUCCESS;
}
