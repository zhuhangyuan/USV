#include "rclcpp/rclcpp.hpp"
#include "robot_msgs/msg/object2_d.hpp"

#include <chrono>
#include <robot_msgs/msg/detail/object2_d__struct.hpp>

using namespace std::chrono_literals;

class Publisher : public rclcpp::Node
{
  public:
    Publisher()
        : Node("publisher")
    {
        publisher = this->create_publisher<robot_msgs::msg::Object2D>("topic", 10);
        timer = this->create_wall_timer(
            500ms, std::bind(&Publisher::timer_callback, this));
    }

  private:
    void timer_callback()
    {
        auto message = robot_msgs::msg::Object2D();
        message.center_x = 1.0;
        message.center_y = 2.0;
        message.label = "Ship";
        message.confidence = 0.8;
        message.width = 6;
        message.height = 4;
        message.header.stamp = this->now();
        message.header.frame_id = "camera_frame";
        publisher->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer;
    rclcpp::Publisher<robot_msgs::msg::Object2D>::SharedPtr publisher;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Publisher>());
    rclcpp::shutdown();
    return 0;
}