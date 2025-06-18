#include "rclcpp/rclcpp.hpp"
#include "robot_msgs/msg/object2_d.hpp"

class Subscriber : public rclcpp::Node
{
  public:
    Subscriber()
        : Node("subscriber")
    {
        subscription_ = this->create_subscription<robot_msgs::msg::Object2D>(
            "topic", 10, std::bind(&Subscriber::topic_callback, this, std::placeholders::_1));
    }

  private:
    void topic_callback(const robot_msgs::msg::Object2D &msg) const
    {
        RCLCPP_INFO(
            this->get_logger(),
            "Received Object2D:\n"
            "  header:\n"
            "    stamp: sec=%d, nanosec=%d\n"
            "    frame_id: %s\n"
            "  label: %s\n"
            "  confidence: %.2f\n"
            "  center: (x=%.2f, y=%.2f)\n"
            "  size: (width=%.2f, height=%.2f)",
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            msg.header.frame_id.c_str(),
            msg.label.c_str(),
            msg.confidence,
            msg.center_x,
            msg.center_y,
            msg.width,
            msg.height);
    }

    rclcpp::Subscription<robot_msgs::msg::Object2D>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    std::cout << "Hello World!" << std::endl;
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Subscriber>());
    rclcpp::shutdown();
    return 0;
}