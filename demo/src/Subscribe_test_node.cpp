#include "rclcpp/rclcpp.hpp"
#include "robot_msgs/msg/frame_detections.hpp"
#include <robot_msgs/msg/detail/frame_detections__struct.hpp>

class Subscriber : public rclcpp::Node
{
  public:
    Subscriber()
        : Node("subscriber")
    {
        subscription_ = this->create_subscription<robot_msgs::msg::FrameDetections>(
            "detection_info", 10, std::bind(&Subscriber::topic_callback, this, std::placeholders::_1));
    }

  private:
    void topic_callback(const robot_msgs::msg::FrameDetections &frame) const
    {
        RCLCPP_INFO(this->get_logger(), "\nReceived frame at %.3f sec with %zu detections:", 
               frame.timestamp, frame.detections.size());
        for(const auto& detection : frame.detections)
        {
            RCLCPP_INFO(
            this->get_logger(),
            "Detection:\n"
            "  class_id: %d\n"
            "  confidence: %.2f\n"
            "  topleft: (%d, %d)\n"
            "  bottomright: (%d, %d)",
            detection.class_id,
            detection.confidence,
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2);
    }
    }

    rclcpp::Subscription<robot_msgs::msg::FrameDetections>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Subscriber>());
    rclcpp::shutdown();
    return 0;
}