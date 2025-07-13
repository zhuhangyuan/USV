#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "algorithm/test_inference.h"

class DetectionNode : public rclcpp::Node
{
  public:
    DetectionNode()
        : Node("detection_3d_node")
    {
        
    }

    ~DetectionNode()
    {
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}