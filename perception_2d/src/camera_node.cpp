#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>



using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node
{
  public:
    CameraNode()
        : Node("camera_node")
    {
        // 声明参数：视频文件路径（默认为空，需要用户指定）
        this->declare_parameter<std::string>("video_path", "");

        // 获取视频路径参数
        std::string video_path = this->get_parameter("video_path").as_string();
        if (video_path.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "必须提供 'video_path' 参数！");
            rclcpp::shutdown();
            return;
        }

        // 打开视频文件
        cap.open(video_path);
        if (!cap.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "无法打开视频文件：%s", video_path.c_str());
            rclcpp::shutdown();
            return;
        }

        // 创建图像发布者
        publisher = this->create_publisher<sensor_msgs::msg::Image>("video_frames", 10);

        // 设置定时器
        timer = this->create_wall_timer(
            33ms, [this]()
            {
                this->publish_call_back();
            });
    }

  private:
    void publish_call_back()
    {
        cv::Mat frame;
        cap >> frame; // 读取一帧
        
        if (frame.empty())
        {
            RCLCPP_WARN(this->get_logger(), "视频结束或帧为空");
            rclcpp::shutdown();
            return;
        }

        // 转换为 ROS Image 消息
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera";

        publisher->publish(*msg);

        RCLCPP_INFO(this->get_logger(), "image szie = %d x %d", frame.cols, frame.rows);
    }

    cv::VideoCapture cap;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher;
    rclcpp::TimerBase::SharedPtr timer;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}