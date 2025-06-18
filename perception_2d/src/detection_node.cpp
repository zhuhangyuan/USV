#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "algorithm/yolov8.h"

using std::placeholders::_1;

class DetectionNode : public rclcpp::Node
{
  public:
    DetectionNode()
        : Node("detection_node")
    {
        // 订阅 CameraNode 发布的图像话题
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video_frames", // 话题名称
            10,             // QoS 队列长度
            std::bind(&DetectionNode::image_callback, this, _1));

        // 初始化 OpenCV 窗口
        cv::namedWindow("Subscribed Image", cv::WINDOW_AUTOSIZE);
    }

    ~DetectionNode()
    {
        cv::destroyAllWindows();
    }

  private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            // 将 ROS Image 消息转换为 cv::Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
                msg,
                msg->encoding // 通常为 "bgr8" 或 "rgb8"
            );

            // 获取 OpenCV 图像
            cv::Mat frame = cv_ptr->image;

            // 在这里添加你的图像处理代码（示例：显示图像）
            cv::imshow("Subscribed Image", frame);
            cv::waitKey(1); // 刷新 OpenCV 窗口

            // 打印图像信息（可选）
            RCLCPP_INFO(
                this->get_logger(),
                "Received image: width=%d, height=%d",
                frame.cols, frame.rows);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}