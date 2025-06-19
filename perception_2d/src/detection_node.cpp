#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "algorithm/detect_service.h"

class DetectionNode : public rclcpp::Node
{
  public:
    DetectionNode()
        : Node("detection_node")
    {
        // 订阅 CameraNode 发布的图像话题
        subscription = this->create_subscription<sensor_msgs::msg::Image>(
            "video_frames", // 话题名称
            10,             // QoS 队列长度
            [&](const sensor_msgs::msg::Image::SharedPtr msg)
            {
                this->image_callback(msg);
            });

        // 读取模型权重
        this->detect_service = std::make_unique<DetectService>();
    }

    ~DetectionNode()
    {
    }

  private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            // 将 ROS Image 消息转换为 cv::Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);

            cv::Mat frame = cv_ptr->image;
            cv::Mat input_img;
            cv::resize(frame, input_img, cv::Size(1024, 1024));

            std::vector<Object> objects;
            auto res = this->detect_service->predict(input_img, objects, 0.5, 0.5, 10, 80);

            cv::imshow("Subscribed Image", res);
            cv::waitKey(1);

            // 打印图像信息
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

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription;
    std::unique_ptr<DetectService> detect_service;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}