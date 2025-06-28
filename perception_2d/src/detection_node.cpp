#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "robot_msgs/msg/frame_detections.hpp"

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
        this->declare_parameter<std::string>("engine_path", "");
        std::string engine_path = this->get_parameter("engine_path").as_string();
        this->declare_parameter<int>("input_size");
        int input_size = this->get_parameter("input_size").as_int();

        this->detect_service = std::make_unique<DetectService>(engine_path, cv::Size(input_size, input_size));

        publisher = this->create_publisher<robot_msgs::msg::FrameDetections>("detection_info", 10);
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

            cv::Mat origin_img = cv_ptr->image;
            cv::Mat input_img;
            cv::resize(origin_img, input_img, this->detect_service->get_input_size());

            std::vector<Object> objects;
            auto res = this->detect_service->predict(input_img, objects, 0.5, 0.5, 10, 1);

            auto frame = robot_msgs::msg::FrameDetections();
            frame.timestamp = this->now().seconds();

            for(const auto& object : objects)
            {
                auto detection = robot_msgs::msg::Detection();
                detection.class_id = object.label;
                detection.confidence = object.prob;
                detection.bbox.x1 = object.rect.x;
                detection.bbox.y1 = object.rect.y;
                detection.bbox.x2 = object.rect.x + object.rect.width;
                detection.bbox.y2 = object.rect.y - object.rect.height;
                frame.detections.push_back(detection);
            }
            publisher->publish(frame);

            cv::imshow("Subscribed Image", res);
            cv::waitKey(1);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription;
    rclcpp::Publisher<robot_msgs::msg::FrameDetections>::SharedPtr publisher;
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