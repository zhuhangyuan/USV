#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
// #include <robot_msgs/msg/detail/video_detections__struct.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "robot_msgs/msg/frame_detections.hpp"
// #include "robot_msgs/msg/video_detections.hpp"

#include "algorithm/detect_service.h"

class DetectionNode : public rclcpp::Node
{
  public:
    DetectionNode()
        : Node("detection_node")
    {
        this->declare_parameter<std::string>("output_file", "output.mp4");
        this->declare_parameter<int>("fps", 30);
        output_file = this->get_parameter("output_file").as_string();
        fps = this->get_parameter("fps").as_int();

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
        // this->declare_parameter<int>("input_size");
        // int input_size = this->get_parameter("input_size").as_int();

        // this->detect_service = std::make_unique<DetectService>(engine_path, cv::Size(input_size, input_size));

        this->declare_parameter("input_size", std::vector<int64_t>{1024, 512});
        std::vector<int64_t> input_size = this->get_parameter("input_size").as_integer_array();
        this->detect_service = std::make_unique<DetectService>(engine_path, cv::Size(static_cast<int>(input_size[0]), static_cast<int>(input_size[1])));

        publisher = this->create_publisher<robot_msgs::msg::FrameDetections>("detection_info", 10);
    }

    ~DetectionNode()
    {
        if (video_writer.isOpened()) {
            video_writer.release();  // 释放资源（新增）
            RCLCPP_INFO(this->get_logger(), "视频文件已关闭：%s", output_file.c_str());
        }
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
            auto res = this->detect_service->predict(input_img, objects, 0.3, 0.5, 30, 1);


            cv::imshow("Subscribed Image", res);
            cv::waitKey(1);

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
            // frames.frames.push_back(frame);
            publisher->publish(frame);

            if (!video_initialized) {
            if (!res.empty()) {
                // 使用实际图像尺寸初始化（修复尺寸不匹配）
                frame_size = cv::Size(res.cols, res.rows);
                int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // 改用兼容编码
                video_writer.open(output_file, codec, fps, frame_size);
                if (video_writer.isOpened()) {
                    video_initialized = true;
                    RCLCPP_INFO(this->get_logger(), "视频写入器初始化成功，尺寸：%dx%d", 
                               frame_size.width, frame_size.height);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "视频初始化失败！请检查路径/编码支持");
                    return;  // 避免后续无效写入
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "首帧图像无效，跳过初始化");
                return;
            }
        }

        // 新增：确保res有效时再写入
        if (!res.empty()) {
            video_writer.write(res);
        } else {
            RCLCPP_WARN(this->get_logger(), "当前帧无效，跳过写入");
        }

            

            
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription;
    rclcpp::Publisher<robot_msgs::msg::FrameDetections>::SharedPtr publisher;
    std::unique_ptr<DetectService> detect_service;

    std::string output_file;
    int fps;
    cv::Size frame_size;
    cv::VideoWriter video_writer;
    bool video_initialized = false;

    // auto frames = robot_msgs::msg::VideoDetections();
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}