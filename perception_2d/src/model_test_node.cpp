#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "algorithm/detect_service.h"

class ModelTestNode : public rclcpp::Node
{
  public:
    ModelTestNode()
        : Node("model_test_node")
    {

        // 读取模型权重
        this->declare_parameter<std::string>("engine_path", "");
        std::string engine_path = this->get_parameter("engine_path").as_string();
        this->declare_parameter<int>("input_size");
        int input_size = this->get_parameter("input_size").as_int();

        this->detect_service = std::make_unique<DetectService>(engine_path, cv::Size(input_size, input_size));

        image();
    }

    ~ModelTestNode()
    {
    }

  private:
    void image()
    {
        std::string folderPath = "/home/jiahan/Desktop/001-mix/WSODD_dataset/images/test/"; // 末尾需要加斜杠

        // 获取文件夹中所有图片文件路径
        std::vector<std::string> imagePaths;
        cv::glob(folderPath + "*.jpg", imagePaths, false); // 支持添加多个扩展名

        // 检查是否找到图片
        if (imagePaths.empty())
        {
            std::cout << "错误：未找到图片文件！请检查路径：" << folderPath << std::endl;
        }

        for(const auto& path : imagePaths)
        {
            cv::Mat img = cv::imread(path);

            cv::Mat origin_img = img;
            cv::Mat input_img;
            cv::resize(origin_img, input_img, this->detect_service->get_input_size());
            std::vector<Object> objects;
            auto res = this->detect_service->predict(input_img, objects, 0.5, 0.5, 10, 14);

            cv::imshow("Image", res);
            cv::waitKey(1000);
        }
    }

    std::unique_ptr<DetectService> detect_service;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ModelTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}