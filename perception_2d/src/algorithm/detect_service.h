#pragma once
#include "opencv2/opencv.hpp"
#include "yolov8.h"
#include <string>

class DetectService
{
  public:
    DetectService(const std::string &engine_file_path, const cv::Size &input_size);
    ~DetectService();
    cv::Size get_input_size() { return input_size; }

    /**
     * @brief 进行一次推理
     * @param image 输入图片
     * @param results 预测出的所有bbox
     * @param score_thres 置信度阈值
     * @param iou_thres NMS阈值
     * @param topk 保留k个结果
     * @param num_labels 类别的数量
     * @return 绘制了结果的图片
     */
    cv::Mat predict(const cv::Mat &image, std::vector<Object> &results,
                    const float score_thres = 0.5f, const float iou_thres = 0.65f,
                    const int topk = 30, const int num_labels = 1);

  private:
    std::unique_ptr<YOLOv8> model;
    std::string engine_file_path;
    cv::Size input_size;

    // const std::vector<std::string> CLASS_NAMES = {
    //     "ship", "rubbish", "harbor", "bridge", "buoy", "ball", "platform", "grass"};

    // const std::vector<std::vector<unsigned int>> COLORS = {
    //     {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}, {255, 255, 255}};


    // const std::vector<std::string> CLASS_NAMES = {
    //     "ship", "boat", "rubbish", "harbor", "bridge", "buoy", "ball", "platform", "grass"};

    // const std::vector<std::vector<unsigned int>> COLORS = {
    //     {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}, {255, 255, 255}, {128, 128, 128}};


    const std::vector<std::string> CLASS_NAMES = {
        "ship"};

    const std::vector<std::vector<unsigned int>> COLORS = {
        {0, 0, 255}};
};
