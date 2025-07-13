#include "test_inference.h"

void test_infer(std::vector<cv::Mat> imgs)
{
    std::string config_file("perception_3d/cfgs/test_configure.yaml");
    YAML::Node config = YAML::LoadFile(config_file);

    // 从配置文件中读取基础参数
    size_t img_N = config["N"].as<size_t>();          // 输入图像数量
    int img_w = config["W"].as<int>();                // 图像宽度（预处理后）
    int img_h = config["H"].as<int>();                // 图像高度（预处理后）
    std::string model_config = config["ModelConfig"].as<std::string>(); // 模型配置文件路径
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>(); // 图像阶段TensorRT引擎路径
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>(); // BEV阶段TensorRT引擎路径
    YAML::Node camconfig = YAML::LoadFile(config["CamConfig"].as<std::string>()); // 加载相机参数配置文件
    std::string lidarbox_fileName = config["OutputLidarBox"].as<std::string>(); // 激光雷达坐标系检测框输出路径
    YAML::Node sample = config["sample"]; // 样本数据配置节点（包含测试图像路径）

    // 从样本配置中收集输入图像路径和对应相机名称
    std::vector<std::string> imgs_file;  // 存储测试图像文件路径
    std::vector<std::string> cam_names;  // 存储对应相机名称（如"CAM_FRONT_LEFT"）
    for (auto file : sample)
    {
        imgs_file.push_back(file.second.as<std::string>()); // 图像路径（如"../sample0/imgs/CAM_FRONT.jpg"）
        cam_names.push_back(file.first.as<std::string>());  // 相机名称（与路径一一对应）
    }

    camsData sampleData;
    sampleData.param = camParams(camconfig, img_N, cam_names); // 从相机配置文件中加载参数

    // 初始化BEVDet推理对象
    BEVDet bevdet(
        model_config,          // 模型配置文件路径
        img_N,                 // 输入图像数量
        sampleData.param.cams_intrin,    // 相机内参（K矩阵）
        sampleData.param.cams2ego_rot,   // 相机到ego坐标系的旋转矩阵
        sampleData.param.cams2ego_trans, // 相机到ego坐标系的平移向量
        imgstage_file,         // 图像阶段引擎路径
        bevstage_file          // BEV阶段引擎路径
    );

    // 读取并预处理输入图像（使用OpenCV）
    // std::vector<cv::Mat> imgs;            // 存储解码后的OpenCV图像矩阵（BGR格式）
    std::vector<std::vector<char>> img_raw_datas; // 存储原始图像二进制数据（JPEG编码）

    // for (auto file : imgs_file)
    // {
    //     imgs.emplace_back(cv::imread(file, cv::IMREAD_COLOR)); // 读取图像
    // }

    // 调整图像尺寸到配置指定的宽高（img_w x img_h）
    for (auto &img : imgs)
    {
        cv::resize(img, img, cv::Size(img_w, img_h));
    }       

    // 将OpenCV图像转换为原始二进制数据（JPEG编码）
    cv2rawData(imgs, img_raw_datas);

    // 分配GPU显存用于存储图像数据
    uchar *imgs_dev = nullptr; // GPU显存指针（存储预处理后的图像数据）
    CHECK_CUDA(cudaMalloc((void **)&imgs_dev, img_N * 3 * img_w * img_h * sizeof(uchar))); // 分配显存（3通道，uchar类型）

    // CPU端解码JPEG图像，并转换为BGRCHW格式后拷贝到GPU显存
    decode_cpu(img_raw_datas, imgs_dev, img_w, img_h); // 解码并传输数据到GPU
    sampleData.imgs_dev = imgs_dev; // 将GPU图像数据指针绑定到样本数据对象

    // 执行BEVDet推理
    std::vector<Box> ego_boxes; // 存储ego坐标系下的检测框（x,y,z,长宽高,旋转角,类别等）
    float time = 0.f;           // 记录推理耗时（毫秒）
    bevdet.DoInfer(sampleData, ego_boxes, time); // 核心推理接口

    Boxes2Txt(ego_boxes, "../sample0/sample0_egobox.txt", false); // 保存ego坐标系结果
}