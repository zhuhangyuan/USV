#include <iostream>
#include <vector>

#include "bevdet.h"
#include "utils.h"

void Getinfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2]);
    }
    printf("\n");
}

void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name,
               bool with_vel = false)
{
    std::ofstream out_file;
    out_file.open(file_name, std::ios::out);
    if (out_file.is_open())
    {
        for (const auto &box : boxes)
        {
            out_file << box.x << " ";
            out_file << box.y << " ";
            out_file << box.z << " ";
            out_file << box.l << " ";
            out_file << box.w << " ";
            out_file << box.h << " ";
            out_file << box.r << " ";
            if (with_vel)
            {
                out_file << box.vx << " ";
                out_file << box.vy << " ";
            }
            out_file << box.score << " ";
            out_file << box.label << "\n";
        }
    }
    out_file.close();
    return;
};

void printBox(const Box &box)
{
    std::cout << "x: " << box.x << " y: " << box.y << " z: " << box.z << " l: " << box.l
              << " w: " << box.w << " h: " << box.h << " yaw: " << box.r << " score: " << box.score
              << " label: " << box.label << std::endl;
}

void Egobox2Lidarbox(const std::vector<Box> &ego_boxes,
                     std::vector<Box> &lidar_boxes,
                     const Eigen::Quaternion<float> &lidar2ego_rot,
                     const Eigen::Translation3f &lidar2ego_trans)
{
    for (size_t i = 0; i < ego_boxes.size(); i++)
    {
        Box b = ego_boxes[i];
        Eigen::Vector3f center(b.x, b.y, b.z);
        center -= lidar2ego_trans.translation();
        center = lidar2ego_rot.inverse().matrix() * center;
        b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        b.x = center.x();
        b.y = center.y();
        b.z = center.z();
        lidar_boxes.push_back(b);
    }
}

void Egobox2Virtualbox(const std::vector<Box> &ego_boxes,
                       std::vector<Box> &virtual_boxes,
                       const Eigen::Quaternion<float> &virtual2ego_rot,
                       const Eigen::Translation3f &virtual2ego_trans)
{
    for (size_t i = 0; i < ego_boxes.size(); i++)
    {
        Box b = ego_boxes[i];
        Eigen::Vector3f center(b.x, b.y, b.z);
        center -= virtual2ego_trans.translation();
        center = virtual2ego_rot.inverse().matrix() * center;
        b.r -= virtual2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        b.x = center.x();
        b.y = center.y();
        b.z = center.z();
        virtual_boxes.push_back(b);
    }
}

int cv2rawData(cv::Mat img, std::vector<char> &raw_data)
{
    if (img.empty())
    {
        std::cerr << "image is empty. " << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<u_char> raw_data_;
    cv::imencode(".jpg", img, raw_data_);
    raw_data = std::vector<char>(raw_data_.begin(), raw_data_.end());
    return EXIT_SUCCESS;
}

int cv2rawData(std::vector<cv::Mat> &imgs, std::vector<std::vector<char>> &raw_datas)
{
    raw_datas.resize(imgs.size());

    for (size_t i = 0; i < raw_datas.size(); i++)
    {
        if (cv2rawData(imgs[i], raw_datas[i]))
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}