# pragma once
#include "bevdet.h"
#include "cpu_jpegdecoder.h"
#include "opencv2/core/types.hpp"
#include "utils.h"
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "opencv2/opencv.hpp"

void test_infer(std::vector<cv::Mat> imgs);