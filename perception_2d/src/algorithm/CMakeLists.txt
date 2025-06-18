set(CUDA_HEAD_PATH "/usr/local/cuda/include")
set(CUDA_LIB_PATH "/usr/local/cuda/lib64")
set(TSRT_HEAD_PATH "/usr/local/TensorRT-8.5.1.7/include")
set(TSRT_LIB_PATH "/usr/local/TensorRT-8.5.1.7/lib")

add_library(algorithm
    SHARED
    yolov8.cpp
)

# 设置头文件路径
target_include_directories(algorithm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TSRT_HEAD_PATH}
    ${CUDA_HEAD_PATH}
)

target_link_directories(algorithm
    PRIVATE
    ${TSRT_LIB_PATH}
    ${CUDA_LIB_PATH}
)
target_link_libraries(algorithm
    PRIVATE
    ${OpenCV_LIBS}

    # c10
    # c10_cuda
    # nvfuser_codegen
    nvinfer
    nvinfer_plugin
)