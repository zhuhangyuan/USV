find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(CUDA_HEAD_PATH "/usr/local/cuda/include")
set(CUDA_LIB_PATH "/usr/local/cuda/lib64")

add_compile_options(-W)
add_compile_options(-std=c++14) # 这里有问题，ros2用的是17，但是bevdet用17编译不了

set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()


set(SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 86 87)

foreach(sm ${SMS})
  set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()

set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -ccbin ${CMAKE_CXX_COMPILER}
)

if(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

# TensorRT路径设置
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64") # orin
  set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
  set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(TENSORRT_INCLUDE_DIRS /usr/local/TensorRT-8.5.1.7/include) # your tensorrt path
  set(TENSORRT_LIBRARY_DIRS /usr/local/TensorRT-8.5.1.7/lib)
endif()

# CUDA路径设置
set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/lib)

# Nvjpeg库设置
find_library(NVJPEG_LIBRARY nvjpeg ${CUDA_LIB_DIRS})

if(NVJPEG_LIBRARY)
  add_definitions(-D__HAVE_NVJPEG__)
  link_libraries(${NVJPEG_LIBRARY})
  message(STATUS ${NVJPEG_LIBRARY})
endif()


# 生成共享库（名称需与父级CMakeLists.txt中target_link_libraries对应）
# 定义算法模块源文件（根据实际文件结构调整）
set(ALGORITHM_SRC

  # C++源文件
  bevdet.cpp
  data.cpp
  nvjpegdecoder.cpp
  cpu_jpegdecoder.cpp
  utils.cpp
  test_inference.cpp

  # CUDA源文件（.cu结尾）
  bevpool.cu
  grid_sampler.cu
  preprocess.cu
  postprocess.cu
  iou3d_nms.cu
)

# 生成共享库（名称需与父级CMakeLists.txt中target_link_libraries对应）
add_library(3d_algorithm SHARED ${ALGORITHM_SRC})

target_include_directories(3d_algorithm
  PUBLIC
  ${CUDA_HEAD_PATH}
  ${TENSORRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}

  ${PROJECT_SOURCE_DIR}/include
)

target_link_directories(3d_algorithm
  PUBLIC
  ${TENSORRT_LIBRARY_DIRS}
  ${CUDA_LIB_PATH}
  ${OpenCV_LIBS}
)

target_link_libraries(3d_algorithm
PRIVATE
    ${OpenCV_LIBS}
    cudart
    nvinfer
    nvinfer_plugin
    yaml-cpp 
    libjpeg.so
)

install(TARGETS 3d_algorithm
  LIBRARY DESTINATION lib/${PROJECT_NAME} # 动态库 (.so)
  ARCHIVE DESTINATION lib/${PROJECT_NAME} # 静态库 (.a)
)
