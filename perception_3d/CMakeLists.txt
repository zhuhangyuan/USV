cmake_minimum_required(VERSION 3.8)
project(perception_3d)

# 全局设置：安装路径、编译命令导出
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 编译选项：C++17标准、警告选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
add_compile_options(-W)
add_compile_options(-std=c++17)

# 查找ROS 2核心依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED) # 传感器消息（如图像）
find_package(cv_bridge REQUIRED) # OpenCV与ROS图像转换
find_package(robot_msgs REQUIRED) # 自定义消息包
find_package(geometry_msgs REQUIRED) # 几何消息（如位姿）

# 查找第三方依赖（OpenCV需显式指定目录）
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# 包含算法模块子目录（生成共享库perception_3d_algorithm）
add_subdirectory(src/algorithm)

# 处理测试（跳过版权和cpplint检查）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE) # 跳过版权检查
  set(ament_cmake_cpplint_FOUND TRUE) # 跳过cpplint（非git仓库）
  ament_lint_auto_find_test_dependencies()
endif()

# 编译测试节点：test_node（示例节点，需根据实际需求调整）
add_executable(test_node src/test_node.cpp)
target_include_directories(test_node PUBLIC include) # 包含自定义头文件
ament_target_dependencies(test_node # 链接ROS依赖
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  robot_msgs
  geometry_msgs
)
target_link_libraries(test_node 3d_algorithm) # 链接算法共享库
install(TARGETS test_node DESTINATION lib/${PROJECT_NAME}) # 安装到ROS库目录

# 安装launch文件（用于启动节点）
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package() # ROS 2包必需声明