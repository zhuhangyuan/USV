cmake_minimum_required(VERSION 3.8)
project(perception_2d)

# 全局设置
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
add_compile_options(-W)
add_compile_options(-std=c++17)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(robot_msgs REQUIRED) # 自定义的消息包

add_subdirectory(src/algorithm)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# add nodes
add_executable(camera_node src/camera_node.cpp)
ament_target_dependencies(camera_node 
  rclcpp std_msgs OpenCV sensor_msgs cv_bridge)
install(TARGETS camera_node DESTINATION lib/${PROJECT_NAME})


add_executable(detection_node src/detection_node.cpp)
ament_target_dependencies(detection_node 
  rclcpp std_msgs OpenCV sensor_msgs cv_bridge robot_msgs)
target_link_libraries(detection_node algorithm)
install(TARGETS detection_node DESTINATION lib/${PROJECT_NAME})

# 安装 launch 文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
