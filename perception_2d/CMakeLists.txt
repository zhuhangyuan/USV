cmake_minimum_required(VERSION 3.8)
project(perception_2d)

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
# find_package(opencv REQUIRED)
find_package(sensor_msgs REQUIRED)
# find_package(cv_bridge REQUIRED)
# find_package(rosidl_default_generators REQUIRED)

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

# # 创建可执行文件
# add_executable(testPublish_node src/testPublish_node.cpp)
# # 添加依赖
# ament_target_dependencies(testPublish_node 
#   rclcpp 
#   sensor_msgs
# )

# # 创建可执行文件
# add_executable(testSubscribe_node src/testSubscribe_node.cpp)
# # 添加依赖
# ament_target_dependencies(testSubscribe_node 
#   rclcpp 
#   sensor_msgs
# )

# # 安装可执行文件
# install(TARGETS 
#   testPublish_node
#   testSubscribe_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# 安装 launch 文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
