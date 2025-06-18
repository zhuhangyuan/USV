cmake_minimum_required(VERSION 3.8)
project(demo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
add_compile_options(-W)
add_compile_options(-std=c++17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(robot_msgs REQUIRED) # 自定义的消息包


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

ament_package()

# add nodes
add_executable(Publish_stdmsg_node src/Publish_stdmsg_node.cpp)
ament_target_dependencies(Publish_stdmsg_node rclcpp std_msgs)
install(TARGETS Publish_stdmsg_node DESTINATION lib/${PROJECT_NAME})

add_executable(Subscribe_stdmsg_node src/Subscribe_stdmsg_node.cpp)
ament_target_dependencies(Subscribe_stdmsg_node rclcpp std_msgs)
install(TARGETS Subscribe_stdmsg_node DESTINATION lib/${PROJECT_NAME})

add_executable(Publish_robotmsg_node src/Publish_robotmsg_node.cpp)
ament_target_dependencies(Publish_robotmsg_node rclcpp robot_msgs)
install(TARGETS Publish_robotmsg_node DESTINATION lib/${PROJECT_NAME})

add_executable(Subscribe_robotmsg_node src/Subscribe_robotmsg_node.cpp)
ament_target_dependencies(Subscribe_robotmsg_node rclcpp robot_msgs)
install(TARGETS Subscribe_robotmsg_node DESTINATION lib/${PROJECT_NAME})

# 安装 launch 文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)