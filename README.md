# USV

## 开发环境

- OS: Ubuntu22.04
- IDE: VSCode(clangd插件)
- 依赖:
  - ROS2 Humble
  - OpenCV 
  - TensorRT
  - CUDA


## 目录结构
```
USV/
├── robot_msgs: 定义自定义消息类型
├── demo: 节点间通信的例子
├── perception_2d: 2D感知模块
├── perception_3d: 3D感知模块
├── control: 控制模块
├── task_allocation: 任务分配模块

```

## 遇到的问题

1. clangd无法对标准库进行索引
- 解决方法：
在CMakeLists.txt中添加
```
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
```

这样clangd就能对标准库进行索引了