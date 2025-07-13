from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_2d',
            executable='model_test_node',
            name='detection',
            parameters=[{
                'engine_path': '/home/jiahan/Desktop/yolo-utils/runs/detect/train12/weights/best.engine',
                # 'engine_path': '/home/jiahan/Desktop/learn_ros/USV/perception_2d/models/yolo11n.engine',
                'input_size': 1024
            }]
        )
    ])