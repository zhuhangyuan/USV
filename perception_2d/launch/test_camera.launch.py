from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_2d',
            executable='camera_node',
            name='camera',
            parameters=[{
                'video_path': '/home/jiahan/Desktop/learn_ros/USV/imgs/person.mp4',
            }],
            arguments=['--ros-args', '--log-level', 'INFO'],
        ),
        Node(
            package='perception_2d',
            executable='detection_node',
            name='detection',
            parameters=[{
                'engine_path': '/home/jiahan/Desktop/learn_ros/USV/perception_2d/models/yolo11n.engine',
                'input_size': 1024
            }]
        )
    ])