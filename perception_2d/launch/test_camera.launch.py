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
            }]
        ),
        Node(
            package='perception_2d',
            executable='detection_node',
            name='detection',
            parameters=[{}]
        )
    ])