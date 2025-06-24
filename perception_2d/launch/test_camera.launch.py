from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_2d',
            executable='camera_node',
            name='camera',
            parameters=[{
                'video_path': '/home/jiahan/Desktop/learn_ros/USV/imgs/ship.mp4',
            }],
            arguments=['--ros-args', '--log-level', 'INFO'],
        ),
        Node(
            package='perception_2d',
            executable='detection_node',
            name='detection',
            parameters=[{
                'engine_path': '/home/jiahan/Desktop/yolo-utils/runs/detect/train11/weights/best.engine',
                # 'engine_path': '/home/jiahan/Desktop/learn_ros/USV/perception_2d/models/yolo11n.engine',
                'input_size': 1024
            }]
        ),

        Node(
            package='demo',
            executable='Subscribe_test_node',
            name='test_subscriber',
            output='screen',
            # 可以添加额外的参数
            # parameters=[{'param_name': 'value'}]
        )
    ])