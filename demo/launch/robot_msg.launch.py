from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='demo',
            executable='Publish_robotmsg_node',
            name='test_publisher',
            output='screen',
            # 可以添加额外的参数
            # parameters=[{'param_name': 'value'}]
        ),
        

        Node(
            package='demo',
            executable='Subscribe_robotmsg_node',
            name='test_subscriber',
            output='screen',
            # 可以添加额外的参数
            # parameters=[{'param_name': 'value'}]
        )
    ])