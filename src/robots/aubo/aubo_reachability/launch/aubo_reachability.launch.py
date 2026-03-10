from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "aubo_type",
            default_value="aubo_i5",
            description="Aubo robot type, e.g. aubo_i5",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="aubo_ros2.xacro",
            description="URDF/XACRO file under aubo_description/urdf/xacro/inc",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="Use fake hardware interfaces in ros2_control tags",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="None",
            description="Robot IP (unused in viewer, passed to xacro)",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz for visualization",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "group_name",
            default_value="manipulator",
            description="MoveIt group name",
        )
    )

    aubo_type = LaunchConfiguration("aubo_type")
    description_file = LaunchConfiguration("description_file")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    robot_ip = LaunchConfiguration("robot_ip")
    launch_rviz = LaunchConfiguration("launch_rviz")
    group_name = LaunchConfiguration("group_name")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("aubo_description"),
                    "urdf",
                    "xacro",
                    "inc",
                    description_file,
                ]
            ),
            " ",
            "aubo_type:=",
            aubo_type,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "robot_ip:=",
            robot_ip,
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    reachability_node = Node(
        package="aubo_reachability",
        executable="reachability_server.py",
        output="screen",
        parameters=[robot_description, {"group_name": group_name}],
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("aubo_description"), "rviz", "view_robot.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription(declared_arguments + [reachability_node, rviz_node])
