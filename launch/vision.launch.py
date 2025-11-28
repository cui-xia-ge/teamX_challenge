import launch
import launch_ros
import os
from ament_index_python.packages import get_package_share_directory
def generate_launch_description():
    
    cameraLaunchPath=[get_package_share_directory('camera_sim_pkg'),
                      'launch','camera.launch.py']
    
    act_includeLaunch=launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            cameraLaunchPath)
    )
    actionVisionNode=launch_ros.actions.Node(
        package='player_pkg',
        executable='VisionNode'
    )
    return launch.LaunchDescription([act_includeLaunch, actionVisionNode])