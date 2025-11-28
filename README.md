#### 使用说明
将本包替换工作空间Vision_Arena_2025下的示例包src/player_pkg
同级目录下应存在target_model_pkg
##### 本地运行命令:
source install/setup.bash
colcon build
ros2 launch camera_sim_pkg camera.launch.py
ros2 run player_pkg VisionNode
ros2 launch referee_pkg referee_pkg_launch.xml \
    TeamName:="TEAMENAME" \
    StageSelect:=0 \
    ModeSelect:=0
