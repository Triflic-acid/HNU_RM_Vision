#!/usr/bin/env bash
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash
source simulation/install_interfaces/setup.bash
export ROS_DOMAIN_ID=89

echo "== 1. 二进制时间（应是刚才编译的）"
ls -l --time-style=+%F_%T build/training

echo "== 2. 进程数（应为 1，0 表示没跑，>1 表示有残留）"
pgrep -af build/training || echo "  （未运行）"

echo "== 3. 节点"
ros2 node list

echo "== 4. 三个关键 Topic"
ros2 topic list -t | grep -E "image_raw|vision_receive|vision_send" || echo "  ❌ 缺失"

echo "== 5. 图像频率（应 ~35 Hz）"
timeout 4 ros2 topic hz /image_raw 2>/dev/null | head -2

echo "== 6. 指令内容"
timeout 4 ros2 topic echo /vision_send_data --once