# SLAM集成Isaac Sim机器人覆盖系统 - 完整运行指南（Socket版本）

## 系统架构概述
本系统通过独立的ROS桥接节点解决了Python版本冲突问题，实现Isaac Sim + MapEx + Cartographer的完整集成。

### 技术栈
- **Isaac Sim 4.5**: 机器人仿真平台 (Python 3.10)
- **ROS桥接节点**: 独立通信节点 (Python 3.8)
- **Cartographer**: SLAM建图系统 (Python 3.8) 
- **MapEx**: 自主探索系统 (Python 3.8)
- **Socket通信**: 避免Python版本冲突

### 解决方案特点
- **无Python版本冲突**: Isaac Sim使用Socket通信，不直接导入ROS库
- **保持完整功能**: 所有SLAM和MapEx功能完全保留
- **独立进程**: 各组件在独立环境中运行，互不干扰

## 文件结构

```
/home/getting/SVSDF/CleanUp_Bench_SVSDF/
├── main_system.py           # 主系统文件（SLAM版本）
├── ros_bridge_node.py       # 新增：独立ROS桥接节点
├── ros_interface.py         # 修改：Socket版本ROS接口
├── path_planner.py         # SLAM版本路径规划器
├── robot_controller.py     # SLAM版本机器人控制器
├── lidar_avoidance.py      # 激光雷达避障（保持原版本）
├── data_structures.py     # 数据结构（无需修改）
├── visualizer.py          # 可视化器（无需修改）
└── quick_start_check.sh   # 系统检查脚本
```

## 详细运行步骤（6个终端）

### 终端1: 启动ROS核心
```bash
# 激活Cartographer环境
conda activate myenv_py38
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/install_isolated/setup.bash

# 启动roscore
roscore
```

### 终端2: 启动ROS桥接节点（新增）
```bash
# 激活Cartographer环境（与ROS同环境）
conda activate myenv_py38
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/install_isolated/setup.bash

# 启动独立ROS桥接节点
cd /home/getting/SVSDF/CleanUp_Bench_SVSDF/
python3 ros_bridge_node.py
```

**重要**: 这个节点负责Isaac Sim与ROS系统的通信，必须在Isaac Sim启动前运行。

### 终端3: 启动Isaac Sim主系统
```bash
# 进入Isaac Sim目录
cd ~/isaacsim

# 启动主系统（使用Isaac Sim的Python 3.10环境）
./python.sh /home/getting/SVSDF/CleanUp_Bench_SVSDF/main_system.py
```

**等待**: 确保看到"ROS桥接连接成功!"消息后再继续下一步。

### 终端4: 启动Cartographer SLAM
```bash
# 激活Cartographer环境
conda activate myenv_py38
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/install_isolated/setup.bash

# 等待Isaac Sim开始发布激光雷达数据后运行
roslaunch cartographer_ros isaac_sim_2d.launch
```

**等待**: 确保Isaac Sim已开始发布`/robot_lidar_pointcloud`话题。

### 终端5: 启动MapEx探索
```bash
# 激活MapEx环境
conda activate lama
cd ~/MapEx/scripts/

# 等待Cartographer开始建图后运行
python3 slam_mapex_bridge.py
```
### 终端6: 验证连接
conda activate myenv_py38
source /opt/ros/noetic/setup.bash

echo "=== 系统连接验证 ==="

echo "1. 检查Isaac Sim激光雷达数据："
timeout 5s rostopic echo /robot_lidar_pointcloud --noarr | head -3

echo "2. 检查Cartographer地图："
timeout 5s rostopic echo /map --noarr | head -3

echo "3. 检查速度指令："
timeout 5s rostopic echo /cmd_vel --noarr | head -3

echo "4. 节点连接图："
rosnode list | grep -E "(bridge|cartographer|isaac)"

echo "5. 话题频率："
echo "激光雷达频率："
timeout 10s rostopic hz /robot_lidar_pointcloud

echo "地图更新频率："
timeout 10s rostopic hz /map

**等待**: 确保Cartographer已开始发布`/map`话题。

## 启动时序图（更新版）

```
时间轴: 0s -----> 5s -----> 15s -----> 25s -----> 35s -----> 开始探索
         |         |          |          |          |
    [终端1]    [终端2]    [终端3]    [终端4]    [终端5]
    roscore → ROS桥接 → Isaac Sim → Cartographer → MapEx
```

### 详细启动序列

1. **0-5秒**: 启动roscore (终端1)
2. **5-10秒**: 启动ROS桥接节点 (终端2)
   - 等待Isaac Sim连接
   - 监听端口9999
3. **10-20秒**: 启动Isaac Sim主系统 (终端3)
   - Isaac Sim初始化
   - 连接到ROS桥接节点
   - 激光雷达开始发布数据
4. **20-30秒**: 启动Cartographer (终端4)
   - 等待看到激光雷达数据
   - 开始SLAM建图
   - 发布地图到 `/map`
5. **30-40秒**: 启动MapEx (终端5)
   - 等待看到SLAM地图数据
   - 开始自主探索
6. **40秒后**: 系统进入自主探索阶段

## 关键ROS话题和Socket通信

### Socket通信端口
- **端口**: localhost:9999
- **协议**: TCP Socket + JSON消息
- **连接**: Isaac Sim (客户端) ↔ ROS桥接节点 (服务器)

### Isaac Sim ↔ ROS桥接节点消息
```json
// Isaac Sim -> ROS桥接节点
{
  "type": "robot_pose",
  "data": {"position": [x, y, z], "yaw": 0.0}
}

{
  "type": "request_goal", 
  "data": {"x": 1.0, "y": 2.0, "yaw": 0.0}
}

// ROS桥接节点 -> Isaac Sim
{
  "type": "map_update",
  "data": {"width": 400, "height": 400, "resolution": 0.05, "data": [...]}
}

{
  "type": "velocity_command",
  "data": {"linear_x": 0.2, "angular_z": 0.1}
}

{
  "type": "exploration_done",
  "data": true
}
```

### ROS话题（保持不变）
- `/robot_lidar_pointcloud`: Isaac Sim发布激光雷达数据
- `/map`: Cartographer发布SLAM地图
- `/cmd_vel`: MapEx发布运动命令
- `/exploration_status`: 探索状态
- `/exploration_done`: 探索完成信号

## 故障排除（更新版）

### 常见问题1: ROS桥接节点连接失败
**症状**: Isaac Sim显示"连接ROS桥接节点失败"
**解决方案**: 
```bash
# 检查ROS桥接节点是否运行
ps aux | grep ros_bridge_node

# 检查端口是否被占用
netstat -tuln | grep 9999

# 重启ROS桥接节点
cd /home/getting/SVSDF/CleanUp_Bench_SVSDF/
python3 ros_bridge_node.py
```

### 常见问题2: Isaac Sim无法发布激光雷达数据
**症状**: Cartographer启动后看不到激光雷达数据
**解决方案**: 
```bash
# 检查Isaac Sim Action Graph是否运行
# 在Isaac Sim界面中检查LiDAR Action Graph状态

# 检话题发布
rostopic hz /robot_lidar_pointcloud

# 如果没有数据，重启Isaac Sim
```

### 常见问题3: 多个Python环境导致的导入错误
**症状**: 各种模块导入失败
**解决方案**:
```bash
# 确保每个终端使用正确的Python环境

# 终端1,2,4: Python 3.8 (myenv_py38)
conda activate myenv_py38
python --version  # 应该显示 Python 3.8.x

# 终端5: Python 3.8 (lama)  
conda activate lama
python --version  # 应该显示 Python 3.8.x

# 终端3: Isaac Sim Python 3.10
cd ~/isaacsim
./python.sh --version  # 应该显示 Python 3.10.x

# 检查ROS环境变量
echo $ROS_MASTER_URI
echo $ROS_IP
```

### 常见问题4: Socket通信超时
**症状**: Isaac Sim连接后很快断开
**解决方案**:
```bash
# 检查防火墙设置
sudo ufw status

# 检查网络连接
telnet localhost 9999

# 检查ROS桥接节点日志
# 查看终端2的输出信息

# 调整超时设置（如果需要）
# 在ros_interface.py中修改timeout参数
```

## 系统状态监控（更新版）

### 监控脚本
```bash
#!/bin/bash
# 保存为 monitor_slam_system.sh

echo "=== SLAM集成系统状态监控（Socket版本） ==="

echo "1. ROS核心状态:"
if pgrep -f roscore > /dev/null; then
    echo "✓ roscore 运行中"
else
    echo "✗ roscore 未运行"
fi

echo "2. ROS桥接节点状态:"
if pgrep -f ros_bridge_node.py > /dev/null; then
    echo "✓ ROS桥接节点 运行中"
else
    echo "✗ ROS桥接节点 未运行"
fi

echo "3. Isaac Sim状态:"
if pgrep -f isaac-sim > /dev/null; then
    echo "✓ Isaac Sim 运行中"
else
    echo "✗ Isaac Sim 未运行"
fi

echo "4. Socket连接状态:"
if netstat -tuln | grep -q 9999; then
    echo "✓ Socket端口9999 监听中"
else
    echo "✗ Socket端口9999 未监听"
fi

echo "5. ROS话题状态:"
if timeout 2s rostopic list | grep -q robot_lidar_pointcloud; then
    echo "✓ 激光雷达话题 活跃"
    rostopic hz /robot_lidar_pointcloud 2>/dev/null | head -1
else
    echo "✗ 激光雷达话题 未发布"
fi

if timeout 2s rostopic list | grep -q "/map"; then
    echo "✓ SLAM地图话题 活跃"
    rostopic hz /map 2>/dev/null | head -1
else
    echo "✗ SLAM地图话题 未发布"
fi

echo "监控完成"
```

## 性能优化建议（更新版）

### Socket通信优化
```python
# 在ros_interface.py中调整缓冲区大小
self.client_socket.recv(8192)  # 增加缓冲区

# 调整消息发送频率
time.sleep(0.01)  # 100Hz -> 50Hz: time.sleep(0.02)
```

### ROS桥接节点优化
```python
# 在ros_bridge_node.py中调整处理频率
rate = rospy.Rate(30)  # 可以调整为20或15以降低CPU使用率
```

### 内存使用优化
- Isaac Sim: 建议16GB+ RAM
- ROS系统: 建议4GB+ RAM  
- GPU: RTX 2060+ 推荐

## 预期结果（Socket版本）

### 正常运行流程
1. **0-40秒**: 系统启动和初始化（5个进程）
2. **40秒-5分钟**: SLAM探索建图阶段
3. **5分钟后**: 基于已知地图的覆盖任务执行阶段

### 成功指标
- ROS桥接节点Socket连接成功
- 激光雷达数据发布频率 > 10Hz
- SLAM地图更新频率 > 1Hz  
- MapEx命令发布频率 > 5Hz
- Socket消息传输延迟 < 10ms
- 探索覆盖率 > 80%
- 任务完成率 > 90%

## 调试命令（更新版）

### Socket调试
```bash
# 测试Socket连接
telnet localhost 9999

# 查看Socket连接状态
ss -tuln | grep 9999

# 监控Socket流量（如果安装了tcpdump）
sudo tcpdump -i lo -A port 9999
```

### ROS调试
```bash
# ROS网络诊断
roswtf

# 查看节点图
rosrun rqt_graph rqt_graph

# 监控消息频率
rostopic hz /robot_lidar_pointcloud
rostopic hz /map
rostopic hz /cmd_vel
```

## 重要提醒

1. **启动顺序严格**: 必须严格按照5个终端的顺序启动
2. **环境变量**: 确保每个终端正确设置了conda环境和ROS环境
3. **Socket端口**: 确保9999端口未被其他程序占用
4. **网络配置**: 如果使用多机器，需要修改socket配置中的host地址
5. **日志监控**: 密切关注ROS桥接节点的输出信息
6. **重启顺序**: 如果系统出错，按相反顺序关闭，然后重新启动

运行出现问题时，请首先检查ROS桥接节点（终端2）的输出信息，它是整个系统通信的关键。