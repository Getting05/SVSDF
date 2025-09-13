#!/usr/bin/env python3
"""
修复版本机器人控制器 - 确保位姿正确发布到ROS
"""

import numpy as np
import math
import torch
import rospy
import time
from data_structures import *
from geometry_msgs.msg import Twist

class SLAMRobotController:
    def __init__(self, mobile_base, world, ros_bridge):
        from isaacsim.core.utils.types import ArticulationAction
        from scipy.spatial.transform import Rotation as R
        
        self.ArticulationAction = ArticulationAction
        self.R = R
        self.mobile_base = mobile_base
        self.world = world
        self.ros_bridge = ros_bridge
        
        # 基础控制参数
        self.max_linear_vel = MAX_LINEAR_VELOCITY
        self.max_angular_vel = MAX_ANGULAR_VELOCITY
        self.linear_kp = 3.0
        self.angular_kp = 4.0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.smooth_factor = 0.8
        
        # 可视化器引用
        self.coverage_visualizer = None
        
        # MapEx和SLAM相关
        self.mapex_interface = ros_bridge.get_mapex_interface()
        self.cartographer_interface = ros_bridge.get_cartographer_interface()
        
        # 导航状态
        self.navigation_timeout = 30.0
        self.position_tolerance = 0.3
        self.mapex_active = False
        
        # 关键修复：位姿发布配置
        self.last_pose_publish_time = time.time()
        self.pose_publish_interval = 0.1  # 10Hz发布频率
        
        print("SLAM机器人控制器初始化完成")
        print("位姿发布已启用：10Hz频率")
    
    def set_coverage_visualizer(self, visualizer):
        """设置覆盖可视化器引用"""
        self.coverage_visualizer = visualizer
    
    def move_to_position_robust(self, target_pos: np.ndarray, target_orientation: float = 0.0) -> bool:
        """鲁棒的移动到位置方法"""
        print(f"直接导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        max_steps = MAX_NAVIGATION_STEPS
        step_count = 0
        
        start_pos = self._get_robot_pose()[0]
        start_distance = np.linalg.norm(start_pos[:2] - target_pos[:2])
        
        while step_count < max_steps:
            current_pos, current_yaw = self._get_robot_pose()
            step_count += 1
            
            # 关键修复：每步都发布位姿到ROS
            self._publish_robot_pose_to_ros(current_pos, current_yaw)
            
            # 超级平滑覆盖标记
            self._mark_coverage_ultra_smooth(current_pos)
            
            # 检查是否到达
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            if pos_error < POSITION_TOLERANCE:
                self._send_zero_velocity()
                print(f"  成功到达! 误差: {pos_error:.3f}m, 步数: {step_count}")
                return True
            
            # 计算控制指令
            self._compute_and_send_control(current_pos, current_yaw, target_pos, target_orientation)
            
            # 步进物理仿真
            self.world.step(render=True)
            
            # 周期性进度报告 - 减少输出频率
            if step_count % 5000 == 0:  # 从500改为5000，减少10倍输出
                progress = max(0, (start_distance - pos_error) / start_distance * 100)
                print(f"  导航中... 距离: {pos_error:.3f}m, 进度: {progress:.1f}%, 步数: {step_count}")
        
        # 导航结束
        self._send_zero_velocity()
        final_pos, _ = self._get_robot_pose()
        final_error = np.linalg.norm(final_pos[:2] - target_pos[:2])
        
        # 最后再发布一次位姿
        self._publish_robot_pose_to_ros(final_pos, self._get_robot_pose()[1])
        
        return final_error < POSITION_TOLERANCE * 1.5
    
    def _publish_robot_pose_to_ros(self, position: np.ndarray, yaw: float):
        """关键修复：强制发布Isaac Sim真值位姿到ROS系统"""
        current_time = time.time()
        
        # 控制发布频率
        if current_time - self.last_pose_publish_time < self.pose_publish_interval:
            return
        
        try:
            # 关键修复：强制发布Isaac Sim的真值位置，覆盖SLAM定位
            if hasattr(self.ros_bridge, 'socket_interface') and self.ros_bridge.socket_interface:
                
                # 发布真值位姿数据 - 这是准确的位置
                success = self.ros_bridge.socket_interface.send_robot_pose(position, yaw)
                
                # 同时直接发布到ROS TF系统，强制覆盖Cartographer的定位
                self._force_publish_ground_truth_tf(position, yaw)
                
                if success:
                    self.last_pose_publish_time = current_time
                    
                    # 定期打印调试信息 - 显示真值位置使用状态
                    if int(current_time) % 10 == 0 and current_time - self.last_pose_publish_time < 0.1:
                        print(f"✅ 真值位姿发布: [{position[0]:.3f}, {position[1]:.3f}], yaw: {np.degrees(yaw):.1f}° (Isaac Sim Ground Truth)")
                else:
                    print("❌ 真值位姿发布失败: socket连接问题")
            else:
                print("⚠️ 警告: ROS接口未连接，真值位姿发布跳过")
                
        except Exception as e:
            print(f"发布真值位姿时出错: {e}")
    
    def _force_publish_ground_truth_tf(self, position: np.ndarray, yaw: float):
        """强制发布Isaac Sim真值位置到ROS TF系统"""
        try:
            # 通过ROS桥接接口发布TF，避免直接导入ROS模块
            if hasattr(self.ros_bridge, 'socket_interface') and self.ros_bridge.socket_interface:
                # 检查socket_interface是否有发送方法
                if hasattr(self.ros_bridge.socket_interface, 'send_robot_pose'):
                    # 使用现有的send_robot_pose方法发送真值位置
                    success = self.ros_bridge.socket_interface.send_robot_pose(position, yaw)
                    
                    if success and not hasattr(self, '_tf_publish_logged'):
                        print(f"✅ 真值位置通过ROS桥接发布成功，防止地图漂移")
                        self._tf_publish_logged = True
                else:
                    # 如果没有发送方法，跳过TF发布但不报错
                    if not hasattr(self, '_tf_skip_logged'):
                        print(f"ℹ️ ROS桥接接口暂不支持TF发布，跳过真值TF变换")
                        self._tf_skip_logged = True
            
        except Exception as e:
            # 降低错误输出频率，避免刷屏
            if not hasattr(self, '_tf_error_logged'):
                print(f"⚠️ 真值TF发布跳过: {e}")
                self._tf_error_logged = True
    
    def _compute_and_send_control(self, current_pos: np.ndarray, current_yaw: float, 
                                 target_pos: np.ndarray, target_yaw: float):
        """计算并发送控制指令"""
        # 计算基础控制指令
        direction = target_pos[:2] - current_pos[:2]
        distance = np.linalg.norm(direction)
        
        if distance < 0.0001:
            self._send_zero_velocity()
            return
        
        # 计算目标角度
        direction_norm = direction / distance
        desired_yaw = np.arctan2(direction_norm[1], direction_norm[0])
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)
        
        # 控制策略：先转向，后前进
        if abs(yaw_error) > ANGLE_TOLERANCE:
            linear_vel = 0.0
            angular_vel = np.clip(self.angular_kp * yaw_error, 
                                -self.max_angular_vel, self.max_angular_vel)
        else:
            linear_vel = np.clip(self.linear_kp * distance, 
                               0.0, self.max_linear_vel)
            angular_vel = np.clip(self.angular_kp * yaw_error * 0.5, 
                                -self.max_angular_vel, self.max_angular_vel)
        
        # 平滑速度变化
        linear_vel = self._smooth_velocity(float(linear_vel), self.prev_linear)
        angular_vel = self._smooth_velocity(float(angular_vel), self.prev_angular)
        
        self.prev_linear = linear_vel
        self.prev_angular = angular_vel
        
        # 发送控制指令
        self._send_velocity_command(linear_vel, angular_vel)
    
    def _mark_coverage_ultra_smooth(self, robot_position: np.ndarray):
        """超级平滑覆盖标记"""
        if self.coverage_visualizer and hasattr(self.coverage_visualizer, 'fluent_coverage_visualizer'):
            self.coverage_visualizer.fluent_coverage_visualizer.mark_coverage_realtime(robot_position)
    
    def _smooth_velocity(self, new_vel: float, prev_vel: float) -> float:
        """平滑速度变化"""
        return self.smooth_factor * prev_vel + (1 - self.smooth_factor) * new_vel
    
    def _send_velocity_command(self, linear_vel: float, angular_vel: float):
        """发送速度指令到轮子"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 轮子参数
        wheel_radius = 0.036
        wheel_base = 0.235
        
        # 计算轮子速度
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        # 构建关节速度
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 设置轮子速度
        left_idx = right_idx = None
        if "left_wheel_joint" in self.mobile_base.dof_names:
            left_idx = self.mobile_base.dof_names.index("left_wheel_joint")
            joint_velocities[left_idx] = float(left_wheel_vel)
        
        if "right_wheel_joint" in self.mobile_base.dof_names:
            right_idx = self.mobile_base.dof_names.index("right_wheel_joint")
            joint_velocities[right_idx] = float(right_wheel_vel)
        
        # 应用控制指令
        action = self.ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def _send_zero_velocity(self):
        """发送零速度指令"""
        self._send_velocity_command(0.0, 0.0)
    
    def _normalize_angle(self, angle):
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _get_robot_pose(self):
        """获取机器人位姿"""
        position, orientation = self.mobile_base.get_world_pose()
        position = np.array(position)
        
        # 四元数转欧拉角
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = self.R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        
        return position, yaw
    
    def get_slam_status(self):
        """获取SLAM状态"""
        return {
            'mapex_active': self.mapex_active,
            'exploration_complete': self.mapex_interface.is_exploration_complete(),
            'map_available': self.cartographer_interface.get_slam_map() is not None,
            'robot_pose': self._get_robot_pose()
        }
    
    def emergency_stop(self):
        """紧急停止"""
        self._send_zero_velocity()
        self.mapex_active = False
        print("紧急停止执行")