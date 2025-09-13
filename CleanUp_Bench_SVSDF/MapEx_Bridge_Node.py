#!/usr/bin/env python3
"""
修复版MapEx桥接节点 - 自动启动探索并确保正确通信
增加RViz目标点可视化功能
"""

import rospy
import json
import socket
import threading
import time
import subprocess
import os
import signal
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, String, Float32MultiArray, ColorRGBA
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

class MapExBridgeNode:
    """修复版MapEx桥接节点 - 自动探索启动"""
    
    def __init__(self):
        rospy.init_node('mapex_bridge_node', anonymous=True)
        
        # Socket服务器配置 - 作为服务器等待MapEx连接
        self.mapex_socket_host = 'localhost'
        self.mapex_socket_port = 9998  # MapEx专用端口
        self.mapex_server_socket = None  # 服务器socket
        self.mapex_client_socket = None  # 客户端连接
        self.mapex_connected = False
        
        # MapEx进程管理
        self.mapex_process = None
        self.mapex_script_path = "/home/getting/SVSDF/MapEx/scripts/explore.py"
        self.mapex_config_path = "/home/getting/SVSDF/MapEx/configs/base.yaml"
        
        # ROS发布器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.exploration_status_pub = rospy.Publisher('/exploration_status', String, queue_size=10)
        self.exploration_done_pub = rospy.Publisher('/exploration_done', Bool, queue_size=10)
        self.mapex_goal_pub = rospy.Publisher('/mapex/goal', Float32MultiArray, queue_size=10)
        
        # 可视化发布器
        self.goal_marker_pub = rospy.Publisher('/mapex/goal_markers', Marker, queue_size=10)  # 改为Marker
        self.current_goal_pub = rospy.Publisher('/mapex/current_goal', PoseStamped, queue_size=10)
        self.goal_reached_pub = rospy.Publisher('/mapex/goal_reached', PoseStamped, queue_size=10)  # 改为PoseStamped
        
        # ROS订阅器
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        self.lidar_sub = rospy.Subscriber('/robot_lidar_pointcloud', PointCloud2, self.lidar_callback, queue_size=1)
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', Float32MultiArray, self.robot_pose_callback, queue_size=10)
        
        # 状态变量
        self.current_map = None
        self.robot_pose = [0.0, 0.0, 0.0]  # x, y, yaw
        self.exploration_active = False
        self.last_map_update = time.time()
        self.map_received_count = 0
        self.pose_received_count = 0
        
        # 目标点可视化相关变量
        self.current_goal = None  # 当前目标点 [x, y, yaw]
        self.goal_history = []  # 历史目标点
        self.goal_id_counter = 0  # 目标点ID计数器
        self.goal_reached_threshold = 0.3  # 到达目标的距离阈值（米）
        self.last_goal_check_time = time.time()
        self.goal_marker_lifetime = rospy.Duration(30.0)  # 标记生存时间
        
        # 关键修复：自动启动探索的条件
        self.auto_start_enabled = True
        self.auto_start_conditions = {
            'map_received': False,
            'pose_received': False,
            'mapex_connected': False,
            'exploration_started': False
        }
        self.min_map_updates = 3  # 至少收到3次地图更新
        self.min_pose_updates = 5  # 至少收到5次位姿更新
        
        # 线程管理
        self.server_thread = None
        self.mapex_monitor_thread = None
        self.running = True
        
        print("修复版MapEx桥接节点初始化完成")
        print("自动探索启动已启用")
    
    def start_socket_server(self):
        """启动Socket服务器 - 修复连接问题"""
        try:
            # 创建服务器socket
            self.mapex_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.mapex_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # 绑定端口并开始监听
            self.mapex_server_socket.bind((self.mapex_socket_host, self.mapex_socket_port))
            self.mapex_server_socket.listen(1)
            
            print(f"MapEx Socket服务器启动，监听 {self.mapex_socket_host}:{self.mapex_socket_port}")
            
            while self.running:
                try:
                    print("等待MapEx连接...")
                    
                    # 关键修复：使用更短的超时时间，避免长时间阻塞
                    self.mapex_server_socket.settimeout(1.0)  # 1秒超时
                    
                    try:
                        self.mapex_client_socket, addr = self.mapex_server_socket.accept()
                        
                        # 立即设置连接状态
                        self.mapex_connected = True
                        self.auto_start_conditions['mapex_connected'] = True
                        
                        print(f"✓ MapEx已连接: {addr}")
                        print(f"✓ 连接状态已更新: {self.mapex_connected}")
                        
                        # 设置客户端socket为非阻塞模式
                        self.mapex_client_socket.settimeout(0.1)
                        
                        # 发送连接确认
                        confirm_sent = self._send_to_mapex({
                            'type': 'bridge_connected',
                            'timestamp': time.time(),
                            'status': 'connected'
                        })
                        
                        if confirm_sent:
                            print("✓ 连接确认消息已发送")
                        else:
                            print("❌ 连接确认消息发送失败")
                        
                        # 启动通信处理 - 强制保持连接
                        print("开始MapEx通信处理...")
                        
                        # 关键修复：一旦连接就不再等待新连接，专注处理当前连接
                        while self.running and self.mapex_connected and self.mapex_client_socket:
                            try:
                                # 发送数据到MapEx
                                self._send_queued_data_to_mapex()
                                
                                # 接收MapEx的命令
                                self._receive_mapex_commands()
                                
                                # 检查自动启动条件
                                self._check_auto_start_conditions()
                                
                                # 强制发送探索命令给已连接的MapEx
                                self._force_exploration_if_needed()
                                
                                # 关键修复：增加适当延时，避免CPU占用过高
                                time.sleep(0.2)  # 5Hz，减缓通信频率
                                
                            except Exception as comm_error:
                                print(f"MapEx通信处理错误: {comm_error}")
                                # 不要因为小错误就断开连接，稍作延时继续
                                time.sleep(0.5)
                                continue
                        
                        print("MapEx通信循环结束，连接将关闭")
                        
                    except socket.timeout:
                        # 接受连接超时，继续循环
                        continue
                    
                except Exception as e:
                    print(f"接受连接时出错: {e}")
                    self.mapex_connected = False
                    self.auto_start_conditions['mapex_connected'] = False
                    if self.mapex_client_socket:
                        self.mapex_client_socket.close()
                        self.mapex_client_socket = None
                
                if not self.running:
                    break
                    
                if self.mapex_connected:
                    print("MapEx连接断开，等待重连...")
                    self.mapex_connected = False
                    self.auto_start_conditions['mapex_connected'] = False
                
                time.sleep(0.5)  # 短暂等待后继续监听
                
        except Exception as e:
            print(f"Socket服务器错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_sockets()
    
    def _force_exploration_if_needed(self):
        """关键修复：强制触发探索如果MapEx空闲太久"""
        current_time = time.time()
        
        # 如果MapEx已连接但长时间没有速度命令，强制触发探索
        if (self.mapex_connected and 
            self.auto_start_conditions['exploration_started'] and
            not hasattr(self, 'last_velocity_command_time')):
            
            # 初始化时间戳
            self.last_velocity_command_time = current_time
            self.idle_start_time = current_time
        
        # 检查是否长时间空闲（超过10秒无速度命令）
        if (hasattr(self, 'last_velocity_command_time') and 
            current_time - self.last_velocity_command_time > 10.0):
            
            if not hasattr(self, 'idle_start_time'):
                self.idle_start_time = current_time
            
            idle_duration = current_time - self.idle_start_time
            
            # 每30秒强制发送一次探索命令
            if idle_duration > 30.0 and int(idle_duration) % 30 == 0:
                print(f"MapEx空闲{idle_duration:.0f}秒，强制触发探索...")
                
                # 发送强制探索命令
                self._send_to_mapex({
                    'type': 'force_exploration',
                    'timestamp': current_time,
                    'data': {
                        'reason': 'idle_timeout',
                        'current_pose': self.robot_pose,
                        'map_info': {
                            'width': self.current_map['width'] if self.current_map else 0,
                            'height': self.current_map['height'] if self.current_map else 0
                        }
                    }
                })
                
                # 重置空闲时间
                self.idle_start_time = current_time
    
    def _check_auto_start_conditions(self):
        """关键修复：检查并触发自动启动探索"""
        if not self.auto_start_enabled or self.auto_start_conditions['exploration_started']:
            return
        
        # 检查所有启动条件
        conditions_met = (
            self.auto_start_conditions['map_received'] and
            self.auto_start_conditions['pose_received'] and
            self.auto_start_conditions['mapex_connected'] and
            self.map_received_count >= self.min_map_updates and
            self.pose_received_count >= self.min_pose_updates
        )
        
        if conditions_met:
            print("自动启动条件满足，开始MapEx探索...")
            print(f"  地图更新: {self.map_received_count}/{self.min_map_updates}")
            print(f"  位姿更新: {self.pose_received_count}/{self.min_pose_updates}")
            print(f"  MapEx连接: {self.auto_start_conditions['mapex_connected']}")
            
            # 发送详细的探索开始命令到MapEx
            exploration_command = {
                'type': 'start_exploration',
                'timestamp': time.time(),
                'data': {
                    'mode': 'autonomous_slam',
                    'initial_pose': self.robot_pose,
                    'map_info': {
                        'width': self.current_map['width'] if self.current_map else 0,
                        'height': self.current_map['height'] if self.current_map else 0,
                        'resolution': self.current_map['resolution'] if self.current_map else 0.05
                    },
                    'exploration_params': {
                        'max_linear_velocity': 0.3,
                        'max_angular_velocity': 2.5,  # 大幅提高角速度从1.0到2.5
                        'exploration_radius': 5.0,
                        'frontier_threshold': 0.1,
                        'use_ground_truth_pose': True,  # 关键修复：指示MapEx使用真值位置
                        'prevent_drift': True,  # 启用防漂移模式
                        'coordinate_source': 'isaac_sim_ground_truth'  # 标明坐标来源
                    }
                }
            }
            
            success = self._send_to_mapex(exploration_command)
            
            if success:
                # 发布探索状态
                self.publish_exploration_status("EXPLORATION_AUTO_STARTED")
                
                # 标记探索已启动
                self.auto_start_conditions['exploration_started'] = True
                self.exploration_active = True
                
                print("MapEx自动探索已启动!")
                
                # 初始化速度命令监控
                self.last_velocity_command_time = time.time()
            else:
                print("发送探索命令失败，MapEx连接异常")
    
    def _send_to_mapex(self, message):
        """发送消息到MapEx - 使用长度头协议"""
        if not self.mapex_connected or not self.mapex_client_socket:
            return False
        
        try:
            msg_type = message.get('type', 'unknown')
            data = json.dumps(message).encode('utf-8')
            data_size = len(data)
            
            print(f"📤 发送消息类型: {msg_type}, JSON大小: {data_size} 字节")
            
            # 关键修复：发送长度头 + 数据
            # 格式：4字节长度头 + JSON数据
            length_header = data_size.to_bytes(4, byteorder='big')
            
            self.mapex_client_socket.settimeout(5.0)  # 增加超时时间
            
            # 先发送长度头
            self.mapex_client_socket.sendall(length_header)
            # 再发送完整数据
            self.mapex_client_socket.sendall(data)
            
            print(f"✅ 消息发送成功: {msg_type} (头部4字节 + 数据{data_size}字节)")
            return True
            
        except Exception as e:
            print(f"❌ 发送失败: {msg_type}, 错误: {e}")
            self.mapex_connected = False
            return False

    def _receive_mapex_commands(self):
        """接收MapEx的命令 - 修复超时问题"""
        if not self.mapex_connected or not self.mapex_client_socket:
            return
        
        try:
            # 关键修复：使用更长的接收超时时间
            self.mapex_client_socket.settimeout(1.0)  # 1秒超时
            data = self.mapex_client_socket.recv(4096)
            
            if data:
                # 处理可能包含多个JSON对象的数据
                data_str = data.decode('utf-8')
                
                # 将接收到的数据添加到缓冲区
                if not hasattr(self, 'message_buffer'):
                    self.message_buffer = ""
                
                self.message_buffer += data_str
                
                # 按行分割处理多个JSON消息
                while '\n' in self.message_buffer:
                    line, self.message_buffer = self.message_buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        try:
                            message = json.loads(line)
                            self._handle_mapex_command(message)
                        except json.JSONDecodeError as e:
                            print(f"MapEx JSON解析错误: {e}")
                            continue
                
        except socket.timeout:
            pass  # 正常超时，继续
        except ConnectionResetError:
            print("MapEx连接被重置")
            self.mapex_connected = False
            self.auto_start_conditions['mapex_connected'] = False
        except Exception as e:
            print(f"接收MapEx命令时出错: {e}")
            # 不要因为小错误就断开连接
            pass
    
    def _handle_mapex_command(self, command):
        """处理来自MapEx的命令 - 增强版本"""
        cmd_type = command.get('type')
        
        if cmd_type == 'velocity_command':
            # 发布速度命令
            data = command.get('data', {})
            twist = Twist()
            twist.linear.x = data.get('linear_x', 0.0)
            twist.angular.z = data.get('angular_z', 0.0)
            self.cmd_vel_pub.publish(twist)
            
            # 关键修复：更新速度命令时间戳
            self.last_velocity_command_time = time.time()
            
            # 强制输出所有速度命令（包括零速度）- 降低频率，渐进式角速度标识
            current_time = time.time()
            if not hasattr(self, 'last_velocity_debug_time') or current_time - self.last_velocity_debug_time > 2.0:
                # 关键修复：根据角速度大小分级显示
                abs_angular = abs(twist.angular.z)
                if abs_angular > 1.5:
                    print(f"🚀 MapEx高角速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f} (⚡快速转弯)")
                elif abs_angular > 0.8:
                    print(f"🚀 MapEx中角速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f} (🔄中速转弯)")
                elif abs_angular > 0.1:
                    print(f"🚀 MapEx低角速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f} (🎯微调)")
                else:
                    print(f"🚀 MapEx速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f}")
                self.last_velocity_debug_time = current_time
            
        elif cmd_type == 'exploration_status':
            # 发布探索状态
            status = command.get('data', 'UNKNOWN')
            self.publish_exploration_status(status)
            
            if status == 'EXPLORATION_COMPLETED':
                # 探索完成
                self.exploration_active = False
                done_msg = Bool()
                done_msg.data = True
                self.exploration_done_pub.publish(done_msg)
                print("MapEx探索完成!")
            elif status == 'EXPLORATION_STARTED':
                print("MapEx确认探索已开始")
                self.last_velocity_command_time = time.time()
            
        elif cmd_type == 'request_goal':
            # MapEx请求新的探索目标
            data = command.get('data', {})
            goal_msg = Float32MultiArray()
            goal_msg.data = [data.get('x', 0.0), data.get('y', 0.0), data.get('yaw', 0.0)]
            self.mapex_goal_pub.publish(goal_msg)
            
            # 更新当前目标并可视化
            new_goal = [data.get('x', 0.0), data.get('y', 0.0), data.get('yaw', 0.0)]
            self._update_current_goal(new_goal)
            print(f"MapEx请求目标: [{new_goal[0]:.2f}, {new_goal[1]:.2f}]")
            
        elif cmd_type == 'heartbeat':
            # 心跳响应
            self._send_to_mapex({
                'type': 'heartbeat_response',
                'timestamp': time.time()
            })
            
        elif cmd_type == 'frontier_info':
            # 前沿信息（调试用）
            data = command.get('data', {})
            frontier_count = data.get('frontier_count', 0)
            print(f"MapEx前沿检测: 发现{frontier_count}个前沿区域")
        
        elif cmd_type == 'new_goal':
            # MapEx发送新的探索目标
            data = command.get('data', {})
            new_goal = [data.get('x', 0.0), data.get('y', 0.0), data.get('yaw', 0.0)]
            self._update_current_goal(new_goal)
            print(f"📍 MapEx设定新目标: [{new_goal[0]:.2f}, {new_goal[1]:.2f}], yaw={np.degrees(new_goal[2]):.1f}°")
        
        elif cmd_type == 'goal_reached':
            # MapEx报告目标已到达
            if self.current_goal:
                self._mark_goal_as_reached()
                print(f"🎯 目标已到达: [{self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}]")
        
        else:
            print(f"⚠️ 未知MapEx命令类型: {cmd_type}")
    
    def _update_current_goal(self, new_goal):
        """更新当前目标并发布RViz可视化标记"""
        if not new_goal or len(new_goal) < 2:
            print("⚠️ 无效的目标坐标")
            return
        
        # 🔧 修复：验证目标坐标合理性
        x, y = new_goal[0], new_goal[1]
        
        # 检查坐标是否在合理范围内（±20米）
        if abs(x) > 20.0 or abs(y) > 20.0:
            print(f"⚠️ 目标坐标超出合理范围: ({x:.2f}, {y:.2f}), 可能存在坐标转换错误")
            # 不直接返回，仍然发布，但给出警告
        
        # 检查坐标是否为NaN或无穷大
        if not (np.isfinite(x) and np.isfinite(y)):
            print(f"❌ 目标坐标包含无效值: ({x}, {y})")
            return
        
        # 🔧 修复：先清除旧目标标记
        if self.current_goal is not None:
            print(f"🧹 清除旧目标: ID={self.goal_id_counter}")
            self._clear_goal_markers()
            
            # 保存旧目标到历史
            self.goal_history.append({
                'goal': self.current_goal.copy(),
                'timestamp': time.time(),
                'status': 'replaced'
            })
        
        # 设置新目标并递增ID
        self.current_goal = new_goal
        self.goal_id_counter += 1
        
        # 发布新目标的可视化
        self._publish_current_goal(new_goal)
        self._publish_goal_marker(new_goal)
        
        print(f"✅ 目标已更新: ID={self.goal_id_counter}, pos=[{new_goal[0]:.2f}, {new_goal[1]:.2f}]")
    
    def _mark_goal_as_reached(self):
        """标记目标为已到达并清除可视化"""
        if not self.current_goal:
            return
        
        # 添加到历史记录
        self.goal_history.append({
            'goal': self.current_goal.copy(),
            'timestamp': time.time(),
            'status': 'reached'
        })
        
        # 发布目标到达消息
        self._publish_goal_reached(self.current_goal)
        
        # 清除当前目标
        self.current_goal = None
        
        # 清除RViz中的目标标记
        self._clear_goal_markers()
        
        print(f"🎯 目标到达确认，标记已清除")
    
    def _publish_current_goal(self, goal):
        """发布当前目标位置"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            
            pose_msg.pose.position.x = float(goal[0])
            pose_msg.pose.position.y = float(goal[1])
            pose_msg.pose.position.z = 0.0
            
            # 设置朝向（如果提供了yaw角）
            if len(goal) > 2:
                yaw = goal[2]
                pose_msg.pose.orientation.z = np.sin(yaw / 2.0)
                pose_msg.pose.orientation.w = np.cos(yaw / 2.0)
            else:
                pose_msg.pose.orientation.w = 1.0
            
            self.current_goal_pub.publish(pose_msg)
            
            print(f"✅ 当前目标位置已发布: [{goal[0]:.2f}, {goal[1]:.2f}]")
            
        except Exception as e:
            print(f"⚠️ 发布当前目标失败: {e}")
    
    def _publish_goal_marker(self, goal):
        """发布目标可视化标记"""
        try:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "mapex_goals"
            marker.id = self.goal_id_counter
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # 设置位置
            marker.pose.position.x = float(goal[0])
            marker.pose.position.y = float(goal[1])
            marker.pose.position.z = 0.2  # 稍微抬高一点
            
            # 设置朝向
            if len(goal) > 2:
                yaw = goal[2]
                marker.pose.orientation.z = np.sin(yaw / 2.0)
                marker.pose.orientation.w = np.cos(yaw / 2.0)
            else:
                marker.pose.orientation.w = 1.0
            
            # 设置大小
            marker.scale.x = 0.5  # 箭头长度
            marker.scale.y = 0.1  # 箭头宽度
            marker.scale.z = 0.1  # 箭头高度
            
            # 设置颜色 - 亮绿色
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            # 设置持续时间
            marker.lifetime = rospy.Duration(0)  # 永久显示，直到手动删除
            
            self.goal_marker_pub.publish(marker)
            
            # 同时发布文字标签
            self._publish_goal_text(goal, self.goal_id_counter)
            
            print(f"✅ 目标标记已发布: 箭头 + 文字, ID={self.goal_id_counter}")
            
        except Exception as e:
            print(f"⚠️ 发布目标标记失败: {e}")
    
    def _publish_goal_text(self, goal, goal_id):
        """发布目标文字标签"""
        try:
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "mapex_goal_text"
            text_marker.id = goal_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # 文字位置（在目标点上方）
            text_marker.pose.position.x = float(goal[0])
            text_marker.pose.position.y = float(goal[1])
            text_marker.pose.position.z = 0.5
            text_marker.pose.orientation.w = 1.0
            
            # 文字内容
            text_marker.text = f"Goal-{goal_id}\n({goal[0]:.1f}, {goal[1]:.1f})"
            
            # 文字大小
            text_marker.scale.z = 0.2
            
            # 文字颜色 - 白色
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            self.goal_marker_pub.publish(text_marker)
            
            print(f"✅ 目标文字标签已发布: Goal-{goal_id}")
            
        except Exception as e:
            print(f"⚠️ 发布目标文字失败: {e}")
    
    def _publish_goal_reached(self, goal):
        """发布目标到达消息"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "map"
            
            pose_msg.pose.position.x = float(goal[0])
            pose_msg.pose.position.y = float(goal[1])
            pose_msg.pose.position.z = 0.0
            
            if len(goal) > 2:
                yaw = goal[2]
                pose_msg.pose.orientation.z = np.sin(yaw / 2.0)
                pose_msg.pose.orientation.w = np.cos(yaw / 2.0)
            else:
                pose_msg.pose.orientation.w = 1.0
            
            self.goal_reached_pub.publish(pose_msg)
            print(f"✅ 目标到达消息已发布: [{goal[0]:.2f}, {goal[1]:.2f}]")
            
        except Exception as e:
            print(f"⚠️ 发布目标到达消息失败: {e}")
    
    def _clear_goal_markers(self):
        """清除所有目标标记"""
        try:
            # 清除当前目标的箭头标记
            clear_marker = Marker()
            clear_marker.header.frame_id = "map"
            clear_marker.header.stamp = rospy.Time.now()
            clear_marker.ns = "mapex_goals"
            clear_marker.id = self.goal_id_counter
            clear_marker.action = Marker.DELETE
            self.goal_marker_pub.publish(clear_marker)
            
            # 清除当前目标的文字标记
            clear_text = Marker()
            clear_text.header.frame_id = "map"
            clear_text.header.stamp = rospy.Time.now()
            clear_text.ns = "mapex_goal_text"
            clear_text.id = self.goal_id_counter
            clear_text.action = Marker.DELETE
            self.goal_marker_pub.publish(clear_text)
            
            # 🔧 修复：清除所有历史标记（防止遗留）
            # 清除最近5个ID的标记，确保没有遗留
            for old_id in range(max(1, self.goal_id_counter - 4), self.goal_id_counter):
                if old_id != self.goal_id_counter:  # 避免重复清除当前ID
                    # 清除旧箭头标记
                    old_marker = Marker()
                    old_marker.header.frame_id = "map"
                    old_marker.header.stamp = rospy.Time.now()
                    old_marker.ns = "mapex_goals"
                    old_marker.id = old_id
                    old_marker.action = Marker.DELETE
                    self.goal_marker_pub.publish(old_marker)
                    
                    # 清除旧文字标记
                    old_text = Marker()
                    old_text.header.frame_id = "map"
                    old_text.header.stamp = rospy.Time.now()
                    old_text.ns = "mapex_goal_text"
                    old_text.id = old_id
                    old_text.action = Marker.DELETE
                    self.goal_marker_pub.publish(old_text)
            
            print(f"✅ 目标标记已清除: 当前ID={self.goal_id_counter} + 历史标记")
            
        except Exception as e:
            print(f"⚠️ 清除目标标记失败: {e}")
    
    def publish_test_goal(self, x=2.0, y=1.5, yaw=0.0):
        """测试函数：发布一个测试目标点"""
        test_goal = [x, y, yaw]
        self._update_current_goal(test_goal)
        print(f"🧪 发布测试目标: [{x}, {y}], yaw={np.degrees(yaw)}°")
    
    def get_goal_history(self):
        """获取目标历史记录"""
        return self.goal_history
    
    def clear_goal_history(self):
        """清除目标历史记录"""
        self.goal_history.clear()
        print("🗑️ 目标历史已清除")
    
    def _publish_test_goal_for_debug(self):
        """调试功能：发布测试目标来验证可视化系统"""
        if not self.mapex_connected:
            return
        
        import random
        
        # 🔧 修复：生成更合理范围的测试目标（室内环境）
        test_x = random.uniform(-5.0, 5.0)   # 限制在±5米范围内
        test_y = random.uniform(-5.0, 5.0)   # 限制在±5米范围内
        test_yaw = random.uniform(-np.pi, np.pi)
        
        test_goal = [test_x, test_y, test_yaw]
        self._update_current_goal(test_goal)
        
        print(f"🧪 [调试] 发布测试目标: [{test_x:.2f}, {test_y:.2f}], yaw={np.degrees(test_yaw):.1f}°")
        print(f"📊 [调试] 当前目标历史数量: {len(self.goal_history)}")
        
        # 模拟3秒后目标到达
        def mark_reached():
            time.sleep(3.0)
            if self.current_goal and self.current_goal == test_goal:
                self._mark_goal_as_reached()
                print(f"🎯 [调试] 测试目标已模拟到达")
        
        # 启动模拟到达线程
        reach_thread = threading.Thread(target=mark_reached)
        reach_thread.daemon = True
        reach_thread.start()

    def _send_queued_data_to_mapex(self):
        """发送队列中的数据到MapEx - 使用真值位置，防止漂移"""
        if not self.mapex_connected:
            return
        
        current_time = time.time()
        
        # 大幅降低地图发送频率
        if (self.current_map and 
            current_time - getattr(self, 'last_map_send_time', 0) > 0.5):  # 改为0.5秒发送一次

            print(f"📤 准备发送地图数据...")
            
            map_message = {
                'type': 'map_update',
                'data': self.current_map,
                'coordinate_frame': 'isaac_sim_ground_truth'  # 标明坐标系
            }
            
            success = self._send_to_mapex(map_message)
            if success:
                self.last_map_send_time = current_time
                print(f"✅ 地图数据发送完成 (真值坐标系)")
            else:
                print(f"❌ 地图数据发送失败")
        
        # 关键修复：发送真值位姿数据，确保MapEx使用精确位置
        if current_time - getattr(self, 'last_pose_send_time', 0) > 0.5:  # 2Hz发送位姿
            pose_message = {
                'type': 'robot_pose',
                'data': {
                    'x': float(self.robot_pose[0]),
                    'y': float(self.robot_pose[1]),
                    'yaw': float(self.robot_pose[2]),
                    'source': 'isaac_sim_ground_truth',  # 标明这是真值位置
                    'prevent_drift': True,  # 启用防漂移标识
                    'coordinate_accuracy': 'sub_millimeter'  # 标明精度级别
                },
                'timestamp': current_time
            }
            success = self._send_to_mapex(pose_message)
            if success:
                self.last_pose_send_time = current_time
    
    def map_callback(self, msg: OccupancyGrid):
        """地图更新回调 - 修复地图数据格式"""
        # 转换地图数据为MapEx可用格式
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        current_time = time.time()
        
        # 关键修复：正确处理地图数据
        map_data_array = np.array(msg.data, dtype=np.int8)
        
        print(f"🔍 原始地图数据: 长度={len(msg.data)}, 预期={width*height}")
        
        # 检查数据长度是否正确
        expected_length = width * height
        if len(msg.data) != expected_length:
            print(f"❌ 地图数据长度不匹配: 收到{len(msg.data)}, 期望{expected_length}")
            return
        
        # 将1D数据重塑为2D，然后再展平（确保格式正确）
        try:
            # Cartographer发送的是行优先的1D数组
            map_data_2d = map_data_array.reshape((height, width))
            
            # 验证重塑是否正确
            print(f"✅ 地图重塑成功: {map_data_2d.shape}")
            
            # 重新展平为列表（行优先）
            map_data_list = map_data_2d.flatten().tolist()
            
            print(f"✅ 地图数据展平: {len(map_data_list)}个元素")
            
        except Exception as e:
            print(f"❌ 地图数据重塑失败: {e}")
            return
        
        # 构建地图消息
        self.current_map = {
            'width': width,
            'height': height,
            'resolution': resolution,
            'origin': origin,
            'data': map_data_list  # 使用正确展平的数据
        }
        
        # 计算地图哈希（用于变化检测）
        map_hash = hash(map_data_array.tobytes())
        
        # 避免发送相同的地图
        if (hasattr(self, 'last_map_hash') and 
            map_hash == self.last_map_hash and 
            current_time - getattr(self, 'last_map_send_time', 0) < 1.0):
            return
        
        self.last_map_update = current_time
        self.last_map_hash = map_hash
        self.map_received_count += 1
        
        # 标记地图接收状态
        if not self.auto_start_conditions['map_received']:
            self.auto_start_conditions['map_received'] = True
            print(f"首次接收地图数据: {width}x{height}, 分辨率: {resolution:.3f}m/cell")
            
            # 数据质量检查
            unknown_count = np.sum(map_data_2d == -1)
            free_count = np.sum((map_data_2d >= 0) & (map_data_2d <= 20))
            occupied_count = np.sum(map_data_2d >= 80)
            total_cells = width * height
            
            print(f"📊 地图统计:")
            print(f"   未知区域: {unknown_count}/{total_cells} ({unknown_count/total_cells*100:.1f}%)")
            print(f"   空闲区域: {free_count}/{total_cells} ({free_count/total_cells*100:.1f}%)")
            print(f"   占用区域: {occupied_count}/{total_cells} ({occupied_count/total_cells*100:.1f}%)")
        
        # 定期状态报告
        if self.map_received_count % 20 == 0:
            unknown_count = np.sum(map_data_2d == -1)
            total_cells = width * height
            known_ratio = (total_cells - unknown_count) / total_cells if total_cells > 0 else 0
            print(f"地图更新计数: {self.map_received_count}, 已知区域: {known_ratio:.1%}")
    
    def robot_pose_callback(self, msg: Float32MultiArray):
        """机器人位姿回调 - 使用Isaac Sim真值位置，避免SLAM漂移"""
        if len(msg.data) >= 3:
            old_pose = self.robot_pose.copy()
            
            # 关键修复：直接使用Isaac Sim真值位置作为机器人位置
            self.robot_pose = [msg.data[0], msg.data[1], msg.data[2]]
            self.pose_received_count += 1
            
            # 标记位姿接收状态
            if not self.auto_start_conditions['pose_received']:
                self.auto_start_conditions['pose_received'] = True
                print(f"✅ 首次收到Isaac Sim真值位姿: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
                print(f"📍 使用真值位置进行MapEx导航，避免SLAM累积误差导致的地图漂移")
            
            # 位姿变化检测和调试输出 - 降低频率
            current_time = time.time()
            if (abs(old_pose[0] - self.robot_pose[0]) > 0.01 or 
                abs(old_pose[1] - self.robot_pose[1]) > 0.01 or 
                abs(old_pose[2] - self.robot_pose[2]) > 0.1):
                
                # 限制输出频率：每3秒最多输出一次位姿变化
                if not hasattr(self, 'last_pose_debug_time') or current_time - self.last_pose_debug_time > 3.0:
                    print(f"🎯 真值位姿变化: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}° (Isaac Ground Truth)")
                    self.last_pose_debug_time = current_time
            
            # 定期状态报告 - 每100次更新报告一次，强调真值位置使用
            if self.pose_received_count % 100 == 0:
                print(f"📊 真值位姿更新计数: {self.pose_received_count}")
                print(f"🎯 当前真值位姿: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
                print(f"🛡️ 防漂移状态: 使用Isaac Sim真值位置，地图坐标精度保证")
    
    def lidar_callback(self, msg: PointCloud2):
        """激光雷达数据回调"""
        # MapEx主要使用地图数据，激光雷达数据可以用于实时避障
        pass
    
    def publish_exploration_status(self, status: str):
        """发布探索状态"""
        msg = String()
        msg.data = status
        self.exploration_status_pub.publish(msg)
        print(f"发布探索状态: {status}")
    
    def _cleanup_sockets(self):
        """清理socket连接"""
        if self.mapex_client_socket:
            try:
                self.mapex_client_socket.close()
            except:
                pass
            self.mapex_client_socket = None
            
        if self.mapex_server_socket:
            try:
                self.mapex_server_socket.close()
            except:
                pass
            self.mapex_server_socket = None
        
        self.mapex_connected = False
        self.auto_start_conditions['mapex_connected'] = False
    
    def print_status_summary(self):
        """打印状态摘要 - 增强版本"""
        print("\n=== MapEx桥接状态摘要 ===")
        print(f"地图接收: {self.auto_start_conditions['map_received']} (计数: {self.map_received_count})")
        print(f"位姿接收: {self.auto_start_conditions['pose_received']} (计数: {self.pose_received_count})")
        print(f"MapEx连接: {self.auto_start_conditions['mapex_connected']}")
        print(f"探索已启动: {self.auto_start_conditions['exploration_started']}")
        print(f"当前位姿: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
        if self.current_map:
            print(f"当前地图: {self.current_map['width']}x{self.current_map['height']}")
        
        # 关键修复：显示速度命令状态
        if hasattr(self, 'last_velocity_command_time'):
            time_since_last_cmd = time.time() - self.last_velocity_command_time
            print(f"最后速度命令: {time_since_last_cmd:.1f}秒前")
        else:
            print(f"最后速度命令: 从未收到")
        
        print("========================\n")
    
    def run(self):
        """运行桥接节点"""
        print("MapEx桥接节点开始运行...")
        
        # 启动Socket服务器线程
        self.server_thread = threading.Thread(target=self.start_socket_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        print("Socket服务器线程已启动")
        
        # 等待其他节点启动
        rospy.sleep(2.0)
        
        # 主循环
        rate = rospy.Rate(10)  # 10Hz
        last_status_time = time.time()
        last_test_goal_time = time.time()
        
        try:
            while not rospy.is_shutdown() and self.running:
                current_time = time.time()
                
                # 关键修复：检查服务器线程状态
                if not self.server_thread.is_alive():
                    print("❌ Socket服务器线程已停止，尝试重启...")
                    self.server_thread = threading.Thread(target=self.start_socket_server)
                    self.server_thread.daemon = True
                    self.server_thread.start()
                
                # 🧪 调试功能：定期发布测试目标验证可视化系统
                if (self.mapex_connected and 
                    current_time - last_test_goal_time > 20.0):  # 每20秒发布一个测试目标
                    self._publish_test_goal_for_debug()
                    last_test_goal_time = current_time
                
                # 定期打印状态摘要
                if current_time - last_status_time > 15.0:
                    self.print_status_summary()
                    last_status_time = current_time
                
                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            print("收到中断信号")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """关闭桥接节点"""
        print("正在关闭MapEx桥接节点...")
        
        self.running = False
        
        # 清理socket连接
        self._cleanup_sockets()
        
        # 等待线程结束
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=3.0)
        
        print("MapEx桥接节点已关闭")

if __name__ == '__main__':
    try:
        bridge = MapExBridgeNode()
        bridge.run()
    except rospy.ROSInterruptException:
        print("MapEx桥接节点被中断")
    except Exception as e:
        print(f"MapEx桥接节点出错: {e}")
        import traceback
        traceback.print_exc()

"""
RViz目标可视化功能说明：

1. 目标发布话题：
   - /mapex/current_goal: 当前活跃的目标点 (PoseStamped)
   - /mapex/goal_markers: 目标的可视化标记 (Marker)
   - /mapex/goal_reached: 已到达的目标点 (PoseStamped)

2. RViz配置：
   在RViz中添加以下显示类型：
   - Marker: 订阅 /mapex/goal_markers 显示目标箭头和文字
   - PoseStamped: 订阅 /mapex/current_goal 显示目标位置

3. 目标生命周期：
   - 新目标: MapEx发送 'new_goal' 命令 → 显示绿色箭头和标签
   - 目标到达: MapEx发送 'goal_reached' 命令 → 清除可视化标记
   - 目标替换: 新目标自动替换旧目标

4. 测试命令：
   在Python中可以调用:
   bridge.publish_test_goal(x=2.0, y=1.5, yaw=0.0)  # 发布测试目标
   bridge.get_goal_history()  # 查看目标历史
   bridge.clear_goal_history()  # 清除历史记录

5. MapEx集成：
   MapEx需要发送以下格式的命令：
   {
     "type": "new_goal",
     "data": {"x": 2.0, "y": 1.5, "yaw": 0.0}
   }
   {
     "type": "goal_reached"
   }
"""