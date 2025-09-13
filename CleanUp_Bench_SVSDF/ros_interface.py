#!/usr/bin/env python3
"""
ROS接口模块 - 通过Socket与ROS桥接节点通信（避免Python版本冲突）
"""

import socket
import json
import threading
import time
import numpy as np
from typing import Optional, Callable, List, Tuple, Dict

class SocketROSInterface:
    """基于Socket的ROS接口 - 与独立ROS桥接节点通信"""
    
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.client_socket = None
        self.connected = False
        
        # 状态变量
        self.current_map = None
        self.exploration_complete = False
        self.mapex_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.robot_pose_tf = None
        
        # 回调函数
        self.map_update_callback = None
        self.exploration_done_callback_func = None
        self.velocity_command_callback = None
        
        # 消息处理线程
        self.receive_thread = None
        self.running = False
        
        print("Socket ROS接口初始化完成")
    
    def connect(self) -> bool:
        """连接到ROS桥接节点"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5.0)  # 5秒超时
            self.client_socket.connect((self.host, self.port))
            
            self.connected = True
            self.running = True
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            print(f"成功连接到ROS桥接节点: {self.host}:{self.port}")
            
            # 发送心跳测试
            self.send_heartbeat()
            return True
            
        except Exception as e:
            print(f"连接ROS桥接节点失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        self.connected = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        print("ROS接口连接已断开")
    
    def _receive_messages(self):
        """接收消息线程"""
        buffer = ""
        
        while self.running and self.connected:
            try:
                if not self.client_socket:
                    break
                
                data = self.client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # 按行分割消息
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            message = json.loads(line.strip())
                            self._handle_message(message)
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"接收消息错误: {e}")
                break
        
        self.connected = False
        print("消息接收线程结束")
    
    def _handle_message(self, message: Dict):
        """处理接收到的消息"""
        msg_type = message.get('type')
        data = message.get('data')
        
        if msg_type == 'map_update':
            self.current_map = data
            if self.map_update_callback:
                self.map_update_callback(data)
                
        elif msg_type == 'velocity_command':
            self.mapex_velocity = data
            if self.velocity_command_callback:
                self.velocity_command_callback(data['linear_x'], data['angular_z'])
                
        elif msg_type == 'exploration_done':
            self.exploration_complete = data
            if self.exploration_done_callback_func:
                self.exploration_done_callback_func(data)
                
        elif msg_type == 'robot_pose_tf':
            self.robot_pose_tf = data
            
        elif msg_type == 'heartbeat_response':
            # 心跳响应
            pass
            
        elif msg_type == 'map_saved':
            print(f"地图保存状态: {data}")
    
    def _send_message(self, message: Dict) -> bool:
        """发送消息到ROS桥接节点"""
        if not self.connected or not self.client_socket:
            return False
        
        try:
            data = json.dumps(message).encode('utf-8')
            self.client_socket.send(data + b'\n')
            return True
        except Exception as e:
            print(f"发送消息失败: {e}")
            self.connected = False
            return False
    
    def set_map_update_callback(self, callback: Callable):
        """设置地图更新回调函数"""
        self.map_update_callback = callback
    
    def set_exploration_done_callback(self, callback: Callable):
        """设置探索完成回调函数"""
        self.exploration_done_callback_func = callback
    
    def set_velocity_command_callback(self, callback: Callable):
        """设置速度命令回调函数"""
        self.velocity_command_callback = callback
    
    def send_robot_pose(self, position: np.ndarray, yaw: float):
        """发送机器人位姿到ROS桥接节点 - 增强真值位置支持"""
        message = {
            'type': 'robot_pose',
            'data': {
                'position': position.tolist(),
                'yaw': float(yaw),
                'source': 'isaac_sim_ground_truth',  # 标明这是真值位置
                'coordinate_accuracy': 'sub_millimeter',  # 标明精度级别
                'prevent_drift': True,  # 启用防漂移标识
                'timestamp': time.time()
            }
        }
        return self._send_message(message)
    
    def send_goal(self, x: float, y: float, yaw: float = 0.0):
        """发送导航目标"""
        message = {
            'type': 'request_goal',
            'data': {
                'x': float(x),
                'y': float(y),
                'yaw': float(yaw)
            }
        }
        return self._send_message(message)
    
    def publish_exploration_status(self, status: str):
        """发布探索状态"""
        message = {
            'type': 'exploration_status',
            'data': status
        }
        return self._send_message(message)
    
    def send_heartbeat(self):
        """发送心跳"""
        message = {
            'type': 'heartbeat',
            'timestamp': time.time()
        }
        return self._send_message(message)
    
    def is_exploration_complete(self) -> bool:
        """检查探索是否完成"""
        return self.exploration_complete
    
    def get_current_map(self) -> Optional[Dict]:
        """获取当前地图"""
        return self.current_map
    
    def get_robot_pose_tf(self) -> Optional[Tuple[List[float], float]]:
        """获取来自TF的机器人位姿"""
        if self.robot_pose_tf:
            return self.robot_pose_tf['position'], self.robot_pose_tf['yaw']
        return None
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected

class CartographerInterface:
    """Cartographer接口 - 通过Socket ROS接口通信"""
    
    def __init__(self, socket_interface: SocketROSInterface):
        self.socket_interface = socket_interface
        self.slam_map = None
        
        print("Cartographer接口初始化完成")
    
    def get_slam_map(self) -> Optional[Dict]:
        """获取SLAM建图结果"""
        return self.socket_interface.get_current_map()
    
    def save_map(self, filename: str) -> bool:
        """保存地图 - 通过ROS桥接节点"""
        message = {
            'type': 'save_map',
            'data': {
                'filename': filename
            }
        }
        return self.socket_interface._send_message(message)

class ROSBridgeManager:
    """ROS桥接管理器 - Socket版本"""
    
    def __init__(self):
        # Socket接口
        self.socket_interface = SocketROSInterface()
        self.cartographer_interface = CartographerInterface(self.socket_interface)
        
        # 连接状态
        self.connection_thread = None
        self.auto_reconnect = True
        
        print("Socket ROS桥接管理器初始化完成")
    
    def start(self):
        """启动ROS接口"""
        self.connection_thread = threading.Thread(target=self._connection_manager)
        self.connection_thread.daemon = True
        self.connection_thread.start()
        print("ROS接口连接管理器已启动")
    
    def stop(self):
        """停止ROS接口"""
        self.auto_reconnect = False
        if self.socket_interface:
            self.socket_interface.disconnect()
        
        if self.connection_thread:
            self.connection_thread.join(timeout=2.0)
        
        print("ROS接口已停止")
    
    def _connection_manager(self):
        """连接管理线程"""
        retry_count = 0
        max_retries = 10
        
        while self.auto_reconnect and retry_count < max_retries:
            if not self.socket_interface.is_connected():
                print(f"尝试连接ROS桥接节点... (尝试 {retry_count + 1}/{max_retries})")
                
                if self.socket_interface.connect():
                    print("ROS桥接连接成功!")
                    retry_count = 0  # 重置重试计数
                    
                    # 保持连接
                    while self.auto_reconnect and self.socket_interface.is_connected():
                        time.sleep(1.0)
                        
                        # 定期发送心跳
                        if retry_count % 10 == 0:
                            self.socket_interface.send_heartbeat()
                        
                        retry_count += 1
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"连接失败，{5}秒后重试...")
                        time.sleep(5.0)
            else:
                time.sleep(1.0)
        
        if retry_count >= max_retries:
            print("达到最大重试次数，停止尝试连接ROS桥接节点")
        
        print("连接管理线程结束")
    
    def get_mapex_interface(self):
        """获取MapEx接口"""
        return self.socket_interface
    
    def get_cartographer_interface(self):
        """获取Cartographer接口"""
        return self.cartographer_interface
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.socket_interface.is_connected()

# 兼容性接口类
class MapExROSInterface:
    """MapEx兼容性接口"""
    
    def __init__(self, socket_interface: SocketROSInterface):
        self.socket_interface = socket_interface
    
    def set_map_update_callback(self, callback: Callable):
        self.socket_interface.set_map_update_callback(callback)
    
    def set_exploration_done_callback(self, callback: Callable):
        self.socket_interface.set_exploration_done_callback(callback)
    
    def set_velocity_command_callback(self, callback: Callable):
        self.socket_interface.set_velocity_command_callback(callback)
    
    def publish_exploration_status(self, status: str):
        return self.socket_interface.publish_exploration_status(status)
    
    def send_goal(self, x: float, y: float, yaw: float = 0.0):
        return self.socket_interface.send_goal(x, y, yaw)
    
    def is_exploration_complete(self) -> bool:
        return self.socket_interface.is_exploration_complete()
    
    def get_current_map(self) -> Optional[Dict]:
        return self.socket_interface.get_current_map()
    
    @property
    def mapex_velocity(self):
        """获取MapEx速度命令"""
        vel = self.socket_interface.mapex_velocity
        
        class VelocityMsg:
            def __init__(self, linear_x, angular_z):
                self.linear = type('linear', (), {'x': linear_x})()
                self.angular = type('angular', (), {'z': angular_z})()
        
        return VelocityMsg(vel['linear_x'], vel['angular_z'])

# 修改ROSBridgeManager以提供兼容的MapExROSInterface
def _get_mapex_interface_original(self):
    return self.socket_interface

def _get_mapex_interface_compatible(self):
    return MapExROSInterface(self.socket_interface)

# 动态替换方法
ROSBridgeManager.get_mapex_interface = _get_mapex_interface_compatible