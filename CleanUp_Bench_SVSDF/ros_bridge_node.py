#!/usr/bin/env python3
"""
修复版本ROS桥接节点 - 正确的TF发布和位姿处理
"""

import rospy
import json
import socket
import threading
import time
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Point, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, Bool, Float32MultiArray
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf2_geometry_msgs
from tf_conversions import transformations

class IsaacSimROSBridge:
    """修复版本的Isaac Sim ROS桥接节点"""
    
    def __init__(self):
        rospy.init_node('isaac_sim_ros_bridge', anonymous=True)
        
        # Socket服务器配置
        self.socket_host = 'localhost'
        self.socket_port = 9999
        self.server_socket = None
        self.client_socket = None
        
        # ROS发布器
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.exploration_status_pub = rospy.Publisher('/exploration_status', String, queue_size=10)
        self.isaac_status_pub = rospy.Publisher('/isaac_sim_status', String, queue_size=10)
        
        # 关键修复：添加机器人位姿发布器
        self.robot_pose_pub = rospy.Publisher('/robot_pose', Float32MultiArray, queue_size=10)
        
        # ROS订阅器
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        self.mapex_cmd_sub = rospy.Subscriber('/cmd_vel', Twist, self.mapex_cmd_callback, queue_size=10)
        self.exploration_done_sub = rospy.Subscriber('/exploration_done', Bool, self.exploration_done_callback, queue_size=1)
        
        # 关键修复：TF广播器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 状态变量
        self.current_map = None
        self.exploration_complete = False
        self.mapex_velocity = Twist()
        
        # 关键修复：机器人位姿状态
        self.robot_pose = None
        self.last_pose_time = rospy.Time.now()
        self.pose_timeout = 2.0  # 位姿超时阈值
        
        # 消息队列
        self.message_queue = []
        self.queue_lock = threading.Lock()
        
        print("修复版Isaac Sim ROS桥接节点初始化完成")
        print("TF广播器已启用")
        print("机器人位姿发布器已启用")
        
    def start_socket_server(self):
        """启动socket服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.socket_host, self.socket_port))
            self.server_socket.listen(1)
            
            print(f"ROS桥接服务器启动，监听 {self.socket_host}:{self.socket_port}")
            
            while not rospy.is_shutdown():
                print("等待Isaac Sim连接...")
                self.client_socket, addr = self.server_socket.accept()
                print(f"Isaac Sim已连接: {addr}")
                
                # 启动通信线程
                comm_thread = threading.Thread(target=self.handle_client_communication)
                comm_thread.daemon = True
                comm_thread.start()
                
                # 发送连接状态
                self.publish_isaac_status("CONNECTED")
                
                # 等待连接断开
                comm_thread.join()
                print("Isaac Sim连接断开")
                
        except Exception as e:
            print(f"Socket服务器错误: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def handle_client_communication(self):
        """处理与Isaac Sim的通信"""
        try:
            while not rospy.is_shutdown() and self.client_socket:
                # 接收Isaac Sim的消息
                try:
                    data = self.client_socket.recv(4096)
                    if not data:
                        break
                    
                    # 解析消息
                    message = json.loads(data.decode('utf-8'))
                    self.handle_isaac_message(message)
                    
                except socket.timeout:
                    continue
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    continue
                
                # 发送排队的消息到Isaac Sim
                self.send_queued_messages()
                
                time.sleep(0.01)  # 100Hz
                
        except Exception as e:
            print(f"通信错误: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
    
    def handle_isaac_message(self, message):
        """处理来自Isaac Sim的消息"""
        msg_type = message.get('type')
        
        if msg_type == 'robot_pose':
            # 关键修复：处理机器人位姿更新
            self.handle_robot_pose_update(message.get('data'))
            
        elif msg_type == 'request_goal':
            # 请求导航目标
            goal_data = message.get('data')
            self.send_navigation_goal(goal_data['x'], goal_data['y'], goal_data.get('yaw', 0.0))
            
        elif msg_type == 'exploration_status':
            # 探索状态更新
            status = message.get('data')
            self.publish_exploration_status(status)
        
        elif msg_type == 'heartbeat':
            # 心跳消息
            self.queue_message({
                'type': 'heartbeat_response',
                'timestamp': time.time()
            })
    
    def handle_robot_pose_update(self, pose_data):
        """关键修复：处理Isaac Sim真值位姿更新并强制覆盖SLAM定位"""
        if not pose_data:
            return
            
        try:
            # 提取Isaac Sim真值位姿数据
            position = pose_data.get('position', [0.0, 0.0, 0.0])
            yaw = pose_data.get('yaw', 0.0)
            
            # 数据验证：检查数值是否合理
            if (abs(position[0]) > 1000 or abs(position[1]) > 1000 or 
                abs(position[2]) > 1000 or abs(yaw) > 10):
                print(f"警告: 收到异常真值位姿数据: pos={position}, yaw={yaw}")
                return
            
            # 更新内部状态为真值位置
            self.robot_pose = {
                'position': position,
                'yaw': yaw,
                'timestamp': rospy.Time.now(),
                'source': 'isaac_sim_ground_truth'  # 标记数据源
            }
            self.last_pose_time = rospy.Time.now()
            
            # 关键修复：强制发布真值TF变换，覆盖Cartographer的定位
            self.publish_ground_truth_tf(position, yaw)
            
            # 发布真值机器人位姿话题
            self.publish_robot_pose_topic(position, yaw)
            
            # 关键修复：发布静态变换，将ground truth位置映射到标准frame
            self.publish_ground_truth_to_base_link_transform(position, yaw)
            
            # 调试输出（每5秒一次）
            current_time = rospy.Time.now()
            if (current_time - getattr(self, 'last_debug_time', rospy.Time(0))).to_sec() > 5.0:
                print(f"✅ 真值位姿更新: pos=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}], yaw={np.degrees(yaw):.1f}° (Isaac Ground Truth)")
                self.last_debug_time = current_time
                
        except Exception as e:
            print(f"处理真值位姿更新时出错: {e}")
    
    def publish_ground_truth_tf(self, position, yaw):
        """发布Isaac Sim真值TF变换"""
        try:
            # 创建真值TF变换消息
            t = TransformStamped()
            
            # 设置坐标系 - 使用特殊的frame_id标识真值
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link_ground_truth"
            
            # 设置真值位置（直接使用Isaac Sim的精确坐标）
            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])
            
            # 设置真值旋转（yaw角转四元数）
            quaternion = transformations.quaternion_from_euler(0, 0, yaw)
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]
            
            # 广播真值TF变换
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            print(f"发布真值TF变换时出错: {e}")
    
    def publish_ground_truth_to_base_link_transform(self, position, yaw):
        """发布从真值位置到标准base_link的变换，强制覆盖SLAM定位"""
        try:
            # 创建覆盖SLAM的TF变换
            t = TransformStamped()
            
            # 关键修复：直接发布到标准的base_link frame，覆盖Cartographer
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"  # 直接覆盖标准frame
            
            # 使用Isaac Sim的真值位置
            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])
            
            # 使用Isaac Sim的真值朝向
            quaternion = transformations.quaternion_from_euler(0, 0, yaw)
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]
            
            # 强制广播，覆盖Cartographer的定位
            self.tf_broadcaster.sendTransform(t)
            
            # 每10秒输出一次覆盖状态
            current_time = rospy.Time.now()
            if not hasattr(self, 'last_override_debug_time'):
                self.last_override_debug_time = rospy.Time(0)
            
            if (current_time - self.last_override_debug_time).to_sec() > 10.0:
                print(f"🔄 TF覆盖: Isaac真值位置覆盖SLAM定位，防止地图漂移")
                self.last_override_debug_time = current_time
            
        except Exception as e:
            print(f"发布TF覆盖变换时出错: {e}")
    
    def publish_robot_tf(self, position, yaw):
        """关键修复：发布正确的TF变换"""
        try:
            # 创建TF变换消息
            t = TransformStamped()
            
            # 设置坐标系
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            
            # 设置平移（直接使用Isaac Sim的坐标）
            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])
            
            # 设置旋转（yaw角转四元数）
            quaternion = transformations.quaternion_from_euler(0, 0, yaw)
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]
            
            # 广播TF变换
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            print(f"发布TF变换时出错: {e}")
    
    def publish_robot_pose_topic(self, position, yaw):
        """关键修复：发布机器人位姿话题"""
        try:
            # 创建位姿消息
            pose_msg = Float32MultiArray()
            pose_msg.data = [
                float(position[0]),  # x
                float(position[1]),  # y
                float(yaw)           # yaw
            ]
            
            # 发布位姿
            self.robot_pose_pub.publish(pose_msg)
            
        except Exception as e:
            print(f"发布位姿话题时出错: {e}")
    
    def queue_message(self, message):
        """将消息加入队列"""
        with self.queue_lock:
            self.message_queue.append(message)
    
    def send_queued_messages(self):
        """发送队列中的消息"""
        if not self.client_socket:
            return
            
        with self.queue_lock:
            while self.message_queue:
                message = self.message_queue.pop(0)
                try:
                    data = json.dumps(message).encode('utf-8')
                    self.client_socket.send(data + b'\n')
                except Exception as e:
                    print(f"发送消息失败: {e}")
                    break
    
    def map_callback(self, msg: OccupancyGrid):
        """地图更新回调"""
        self.current_map = msg
        
        # 转换地图数据
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        # 转换占用栅格数据
        map_data = np.array(msg.data).reshape((height, width)).tolist()
        
        # 发送地图更新到Isaac Sim
        map_message = {
            'type': 'map_update',
            'data': {
                'width': width,
                'height': height,
                'resolution': resolution,
                'origin': origin,
                'data': map_data
            }
        }
        self.queue_message(map_message)
        
        # 定期打印地图状态
        if not hasattr(self, 'last_map_debug_time'):
            self.last_map_debug_time = rospy.Time.now()
        
        if (rospy.Time.now() - self.last_map_debug_time).to_sec() > 10.0:
            print(f"地图更新: {width}x{height}, 分辨率: {resolution:.3f}m/cell")
            self.last_map_debug_time = rospy.Time.now()
    
    def mapex_cmd_callback(self, msg: Twist):
        """MapEx命令回调"""
        self.mapex_velocity = msg
        
        # 发送速度命令到Isaac Sim
        velocity_message = {
            'type': 'velocity_command',
            'data': {
                'linear_x': msg.linear.x,
                'angular_z': msg.angular.z
            }
        }
        self.queue_message(velocity_message)
        
        # 强制输出所有速度命令（包括零速度）- 降低频率
        current_time = rospy.Time.now()
        if not hasattr(self, 'last_velocity_debug_time') or (current_time - self.last_velocity_debug_time).to_sec() > 2.0:
            print(f"🚀 MapEx速度命令: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
            self.last_velocity_debug_time = current_time
    
    def exploration_done_callback(self, msg: Bool):
        """探索完成回调"""
        self.exploration_complete = msg.data
        
        # 发送探索完成状态到Isaac Sim
        done_message = {
            'type': 'exploration_done',
            'data': msg.data
        }
        self.queue_message(done_message)
        
        if msg.data:
            print("收到MapEx探索完成信号")
    
    def send_navigation_goal(self, x: float, y: float, yaw: float = 0.0):
        """发送导航目标"""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # 转换航向角为四元数
        quaternion = transformations.quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        
        self.goal_pub.publish(goal)
        rospy.loginfo(f"发送导航目标: ({x:.2f}, {y:.2f}, {np.degrees(yaw):.1f}°)")
    
    def publish_exploration_status(self, status: str):
        """发布探索状态"""
        msg = String()
        msg.data = status
        self.exploration_status_pub.publish(msg)
        print(f"探索状态更新: {status}")
    
    def publish_isaac_status(self, status: str):
        """发布Isaac Sim状态"""
        msg = String()
        msg.data = status
        self.isaac_status_pub.publish(msg)
    
    def save_cartographer_map(self, filename: str) -> bool:
        """保存Cartographer地图"""
        try:
            # 调用Cartographer保存状态服务
            rospy.wait_for_service('/write_state', timeout=5.0)
            from cartographer_ros_msgs.srv import WriteState
            
            write_state = rospy.ServiceProxy('/write_state', WriteState)
            response = write_state(filename, True)  # include_unfinished_submaps=True
            
            if response.status.code == 0:  # SUCCESS
                rospy.loginfo(f"成功保存SLAM地图到: {filename}")
                
                # 通知Isaac Sim
                save_message = {
                    'type': 'map_saved',
                    'data': {
                        'filename': filename,
                        'success': True
                    }
                }
                self.queue_message(save_message)
                return True
            else:
                rospy.logerr(f"保存地图失败: {response.status.message}")
                return False
                
        except Exception as e:
            rospy.logerr(f"保存地图时发生异常: {e}")
            return False
    
    def check_robot_pose_status(self):
        """检查机器人位姿状态"""
        if self.robot_pose is None:
            return "NO_POSE"
        
        time_since_last_pose = (rospy.Time.now() - self.last_pose_time).to_sec()
        if time_since_last_pose > self.pose_timeout:
            return "POSE_TIMEOUT"
        
        return "POSE_OK"
    
    def run(self):
        """运行ROS桥接节点"""
        # 启动socket服务器线程
        server_thread = threading.Thread(target=self.start_socket_server)
        server_thread.daemon = True
        server_thread.start()
        
        # ROS主循环
        rate = rospy.Rate(30)  # 30Hz
        
        while not rospy.is_shutdown():
            # 检查机器人位姿状态
            pose_status = self.check_robot_pose_status()
            
            # 定期状态报告
            if not hasattr(self, 'last_status_time'):
                self.last_status_time = rospy.Time.now()
            
            if (rospy.Time.now() - self.last_status_time).to_sec() > 30.0:
                print(f"桥接状态: 位姿={pose_status}, 地图={'有' if self.current_map else '无'}, 探索={'完成' if self.exploration_complete else '进行中'}")
                self.last_status_time = rospy.Time.now()
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        print("ROS桥接节点关闭")

if __name__ == '__main__':
    try:
        bridge = IsaacSimROSBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        print("ROS桥接节点被中断")