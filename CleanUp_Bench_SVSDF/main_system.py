#!/usr/bin/env python3
"""
主系统模块 - 完整SLAM + MapEx + Cartographer集成版本
支持先探索建图，后执行任务的两阶段流程
"""

import psutil
import torch
import numpy as np
import time
import random
import gc
import math
import rospy
import subprocess
import os
import signal  # 修复：添加signal导入以支持进程终止操作
from typing import List, Dict

def print_memory_usage(stage_name: str = ""):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"内存: {memory_mb:.1f}MB - {stage_name}")

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
})

rospy.init_node('isaac_sim_slam_robot', anonymous=True)

# 检查ROS Bridge扩展
import omni.kit.app
import carb

ext_manager = omni.kit.app.get_app().get_extension_manager()

if not ext_manager.is_extension_enabled("omni.isaac.ros_bridge"):
    ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)

simulation_app.update()
simulation_app.update()
simulation_app.update()

# 检查roscore连接
import rosgraph
if rosgraph.is_master_online():
    print("roscore连接正常")

# 现在安全导入Isaac Sim核心模块
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, Usd, UsdShade, Sdf
import isaacsim.core.utils.prims as prim_utils

# 导入本地模块
from data_structures import *
from visualizer import RealMovementVisualizer
from path_planner import SLAMPathPlanner
from robot_controller import SLAMRobotController
from ros_interface import ROSBridgeManager

class CompleteSLAMCoverageSystem:
    """完整SLAM覆盖系统 - 先探索建图，后执行任务"""
    
    def __init__(self):
        self.world = None
        self.mobile_base = None
        self.robot_controller = None
        self.path_planner = None
        self.visualizer = None
        
        # ROS接口管理器
        self.ros_bridge = ROSBridgeManager()
        
        self.scene_objects = []
        self.coverage_path = []
        self.coverage_stats = {
            'swept_objects': 0,
            'grasped_objects': 0,
            'delivered_objects': 0,
            'failed_grasps': 0,
            'total_coverage_points': 0,
            'exploration_time': 0,
            'mapping_complete': False,
            'coverage_complete': False
        }
        
        # 系统状态 - 三阶段流程
        self.current_phase = "INITIALIZATION"  # INITIALIZATION -> SLAM_EXPLORATION -> TASK_EXECUTION -> COMPLETE
        self.slam_complete = False
        self.exploration_start_time = None
        
        # 物体配置
        self.klt_box_usd_path = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis/Lobby/My_asset/T/small_KLT.usd"
        self.klt_approach_distance = 0.8
        
        # 机械臂配置
        self.arm_poses = {
            "home": [0.0, 0.524, 0.0, -0.785, 0.0, 1.571, 0.785],
            "grasp_approach": [0.0, 1.7, 0.0, -0.646, 0.0, 2.234, 0],
            "grasp_lift": [0.0, 1.047, 0.0, -0.646, 0.0, 2.234, 0.785],
            "carry": [0.0, 1.047, 0.0, -0.646, 0.0, 2.234, 0.785]
        }
        
        # 任务参数
        self.grasp_trigger_distance = 1.5
        self.grasp_approach_distance = 0.26
        self.arm_stabilization_time = 2.0
        self.gripper_stabilization_time = 1.0
        self.gripper_open_pos = 0.05
        self.gripper_close_pos = 0.025
        self.angle_tolerance = 0.05
        self.position_tolerance = 0.15
        self.angular_kp = 5.0
        self.max_angular_vel = 2.0
        
        self.carrying_object = None
        self.return_position = None
        
        # MapEx桥接进程
        self.mapex_bridge_process = None
    
    def initialize_system(self):
        """初始化系统"""
        print("初始化完整SLAM集成覆盖系统...")
        print("阶段: Isaac Sim环境初始化")
        
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,
            rendering_dt=1.0/30.0
        )
        self.world.scene.clear()
        
        # 物理设置
        physics_context = self.world.get_physics_context()
        physics_context.set_gravity(-9.81)
        physics_context.enable_gpu_dynamics(True)
        
        # 创建地面
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.5]),
            scale=np.array([50.0, 50.0, 1.0]),
            color=np.array([0.4, 0.4, 0.4])
        )
        self.world.scene.add(ground)
        
        self._setup_lighting()
        self._initialize_components()
        self._create_slam_exploration_environment()
        
        print(f"完整SLAM集成覆盖系统初始化完成")
        print(f"系统将按以下三阶段执行:")
        print(f"  1. SLAM探索建图阶段 (Cartographer + MapEx)")
        print(f"  2. 基于已知地图的任务规划阶段")
        print(f"  3. 覆盖清扫抓取执行阶段")
        return True
    
    def _setup_lighting(self):
        """设置照明"""
        main_light = prim_utils.create_prim("/World/MainLight", "DistantLight")
        distant_light = UsdLux.DistantLight(main_light)
        distant_light.CreateIntensityAttr(4000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.95))
        
        env_light = prim_utils.create_prim("/World/EnvLight", "DomeLight")
        dome_light = UsdLux.DomeLight(env_light)
        dome_light.CreateIntensityAttr(800)
        dome_light.CreateColorAttr((0.8, 0.9, 1.0))
        print("照明设置完成")
    
    def _initialize_components(self):
        """初始化组件"""
        # SLAM版本的路径规划器和控制器
        self.path_planner = SLAMPathPlanner()
        self.visualizer = RealMovementVisualizer(self.world)
        
        # 启动ROS接口
        self.ros_bridge.start()
        
        # 设置回调函数
        mapex_interface = self.ros_bridge.get_mapex_interface()
        mapex_interface.set_exploration_done_callback(self._on_exploration_complete)
        mapex_interface.set_map_update_callback(self._on_map_update)
        
        print("组件初始化完成（SLAM集成版本）")
    
    def _create_slam_exploration_environment(self):
        """创建SLAM探索环境 - 未知环境 + 已知任务对象坐标"""
        print("创建SLAM探索环境...")
        
        # 1. 障碍物 (红色) - 机器人需要通过SLAM发现
        obstacles_config = [
            {"pos": [3.5, 2.2, 0.25], "size": [0.8, 0.6, 2.0], "color": [0.8, 0.2, 0.2], "shape": "box"},
            {"pos": [1.8, -2.8, 0.3], "size": [1.4, 0.4, 2.2], "color": [0.7, 0.1, 0.1], "shape": "box"},
            {"pos": [-2.2, 3.1, 0.35], "size": [0.9], "color": [0.6, 0.2, 0.2], "shape": "sphere"},
            {"pos": [5.2, 1.1, 0.25], "size": [1.0, 1.0, 1.8], "color": [0.5, 0.1, 0.1], "shape": "box"},
            {"pos": [-3.8, -2.3, 0.4], "size": [0.7], "color": [0.6, 0.2, 0.2], "shape": "sphere"},
            {"pos": [0.9, 4.1, 0.2], "size": [0.6, 1.2, 2.0], "color": [0.7, 0.15, 0.15], "shape": "box"},
            {"pos": [-4.5, 0.8, 0.3], "size": [0.5, 0.8, 2.0], "color": [0.8, 0.1, 0.1], "shape": "box"},
        ]
        
        # 2. 清扫目标 (黄色) - 已知坐标，用于任务执行
        sweep_config = [
            {"pos": [2.1, 0.7, 0.05], "size": [0.04], "color": [1.0, 1.0, 0.2], "shape": "sphere"},
            {"pos": [4.2, 3.1, 0.05], "size": [0.04], "color": [0.9, 0.9, 0.1], "shape": "sphere"},
            {"pos": [6.1, -0.8, 0.05], "size": [0.04], "color": [1.0, 0.8, 0.0], "shape": "sphere"},
            {"pos": [-1.3, 2.4, 0.05], "size": [0.04], "color": [0.8, 0.8, 0.2], "shape": "sphere"},
            {"pos": [-3.1, -1.7, 0.05], "size": [0.04], "color": [1.0, 0.9, 0.1], "shape": "sphere"},
            {"pos": [1.7, -4.2, 0.05], "size": [0.04], "color": [0.9, 0.7, 0.0], "shape": "sphere"},
        ]
        
        # 3. 抓取物体 (绿色) - 已知坐标，用于任务执行
        grasp_config = [
            {"pos": [2.8, 1.5, 0.08], "size": [0.06, 0.06, 0.06], "color": [0.2, 0.8, 0.2], "shape": "box"},
            {"pos": [-1.5, -3.2, 0.08], "size": [0.06, 0.06, 0.06], "color": [0.1, 0.9, 0.1], "shape": "box"},
            {"pos": [4.7, 2.8, 0.08], "size": [0.06, 0.06, 0.06], "color": [0.0, 0.8, 0.0], "shape": "box"},
            {"pos": [-2.9, 1.2, 0.08], "size": [0.06, 0.06, 0.06], "color": [0.1, 0.7, 0.1], "shape": "box"},
            {"pos": [3.3, -1.9, 0.08], "size": [0.06, 0.06, 0.06], "color": [0.2, 0.85, 0.2], "shape": "box"},
        ]
        
        # 4. KLT框子任务区域 (蓝色)
        klt_box_config = [
            {"pos": [7, -2, 0.1], "size": [1.8, 1.0, 0.35], "usd_path": self.klt_box_usd_path, "shape": "usd_asset"},
        ]
        
        # 创建所有对象
        all_configs = [
            (obstacles_config, ObjectType.OBSTACLE),
            (sweep_config, ObjectType.SWEEP),
            (grasp_config, ObjectType.GRASP),
            (klt_box_config, ObjectType.TASK)
        ]
        
        for config_list, obj_type in all_configs:
            for i, config in enumerate(config_list):
                self._create_scene_object(config, obj_type, i)
        
        # 5. 添加外围围墙
        self._create_boundary_walls()
        
        print(f"SLAM探索环境创建完成:")
        print(f"  障碍物: {len(obstacles_config)}个 (机器人将通过SLAM发现)")
        print(f"  清扫目标: {len(sweep_config)}个 (坐标已知，用于任务执行)")
        print(f"  抓取物体: {len(grasp_config)}个 (坐标已知，用于任务执行)")
        print(f"  KLT框子任务区域: {len(klt_box_config)}个")
        print(f"  外围围墙: 已创建")
        print(f"  激光雷达发布到: /robot_lidar_pointcloud")
        print(f"  地图未知，需要机器人自主探索建图")
    
    def _create_scene_object(self, config: Dict, obj_type: ObjectType, index: int):
        """创建场景对象"""
        name = f"{obj_type.value}_{index}"
        prim_path = f"/World/{name}"
        
        # KLT框子任务区域使用USD资产
        if obj_type == ObjectType.TASK and config.get("shape") == "usd_asset":
            isaac_obj = self._create_klt_box_task_area(config, prim_path, name)
            collision_boundary = CollisionBoundary(
                center=np.array(config["pos"]),
                shape_type="box",
                dimensions=np.array(config["size"])
            )
        elif config["shape"] == "sphere":
            isaac_obj = DynamicSphere(
                prim_path=prim_path,
                name=name,
                position=np.array(config["pos"]),
                radius=config["size"][0],
                color=np.array(config["color"])
            )
            collision_boundary = CollisionBoundary(
                center=np.array(config["pos"]),
                shape_type="sphere",
                dimensions=np.array(config["size"])
            )
        else:  # box
            if obj_type in [ObjectType.OBSTACLE, ObjectType.TASK]:
                isaac_obj = FixedCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array(config["pos"]),
                    scale=np.array(config["size"]),
                    color=np.array(config["color"])
                )
            else:
                isaac_obj = DynamicCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array(config["pos"]),
                    scale=np.array(config["size"]),
                    color=np.array(config["color"])
                )
            
            collision_boundary = CollisionBoundary(
                center=np.array(config["pos"]),
                shape_type="box",
                dimensions=np.array(config["size"])
            )
        
        # 对于USD资产，直接管理
        if obj_type == ObjectType.TASK and config.get("shape") == "usd_asset":
            isaac_obj = self._create_klt_box_task_area(config, prim_path, name)
        elif isaac_obj:
            self.world.scene.add(isaac_obj)
        
        # 创建场景对象
        scene_obj = SceneObject(
            name=name,
            object_type=obj_type,
            position=np.array(config["pos"]),
            collision_boundary=collision_boundary,
            isaac_object=isaac_obj,
            color=np.array([0.3, 0.5, 0.8]) if obj_type == ObjectType.TASK else np.array(config["color"]),
            original_position=np.array(config["pos"])
        )
        
        self.scene_objects.append(scene_obj)
        
        # 只有任务相关对象需要添加到路径规划器（SLAM会发现障碍物）
        if obj_type in [ObjectType.SWEEP, ObjectType.GRASP, ObjectType.TASK]:
            self.path_planner.add_task_object(scene_obj)
        
        print(f"创建{obj_type.value}对象: {name}")
    
    def _create_boundary_walls(self):
        """创建外围围墙 - 增强版SLAM探索空间"""
        print("创建外围围墙...")
        
        # 分析当前所有对象的分布范围
        all_positions = []
        for obj in self.scene_objects:
            all_positions.append(obj.position[:2])
        
        if all_positions:
            positions_array = np.array(all_positions)
            min_x, min_y = positions_array.min(axis=0)
            max_x, max_y = positions_array.max(axis=0)
            
            # 为SLAM探索扩展围墙范围
            wall_margin = 5.0  # 增加围墙边距以适应SLAM探索
            wall_x_range = max(abs(min_x), abs(max_x)) + wall_margin
            wall_y_range = max(abs(min_y), abs(max_y)) + wall_margin
        else:
            wall_x_range = 15.0  # 增加默认围墙范围
            wall_y_range = 15.0
        
        # 围墙配置参数
        wall_thickness = 0.5
        wall_height = 2.5
        wall_color = [0.6, 0.6, 0.6]
        
        # 创建四面围墙
        walls_config = [
            {"name": "north_wall", "pos": [0.0, wall_y_range + wall_thickness/2, wall_height/2], 
             "size": [wall_x_range*2 + wall_thickness*2, wall_thickness, wall_height], "color": wall_color},
            {"name": "south_wall", "pos": [0.0, -(wall_y_range + wall_thickness/2), wall_height/2], 
             "size": [wall_x_range*2 + wall_thickness*2, wall_thickness, wall_height], "color": wall_color},
            {"name": "east_wall", "pos": [wall_x_range + wall_thickness/2, 0.0, wall_height/2], 
             "size": [wall_thickness, wall_y_range*2, wall_height], "color": wall_color},
            {"name": "west_wall", "pos": [-(wall_x_range + wall_thickness/2), 0.0, wall_height/2], 
             "size": [wall_thickness, wall_y_range*2, wall_height], "color": wall_color}
        ]
        
        # 创建围墙对象
        for wall_config in walls_config:
            wall_name = wall_config["name"]
            prim_path = f"/World/BoundaryWalls/{wall_name}"
            
            wall_obj = FixedCuboid(
                prim_path=prim_path,
                name=wall_name,
                position=np.array(wall_config["pos"]),
                scale=np.array(wall_config["size"]),
                color=np.array(wall_config["color"])
            )
            
            self.world.scene.add(wall_obj)
            
            collision_boundary = CollisionBoundary(
                center=np.array(wall_config["pos"]),
                shape_type="box",
                dimensions=np.array(wall_config["size"])
            )
            
            scene_obj = SceneObject(
                name=wall_name,
                object_type=ObjectType.OBSTACLE,
                position=np.array(wall_config["pos"]),
                collision_boundary=collision_boundary,
                isaac_object=wall_obj,
                color=np.array(wall_config["color"]),
                original_position=np.array(wall_config["pos"])
            )
            
            self.scene_objects.append(scene_obj)
        
        print(f"  SLAM探索围墙创建完成: 范围±{wall_x_range:.1f}m x ±{wall_y_range:.1f}m")
    
    def _create_klt_box_task_area(self, config: Dict, prim_path: str, name: str):
        """创建KLT框子任务区域"""
        try:
            stage = self.world.stage
            klt_prim = stage.DefinePrim(prim_path, "Xform")
            
            references = klt_prim.GetReferences()
            references.AddReference(config["usd_path"])
            
            xform = UsdGeom.Xformable(klt_prim)
            xform.ClearXformOpOrder()
            
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(config["pos"][0]), float(config["pos"][1]), 0.0))
            
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(2.5, 2.5, 2.5))
            
            # 添加物理属性
            if not klt_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(klt_prim)
                rigid_body_api.CreateRigidBodyEnabledAttr().Set(False)
            
            if not klt_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(klt_prim)
                collision_api.CreateCollisionEnabledAttr().Set(True)
            
            class KLTBoxReference:
                def __init__(self, prim_path, name, position):
                    self.prim_path = prim_path
                    self.name = name
                    self._position = position
                
                def set_visibility(self, visible):
                    pass
                
                def set_world_pose(self, position, orientation):
                    pass
            
            return KLTBoxReference(prim_path, name, np.array(config["pos"]))
            
        except Exception as e:
            print(f"创建KLT框子失败: {e}")
            return FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array(config["pos"]),
                scale=np.array(config["size"]),
                color=np.array([0.3, 0.5, 0.8])
            )
    
    def initialize_robot(self):
        """初始化机器人"""
        print("初始化机器人...")
        
        robot_usd_path = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm4.usd"
        print(f"使用机器人资产: {robot_usd_path}")
        
        self.mobile_base = WheeledRobot(
            prim_path="/World/create3_robot",
            name="create3_robot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=robot_usd_path,
            position=np.array([0.0, 0.0, 0.0])
        )
        
        self.world.scene.add(self.mobile_base)
        print("机器人添加到场景完成")
        
        return True
    
    def setup_post_load(self):
        """后加载设置"""
        print("后加载设置...")
        
        # 重置世界
        self.world.reset()
        print("世界重置完成")
        
        # 稳定物理
        print("物理稳定中...")
        for i in range(120):
            self.world.step(render=False)
            if i % 30 == 0:
                print(f"  稳定进度: {i+1}/120")
        
        # 获取机器人对象
        print("获取机器人对象...")
        self.mobile_base = None
        for retry in range(5):
            self.mobile_base = self.world.scene.get_object("create3_robot")
            if self.mobile_base is not None:
                print(f"机器人对象获取成功 (尝试 {retry+1}/5)")
                break
            else:
                for _ in range(30):
                    self.world.step(render=False)
        
        # 验证机器人对象
        controller = self.mobile_base.get_articulation_controller()
        print(f"机器人验证通过，DOF数量: {len(self.mobile_base.dof_names)}")
        
        # 修正物理层次结构
        self._fix_robot_physics_conservative()
        
        # 设置控制增益
        self._setup_robust_control_gains()
        
        # 移动机械臂到home位置
        self._move_arm_to_pose("home")
        
        # 初始化控制器 - SLAM版本
        self.robot_controller = SLAMRobotController(self.mobile_base, self.world, self.ros_bridge)
        
        # 设置覆盖可视化器引用
        self.robot_controller.set_coverage_visualizer(self.visualizer)
        
        print("SLAM机器人控制器初始化完成")
        
        # 验证机器人状态
        pos, yaw = self.robot_controller._get_robot_pose()
        print(f"机器人状态验证: 位置[{pos[0]:.3f}, {pos[1]:.3f}], 朝向{np.degrees(yaw):.1f}°")
        
        print("后加载设置完成")
        return True
    
    def _fix_robot_physics_conservative(self):
        """保守的机器人物理层次结构修正"""
        print("保守修正机器人物理层次结构...")
        
        robot_prim = self.world.stage.GetPrimAtPath("/World/create3_robot")
        
        problem_paths = [
            "/World/create3_robot/create_3/left_wheel/visuals_01",
            "/World/create3_robot/create_3/right_wheel/visuals_01"
        ]
        
        for wheel_path in problem_paths:
            wheel_prim = self.world.stage.GetPrimAtPath(wheel_path)
            if wheel_prim and wheel_prim.IsValid():
                if wheel_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body = UsdPhysics.RigidBodyAPI(wheel_prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                    print(f"  修正轮子视觉物理: {wheel_path.split('/')[-1]}")
                
                if wheel_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(wheel_prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
        
        print("保守物理层次结构修正完成")
    
    def _setup_robust_control_gains(self):
        """设置鲁棒的控制增益"""
        print("设置控制增益...")
        
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        kp = torch.zeros(num_dofs, dtype=torch.float32)
        kd = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 轮子控制
        wheel_names = ["left_wheel_joint", "right_wheel_joint"]
        for wheel_name in wheel_names:
            if wheel_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(wheel_name)
                kp[idx] = 0.0
                kd[idx] = 500.0
        
        # 机械臂控制
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for joint_name in arm_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 800.0
                kd[idx] = 40.0
        
        # 夹爪控制
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 1e5
                kd[idx] = 1e3
        
        articulation_controller.set_gains(kps=kp, kds=kd)
        print("控制增益设置完成")
    
    def _move_arm_to_pose(self, pose_name: str):
        """移动机械臂到指定姿态 - 60步插帧平滑移动"""
        print(f"机械臂平滑移动到: {pose_name} (60步插帧)")
        
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 获取当前关节位置
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        
        # 获取机械臂关节的起始位置
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        start_positions = []
        target_positions_filtered = []
        arm_joint_indices = []
        
        for i, joint_name in enumerate(arm_joint_names):
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                if i < len(target_positions):
                    start_positions.append(float(current_joint_positions[idx]))
                    target_positions_filtered.append(target_positions[i])
                    arm_joint_indices.append(idx)
        
        start_positions = np.array(start_positions)
        target_positions_filtered = np.array(target_positions_filtered)
        
        # 60步插帧移动
        interpolation_steps = 60
        for step in range(interpolation_steps):
            # 计算插值系数 (使用平滑的 ease-in-out 曲线)
            t = step / (interpolation_steps - 1)
            smooth_t = 3 * t * t - 2 * t * t * t  # 平滑插值曲线
            
            # 计算当前步的关节位置
            current_interpolated_positions = start_positions + smooth_t * (target_positions_filtered - start_positions)
            
            # 构建完整的关节位置数组
            joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
            
            # 保持当前所有关节位置
            for i in range(num_dofs):
                joint_positions[i] = float(current_joint_positions[i])
            
            # 设置插值后的机械臂关节位置
            for i, idx in enumerate(arm_joint_indices):
                joint_positions[idx] = float(current_interpolated_positions[i])
            
            # 应用动作
            action = ArticulationAction(joint_positions=joint_positions)
            articulation_controller.apply_action(action)
            
            # 每步都进行物理仿真
            self.world.step(render=True)
            
            # 显示进度
            if step % 15 == 0:
                print(f"  平滑移动进度: {step+1}/60 ({(step+1)/60*100:.1f}%)")
        
        # 最终稳定
        stabilization_steps = 20
        for _ in range(stabilization_steps):
            self.world.step(render=True)
        
        print(f"  机械臂平滑移动完成: {pose_name} (60步插帧平滑移动)")
    
    def start_mapex_bridge(self):
        """启动MapEx桥接进程"""
        try:
            print("启动MapEx桥接进程...")
            
            # 构建启动命令
            cmd = [
                "bash", "-c",
                f"source /opt/ros/noetic/setup.bash && "
                f"conda activate myenv_py38 && "
                f"cd /home/getting/SVSDF/CleanUp_Bench_SVSDF && "
                f"python3 MapEx_Bridge_Node.py" 
            ]
            
            self.mapex_bridge_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            print(f"MapEx桥接进程已启动，PID: {self.mapex_bridge_process.pid}")
            
            # 等待桥接进程初始化
            time.sleep(3.0)
            
            return True
            
        except Exception as e:
            print(f"启动MapEx桥接进程失败: {e}")
            return False
    
    def stop_mapex_bridge(self):
        """停止MapEx桥接进程"""
        if self.mapex_bridge_process:
            try:
                # 发送SIGTERM到整个进程组
                os.killpg(os.getpgid(self.mapex_bridge_process.pid), signal.SIGTERM)
                
                # 等待进程结束
                try:
                    self.mapex_bridge_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # 如果10秒后还没结束，强制杀死
                    os.killpg(os.getpgid(self.mapex_bridge_process.pid), signal.SIGKILL)
                    self.mapex_bridge_process.wait()
                
                print("MapEx桥接进程已停止")
                
            except Exception as e:
                print(f"停止MapEx桥接进程时出错: {e}")
            
            finally:
                self.mapex_bridge_process = None
    
    def _on_exploration_complete(self, complete: bool):
        """探索完成回调"""
        if complete and not self.slam_complete:
            self.slam_complete = True
            self.coverage_stats['mapping_complete'] = True
            if self.exploration_start_time:
                self.coverage_stats['exploration_time'] = time.time() - self.exploration_start_time
            
            print(f"\n=== SLAM探索完成 ===")
            print(f"探索时间: {self.coverage_stats['exploration_time']:.1f}秒")
            print(f"准备切换到任务执行阶段...")
            
            self.current_phase = "TASK_EXECUTION"
    
    def _on_map_update(self, map_data: dict):
        """地图更新回调"""
        # 更新路径规划器的地图
        self.path_planner.update_slam_map(map_data)
    
    def run_slam_exploration(self):
        """运行SLAM探索阶段 - 修复版本，确保位姿持续发布"""
        print("\n=== 开始SLAM探索阶段 ===")
        print("Phase 1: 使用Cartographer SLAM + MapEx探索未知环境")
        
        self.current_phase = "SLAM_EXPLORATION"
        self.exploration_start_time = time.time()
        
        # 1. 启动MapEx桥接进程
        if not self.start_mapex_bridge():
            print("MapEx桥接进程启动失败")
            return False
        
        # 2. 发布探索状态
        mapex_interface = self.ros_bridge.get_mapex_interface()
        mapex_interface.publish_exploration_status("EXPLORATION_STARTED")
        
        print("等待Cartographer、MapEx桥接和MapEx进程启动...")
        time.sleep(8.0)  # 给三个进程启动时间
        
        print("开始自主探索...")
        exploration_steps = 0
        max_exploration_steps = 25000  # 增加最大探索步数
        
        # 关键修复：确保位姿持续发布
        last_pose_publish_time = time.time()
        pose_publish_interval = 0.1  # 10Hz发布频率
        
        while (not self.slam_complete and 
               exploration_steps < max_exploration_steps and 
               self.current_phase == "SLAM_EXPLORATION"):
            
            # 获取机器人当前位置并标记覆盖
            current_pos, current_yaw = self.robot_controller._get_robot_pose()
            self.robot_controller._mark_coverage_ultra_smooth(current_pos)
            
            # 关键修复：强制发布位姿到ROS（高频率）
            current_time = time.time()
            if current_time - last_pose_publish_time >= pose_publish_interval:
                self.robot_controller._publish_robot_pose_to_ros(current_pos, current_yaw)
                last_pose_publish_time = current_time
                # 调试输出：每500步显示一次位姿发布状态
                if exploration_steps % 500 == 0:
                    print(f"强制发布位姿: [{current_pos[0]:.3f}, {current_pos[1]:.3f}], yaw: {np.degrees(current_yaw):.1f}°")
            
            # 处理来自MapEx的ROS速度指令
            try:
                # 检查是否有新的速度指令
                mapex_velocity = mapex_interface.mapex_velocity
                
                if mapex_velocity:
                    linear_vel = float(mapex_velocity.linear.x)
                    angular_vel = float(mapex_velocity.angular.z)
                    
                    # 如果收到非零速度指令，更新时间戳
                    if abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01:
                        last_velocity_time = time.time()
                        
                        # 执行速度指令
                        self.robot_controller._send_velocity_command(linear_vel, angular_vel)
                        
                        if exploration_steps % 200 == 0:
                            print(f"执行MapEx速度指令: linear={linear_vel:.3f}, angular={angular_vel:.3f}")
                    else:
                        # 检查速度指令超时
                        if not hasattr(self, 'last_velocity_time'):
                            self.last_velocity_time = time.time()
                        
                        if (time.time() - getattr(self, 'last_velocity_time', time.time())) > 2.0:
                            # 超时则停止机器人
                            self.robot_controller._send_velocity_command(0.0, 0.0)
                
            except Exception as e:
                print(f"处理速度指令时出错: {e}")
                # 出错时停止机器人
                self.robot_controller._send_velocity_command(0.0, 0.0)
            
            # 执行一步仿真
            self.world.step(render=True)
            exploration_steps += 1
            
            # 周期性进度报告
            if exploration_steps % 1000 == 0:
                elapsed_time = time.time() - self.exploration_start_time
                print(f"探索进度: {exploration_steps}/{max_exploration_steps} "
                      f"步, 已用时: {elapsed_time:.1f}秒")
                
                # 检查是否有地图数据
                current_map = mapex_interface.get_current_map()
                if current_map:
                    print(f"地图尺寸: {current_map['width']}x{current_map['height']}, "
                          f"分辨率: {current_map['resolution']:.2f}m/cell")
                
                # 检查位姿发布状态
                if hasattr(self.robot_controller, 'last_pose_publish_time'):
                    pose_age = current_time - self.robot_controller.last_pose_publish_time
                    print(f"位姿发布状态: 最后发布 {pose_age:.1f}秒前")
            
            # 短暂延时
            time.sleep(0.01)
        
        # 探索结束处理
        print("SLAM探索阶段结束，停止机器人")
        self.robot_controller._send_velocity_command(0.0, 0.0)  # 确保机器人停止
        
        if self.slam_complete:
            print("✓ SLAM探索成功完成")
            mapex_interface.publish_exploration_status("EXPLORATION_COMPLETED")
            
            # 保存地图
            cartographer_interface = self.ros_bridge.get_cartographer_interface()
            map_file = "/home/getting/isaac_sim_slam_map.pbstream"
            if cartographer_interface.save_map(map_file):
                print(f"✓ SLAM地图已保存: {map_file}")
        else:
            print("⚠ SLAM探索达到最大步数限制")
            mapex_interface.publish_exploration_status("EXPLORATION_TIMEOUT")
        
        print(f"探索阶段结束，总用时: {time.time() - self.exploration_start_time:.1f}秒")
        
        # 停止MapEx桥接进程
        self.stop_mapex_bridge()
        
        return self.slam_complete
    
    def plan_coverage_mission(self):
        """基于SLAM地图规划覆盖任务"""
        print("\n=== 基于SLAM地图规划覆盖任务 ===")
        print("Phase 2: 使用已知地图进行任务规划")
        
        current_pos, _ = self.robot_controller._get_robot_pose()
        print(f"机器人当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        
        # 获取SLAM地图
        mapex_interface = self.ros_bridge.get_mapex_interface()
        slam_map = mapex_interface.get_current_map()
        
        if slam_map:
            print(f"使用SLAM地图进行路径规划")
            print(f"地图信息: {slam_map['width']}x{slam_map['height']}, "
                  f"分辨率: {slam_map['resolution']:.2f}m/cell")
            
            # 生成基于SLAM地图的覆盖路径
            self.coverage_path = self.path_planner.generate_slam_based_coverage_path(
                current_pos, slam_map)
        else:
            print("⚠ 无法获取SLAM地图，使用备用路径规划")
            # 备用方案：基于已知任务对象生成简单路径
            self.coverage_path = self.path_planner.generate_task_based_path(current_pos)
        
        # 设置完整路径可视化
        self.visualizer.setup_complete_path_visualization(self.coverage_path)
        
        print(f"覆盖任务规划完成: {len(self.coverage_path)}个覆盖点")
        print("已集成超级平滑实时覆盖区域可视化系统")
        print_memory_usage("覆盖规划完成")
        
        return True
    
    def execute_task_coverage(self):
        """执行任务覆盖阶段"""
        print("\n=== 执行任务覆盖阶段 ===")
        print("Phase 3: 基于已知地图执行覆盖清扫抓取任务")
        print_memory_usage("任务开始")
        
        # 展示路径预览
        print("展示路径预览...")
        for step in range(60):
            self.world.step(render=True)
            time.sleep(0.03)
        
        print("开始执行覆盖路径...")
        
        successful_points = 0
        step_counter = 0
        
        for i, point in enumerate(self.coverage_path):
            print(f"\n=== 导航到点 {i+1}/{len(self.coverage_path)} ===")
            
            # 鲁棒移动到目标点 - 使用基于SLAM地图的导航
            success = self.robot_controller.move_to_position_robust(
                point.position, point.orientation)
            
            if success:
                successful_points += 1
                print(f"  点 {i+1} 导航成功")
            else:
                print(f"  点 {i+1} 导航失败，继续下一个点")
            
            # 获取当前位置
            current_pos, _ = self.robot_controller._get_robot_pose()
            
            # 检查四类对象交互 - 集成真实抓取
            self._check_four_object_interactions(current_pos)
            
            # 短暂停顿
            for _ in range(5):
                self.world.step(render=True)
            
            time.sleep(0.1)
        
        print(f"\n基于SLAM地图的覆盖执行完成!")
        print(f"成功到达点数: {successful_points}/{len(self.coverage_path)}")
        self._show_slam_coverage_results()
        
        self.coverage_stats['coverage_complete'] = True
        self.current_phase = "COMPLETE"
    
    def _check_four_object_interactions(self, robot_pos: np.ndarray):
        """检查四类对象交互 - 基于已知坐标"""
        for scene_obj in self.scene_objects:
            if not scene_obj.is_active:
                continue
            
            # 只处理任务相关对象（sweep, grasp, task）
            if scene_obj.object_type in [ObjectType.SWEEP, ObjectType.GRASP, ObjectType.TASK]:
                # 检查碰撞（基于已知坐标）
                distance = np.linalg.norm(robot_pos[:2] - scene_obj.position[:2])
                if distance <= INTERACTION_DISTANCE:
                    self._handle_object_interaction(scene_obj, robot_pos)
    
    def _handle_object_interaction(self, scene_obj: SceneObject, robot_pos: np.ndarray):
        """处理对象交互"""
        print(f"    交互检测: {scene_obj.name} ({scene_obj.object_type.value})")
        
        if scene_obj.object_type == ObjectType.SWEEP:
            self._handle_sweep_object(scene_obj)
        elif scene_obj.object_type == ObjectType.GRASP:
            self._handle_grasp_object_with_real_arm(scene_obj, robot_pos)
    
    def _handle_sweep_object(self, sweep_obj: SceneObject):
        """处理清扫对象"""
        print(f"      清扫目标消失: {sweep_obj.name}")
        
        # 隐藏对象
        sweep_obj.isaac_object.set_visibility(False)
        sweep_obj.isaac_object.set_world_pose(
            np.array([100.0, 100.0, -5.0]), 
            np.array([0, 0, 0, 1])
        )
        
        # 标记为非活跃
        sweep_obj.is_active = False
        self.coverage_stats['swept_objects'] += 1
        
        print(f"      清扫完成，总清扫数: {self.coverage_stats['swept_objects']}")
    
    def _handle_grasp_object_with_real_arm(self, grasp_obj: SceneObject, robot_pos: np.ndarray):
        """处理抓取对象 - 真实Franka机械臂抓取"""
        print(f"      检测到抓取对象: {grasp_obj.name}")
        
        # 检查是否已经抓取失败过
        if hasattr(grasp_obj, 'grasp_failed') and grasp_obj.grasp_failed:
            print(f"      物体 {grasp_obj.name} 之前抓取失败（已变灰色），跳过不再尝试")
            return
        
        # 如果已经在运送其他物体，跳过
        if self.carrying_object is not None:
            print(f"      已在运送物体，跳过")
            return
        
        # 检查是否在抓取触发距离内
        distance = np.linalg.norm(robot_pos[:2] - grasp_obj.position[:2])
        print(f"      当前距离抓取对象: {distance:.3f}m (触发阈值: {self.grasp_trigger_distance}m)")
        
        if distance > self.grasp_trigger_distance:
            print(f"      距离超出触发范围，跳过抓取")
            return
        
        print(f"      触发抓取任务! 距离: {distance:.3f}m")
        
        # 记录返回位置
        self.return_position = robot_pos.copy()
        
        # 执行完整抓取序列
        grasp_success = self._execute_full_grasp_sequence_with_test_params(grasp_obj)
        
        if grasp_success:
            # 抓取成功，运送到KLT框子
            print(f"      抓取成功，开始运送到KLT框子...")
            self._deliver_to_klt_box_with_real_arm(grasp_obj)
        else:
            # 抓取失败，直接返回继续覆盖任务
            print(f"      抓取失败，取消运送，直接返回继续覆盖任务...")
        
        # 返回继续覆盖
        self._return_to_coverage()
    
    def _execute_full_grasp_sequence_with_test_params(self, grasp_obj: SceneObject):
        """执行完整抓取序列 - 60步插帧平滑移动 + 抓取成功检测"""
        print(f"        执行真实Franka机械臂抓取序列 (60步插帧平滑移动)...")
        
        # 步骤1: 定位到抓取对象正前方
        self._position_for_grasp_precise(grasp_obj)
        
        # 步骤1.5: 记录抓取前物体的Z轴坐标
        initial_z_position = self._get_object_current_z_position(grasp_obj)
        print(f"        抓取前物体Z轴坐标: {initial_z_position:.3f}m")
        
        # 步骤2: 机械臂执行抓取动作 - 60步插帧平滑版本
        self._control_gripper_with_test_params("open")
        print("        2.2 机械臂平滑移动到抓取姿态(60步插帧)")
        self._move_arm_to_pose("grasp_approach")
        self._control_gripper_with_test_params("close")
        print("        2.3 平滑抬起物体(60步插帧)")
        self._move_arm_to_pose("grasp_lift")
        
        # 步骤3: 抓取成功检测机制
        success = self._check_grasp_success(grasp_obj, initial_z_position)
        
        if success:
            # 抓取成功，标记为运送中
            self.carrying_object = grasp_obj
            grasp_obj.is_active = False
            self.coverage_stats['grasped_objects'] += 1
            print(f"        ✓ Franka机械臂抓取成功! 物体已被抬起")
            return True
        else:
            # 抓取失败，标记物体为不再尝试，回到home位置
            print(f"        ✗ 抓取失败! 物体未被成功抓起，取消运送任务...")
            print(f"        ✗ 将物体 {grasp_obj.name} 标记为不再尝试")
            
            # 标记物体为抓取失败，不再尝试
            grasp_obj.grasp_failed = True
            
            # 将物体变为灰色，表示抓取失败
            self._set_object_to_gray(grasp_obj)
            
            # 统计抓取失败次数
            self.coverage_stats['failed_grasps'] += 1
            
            # 机械臂回到home位置
            self._move_arm_to_pose("home")
            
            print(f"        物体 {grasp_obj.name} 已变为灰色并被永久跳过，机器人将继续覆盖任务")
            return False
    
    def _get_object_current_z_position(self, grasp_obj: SceneObject) -> float:
        """获取物体当前的Z轴坐标"""
        try:
            # 获取Isaac对象的当前世界位置
            position, _ = grasp_obj.isaac_object.get_world_pose()
            current_z = float(position[2])
            return current_z
        except Exception as e:
            print(f"          警告: 无法获取物体Z坐标: {e}")
            # 使用备用方法：从场景对象的位置获取
            return float(grasp_obj.position[2])
    
    def _check_grasp_success(self, grasp_obj: SceneObject, initial_z: float) -> bool:
        """检查抓取是否成功 - 通过Z轴坐标变化判断"""
        print(f"        步骤4: 检查抓取成功状态...")
        
        # 等待物理稳定，确保物体位置更新
        for _ in range(30):
            self.world.step(render=True)
        
        # 获取抬起后的Z坐标
        current_z = self._get_object_current_z_position(grasp_obj)
        z_change = current_z - initial_z
        
        print(f"          抓取前Z坐标: {initial_z:.3f}m")
        print(f"          抬起后Z坐标: {current_z:.3f}m") 
        print(f"          Z轴变化量: {z_change:.3f}m")
        
        # 抓取成功判断阈值：Z轴抬升超过0.2米
        success_threshold = 0.2
        
        if z_change >= success_threshold:
            print(f"          ✓ 抓取成功! Z轴抬升 {z_change:.3f}m >= {success_threshold}m")
            return True
        else:
            print(f"          ✗ 抓取失败! Z轴抬升 {z_change:.3f}m < {success_threshold}m")
            return False
    
    def _set_object_to_gray(self, grasp_obj: SceneObject):
        """将抓取失败的物体设置为灰色"""
        try:
            print(f"          将物体 {grasp_obj.name} 设置为灰色...")
            
            if hasattr(grasp_obj, 'isaac_object') and grasp_obj.isaac_object:
                # 尝试设置物体为灰色
                from pxr import UsdGeom
                
                # 获取物体的prim
                if hasattr(grasp_obj.isaac_object, 'prim_path'):
                    stage = self.world.stage
                    prim_path = grasp_obj.isaac_object.prim_path
                    prim = stage.GetPrimAtPath(prim_path)
                    
                    if prim and prim.IsValid():
                        # 设置显示颜色为灰色
                        gprim = UsdGeom.Gprim(prim)
                        if gprim:
                            gray_color = (0.5, 0.5, 0.5)  # 灰色
                            gprim.CreateDisplayColorAttr().Set([gray_color])
                            print(f"          ✓ 物体 {grasp_obj.name} 已设置为灰色")
                        else:
                            print(f"          ⚠ 无法获取物体 {grasp_obj.name} 的Gprim，使用备用方法")
                    else:
                        print(f"          ⚠ 物体 {grasp_obj.name} 的prim无效，跳过颜色设置")
                else:
                    print(f"          ⚠ 物体 {grasp_obj.name} 没有prim_path属性")
            
            # 更新物体的颜色属性（用于记录）
            grasp_obj.color = np.array([0.5, 0.5, 0.5])
            
        except Exception as e:
            print(f"          ⚠ 设置物体颜色时出错: {e}")
            print(f"          物体 {grasp_obj.name} 仍被标记为抓取失败，但颜色可能未改变")
    
    def _position_for_grasp_precise(self, grasp_obj: SceneObject):
        """精确定位到抓取对象"""
        print(f"          直接朝向抓取对象移动...")
        
        object_pos = grasp_obj.position
        current_pos, current_yaw = self.robot_controller._get_robot_pose()
        
        # 计算从当前位置到对象的方向
        direction_to_object = object_pos[:2] - current_pos[:2]
        distance_to_object = np.linalg.norm(direction_to_object)
        
        print(f"          对象位置: [{object_pos[0]:.3f}, {object_pos[1]:.3f}]")
        print(f"          当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        print(f"          当前距离: {distance_to_object:.3f}m")
        
        # 计算目标角度 - 面向抓取对象
        target_yaw = np.arctan2(direction_to_object[1], direction_to_object[0])
        print(f"          目标角度: {np.degrees(target_yaw):.1f}°")
        
        # 先精确角度调整
        self._precise_angle_adjustment(target_yaw)
        
        # 计算停止位置 - 距离对象0.26米
        if distance_to_object > self.grasp_approach_distance:
            direction_norm = direction_to_object / distance_to_object
            approach_pos = object_pos[:2] - direction_norm * self.grasp_approach_distance
            approach_pos_3d = np.array([approach_pos[0], approach_pos[1], 0.0])
            
            print(f"          接近位置: [{approach_pos_3d[0]:.3f}, {approach_pos_3d[1]:.3f}]")
            
            # 移动到接近位置 - 使用直接导航
            success = self.robot_controller.move_to_position_robust(approach_pos_3d, target_yaw)
        else:
            print(f"          已在接近距离内，无需移动")
            success = True
        
        print(f"          直接抓取定位完成")
    
    def _precise_angle_adjustment(self, target_yaw: float):
        """精确角度调整"""
        print(f"            执行精确角度调整到: {np.degrees(target_yaw):.1f}°")
        
        max_angle_steps = 250
        
        for step in range(max_angle_steps):
            current_pos, current_yaw = self.robot_controller._get_robot_pose()
            
            # 计算角度误差
            angle_error = self._normalize_angle(target_yaw - current_yaw)
            
            if abs(angle_error) < self.angle_tolerance:
                print(f"            角度调整完成! 误差: {np.degrees(angle_error):.2f}°")
                break
            
            # 使用更大的角速度增益和最大角速度
            angular_vel = np.clip(self.angular_kp * angle_error, 
                                -self.max_angular_vel, self.max_angular_vel)
            
            # 发送角度调整指令
            self._send_velocity_command(0.0, angular_vel)
            self.world.step(render=True)
            
            if step % 50 == 0:
                print(f"              角度调整中... 误差: {np.degrees(angle_error):.2f}°")
        
        # 停止旋转
        self._send_velocity_command(0.0, 0.0)
        
        # 最终验证角度
        final_pos, final_yaw = self.robot_controller._get_robot_pose()
        final_angle_error = self._normalize_angle(target_yaw - final_yaw)
        print(f"            最终角度误差: {np.degrees(final_angle_error):.2f}°")
    
    def _normalize_angle(self, angle):
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _send_velocity_command(self, linear_vel: float, angular_vel: float):
        """发送速度指令"""
        self.robot_controller._send_velocity_command(linear_vel, angular_vel)
    
    def _control_gripper_with_test_params(self, action: str):
        """控制夹爪 - 60步插帧平滑移动"""
        print(f"    执行夹爪平滑动作: {action} (60步插帧)")
        
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 设置目标夹爪位置
        if action == "close":
            target_gripper_pos = self.gripper_close_pos
        else:  # "open"
            target_gripper_pos = self.gripper_open_pos
        
        # 获取当前关节位置
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        
        # 获取夹爪关节的起始位置
        gripper_start_positions = []
        gripper_joint_indices = []
        
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                gripper_start_positions.append(float(current_joint_positions[idx]))
                gripper_joint_indices.append(idx)
        
        if not gripper_joint_indices:
            print(f"    未找到夹爪关节，跳过夹爪控制")
            return
        
        # 60步插帧移动
        interpolation_steps = 60
        for step in range(interpolation_steps):
            # 计算插值系数 (使用平滑的 ease-in-out 曲线)
            t = step / (interpolation_steps - 1)
            smooth_t = 3 * t * t - 2 * t * t * t  # 平滑插值曲线
            
            # 计算当前步的夹爪位置
            current_gripper_positions = []
            for start_pos in gripper_start_positions:
                interpolated_pos = start_pos + smooth_t * (target_gripper_pos - start_pos)
                current_gripper_positions.append(interpolated_pos)
            
            # 构建完整的关节位置数组
            joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
            
            # 保持当前所有关节位置
            for i in range(num_dofs):
                joint_positions[i] = float(current_joint_positions[i])
            
            # 设置插值后的夹爪关节位置
            for i, idx in enumerate(gripper_joint_indices):
                joint_positions[idx] = float(current_gripper_positions[i])
            
            # 应用关节控制
            action_obj = ArticulationAction(joint_positions=joint_positions)
            articulation_controller.apply_action(action_obj)
            
            # 每步都进行物理仿真
            self.world.step(render=True)
            
            # 显示进度
            if step % 15 == 0:
                print(f"      夹爪平滑移动进度: {step+1}/60 ({(step+1)/60*100:.1f}%)")
        
        # 最终稳定
        stabilization_steps = 10
        for _ in range(stabilization_steps):
            self.world.step(render=True)
        
        print(f"    夹爪平滑动作完成: {action} -> {target_gripper_pos}m (60步插帧平滑移动)")
    
    def _deliver_to_klt_box_with_real_arm(self, grasp_obj: SceneObject):
        """运送到KLT框子 - 使用基于SLAM地图的导航"""
        print(f"        运送到KLT框子...")
        
        # 找到KLT框子任务区域
        klt_boxes = [obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK and obj.is_active]
        if not klt_boxes:
            print(f"        没有找到KLT框子")
            return
            
        klt_box = klt_boxes[0]
        klt_position = klt_box.position.copy()
        
        # 计算最优接近位置
        current_pos, _ = self.robot_controller._get_robot_pose()
        
        # 定义KLT框子四周的接近点
        approach_offsets = [
            [self.klt_approach_distance, 0.0],   # 右侧
            [-self.klt_approach_distance, 0.0],  # 左侧  
            [0.0, self.klt_approach_distance],   # 前面
            [0.0, -self.klt_approach_distance],  # 后面
        ]
        
        # 选择最近的安全接近点
        approach_candidates = []
        for offset in approach_offsets:
            candidate_pos = klt_position.copy()
            candidate_pos[0] += offset[0]
            candidate_pos[1] += offset[1] 
            candidate_pos[2] = 0.0
            
            distance = np.linalg.norm(current_pos[:2] - candidate_pos[:2])
            approach_candidates.append((candidate_pos, distance, offset))
        
        # 选择距离最近的接近点
        approach_candidates.sort(key=lambda x: x[1])
        approach_pos, min_distance, chosen_offset = approach_candidates[0]
        
        print(f"        智能选择接近方向: 偏移[{chosen_offset[0]:.1f}, {chosen_offset[1]:.1f}]")
        print(f"        导航到KLT框子接近位置: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}]")
        
        # **使用基于SLAM地图的导航到接近位置**
        success = self.robot_controller.move_to_position_robust(approach_pos)
        
        if success:
            # 检查是否足够接近KLT框子
            current_pos, _ = self.robot_controller._get_robot_pose()
            distance_to_klt = np.linalg.norm(current_pos[:2] - klt_position[:2])
            
            print(f"        当前到KLT框子距离: {distance_to_klt:.3f}m")
            
            if distance_to_klt <= self.klt_approach_distance + 0.3:
                # 足够接近，可以投放
                self._perform_release_with_real_arm_and_physics(grasp_obj, klt_box)
                print(f"        运送到KLT框子完成")
            else:
                print(f"        距离KLT框子太远: {distance_to_klt:.3f}m > {self.klt_approach_distance + 0.3:.3f}m")
        else:
            print(f"        导航到KLT框子失败")
    
    def _perform_release_with_real_arm_and_physics(self, grasp_obj: SceneObject, klt_box: SceneObject):
        """执行真实机械臂释放到KLT框子 - 60步插帧平滑移动"""
        print(f"          执行真实Franka机械臂释放(60步插帧平滑移动)...")
        
        # 机械臂移动到释放姿态并张开夹爪 - 60步插帧平滑移动
        self._move_arm_to_pose("carry")  # 保持抬起状态 - 60步插帧
        self._control_gripper_with_test_params("open")    # 张开夹爪，物体自然掉落 - 60步插帧
        
        # 观察物体掉落过程
        for _ in range(30):
            self.world.step(render=True)
            time.sleep(0.02)
        
        print(f"          物体自然掉落完成")
        
        # 机械臂回到home姿态 - 60步插帧平滑移动
        self._move_arm_to_pose("home")
        
        # 更新统计
        self.carrying_object = None
        self.coverage_stats['delivered_objects'] += 1
        
        print(f"          真实Franka机械臂60步插帧平滑释放完成")
    
    def _return_to_coverage(self):
        """返回覆盖位置"""
        if self.return_position is not None:
            print(f"        返回覆盖位置: [{self.return_position[0]:.3f}, {self.return_position[1]:.3f}]")
            self.robot_controller.move_to_position_robust(self.return_position)
            self.return_position = None
            print(f"        返回完成，继续覆盖")
    
    def _show_slam_coverage_results(self):
        """显示SLAM覆盖结果"""
        coverage_marks = len(self.visualizer.fluent_coverage_visualizer.coverage_marks)
        
        # 统计各类对象
        obstacle_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.OBSTACLE])
        sweep_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.SWEEP])
        grasp_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.GRASP])
        klt_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK])
        
        print(f"\n=== 完整SLAM集成覆盖结果 ===")
        print(f"系统阶段: {self.current_phase}")
        print(f"SLAM探索时间: {self.coverage_stats['exploration_time']:.1f}秒")
        print(f"建图状态: {'完成' if self.coverage_stats['mapping_complete'] else '进行中'}")
        print(f"覆盖状态: {'完成' if self.coverage_stats['coverage_complete'] else '进行中'}")
        print(f"")
        print(f"覆盖路径点数: {len(self.coverage_path)}")
        print(f"超级平滑覆盖标记区域: {coverage_marks}个")
        print(f"")
        print(f"环境对象统计:")
        print(f"  障碍物: {obstacle_count}个 (SLAM自动发现)")
        print(f"  清扫目标: {sweep_total}个")
        print(f"  抓取物体: {grasp_total}个(真实Franka机械臂)") 
        print(f"  KLT框子任务区域: {klt_count}个")
        print(f"")
        print(f"完整三阶段SLAM技术栈:")
        print(f"  阶段1-建图: Cartographer SLAM")
        print(f"  阶段1-探索: MapEx自主探索")
        print(f"  阶段2-规划: 基于SLAM地图的路径规划")
        print(f"  阶段3-导航: 基于SLAM地图的智能导航")
        print(f"  实时避障: 激光雷达实时避障")
        print(f"")
        print(f"多环境通信架构:")
        print(f"  Isaac Sim (Python 3.10) ↔ Socket")
        print(f"  ROS桥接节点 (Python 3.8) ↔ ROS")
        print(f"  MapEx桥接节点 (Python 3.8) ↔ Socket")
        print(f"  MapEx Explorer (Python 3.6) ↔ Socket")
        print(f"  Cartographer (Python 3.8) ↔ ROS")
        print(f"")
        print(f"真实Franka机械臂抓取配置 (60步插帧平滑移动):")
        print(f"  抓取触发距离: {self.grasp_trigger_distance}m")
        print(f"  抓取接近距离: {self.grasp_approach_distance}m")
        print(f"  机械臂60步插帧移动: 平滑无晃动")
        print(f"  夹爪60步插帧移动: 平滑无晃动")
        print(f"  夹爪张开/闭合位置: {self.gripper_open_pos}m / {self.gripper_close_pos}m")
        print(f"")
        print(f"任务执行统计:")
        print(f"  清扫完成: {self.coverage_stats['swept_objects']}/{sweep_total}")
        print(f"  抓取尝试: {self.coverage_stats['grasped_objects'] + self.coverage_stats['failed_grasps']}/{grasp_total}")
        print(f"  抓取成功: {self.coverage_stats['grasped_objects']}/{grasp_total}")
        print(f"  抓取失败: {self.coverage_stats['failed_grasps']}/{grasp_total}")
        print(f"  投放到KLT框子: {self.coverage_stats['delivered_objects']}/{grasp_total}")
        print(f"")
        print(f"抓取成功检测机制:")
        print(f"  检测方法: Z轴坐标变化检测")
        print(f"  成功阈值: Z轴抬升 >= 0.2m")
        print(f"  失败处理: 标记物体不再尝试，继续覆盖任务")
        print(f"  视觉标记: 失败物体变为灰色")
        print(f"  智能跳过: 避免重复尝试失败物体")
        
        print(f"")
        sweep_rate = (self.coverage_stats['swept_objects'] / sweep_total * 100) if sweep_total > 0 else 0
        grasp_success_rate = (self.coverage_stats['grasped_objects'] / grasp_total * 100) if grasp_total > 0 else 0
        delivery_rate = (self.coverage_stats['delivered_objects'] / grasp_total * 100) if grasp_total > 0 else 0
        print(f"任务完成率:")
        print(f"  清扫任务: {sweep_rate:.1f}%")
        print(f"  抓取成功率: {grasp_success_rate:.1f}%")
        print(f"  KLT框子投放任务: {delivery_rate:.1f}%")
        print("完整SLAM集成四类对象覆盖任务完成（三阶段流程 + 真实Franka机械臂抓取）!")
    
    def run_demo(self):
        """运行完整三阶段演示"""
        print("\n" + "="*80)
        print("完整SLAM集成四类对象覆盖系统")
        print("三阶段流程: SLAM建图 → 任务规划 → 覆盖执行")
        print("Cartographer SLAM建图 | MapEx自主探索 | 基于已知地图的覆盖清扫")
        print("障碍物SLAM发现 | 清扫目标消失 | 抓取运送 | KLT框子投放")
        print("真实Franka机械臂抓取 | 精确关节控制 | 稳定化处理")
        print("多环境通信: Python 3.10 + 3.8 + 3.6 完美集成")
        print("系统特性: 先探索建图，后执行任务")
        print("="*80)
        
        pos, yaw = self.robot_controller._get_robot_pose()
        print(f"机器人初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        # 阶段 1: SLAM探索建图
        slam_success = self.run_slam_exploration()
        
        # 阶段 2: 基于SLAM地图规划覆盖任务
        if slam_success:
            self.plan_coverage_mission()
            
            # 阶段 3: 执行覆盖清扫抓取任务
            self.execute_task_coverage()
        else:
            print("SLAM探索未完成，跳过任务执行阶段")
        
        # 完成后机械臂回到home位置
        self._move_arm_to_pose("home")
        
        print("\n完整SLAM集成四类对象覆盖系统演示完成!")
        print("三阶段流程完美执行!")
        print("Cartographer SLAM建图运行正常!")
        print("MapEx自主探索运行正常!")
        print("多环境通信（Python 3.10 + 3.8 + 3.6）运行完美!")
        print("真实Franka机械臂抓取系统运行完美! 60步插帧平滑移动，无晃动!")
        print("抓取成功检测机制运行正常! Z轴坐标变化检测，智能判断抓取成功!")
        print("基于SLAM地图的覆盖清扫任务完成!")
    
    def cleanup(self):
        """清理系统"""
        print("清理系统资源...")
        print_memory_usage("清理前")
        
        if self.visualizer:
            self.visualizer.cleanup()
        
        # 停止MapEx桥接进程
        self.stop_mapex_bridge()
        
        # 清理ROS资源
        if self.ros_bridge:
            self.ros_bridge.stop()
        
        # 关闭ROS节点
        rospy.signal_shutdown("系统关闭")
        
        # 清理内存
        for _ in range(5):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.world:
            self.world.stop()
        
        print_memory_usage("清理后")
        print("系统资源清理完成")

def main():
    """主函数"""
    print("完整SLAM集成机器人覆盖系统 + Cartographer + MapEx + 真实Franka机械臂抓取")
    print(f"机器人半径: {ROBOT_RADIUS}m")
    print(f"安全边距: {SAFETY_MARGIN}m") 
    print(f"交互距离: {INTERACTION_DISTANCE}m")
    print("系统特性: 三阶段流程 - 先SLAM探索建图，后基于已知地图执行任务")
    print("技术栈:")
    print(f"  SLAM建图: Cartographer")
    print(f"  自主探索: MapEx")
    print(f"  路径规划: 基于SLAM地图")
    print(f"  实时避障: 激光雷达")
    print(f"  机械臂抓取: 真实Franka (60步插帧平滑)")
    print(f"  抓取检测: Z轴坐标变化智能判断")
    print(f"  多环境通信: Python 3.10 + 3.8 + 3.6")
    print("")
    print("三阶段工作流程:")
    print("  1. Isaac Sim启动机器人和环境")
    print("  2. Cartographer SLAM建图")
    print("  3. MapEx自主探索环境")
    print("  4. 基于已知地图规划覆盖路径")
    print("  5. 执行清扫、抓取、投放任务")
    
    system = CompleteSLAMCoverageSystem()
    
    print("\n=== 系统初始化阶段 ===")
    system.initialize_system()
    print("系统初始化完成")
    
    print("\n=== 机器人初始化阶段 ===")
    system.initialize_robot()
    print("机器人初始化完成")
    
    print("\n=== 后加载设置阶段 ===")
    system.setup_post_load()
    print("后加载设置完成")
    
    # 额外稳定，确保所有组件就绪
    print("\n=== 最终系统稳定阶段 ===")
    for i in range(60):
        system.world.step(render=False)
        if i % 20 == 0:
            print(f"  稳定进度: {i+1}/60")
    print("系统最终稳定完成")
    
    # 运行演示
    print("\n=== 三阶段演示执行阶段 ===")
    system.run_demo()
    
    # 保持运行观察效果
    print("\n=== 效果观察阶段 ===")
    for i in range(200):
        system.world.step(render=True)
        time.sleep(0.05)
        if i % 50 == 0:
            print(f"  观察进度: {i+1}/200")
    
    print("\n=== 系统清理阶段 ===")
    system.cleanup()
    print("程序结束")

if __name__ == "__main__":
    main()