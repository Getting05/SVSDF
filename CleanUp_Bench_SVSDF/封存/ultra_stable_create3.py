#!/usr/bin/env python3
"""
Isaac Sim 4.5 四类对象真实移动覆盖算法机器人系统 - 流畅实时可视化优化版
- 障碍物(obstacles): 避障
- 清扫目标(sweep): 触碰消失
- 抓取物体(grasp): 触碰消失，运送到任务区域
- 任务区域(task): 放置抓取物体
- 支持不规则几何体碰撞检测
- 优化路径规划，避免线条贯穿障碍物
- 实现高效弓字形避障算法，确保高覆盖率
- 集成流畅实时覆盖区域可视化系统
"""

import psutil
import torch
from enum import Enum
from dataclasses import dataclass

def print_memory_usage(stage_name: str = ""):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"内存: {memory_mb:.1f}MB - {stage_name}")

# 保持原有参数不变
MAX_LINEAR_VELOCITY = 0.3     
MAX_ANGULAR_VELOCITY = 1.5     
COVERAGE_CELL_SIZE = 0.5       
COVERAGE_AREA_SIZE = 10.0       
ROBOT_RADIUS = 0.45            
PATH_TOLERANCE = 0.2          
POSITION_TOLERANCE = 0.2     
ANGLE_TOLERANCE = 0.2        
MAX_NAVIGATION_STEPS = 10000   
MAX_GHOST_ROBOTS = 10        

# 新增参数
SAFETY_MARGIN = 0.2  # 额外安全距离
INTERACTION_DISTANCE = 0.7  # 交互距离

# 流畅可视化参数（从第一个文档提取）
FINE_GRID_SIZE = 0.1          # 精细网格大小(m) - 流畅可视化
COVERAGE_UPDATE_FREQUENCY = 1  # 覆盖标记更新频率（每N步更新一次）
COVERAGE_MARK_RADIUS = 0.45    # 覆盖标记半径(m) - 实际底盘半径

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
})

import numpy as np
import math
import time
import random
from collections import deque
from typing import List, Tuple, Optional, Dict
import gc

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation as R
from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, Usd, UsdShade, Sdf
import isaacsim.core.utils.prims as prim_utils

class ObjectType(Enum):
    """对象类型枚举"""
    OBSTACLE = "obstacle"
    SWEEP = "sweep"
    GRASP = "grasp"
    TASK = "task"

@dataclass
class CollisionBoundary:
    """碰撞边界"""
    center: np.ndarray
    shape_type: str  # 'box', 'sphere', 'cylinder'
    dimensions: np.ndarray  # [width, length, height] for box, [radius] for sphere
    rotation: float = 0.0  # rotation around z-axis

@dataclass
class SceneObject:
    """场景对象"""
    name: str
    object_type: ObjectType
    position: np.ndarray
    collision_boundary: CollisionBoundary
    isaac_object: object = None
    color: np.ndarray = None
    is_active: bool = True
    original_position: np.ndarray = None

@dataclass
class CoveragePoint:
    position: np.ndarray
    orientation: float
    has_object: bool = False

class FluentCoverageVisualizer:
    """流畅覆盖区域可视化器 - 实时跟随机器人移动"""
    
    def __init__(self, world: World):
        self.world = world
        self.coverage_marks = {}  # 精细位置 -> 覆盖次数
        self.mark_prims = {}      # 精细位置 -> prim路径
        self.coverage_container = "/World/CoverageMarks"
        self.last_marked_position = None
        self.mark_counter = 0
        
        print("流畅覆盖可视化器初始化")
    
    def mark_coverage_realtime(self, robot_position: np.ndarray):
        """实时标记覆盖区域 - 流畅跟随机器人移动"""
        self.mark_counter += 1
        
        # 提高更新频率，每步都可能更新
        if self.mark_counter % COVERAGE_UPDATE_FREQUENCY != 0:
            return
            
        # 使用精细网格进行流畅标记
        fine_grid_pos = self._fine_quantize_position(robot_position)
        
        # 检查是否需要新标记
        if self._should_create_new_mark(fine_grid_pos):
            self._create_fluent_coverage_mark(fine_grid_pos)
            self.last_marked_position = fine_grid_pos.copy()
    
    def _should_create_new_mark(self, current_pos: np.ndarray) -> bool:
        """判断是否应该创建新标记"""
        if self.last_marked_position is None:
            return True
            
        # 距离上次标记位置足够远时创建新标记
        distance = np.linalg.norm(current_pos[:2] - self.last_marked_position[:2])
        return distance >= FINE_GRID_SIZE
    
    def _fine_quantize_position(self, position: np.ndarray) -> np.ndarray:
        """精细量化位置 - 更小的网格实现流畅效果"""
        # 使用精细网格，确保标记连贯流畅
        x = round(position[0] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        y = round(position[1] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        return np.array([x, y, 0.02])  # 略高于地面
    
    def _create_fluent_coverage_mark(self, position: np.ndarray):
        """创建流畅的覆盖标记"""
        stage = self.world.stage
        
        # 确保容器存在
        if not stage.GetPrimAtPath(self.coverage_container):
            stage.DefinePrim(self.coverage_container, "Xform")
        
        # 创建唯一的标记路径
        mark_id = len(self.coverage_marks)
        x_str = f"{position[0]:.2f}".replace(".", "p").replace("-", "N")
        y_str = f"{position[1]:.2f}".replace(".", "p").replace("-", "N")
        pos_key = f"{x_str}_{y_str}"
        mark_path = f"{self.coverage_container}/FluentMark_{mark_id}_{pos_key}"
        
        # 记录覆盖
        if pos_key in self.coverage_marks:
            self.coverage_marks[pos_key] += 1
            return  # 已存在标记，直接返回
        else:
            self.coverage_marks[pos_key] = 1
        
        coverage_count = self.coverage_marks[pos_key]
        
        # 创建圆形标记
        mark_prim = stage.DefinePrim(mark_path, "Cylinder")
        cylinder_geom = UsdGeom.Cylinder(mark_prim)
        
        # 设置为扁平圆盘，半径稍小于底盘半径实现更精细的标记
        mark_radius = COVERAGE_MARK_RADIUS * 0.8  # 稍小一些，更精细
        cylinder_geom.CreateRadiusAttr().Set(mark_radius)
        cylinder_geom.CreateHeightAttr().Set(0.01)  # 更薄的圆盘
        
        # 设置位置 - 修复变换操作冲突
        xform = UsdGeom.Xformable(mark_prim)
        xform.ClearXformOpOrder()  # 清除现有变换操作
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        
        # 禁用物理
        try:
            UsdPhysics.RigidBodyAPI.Apply(mark_prim)
            rigid_body = UsdPhysics.RigidBodyAPI(mark_prim)
            rigid_body.CreateRigidBodyEnabledAttr().Set(False)
        except:
            pass
        
        # 设置流畅渐变颜色
        self._set_fluent_color(mark_prim, coverage_count)
        
        # 记录标记
        self.mark_prims[pos_key] = mark_path
        
        # 流畅进度显示
        if len(self.coverage_marks) % 20 == 0:
            print(f"流畅覆盖进度: {len(self.coverage_marks)}个精细标记")
    
    def _set_fluent_color(self, mark_prim, coverage_count: int):
        """设置10档灰度渐变颜色 - 浅灰色到深灰色"""
        # 将覆盖次数限制在1-10档范围内
        coverage_level = min(coverage_count, 10)
        
        # 计算灰度值：从0.9(浅灰)到0.1(深灰)，分10档
        gray_intensity = 0.9 - (coverage_level - 1) * 0.08  # (0.9-0.1)/10 = 0.08
        
        # RGB都使用相同的灰度值
        color_value = float(gray_intensity)
        
        gprim = UsdGeom.Gprim(mark_prim)
        gprim.CreateDisplayColorAttr().Set([(color_value, color_value, color_value)])
    
    def cleanup(self):
        """清理覆盖标记"""
        stage = self.world.stage
        
        # 删除所有标记prims
        for pos_key, mark_path in self.mark_prims.items():
            if stage.GetPrimAtPath(mark_path):
                stage.RemovePrim(mark_path)
        
        # 删除容器
        if stage.GetPrimAtPath(self.coverage_container):
            stage.RemovePrim(self.coverage_container)
            
        self.coverage_marks.clear()
        self.mark_prims.clear()
        self.last_marked_position = None
        
        print("流畅覆盖标记清理完成")

class FourObjectPathPlanner:
    """四类对象路径规划器 - 高效弓字形避障优化版"""
    
    def __init__(self):
        self.world_size = COVERAGE_AREA_SIZE
        self.cell_size = COVERAGE_CELL_SIZE
        self.grid_size = int(self.world_size / self.cell_size)
        self.robot_radius = ROBOT_RADIUS
        self.safety_margin = SAFETY_MARGIN
        
        # 高分辨率障碍地图用于精确路径检查
        self.fine_grid_size = self.grid_size * 4  # 4倍精度
        self.fine_cell_size = self.world_size / self.fine_grid_size
        
        self.obstacle_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.fine_obstacle_map = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=bool)
        self.scene_objects = []
        
        print(f"高效弓字形路径规划器初始化:")
        print(f"  基础网格: {self.grid_size}x{self.grid_size}")
        print(f"  精细网格: {self.fine_grid_size}x{self.fine_grid_size}")
        print(f"  机器人半径: {self.robot_radius}m")
        print(f"  安全边距: {self.safety_margin}m")
    
    def add_scene_object(self, scene_obj: SceneObject):
        """添加场景对象"""
        self.scene_objects.append(scene_obj)
        
        # 只有障碍物需要加入碰撞地图用于路径规划避障
        if scene_obj.object_type == ObjectType.OBSTACLE:
            self._mark_collision_cells(scene_obj)
        
        print(f"添加{scene_obj.object_type.value}对象: {scene_obj.name}")
    
    def _mark_collision_cells(self, scene_obj: SceneObject):
        """标记碰撞单元格 - 双分辨率标记"""
        boundary = scene_obj.collision_boundary
        center = boundary.center
        
        # 计算扩展半径（机器人半径 + 安全边距）
        expansion = self.robot_radius + self.safety_margin
        
        # 在两种分辨率网格上标记障碍物
        if boundary.shape_type == 'sphere':
            radius = boundary.dimensions[0] + expansion
            self._mark_circular_area(center, radius, self.obstacle_map, self.grid_size, self.cell_size)
            self._mark_circular_area(center, radius, self.fine_obstacle_map, self.fine_grid_size, self.fine_cell_size)
        
        elif boundary.shape_type == 'box':
            half_w = boundary.dimensions[0] / 2 + expansion
            half_l = boundary.dimensions[1] / 2 + expansion
            self._mark_rectangular_area(center, half_w, half_l, boundary.rotation, 
                                      self.obstacle_map, self.grid_size, self.cell_size)
            self._mark_rectangular_area(center, half_w, half_l, boundary.rotation,
                                      self.fine_obstacle_map, self.fine_grid_size, self.fine_cell_size)
        
        elif boundary.shape_type == 'cylinder':
            radius = boundary.dimensions[0] + expansion
            self._mark_circular_area(center, radius, self.obstacle_map, self.grid_size, self.cell_size)
            self._mark_circular_area(center, radius, self.fine_obstacle_map, self.fine_grid_size, self.fine_cell_size)
    
    def _mark_circular_area(self, center: np.ndarray, radius: float, grid_map: np.ndarray, 
                           grid_size: int, cell_size: float):
        """标记圆形区域"""
        center_x = int((center[0] + self.world_size/2) / cell_size)
        center_y = int((center[1] + self.world_size/2) / cell_size)
        
        grid_radius = int(radius / cell_size) + 1
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                gx, gy = center_x + dx, center_y + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    world_x = (gx * cell_size) - self.world_size/2
                    world_y = (gy * cell_size) - self.world_size/2
                    
                    distance = np.linalg.norm([world_x - center[0], world_y - center[1]])
                    if distance <= radius:
                        grid_map[gx, gy] = True
    
    def _mark_rectangular_area(self, center: np.ndarray, half_w: float, half_l: float, 
                             rotation: float, grid_map: np.ndarray, grid_size: int, cell_size: float):
        """标记矩形区域（考虑旋转）"""
        center_x = int((center[0] + self.world_size/2) / cell_size)
        center_y = int((center[1] + self.world_size/2) / cell_size)
        
        # 计算包含旋转矩形的外接矩形
        max_extent = max(half_w, half_l) * 1.42  # sqrt(2) approximation
        grid_extent = int(max_extent / cell_size) + 1
        
        cos_rot = np.cos(-rotation)  # 逆旋转
        sin_rot = np.sin(-rotation)
        
        for dx in range(-grid_extent, grid_extent + 1):
            for dy in range(-grid_extent, grid_extent + 1):
                gx, gy = center_x + dx, center_y + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    world_x = (gx * cell_size) - self.world_size/2
                    world_y = (gy * cell_size) - self.world_size/2
                    
                    # 转换到对象局部坐标系
                    rel_x = world_x - center[0]
                    rel_y = world_y - center[1]
                    
                    # 旋转到对象坐标系
                    local_x = cos_rot * rel_x - sin_rot * rel_y
                    local_y = sin_rot * rel_x + cos_rot * rel_y
                    
                    # 检查是否在矩形内
                    if abs(local_x) <= half_w and abs(local_y) <= half_l:
                        grid_map[gx, gy] = True
    
    def check_collision_with_object(self, robot_pos: np.ndarray, scene_obj: SceneObject) -> bool:
        """检查机器人与对象的碰撞"""
        if not scene_obj.is_active:
            return False
            
        boundary = scene_obj.collision_boundary
        center = boundary.center
        
        distance_2d = np.linalg.norm(robot_pos[:2] - center[:2])
        
        if boundary.shape_type == 'sphere':
            return distance_2d <= (boundary.dimensions[0] + INTERACTION_DISTANCE)
        
        elif boundary.shape_type == 'box':
            # 转换到对象局部坐标系
            rel_pos = robot_pos[:2] - center[:2]
            cos_rot = np.cos(-boundary.rotation)
            sin_rot = np.sin(-boundary.rotation)
            
            local_x = cos_rot * rel_pos[0] - sin_rot * rel_pos[1]
            local_y = sin_rot * rel_pos[0] + cos_rot * rel_pos[1]
            
            half_w = boundary.dimensions[0] / 2 + INTERACTION_DISTANCE
            half_l = boundary.dimensions[1] / 2 + INTERACTION_DISTANCE
            
            return abs(local_x) <= half_w and abs(local_y) <= half_l
        
        elif boundary.shape_type == 'cylinder':
            return distance_2d <= (boundary.dimensions[0] + INTERACTION_DISTANCE)
        
        return False
    
    def generate_coverage_path(self, start_pos: np.ndarray) -> List[CoveragePoint]:
        """生成高效弓字形覆盖路径"""
        print("生成高效弓字形覆盖路径...")
        
        # 生成安全覆盖点网格
        coverage_points = self._generate_safe_coverage_grid()
        
        # 生成高效弓字形避障路径
        path_points = self._create_efficient_bow_pattern_path(coverage_points, start_pos)
        
        # 路径连通性验证和修复
        validated_path = self._validate_and_fix_path_connectivity(path_points)
        
        print(f"弓字形覆盖路径生成完成: {len(validated_path)}个点")
        return validated_path
    
    def _generate_safe_coverage_grid(self) -> List[Tuple[int, int]]:
        """生成安全可覆盖网格点"""
        coverage_points = []
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # 基本障碍检查
                if self.obstacle_map[x, y]:
                    continue
                
                # 边界安全检查
                if not self._is_safe_boundary_point(x, y):
                    continue
                
                # 机器人占地空间检查
                if not self._is_robot_space_clear(x, y):
                    continue
                
                coverage_points.append((x, y))
        
        print(f"生成安全覆盖点: {len(coverage_points)}个")
        return coverage_points
    
    def _is_safe_boundary_point(self, grid_x: int, grid_y: int) -> bool:
        """检查网格点是否距离边界足够远"""
        world_x = (grid_x * self.cell_size) - self.world_size/2
        world_y = (grid_y * self.cell_size) - self.world_size/2
        
        # 确保距离边界足够远
        margin = self.robot_radius + self.safety_margin + 0.1
        return (abs(world_x) < self.world_size/2 - margin and 
                abs(world_y) < self.world_size/2 - margin)
    
    def _is_robot_space_clear(self, center_x: int, center_y: int) -> bool:
        """检查机器人占地空间是否完全清空"""
        # 计算机器人在网格中的占地范围
        robot_grid_radius = max(1, int((self.robot_radius + self.safety_margin * 0.5) / self.cell_size))
        
        # 检查机器人周围网格是否有障碍
        for dx in range(-robot_grid_radius, robot_grid_radius + 1):
            for dy in range(-robot_grid_radius, robot_grid_radius + 1):
                check_x, check_y = center_x + dx, center_y + dy
                
                # 边界检查
                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    continue
                
                # 计算实际距离
                world_center_x = (center_x * self.cell_size) - self.world_size/2
                world_center_y = (center_y * self.cell_size) - self.world_size/2
                world_check_x = (check_x * self.cell_size) - self.world_size/2
                world_check_y = (check_y * self.cell_size) - self.world_size/2
                
                distance = np.sqrt((world_check_x - world_center_x)**2 + 
                                 (world_check_y - world_center_y)**2)
                
                # 在机器人半径内且有障碍物
                if distance <= self.robot_radius + self.safety_margin * 0.5 and self.obstacle_map[check_x, check_y]:
                    return False
        
        return True
    
    def _create_efficient_bow_pattern_path(self, coverage_points: List[Tuple[int, int]], 
                                         start_pos: np.ndarray) -> List[CoveragePoint]:
        """创建高效弓字形避障路径 - 核心算法"""
        print("创建高效弓字形避障路径...")
        
        path_points = []
        
        # 按Y坐标分组形成行
        rows = {}
        for x, y in coverage_points:
            if y not in rows:
                rows[y] = []
            rows[y].append(x)
        
        # 按Y坐标排序行
        sorted_row_keys = sorted(rows.keys())
        
        # 处理每一行，实现弓字形模式
        for row_idx, y in enumerate(sorted_row_keys):
            row_x_coords = sorted(rows[y])
            
            # 奇数行反向形成弓字形
            if row_idx % 2 == 1:
                row_x_coords.reverse()
            
            # 找到行中的连续段，跳过障碍物区间
            continuous_segments = self._find_continuous_segments(row_x_coords, y)
            
            # 对每个连续段生成路径点
            for segment in continuous_segments:
                segment_points = self._create_segment_path(segment, y, row_idx % 2 == 1)
                
                # 检查与上一个点的连通性
                if path_points and segment_points:
                    connection_points = self._create_safe_connection(path_points[-1], segment_points[0])
                    path_points.extend(connection_points)
                
                path_points.extend(segment_points)
        
        print(f"弓字形路径创建完成: {len(path_points)}个点")
        return path_points
    
    def _find_continuous_segments(self, x_coords: List[int], y: int) -> List[List[int]]:
        """找到行中的连续可达段"""
        segments = []
        current_segment = []
        
        for i, x in enumerate(x_coords):
            current_segment.append(x)
            
            # 检查是否应该结束当前段
            should_break = False
            
            if i < len(x_coords) - 1:
                next_x = x_coords[i + 1]
                # 如果下一个点距离太远，或者路径被障碍物阻断
                if abs(next_x - x) > 2 or not self._is_segment_connection_safe(x, next_x, y):
                    should_break = True
            else:
                # 最后一个点
                should_break = True
            
            if should_break:
                if len(current_segment) >= 1:  # 至少有一个点的段才有效
                    segments.append(current_segment.copy())
                current_segment = []
        
        return segments
    
    def _is_segment_connection_safe(self, x1: int, x2: int, y: int) -> bool:
        """检查段内连接是否安全"""
        # 检查两点间是否有障碍物阻挡
        start_x, end_x = min(x1, x2), max(x1, x2)
        
        for check_x in range(start_x, end_x + 1):
            if self.obstacle_map[check_x, y]:
                return False
        
        return True
    
    def _create_segment_path(self, segment: List[int], y: int, reverse: bool) -> List[CoveragePoint]:
        """为连续段创建路径点"""
        points = []
        
        for x in segment:
            world_pos = self._grid_to_world(x, y)
            
            # 计算朝向
            if reverse:
                orientation = math.pi  # 反向
            else:
                orientation = 0.0  # 正向
            
            # 检查是否有目标对象
            has_object = self._has_nearby_target_object(world_pos)
            
            point = CoveragePoint(
                position=world_pos,
                orientation=orientation,
                has_object=has_object
            )
            
            points.append(point)
        
        return points
    
    def _create_safe_connection(self, from_point: CoveragePoint, to_point: CoveragePoint) -> List[CoveragePoint]:
        """创建两点间的安全连接路径"""
        connection_points = []
        
        # 检查直线连接是否安全
        if self._is_path_safe_high_res(from_point.position, to_point.position):
            return connection_points  # 直接连接，不需要中间点
        
        # 需要绕行，使用90度转弯策略
        mid_points = self._find_90_degree_detour(from_point.position, to_point.position)
        
        for mid_pos in mid_points:
            mid_point = CoveragePoint(
                position=mid_pos,
                orientation=np.arctan2(to_point.position[1] - mid_pos[1], 
                                     to_point.position[0] - mid_pos[0]),
                has_object=False
            )
            connection_points.append(mid_point)
        
        return connection_points
    
    def _find_90_degree_detour(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        """找到90度转弯的绕行路径"""
        detour_points = []
        
        # 尝试L形路径：先水平后垂直，或先垂直后水平
        option1 = np.array([end_pos[0], start_pos[1], start_pos[2]])  # 先水平
        option2 = np.array([start_pos[0], end_pos[1], start_pos[2]])  # 先垂直
        
        # 检查两种L形路径的安全性
        for option in [option1, option2]:
            if (self._is_path_safe_high_res(start_pos, option) and 
                self._is_path_safe_high_res(option, end_pos) and
                self._is_position_safe_high_res(option)):
                detour_points.append(option)
                break
        
        return detour_points
    
    def _is_path_safe_high_res(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """使用高分辨率网格检查路径安全性"""
        distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
        steps = max(1, int(distance / (self.fine_cell_size * 0.5)))
        
        for i in range(steps + 1):
            t = i / steps
            check_pos = start_pos + t * (end_pos - start_pos)
            
            if not self._is_position_safe_high_res(check_pos):
                return False
        
        return True
    
    def _is_position_safe_high_res(self, position: np.ndarray) -> bool:
        """使用高分辨率网格检查位置安全性"""
        # 转换到精细网格坐标
        fine_x = int((position[0] + self.world_size/2) / self.fine_cell_size)
        fine_y = int((position[1] + self.world_size/2) / self.fine_cell_size)
        
        # 边界检查
        if not (0 <= fine_x < self.fine_grid_size and 0 <= fine_y < self.fine_grid_size):
            return False
        
        # 检查机器人占地空间
        robot_fine_radius = max(1, int((self.robot_radius + self.safety_margin * 0.5) / self.fine_cell_size))
        
        for dx in range(-robot_fine_radius, robot_fine_radius + 1):
            for dy in range(-robot_fine_radius, robot_fine_radius + 1):
                check_x, check_y = fine_x + dx, fine_y + dy
                
                if not (0 <= check_x < self.fine_grid_size and 0 <= check_y < self.fine_grid_size):
                    continue
                
                # 计算实际距离
                world_center_x = (fine_x * self.fine_cell_size) - self.world_size/2
                world_center_y = (fine_y * self.fine_cell_size) - self.world_size/2
                world_check_x = (check_x * self.fine_cell_size) - self.world_size/2
                world_check_y = (check_y * self.fine_cell_size) - self.world_size/2
                
                distance = np.sqrt((world_check_x - world_center_x)**2 + 
                                 (world_check_y - world_center_y)**2)
                
                if distance <= self.robot_radius + self.safety_margin * 0.5:
                    if self.fine_obstacle_map[check_x, check_y]:
                        return False
        
        return True
    
    def _validate_and_fix_path_connectivity(self, path_points: List[CoveragePoint]) -> List[CoveragePoint]:
        """验证和修复路径连通性"""
        if len(path_points) <= 1:
            return path_points
        
        validated_path = [path_points[0]]
        
        for i in range(1, len(path_points)):
            current = validated_path[-1]
            next_point = path_points[i]
            
            # 检查连接是否安全
            if self._is_path_safe_high_res(current.position, next_point.position):
                validated_path.append(next_point)
            else:
                # 需要修复连接
                repair_points = self._repair_connection(current, next_point)
                validated_path.extend(repair_points)
                validated_path.append(next_point)
        
        print(f"路径验证完成: 原{len(path_points)}点 -> 验证{len(validated_path)}点")
        return validated_path
    
    def _repair_connection(self, from_point: CoveragePoint, to_point: CoveragePoint) -> List[CoveragePoint]:
        """修复两点间的连接"""
        # 使用A*算法寻找安全路径
        repair_path = self._find_safe_path_astar(from_point.position, to_point.position)
        
        repair_points = []
        for pos in repair_path:
            repair_point = CoveragePoint(
                position=pos,
                orientation=np.arctan2(to_point.position[1] - pos[1], 
                                     to_point.position[0] - pos[0]),
                has_object=False
            )
            repair_points.append(repair_point)
        
        return repair_points
    
    def _find_safe_path_astar(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """使用简化A*算法寻找安全路径"""
        # 转换到网格坐标
        start_grid = (int((start_pos[0] + self.world_size/2) / self.cell_size),
                     int((start_pos[1] + self.world_size/2) / self.cell_size))
        goal_grid = (int((goal_pos[0] + self.world_size/2) / self.cell_size),
                    int((goal_pos[1] + self.world_size/2) / self.cell_size))
        
        # 简化的A*实现，优先90度移动
        open_set = [(0, start_grid, [])]
        closed_set = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上右下左，90度移动优先
        
        while open_set:
            open_set.sort(key=lambda x: x[0])
            cost, current, path = open_set.pop(0)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            if current == goal_grid:
                # 找到路径，转换回世界坐标
                world_path = []
                for grid_pos in path:
                    world_pos = self._grid_to_world(grid_pos[0], grid_pos[1])
                    world_path.append(world_pos)
                return world_path
            
            # 探索邻居
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor not in closed_set and 
                    0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and
                    not self.obstacle_map[neighbor[0], neighbor[1]]):
                    
                    new_cost = cost + 1
                    heuristic = abs(neighbor[0] - goal_grid[0]) + abs(neighbor[1] - goal_grid[1])
                    total_cost = new_cost + heuristic
                    
                    new_path = path + [current]
                    open_set.append((total_cost, neighbor, new_path))
        
        # 如果找不到路径，返回空列表
        return []
    
    def _has_nearby_target_object(self, position: np.ndarray) -> bool:
        """检查附近是否有目标对象"""
        for obj in self.scene_objects:
            if obj.object_type in [ObjectType.SWEEP, ObjectType.GRASP] and obj.is_active:
                distance = np.linalg.norm(position[:2] - obj.position[:2])
                if distance < self.robot_radius * 2:
                    return True
        return False
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """网格坐标转世界坐标"""
        x = (grid_x * self.cell_size) - self.world_size/2
        y = (grid_y * self.cell_size) - self.world_size/2
        return np.array([x, y, 0.0])
    
    def get_active_objects_by_type(self, obj_type: ObjectType) -> List[SceneObject]:
        """获取指定类型的活跃对象"""
        return [obj for obj in self.scene_objects if obj.object_type == obj_type and obj.is_active]

class RealMovementVisualizer:
    """真实移动可视化器 - 集成流畅覆盖可视化"""
    
    def __init__(self, world: World):
        self.world = world
        self.stage = world.stage
        
        # 使用不同的USD资产
        self.real_robot_usd = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_robot_usd = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"
        
        # 容器路径
        self.path_container = "/World/CompletePath"
        self.ghost_container = "/World/SyncGhosts" 
        
        # 状态变量
        self.all_path_points = []
        self.ghost_robots = []
        
        # 集成流畅覆盖可视化器
        self.fluent_coverage_visualizer = FluentCoverageVisualizer(world)
        
        print("初始化真实移动可视化器（集成流畅覆盖）")
    
    def setup_complete_path_visualization(self, path_points: List[CoveragePoint]):
        """设置完整路径可视化 - 优化版，避免线条贯穿障碍物"""
        self.all_path_points = path_points
        print(f"设置优化路径可视化: {len(path_points)}个点")
        
        # 清理旧可视化
        self._cleanup_all()
        
        # 创建分段路径线条，避免贯穿
        self._create_segmented_path_lines()
        
        # 创建虚影机器人
        self._create_ghost_robots()
        
        print("优化路径可视化设置完成")
    
    def _create_segmented_path_lines(self):
        """创建分段路径线条，避免贯穿障碍物"""
        self.stage.DefinePrim(self.path_container, "Xform")
        
        if len(self.all_path_points) < 2:
            return
        
        # 将路径分段，每段确保不贯穿障碍物
        path_segments = self._identify_safe_path_segments()
        
        # 为每个安全段创建线条
        for seg_idx, segment in enumerate(path_segments):
            if len(segment) < 2:
                continue
                
            line_path = f"{self.path_container}/PathSegment_{seg_idx}"
            line_prim = self.stage.DefinePrim(line_path, "BasisCurves")
            line_geom = UsdGeom.BasisCurves(line_prim)
            
            line_geom.CreateTypeAttr().Set("linear")
            line_geom.CreateBasisAttr().Set("bspline")
            
            # 构建段内路径点
            segment_points = []
            for point_idx in segment:
                point = self.all_path_points[point_idx]
                world_pos = Gf.Vec3f(float(point.position[0]), float(point.position[1]), 0.05)
                segment_points.append(world_pos)
            
            line_geom.CreatePointsAttr().Set(segment_points)
            line_geom.CreateCurveVertexCountsAttr().Set([len(segment_points)])
            line_geom.CreateWidthsAttr().Set([0.06] * len(segment_points))
            
            # 设置路径材质
            self._setup_path_material(line_prim)
        
        print(f"创建分段路径线条: {len(path_segments)}段")
    
    def _identify_safe_path_segments(self) -> List[List[int]]:
        """识别安全的路径段，避免贯穿障碍物"""
        segments = []
        current_segment = [0]  # 从第一个点开始
        
        for i in range(1, len(self.all_path_points)):
            prev_point = self.all_path_points[i-1]
            curr_point = self.all_path_points[i]
            
            # 检查两点间连接是否安全（简化检查）
            if self._is_connection_visually_safe(prev_point.position, curr_point.position):
                current_segment.append(i)
            else:
                # 结束当前段，开始新段
                if len(current_segment) >= 2:
                    segments.append(current_segment.copy())
                current_segment = [i-1, i]  # 新段从上一个点开始
        
        # 添加最后一段
        if len(current_segment) >= 2:
            segments.append(current_segment)
        
        return segments
    
    def _is_connection_visually_safe(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """检查两点连线是否视觉安全（简化版，主要检查距离）"""
        distance = np.linalg.norm(pos2[:2] - pos1[:2])
        
        # 如果距离太大，可能跨越了障碍物
        if distance > COVERAGE_CELL_SIZE * 3:
            return False
        
        # 检查中点是否远离已知障碍物（简化检查）
        mid_point = (pos1 + pos2) / 2
        
        # 这里可以添加更复杂的障碍物检查，但为了性能简化处理
        return True
    
    def _create_ghost_robots(self):
        """创建虚影机器人"""
        self.stage.DefinePrim(self.ghost_container, "Xform")
        
        # 选择关键位置放置虚影机器人
        ghost_indices = self._select_ghost_positions()
        
        for i, point_idx in enumerate(ghost_indices):
            point = self.all_path_points[point_idx]
            ghost_path = f"{self.ghost_container}/GhostRobot_{i}"
            
            # 创建虚影机器人
            ghost_prim = self.stage.DefinePrim(ghost_path, "Xform")
            references = ghost_prim.GetReferences()
            references.AddReference(self.ghost_robot_usd)
            
            # 设置位置和朝向
            position = Gf.Vec3d(float(point.position[0]), float(point.position[1]), 0.0)
            yaw_degrees = float(np.degrees(point.orientation))
            
            xform = UsdGeom.Xformable(ghost_prim)
            xform.ClearXformOpOrder()
            
            translate_op = xform.AddTranslateOp()
            translate_op.Set(position)
            
            if abs(yaw_degrees) > 1.0:
                rotate_op = xform.AddRotateZOp()
                rotate_op.Set(yaw_degrees)
            
            # 禁用物理
            self._disable_ghost_physics_safe(ghost_prim)
            
            # 设置半透明绿色
            self._set_ghost_material(ghost_prim, 0.4)
            
            self.ghost_robots.append(ghost_path)
        
        print(f"创建虚影机器人: {len(ghost_indices)}个")
    
    def _select_ghost_positions(self) -> List[int]:
        """智能选择虚影位置"""
        total_points = len(self.all_path_points)
        max_ghosts = min(MAX_GHOST_ROBOTS, total_points)
        
        if total_points <= max_ghosts:
            return list(range(total_points))
        
        selected_indices = []
        
        # 必须包含起点和终点
        selected_indices.append(0)
        if total_points > 1:
            selected_indices.append(total_points - 1)
        
        # 优先选择有物体的点
        object_indices = [i for i, p in enumerate(self.all_path_points) if p.has_object]
        for idx in object_indices[:max_ghosts//2]:
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        # 均匀选择其他点
        remaining_slots = max_ghosts - len(selected_indices)
        if remaining_slots > 0:
            step = max(1, total_points // remaining_slots)
            
            for i in range(step, total_points, step):
                if len(selected_indices) >= max_ghosts:
                    break
                if i not in selected_indices:
                    selected_indices.append(i)
        
        return sorted(selected_indices)
    
    def mark_coverage_realtime(self, robot_position: np.ndarray, step_count: int):
        """实时标记覆盖区域 - 使用流畅覆盖可视化器"""
        # 调用集成的流畅覆盖可视化器
        self.fluent_coverage_visualizer.mark_coverage_realtime(robot_position)
    
    def _disable_ghost_physics_safe(self, root_prim):
        """安全禁用虚影物理属性"""
        def disable_recursive(prim):
            try:
                if prim.IsA(UsdGeom.Xformable):
                    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                    
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(prim)
                    collision = UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
            except:
                pass
            
            for child in prim.GetChildren():
                disable_recursive(child)
        
        disable_recursive(root_prim)
    
    def _set_ghost_material(self, ghost_prim, opacity: float):
        """设置虚影材质"""
        material_path = f"/World/Materials/GhostMaterial_{hash(str(ghost_prim.GetPath())) % 1000}"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.9, 0.2))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 0.1))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        # 应用到mesh
        def apply_material_recursive(prim):
            if prim.GetTypeName() == "Mesh":
                UsdShade.MaterialBindingAPI(prim).Bind(material)
            for child in prim.GetChildren():
                apply_material_recursive(child)
        
        apply_material_recursive(ghost_prim)
    
    def _setup_path_material(self, line_prim):
        """设置路径材质"""
        material_path = "/World/Materials/PathMaterial"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 1.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.05, 0.25, 0.5))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        UsdShade.MaterialBindingAPI(line_prim).Bind(material)
    
    def _cleanup_all(self):
        """清理所有可视化"""
        containers = [self.path_container, self.ghost_container]
        for container in containers:
            if self.stage.GetPrimAtPath(container):
                self.stage.RemovePrim(container)
        
        # 清理材质
        materials_path = "/World/Materials"
        if self.stage.GetPrimAtPath(materials_path):
            materials_prim = self.stage.GetPrimAtPath(materials_path)
            for child in materials_prim.GetChildren():
                if any(name in str(child.GetPath()) for name in ["PathMaterial", "GhostMaterial"]):
                    self.stage.RemovePrim(child.GetPath())
    
    def cleanup(self):
        """清理资源"""
        self._cleanup_all()
        # 清理流畅覆盖可视化器
        if self.fluent_coverage_visualizer:
            self.fluent_coverage_visualizer.cleanup()
        print("可视化资源清理完成")

class FixedRobotController:
    """修复版机器人控制器"""
    
    def __init__(self, mobile_base, world):
        self.mobile_base = mobile_base
        self.world = world
        self.max_linear_vel = MAX_LINEAR_VELOCITY
        self.max_angular_vel = MAX_ANGULAR_VELOCITY
        
        # 控制参数
        self.linear_kp = 3.0
        self.angular_kp = 4.0
        
        # 速度平滑
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.smooth_factor = 0.8
        
        print("修复版机器人控制器初始化")
        print(f"线速度上限: {self.max_linear_vel}m/s")
        print(f"角速度上限: {self.max_angular_vel}rad/s")
    
    def move_to_position_robust(self, target_pos: np.ndarray, target_orientation: float = 0.0) -> bool:
        """鲁棒的移动到位置方法"""
        print(f"开始导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        max_steps = MAX_NAVIGATION_STEPS
        step_count = 0
        stuck_counter = 0
        prev_position = None
        
        # 记录初始位置用于检测是否有进展
        start_pos = self._get_robot_pose()[0]
        start_distance = np.linalg.norm(start_pos[:2] - target_pos[:2])
        
        while step_count < max_steps:
            current_pos, current_yaw = self._get_robot_pose()
            step_count += 1
            
            # 检查是否到达
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            if pos_error < POSITION_TOLERANCE:
                self._send_zero_velocity()
                print(f"  成功到达! 误差: {pos_error:.3f}m, 步数: {step_count}")
                return True
            
            # 检测卡住状态
            if prev_position is not None:
                movement = np.linalg.norm(current_pos[:2] - prev_position[:2])
                if movement < 0.001:
                    stuck_counter += 1
                    if stuck_counter > 600:
                        print(f"  检测到卡住状态，尝试脱困...")
                        self._unstuck_maneuver()
                        stuck_counter = 0
                else:
                    stuck_counter = 0
            
            prev_position = current_pos.copy()
            
            # 计算控制指令
            self._compute_and_send_control(current_pos, current_yaw, target_pos, target_orientation)
            
            # 步进物理仿真
            try:
                self.world.step(render=True)
            except Exception:
                break
            
            # 周期性进度报告
            if step_count % 500 == 0:
                progress = max(0, (start_distance - pos_error) / start_distance * 100)
                print(f"  导航中... 距离: {pos_error:.3f}m, 进度: {progress:.1f}%, 步数: {step_count}")
        
        # 导航结束
        self._send_zero_velocity()
        final_pos, _ = self._get_robot_pose()
        final_error = np.linalg.norm(final_pos[:2] - target_pos[:2])
        
        if final_error < POSITION_TOLERANCE * 1.5:
            print(f"  接近成功! 最终误差: {final_error:.3f}m")
            return True
        else:
            print(f"  导航失败! 最终误差: {final_error:.3f}m, 用时: {step_count}步")
            return False
    
    def _compute_and_send_control(self, current_pos: np.ndarray, current_yaw: float, 
                                  target_pos: np.ndarray, target_yaw: float):
        """计算并发送控制指令"""
        try:
            # 计算目标方向
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
                # 需要转向
                linear_vel = 0.0
                angular_vel = np.clip(self.angular_kp * yaw_error, 
                                    -self.max_angular_vel, self.max_angular_vel)
            else:
                # 可以前进
                linear_vel = np.clip(self.linear_kp * distance, 
                                   0.0, self.max_linear_vel)
                # 保持轻微转向修正
                angular_vel = np.clip(self.angular_kp * yaw_error * 0.5, 
                                    -self.max_angular_vel, self.max_angular_vel)
            
            # 平滑速度变化
            linear_vel = self._smooth_velocity(linear_vel, self.prev_linear)
            angular_vel = self._smooth_velocity(angular_vel, self.prev_angular)
            
            self.prev_linear = linear_vel
            self.prev_angular = angular_vel
            
            # 发送控制指令
            self._send_velocity_command(linear_vel, angular_vel)
            
        except Exception:
            pass
    
    def _smooth_velocity(self, new_vel: float, prev_vel: float) -> float:
        """平滑速度变化"""
        return self.smooth_factor * prev_vel + (1 - self.smooth_factor) * new_vel
    
    def _unstuck_maneuver(self):
        """脱困机动"""
        print("    执行脱困机动...")
        
        # 后退
        for _ in range(50):
            self._send_velocity_command(-self.max_linear_vel * 0.3, 0.0)
            self.world.step(render=True)
        
        # 转向
        for _ in range(60):
            self._send_velocity_command(0.0, self.max_angular_vel * 0.5)
            self.world.step(render=True)
        
        # 停止
        self._send_zero_velocity()
        for _ in range(10):
            self.world.step(render=True)
        
        print("    脱困机动完成")
    
    def _send_velocity_command(self, linear_vel: float, angular_vel: float):
        """发送速度指令到轮子"""
        try:
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
            if "left_wheel_joint" in self.mobile_base.dof_names:
                left_idx = self.mobile_base.dof_names.index("left_wheel_joint")
                joint_velocities[left_idx] = float(left_wheel_vel)
            
            if "right_wheel_joint" in self.mobile_base.dof_names:
                right_idx = self.mobile_base.dof_names.index("right_wheel_joint")
                joint_velocities[right_idx] = float(right_wheel_vel)
            
            # 应用控制指令
            action = ArticulationAction(joint_velocities=joint_velocities)
            articulation_controller.apply_action(action)
            
        except Exception:
            pass
    
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
        try:
            position, orientation = self.mobile_base.get_world_pose()
            position = np.array(position)
            
            # 四元数转欧拉角
            quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
            r = R.from_quat(quat)
            yaw = r.as_euler('xyz')[2]
            
            return position, yaw
        except Exception:
            return np.array([0.0, 0.0, 0.0]), 0.0

class FourObjectCoverageSystem:
    """四类对象覆盖系统"""
    
    def __init__(self):
        self.world = None
        self.mobile_base = None
        self.robot_controller = None
        self.path_planner = None
        self.visualizer = None
        
        self.scene_objects = []
        self.coverage_path = []
        self.coverage_stats = {
            'swept_objects': 0,
            'grasped_objects': 0,
            'delivered_objects': 0,
            'total_coverage_points': 0
        }
        
        # 机械臂配置
        self.arm_poses = {
            "home": [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.785],
            "pickup": [0.0, 0.3, 0.0, -1.5, 0.0, 2.2, 0.785],
            "carry": [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.785]
        }
        
        # 抓取状态
        self.carrying_object = None
        self.return_position = None
    
    def initialize_system(self):
        """初始化系统"""
        print("初始化四类对象覆盖系统（流畅实时可视化）...")
        
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
        self._create_four_type_environment()
        
        print("四类对象系统初始化完成（已集成流畅实时覆盖可视化）")
        return True
    
    def _setup_lighting(self):
        """设置照明"""
        try:
            main_light = prim_utils.create_prim("/World/MainLight", "DistantLight")
            distant_light = UsdLux.DistantLight(main_light)
            distant_light.CreateIntensityAttr(4000)
            distant_light.CreateColorAttr((1.0, 1.0, 0.95))
            
            env_light = prim_utils.create_prim("/World/EnvLight", "DomeLight")
            dome_light = UsdLux.DomeLight(env_light)
            dome_light.CreateIntensityAttr(800)
            dome_light.CreateColorAttr((0.8, 0.9, 1.0))
            print("照明设置完成")
        except Exception as e:
            print(f"照明设置错误: {e}")
    
    def _initialize_components(self):
        """初始化组件"""
        self.path_planner = FourObjectPathPlanner()
        self.visualizer = RealMovementVisualizer(self.world)  # 已集成流畅覆盖可视化
        print("组件初始化完成（含流畅实时覆盖可视化）")
    
    def _create_four_type_environment(self):
        """创建四类对象环境"""
        print("创建四类对象环境...")
        
        # 1. 障碍物 (红色) - 路径规划避障
        obstacles_config = [
            {"pos": [1.2, 0.8, 0.15], "size": [0.6, 0.4, 0.3], "color": [0.8, 0.2, 0.2], "shape": "box"},
            {"pos": [0.5, -1.5, 0.2], "size": [1.2, 0.3, 0.4], "color": [0.8, 0.1, 0.1], "shape": "box"},
            {"pos": [-1.0, 1.2, 0.25], "size": [0.5], "color": [0.7, 0.2, 0.2], "shape": "sphere"},
        ]
        
        # 2. 清扫目标 (黄色) - 触碰消失
        sweep_config = [
            {"pos": [0.8, 0.2, 0.05], "size": [0.1], "color": [1.0, 1.0, 0.2], "shape": "sphere"},
            {"pos": [1.5, 1.5, 0.05], "size": [0.1], "color": [0.9, 0.9, 0.1], "shape": "sphere"},
            {"pos": [-0.8, -0.8, 0.05], "size": [0.1], "color": [1.0, 0.8, 0.0], "shape": "sphere"},
            {"pos": [2.0, 0.5, 0.05], "size": [0.1], "color": [0.8, 0.8, 0.2], "shape": "sphere"},
        ]
        
        # 3. 抓取物体 (绿色) - 运送到任务区域
        grasp_config = [
            {"pos": [1.8, -1.2, 0.08], "size": [0.16, 0.16, 0.16], "color": [0.2, 0.8, 0.2], "shape": "box"},
            {"pos": [-1.5, 0.5, 0.08], "size": [0.16, 0.16, 0.16], "color": [0.1, 0.9, 0.1], "shape": "box"},
            {"pos": [0.3, 1.8, 0.08], "size": [0.16, 0.16, 0.16], "color": [0.0, 0.8, 0.0], "shape": "box"},
        ]
        
        # 4. 任务区域 (蓝色) - 放置区域
        task_config = [
            {"pos": [-2.2, -2.2, 0], "size": [0.8, 0.8, 0.2], "color": [0.2, 0.2, 0.8], "shape": "box"},
        ]
        
        # 创建所有对象
        all_configs = [
            (obstacles_config, ObjectType.OBSTACLE),
            (sweep_config, ObjectType.SWEEP),
            (grasp_config, ObjectType.GRASP),
            (task_config, ObjectType.TASK)
        ]
        
        for config_list, obj_type in all_configs:
            for i, config in enumerate(config_list):
                self._create_scene_object(config, obj_type, i)
        
        print(f"四类对象环境创建完成:")
        print(f"  障碍物: {len(obstacles_config)}个")
        print(f"  清扫目标: {len(sweep_config)}个") 
        print(f"  抓取物体: {len(grasp_config)}个")
        print(f"  任务区域: {len(task_config)}个")
    
    def _create_scene_object(self, config: Dict, obj_type: ObjectType, index: int):
        """创建场景对象"""
        try:
            name = f"{obj_type.value}_{index}"
            prim_path = f"/World/{name}"
            
            # 创建Isaac对象
            if config["shape"] == "sphere":
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
                # 对于障碍物和任务区域，使用FixedCuboid以避免物理问题
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
            
            self.world.scene.add(isaac_obj)
            
            # 对任务区域禁用所有物理属性
            if obj_type == ObjectType.TASK:
                self._disable_task_physics(isaac_obj)
            
            # 创建场景对象
            scene_obj = SceneObject(
                name=name,
                object_type=obj_type,
                position=np.array(config["pos"]),
                collision_boundary=collision_boundary,
                isaac_object=isaac_obj,
                color=np.array(config["color"]),
                original_position=np.array(config["pos"])
            )
            
            self.scene_objects.append(scene_obj)
            self.path_planner.add_scene_object(scene_obj)
            
            print(f"创建{obj_type.value}对象: {name}")
            
        except Exception as e:
            print(f"创建对象失败 {name}: {e}")
    
    def _disable_task_physics(self, isaac_obj):
        """禁用任务区域的物理属性，确保无物理交互"""
        try:
            # 获取物体的prim路径
            prim_path = isaac_obj.prim_path
            prim = self.world.stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                # 禁用刚体物理
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                else:
                    # 如果没有刚体API，先应用再禁用
                    UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                
                # 禁用碰撞检测
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
                else:
                    # 如果没有碰撞API，先应用再禁用
                    UsdPhysics.CollisionAPI.Apply(prim)
                    collision = UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
                
                # 递归处理子节点
                for child in prim.GetChildren():
                    self._disable_child_physics(child)
                
                print(f"已禁用任务区域物理属性: {isaac_obj.name}")
                
        except Exception as e:
            print(f"禁用任务区域物理属性失败 {isaac_obj.name}: {e}")
    
    def _disable_child_physics(self, child_prim):
        """递归禁用子节点的物理属性"""
        try:
            if child_prim.IsValid():
                # 禁用刚体物理
                if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body = UsdPhysics.RigidBodyAPI(child_prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                
                # 禁用碰撞检测
                if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(child_prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
                
                # 递归处理更深层的子节点
                for grandchild in child_prim.GetChildren():
                    self._disable_child_physics(grandchild)
                    
        except Exception as e:
            pass  # 忽略子节点处理错误
    
    def _disable_dropped_object_physics(self, isaac_obj):
        """禁用放置物体的物理属性，避免碰撞和掉落"""
        try:
            # 获取物体的prim路径
            prim_path = isaac_obj.prim_path
            prim = self.world.stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                # 禁用刚体物理（避免掉落）
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                else:
                    UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                
                # 禁用碰撞检测（避免与机器人碰撞）
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
                else:
                    UsdPhysics.CollisionAPI.Apply(prim)
                    collision = UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
                
                # 递归处理子节点
                for child in prim.GetChildren():
                    self._disable_child_physics(child)
                
                print(f"已禁用放置物体物理属性: {isaac_obj.name}")
                
        except Exception as e:
            print(f"禁用放置物体物理属性失败 {isaac_obj.name}: {e}")
    
    def _safe_disable_dropped_object_physics(self, isaac_obj):
        """安全地禁用放置物体的物理属性，避免张量视图失效"""
        try:
            # 获取物体的prim路径
            prim_path = isaac_obj.prim_path
            prim = self.world.stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                # 对于FixedCuboid，主要是确保碰撞检测被禁用
                # 不需要处理刚体物理，因为FixedCuboid本来就是静态的
                
                # 禁用碰撞检测（避免与机器人碰撞）
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                
                collision = UsdPhysics.CollisionAPI(prim)
                collision.CreateCollisionEnabledAttr().Set(False)
                
                # 递归处理子节点的碰撞
                for child in prim.GetChildren():
                    self._safe_disable_child_collision(child)
                
                print(f"已安全禁用放置物体碰撞: {isaac_obj.name}")
                
        except Exception as e:
            print(f"安全禁用放置物体物理属性失败 {isaac_obj.name}: {e}")
    
    def _safe_disable_child_collision(self, child_prim):
        """安全地禁用子节点的碰撞检测"""
        try:
            if child_prim.IsValid():
                # 只处理碰撞检测，不处理刚体物理
                if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(child_prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
                
                # 递归处理更深层的子节点
                for grandchild in child_prim.GetChildren():
                    self._safe_disable_child_collision(grandchild)
                    
        except Exception as e:
            pass  # 忽略子节点处理错误
    
    def _disable_collision_completely(self, isaac_obj):
        """完全禁用物体的碰撞属性，确保无任何碰撞交互"""
        try:
            # 获取物体的prim路径
            prim_path = isaac_obj.prim_path
            prim = self.world.stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                # 完全禁用碰撞检测
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                
                collision = UsdPhysics.CollisionAPI(prim)
                collision.CreateCollisionEnabledAttr().Set(False)
                
                # 递归处理所有子节点，确保完全无碰撞
                self._recursive_disable_collision(prim)
                
                print(f"已完全禁用物体碰撞: {isaac_obj.name}")
                
        except Exception as e:
            print(f"禁用物体碰撞失败 {isaac_obj.name}: {e}")
    
    def _recursive_disable_collision(self, prim):
        """递归禁用所有子节点的碰撞"""
        try:
            for child in prim.GetChildren():
                if child.IsValid():
                    # 禁用子节点碰撞
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(child)
                    
                    collision = UsdPhysics.CollisionAPI(child)
                    collision.CreateCollisionEnabledAttr().Set(False)
                    
                    # 继续递归处理更深层的子节点
                    self._recursive_disable_collision(child)
                    
        except Exception as e:
            pass  # 忽略处理错误
    
    def initialize_robot(self):
        """初始化机器人"""
        print("初始化机器人...")
        
        try:
            robot_usd_path = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
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
            
        except Exception as e:
            print(f"机器人初始化失败: {e}")
            return False
        
        return True
    
    def setup_post_load(self):
        """后加载设置"""
        print("后加载设置...")
        
        try:
            # 重置世界
            self.world.reset()
            print("世界重置完成")
            
            # 稳定物理 - 更多步数
            print("物理稳定中...")
            for i in range(120):  # 增加稳定步数
                self.world.step(render=False)
                if i % 30 == 0:
                    print(f"  稳定进度: {i+1}/120")
            
            # 获取机器人对象 - 添加重试机制
            print("获取机器人对象...")
            self.mobile_base = None
            for retry in range(5):  # 重试5次
                try:
                    self.mobile_base = self.world.scene.get_object("create3_robot")
                    if self.mobile_base is not None:
                        print(f"机器人对象获取成功 (尝试 {retry+1}/5)")
                        break
                    else:
                        print(f"  尝试 {retry+1}/5 失败，继续尝试...")
                        # 额外的稳定步骤
                        for _ in range(30):
                            self.world.step(render=False)
                except Exception as e:
                    print(f"  尝试 {retry+1}/5 异常: {e}")
                    continue
            
            # 验证机器人对象的关键属性
            try:
                controller = self.mobile_base.get_articulation_controller()
                print(f"机器人验证通过，DOF数量: {len(self.mobile_base.dof_names)}")
                
            except Exception as e:
                print(f"机器人对象验证失败: {e}")
                return False
            
            # 修复物理层次结构 - 更保守的方法
            self._fix_robot_physics_conservative()
            
            # 设置控制增益
            self._setup_robust_control_gains()
            
            # 移动机械臂到home位置
            self._move_arm_to_pose("home")
            
            # 初始化控制器
            try:
                self.robot_controller = FixedRobotController(self.mobile_base, self.world)
                print("控制器初始化完成")
            except Exception as e:
                print(f"控制器初始化失败: {e}")
                return False
            
            # 验证机器人状态
            try:
                pos, yaw = self.robot_controller._get_robot_pose()
                print(f"机器人状态验证: 位置[{pos[0]:.3f}, {pos[1]:.3f}], 朝向{np.degrees(yaw):.1f}°")
            except Exception as e:
                print(f"机器人状态验证警告: {e}")
                # 不返回False，允许继续
            
        except Exception as e:
            print(f"后加载设置失败: {e}")
            return False
        
        print("后加载设置完成")
        return True
    
    def _fix_robot_physics_conservative(self):
        """保守的机器人物理层次结构修复"""
        print("保守修复机器人物理层次结构...")
        
        try:
            robot_prim = self.world.stage.GetPrimAtPath("/World/create3_robot")
            
            # 只修复明确的问题轮子，避免过度修复
            problem_paths = [
                "/World/create3_robot/create_3/left_wheel/visuals_01",
                "/World/create3_robot/create_3/right_wheel/visuals_01"
            ]
            
            for wheel_path in problem_paths:
                wheel_prim = self.world.stage.GetPrimAtPath(wheel_path)
                if wheel_prim and wheel_prim.IsValid():
                    try:
                        # 仅禁用视觉元素的刚体物理
                        if wheel_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            rigid_body = UsdPhysics.RigidBodyAPI(wheel_prim)
                            rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                            print(f"  修复轮子视觉物理: {wheel_path.split('/')[-1]}")
                        
                        # 禁用碰撞但保留变换
                        if wheel_prim.HasAPI(UsdPhysics.CollisionAPI):
                            collision = UsdPhysics.CollisionAPI(wheel_prim)
                            collision.CreateCollisionEnabledAttr().Set(False)
                            
                    except Exception as e:
                        print(f"  修复轮子 {wheel_path.split('/')[-1]} 部分失败: {e}")
                        continue
            
            print("保守物理层次结构修复完成")
            
        except Exception as e:
            print(f"保守物理修复失败: {e}")
            # 不抛出异常，允许程序继续
    
    def _setup_robust_control_gains(self):
        """设置鲁棒的控制增益"""
        print("设置控制增益...")
        
        try:
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
                    print(f"  设置轮子控制: {wheel_name}")
            
            # 机械臂控制
            arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
            arm_found = 0
            for joint_name in arm_joint_names:
                if joint_name in self.mobile_base.dof_names:
                    idx = self.mobile_base.dof_names.index(joint_name)
                    kp[idx] = 800.0
                    kd[idx] = 40.0
                    arm_found += 1
            
            if arm_found > 0:
                print(f"  设置机械臂控制: {arm_found}个关节")
            
            # 夹爪控制
            gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
            gripper_found = 0
            for joint_name in gripper_names:
                if joint_name in self.mobile_base.dof_names:
                    idx = self.mobile_base.dof_names.index(joint_name)
                    kp[idx] = 1e5
                    kd[idx] = 1e3
                    gripper_found += 1
            
            if gripper_found > 0:
                print(f"  设置夹爪控制: {gripper_found}个关节")
            
            # 应用增益
            articulation_controller.set_gains(kps=kp, kds=kd)
            print("控制增益设置完成")
            
        except Exception as e:
            print(f"控制增益设置失败: {e}")
    
    def _move_arm_to_pose(self, pose_name: str):
        """移动机械臂到指定姿态"""
        
        print(f"移动机械臂到: {pose_name}")
        
        try:
            target_positions = self.arm_poses[pose_name]
            articulation_controller = self.mobile_base.get_articulation_controller()
            
            num_dofs = len(self.mobile_base.dof_names)
            joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
            
            # 设置机械臂关节位置
            arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
            set_joints = 0
            
            for i, joint_name in enumerate(arm_joint_names):
                if joint_name in self.mobile_base.dof_names:
                    idx = self.mobile_base.dof_names.index(joint_name)
                    if i < len(target_positions):
                        joint_positions[idx] = target_positions[i]
                        set_joints += 1
            
            # 应用动作
            action = ArticulationAction(joint_positions=joint_positions)
            articulation_controller.apply_action(action)
            
            # 等待机械臂移动到位
            for _ in range(30):
                self.world.step(render=False)
            
            print(f"  机械臂移动完成: {pose_name} (设置{set_joints}个关节)")
            
        except Exception as e:
            print(f"  机械臂移动失败: {e}")
    
    def plan_coverage_mission(self):
        """规划覆盖任务"""
        print("规划四类对象覆盖任务...")
        
        try:
            current_pos, _ = self.robot_controller._get_robot_pose()
            print(f"机器人当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
            
            # 生成覆盖路径
            self.coverage_path = self.path_planner.generate_coverage_path(current_pos)
            
            # 设置完整路径可视化
            self.visualizer.setup_complete_path_visualization(self.coverage_path)
            
            print(f"四类对象覆盖规划完成: {len(self.coverage_path)}个覆盖点")
            print("已集成流畅实时覆盖区域可视化系统")
            print_memory_usage("覆盖规划完成")
            
        except Exception as e:
            print(f"覆盖规划失败: {e}")
            return False
        
        return True
    
    def execute_four_object_coverage(self):
        """执行四类对象覆盖"""
        print("\n开始执行四类对象覆盖（流畅实时可视化）...")
        print_memory_usage("任务开始")
        
        # 展示路径预览
        print("展示路径预览...")
        for step in range(60):
            self.world.step(render=True)
            time.sleep(0.03)
        
        print("开始执行路径...")
        
        successful_points = 0
        step_counter = 0
        
        for i, point in enumerate(self.coverage_path):
            print(f"\n=== 导航到点 {i+1}/{len(self.coverage_path)} ===")
            
            try:
                # 鲁棒移动到目标点
                success = self.robot_controller.move_to_position_robust(point.position, point.orientation)
                
                if success:
                    successful_points += 1
                    print(f"  点 {i+1} 导航成功")
                else:
                    print(f"  点 {i+1} 导航失败，继续下一个点")
                
                # 获取当前位置
                current_pos, _ = self.robot_controller._get_robot_pose()
                
                # 实时流畅覆盖标记 - 使用集成的流畅可视化器
                self.visualizer.mark_coverage_realtime(current_pos, step_counter)
                step_counter += 1
                
                # 检查四类对象交互
                self._check_four_object_interactions(current_pos)
                
                # 短暂停顿
                for _ in range(5):
                    self.world.step(render=True)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  点 {i+1} 处理失败: {e}")
                continue
        
        print(f"\n四类对象覆盖执行完成!")
        print(f"成功到达点数: {successful_points}/{len(self.coverage_path)}")
        self._show_four_object_results()
    
    def _check_four_object_interactions(self, robot_pos: np.ndarray):
        """检查四类对象交互"""
        for scene_obj in self.scene_objects:
            if not scene_obj.is_active:
                continue
            
            # 检查碰撞
            if self.path_planner.check_collision_with_object(robot_pos, scene_obj):
                self._handle_object_interaction(scene_obj, robot_pos)
    
    def _handle_object_interaction(self, scene_obj: SceneObject, robot_pos: np.ndarray):
        """处理对象交互"""
        print(f"    交互检测: {scene_obj.name} ({scene_obj.object_type.value})")
        
        if scene_obj.object_type == ObjectType.SWEEP:
            # 清扫目标：直接消失
            self._handle_sweep_object(scene_obj)
        
        elif scene_obj.object_type == ObjectType.GRASP:
            # 抓取物体：抓取并运送到任务区域
            self._handle_grasp_object(scene_obj, robot_pos)
    
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
    
    def _handle_grasp_object(self, grasp_obj: SceneObject, robot_pos: np.ndarray):
        """处理抓取对象"""
        print(f"      抓取物体: {grasp_obj.name}")
        
        # 如果已经在运送其他物体，跳过
        if self.carrying_object is not None:
            print(f"      已在运送物体，跳过")
            return
        
        # 记录返回位置
        self.return_position = robot_pos.copy()
        
        # 执行抓取
        self._perform_grasp_sequence(grasp_obj)
        
        # 运送到任务区域
        self._deliver_to_task_area(grasp_obj)
        
        # 返回继续覆盖
        self._return_to_coverage()
    
    def _perform_grasp_sequence(self, grasp_obj: SceneObject):
        """执行抓取序列"""
        print(f"        执行抓取动作...")
        
        # 机械臂抓取动作
        self._move_arm_to_pose("pickup")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        # 隐藏原物体
        grasp_obj.isaac_object.set_visibility(False)
        grasp_obj.isaac_object.set_world_pose(
            np.array([100.0, 100.0, -5.0]), 
            np.array([0, 0, 0, 1])
        )
        
        # 标记为运送中
        self.carrying_object = grasp_obj
        grasp_obj.is_active = False
        self.coverage_stats['grasped_objects'] += 1
        
        print(f"        抓取完成")
    
    def _deliver_to_task_area(self, grasp_obj: SceneObject):
        """运送到任务区域"""
        print(f"        运送到任务区域...")
        
        # 找到任务区域
        task_areas = [obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK and obj.is_active]
        if not task_areas:
            print(f"        没有找到任务区域")
            return
            
        task_area = task_areas[0]  # 使用第一个任务区域
        target_pos = task_area.position.copy()
        target_pos[2] = 0.0  # 地面高度
        
        # 导航到任务区域
        print(f"        导航到任务区域: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        success = self.robot_controller.move_to_position_robust(target_pos)
        
        if success:
            # 放置物体
            self._perform_release_sequence(grasp_obj, task_area)
            print(f"        运送完成")
        else:
            print(f"        导航到任务区域失败")
    
    def _perform_release_sequence(self, grasp_obj: SceneObject, task_area: SceneObject):
        """执行释放序列"""
        print(f"          执行放置动作...")
        
        # 机械臂放置动作
        self._move_arm_to_pose("pickup")
        self._control_gripper("open")
        self._move_arm_to_pose("home")
        
        # 在任务区域显示物体
        drop_position = task_area.position.copy()
        # 让物体与任务区域贴合，没有高度差
        drop_position[2] = task_area.position[2]  # 与任务区域同一高度
        
        # 创建新的投放物体（显示效果）
        try:
            drop_name = f"delivered_{grasp_obj.name}"
            drop_prim_path = f"/World/{drop_name}"
            
            # 使用FixedCuboid，本身就没有动态物理属性
            drop_obj = FixedCuboid(
                prim_path=drop_prim_path,
                name=drop_name,
                position=drop_position,
                scale=grasp_obj.collision_boundary.dimensions,
                color=grasp_obj.color * 0.7  # 稍微暗一点表示已放置
            )
            
            self.world.scene.add(drop_obj)
            
            # 等待物体创建完成
            for _ in range(2):
                self.world.step(render=False)
            
            # 完全禁用碰撞属性
            self._disable_collision_completely(drop_obj)
            
            print(f"          物体放置在任务区域（无碰撞，贴合表面）")
            
        except Exception as e:
            print(f"          放置物体失败: {e}")
        
        # 更新统计
        self.carrying_object = None
        self.coverage_stats['delivered_objects'] += 1
    
    def _return_to_coverage(self):
        """返回覆盖位置"""
        if self.return_position is not None:
            print(f"        返回覆盖位置: [{self.return_position[0]:.3f}, {self.return_position[1]:.3f}]")
            self.robot_controller.move_to_position_robust(self.return_position)
            self.return_position = None
            print(f"        返回完成，继续覆盖")
    
    def _control_gripper(self, action):
        """控制夹爪"""
        try:
            gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
            
            # 检查夹爪关节是否存在
            available_gripper_joints = [name for name in gripper_names if name in self.mobile_base.dof_names]
            
            articulation_controller = self.mobile_base.get_articulation_controller()
            
            gripper_pos = 0.0 if action == "close" else 0.04
            
            num_dofs = len(self.mobile_base.dof_names)
            joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
            
            controlled_joints = 0
            for joint_name in available_gripper_joints:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = gripper_pos
                controlled_joints += 1
            
            # 应用动作
            action_obj = ArticulationAction(joint_positions=joint_positions)
            articulation_controller.apply_action(action_obj)
            
            # 等待夹爪动作完成
            for _ in range(15):
                self.world.step(render=False)
            
            print(f"    夹爪控制完成: {action} ({controlled_joints}个关节)")
                
        except Exception as e:
            print(f"    夹爪控制失败: {e}")
    
    def _show_four_object_results(self):
        """显示四类对象结果"""
        # 获取流畅覆盖标记数量
        coverage_marks = len(self.visualizer.fluent_coverage_visualizer.coverage_marks)
        
        # 统计各类对象
        obstacle_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.OBSTACLE])
        sweep_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.SWEEP])
        grasp_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.GRASP])
        task_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK])
        
        print(f"\n=== 四类对象覆盖结果（流畅实时可视化版） ===")
        print(f"覆盖路径点数: {len(self.coverage_path)}")
        print(f"流畅覆盖标记区域: {coverage_marks}个")
        print(f"")
        print(f"环境对象统计:")
        print(f"  障碍物: {obstacle_count}个 (避障)")
        print(f"  清扫目标: {sweep_total}个")
        print(f"  抓取物体: {grasp_total}个") 
        print(f"  任务区域: {task_count}个")
        print(f"")
        print(f"任务执行统计:")
        print(f"  清扫完成: {self.coverage_stats['swept_objects']}/{sweep_total}")
        print(f"  抓取完成: {self.coverage_stats['grasped_objects']}/{grasp_total}")
        print(f"  运送完成: {self.coverage_stats['delivered_objects']}/{grasp_total}")
        print(f"")
        sweep_rate = (self.coverage_stats['swept_objects'] / sweep_total * 100) if sweep_total > 0 else 0
        grasp_rate = (self.coverage_stats['delivered_objects'] / grasp_total * 100) if grasp_total > 0 else 0
        print(f"任务完成率:")
        print(f"  清扫任务: {sweep_rate:.1f}%")
        print(f"  抓取任务: {grasp_rate:.1f}%")
        print(f"")
        print(f"流畅实时可视化特性:")
        print(f"  精细网格大小: {FINE_GRID_SIZE}m")
        print(f"  覆盖标记半径: {COVERAGE_MARK_RADIUS}m")
        print(f"  更新频率: 每{COVERAGE_UPDATE_FREQUENCY}步")
        print(f"  渐变灰度: 浅灰到深灰10档")
        print(f"  实时跟随: 机器人移动路径")
        print("四类对象覆盖任务完成（集成流畅实时覆盖可视化）!")
    
    def run_demo(self):
        """运行演示"""
        print("\n" + "="*80)
        print("四类对象真实移动覆盖算法机器人系统 - 流畅实时可视化优化版")
        print("障碍物避障 | 清扫目标消失 | 抓取运送 | 任务区域投放")
        print("优化路径规划 | 避免线条贯穿障碍物 | 智能弓字形避障")
        print("集成流畅实时覆盖区域可视化 | 精细网格标记 | 渐变颜色系统")
        print("="*80)
        
        try:
            pos, yaw = self.robot_controller._get_robot_pose()
            print(f"机器人初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
            
            self.plan_coverage_mission()
            
            self.execute_four_object_coverage()
            
            self._move_arm_to_pose("home")
            
            print("\n四类对象覆盖系统演示完成（流畅实时可视化版）!")
            
        except Exception as e:
            print(f"\n演示执行失败: {e}")
    
    def cleanup(self):
        """清理系统"""
        print("清理系统资源...")
        print_memory_usage("清理前")
        
        try:
            if self.visualizer:
                self.visualizer.cleanup()
            
            # 清理内存
            for _ in range(5):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if self.world:
                self.world.stop()
                
        except Exception as e:
            print(f"清理失败: {e}")
        
        print_memory_usage("清理后")
        print("系统资源清理完成")

def main():
    """主函数"""
    print("启动流畅实时可视化优化版四类对象覆盖算法机器人系统...")
    print(f"对象类型:")
    print(f"  1. 障碍物(红色) - 路径规划避障")
    print(f"  2. 清扫目标(黄色) - 触碰消失") 
    print(f"  3. 抓取物体(绿色) - 运送到任务区域")
    print(f"  4. 任务区域(蓝色) - 物体放置区域 (无物理属性)")
    print(f"")
    print(f"机器人半径: {ROBOT_RADIUS}m")
    print(f"安全边距: {SAFETY_MARGIN}m") 
    print(f"交互距离: {INTERACTION_DISTANCE}m")
    print(f"优化特性: 高效弓字形避障路径，避免贯穿障碍物，90度转弯优先")
    print(f"物理设置: 任务区域无物理属性，放置物体与区域贴合且完全无碰撞")
    print(f"")
    print(f"流畅实时可视化特性:")
    print(f"  精细网格: {FINE_GRID_SIZE}m")
    print(f"  覆盖标记半径: {COVERAGE_MARK_RADIUS}m")
    print(f"  更新频率: {COVERAGE_UPDATE_FREQUENCY}")
    print(f"  渐变颜色: 10档灰度系统")
    print(f"  实时跟随: 流畅标记机器人移动轨迹")
    
    system = FourObjectCoverageSystem()
    
    try:
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
        print("\n=== 演示执行阶段 ===")
        system.run_demo()
        
        # 保持运行观察效果
        print("\n=== 效果观察阶段 ===")
        for i in range(200):
            system.world.step(render=True)
            time.sleep(0.05)
            if i % 50 == 0:
                print(f"  观察进度: {i+1}/200")
    
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n系统运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== 系统清理阶段 ===")
        try:
            system.cleanup()
        except Exception as e:
            print(f"清理过程出现问题: {e}")
        print("程序结束")

if __name__ == "__main__":
    main()