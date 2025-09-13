#!/usr/bin/env python3
"""
SLAM版本路径规划模块 - 基于已知地图进行路径规划
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from data_structures import *

class SLAMPathPlanner:
    """SLAM版本路径规划器 - 基于Cartographer地图"""
    
    def __init__(self):
        self.world_size = COVERAGE_AREA_SIZE
        self.cell_size = COVERAGE_CELL_SIZE
        self.robot_radius = ROBOT_RADIUS
        self.safety_margin = SAFETY_MARGIN
        
        # SLAM地图相关
        self.slam_map = None
        self.slam_map_resolution = 0.05  # Cartographer默认分辨率
        self.slam_map_origin = [0.0, 0.0]
        self.slam_map_width = 0
        self.slam_map_height = 0
        
        # 任务对象（已知坐标）
        self.task_objects = []  # sweep, grasp, task对象
        
        print(f"SLAM路径规划器初始化完成")
    
    def add_task_object(self, scene_obj: SceneObject):
        """添加任务对象（已知坐标的清扫、抓取、任务对象）"""
        if scene_obj.object_type in [ObjectType.SWEEP, ObjectType.GRASP, ObjectType.TASK]:
            self.task_objects.append(scene_obj)
            print(f"添加任务对象: {scene_obj.name} ({scene_obj.object_type.value})")
    
    def update_slam_map(self, map_data: Dict):
        """更新SLAM地图数据"""
        # 关键修复：确保slam_map是numpy数组而不是列表
        map_data_raw = map_data['data']
        if isinstance(map_data_raw, list):
            # 如果是1D列表，重新reshape为2D numpy数组
            map_width = map_data['width']
            map_height = map_data['height']
            self.slam_map = np.array(map_data_raw).reshape(map_height, map_width)
            print(f"SLAM地图数据从列表转换为numpy数组: {map_width}x{map_height}")
        else:
            # 如果已经是numpy数组，直接使用
            self.slam_map = np.array(map_data_raw)
            
        self.slam_map_width = map_data['width']
        self.slam_map_height = map_data['height']
        self.slam_map_resolution = map_data['resolution']
        self.slam_map_origin = map_data['origin']
        
        print(f"SLAM地图更新: {self.slam_map_width}x{self.slam_map_height}, "
              f"分辨率: {self.slam_map_resolution:.3f}m/cell, 数据类型: {type(self.slam_map)}")
        print(f"SLAM地图形状: {self.slam_map.shape}")
    
    def generate_slam_based_coverage_path(self, start_pos: np.ndarray, slam_map: Dict) -> List[CoveragePoint]:
        """基于SLAM地图生成覆盖路径"""
        print(f"=== 基于SLAM地图生成覆盖路径 ===")
        
        # 更新地图
        self.update_slam_map(slam_map)
        
        # 生成基于SLAM地图的安全覆盖点
        coverage_points = self._generate_slam_safe_coverage_grid()
        print(f"SLAM安全覆盖点: {len(coverage_points)}个")
        
        # 创建任务优先路径
        task_priority_path = self._create_task_priority_path(coverage_points, start_pos)
        print(f"任务优先路径: {len(task_priority_path)}个点")
        
        # 验证路径连通性
        validated_path = self._validate_slam_path_connectivity(task_priority_path)
        print(f"验证后路径: {len(validated_path)}个点")
        
        return validated_path
    
    def generate_task_based_path(self, start_pos: np.ndarray) -> List[CoveragePoint]:
        """备用方案：基于已知任务对象生成简单路径"""
        print(f"=== 使用备用任务路径规划 ===")
        
        path_points = []
        
        # 添加起始点
        start_point = CoveragePoint(
            position=start_pos,
            orientation=0.0,
            has_object=False
        )
        path_points.append(start_point)
        
        # 访问所有任务对象
        for task_obj in self.task_objects:
            if task_obj.is_active:
                # 创建接近点
                approach_pos = task_obj.position.copy()
                approach_pos[2] = 0.0
                
                task_point = CoveragePoint(
                    position=approach_pos,
                    orientation=0.0,
                    has_object=True
                )
                path_points.append(task_point)
        
        print(f"备用路径生成完成: {len(path_points)}个点")
        return path_points
    
    def _generate_slam_safe_coverage_grid(self) -> List[Tuple[float, float]]:
        """基于SLAM地图生成安全覆盖点"""
        coverage_points = []
        
        if self.slam_map is None:
            print("⚠ SLAM地图不可用，使用默认覆盖网格")
            return self._generate_default_coverage_grid()
        
        # 转换SLAM地图坐标系
        map_data = self.slam_map
        resolution = self.slam_map_resolution
        origin = self.slam_map_origin
        
        # 遍历地图网格，找到安全的覆盖点
        step_size = max(1, int(self.cell_size / resolution))  # 覆盖点间距
        robot_radius_cells = int((self.robot_radius + self.safety_margin) / resolution)
        
        for map_y in range(0, self.slam_map_height, step_size):
            for map_x in range(0, self.slam_map_width, step_size):
                # 检查该点是否安全
                if self._is_slam_position_safe(map_x, map_y, robot_radius_cells):
                    # 转换为世界坐标
                    world_x = origin[0] + map_x * resolution
                    world_y = origin[1] + map_y * resolution
                    
                    coverage_points.append((world_x, world_y))
        
        print(f"从SLAM地图提取安全覆盖点: {len(coverage_points)}个")
        return coverage_points
    
    def _generate_default_coverage_grid(self) -> List[Tuple[float, float]]:
        """生成默认覆盖网格（备用方案）"""
        coverage_points = []
        
        # 简单的网格生成
        grid_range = 8.0  # ±8米范围
        step = self.cell_size
        
        for x in np.arange(-grid_range, grid_range + step, step):
            for y in np.arange(-grid_range, grid_range + step, step):
                # 避开边界
                if abs(x) < grid_range - 1.0 and abs(y) < grid_range - 1.0:
                    coverage_points.append((x, y))
        
        return coverage_points
    
    def _is_slam_position_safe(self, map_x: int, map_y: int, radius_cells: int) -> bool:
        """检查SLAM地图中的位置是否安全"""
        if self.slam_map is None:
            return False
        
        # 检查机器人占地空间是否全部安全
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                # 检查边界
                if (check_x < 0 or check_x >= self.slam_map_width or
                    check_y < 0 or check_y >= self.slam_map_height):
                    return False
                
                # 检查占用状态
                # Cartographer: -1=未知, 0=空闲, 100=占用
                cell_value = self.slam_map[check_y, check_x]
                
                if cell_value > 50 or cell_value < 0:  # 占用或未知区域
                    return False
                
                # 检查距离约束
                distance = math.sqrt(dx*dx + dy*dy) * self.slam_map_resolution
                if distance <= self.robot_radius + self.safety_margin:
                    if cell_value > 20:  # 不够安全的区域
                        return False
        
        return True
    
    def _create_task_priority_path(self, coverage_points: List[Tuple[float, float]], 
                                 start_pos: np.ndarray) -> List[CoveragePoint]:
        """创建任务优先的路径"""
        print("创建任务优先覆盖路径...")
        
        path_points = []
        
        # 1. 添加起始点
        start_point = CoveragePoint(
            position=start_pos,
            orientation=0.0,
            has_object=False
        )
        path_points.append(start_point)
        
        # 2. 按距离排序覆盖点
        coverage_points_sorted = sorted(coverage_points, 
                                      key=lambda p: np.linalg.norm([p[0] - start_pos[0], p[1] - start_pos[1]]))
        
        # 3. 创建弓字形覆盖模式
        bow_pattern_points = self._create_bow_pattern_from_points(coverage_points_sorted, start_pos)
        
        # 4. 插入任务对象访问点
        enhanced_path = self._insert_task_object_visits(bow_pattern_points)
        
        print(f"任务优先路径创建完成: {len(enhanced_path)}个点")
        return enhanced_path
    
    def _create_bow_pattern_from_points(self, points: List[Tuple[float, float]], 
                                      start_pos: np.ndarray) -> List[CoveragePoint]:
        """从覆盖点创建弓字形模式"""
        if not points:
            return []
        
        path_points = []
        
        # 按Y坐标分组形成行
        rows = {}
        for x, y in points:
            row_key = round(y / self.cell_size)  # 量化Y坐标
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x, y))
        
        # 按Y坐标排序行
        sorted_row_keys = sorted(rows.keys())
        
        # 处理每一行，实现弓字形模式
        for row_idx, row_key in enumerate(sorted_row_keys):
            row_points = sorted(rows[row_key], key=lambda p: p[0])  # 按X排序
            
            # 奇数行反向形成弓字形
            if row_idx % 2 == 1:
                row_points.reverse()
            
            # 为每个点创建CoveragePoint
            for point_idx, (x, y) in enumerate(row_points):
                # 计算朝向
                if point_idx < len(row_points) - 1:
                    next_x, next_y = row_points[point_idx + 1]
                    orientation = np.arctan2(next_y - y, next_x - x)
                elif row_idx % 2 == 0:
                    orientation = 0.0  # 正向
                else:
                    orientation = math.pi  # 反向
                
                # 检查附近是否有任务对象
                has_object = self._has_nearby_task_object(np.array([x, y, 0.0]))
                
                coverage_point = CoveragePoint(
                    position=np.array([x, y, 0.0]),
                    orientation=orientation,
                    has_object=has_object
                )
                
                path_points.append(coverage_point)
        
        return path_points
    
    def _insert_task_object_visits(self, base_path: List[CoveragePoint]) -> List[CoveragePoint]:
        """在基础路径中插入任务对象访问点"""
        enhanced_path = []
        
        for point in base_path:
            enhanced_path.append(point)
            
            # 检查附近是否有未访问的任务对象
            nearby_tasks = self._find_nearby_task_objects(point.position)
            
            for task_obj in nearby_tasks:
                if task_obj.is_active:
                    # 创建任务对象访问点
                    task_visit_point = CoveragePoint(
                        position=task_obj.position.copy(),
                        orientation=point.orientation,
                        has_object=True
                    )
                    enhanced_path.append(task_visit_point)
        
        return enhanced_path
    
    def _find_nearby_task_objects(self, position: np.ndarray, radius: float = 2.0) -> List[SceneObject]:
        """找到附近的任务对象"""
        nearby_objects = []
        
        for task_obj in self.task_objects:
            if task_obj.is_active:
                distance = np.linalg.norm(position[:2] - task_obj.position[:2])
                if distance <= radius:
                    nearby_objects.append(task_obj)
        
        return nearby_objects
    
    def _has_nearby_task_object(self, position: np.ndarray) -> bool:
        """检查附近是否有任务对象"""
        nearby_objects = self._find_nearby_task_objects(position, radius=1.0)
        return len(nearby_objects) > 0
    
    def _validate_slam_path_connectivity(self, path_points: List[CoveragePoint]) -> List[CoveragePoint]:
        """验证基于SLAM地图的路径连通性"""
        if len(path_points) <= 1:
            return path_points
        
        validated_path = [path_points[0]]
        
        for i in range(1, len(path_points)):
            current = validated_path[-1]
            next_point = path_points[i]
            
            # 检查路径是否安全（基于SLAM地图）
            if self._is_slam_path_safe(current.position, next_point.position):
                validated_path.append(next_point)
            else:
                # 需要绕行
                detour_points = self._find_slam_detour(current.position, next_point.position)
                validated_path.extend(detour_points)
                validated_path.append(next_point)
        
        print(f"SLAM路径验证完成: 原{len(path_points)}点 -> 验证{len(validated_path)}点")
        return validated_path
    
    def _is_slam_path_safe(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """基于SLAM地图检查路径安全性"""
        if self.slam_map is None:
            return True  # 没有地图时假设安全
        
        # 沿路径采样检查
        distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
        steps = max(1, int(distance / (self.slam_map_resolution * 2)))
        
        for i in range(steps + 1):
            t = i / max(1, steps)
            check_pos = start_pos + t * (end_pos - start_pos)
            
            # 转换为地图坐标
            map_x = int((check_pos[0] - self.slam_map_origin[0]) / self.slam_map_resolution)
            map_y = int((check_pos[1] - self.slam_map_origin[1]) / self.slam_map_resolution)
            
            # 检查机器人半径范围内的安全性
            if not self._is_slam_position_safe(map_x, map_y, 
                                             int((self.robot_radius + self.safety_margin) / self.slam_map_resolution)):
                return False
        
        return True
    
    def _find_slam_detour(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[CoveragePoint]:
        """基于SLAM地图寻找绕行路径"""
        detour_points = []
        
        # 简化的绕行策略：尝试90度转向
        direction = goal_pos[:2] - start_pos[:2]
        if np.linalg.norm(direction) < 0.1:
            return detour_points
        
        direction_norm = direction / np.linalg.norm(direction)
        
        # 垂直方向
        perpendicular = np.array([-direction_norm[1], direction_norm[0]])
        
        # 尝试左右绕行
        for side_multiplier in [1, -1]:
            detour_offset = perpendicular * side_multiplier * 1.0  # 1米偏移
            mid_point = (start_pos[:2] + goal_pos[:2]) / 2 + detour_offset
            mid_pos_3d = np.array([mid_point[0], mid_point[1], 0.0])
            
            # 检查绕行点是否安全
            map_x = int((mid_pos_3d[0] - self.slam_map_origin[0]) / self.slam_map_resolution)
            map_y = int((mid_pos_3d[1] - self.slam_map_origin[1]) / self.slam_map_resolution)
            
            if self._is_slam_position_safe(map_x, map_y, 
                                         int((self.robot_radius + self.safety_margin) / self.slam_map_resolution)):
                # 安全的绕行点
                detour_point = CoveragePoint(
                    position=mid_pos_3d,
                    orientation=np.arctan2(direction_norm[1], direction_norm[0]),
                    has_object=False
                )
                detour_points.append(detour_point)
                break
        
        return detour_points
    
    def check_collision_with_object(self, robot_pos: np.ndarray, scene_obj: SceneObject) -> bool:
        """检查机器人与对象的碰撞（用于任务对象交互）"""
        if not scene_obj.is_active:
            return False
            
        boundary = scene_obj.collision_boundary
        center = boundary.center
        
        distance_2d = np.linalg.norm(robot_pos[:2] - center[:2])
        
        if boundary.shape_type == 'sphere':
            return distance_2d <= (boundary.dimensions[0] + INTERACTION_DISTANCE)
        
        elif boundary.shape_type == 'box':
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
    
    def get_active_objects_by_type(self, obj_type: ObjectType) -> List[SceneObject]:
        """获取指定类型的活跃对象"""
        return [obj for obj in self.task_objects if obj.object_type == obj_type and obj.is_active]