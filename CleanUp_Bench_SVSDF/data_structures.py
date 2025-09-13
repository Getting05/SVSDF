#!/usr/bin/env python3
"""
数据结构和类型定义模块
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass

# 系统配置参数
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
SAFETY_MARGIN = 0.2
INTERACTION_DISTANCE = 1

# 可视化参数
FINE_GRID_SIZE = 0.03          # 精细网格大小
COVERAGE_UPDATE_FREQUENCY = 1  # 覆盖更新频率
COVERAGE_MARK_RADIUS = 0.45    # 覆盖标记半径
MARK_DISTANCE_THRESHOLD = 0.02 # 标记距离阈值

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
    shape_type: str
    dimensions: np.ndarray
    rotation: float = 0.0

@dataclass
class SceneObject:
    """场景对象"""
    name: str
    object_type: ObjectType
    position: np.ndarray
    collision_boundary: CollisionBoundary
    isaac_object: object = None
    color: np.ndarray = np.array([0.5, 0.5, 0.5])
    is_active: bool = True
    original_position: np.ndarray = np.array([0.0, 0.0, 0.0])
    grasp_failed: bool = False

@dataclass
class CoveragePoint:
    position: np.ndarray
    orientation: float
    has_object: bool = False