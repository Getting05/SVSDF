#!/usr/bin/env python3
"""
SLAM集成MapEx探索器 - 官方算法适配版
保持原版MapEx算法逻辑，集成SLAM实时地图数据和机器人控制
"""

import numpy as np
import os 
import cv2
import time
import math
import threading
import socket
import gc
from omegaconf import OmegaConf
import hydra 
import torch 
import matplotlib
import matplotlib.pyplot as plt
import pyastar2d    
import json
import traceback
import argparse
from collections import deque
from queue import PriorityQueue
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from datetime import datetime

# 导入MapEx工具
from lama_pred_utils import load_lama_model, visualize_prediction, get_lama_transform, convert_obsimg_to_model_input
import sys
sys.path.append('../')
from scripts import simple_mask_utils as smu 
import scripts.sim_utils as sim_utils
import upen_baseline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_options_dict_from_yml(config_name):
    cwd = os.getcwd()
    hydra_config_dir_path = os.path.join(cwd, '../configs')
    with hydra.initialize_config_dir(config_dir=hydra_config_dir_path):
        cfg = hydra.compose(config_name=config_name)
    return cfg

def update_mission_status(start_time, cur_step, mission_complete, fail_reason, mission_status_save_path):
    mission_status = {}
    mission_status['start_time'] = start_time
    mission_status["cur_step"] = cur_step
    mission_status["mission_complete"] = mission_complete
    mission_status["fail_reason"] = fail_reason
    mission_status["last_exp_time_s"] = time.time() - mission_status['start_time']
    with open(mission_status_save_path, 'w') as f:
        json.dump(mission_status, f)

def get_lama_pred_from_obs(cur_obs_img, lama_model, lama_map_transform, device):
    cur_obs_img_3chan = np.stack([cur_obs_img, cur_obs_img, cur_obs_img], axis=2)
    input_lama_batch, lama_mask = convert_obsimg_to_model_input(cur_obs_img_3chan, lama_map_transform, device)
    lama_pred_alltrain = lama_model(input_lama_batch)
    lama_pred_alltrain_viz = visualize_prediction(lama_pred_alltrain, lama_mask)
    return cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz

def get_pred_maputils_from_viz(viz_map):
    pred_maputils = np.zeros((viz_map.shape[0], viz_map.shape[1]))
    pred_maputils[viz_map[:,:,0] > 128] = 1
    return pred_maputils

def get_padded_obs_map(obs_map):
    import albumentations as A
    transform = A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, 
                        pad_width_divisor=16, border_mode=cv2.BORDER_CONSTANT, value=0)
    return transform(image=obs_map)['image']

def is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter):
    if locked_frontier_center is None:
        return False
    if occ_grid_pyastar[locked_frontier_center[0], locked_frontier_center[1]] == np.inf:
        return False
    if np.linalg.norm(locked_frontier_center - cur_pose) < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter:
        return False
    return True

def reselect_frontier_from_frontier_region_centers(frontier_region_centers, total_cost, t, start_exp_time):
    if len(frontier_region_centers) <= 1:
        return False, None, frontier_region_centers, total_cost
    
    frontier_region_centers = np.delete(frontier_region_centers, np.argmin(total_cost), axis=0)
    total_cost = np.delete(total_cost, np.argmin(total_cost), axis=0)
    
    if len(frontier_region_centers) == 0:
        return False, None, frontier_region_centers, total_cost
        
    locked_frontier_center = frontier_region_centers[np.argmin(total_cost)]
    return True, locked_frontier_center, frontier_region_centers, total_cost

def determine_local_planner(mode):
    if mode == 'upen':
        return 'astar'
    elif mode in ['nearest', 'visvar', 'visunk', 'obsunk', 'onlyvar', 'visvarprob']:
        return 'astar'
    elif mode == 'hector' or mode == 'hectoraug':
        return 'gradient'
    else:
        raise ValueError("Invalid mode: {}".format(mode))

def determine_use_model(mode):
    possible_mode_list = ['nearest', 'obsunk', 'onlyvar', 'visunk', 'visvar', 'visvarprob', 'upen', 'hectoraug']
    assert mode in possible_mode_list, "mode must be one of {}".format(possible_mode_list)
    return mode in ['onlyvar', 'visunk', 'visvar', 'visvarprob', 'hectoraug', 'upen']

def get_hector_exploration_transform_map(occgrid, frontiers, init_cost, mode, infogain_val_list, info_gain_weight):
    if mode == 'hector':
        assert(infogain_val_list is None), "init_frontier_value must be None for hector mode"
    elif mode == 'hectoraug':
        assert(len(infogain_val_list) == len(frontiers)), "init_frontier_value must be the same length as frontiers for hector aug"
    
    if mode == 'hectoraug':
        max_infogain = np.max(infogain_val_list)
        infogain_cost_list = []
        for infogain_val in infogain_val_list:
            infogain_cost_list.append(int(info_gain_weight * (np.sqrt(max_infogain) - np.sqrt(infogain_val))))
    
    cost_map =  np.full_like(occgrid, np.inf)    
    queue = PriorityQueue()
    for frontier_i, frontier in enumerate(frontiers):
        queue.put((0, (frontier[0], frontier[1])))
        if mode == 'hector':
            cost_map[frontier[0], frontier[1]] = 0
        elif mode == 'hectoraug':
            cost_map[frontier[0], frontier[1]] = infogain_cost_list[frontier_i]
        else:
            raise ValueError("Invalid mode: {}".format(mode))
    
    neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    count = 0
    while not queue.empty():
        cur_ind = queue.get()[1]
        cur_cost = cost_map[cur_ind[0], cur_ind[1]]
        for neighbor in neighbors:
            new_x = cur_ind[0] + neighbor[0]
            new_y = cur_ind[1] + neighbor[1]
            if 0 <= new_x < cost_map.shape[0] and 0 <= new_y < cost_map.shape[1] and (occgrid[new_x, new_y] == 0):
                new_cost = (cur_cost + 1) + init_cost[new_x, new_y]
                if cost_map[new_x, new_y] > new_cost:
                    cost_map[new_x, new_y] = new_cost
                    queue.put((new_cost, (new_x, new_y)))
        count += 1
    return cost_map

def gradient_planner(cur_pose, cost_transform_map):
    neighbor_inds = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    cur_query_pose = np.array([cur_pose[0], cur_pose[1]])
    cur_max_grad = np.inf 
    
    for neighbor_ind in neighbor_inds:
        neighbor_pose = cur_query_pose + neighbor_ind
        grad_to_neighbor = cost_transform_map[neighbor_pose[0], neighbor_pose[1]] - cost_transform_map[cur_query_pose[0], cur_query_pose[1]]
        if grad_to_neighbor < cur_max_grad:
            cur_max_grad = grad_to_neighbor
            next_pose = neighbor_pose
    return next_pose

def compute_velocity_to_target_safe(target, robot_pose, config):
    """安全速度控制 - 优化转弯性能和动态参数调整"""
    current_pos = np.array([robot_pose[0], robot_pose[1]])
    current_yaw = robot_pose[2]
    
    direction = target - current_pos
    distance = np.linalg.norm(direction)
    
    distance_threshold = 0.3  # 减小距离阈值，更早开始目标切换
    if distance < distance_threshold:
        return 0.0, 0.0
    
    target_yaw = np.arctan2(direction[1], direction[0])
    yaw_error = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))

    angle_threshold = np.radians(3)  # 减小角度死区，提高转向精度
    exploration_params = config.get('exploration_params', {})
    
   # 动态调整速度参数 - 根据距离和角度误差
    base_linear_vel = exploration_params.get('max_linear_velocity', 0.35)
    base_angular_vel = exploration_params.get('max_angular_velocity', 2.5)
    
    # 基于距离的动态调整
    if distance > 2.0:  # 远距离目标
        max_linear_vel = min(0.5, base_linear_vel * 1.2)  # 提高远距离线速度
        max_angular_vel = base_angular_vel
    elif distance < 1.0:  # 近距离目标
        max_linear_vel = base_linear_vel * 0.8  # 降低近距离线速度，提高精度
        max_angular_vel = base_angular_vel * 1.1  # 提高近距离角速度
    else:  # 中等距离
        max_linear_vel = base_linear_vel
        max_angular_vel = base_angular_vel

    
    angle_deadzone = angle_threshold
    turn_threshold = np.radians(15)  # 进一步减小转弯阈值，更积极转弯
    
    if abs(yaw_error) < angle_deadzone:
        # 角度误差很小，直线前进
        linear_vel = min(max_linear_vel, 0.6 * distance + 0.2)  # 增加基础速度和距离系数
        angular_vel = 0.0
    elif abs(yaw_error) > turn_threshold:
        # 角度误差大，纯转弯模式 - 更激进的转弯策略
        linear_vel = 0.0  # 完全停止前进，专注转弯
        abs_error = abs(yaw_error)
        
        # 根据角度误差调整转弯速度 - 更细粒度的控制
        if abs_error > 2.5:  # 角度误差极大 (>143度)
            speed_factor = 1.0  # 最高转弯速度
        elif abs_error > 2.0:  # 角度误差非常大 (>114度)
            speed_factor = 0.95
        elif abs_error > 1.5:  # 角度误差大 (>86度)
            speed_factor = 0.9
        elif abs_error > 1.0:  # 角度误差中等 (>57度)
            speed_factor = 0.85
        elif abs_error > 0.5:  # 角度误差较小 (>29度)
            speed_factor = 0.8
        else:  # 角度误差很小 (15-29度)
            speed_factor = 0.75
        
        angular_vel = speed_factor * max_angular_vel * (1.0 if yaw_error > 0 else -1.0)
    else:
        # 中等角度误差，边转弯边前进（但以转弯为主）
        linear_vel = min(max_linear_vel * 0.3, 0.15 * distance + 0.05)  # 进一步降低前进速度
        angular_vel = np.clip(2.5 * yaw_error, -2.0, 2.0)  # 增加角速度系数和限制
    
    # 安全检查 - 确保不超过物理限制
    linear_vel = np.clip(linear_vel, 0.0, 0.6)  # 硬限制最大线速度
    angular_vel = np.clip(angular_vel, -3.0, 3.0)  # 硬限制最大角速度
    
    return linear_vel, angular_vel

def check_and_handle_stuck_situation(self, robot_pose, obs_map):
    """检测和处理机器人卡死情况"""
    current_time = time.time()
    current_pos = np.array([robot_pose[0], robot_pose[1]])
    
    # 初始化上次位置
    if self.last_position is None:
        self.last_position = current_pos.copy()
        self.last_move_time = current_time
        return False, None, None
    
    # 计算移动距离
    move_distance = np.linalg.norm(current_pos - self.last_position)
    
    # 如果移动距离足够大，更新位置和时间
    if move_distance > self.stuck_distance_threshold:
        self.last_position = current_pos.copy()
        self.last_move_time = current_time
        self.in_anti_stuck_mode = False
        return False, None, None
    
    # 检查是否卡死
    stuck_duration = current_time - self.last_move_time
    if stuck_duration > self.stuck_threshold_time:
        print(f"⚠️  检测到机器人卡死！已卡住 {stuck_duration:.1f} 秒，距离移动: {move_distance:.3f}m")
        
        if not self.in_anti_stuck_mode:
            self.in_anti_stuck_mode = True
            self.anti_stuck_start_time = current_time
            print("🔄 启动防卡死模式...")
        
        # 生成防卡死移动指令
        linear_vel, angular_vel = self._generate_anti_stuck_command(robot_pose, obs_map, current_time)
        return True, linear_vel, angular_vel
    
    return False, None, None

def _generate_anti_stuck_command(self, robot_pose, obs_map, current_time):
    """生成防卡死移动指令"""
    anti_stuck_duration = current_time - self.anti_stuck_start_time
    
    # 第一阶段：后退 (0-3秒)
    if anti_stuck_duration < 3.0:
        print("🔙 防卡死阶段1: 后退")
        return -0.2, 0.0  # 后退
    
    # 第二阶段：智能转向，寻找未知区域 (3-8秒)
    elif anti_stuck_duration < 8.0:
        print("🔄 防卡死阶段2: 智能转向寻找未知区域")
        optimal_direction = self._find_optimal_turn_direction(robot_pose, obs_map)
        return 0.0, optimal_direction
    
    # 第三阶段：小步前进测试 (8-10秒)
    elif anti_stuck_duration < 10.0:
        print("➡️  防卡死阶段3: 小步前进测试")
        return 0.1, 0.0  # 小步前进
    
    # 如果10秒后仍然卡死，重置防卡死模式，重新开始
    else:
        print("🔄 防卡死模式重置，重新开始...")
        self.anti_stuck_start_time = current_time
        return -0.2, 0.0  # 重新开始后退

def _find_optimal_turn_direction(self, robot_pose, obs_map):
    """根据周围环境和未知区域寻找最优转向方向"""
    if obs_map is None:
        return 1.5  # 默认向左转
    
    # 获取机器人周围的环境信息
    robot_x = int(robot_pose[0])
    robot_y = int(robot_pose[1])
    search_radius = 20  # 搜索半径（像素）
    
    # 确保坐标在地图范围内
    map_height, map_width = obs_map.shape
    robot_x = np.clip(robot_x, search_radius, map_height - search_radius - 1)
    robot_y = np.clip(robot_y, search_radius, map_width - search_radius - 1)
    
    # 分析不同方向的环境
    directions = {
        'left': (-1.8, 0),      # 向左转
        'right': (1.8, 0),      # 向右转
        'left_fast': (-2.2, 0), # 快速向左转
        'right_fast': (2.2, 0)  # 快速向右转
    }
    
    best_score = -1
    best_direction = 1.5  # 默认向左转
    
    for direction_name, (angular_vel, _) in directions.items():
        # 计算该方向的评分
        score = self._evaluate_direction_score(
            robot_x, robot_y, obs_map, angular_vel, search_radius
        )
        
        if score > best_score:
            best_score = score
            best_direction = angular_vel
    
    print(f"🎯 选择最优转向方向: {best_direction:.1f} rad/s (评分: {best_score:.2f})")
    return best_direction

def _evaluate_direction_score(self, robot_x, robot_y, obs_map, angular_vel, search_radius):
    """评估特定转向方向的优劣"""
    # 根据角速度计算检查的扇形区域
    if angular_vel > 0:  # 向左转，检查左侧区域
        angle_start = np.pi/4
        angle_end = 3*np.pi/4
    else:  # 向右转，检查右侧区域
        angle_start = -3*np.pi/4  
        angle_end = -np.pi/4
    
    free_space_count = 0
    unknown_space_count = 0
    obstacle_count = 0
    total_count = 0
    
    # 扫描扇形区域
    for r in range(5, search_radius, 2):  # 从距离5开始，避免检查太近的区域
        for angle in np.linspace(angle_start, angle_end, 10):
            dx = int(r * np.cos(angle))
            dy = int(r * np.sin(angle))
            
            check_x = robot_x + dx
            check_y = robot_y + dy
            
            # 确保坐标在地图范围内
            if 0 <= check_x < obs_map.shape[0] and 0 <= check_y < obs_map.shape[1]:
                cell_value = obs_map[check_x, check_y]
                total_count += 1
                
                if cell_value == 0:  # 自由空间
                    free_space_count += 1
                elif cell_value == 0.5:  # 未知区域
                    unknown_space_count += 1
                elif cell_value == 1:  # 障碍物
                    obstacle_count += 1
    
    if total_count == 0:
        return 0
    
    # 计算评分：优先选择有更多未知区域和自由空间的方向
    free_ratio = free_space_count / total_count
    unknown_ratio = unknown_space_count / total_count  
    obstacle_ratio = obstacle_count / total_count
    
    # 评分公式：未知区域权重最高，自由空间次之，障碍物惩罚
    score = unknown_ratio * 3.0 + free_ratio * 2.0 - obstacle_ratio * 1.0
    
    return score

def analyze_map_statistics(slam_map_data, obstacle_config):
    """分析地图统计信息"""
    if not slam_map_data:
        return None
    
    width = slam_map_data['width']
    height = slam_map_data['height']
    map_data = np.array(slam_map_data['data']).reshape((height, width))
    
    # 获取阈值配置
    free_threshold = obstacle_config.get('free_threshold', 10)
    obstacle_threshold = obstacle_config.get('obstacle_threshold', 90)
    unknown_threshold = obstacle_config.get('unknown_threshold', -1)
    
    total_cells = width * height
    
    # 统计不同区域
    unknown_cells = np.sum(map_data < unknown_threshold)
    free_cells = np.sum(map_data <= free_threshold)
    obstacle_cells = np.sum(map_data >= obstacle_threshold)
    
    # 中等概率区域（不确定区域）
    uncertain_mask = (map_data > free_threshold) & (map_data < obstacle_threshold) & (map_data >= 0)
    uncertain_cells = np.sum(uncertain_mask)
    
    # 计算比例
    unknown_ratio = unknown_cells / total_cells * 100
    free_ratio = free_cells / total_cells * 100
    obstacle_ratio = obstacle_cells / total_cells * 100
    uncertain_ratio = uncertain_cells / total_cells * 100
    
    # 计算累计概率
    valid_cells = map_data[map_data >= 0]  # 排除未知区域
    if len(valid_cells) > 0:
        avg_probability = np.mean(valid_cells)
        std_probability = np.std(valid_cells)
    else:
        avg_probability = 0
        std_probability = 0
    
    stats = {
        'total_cells': total_cells,
        'unknown_cells': unknown_cells,
        'free_cells': free_cells,
        'obstacle_cells': obstacle_cells,
        'uncertain_cells': uncertain_cells,
        'unknown_ratio': unknown_ratio,
        'free_ratio': free_ratio,
        'obstacle_ratio': obstacle_ratio,
        'uncertain_ratio': uncertain_ratio,
        'avg_probability': avg_probability,
        'std_probability': std_probability,
        'thresholds': {
            'free_threshold': free_threshold,
            'obstacle_threshold': obstacle_threshold,
            'unknown_threshold': unknown_threshold
        }
    }
    
    return stats

def visualize_and_save_map(slam_map_data, robot_pose, step, output_dir, obstacle_config, 
                          frontier_centers=None, inflated_map=None):
    """可视化地图并保存为PNG - 包含膨胀地图显示"""
    if not slam_map_data:
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    width = slam_map_data['width']
    height = slam_map_data['height']
    resolution = slam_map_data['resolution']
    map_data = np.array(slam_map_data['data']).reshape((height, width))
    
    # 获取阈值配置
    free_threshold = obstacle_config.get('free_threshold', 20)
    obstacle_threshold = obstacle_config.get('obstacle_threshold', 80)
    
    # 创建可视化地图
    vis_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 未知区域 - 灰色
    unknown_mask = map_data < 0
    vis_map[unknown_mask] = [128, 128, 128]
    
    # 自由空间 - 白色
    free_mask = (map_data >= 0) & (map_data <= free_threshold)
    vis_map[free_mask] = [255, 255, 255]
    
    # 障碍物 - 黑色
    obstacle_mask = map_data >= obstacle_threshold
    vis_map[obstacle_mask] = [0, 0, 0]
    
    # 不确定区域 - 淡灰色
    uncertain_mask = (map_data > free_threshold) & (map_data < obstacle_threshold) & (map_data >= 0)
    vis_map[uncertain_mask] = [200, 200, 200]
    
    # 创建图形 - 增加子图数量以包含膨胀地图
    plt.figure(figsize=(20, 12))
    
    # 原始地图显示
    plt.subplot(2, 3, 1)
    plt.imshow(vis_map, origin='lower')
    plt.title(f'Original Map - Step {step}', fontsize=12, fontweight='bold')
    
    # 添加机器人位置
    robot_x, robot_y = None, None
    if robot_pose:
        robot_x = int(robot_pose[0] / resolution + width // 2)
        robot_y = int(robot_pose[1] / resolution + height // 2)
        if 0 <= robot_x < width and 0 <= robot_y < height:
            plt.plot(robot_x, robot_y, 'ro', markersize=8, label='Robot Position')
    
    # 添加前沿点
    if frontier_centers is not None and len(frontier_centers) > 0:
        for i, frontier in enumerate(frontier_centers[:5]):  # 只显示前5个前沿
            plt.plot(frontier[1], frontier[0], 'b*', markersize=10, 
                    label='Frontier' if i == 0 else "")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    # 膨胀地图显示（新增）
    if inflated_map is not None:
        plt.subplot(2, 3, 2)
        # 创建膨胀地图可视化
        inflated_vis = np.zeros((inflated_map.shape[0], inflated_map.shape[1], 3), dtype=np.uint8)
        
        # 自由空间 - 白色
        free_inflated = inflated_map == 0
        inflated_vis[free_inflated] = [255, 255, 255]
        
        # 膨胀障碍物 - 红色（与原始障碍物区分）
        inflated_obstacles = inflated_map == np.inf
        inflated_vis[inflated_obstacles] = [255, 0, 0]
        
        plt.imshow(inflated_vis, origin='lower')
        plt.title('Inflated Map for Planning', fontsize=12, fontweight='bold')
        
        # 添加机器人位置
        if robot_pose and robot_x is not None and robot_y is not None:
            # 转换到膨胀地图坐标系
            inflated_robot_x = min(robot_x, inflated_map.shape[1] - 1)
            inflated_robot_y = min(robot_y, inflated_map.shape[0] - 1)
            plt.plot(inflated_robot_x, inflated_robot_y, 'go', markersize=8, label='Robot Position')
            
        # 添加前沿点 - 检查在膨胀地图中的可达性
        if frontier_centers is not None and len(frontier_centers) > 0:
            reachable_count = 0
            unreachable_count = 0
            for frontier in frontier_centers[:5]:
                # 检查前沿在膨胀地图中是否可达
                if (0 <= frontier[0] < inflated_map.shape[0] and 
                    0 <= frontier[1] < inflated_map.shape[1]):
                    if inflated_map[frontier[0], frontier[1]] == np.inf:
                        # 不可达前沿 - 红色X
                        plt.plot(frontier[1], frontier[0], 'rx', markersize=12, 
                                label='Unreachable Frontier' if unreachable_count == 0 else "")
                        unreachable_count += 1
                    else:
                        # 可达前沿 - 绿色星号
                        plt.plot(frontier[1], frontier[0], 'g*', markersize=10, 
                                label='Reachable Frontier' if reachable_count == 0 else "")
                        reachable_count += 1
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
    
    # 概率分布热力图
    plt.subplot(2, 3, 3)
    prob_map = map_data.copy().astype(float)
    prob_map[prob_map < 0] = np.nan  # 未知区域设为NaN
    im = plt.imshow(prob_map, cmap='RdYlBu_r', origin='lower', vmin=0, vmax=100)
    plt.title('Occupancy Probability Heatmap', fontsize=12)
    plt.colorbar(im, label='Occupancy Probability (%)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    # 统计信息图表
    stats = analyze_map_statistics(slam_map_data, obstacle_config)
    if stats:
        plt.subplot(2, 3, 4)
        categories = ['Unknown', 'Free Space', 'Obstacles', 'Uncertain']
        ratios = [stats['unknown_ratio'], stats['free_ratio'], 
                 stats['obstacle_ratio'], stats['uncertain_ratio']]
        colors = ['gray', 'white', 'black', 'lightgray']
        
        bars = plt.bar(categories, ratios, color=colors, edgecolor='black')
        plt.title('Area Distribution (%)', fontsize=12)
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, ratio in zip(bars, ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 膨胀效果对比（新增）
    if inflated_map is not None:
        plt.subplot(2, 3, 5)
        
        # 计算膨胀前后的差异
        original_obstacles = obstacle_mask.astype(int)
        inflated_obstacles_2d = (inflated_map == np.inf).astype(int)
        
        # 确保尺寸匹配
        min_h = min(original_obstacles.shape[0], inflated_obstacles_2d.shape[0])
        min_w = min(original_obstacles.shape[1], inflated_obstacles_2d.shape[1])
        
        original_crop = original_obstacles[:min_h, :min_w]
        inflated_crop = inflated_obstacles_2d[:min_h, :min_w]
        
        # 计算膨胀区域
        inflation_area = inflated_crop - original_crop
        inflation_area[inflation_area < 0] = 0  # 确保没有负值
        inflation_ratio = np.sum(inflation_area) / (min_h * min_w) * 100
        
        # 创建对比可视化
        comparison_vis = np.zeros((min_h, min_w, 3), dtype=np.uint8)
        comparison_vis[original_crop == 1] = [0, 0, 0]  # 原始障碍物 - 黑色
        comparison_vis[inflation_area == 1] = [255, 100, 100]  # 膨胀区域 - 浅红色
        comparison_vis[(original_crop == 0) & (inflated_crop == 0)] = [255, 255, 255]  # 自由空间 - 白色
        
        plt.imshow(comparison_vis, origin='lower')
        plt.title(f'Inflation Effect\n({inflation_ratio:.1f}% area inflated)', fontsize=12)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
    
    # 详细信息文本
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    if stats:
        # 添加膨胀配置信息
        pixel_per_meter = 1.0 / resolution
        dilate_diam = 15  # 从配置中获取
        safe_radius = (dilate_diam / 2) / pixel_per_meter
        
        inflation_info = f"""
Inflation Configuration:
• Diameter: {dilate_diam} pixels ({safe_radius:.1f}m)
• Safe distance: {safe_radius:.1f}m radius
• Pixel/meter: {pixel_per_meter:.1f}
"""
        
        info_text = f"""Map Statistics (Step {step}):

Total Cells: {stats['total_cells']:,}
Resolution: {resolution:.3f} m/pixel

Area Distribution:
• Unknown: {stats['unknown_cells']:,} cells ({stats['unknown_ratio']:.1f}%)
• Free Space: {stats['free_cells']:,} cells ({stats['free_ratio']:.1f}%)
• Obstacles: {stats['obstacle_cells']:,} cells ({stats['obstacle_ratio']:.1f}%)
• Uncertain: {stats['uncertain_cells']:,} cells ({stats['uncertain_ratio']:.1f}%)

Probability Statistics:
• Average: {stats['avg_probability']:.1f}%
• Std Dev: {stats['std_probability']:.1f}%

Threshold Settings:
• Free Space: ≤ {stats['thresholds']['free_threshold']}%
• Obstacles: ≥ {stats['thresholds']['obstacle_threshold']}%
• Unknown: < {stats['thresholds']['unknown_threshold']}%
{inflation_info}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片 - 支持字符串和整数类型的step参数
    if isinstance(step, str):
        filename = f'map_exploration_step_{step}.png'
    else:
        filename = f'map_exploration_step_{step:04d}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced map visualization saved to: {filepath}")
    
    return stats

def print_detailed_map_stats(stats, step):
    """打印详细的地图统计信息"""
    if not stats:
        return
    
    print(f"\n{'='*80}")
    print(f"DETAILED MAP ANALYSIS - STEP {step}")
    print(f"{'='*80}")
    print(f"Map Dimensions: {int(np.sqrt(stats['total_cells']))}x{int(np.sqrt(stats['total_cells']))} pixels")
    print(f"Total Cells: {stats['total_cells']:,}")
    print(f"\nAREA DISTRIBUTION:")
    print(f"├── Unknown Areas:     {stats['unknown_cells']:8,} cells ({stats['unknown_ratio']:6.2f}%)")
    print(f"├── Free Space:        {stats['free_cells']:8,} cells ({stats['free_ratio']:6.2f}%)")
    print(f"├── Obstacle Areas:    {stats['obstacle_cells']:8,} cells ({stats['obstacle_ratio']:6.2f}%)")
    print(f"└── Uncertain Areas:   {stats['uncertain_cells']:8,} cells ({stats['uncertain_ratio']:6.2f}%)")
    
    print(f"\nPROBABILITY STATISTICS:")
    print(f"├── Average Probability: {stats['avg_probability']:6.2f}%")
    print(f"└── Standard Deviation:  {stats['std_probability']:6.2f}%")
    
    print(f"\nTHRESHOLD CONFIGURATION:")
    print(f"├── Free Space Threshold:  ≤ {stats['thresholds']['free_threshold']:3d}%")
    print(f"├── Obstacle Threshold:    ≥ {stats['thresholds']['obstacle_threshold']:3d}%")
    print(f"└── Unknown Threshold:     < {stats['thresholds']['unknown_threshold']:3d}%")
    
    # 探索进度分析
    known_ratio = stats['free_ratio'] + stats['obstacle_ratio'] + stats['uncertain_ratio']
    print(f"\nEXPLORATION PROGRESS:")
    print(f"├── Known Areas:      {known_ratio:6.2f}%")
    print(f"└── Unknown Areas:    {stats['unknown_ratio']:6.2f}%")
    
    print(f"{'='*80}\n")

class SLAMMapExExplorer:
    def __init__(self, config_name='base.yaml'):
        self.config_name = config_name
        self.collect_opts = get_options_dict_from_yml(config_name)
        
        # Socket通信配置
        socket_config = self.collect_opts.get('socket_bridge_config', {})
        self.bridge_host = socket_config.get('host', 'localhost')
        self.bridge_port = socket_config.get('port', 9998)
        self.socket_client = None
        self.connected = False
        
        # SLAM地图状态
        self.current_slam_map = None
        self.robot_pose = [0.0, 0.0, 0.0]
        self.exploration_active = False
        self.exploration_complete = False
        
        # 地图尺寸变化时的坐标同步
        self.map_size_changed = False
        self.old_map_size = None
        
        # 防卡死机制状态变量
        self.last_position = None
        self.last_move_time = time.time()
        self.stuck_threshold_time = 100.0  # 10秒没移动认为卡死
        self.stuck_distance_threshold = 0.1  # 移动距离小于0.1米认为没有移动
        self.in_anti_stuck_mode = False
        self.anti_stuck_start_time = None
        
        # MapEx组件
        self.lama_model = None
        self.lama_map_transform = None
        self.mapper = None
        self.frontier_planner = None
        self.model_list = []
        
        # 线程管理
        self.socket_thread = None
        self.exploration_thread = None
        self.running = True
        
        # 探索参数
        self.last_map_process_time = 0
        
        # 调试和可视化配置
        self.debug_output_dir = "/home/getting/SVSDF/CleanUp_Bench_SVSDF/results"
        self.visualization_frequency = 10  # 每10步保存一次可视化
        self.stats_frequency = 10  # 每10步打印一次统计信息
        
        # 确保输出目录存在
        os.makedirs(self.debug_output_dir, exist_ok=True)
        
        print("=== SLAM MapEx Explorer (官方算法适配版) ===")
        print("初始化完成")
        print(f"调试输出目录: {self.debug_output_dir}")
        print(f"可视化频率: 每{self.visualization_frequency}步")
        print(f"统计信息频率: 每{self.stats_frequency}步")
    
    def initialize_mapex_components(self):
        """初始化MapEx组件 - 保持原版逻辑"""
        print("初始化MapEx组件...")
        
        device = self.collect_opts.lama_device
        
        # 加载主模型
        if self.collect_opts.big_lama_model_folder_name:
            model_path = os.path.join(
                self.collect_opts.root_path, 
                'pretrained_models', 
                self.collect_opts.big_lama_model_folder_name
            )
            self.lama_model = load_lama_model(model_path, device=device)
            print(f"LAMA模型加载完成: {model_path}")
        
        # 加载ensemble模型
        if self.collect_opts.ensemble_folder_name:
            ensemble_folder_name = self.collect_opts.ensemble_folder_name
            ensemble_model_dirs = sorted(os.listdir(os.path.join(self.collect_opts.root_path, 'pretrained_models', ensemble_folder_name)))
            for ensemble_model_dir in ensemble_model_dirs:
                ensemble_model_path = os.path.join(self.collect_opts.root_path, 'pretrained_models', ensemble_folder_name, ensemble_model_dir)
                model = load_lama_model(ensemble_model_path, device=device)
                self.model_list.append(model)
                print(f"Ensemble模型加载: {ensemble_model_dir}")
        
        # 设置变换
        self.lama_map_transform = get_lama_transform(
            self.collect_opts.lama_transform_variant, 
            self.collect_opts.lama_out_size
        )
        
        # 初始化前沿规划器
        if hasattr(self.collect_opts, 'modes_to_test') and self.collect_opts.modes_to_test:
            mode = self.collect_opts.modes_to_test[0]
            print(f"初始化前沿规划器，模式: {mode}")
            
            if mode != 'upen':
                self.frontier_planner = sim_utils.FrontierPlanner(score_mode=mode)
                print(f"前沿规划器初始化完成: {mode}")
        else:
            self.frontier_planner = sim_utils.FrontierPlanner(score_mode='visvarprob')
        
        print("MapEx组件初始化完成")
        return True
    
    def start_socket_communication(self):
        self.socket_thread = threading.Thread(target=self._socket_communication_loop)
        self.socket_thread.daemon = True
        self.socket_thread.start()
        print("Socket通信线程已启动")
    
    def _socket_communication_loop(self):
        retry_count = 0
        max_retries = self.collect_opts.get('socket_bridge_config', {}).get('retry_count', 30)
        
        while self.running and retry_count < max_retries:
            if not self.connected:
                print(f"尝试连接桥接节点... (尝试 {retry_count + 1}/{max_retries})")
                
                try:
                    self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket_client.settimeout(5.0)
                    self.socket_client.connect((self.bridge_host, self.bridge_port))
                    
                    self.connected = True
                    retry_count = 0
                    print("桥接节点连接成功!")
                except Exception as e:
                    print(f"连接失败: {e}")
                    retry_count += 1
                    time.sleep(2.0)
                    continue
            
            try:
                while self.running and self.connected:
                    self._receive_bridge_data()
                    time.sleep(0.1)
            except Exception as e:
                print(f"通信错误: {e}")
                self.connected = False
                if self.socket_client:
                    self.socket_client.close()
                    self.socket_client = None
                    
        print("Socket通信线程结束")
    
    def _receive_bridge_data(self):
        if not self.connected or not self.socket_client:
            return
        
        try:
            length_data = self._recv_exact(4)
            if not length_data:
                return
                
            message_length = int.from_bytes(length_data, byteorder='big')
            message_data = self._recv_exact(message_length)
            if not message_data:
                return
            
            message_str = message_data.decode('utf-8')
            message = json.loads(message_str)
            self._handle_bridge_message(message)
        except Exception as e:
            print(f"接收数据错误: {e}")
            return

    def _recv_exact(self, size):
        data = b''
        while len(data) < size:
            try:
                self.socket_client.settimeout(2.0)
                chunk = self.socket_client.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except Exception:
                return None
        return data

    def _handle_bridge_message(self, message):
        msg_type = message.get('type')
        current_time = time.time()
        
        if msg_type == 'map_update':
            map_data = message.get('data')
            if map_data:
                width = map_data.get('width', 0)
                height = map_data.get('height', 0)
                raw_data = map_data.get('data')
                
                if raw_data and len(raw_data) == width * height:
                    self.current_slam_map = map_data
                    if current_time - getattr(self, 'last_map_process_time', 0) > 0.5:
                        self._update_mapper_with_slam_map()
                        self.last_map_process_time = current_time
                        
        elif msg_type == 'robot_pose':
            if current_time - getattr(self, 'last_pose_process_time', 0) > 0.2:
                data = message.get('data', {})
                self.robot_pose = [
                    data.get('x', 0.0),
                    data.get('y', 0.0),
                    data.get('yaw', 0.0)
                ]
                self.last_pose_process_time = current_time
                
        elif msg_type == 'start_exploration':
            print("收到探索启动指令")
            if not self.exploration_active:
                self.start_exploration()
                
        elif msg_type == 'heartbeat':
            self._send_to_bridge({
                'type': 'heartbeat_response',
                'timestamp': current_time
            })
        
        # 定期清理内存
        if not hasattr(self, 'last_gc_time'):
            self.last_gc_time = current_time
        elif current_time - self.last_gc_time > 30.0:
            gc.collect()
            self.last_gc_time = current_time
    
    def _update_mapper_with_slam_map(self):
        """使用SLAM地图更新mapper - 适配原版Mapper接口"""
        if not self.current_slam_map:
            return
        
        width = self.current_slam_map['width']
        height = self.current_slam_map['height']
        resolution = self.current_slam_map['resolution']
        map_data = np.array(self.current_slam_map['data']).reshape((height, width))
        
        # 获取障碍物检测配置
        obstacle_config = self.collect_opts.get('obstacle_detection', {})
        free_threshold = obstacle_config.get('free_threshold', 20)
        obstacle_threshold = obstacle_config.get('obstacle_threshold', 80)
        unknown_threshold = obstacle_config.get('unknown_threshold', -1)
        
        # 转换地图格式 - 适配原版MapEx格式
        mapex_map = np.zeros_like(map_data, dtype=np.float32)
        
        # 明确的空闲区域：占用概率 <= free_threshold
        mapex_map[map_data <= free_threshold] = 0.0
        
        # 未知区域：负值表示未探索
        mapex_map[map_data < unknown_threshold] = 0.5
        
        # 明确的障碍物：占用概率 >= obstacle_threshold
        mapex_map[map_data >= obstacle_threshold] = 1.0
        
        # 中等概率区域：更保守地处理
        uncertain_mask = (map_data > free_threshold) & (map_data < obstacle_threshold) & (map_data >= 0)
        conservative_obstacle_threshold = (free_threshold + obstacle_threshold) * 0.75
        conservative_obstacles = uncertain_mask & (map_data >= conservative_obstacle_threshold)
        mapex_map[conservative_obstacles] = 0.8
        remaining_uncertain = uncertain_mask & ~conservative_obstacles  
        mapex_map[remaining_uncertain] = 0.5
        
        if self.mapper is None:
            # 首次创建 - 使用原版Mapper接口
            # 修正后的代码
            lidar_configs = {
                'pixel_per_meter': int(1.0 / resolution),
                'laser_range_m': self.collect_opts.get('lidar_sim_configs', {}).get('laser_range_m', 20),
                'num_laser': self.collect_opts.get('lidar_sim_configs', {}).get('num_laser', 2500),
                'dilate_diam_for_planning': self.collect_opts.get('lidar_sim_configs', {}).get('dilate_diam_for_planning', 3),
            }
            
            self.mapper = sim_utils.Mapper(
                mapex_map, 
                lidar_configs, 
                use_distance_transform_for_planning=self.collect_opts.use_distance_transform_for_planning
            )
            print(f"Mapper首次创建: {width}x{height}")
            self.last_map_shape = (width, height)
            
            if not self.exploration_active:
                self.start_exploration()
        else:
            # 地图更新 - 保守更新策略
            if (width, height) != getattr(self, 'last_map_shape', (0, 0)):
                print(f"检测到地图尺寸变化: {getattr(self, 'last_map_shape', (0, 0))} -> {(width, height)}")
                
                # 标记地图尺寸已变化
                self.map_size_changed = True
                self.map_size_changed_last_step = True  # 新增：标记下一步需要重新检测前沿
                self.old_map_size = getattr(self, 'last_map_shape', (0, 0))
                
                old_height, old_width = getattr(self, 'last_map_shape', (0, 0))
                if old_height > 0 and old_width > 0:
                    overlap_h = min(height, old_height)
                    overlap_w = min(width, old_width)
                    
                    old_obs_map = self.mapper.obs_map.copy()
                    new_obs_map = np.full((height, width), 0.5, dtype=np.float32)
                    
                    actual_old_h, actual_old_w = old_obs_map.shape
                    safe_overlap_h = min(overlap_h, actual_old_h, height)
                    safe_overlap_w = min(overlap_w, actual_old_w, width)
                    
                    new_obs_map[:safe_overlap_h, :safe_overlap_w] = old_obs_map[:safe_overlap_h, :safe_overlap_w]
                    
                    unknown_mask = (new_obs_map == 0.5)
                    free_space_override = (new_obs_map == 0.0) & (mapex_map >= 0.5)
                    
                    new_obs_map[unknown_mask] = mapex_map[unknown_mask]
                    new_obs_map[free_space_override] = 0.0
                    
                    self.mapper.obs_map = new_obs_map
                    self.mapper.gt_map = new_obs_map
                    
                else:
                    self.mapper.obs_map = mapex_map
                    self.mapper.gt_map = mapex_map
                
                self.last_map_shape = (width, height)
            else:
                current_obs = self.mapper.obs_map.copy()
                update_mask = (current_obs == 0.5) | (mapex_map >= 0.9)
                protect_free_space = (current_obs == 0.0) & (mapex_map > 0.0) & (mapex_map < 0.9)
                
                current_obs[update_mask & ~protect_free_space] = mapex_map[update_mask & ~protect_free_space]
                
                self.mapper.obs_map = current_obs
                self.mapper.gt_map = current_obs

    def start_exploration(self):
        if self.exploration_active:
            print("探索已经在进行中")
            return
        
        print("开始MapEx探索...")
        self.exploration_active = True
        self.exploration_complete = False
        
        self._send_to_bridge({
            'type': 'exploration_status',
            'data': 'EXPLORATION_STARTED'
        })
        
        self.exploration_thread = threading.Thread(target=self._run_official_exploration)
        self.exploration_thread.daemon = True
        self.exploration_thread.start()
        
        print("MapEx探索已启动")
    
    def _run_official_exploration(self):
        """运行官方MapEx探索算法 - 保持原版逻辑"""
        print("=== 官方MapEx探索循环启动 ===")
        
        if not self.mapper:
            print("等待地图数据...")
            start_wait_time = time.time()
            last_debug_time = start_wait_time
            
            while self.running and not self.mapper:
                current_time = time.time()
                
                if current_time - last_debug_time >= 5.0:
                    wait_duration = int(current_time - start_wait_time)
                    print(f"等待地图数据中... ({wait_duration}s)")
                    last_debug_time = current_time
                time.sleep(1.0)
            
            time.sleep(1.0)
        
        # 初始化探索参数 - 保持原版逻辑
        start_exp_time = time.time()
        t = 0
        max_steps = self.collect_opts.mission_time
        
        # 坐标转换
        pixel_per_meter = getattr(self.mapper, 'pixel_per_meter', 20)
        map_height, map_width = self.mapper.obs_map.shape
        
        print(f"地图信息: {map_width}x{map_height}, 像素密度: {pixel_per_meter} 像素/米")
        
        # 机器人初始位置
        cur_pose_x = int(round(self.robot_pose[0] * pixel_per_meter + map_height // 2))
        cur_pose_y = int(round(self.robot_pose[1] * pixel_per_meter + map_width // 2))
        
        cur_pose_x = np.clip(cur_pose_x, 0, map_height - 1)
        cur_pose_y = np.clip(cur_pose_y, 0, map_width - 1)
        
        cur_pose = np.array([cur_pose_x, cur_pose_y])
        
        print(f"机器人像素坐标: [{cur_pose[0]}, {cur_pose[1]}]")
        
        # 验证坐标有效性并寻找附近空闲空间
        if (cur_pose[0] < 0 or cur_pose[0] >= map_height or 
            cur_pose[1] < 0 or cur_pose[1] >= map_width):
            print(f"机器人坐标超出地图范围: {cur_pose}")
            return
        
        if self.mapper.obs_map[cur_pose[0], cur_pose[1]] == 1:
            print(f"机器人起始位置在障碍物中，寻找附近空闲空间...")
            found_free_space = False
            for offset in range(1, 10):
                if found_free_space:
                    break
                for dx in [-offset, 0, offset]:
                    if found_free_space:
                        break
                    for dy in [-offset, 0, offset]:
                        new_x = cur_pose[0] + dx
                        new_y = cur_pose[1] + dy
                        if (0 <= new_x < map_height and 0 <= new_y < map_width and
                            self.mapper.obs_map[new_x, new_y] == 0):
                            cur_pose = np.array([new_x, new_y])
                            print(f"找到附近空闲位置: {cur_pose}")
                            found_free_space = True
                            break
        
        pose_list = np.atleast_2d(cur_pose)
        ind_to_move_per_step = 3
        
        # 初始观测 - 由于SLAM系统已经提供地图，这里跳过observe操作
        print("跳过初始观测（使用SLAM地图）")
        
        locked_frontier_center = None
        
        # 确定使用的探索模式
        mode = self.collect_opts.modes_to_test[0] if self.collect_opts.modes_to_test else 'visvarprob'
        use_model = determine_use_model(mode)
        
        print(f"开始官方MapEx探索算法，模式: {mode}")
        
        # === 主探索循环 - 保持原版逻辑 ===
        while (self.running and self.exploration_active and 
               t < max_steps and not self.exploration_complete):
            
            t += 1
            start_mission_i_time = time.time()
            show_plt = (t % self.collect_opts.show_plt_freq == 0) or (t == max_steps - 1)
            
            debug_step = (t % 100 == 0)
            
            # === 检查地图尺寸变化并调整坐标 ===
            if getattr(self, 'map_size_changed', False):
                old_height, old_width = self.old_map_size if self.old_map_size else (0, 0)
                new_height, new_width = self.mapper.obs_map.shape
                
                print(f"地图尺寸变化检测：{(old_height, old_width)} -> {(new_height, new_width)}")
                
                # 调整当前位置坐标
                if old_height > 0 and old_width > 0:
                    # 检查当前坐标是否超出新地图范围
                    if (cur_pose[0] >= new_height or cur_pose[1] >= new_width or
                        cur_pose[0] < 0 or cur_pose[1] < 0):
                        print(f"当前位置 {cur_pose} 超出新地图范围，调整...")
                        cur_pose = np.array([
                            np.clip(cur_pose[0], 0, new_height - 1),
                            np.clip(cur_pose[1], 0, new_width - 1)
                        ])
                        print(f"调整后位置: {cur_pose}")
                        
                        # 更新pose_list的最后一个位置
                        pose_list[-1] = cur_pose
                
                # 重置标记
                self.map_size_changed = False
                self.old_map_size = None
            
            # === 地图统计分析和可视化 - 新增调试功能 ===
            if self.current_slam_map and (t % self.stats_frequency == 0 or t == 1):
                # 获取障碍物检测配置
                obstacle_config = self.collect_opts.get('obstacle_detection', {})
                
                # 打印详细统计信息
                stats = analyze_map_statistics(self.current_slam_map, obstacle_config)
                if stats:
                    print_detailed_map_stats(stats, t)
            
            # 定期保存可视化地图
            if self.current_slam_map and (t % self.visualization_frequency == 0 or t == 1):
                obstacle_config = self.collect_opts.get('obstacle_detection', {})
                
                # 获取当前前沿信息（如果有的话）
                current_frontiers = None
                if 'frontier_region_centers_unscored' in locals():
                    current_frontiers = frontier_region_centers_unscored
                elif 'frontier_region_centers' in locals():
                    current_frontiers = frontier_region_centers
                
                # 获取膨胀地图用于可视化
                inflated_planning_map = None
                if hasattr(self, 'mapper') and self.mapper:
                    try:
                        inflated_planning_map = self.mapper.get_inflated_planning_maps(
                            unknown_as_occ=self.collect_opts.unknown_as_occ
                        )
                    except Exception as e:
                        print(f"获取膨胀地图失败: {e}")
                        inflated_planning_map = None
                
                # 保存增强的可视化地图
                visualization_stats = visualize_and_save_map(
                    self.current_slam_map, 
                    self.robot_pose, 
                    t, 
                    self.debug_output_dir, 
                    obstacle_config,
                    frontier_centers=current_frontiers,
                    inflated_map=inflated_planning_map  # 新增参数
                )
                
                if visualization_stats:
                    print("Step {}: 已保存增强地图可视化和统计信息".format(t))
            
            # === 前沿检测 - 提高检测频率，确保及时发现新前沿 ===
            frontier_detection_needed = False
            
            if t == 1:
                # 第一步必须检测前沿
                frontier_detection_needed = True
                print("Step {}: 初次前沿检测".format(t))
            elif mode != 'upen':
                # 定期重新检测前沿，确保捕获新的可探索区域
                frontier_check_frequency = 10  # 每10步检查一次前沿（根据用户要求调整）
                if t % frontier_check_frequency == 0:
                    frontier_detection_needed = True
                    if debug_step:
                        print("Step {}: 定期前沿重检测".format(t))
                
                # 地图大小变化时立即重新检测
                if getattr(self, 'map_size_changed_last_step', False):
                    frontier_detection_needed = True
                    print("Step {}: 地图变化触发前沿重检测".format(t))
                    self.map_size_changed_last_step = False
            
            # 执行前沿检测
            if frontier_detection_needed and mode != 'upen':
                start_frontier_time = time.time()
                frontier_region_centers_unscored, filtered_map, num_large_regions = self.frontier_planner.get_frontier_centers_given_obs_map(self.mapper.obs_map)
                frontier_detection_time = time.time() - start_frontier_time
                
                print("Step {}: 前沿检测完成 - 发现{}个前沿 (耗时:{:.3f}s)".format(
                    t, len(frontier_region_centers_unscored), frontier_detection_time))
                
                # 详细前沿信息
                if len(frontier_region_centers_unscored) > 0:
                    print("  📍 前沿位置预览:")
                    for i, frontier in enumerate(frontier_region_centers_unscored[:5]):  # 显示前5个
                        distance = np.linalg.norm(frontier - cur_pose)
                        print(f"    前沿{i+1}: [{frontier[0]:3.0f}, {frontier[1]:3.0f}] 距离:{distance:.1f}像素")
                    if len(frontier_region_centers_unscored) > 5:
                        print(f"    ... 还有{len(frontier_region_centers_unscored)-5}个前沿")
                else:
                    print("  ❌ 未发现任何前沿")
                
                if len(frontier_region_centers_unscored) == 0:
                    print("Step {}: 没有发现前沿，原地旋转360度后再次检测...".format(t))
                    # 原地旋转360度
                    rotate_steps = 12  # 每次转动30度，共12次
                    for i in range(rotate_steps):
                        velocity_command = {
                            'type': 'velocity_command',
                            'data': {
                                'linear_x': 0.0,
                                'angular_z': 2.0  # 提高旋转速度
                            }
                        }
                        self._send_to_bridge(velocity_command)
                        time.sleep(0.4)  # 减少每次旋转时间，加快扫描
                    
                    # 旋转后再次检测前沿
                    frontier_region_centers_unscored, filtered_map, num_large_regions = self.frontier_planner.get_frontier_centers_given_obs_map(self.mapper.obs_map)
                    
                    if len(frontier_region_centers_unscored) == 0:
                        print("Step {}: 旋转后仍未发现前沿，探索完成".format(t))
                        break
                    else:
                        print("Step {}: 旋转后发现{}个前沿，继续探索".format(t, len(frontier_region_centers_unscored)))
            
            # === Hector模式特殊处理 ===
            if mode == 'hector' or mode == 'hectoraug':
                do_hector_plan = (t % 1 == 0)
                if do_hector_plan:
                    frontier_region_centers_unscored, filtered_map, num_large_regions = self.frontier_planner.get_frontier_centers_given_obs_map(self.mapper.obs_map)
                    frontier_region_centers = frontier_region_centers_unscored
                    frontier_cost_list = np.zeros(len(frontier_region_centers))
                    
                    if mode == 'hectoraug':
                        hector_cur_obs_img = self.mapper.obs_map.copy()
                        cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz = \
                            get_lama_pred_from_obs(hector_cur_obs_img, self.lama_model, self.lama_map_transform, self.collect_opts.lama_device)
                        pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)
                        hector_frontier_region_centers_unscored = frontier_region_centers_unscored
                        padded_obs_map = get_padded_obs_map(self.mapper.obs_map)
                        
                        frontier_region_centers, frontier_cost_list, viz_most_flooded_grid, viz_medium_flooded_grid, best_ind, medium_ind = \
                                self.frontier_planner.score_frontiers(hector_frontier_region_centers_unscored, cur_pose, \
                                                                    pose_list, pred_maputils, self.collect_opts.pred_vis_configs, \
                                                                        obs_map=padded_obs_map, mean_map=None, var_map=None)

            # === 获取膨胀障碍物地图用于局部规划 ===
            occ_grid_pyastar = self.mapper.get_inflated_planning_maps(unknown_as_occ=self.collect_opts.unknown_as_occ)
            
            # === 非Hector方法的前沿选择 ===
            if mode not in ['hector', 'hectoraug']:
                # 检查是否接近锁定的前沿中心
                if locked_frontier_center is not None:
                    if np.linalg.norm(locked_frontier_center - cur_pose) < self.collect_opts.cur_pose_dist_threshold_m * pixel_per_meter:
                        locked_frontier_center = None

                need_new_locked_frontier = False
                if mode == 'upen':
                    if locked_frontier_center is None:
                        need_new_locked_frontier = True
                    else: 
                        upen_goal_pose_freq = self.collect_opts.upen_config['goal_pose_freq']
                        at_goal_pose_freq = t % upen_goal_pose_freq == 0
                        lock_frontier_center_is_invalid = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, self.collect_opts, pixel_per_meter)  
                        cannot_reach_frontier_center = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False) is None
                        need_new_locked_frontier = at_goal_pose_freq or lock_frontier_center_is_invalid or cannot_reach_frontier_center
                        if need_new_locked_frontier and debug_step:
                            print("UPEN: freq: {}, invalid: {}, cannot reach: {}".format(at_goal_pose_freq, lock_frontier_center_is_invalid, cannot_reach_frontier_center))
                else:
                    need_new_locked_frontier = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, self.collect_opts, pixel_per_meter)
                                
                if need_new_locked_frontier:
                    show_plt = True
                                                    
                    # === 地图预测 - 原版逻辑 ===
                    pred_maputils = None
                    var_map = None
                    mean_map = None
                    
                    # 先进行地图填充，确保尺寸一致性
                    padded_obs_map = get_padded_obs_map(self.mapper.obs_map)
                    
                    if use_model and self.lama_model:
                        # 使用填充后的地图进行LAMA预测，确保尺寸一致
                        cur_obs_img = padded_obs_map.copy()
                        
                        # LAMA全局预测
                        cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz = \
                            get_lama_pred_from_obs(cur_obs_img, self.lama_model, self.lama_map_transform, self.collect_opts.lama_device)
                        
                        # 获取ensemble预测
                        lama_pred_list = []
                        for model_i, model in enumerate(self.model_list):
                            if debug_step:
                                print("predicting with model: ", model_i)
                            pred_time_start = time.time()
                            lama_pred = model(input_lama_batch)
                            if debug_step:
                                print("Prediction took {} seconds".format(np.round(time.time() - pred_time_start, 2)))
                            lama_pred_onechan = lama_pred['inpainted'][0][0]
                            lama_pred_list.append(lama_pred_onechan)

                        # 获取方差
                        if len(lama_pred_list) > 0:
                            lama_pred_list = torch.stack(lama_pred_list)
                            var_map = torch.var(lama_pred_list, dim=0)
                            mean_map = np.mean(lama_pred_list.cpu().numpy(), axis=0)

                        pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)

                    if mode == 'upen':
                        # UPEN RRT规划 - 原版逻辑
                        upen_path = None
                        while upen_path is None:
                            assert self.collect_opts.upen_config is not None, "upen_config must be provided in yml"
                            upen_rrt_max_iters = self.collect_opts.upen_config['rrt_max_iters']
                            upen_expand_dis = self.collect_opts.upen_config['expand_dis']
                            upen_goal_sample_rate = self.collect_opts.upen_config['goal_sample_rate']
                            upen_connect_circle_dist = self.collect_opts.upen_config['connect_circle_dist']
                            upen_rrt_num_path = self.collect_opts.upen_config['rrt_num_path']
                            upen_rrt_straight_line = self.collect_opts.upen_config['rrt_straight_line']
                            upen_reach_horizon = self.collect_opts.upen_config['reach_horizon']
                            
                            pose_coords = torch.tensor([[[cur_pose[1], cur_pose[0]]]]).cuda()
                            planning_grid = torch.tensor(padded_obs_map).cuda() 
                            end_pose = [planning_grid.shape[0]-20, planning_grid.shape[1]-20]
                            goal_pose_coords = torch.tensor([[[end_pose[1], end_pose[0]]]]).cuda()
                            
                            occ_chan = 1
                            ensemble = torch.zeros((lama_pred_list.shape[0], 1, 2, planning_grid.shape[0], planning_grid.shape[1])).cuda()
                            ensemble[:, 0, occ_chan, :, :] = lama_pred_list
                            
                            upen_rrt_goal, upen_rrt_best_path, upen_path_dict = upen_baseline.get_rrt_goal(
                                pose_coords=pose_coords.clone(), goal=goal_pose_coords.clone(), grid=planning_grid, ensemble=ensemble,  
                                rrt_max_iters=upen_rrt_max_iters, expand_dis=upen_expand_dis, goal_sample_rate=upen_goal_sample_rate, 
                                connect_circle_dist=upen_connect_circle_dist, rrt_num_path=upen_rrt_num_path,  
                                rrt_straight_line=upen_rrt_straight_line, reach_horizon=upen_reach_horizon, upen_mode='exploration')
                            
                            if upen_rrt_goal is not None:
                                intermediate_goal_pose = upen_rrt_goal[0][0].cpu().numpy()
                                intermediate_goal_pose = intermediate_goal_pose[::-1]
                            else:
                                print("No intermediate goal found, using A* to get intermediate goal")
                                astar_path_to_goal = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, end_pose, allow_diagonal=False)
                                if astar_path_to_goal is not None:
                                    intermediate_goal_pose = astar_path_to_goal[np.min([20, len(astar_path_to_goal)-1])]
                                else:
                                    print("A* path to goal also failed")
                                    break
                            
                            locked_frontier_center = intermediate_goal_pose
                            upen_path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    
                    else:
                        # 基于前沿的方法：重新计算所有前沿中心分数 - 原版逻辑
                        frontier_region_centers_unscored, filtered_map, num_large_regions = self.frontier_planner.get_frontier_centers_given_obs_map(self.mapper.obs_map)
                        
                        if len(frontier_region_centers_unscored) == 0:
                            print("Step {}: 没有发现前沿，探索完成".format(t))
                            break
                        
                        frontier_region_centers, frontier_cost_list, viz_most_flooded_grid, viz_medium_flooded_grid, best_ind, medium_ind = \
                                self.frontier_planner.score_frontiers(frontier_region_centers_unscored, cur_pose, \
                                                                    pose_list, pred_maputils, self.collect_opts.pred_vis_configs, \
                                                                        obs_map=padded_obs_map, mean_map=mean_map, var_map=var_map)
                        
                        if len(frontier_region_centers) == 0:
                            print("Step {}: 前沿评分后无有效前沿，探索完成".format(t))
                            break
                        
                        locked_frontier_center = frontier_region_centers[np.argmin(frontier_cost_list)]

                        # 前沿验证
                        while not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, self.collect_opts, pixel_per_meter):
                            frontier_selected, locked_frontier_center, frontier_region_centers, frontier_cost_list = reselect_frontier_from_frontier_region_centers(frontier_region_centers, frontier_cost_list, t, start_exp_time)
                            if not frontier_selected:
                                print("Step {}: 所有前沿均不可达，探索完成".format(t))
                                self.exploration_complete = True
                                break

            else:
                # Hector Explorer Baseline - 原版逻辑
                if do_hector_plan:
                    obstacle_transform_map = np.ones(self.mapper.obs_map.shape)
                    obs_obstacles = np.where(self.mapper.obs_map == 1)
                    obstacle_transform_map[obs_obstacles] = 0
                    obstacle_transform_map = cv2.distanceTransform(obstacle_transform_map.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) 
                    
                    obs_ramp_dist = 15
                    obs_min_dist = 2
                    danger_transform_map = np.zeros(self.mapper.obs_map.shape)
                    danger_transform_map[obstacle_transform_map < obs_ramp_dist] = (obs_ramp_dist - obstacle_transform_map[obstacle_transform_map < obs_ramp_dist]) * 5                
                    
                    if mode == 'hector':
                        init_frontier_value = None
                    elif mode == 'hectoraug':
                        init_frontier_value = -1 * frontier_cost_list
                    else:
                        raise ValueError("Invalid mode: {}, mode not available for getting init_frontier_value".format(mode))
                    
                    cost_transform_map = get_hector_exploration_transform_map(self.mapper.obs_map, frontier_region_centers, init_cost=danger_transform_map, mode=mode, infogain_val_list=init_frontier_value, info_gain_weight=200)

            if self.exploration_complete:
                break
            
            # === 局部规划 - 原版逻辑 ===
            chosen_local_planner = determine_local_planner(mode)
            
            if chosen_local_planner == 'astar':
                # 关键修复：验证目标坐标是否在当前地图范围内
                map_height, map_width = occ_grid_pyastar.shape
                
                if locked_frontier_center is None:
                    print("Step {}: 没有有效的前沿目标，探索完成".format(t))
                    break
                
                # 检查并调整目标坐标
                if (locked_frontier_center[0] >= map_height or locked_frontier_center[1] >= map_width or
                    locked_frontier_center[0] < 0 or locked_frontier_center[1] < 0):
                    print("目标坐标 {} 超出地图范围 {}".format(locked_frontier_center, (map_height, map_width)))
                    
                    adjusted_target = np.array([
                        np.clip(locked_frontier_center[0], 0, map_height - 1),
                        np.clip(locked_frontier_center[1], 0, map_width - 1)
                    ])
                    
                    if occ_grid_pyastar[adjusted_target[0], adjusted_target[1]] != np.inf:
                        print("调整目标坐标为: {}".format(adjusted_target))
                        locked_frontier_center = adjusted_target
                    else:
                        print("调整后的坐标仍在障碍物中，寻找附近可用坐标...")
                        found_valid = False
                        for radius in range(1, 20):
                            for dx in [-radius, 0, radius]:
                                for dy in [-radius, 0, radius]:
                                    test_x = np.clip(adjusted_target[0] + dx, 0, map_height - 1)
                                    test_y = np.clip(adjusted_target[1] + dy, 0, map_width - 1)
                                    if occ_grid_pyastar[test_x, test_y] != np.inf:
                                        locked_frontier_center = np.array([test_x, test_y])
                                        print("找到附近可用坐标: {}".format(locked_frontier_center))
                                        found_valid = True
                                        break
                                if found_valid:
                                    break
                            if found_valid:
                                break
                        
                        if not found_valid:
                            print("无法找到有效的目标坐标，跳过此目标")
                            locked_frontier_center = None
                
                path = None
                if locked_frontier_center is not None:
                    try:
                        path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    except ValueError as e:
                        print("A*路径规划参数错误: {}".format(e))
                        path = None
                
                # A*路径规划失败处理 - 原版逻辑
                while path is None:
                    if debug_step:
                        print("A*路径规划失败，尝试重新选择前沿...")
                    frontier_selected, locked_frontier_center, frontier_region_centers, frontier_cost_list = reselect_frontier_from_frontier_region_centers(frontier_region_centers, frontier_cost_list, t, start_exp_time)
                    if not frontier_selected:
                        print("路径规划失败，所有前沿均不可达，探索完成")
                        self.exploration_complete = True
                        break
                    
                    # 验证新选择的目标
                    if (locked_frontier_center[0] >= map_height or locked_frontier_center[1] >= map_width or
                        locked_frontier_center[0] < 0 or locked_frontier_center[1] < 0):
                        print("新目标 {} 也超出范围，调整...".format(locked_frontier_center))
                        locked_frontier_center = np.array([
                            np.clip(locked_frontier_center[0], 0, map_height - 1),
                            np.clip(locked_frontier_center[1], 0, map_width - 1)
                        ])
                    
                    try:
                        path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    except ValueError as e:
                        print("新目标A*规划也失败: {}".format(e))
                        path = None
                
                if self.exploration_complete:
                    break
                
                if debug_step:
                    print("A*路径规划成功: 路径长度 {}".format(len(path)))
                
                plan_x = path[:,0]
                plan_y = path[:,1]        
                    
                next_pose = sim_utils.psuedo_traj_controller(plan_x, plan_y, plan_ind_to_use=ind_to_move_per_step)
                
            elif chosen_local_planner == 'gradient':
                for _ in range(ind_to_move_per_step):
                    next_pose = gradient_planner(cur_pose, cost_transform_map=cost_transform_map) 
                    cur_pose = next_pose
            else:
                raise ValueError("Invalid local planner: {}".format(chosen_local_planner))

            # === 执行移动（使用优化的安全速度控制）===
            # 转换像素坐标到世界坐标
            world_target = np.array([
                (next_pose[0] - self.mapper.obs_map.shape[0]//2) / pixel_per_meter,
                (next_pose[1] - self.mapper.obs_map.shape[1]//2) / pixel_per_meter
            ])
            
            # === 防卡死检测和处理 ===
            is_stuck, anti_stuck_linear, anti_stuck_angular = check_and_handle_stuck_situation(
                self, self.robot_pose, self.mapper.obs_map
            )
            
            if is_stuck:
                # 使用防卡死速度命令
                linear_vel = anti_stuck_linear
                angular_vel = anti_stuck_angular
                print(f"🚨 使用防卡死速度命令: linear={linear_vel:.2f}, angular={angular_vel:.2f}")
            else:
                # 计算正常的安全速度命令
                linear_vel, angular_vel = compute_velocity_to_target_safe(
                    world_target, self.robot_pose, self.collect_opts
                )
            
            # 发送速度命令
            velocity_command = {
                'type': 'velocity_command',
                'data': {
                    'linear_x': float(linear_vel),
                    'angular_z': float(angular_vel)
                }
            }
            
            success = self._send_to_bridge(velocity_command)
            
            if debug_step or is_stuck:
                status_msg = " [防卡死模式]" if is_stuck else ""
                print("Step {}: Target {}, Vel: [{:.2f}, {:.2f}]{}".format(
                    t, world_target, linear_vel, angular_vel, status_msg))
            
            # === 更新位姿 - 原版逻辑 ===
            # 检查坐标是否在地图范围内
            map_height, map_width = self.mapper.gt_map.shape
            if (next_pose[0] >= map_height or next_pose[1] >= map_width or
                next_pose[0] < 0 or next_pose[1] < 0):
                print("警告：next_pose {} 超出地图范围 {}，调整坐标".format(next_pose, (map_height, map_width)))
                next_pose = np.array([
                    np.clip(next_pose[0], 0, map_height - 1),
                    np.clip(next_pose[1], 0, map_width - 1)
                ])
                print("调整后坐标: {}".format(next_pose))
            
            # 检查是否撞墙
            if self.mapper.gt_map[next_pose[0], next_pose[1]] == 1:
                print("Hit wall!")
                break
            
            pose_list = np.concatenate([pose_list, np.atleast_2d(next_pose)], axis=0)
            cur_pose = next_pose
            
            # 观测：由于使用SLAM地图，跳过mapper的observe操作
            # mapper.observe_and_accumulate_given_pose(cur_pose)  # 注释掉，使用SLAM地图更新
            
            if debug_step:
                print("Total time for step {} is {} seconds".format(t, np.round(time.time() - start_mission_i_time, 2)))
                
                # 防卡死状态监控
                current_time = time.time()
                if hasattr(self, 'last_move_time'):
                    idle_time = current_time - self.last_move_time
                    if idle_time > 5.0:  # 超过5秒没移动就显示警告
                        print(f"⚠️  机器人空闲时间: {idle_time:.1f}s (阈值: {self.stuck_threshold_time}s)")
                
                if hasattr(self, 'in_anti_stuck_mode') and self.in_anti_stuck_mode:
                    anti_stuck_duration = current_time - self.anti_stuck_start_time
                    print(f"🔄 防卡死模式运行中: {anti_stuck_duration:.1f}s")
            
            # 短暂延时，确保系统有时间处理新数据和更新地图
            time.sleep(0.2)  # 增加到200ms，给数据处理更多时间
            
            # 额外的数据同步等待机制
            if t % 5 == 0:  # 每5步进行一次较长的同步等待
                print(f"Step {t}: 数据同步等待中...")
                time.sleep(0.5)  # 额外500ms等待

        print("=== 官方MapEx探索循环结束 ===")
        
        # === 生成最终统计报告 ===
        if self.current_slam_map:
            print("\n" + "="*80)
            print("FINAL EXPLORATION REPORT")
            print("="*80)
            
            obstacle_config = self.collect_opts.get('obstacle_detection', {})
            final_stats = analyze_map_statistics(self.current_slam_map, obstacle_config)
            
            if final_stats:
                print_detailed_map_stats(final_stats, t)
                
                # 保存最终可视化地图
                current_frontiers = None
                if 'frontier_region_centers_unscored' in locals():
                    current_frontiers = frontier_region_centers_unscored
                elif 'frontier_region_centers' in locals():
                    current_frontiers = frontier_region_centers
                
                # 获取最终膨胀地图
                final_inflated_map = None
                if hasattr(self, 'mapper') and self.mapper:
                    try:
                        final_inflated_map = self.mapper.get_inflated_planning_maps(
                            unknown_as_occ=self.collect_opts.unknown_as_occ
                        )
                    except Exception as e:
                        print("获取最终膨胀地图失败: {}".format(e))
                        final_inflated_map = None
                
                final_visualization_stats = visualize_and_save_map(
                    self.current_slam_map, 
                    self.robot_pose, 
                    f"final_{t}", 
                    self.debug_output_dir, 
                    obstacle_config,
                    frontier_centers=current_frontiers,
                    inflated_map=final_inflated_map  # 新增参数
                )
                
                # 保存最终统计数据到JSON文件
                final_report = {
                    'exploration_summary': {
                        'total_steps': t,
                        'exploration_time': time.time() - start_exp_time,
                        'final_pose': self.robot_pose,
                        'map_statistics': final_stats
                    },
                    'threshold_analysis': {
                        'current_thresholds': final_stats['thresholds'],
                        'recommendations': self._generate_threshold_recommendations(final_stats)
                    }
                }
                
                report_file = os.path.join(self.debug_output_dir, f'exploration_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                os.makedirs(self.debug_output_dir, exist_ok=True)
                with open(report_file, 'w') as f:
                    json.dump(final_report, f, indent=2)
                
                print("Final exploration report saved to: {}".format(report_file))
                print("="*80)
        
        self._finish_exploration()
    
    def _send_to_bridge(self, message):
        if not self.connected or not self.socket_client:
            return False
        
        try:
            data = json.dumps(message).encode('utf-8')
            self.socket_client.setblocking(False)
            self.socket_client.send(data + b'\n')
            return True
        except Exception as e:
            return False
    
    def _finish_exploration(self):
        print("完成MapEx探索")
        
        # 停止机器人
        self._send_to_bridge({
            'type': 'velocity_command',
            'data': {
                'linear_x': 0.0,
                'angular_z': 0.0
            }
        })
        
        # 发送完成状态
        self._send_to_bridge({
            'type': 'exploration_status',
            'data': 'EXPLORATION_COMPLETED'
        })
        
        self.exploration_active = False
        self.exploration_complete = True

    def _generate_threshold_recommendations(self, stats):
        """基于统计数据生成阈值推荐"""
        recommendations = {}
        
        # 分析当前阈值效果
        free_ratio = stats['free_ratio']
        obstacle_ratio = stats['obstacle_ratio']
        uncertain_ratio = stats['uncertain_ratio']
        avg_prob = stats['avg_probability']
        std_prob = stats['std_probability']
        
        current_free_thresh = stats['thresholds']['free_threshold']
        current_obstacle_thresh = stats['thresholds']['obstacle_threshold']
        
        # 推荐策略
        if uncertain_ratio > 30:  # 不确定区域太多
            recommendations['status'] = 'Too much uncertain area detected'
            recommendations['free_threshold'] = {
                'current': current_free_thresh,
                'recommended': max(10, current_free_thresh - 10),
                'reason': 'Increase free space sensitivity to reduce uncertain areas'
            }
            recommendations['obstacle_threshold'] = {
                'current': current_obstacle_thresh,
                'recommended': min(90, current_obstacle_thresh + 5),
                'reason': 'Increase obstacle confidence to reduce false positives'
            }
        elif uncertain_ratio < 5:  # 不确定区域太少，可能阈值太激进
            recommendations['status'] = 'Very low uncertain area - thresholds might be too aggressive'
            recommendations['free_threshold'] = {
                'current': current_free_thresh,
                'recommended': min(30, current_free_thresh + 5),
                'reason': 'Slight increase to be more conservative about free space'
            }
            recommendations['obstacle_threshold'] = {
                'current': current_obstacle_thresh,
                'recommended': max(70, current_obstacle_thresh - 5),
                'reason': 'Slight decrease to be more sensitive to obstacles'
            }
        else:
            recommendations['status'] = 'Thresholds appear well-balanced'
            recommendations['free_threshold'] = {
                'current': current_free_thresh,
                'recommended': current_free_thresh,
                'reason': 'Current threshold is working well'
            }
            recommendations['obstacle_threshold'] = {
                'current': current_obstacle_thresh,
                'recommended': current_obstacle_thresh,
                'reason': 'Current threshold is working well'
            }
        
        # 基于标准差的额外建议
        if std_prob > 25:
            recommendations['note'] = 'High probability variance detected - consider environment-specific tuning'
        
        return recommendations

    def run(self):
        print("启动SLAM MapEx Explorer...")
        
        # 1. 初始化MapEx组件
        if not self.initialize_mapex_components():
            print("MapEx组件初始化失败")
            return False
        
        # 2. 启动Socket通信
        self.start_socket_communication()
        
        # 3. 等待连接
        for i in range(60):
            if self.connected:
                break
            time.sleep(1.0)
            print("等待桥接连接... ({}/60)".format(i+1))
        
        if not self.connected:
            print("连接桥接节点超时")
            return False
        
        print("SLAM MapEx Explorer已准备就绪，等待探索命令...")
        
        # 4. 主循环
        while self.running:
            time.sleep(1.0)
            
            if self.exploration_complete:
                print("探索已完成，准备退出...")
                break
        
        return True
    
    def shutdown(self):
        """关闭Explorer"""
        print("正在关闭SLAM MapEx Explorer...")
        
        self.running = False
        
        if self.exploration_active:
            self.exploration_active = False
        
        self.connected = False
        if self.socket_client:
            self.socket_client.close()
            self.socket_client = None
        
        if self.socket_thread and self.socket_thread.is_alive():
            self.socket_thread.join(timeout=3.0)
        
        print("SLAM MapEx Explorer已关闭")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='base.yaml', help='配置文件名')
    args = parser.parse_args()
    
    print("=== SLAM集成MapEx探索器 (官方算法适配版) ===")
    print("配置文件: {}".format(args.config_name))
    print("模式: 官方MapEx算法 + SLAM地图输入 + Socket通信")
    print("适配: 保持原版算法逻辑，替换数据源和控制输出")
    
    explorer = SLAMMapExExplorer(config_name=args.config_name)
    
    try:
        success = explorer.run()
        
        if success:
            print("SLAM MapEx探索器正常结束")
        else:
            print("SLAM MapEx探索器异常结束")
    except KeyboardInterrupt:
        print("接收到中断信号，正在关闭...")
        explorer.shutdown()
    except Exception as e:
        print("探索器运行错误: {}".format(e))
        explorer.shutdown()

if __name__ == '__main__':
    main()