#!/usr/bin/env python3
"""
可视化模块
"""

import numpy as np
from typing import List
from data_structures import *

class FluentCoverageVisualizer:
    """流畅覆盖区域可视化器"""
    
    def __init__(self, world):
        from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, UsdShade, Sdf
        
        self.UsdPhysics = UsdPhysics
        self.UsdGeom = UsdGeom
        self.Gf = Gf
        
        self.world = world
        self.coverage_marks = {}
        self.mark_prims = {}
        self.coverage_container = "/World/CoverageMarks"
        self.last_marked_position = None
        self.mark_counter = 0
    
    def mark_coverage_realtime(self, robot_position: np.ndarray):
        """实时标记"""
        self.mark_counter += 1
        fine_grid_pos = self._ultra_fine_quantize_position(robot_position)
        
        if self._should_create_intelligent_dispersed_mark(fine_grid_pos):
            self._create_ultra_smooth_coverage_mark(fine_grid_pos)
            self.last_marked_position = fine_grid_pos.copy()
    
    def _should_create_intelligent_dispersed_mark(self, current_pos: np.ndarray) -> bool:
        """判断是否应该创建新标记"""
        if self.last_marked_position is None:
            return True
        
        from data_structures import MARK_DISTANCE_THRESHOLD
        distance_to_last = np.linalg.norm(current_pos[:2] - self.last_marked_position[:2])
        return bool(distance_to_last >= MARK_DISTANCE_THRESHOLD)
    
    def _ultra_fine_quantize_position(self, position: np.ndarray) -> np.ndarray:
        """量化位置"""
        from data_structures import FINE_GRID_SIZE
        x = round(position[0] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        y = round(position[1] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        return np.array([x, y, 0.015])
    
    def _create_ultra_smooth_coverage_mark(self, position: np.ndarray):
        """创建覆盖标记"""
        stage = self.world.stage
        
        if not stage.GetPrimAtPath(self.coverage_container):
            stage.DefinePrim(self.coverage_container, "Xform")
        
        mark_id = len(self.coverage_marks)
        x_str = f"{position[0]:.3f}".replace(".", "p").replace("-", "N")
        y_str = f"{position[1]:.3f}".replace(".", "p").replace("-", "N")
        pos_key = f"{x_str}_{y_str}"
        mark_path = f"{self.coverage_container}/SmoothMark_{mark_id}_{pos_key}"
        
        if pos_key in self.coverage_marks:
            self.coverage_marks[pos_key] += 1
            return
        else:
            self.coverage_marks[pos_key] = 1
        
        coverage_count = self.coverage_marks[pos_key]
        
        mark_prim = stage.DefinePrim(mark_path, "Cylinder")
        cylinder_geom = self.UsdGeom.Cylinder(mark_prim)
        
        from data_structures import COVERAGE_MARK_RADIUS
        mark_radius = COVERAGE_MARK_RADIUS * 0.7
        cylinder_geom.CreateRadiusAttr().Set(mark_radius)
        cylinder_geom.CreateHeightAttr().Set(0.008)
        
        xform = self.UsdGeom.Xformable(mark_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(self.Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        
        self.UsdPhysics.RigidBodyAPI.Apply(mark_prim)
        rigid_body = self.UsdPhysics.RigidBodyAPI(mark_prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(False)
        
        self._set_ultra_smooth_color(mark_prim, coverage_count)
        self.mark_prims[pos_key] = mark_path
    
    def _set_ultra_smooth_color(self, mark_prim, coverage_count: int):
        """设置5档超精细灰度渐变 - 从极浅灰到深黑"""
        # 5档精细渐变
        coverage_level = min(coverage_count, 5)
        
        # 从0.95(极浅灰)到0.05(深黑)，分5档
        gray_intensity = 0.95 - (coverage_level - 1) * 0.06  # (0.95-0.05)/5 = 0.06
        
        color_value = float(gray_intensity)
        
        gprim = self.UsdGeom.Gprim(mark_prim)
        gprim.CreateDisplayColorAttr().Set([(color_value, color_value, color_value)])
    
    def cleanup(self):
        """清理覆盖标记"""
        stage = self.world.stage
        
        for pos_key, mark_path in self.mark_prims.items():
            if stage.GetPrimAtPath(mark_path):
                stage.RemovePrim(mark_path)
        
        if stage.GetPrimAtPath(self.coverage_container):
            stage.RemovePrim(self.coverage_container)
            
        self.coverage_marks.clear()
        self.mark_prims.clear()
        self.last_marked_position = None
        
        print("超丝滑覆盖标记清理完成")

class RealMovementVisualizer:
    """真实移动可视化器 - 集成超丝滑流畅覆盖可视化"""
    
    def __init__(self, world):
        from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, UsdShade, Sdf
        
        self.UsdPhysics = UsdPhysics
        self.UsdGeom = UsdGeom
        self.UsdShade = UsdShade
        self.Gf = Gf
        self.Sdf = Sdf
        
        self.world = world
        self.stage = world.stage
        
        # USD资产路径
        self.real_robot_usd = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm4.usd"
        self.ghost_robot_usd = "/home/getting/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"
        
        # 容器路径
        self.path_container = "/World/CompletePath"
        self.ghost_container = "/World/SyncGhosts" 
        
        # 状态变量
        self.all_path_points = []
        self.ghost_robots = []
        
        # 集成超丝滑流畅覆盖可视化器
        self.fluent_coverage_visualizer = FluentCoverageVisualizer(world)
        
        print("初始化真实移动可视化器（集成超丝滑覆盖 + 智能物理距离分散）")
    
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
            line_geom = self.UsdGeom.BasisCurves(line_prim)
            
            line_geom.CreateTypeAttr().Set("linear")
            line_geom.CreateBasisAttr().Set("bspline")
            
            # 构建段内路径点
            segment_points = []
            for point_idx in segment:
                point = self.all_path_points[point_idx]
                world_pos = self.Gf.Vec3f(float(point.position[0]), float(point.position[1]), 0.05)
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
        
        # 距离太大可能跨越障碍物
        if distance > COVERAGE_CELL_SIZE * 3:
            return False
        
        return True
    
    def _create_ghost_robots(self):
        """创建虚影机器人 - 基于物理距离智能分散"""
        self.stage.DefinePrim(self.ghost_container, "Xform")
        
        # 选择关键位置放置虚影机器人 - 使用智能分散算法
        ghost_indices = self._select_ghost_positions()
        
        print(f"    创建智能分散虚影机器人...")
        
        for i, point_idx in enumerate(ghost_indices):
            point = self.all_path_points[point_idx]
            ghost_path = f"{self.ghost_container}/GhostRobot_{i}"
            
            # 创建虚影机器人
            ghost_prim = self.stage.DefinePrim(ghost_path, "Xform")
            references = ghost_prim.GetReferences()
            references.AddReference(self.ghost_robot_usd)
            
            # 设置位置和朝向
            position = self.Gf.Vec3d(float(point.position[0]), float(point.position[1]), 0.0)
            yaw_degrees = float(np.degrees(point.orientation))
            
            xform = self.UsdGeom.Xformable(ghost_prim)
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
            
            print(f"      虚影{i+1}: 位置[{point.position[0]:.2f}, {point.position[1]:.2f}], 朝向{yaw_degrees:.1f}°")
        
        print(f"    智能分散虚影机器人创建完成: {len(ghost_indices)}个, 物理距离优化分布")
    
    def _select_ghost_positions(self) -> List[int]:
        """智能选择虚影位置 - 基于物理距离分散算法"""
        total_points = len(self.all_path_points)
        max_ghosts = min(MAX_GHOST_ROBOTS, total_points)
        
        if total_points <= max_ghosts:
            return list(range(total_points))
        
        print(f"    开始智能物理距离分散算法: {total_points}个路径点，目标{max_ghosts}个虚影")
        
        selected_indices = []
        
        # 1. 必须包含起点和终点
        selected_indices.append(0)
        if total_points > 1:
            selected_indices.append(total_points - 1)
        
        # 2. 使用贪心算法选择物理距离最分散的点
        min_distance_threshold = 2.0  # 最小间距阈值(米)
        remaining_slots = max_ghosts - len(selected_indices)
        
        # 候选点池：排除已选择的起点和终点
        candidate_indices = [i for i in range(1, total_points - 1)]
        
        for slot in range(remaining_slots):
            if not candidate_indices:
                break
            
            best_candidate = None
            best_min_distance = 0
            
            # 对每个候选点，计算它到已选点的最小距离
            for candidate_idx in candidate_indices:
                candidate_pos = self.all_path_points[candidate_idx].position
                
                # 计算到所有已选点的最小距离
                min_distance_to_selected = float('inf')
                for selected_idx in selected_indices:
                    selected_pos = self.all_path_points[selected_idx].position
                    distance = np.linalg.norm(candidate_pos[:2] - selected_pos[:2])
                    min_distance_to_selected = min(min_distance_to_selected, distance)
                
                # 选择与已选点距离最远的候选点
                if min_distance_to_selected > best_min_distance:
                    best_min_distance = min_distance_to_selected
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                candidate_indices.remove(best_candidate)
                print(f"      选择虚影点 {len(selected_indices)}/{max_ghosts}: 索引{best_candidate}, 最小间距{best_min_distance:.2f}m")
            else:
                break
        
        # 3. 如果还有空位，优先选择有物体交互的点（但要满足距离约束）
        if len(selected_indices) < max_ghosts:
            object_indices = [i for i, p in enumerate(self.all_path_points) if p.has_object and i not in selected_indices]
            
            for obj_idx in object_indices:
                if len(selected_indices) >= max_ghosts:
                    break
                
                obj_pos = self.all_path_points[obj_idx].position
                
                # 检查是否满足最小距离约束
                min_distance_to_selected = float('inf')
                for selected_idx in selected_indices:
                    selected_pos = self.all_path_points[selected_idx].position
                    distance = np.linalg.norm(obj_pos[:2] - selected_pos[:2])
                    min_distance_to_selected = min(min_distance_to_selected, distance)
                
                if min_distance_to_selected >= min_distance_threshold * 0.7:  # 对物体点放宽约束
                    selected_indices.append(obj_idx)
                    print(f"      额外选择物体交互点: 索引{obj_idx}, 间距{min_distance_to_selected:.2f}m")
        
        # 4. 验证分散效果
        selected_indices_sorted = sorted(selected_indices)
        print(f"    物理距离分散验证:")
        for i, idx in enumerate(selected_indices_sorted):
            pos = self.all_path_points[idx].position
            print(f"      虚影{i+1}: 索引{idx}, 位置[{pos[0]:.2f}, {pos[1]:.2f}]")
        
        # 计算平均间距
        if len(selected_indices_sorted) > 1:
            total_distance = 0
            distance_count = 0
            for i in range(len(selected_indices_sorted)):
                for j in range(i+1, len(selected_indices_sorted)):
                    pos1 = self.all_path_points[selected_indices_sorted[i]].position
                    pos2 = self.all_path_points[selected_indices_sorted[j]].position
                    distance = np.linalg.norm(pos1[:2] - pos2[:2])
                    total_distance += distance
                    distance_count += 1
            
            avg_distance = total_distance / distance_count if distance_count > 0 else 0
            print(f"    平均虚影间距: {avg_distance:.2f}m")
            
            # 找出最小间距
            min_distance = float('inf')
            for i in range(len(selected_indices_sorted)):
                for j in range(i+1, len(selected_indices_sorted)):
                    pos1 = self.all_path_points[selected_indices_sorted[i]].position
                    pos2 = self.all_path_points[selected_indices_sorted[j]].position
                    distance = np.linalg.norm(pos1[:2] - pos2[:2])
                    min_distance = min(min_distance, distance)
            print(f"    最小虚影间距: {min_distance:.2f}m")
        
        print(f"    智能物理距离分散完成: 选择{len(selected_indices_sorted)}个虚影点")
        
        return selected_indices_sorted
    
    def mark_coverage_realtime(self, robot_position: np.ndarray, step_count: int):
        """超丝滑实时标记覆盖区域 - 无延迟标记"""
        # 调用集成的超丝滑流畅覆盖可视化器
        self.fluent_coverage_visualizer.mark_coverage_realtime(robot_position)
    
    def _disable_ghost_physics_safe(self, root_prim):
        """安全禁用虚影物理属性"""
        def disable_recursive(prim):
            try:
                if prim.IsA(self.UsdGeom.Xformable):
                    if not prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                        self.UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body = self.UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                    
                    if not prim.HasAPI(self.UsdPhysics.CollisionAPI):
                        self.UsdPhysics.CollisionAPI.Apply(prim)
                    collision = self.UsdPhysics.CollisionAPI(prim)
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
        material = self.UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = self.UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set((0.2, 0.9, 0.2))
        shader.CreateInput("emissiveColor", self.Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 0.1))
        shader.CreateInput("opacity", self.Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.8)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", self.Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        # 应用到mesh
        def apply_material_recursive(prim):
            if prim.GetTypeName() == "Mesh":
                self.UsdShade.MaterialBindingAPI(prim).Bind(material)
            for child in prim.GetChildren():
                apply_material_recursive(child)
        
        apply_material_recursive(ghost_prim)
    
    def _setup_path_material(self, line_prim):
        """设置路径材质"""
        material_path = "/World/Materials/PathMaterial"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = self.UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = self.UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 1.0))
        shader.CreateInput("emissiveColor", self.Sdf.ValueTypeNames.Color3f).Set((0.05, 0.25, 0.5))
        shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.2)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", self.Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        self.UsdShade.MaterialBindingAPI(line_prim).Bind(material)
    
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
        # 清理超丝滑流畅覆盖可视化器
        if self.fluent_coverage_visualizer:
            self.fluent_coverage_visualizer.cleanup()
        print("可视化资源清理完成")