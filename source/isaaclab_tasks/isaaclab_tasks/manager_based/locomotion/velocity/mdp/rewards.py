# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

# ==============================================================================
# TERRAIN-ADAPTIVE REWARD FUNCTIONS (Go2 SDK-BASED)
# ==============================================================================
# These functions leverage Go2 SDK sensors for terrain-aware locomotion control:
# - IMU sensors (roll, pitch, yaw from quaternion/euler angles)
# - Height scanner (terrain slope detection)
# - Joint position sensors (leg configuration adaptation) 
# - Foot force sensors (contact and gait patterns)
# ==============================================================================

def base_orientation_alignment_to_terrain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    alignment_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward robot body orientation alignment to terrain slope using Go2 SDK sensors.
    
    This function uses:
    - Go2's IMU (quaternion → roll/pitch/yaw) for current body orientation
    - Height scanner for terrain slope estimation
    - Adaptive threshold for orientation tolerance
    
    On flat terrain: Rewards keeping body parallel to ground (roll=0, pitch=0)
    On slopes/stairs: Rewards aligning body parallel to terrain slope
    
    Args:
        env: The learning environment
        asset_cfg: Config for robot asset (provides IMU data via quaternion)
        sensor_cfg: Config for height scanner sensor (terrain slope detection) 
        alignment_threshold: Tolerance for orientation alignment in radians
        
    Returns:
        Reward tensor: Higher values for better terrain alignment
    """
    # Extract robot and sensor data
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get current body orientation from Go2's IMU (quaternion)
    body_quat = asset.data.root_quat_w  # [N, 4] - w, x, y, z quaternion
    
    # Extract roll, pitch, yaw from quaternion using Isaac Lab function
    body_roll, body_pitch, body_yaw = euler_xyz_from_quat(body_quat)  # Returns tuple of 3 tensors
    
    # Check if height scanner is available for terrain slope estimation
    if sensor_cfg.name in env.scene.sensors:
        height_scanner: RayCaster = env.scene[sensor_cfg.name]
        # Estimate terrain slope from height scanner data
        ray_hits = height_scanner.data.ray_hits_w  # [N, num_rays, 3] - world coordinates
    else:
        # Fallback: assume flat terrain if no height scanner
        ray_hits = None
    
    # Calculate terrain normal vector from height differences
    # Use cross-product of terrain vectors to get surface normal
    if ray_hits is not None and ray_hits.shape[1] >= 4:  # Need at least 4 rays for robust normal estimation
        # Get corner rays for cross-product calculation
        front_left = ray_hits[:, 0, :]   # Front-left ray hit
        front_right = ray_hits[:, 1, :]  # Front-right ray hit  
        back_left = ray_hits[:, 2, :]    # Back-left ray hit
        
        # Calculate terrain vectors
        terrain_vec1 = front_right - front_left   # Left-right vector
        terrain_vec2 = back_left - front_left     # Front-back vector
        
        # Cross product gives terrain normal vector
        terrain_normal = torch.cross(terrain_vec1, terrain_vec2, dim=1)
        terrain_normal = torch.nn.functional.normalize(terrain_normal, dim=1)
        
        # Extract desired roll and pitch from terrain normal
        # Normal vector [nx, ny, nz] → roll = atan2(ny, nz), pitch = atan2(-nx, sqrt(ny² + nz²))
        desired_roll = torch.atan2(terrain_normal[:, 1], terrain_normal[:, 2])
        desired_pitch = torch.atan2(-terrain_normal[:, 0], 
                                   torch.sqrt(terrain_normal[:, 1]**2 + terrain_normal[:, 2]**2))
    else:
        # Fallback to flat terrain assumption if insufficient rays
        desired_roll = torch.zeros_like(body_roll)
        desired_pitch = torch.zeros_like(body_pitch) 
    
    # Calculate orientation alignment errors
    roll_error = torch.abs(body_roll - desired_roll)
    pitch_error = torch.abs(body_pitch - desired_pitch)
    
    # Total orientation error
    total_error = roll_error + pitch_error
    
    # Exponential reward function: higher reward for smaller alignment errors
    # alignment_threshold acts as standard deviation for exponential kernel
    alignment_reward = torch.exp(-total_error / (alignment_threshold**2))
    
    return alignment_reward


def base_height_terrain_adaptive(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_tolerance: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Reward maintaining proper height above terrain using Go2 SDK sensors.
    
    This function uses:
    - Height scanner to measure distance to terrain surface
    - Adaptive target height based on local terrain elevation  
    - Joint position feedback for leg configuration awareness
    
    Args:
        env: The learning environment
        target_height: Desired height above terrain surface (meters)
        height_tolerance: Acceptable height variation (meters)
        asset_cfg: Config for robot asset
        sensor_cfg: Config for height scanner sensor
        
    Returns:
        Reward tensor: Higher values for maintaining proper terrain clearance
    """
    # Extract robot and sensor data
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get robot's current height (Z position in world frame)
    current_height = asset.data.root_pos_w[:, 2]
    
    # Get terrain height beneath robot from height scanner if available
    if sensor_cfg.name in env.scene.sensors:
        height_scanner: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = height_scanner.data.ray_hits_w  # [N, num_rays, 3]
        # Calculate average terrain height beneath robot
        terrain_height = torch.mean(ray_hits[:, :, 2], dim=1)  # Average Z coordinate of ray hits
    else:
        # Fallback: assume ground level at 0 if no height scanner
        terrain_height = torch.zeros_like(current_height)
    
    # Calculate actual height above terrain
    height_above_terrain = current_height - terrain_height
    
    # Calculate height error from target
    height_error = torch.abs(height_above_terrain - target_height)
    
    # Reward function: exponential decay with height_tolerance as std dev
    height_reward = torch.exp(-height_error / (height_tolerance**2))
    
    # Additional penalty for being too close to or below terrain
    ground_clearance_penalty = torch.clamp(0.1 - height_above_terrain, min=0.0) * 10.0
    
    return height_reward - ground_clearance_penalty


def air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize variance in feet air time to encourage coordinated gait patterns.
    
    This function promotes synchronized stepping patterns essential for stair climbing
    by penalizing large differences in air time between feet.
    
    Args:
        env: The learning environment
        sensor_cfg: Config for contact sensor (foot force sensors)
        
    Returns:
        Penalty tensor: Higher values for uncoordinated stepping
    """
    # Extract contact sensor data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get current air time for each foot
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [N, 4]
    
    # Calculate variance in air time across feet
    air_time_var = torch.var(air_time, dim=1)  # [N]
    
    return air_time_var


def feet_slide_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet sliding during ground contact using Go2 foot force sensors.
    
    This function uses Go2's foot force sensors to detect ground contact and
    penalizes horizontal foot movement during contact phases - critical for
    stable stair climbing.
    
    Args:
        env: The learning environment
        asset_cfg: Config for robot asset (foot bodies)
        sensor_cfg: Config for contact sensor (foot force sensors)
        
    Returns:
        Penalty tensor: Higher values for sliding feet
    """
    # Extract robot and contact sensor data
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get foot contact forces and determine contact status
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = torch.norm(net_contact_forces, dim=-1).max(dim=1)[0] > 1.0  # [N, 4]
    
    # Get foot linear velocities in world frame
    foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # [N, 4, 2] (x, y only)
    
    # Calculate sliding magnitude (horizontal velocity during contact)
    sliding_magnitude = torch.norm(foot_velocities, dim=-1)  # [N, 4]
    
    # Penalize sliding only when foot is in contact with ground
    sliding_penalty = sliding_magnitude * is_contact.float()
    
    # Sum penalty across all feet
    return torch.sum(sliding_penalty, dim=1)
