# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.envs import mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ============================================================================
        # GO2 SDK-FOCUSED REWARD CONFIGURATION FOR ROUGH TERRAIN LOCOMOTION
        # ============================================================================
        # This configuration emphasizes joint-based reinforcement learning using
        # sensor data available through the Go2 SDK:
        # 
        # 1. Joint Motors (12 total): Position, Velocity, Torque feedback per joint
        # 2. IMU Sensors: 3-axis accelerometer, gyroscope, orientation (quaternion/euler)
        # 3. Foot Force Sensors (4 total): Ground contact force and binary contact state
        # 4. Reduced dependency on LiDAR and camera for more robust joint-based control
        # ============================================================================

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.height_scanner.debug_vis = False
        #self.scene.lidar_scanner.debug_vis = False
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Optimize action scale for Go2's joint range and precision
        self.actions.joint_pos.scale = 0.3  # Slightly increased for more responsive joint control

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        # More diverse joint initialization for better joint-space exploration
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        # ============ JOINT-BASED REWARD SYSTEM (Go2 SDK Focused) ============
        # Core locomotion rewards - moderate weights to maintain velocity tracking
        # Linear Velocity Tracking - 페널티 합(~6.6)의 절반(3.3)보다 크게
        self.rewards.track_lin_vel_xy_exp.weight = 7.0  # 다른 모든 페널티의 합의 절반보다 커야 함
        self.rewards.track_ang_vel_z_exp.weight = 1.0   # Angular velocity tracking
        
        # ============ JOINT MOTOR REWARDS (Based on Go2 SDK 12 Joint Motors) ============
        # Joint position smoothness - reward natural joint movements
        self.rewards.joint_deviation_l1 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,  # Penalize large deviations from default joint positions
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # Joint velocity smoothness - encourage coordinated leg movement
        self.rewards.joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.01,  # Penalize excessive joint velocities
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # Joint torque efficiency - based on Go2's motor torque feedback
        self.rewards.dof_torques_l2.weight = -0.0002  # Increased penalty for motor efficiency
        
        # Joint acceleration smoothness - prevent jerky movements
        self.rewards.dof_acc_l2.weight = -2.5e-6  # Encourage smooth joint accelerations
        
        # Joint position limits - keep joints within safe operating range  
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-0.1,  # Penalize joint limit violations
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # ============ FOOT CONTACT REWARDS (Using Go2's 4 Foot Force Sensors) ============
        # Feet air time - based on foot force sensor feedback
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.4  # Encourage proper stepping patterns
        
        # Remove unwanted contact penalties to allow natural joint-based adaptation
        self.rewards.undesired_contacts = None
        
        # ============ BODY STABILITY REWARDS (IMU-based from Go2 SDK) ============
        # Body orientation - based on Go2's IMU quaternion data
        self.rewards.flat_orientation_l2.weight = -2.0  # Moderate penalty for body tilting
        
        # Vertical motion control - based on Go2's IMU acceleration data
        self.rewards.lin_vel_z_l2.weight = -1.0  # Allow some vertical movement for rough terrain
        self.rewards.ang_vel_xy_l2.weight = -0.05  # Light penalty for roll/pitch rates
        
        # Base height maintenance - use joint-based height estimation
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-3.0,  # Moderate penalty for height deviations
            params={
                "target_height": 0.34,  # Go2's normal standing height
                "sensor_cfg": SceneEntityCfg("height_scanner")
            }
        )
        
        # ============ ACTION SMOOTHNESS (Joint Command Smoothness) ============
        # Action rate smoothness - encourage smooth joint commands
        self.rewards.action_rate_l2 = RewTerm(
            func=mdp.action_rate_l2,
            weight=-0.02,  # Increased penalty for jerky actions
        )
        
        # ============ GO2-SPECIFIC JOINT COORDINATION REWARDS ============
        # Joint power efficiency - simulate Go2's power consumption monitoring
        self.rewards.joint_power_l2 = RewTerm(
            func=mdp.joint_torques_l2,  # Use torque as proxy for power consumption
            weight=-0.0001,  # Encourage energy-efficient movements
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None