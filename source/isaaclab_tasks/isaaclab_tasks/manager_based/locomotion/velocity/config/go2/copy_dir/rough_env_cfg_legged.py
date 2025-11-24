# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


##
# Custom Rewards Configuration (legged-loco style with local MDP)
##
@configclass
class Go2RoughRewardsCfg(RewardsCfg):
    """Go2 rewards configuration based on legged-loco patterns using local IsaacLab MDP."""
    
    # Hip deviation penalty - critical for Go2 stability
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    
    # Joint deviation for thigh and calf joints
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    
    # Base height maintenance
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.32, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    
    # Action smoothness penalty
    action_smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.02,
    )
    
    # Joint power penalty
    joint_power = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Custom Observations Configuration (legged-loco style)
##
@configclass
class Go2ObservationsCfg:
    """Observation specifications for Go2 with legged-loco style policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Core observations for locomotion policy
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        
        # Extended observations for critic
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Go2 rough terrain locomotion with legged-loco training patterns."""
    
    # Override with custom configurations
    rewards: Go2RoughRewardsCfg = Go2RoughRewardsCfg()
    observations: Go2ObservationsCfg = Go2ObservationsCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ============================================================================
        # GO2 LEGGED-LOCO INTEGRATION WITH LOCAL ISAACLAB MDP
        # ============================================================================
        # This configuration applies legged-loco training patterns using local
        # IsaacLab MDP functions for better compatibility and performance
        # ============================================================================

        # Robot configuration
        from copy import deepcopy
        self.scene.robot = deepcopy(UNITREE_GO2_CFG)
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.height_scanner.debug_vis = False
        
        # Simulation settings
        self.decimation = 4
        self.sim.render_interval = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        
        # Action configuration - optimized for Go2
        self.actions.joint_pos.scale = 0.25  # Reduced for better stability

        # ============ REWARD CONFIGURATION (legged-loco style) ============
        # Core locomotion rewards
        # Linear Velocity Tracking - 페널티 합(~7.5)의 절반(3.75)보다 크게
        self.rewards.track_lin_vel_xy_exp.weight = 8.0  # 다른 모든 페널티의 합의 절반보다 커야 함
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        
        # Body stability
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        
        # Joint and action penalties
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.02
        
        # Foot contact rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.2
        
        # Fix unwanted contacts body name pattern and disable for better terrain adaptation
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.rewards.undesired_contacts.weight = 0.0
        

        # ============ EVENT CONFIGURATION ============
        # Mass randomization
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        
        # Joint initialization
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        
        # ============ COMMAND CONFIGURATION ============
        # Velocity command ranges optimized for Go2
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        
        # ============ TERMINATION CONFIGURATION ============
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        
        # Update sensor periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    """Configuration for Go2 rough terrain locomotion in play/demo mode."""
    
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
        
        # Disable external forces for stable demonstration
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        
        # Set forward motion commands for demonstration
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)