# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Go2 rough terrain locomotion with legged-loco training patterns."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Go2 robot configuration with legged-loco training patterns

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
        # Linear Velocity Tracking - 전체 가중치 5 미만으로 조정
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 주요 보상
        self.rewards.track_ang_vel_z_exp.weight = 0.6   # 각속도 추적 보상
        
        # Body stability  
        self.rewards.flat_orientation_l2.weight = -1.0  # 자세 페널티
        self.rewards.lin_vel_z_l2.weight = -0.4        # 수직 속도 페널티
        self.rewards.ang_vel_xy_l2.weight = -0.02      # 각속도 페널티
        
        # Joint and action penalties
        self.rewards.dof_torques_l2.weight = -0.0001  # 토크 페널티
        self.rewards.dof_acc_l2.weight = -1.0e-7      # 가속도 페널티
        self.rewards.action_rate_l2.weight = -0.01    # 액션 변화율 페널티
        
        # Foot contact rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.15  # 발 접촉 보상
        
        # Fix unwanted contacts body name pattern and disable for better terrain adaptation
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.rewards.undesired_contacts.weight = -0.0001  # 약한 페널티로 조정
        

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
        
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None