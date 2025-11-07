# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidStairsTerrainCfg

import isaaclab.sim as sim_utils

from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.envs import mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2StairEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        self.viewer=None
        #self.viewer.env_index=0

        # Configure as TerrainGenerator
        pyramid_terrain_cfg = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=2.0,
            num_rows=3,
            num_cols=3,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            sub_terrains={ # Configure alternating pyramid and inverted pyramid stairs
                "pyramid_stairs": HfPyramidStairsTerrainCfg(
                    proportion=0.5,  # 50% normal pyramids
                    step_height_range=(0.05, 0.20),
                    step_width=0.30,
                    platform_width=2.0,
                    inverted=False,
                ),
                "inverted_pyramid_stairs": HfPyramidStairsTerrainCfg(
                    proportion=0.5,  # 50% inverted pyramids  
                    step_height_range=(0.05, 0.20),
                    step_width=0.30,
                    platform_width=2.0,
                    inverted=True,
                )
            },
            curriculum=False,  # No curriculum for visualization
        )

        # Create terrain importer configuration
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=pyramid_terrain_cfg,
            max_init_terrain_level=0,
            collision_group=0,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
                metallic=0.0,
                roughness=1.0,
            ),
            debug_vis=False,
        )
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.5  # Increased from 0.25
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-10.0,
            params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base")}
        ) # Added to penalize base contacts
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # ============ ANTI-CRAWLING REWARD MODIFICATIONS ============
        # Significantly increase penalty for body tilting (was -2.5)
        self.rewards.flat_orientation_l2.weight = -10.0
        # Increase penalties for unwanted movements to prevent crawling
        self.rewards.lin_vel_z_l2.weight = -4.0    # Prevent bouncing (was -2.0)
        self.rewards.ang_vel_xy_l2.weight = -0.2    # Prevent roll/pitch (was -0.05)
        # Add penalty for maintaining wrong height (prevents crawling low and Lidar sensor contact)
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={
                "target_height": 0.34,  # Go2's normal standing height
                "sensor_cfg": SceneEntityCfg("height_scanner")  # Use terrain-adjusted height for rough terrain
            }
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2StairEnvCfg_PLAY(UnitreeGo2StairEnvCfg):
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