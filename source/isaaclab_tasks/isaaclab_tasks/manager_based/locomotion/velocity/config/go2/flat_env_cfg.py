# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ============ ANTI-CRAWLING REWARD MODIFICATIONS ============
        # Significantly increase penalty for body tilting (was -2.5)
        self.rewards.flat_orientation_l2.weight = -10.0
        
        # Add penalty for maintaining wrong height (prevents crawling low)
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={"target_height": 0.34}  # Go2's normal standing height
        )
        
        # Increase penalties for unwanted movements to prevent crawling
        self.rewards.lin_vel_z_l2.weight = -4.0    # Prevent bouncing (was -2.0)
        self.rewards.ang_vel_xy_l2.weight = -0.2    # Prevent roll/pitch (was -0.05)
        
        # Enhanced foot rewards for proper walking (not shuffling/crawling)
        self.rewards.feet_air_time.weight = 0.5  # Increased from 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
