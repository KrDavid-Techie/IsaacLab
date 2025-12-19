# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg

@configclass
class DreamWaQPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.experiment_name = "go2_dreamwaq"
        self.run_name = ""
        
        # Use custom DreamWaQ ActorCritic
        self.policy_class_name = "ActorCritic_DWAQ"
        # Ensure algorithm is PPO (standard name, but our custom runner uses custom PPO class)
        self.algorithm_class_name = "PPO"
