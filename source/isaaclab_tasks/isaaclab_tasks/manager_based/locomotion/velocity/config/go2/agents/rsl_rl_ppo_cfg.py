# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        #noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class UnitreeGo2StairPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_stair"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.8,
    )

@configclass
class UnitreeGo2RoughRMAPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = 'unitree_go2_rough_rma_phase1'

@configclass
class UnitreeGo2RoughRMADistillationRunnerCfg(RslRlDistillationRunnerCfg):
    experiment_name = 'unitree_go2_rough_rma_phase2'
    load_run = 'unitree_go2_rough_rma_phase1'
    
    policy = RslRlDistillationStudentTeacherCfg(
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[512, 256, 128],
        activation='elu',
        student_obs_normalization=False,
        teacher_obs_normalization=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-3,
    )

@configclass
class UnitreeGo2ShipPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "unitree_go2_ship"

@configclass
class UnitreeGo2ShipRMAPPORunnerCfg(UnitreeGo2ShipPPORunnerCfg):
    experiment_name = "unitree_go2_ship_rma_phase1"

@configclass
class UnitreeGo2ShipRMADistillationRunnerCfg(RslRlDistillationRunnerCfg):
    experiment_name = 'unitree_go2_ship_rma_phase2'
    load_run = 'unitree_go2_ship_rma_phase1'
    
    policy = RslRlDistillationStudentTeacherCfg(
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[512, 256, 128],
        activation='elu',
        student_obs_normalization=False,
        teacher_obs_normalization=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-3,
    )
