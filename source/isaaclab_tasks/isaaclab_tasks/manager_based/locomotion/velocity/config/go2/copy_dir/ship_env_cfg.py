# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, LocomotionVelocityRoughEnvCfg_RMA

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

@configclass
class UnitreeGo2ShipEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ============ ROBOT CONFIGURATION ============
        # Go2 robot configuration with legged-loco training patterns
        from copy import deepcopy
        self.scene.robot = deepcopy(UNITREE_GO2_CFG)
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # self.scene.robot.actuators["base_legs"] = ActuatorNetMLPCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #     network_file="/path/to/your/trained_actuator_net.pt", # 실제 .pt 파일 경로
        #     pos_scale=-1.0,           # 네트워크 학습 시 사용된 스케일링 값 (데이터 수집 방식에 따라 다름)
        #     vel_scale=1.0,
        #     torque_scale=1.0,
        #     input_order="pos_vel",    # 입력 순서 (pos_error, vel) 또는 (vel, pos_error) 등
        #     input_idx=[0, 1, 2],      # 입력 인덱스
        #     effort_limit=23.5,        # Go2 스펙
        #     velocity_limit=30.0,      # Go2 스펙
        #     saturation_effort=23.5,
        # )

        # ============ TERRAIN CONFIGURATION ============
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ============ ACTION CONFIGURATION ============
        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # [추가] 액션 클리핑: 목표 관절 위치가 물리적 한계를 넘지 않도록 강제 제한 (안전 장치)
        # 값은 라디안 단위이며, 필요에 따라 로봇의 실제 관절 한계값으로 더 좁게 설정 가능
        self.actions.joint_pos.clip = {
            ".*_hip_joint": (-0.84, 0.84),        # 좌우 벌림 제한
            ".*_thigh_joint": (-4.0, 1.5),        # 앞뒤 허벅지 (앞/뒤 다리 통합하여 넓게 잡음)
            ".*_calf_joint": (-2.72, -0.84),      # 종아리 (무릎 굽힘 제한)
        }

        # ============ COMMAND CONFIGURATION ============
        # Velocity command ranges optimized for Go2
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1

        # ============ EVENT CONFIGURATION ============
        self.events.base_com = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
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

        # ============ GRAVITY RANDOMIZATION ============
        # Randomize gravity vector direction and magnitude during training
        self.events.randomize_gravity = EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="interval",
            interval_range_s=(1.0, 5.0),
            params={
                "gravity_distribution_params": ([-2.0, -2.0, -10.0], [2.0, 2.0, -9.0]),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        # ============ REWARD CONFIGURATION (Total Rewards: 2.5, Total Penalties: -2.5) ============
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -0.8
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.stand_still_joint_deviation_l1.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7

        self.rewards.undesired_contacts.weight = -0.45
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh|.*_hip|Head_lower"

        # # Core locomotion rewards
        # # Linear Velocity Tracking
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5  # 주요 보상
        # self.rewards.track_ang_vel_z_exp.weight = 0.5   # 각속도 추적 보상
        # # Body stability  
        # self.rewards.flat_orientation_l2.weight = -1.0  # 자세 페널티
        # self.rewards.lin_vel_z_l2.weight = -0.5         # 수직 속도 페널티
        # self.rewards.ang_vel_xy_l2.weight = -0.05       # 각속도 페널티
        # # Joint and action penalties
        # self.rewards.dof_torques_l2.weight = -0.0005    # 토크 페널티
        # self.rewards.dof_acc_l2.weight = -6.25e-7       # 가속도 페널티
        # self.rewards.dof_pos_limits.weight = -6.25e-7   # 관절 위치 한계 페널티
        # self.rewards.action_rate_l2.weight = -0.1       # 액션 변화율 페널티

        # # Foot contact rewards
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.5         # 발 접촉 보상 
        # # Fix unwanted contacts body name pattern and disable for better terrain adaptation
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh|.*_hip|Head_lower"
        # self.rewards.undesired_contacts.weight = -0.4   # 접촉 페널티

        # ============ TERMINATION CONFIGURATION ============
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # ============ SENSOR CONFIGURATION ============
        # Update sensor periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        # self.scene.lidar_scanner.update_period = self.sim.dt * self.decimation

@configclass
class UnitreeGo2ShipEnvCfg_PLAY(UnitreeGo2ShipEnvCfg):
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

        # # [권장] 랜덤 명령어가 자동으로 바뀌지 않게 설정 (무한대 시간 설정)
        # self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9) 
        # self.commands.base_velocity.debug_vis = True # 화살표로 명령 방향 보기

@configclass
class UnitreeGo2ShipEnvCfg_RMA(UnitreeGo2ShipEnvCfg):
    observations: LocomotionVelocityRoughEnvCfg_RMA.ObservationsCfg = LocomotionVelocityRoughEnvCfg_RMA.ObservationsCfg()
