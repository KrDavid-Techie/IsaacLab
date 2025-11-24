# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.envs import mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
# Import terrain-adaptive reward functions from main MDP module
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp

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
        # JOINT-ONLY REINFORCEMENT LEARNING FOR STAIR CLIMBING & WALKING
        # ============================================================================
        # This configuration uses ONLY joint values and foot pressure sensors for RL:
        # 
        # PRIMARY SENSORS:
        # 1. Joint Motors (12 total): Position, Velocity, Torque feedback per joint
        # 2. Foot Pressure Sensors (4 total): Ground contact force and binary contact state
        # 
        # EXCLUDED SENSORS (for pure joint-based RL):
        # - IMU sensors (accelerometer, gyroscope, orientation)
        # - Height scanner / LiDAR
        # - Camera systems
        # ============================================================================

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Disable external sensors for joint-only RL approach
        self.scene.height_scanner = None  # Remove height scanner dependency
        # self.scene.lidar_scanner = None   # Remove LiDAR dependency
        # Enhanced terrain configuration for stair climbing and rough terrain navigation
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.15)  # Larger obstacles
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.08)  # More challenging
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        
        # Add pyramid stairs configuration for stair climbing training
        if hasattr(self.scene.terrain.terrain_generator.sub_terrains, 'pyramid_stairs'):
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_width = 0.3

        # Optimize action scale for aggressive stair climbing joint control
        self.actions.joint_pos.scale = 0.5  # High responsiveness for dynamic stair climbing

        # event - enhanced for rough terrain training
        self.events.push_robot = None  # Disable random pushing for stable stair climbing
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 2.0)  # Reduced mass variation
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        # Enhanced joint initialization for rough terrain adaptation
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)  # More stable initial positions
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
        # ============================================================================
        # 4족 보행 로봇 핵심 보상 시스템 (연구 기반 설계 철학)
        # ============================================================================
        # 이 보상 시스템은 4족 보행 로봇의 핵심 보상 컴포넌트 연구를 바탕으로 설계됨
        # PRIMARY SENSORS: 관절 모터(12개) + 발 압력 센서(4개)만 사용
        # ============================================================================
        
        # ============ 1. 생존 (SURVIVAL) - 필수 ============
        # Alive Bonus는 일반적으로 환경 자체에서 제공되므로 여기서는 조기 종료 방지에 집중
        
        # ============ 2. 임무 (TASK) - 필수 ============
        # Linear Velocity Tracking - 핵심 긍정적 보상
        # exp() 형태로 보상을 [0, weight]로 제한하여 안정성 확보
        self.rewards.track_lin_vel_xy_exp.weight = 10.0  # 다른 모든 페널티의 합(~18.86)의 절반(9.43)보다 커야 함
        
        # Angular Velocity Tracking - 회전 속도 추적
        self.rewards.track_ang_vel_z_exp.weight = 1.0   # 선형 속도와 동일한 철학
        
        # ============ 3. 안정성 (STABILITY) - 권장 ============
        # Roll/Pitch Stabilization - 몸체 수평 유지 (넘어짐 방지)
        self.rewards.flat_orientation_l2.weight = -1.0  # 조기 종료 전 부드러운 페널티 제공
        
        # Base Height Penalty - 원하는 고도 유지 ("기어가는" 정책 방지)
        # lin_vel_z_l2가 수직 속도 페널티 역할을 하여 높이 안정성에 기여
        self.rewards.lin_vel_z_l2.weight = -2.0
        
        # Angular velocity penalties - 몸체 안정성 확보
        self.rewards.ang_vel_xy_l2.weight = -0.5
        
        # ============ 4. 효율성 (EFFICIENCY) - 매우 주의! ============
        # Torque/Energy Penalty - "기어가는" 정책의 주요 원인
        # 매우 낮은 값으로 시작하거나 훈련 후반에 도입
        self.rewards.dof_torques_l2.weight = -0.0002  # 매우 낮은 토크 페널티
        
        # ============ 5. 부드러움 (SMOOTHNESS) - 권장 ============
        # Action Rate Penalty - 연속 스텝의 부드러운 행동 변화
        # "Jerky"한 동작 방지, Sim2Real 격차 감소에 매우 중요
        self.rewards.action_rate_l2.weight = -0.01
        
        # Joint Acceleration Penalty - 관절의 급격한 가속 방지
        # Action Rate와 유사한 물리적 효과, 스케일링에 매우 민감
        self.rewards.dof_acc_l2.weight = -2.5e-7
        
        # ============ 6. 안전 (SAFETY) - 권장 ============  
        # Self-Collision Penalty - 무릎이나 다리가 몸체에 충돌 방지
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*thigh.*|.*calf.*"
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        
        # Joint limits safety - 관절 한계 위반 방지
        self.rewards.dof_pos_limits.weight = -10.0
        
        # ============ 7. 보행 패턴 최적화 (GAIT OPTIMIZATION) ============
        # Feet air time - 적절한 스텝 패턴 유도
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.5  # 보행 패턴 개선
        
        # Feet sliding penalty - 안정적인 locomotion을 위한 미끄러짐 방지
        self.rewards.feet_slide = RewTerm(
            func=velocity_mdp.feet_slide,
            weight=-0.25,  # 미끄러짐 페널티
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")
            }
        )
        
        # ============ 8. 저속 명령 시 안정성 (LOW-SPEED STABILITY) ============
        # Stand still when commanded - 작은 명령에서의 정지 행동 유도
        self.rewards.stand_still = RewTerm(
            func=velocity_mdp.stand_still_joint_deviation_l1,
            weight=-0.1,  # 정지 명령 시 움직임 페널티
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.06,
                "asset_cfg": SceneEntityCfg("robot")
            }
        )
        
        # ============ 보상 시스템 설계 철학 요약 ============
        # 1. 생존: 조기 종료 방지를 위한 안정성 보상
        # 2. 임무: 핵심 긍정적 보상 (속도 추적)이 모든 페널티보다 우선
        # 3. 안정성: 넘어짐 및 "기어가는" 정책 방지
        # 4. 효율성: 토크 페널티는 매우 주의깊게 조정 (낮은 값)
        # 5. 부드러움: Sim2Real 전이를 위한 부드러운 동작 유도
        # 6. 안전성: 충돌 및 관절 한계 위반 방지
        # 7. 관절+발압력 센서만 사용하는 순수 proprioceptive RL
        
        # ============ OBSERVATIONS (Joint-Only RL) ============
        # Remove height scanner dependency from observations
        self.observations.policy.height_scan = None
        
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