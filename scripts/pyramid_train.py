# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    isaaclab.bat -p scripts/pyramid_spawn.py

"""

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



"""Rest everything follows."""
import torch
import numpy as np
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidStairsTerrainCfg
from isaaclab.sim.spawners.lights import spawn_light
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.envs.common import ViewerCfg

# Go2 Robot Configuration
GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="c:/Users/User/github/unitree_model/Go2/usd/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Start 0.5m above ground
        joint_pos={
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
    ),
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=150.0,  # Conservative effort limit
            velocity_limit=100.0,
            stiffness=25.0,      # Lower stiffness for stability
            damping=0.5,
        ),
    },
)


def create_pyramid_terrain() -> TerrainImporter:
    """Create pyramid stairs terrain in the scene."""
    
    # Configure as TerrainGenerator with alternating pyramid and inverted pyramid
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
    terrain_cfg = TerrainImporterCfg(
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
    
    # Create terrain importer
    terrain_importer = TerrainImporter(terrain_cfg)
    
    return terrain_importer


def create_flat_terrain():
    """Create completely flat terrain using ground plane."""
    
    # Create a simple flat ground plane
    cfg_ground = sim_utils.GroundPlaneCfg(
        size=(20.0, 20.0),  # Large flat area
        color=(0.6, 0.6, 0.6),  # Gray color
    )
    cfg_ground.func("/World/ground", cfg_ground)
    
    return None  # Ground plane doesn't return terrain importer


def light_setup():
    """Setup lights in the scene."""
    # Add distant light (sun-like lighting)
    spawn_light(
        prim_path="/World/DistantLight",
        cfg=sim_utils.DistantLightCfg(
            intensity=1000.0,
            color=(1.0, 1.0, 1.0),
            angle=0.53,  # Sun-like angular size
        ),
        translation=(0.0, 0.0, 10.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    # Add dome light for ambient lighting
    spawn_light(
        prim_path="/World/DomeLight",
        cfg=sim_utils.DomeLightCfg(
            intensity=300.0,
            color=(1.0, 1.0, 1.0),
        ),
    )


class PyramidSceneCfg(InteractiveSceneCfg):
    """Scene configuration with pyramid terrain and Go2 robot."""
    
    # Terrain - will be created by the terrain importer
    # Robot
    go2_robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")


def spawn_go2_robot(scene: InteractiveScene):
    """Spawn Go2 robot in the scene."""
    # The robot is automatically spawned when the scene is created
    robot = scene["go2_robot"]
    
    print(f"[INFO]: Go2 robot spawned at: {robot.data.root_pos_w}")
    print(f"[INFO]: Go2 robot has {robot.num_joints} joints")
    print(f"[INFO]: Joint names: {robot.data.joint_names}")
    
    return robot


def update_chase_camera(sim: SimulationContext, robot, camera_offset=(3.0, 3.0, 2.0)):
    """Update chase camera to follow the Go2 robot."""
    # Get robot position (use first environment)
    robot_pos = robot.data.root_pos_w[0].cpu().numpy()  # [x, y, z]
    
    # Calculate camera position relative to robot
    camera_pos = robot_pos + np.array(camera_offset)
    
    # Look at point slightly ahead of the robot
    lookat_pos = robot_pos + np.array([0.0, 0.0, 0.3])  # Look at robot center + slight height offset
    
    # Update camera view
    sim.set_camera_view(camera_pos.tolist(), lookat_pos.tolist())


def run_simulation_with_robot(sim: SimulationContext, scene: InteractiveScene):
    """Run simulation with Go2 robot on flat terrain with LiDAR."""
    robot = scene["go2_robot"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    print("[INFO]: Starting simulation with Go2 robot on flat terrain...")
    print("[INFO]: Chase camera enabled - camera will follow the robot!")
    print("[INFO]: LiDAR sensor enabled - scanning environment!")
    
    while simulation_app.is_running():
        # Reset robot periodically
        if count % 500 == 0:
            # Reset robot to initial state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins  # Add environment offset
            
            # Set robot position and orientation
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset joint states
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Clear scene buffers
            scene.reset()
            print(f"[INFO]: Reset robot at sim_time: {sim_time:.2f}s")
        
        # Simple standing behavior - maintain default joint positions
        target_joint_pos = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(target_joint_pos)
        
        # Write data and step simulation
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
    
        # Update chase camera every frame
        if count % 5 == 0:  # Update every 5 frames for performance
            update_chase_camera(sim, robot)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = SimulationContext(sim_cfg)
    
    # Add lighting to the scene
    print("[INFO]: Adding lighting to the scene...")
    light_setup()
        
    # Create flat terrain
    print("[INFO]: Creating flat terrain...")
    terrain = create_flat_terrain()
    
    # Create scene with robot
    print("[INFO]: Creating scene with Go2 robot...")
    scene_cfg = PyramidSceneCfg(num_envs=5, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset the simulator to generate terrain and spawn robot
    print("[INFO]: Generating terrain and spawning robot...")
    sim.reset()
    
    # Spawn and setup Go2 robot
    print("[INFO]: Setting up Go2 robot...")
    robot = spawn_go2_robot(scene)

    
    # Initialize the sensor with the correct number of environments
    print("[INFO]: Initializing LiDAR sensor...")
    # The sensor needs to be initialized after the simulation reset
    
    # Set initial chase camera position
    print("[INFO]: Setting up chase camera...")
    update_chase_camera(sim, robot, camera_offset=(4.0, 4.0, 3.0))
    
    # Now we are ready!
    print("[INFO]: Setup complete! You should see flat terrain and Go2 robot in the viewport.")
    print("[INFO]: Robot will maintain standing pose on the terrain.")
    print("[INFO]: Chase camera will follow the robot automatically!")

    # Run simulation with robot and LiDAR
    run_simulation_with_robot(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()