# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log_joints", action="store_true", default=False, help="Log joint positions and velocities.")
parser.add_argument("--log_commands", action="store_true", default=False, help="Log velocity commands.")
parser.add_argument("--log_interval", type=int, default=10, help="Interval (in steps) for logging joint data.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import csv

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # setup joint logging if requested
    joint_log_files = []
    joint_writers = []
    if args_cli.log_joints:
        joint_log_dir = os.path.join(log_dir, "joint_logs")
        os.makedirs(joint_log_dir, exist_ok=True)
        
        # Get joint names from the environment
        robot = env.unwrapped.scene["robot"]
        joint_names = robot.data.joint_names
        num_envs = env.unwrapped.scene.num_envs
        
        print(f"[INFO] Logging joint data for {num_envs} environments")
        print(f"[INFO] Joint names: {joint_names}")
        print(f"[INFO] Logging interval: {args_cli.log_interval} steps")
        
        # Create CSV file for each environment
        for env_idx in range(num_envs):
            log_file_path = os.path.join(joint_log_dir, f"env_{env_idx:03d}_joints.csv")
            log_file = open(log_file_path, 'w', newline='')
            joint_log_files.append(log_file)
            
            # Create CSV writer with header
            fieldnames = ['timestep', 'time'] + [f"{name}_pos" for name in joint_names] + [f"{name}_vel" for name in joint_names]
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()
            joint_writers.append(writer)
    
    # setup command logging if requested
    command_log_files = []
    command_writers = []
    if args_cli.log_commands:
        command_log_dir = os.path.join(log_dir, "command_logs")
        os.makedirs(command_log_dir, exist_ok=True)
        
        num_envs = env.unwrapped.scene.num_envs
        
        print(f"[INFO] Logging command data for {num_envs} environments")
        print(f"[INFO] Command logging interval: {args_cli.log_interval} steps")
        
        # Create CSV file for each environment
        for env_idx in range(num_envs):
            log_file_path = os.path.join(command_log_dir, f"env_{env_idx:03d}_commands.csv")
            log_file = open(log_file_path, 'w', newline='')
            command_log_files.append(log_file)
            
            # Create CSV writer with header
            fieldnames = ['timestep', 'time', 'cmd_lin_vel_x', 'cmd_lin_vel_y', 'cmd_ang_vel_z', 
                         'actual_lin_vel_x', 'actual_lin_vel_y', 'actual_ang_vel_z']
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()
            command_writers.append(writer)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        # log joint data if requested
        if args_cli.log_joints and timestep % args_cli.log_interval == 0:
            robot = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos.cpu().numpy()  # Shape: [num_envs, num_joints]
            joint_vel = robot.data.joint_vel.cpu().numpy()  # Shape: [num_envs, num_joints]
            current_time = timestep * dt
            
            # Log data for each environment
            for env_idx in range(env.unwrapped.scene.num_envs):
                row_data = {
                    'timestep': timestep,
                    'time': current_time
                }
                
                # Add joint positions
                for joint_idx, joint_name in enumerate(robot.data.joint_names):
                    row_data[f"{joint_name}_pos"] = joint_pos[env_idx, joint_idx]
                    row_data[f"{joint_name}_vel"] = joint_vel[env_idx, joint_idx]
                
                joint_writers[env_idx].writerow(row_data)
                # Flush to ensure data is written
                joint_log_files[env_idx].flush()
        
        # log command data if requested
        if args_cli.log_commands and timestep % args_cli.log_interval == 0:
            robot = env.unwrapped.scene["robot"]
            current_time = timestep * dt
            
            # Get velocity commands from the command manager
            if hasattr(env.unwrapped, 'command_manager') and hasattr(env.unwrapped.command_manager, 'get_command'):
                try:
                    # Try to get velocity commands
                    velocity_commands = env.unwrapped.command_manager.get_command("base_velocity")  # Shape: [num_envs, 3]
                    velocity_commands_np = velocity_commands.cpu().numpy()
                    
                    # Get actual robot velocities
                    actual_lin_vel = robot.data.root_lin_vel_b.cpu().numpy()  # Shape: [num_envs, 3]
                    actual_ang_vel = robot.data.root_ang_vel_b.cpu().numpy()  # Shape: [num_envs, 3]
                    
                    # Log data for each environment
                    for env_idx in range(env.unwrapped.scene.num_envs):
                        row_data = {
                            'timestep': timestep,
                            'time': current_time,
                            'cmd_lin_vel_x': velocity_commands_np[env_idx, 0],
                            'cmd_lin_vel_y': velocity_commands_np[env_idx, 1], 
                            'cmd_ang_vel_z': velocity_commands_np[env_idx, 2],
                            'actual_lin_vel_x': actual_lin_vel[env_idx, 0],
                            'actual_lin_vel_y': actual_lin_vel[env_idx, 1],
                            'actual_ang_vel_z': actual_ang_vel[env_idx, 2]
                        }
                        
                        command_writers[env_idx].writerow(row_data)
                        # Flush to ensure data is written
                        command_log_files[env_idx].flush()
                        
                except Exception as e:
                    if timestep == 0:  # Only print once
                        print(f"[WARNING] Could not access velocity commands: {e}")
            else:
                if timestep == 0:  # Only print once
                    print("[WARNING] Command manager not accessible for logging commands")
        
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close joint log files if opened
    # if args_cli.log_joints:
    #     for log_file in joint_log_files:
    #         log_file.close()
    #     print(f"[INFO] Joint logging completed. Files saved in: {joint_log_dir}")
    
    # # close command log files if opened
    # if args_cli.log_commands:
    #     for log_file in command_log_files:
    #         log_file.close()
    #     print(f"[INFO] Command logging completed. Files saved in: {command_log_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
