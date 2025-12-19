# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RMA Adaptation Module (Phase 2)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train RMA Adaptation Module.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_interval", type=int, default=100, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--max_iterations", type=int, default=100, help="Training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Train Adaptation Module."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    # Save to logs/rma with _phase2 suffix to distinguish from Phase 1
    # Use experiment_name from agent_cfg if available, otherwise default to unitree_go2_rough_rma_phase2
    experiment_name = getattr(agent_cfg, "experiment_name", "unitree_go2_rough_rma_phase2")
    # If the experiment name is phase 1, change it to phase 2 for saving
    if "phase1" in experiment_name:
        experiment_name = experiment_name.replace("phase1", "phase2")
    elif "phase2" not in experiment_name:
        experiment_name = f"{experiment_name}_phase2"
        
    log_root_path = os.path.join("logs", "rma", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load Teacher Policy (Phase 1)
    # We assume the user provided --load_run or --checkpoint pointing to Phase 1
    if agent_cfg.load_run or agent_cfg.load_checkpoint:
        # Check if logs are in logs/ram (custom), logs/rma, or logs/rsl_rl (default)
        # We use absolute paths to avoid issues with os.path.join in get_checkpoint_path
        # because DirEntry objects from os.scandir return paths relative to the scandir argument.
        # If we pass a relative path, get_checkpoint_path joins it with the DirEntry path (which is also relative),
        # causing duplication (e.g. logs/exp/logs/exp/run).
        # If we pass an absolute path, os.path.join handles the second absolute path correctly.
        
        base_path = os.path.abspath("logs")

        if os.path.exists(os.path.join(base_path, "rma", agent_cfg.experiment_name)):
            log_root = os.path.join(base_path, "rma")
        else:
            log_root = os.path.join(base_path, "rsl_rl")
            
        # Handle common user error where load_run is set to experiment_name
        if agent_cfg.load_run == agent_cfg.experiment_name:
            print(f"[WARNING] --load_run matches experiment_name. Assuming you meant to load the latest run. Setting load_run to '.*'")
            agent_cfg.load_run = ".*"
            
        resume_path = get_checkpoint_path(os.path.join(log_root, agent_cfg.experiment_name), agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading teacher policy from: {resume_path}")
        
        # Save loading info to file
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "teacher_policy_info.txt"), "w") as f:
            f.write(f"Teacher Policy Path: \n {resume_path}\n")
    else:
        raise ValueError("Please provide --load_run or --checkpoint to load the Teacher policy (Phase 1).")

    # Create runner to load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)
    
    # Debugging: Print attributes of runner.alg to find the correct policy attribute
    print(f"[INFO] runner.alg attributes: {dir(runner.alg)}")
    
    # Try to get policy
    if hasattr(runner.alg, "actor_critic"):
        teacher_policy = runner.alg.actor_critic
    elif hasattr(runner.alg, "policy"):
        teacher_policy = runner.alg.policy
    else:
        # Fallback to get_inference_policy if available, but we need the full model for .act() usually?
        # Actually get_inference_policy returns the actor forward pass usually.
        print("[WARNING] Could not find 'actor_critic' or 'policy' in runner.alg. Using get_inference_policy.")
        teacher_policy = runner.get_inference_policy(device=env.device)
        
    # teacher_policy.eval() # get_inference_policy might not have .eval()
    if hasattr(teacher_policy, "eval"):
        teacher_policy.eval()
    
    # Initialize Adaptation Module
    # Input: Proprioception History
    # Output: Extrinsics (Privileged Info)
    
    # Set normalizer to eval mode if it exists to stop updating statistics
    if hasattr(runner, "obs_normalizer") and runner.obs_normalizer is not None:
        if hasattr(runner.obs_normalizer, "eval"):
            runner.obs_normalizer.eval()
            print("[INFO] Set observation normalizer to eval mode.")

    # Get dimensions
    # Note: RslRlVecEnvWrapper flattens observations, but we can access the manager via unwrapped
    obs_manager = env.unwrapped.observation_manager
    proprio_dim = obs_manager.group_obs_dim["proprio"][0]
    privileged_dim = obs_manager.group_obs_dim["privileged"][0]
    
    print(f"[INFO] Proprioception Dim: {proprio_dim}")
    print(f"[INFO] Privileged Dim: {privileged_dim}")
    
    adaptation_module = nn.Sequential(
        nn.Linear(proprio_dim, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, privileged_dim)
    ).to(env.device)
    
    optimizer = optim.Adam(adaptation_module.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Training Loop
    max_iterations = args_cli.max_iterations if args_cli.max_iterations else 1000
    
    obs = env.get_observations()
    
    print("[INFO] Starting Adaptation Training...")
    for i in range(max_iterations):
        # 1. Run Teacher Policy
        with torch.no_grad():
            # Teacher uses "policy" group (Proprio + Extrinsics)
            teacher_obs = obs["policy"]
            
            # Handle normalization if runner has it
            if hasattr(runner, "obs_normalizer") and runner.obs_normalizer is not None:
                teacher_obs = runner.obs_normalizer(teacher_obs)
            
            # Generate actions
            # RSL-RL ActorCritic expects a dictionary of observations if obs_groups is used
            if hasattr(teacher_policy, "act"):
                # Check if act expects a dict (common in newer RSL-RL with obs_groups)
                # We reconstruct the dict with the (potentially normalized) policy obs
                teacher_input = {"policy": teacher_obs}
                actions = teacher_policy.act(teacher_input)
            else:
                # Assume it's a callable (like from get_inference_policy)
                actions = teacher_policy(teacher_obs)
            
        # 2. Step Environment
        obs, rew, dones, extras = env.step(actions)
        
        # 3. Adaptation Training
        # Input: Proprioception with history
        proprio_obs = obs["proprio"]
        # Target: Privileged Info (Extrinsics)
        target = obs["privileged"]
        
        pred = adaptation_module(proprio_obs)
        loss = loss_fn(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Iter {i}, Loss: {loss.item():.6f}")
            
    # Save Adaptation Module
    save_path = os.path.join(log_dir, "adaptation_module.pt")
    torch.save(adaptation_module.state_dict(), save_path)
    print(f"[INFO] Saved Adaptation Module to: {save_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
