# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play RMA Student Policy (Teacher + Adaptation Module)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play RMA Student Policy.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default=None, help="Name of the task."
)
parser.add_argument(
    "--run_phase2", type=str, required=True, help="Name of the Adaptation Module run (Phase 2)."
)
parser.add_argument(
    "--ckpt_phase2", type=str, default="adaptation_module.pt", help="Checkpoint file for Adaptation Module."
)
parser.add_argument(
    "--run_phase1", type=str, default=None, help="Name of the Teacher Policy run (Phase 1). Optional if teacher_policy_info.txt exists."
)
parser.add_argument(
    "--ckpt_phase1", type=str, default=None, help="Checkpoint file for Teacher Policy. Optional if teacher_policy_info.txt exists."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def get_log_root(run_name, folder_name="rsl_rl"):
    """Helper to find log root."""
    # Check custom paths first
    if os.path.exists(os.path.join("logs", "ram", run_name)):
        return os.path.join("logs", "ram")
    elif os.path.exists(os.path.join("logs", "rma", run_name)):
        return os.path.join("logs", "rma")
    else:
        return os.path.join("logs", folder_name)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Play with RMA Student Policy."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------------------------------------------------
    # 0. Locate Phase 2 Run and Metadata
    # ---------------------------------------------------------
    log_root_p2 = get_log_root(args_cli.run_phase2, folder_name="rma")
    
    # Try to find the run folder
    run_path_p2 = os.path.join(log_root_p2, args_cli.run_phase2)
    # If it's a date folder inside
    if os.path.exists(run_path_p2):
        # Check if there are date folders or if it's direct
        subdirs = [d for d in os.listdir(run_path_p2) if os.path.isdir(os.path.join(run_path_p2, d))]
        subdirs.sort()
        if subdirs:
            # Use latest date folder
            run_path_p2 = os.path.join(run_path_p2, subdirs[-1])
            
    print(f"[INFO] Phase 2 Run Directory: {run_path_p2}")

    # ---------------------------------------------------------
    # 1. Load Teacher Policy (Phase 1)
    # ---------------------------------------------------------
    resume_path_p1 = None
    
    # Try to read from info file
    info_file = os.path.join(run_path_p2, "teacher_policy_info.txt")
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("Teacher Policy Path:"):
                    # Try same line first
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        resume_path_p1 = parts[1].strip()
                    # Try next line if same line is empty
                    elif i + 1 < len(lines):
                        resume_path_p1 = lines[i+1].strip()
                    
                    if resume_path_p1:
                        print(f"[INFO] Found Teacher Policy path in info file: {resume_path_p1}")
                        break
    
    # Override if arguments provided
    if args_cli.run_phase1:
        log_root_p1 = get_log_root(args_cli.run_phase1)
        ckpt = args_cli.ckpt_phase1 if args_cli.ckpt_phase1 else "model_.*.pt"
        resume_path_p1 = get_checkpoint_path(
            os.path.join(log_root_p1, args_cli.run_phase1), 
            ".*", 
            ckpt
        )
        
    if not resume_path_p1:
        raise ValueError("Could not determine Teacher Policy path. Please provide --run_phase1 or ensure teacher_policy_info.txt exists in Phase 2 run.")

    print(f"[INFO]: Loading Teacher Policy from: {resume_path_p1}")

    # Create runner to load policy
    # We use a dummy log dir since we are just playing
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="logs/play_dummy", device=agent_cfg.device)
    runner.load(resume_path_p1)
    
    # Get policy
    if hasattr(runner.alg, "actor_critic"):
        teacher_policy = runner.alg.actor_critic
    elif hasattr(runner.alg, "policy"):
        teacher_policy = runner.alg.policy
    else:
        teacher_policy = runner.get_inference_policy(device=env.device)
    
    if hasattr(teacher_policy, "eval"):
        teacher_policy.eval()

    # Set normalizer to eval mode
    if hasattr(runner, "obs_normalizer") and runner.obs_normalizer is not None:
        if hasattr(runner.obs_normalizer, "eval"):
            runner.obs_normalizer.eval()

    # ---------------------------------------------------------
    # 2. Load Adaptation Module (Phase 2)
    # ---------------------------------------------------------
    # We already located run_path_p2 in step 0
    
    adapt_ckpt_path = os.path.join(run_path_p2, args_cli.ckpt_phase2)
    print(f"[INFO]: Loading Adaptation Module from: {adapt_ckpt_path}")
    
    # Recreate model architecture
    obs_manager = env.unwrapped.observation_manager
    proprio_dim = obs_manager.group_obs_dim["proprio"][0]
    privileged_dim = obs_manager.group_obs_dim["privileged"][0]
    
    adaptation_module = nn.Sequential(
        nn.Linear(proprio_dim, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, privileged_dim)
    ).to(env.device)
    
    adaptation_module.load_state_dict(torch.load(adapt_ckpt_path))
    adaptation_module.eval()

    # ---------------------------------------------------------
    # 3. Run Inference Loop
    # ---------------------------------------------------------
    obs = env.get_observations()
    
    print("[INFO] Starting RMA Inference...")
    while simulation_app.is_running():
        with torch.no_grad():
            # A. Get Inputs
            proprio_hist = obs["proprio"]
            policy_obs_gt = obs["policy"] # Contains Ground Truth Extrinsics
            
            # B. Predict Extrinsics
            pred_extrinsics = adaptation_module(proprio_hist)
            
            # C. Construct Student Observation
            # We assume extrinsics are at the END of the policy observation vector
            # Split GT obs
            split_idx = policy_obs_gt.shape[1] - privileged_dim
            base_obs = policy_obs_gt[:, :split_idx]
            
            # Concatenate Base + Predicted
            student_obs_raw = torch.cat([base_obs, pred_extrinsics], dim=1)
            
            # D. Normalize (using Teacher's statistics)
            if hasattr(runner, "obs_normalizer") and runner.obs_normalizer is not None:
                student_obs = runner.obs_normalizer(student_obs_raw)
            else:
                student_obs = student_obs_raw
            
            # E. Policy Inference
            # RSL-RL ActorCritic expects a dictionary if obs_groups is used
            if hasattr(teacher_policy, "act"):
                teacher_input = {"policy": student_obs}
                actions = teacher_policy.act(teacher_input)
            else:
                actions = teacher_policy(student_obs)
        
        # Step
        obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
