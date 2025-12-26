# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL and evaluate performance."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

# Set KMP_DUPLICATE_LIB_OK to TRUE to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from isaaclab.app import AppLauncher

# local imports
# Add the path to the rsl_rl script directory to sys.path to import cli_args
sys.path.append(os.path.join(os.path.dirname(__file__), "../reinforcement_learning/rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
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
parser.add_argument("--evaluation_time", type=float, default=10.0, help="Time duration for evaluation in seconds.")

# MinIO Arguments
parser.add_argument("--minio_endpoint", type=str, default=None, help="MinIO endpoint URL")
parser.add_argument("--minio_access_key", type=str, default=None, help="MinIO access key")
parser.add_argument("--minio_secret_key", type=str, default=None, help="MinIO secret key")
parser.add_argument("--minio_bucket", type=str, default=None, help="MinIO bucket name")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Set headless to True by default
parser.set_defaults(headless=True)
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

import torch
import numpy as np
import gymnasium as gym
import csv
from datetime import datetime

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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
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
    
    # Set evaluation output directory to scripts/evaluation/result
    eval_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Disable environment logging to prevent events.out.tfevents generation
    env_cfg.log_dir = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(eval_output_dir, "videos", agent_cfg.experiment_name),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
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

    # reset environment
    obs, _ = env.reset()
    
    # Access the robot object
    # Note: This assumes the robot is named "robot" in the scene. 
    # If not, we might need to inspect env.unwrapped.scene.keys()
    robot = env.unwrapped.scene["robot"]
    
    # Evaluation Metrics Storage
    metrics = {
        "velocity_tracking_error": [],
        "power_consumption": [],
        "torque_smoothness": [],
        "undesired_contacts": 0,
        "command_velocity": [],
        "measured_velocity": []
    }
    
    # Extended metrics for Sim-to-Real (Time-series data)
    sim_log = {
        "timestamp": [],
        "command_vel": [],
        "base_lin_vel": [],
        "dof_pos": [],
        "dof_vel": [],
        "dof_torque": []
    }
    
    # Simulation parameters
    dt = env.unwrapped.step_dt
    num_steps = int(args_cli.evaluation_time / dt)
    
    print(f"[INFO] Starting evaluation for {args_cli.evaluation_time} seconds ({num_steps} steps)...")

    # Previous torque for smoothness calculation
    prev_torque = None

    for i in range(num_steps):
        # run the policy
        with torch.inference_mode():
            actions = policy(obs)
        
        # step the environment
        # Note: RslRlVecEnvWrapper.step returns (obs, rew, terminated, truncated, info)
        # However, some older versions or wrappers might return (obs, rew, done, info)
        # Let's handle both cases
        step_result = env.step(actions)
        if len(step_result) == 5:
            obs, _, _, _, _ = step_result
        elif len(step_result) == 4:
            obs, _, _, _ = step_result
        else:
            raise ValueError(f"Unexpected step return length: {len(step_result)}")
        
        # --- Data Collection for Metrics ---
        
        # Get command velocity (Target)
        cmd_vel = None
        if hasattr(env.unwrapped, "command_manager"):
            # Try to find a velocity command
            for term_name in env.unwrapped.command_manager.active_terms:
                term = env.unwrapped.command_manager.get_term(term_name)
                if "vel" in term_name.lower():
                    cmd_vel = term.command
                    break
        
        # Skip logging if no command velocity is found
        if cmd_vel is None:
            continue

        # Collect detailed time-series data for Sim-to-Real comparison (Env 0 only)
        sim_log["timestamp"].append(i * dt)
        sim_log["base_lin_vel"].append(robot.data.root_lin_vel_b[0].cpu().numpy())
        sim_log["dof_pos"].append(robot.data.joint_pos[0].cpu().numpy())
        sim_log["dof_vel"].append(robot.data.joint_vel[0].cpu().numpy())
        sim_log["dof_torque"].append(robot.data.applied_torque[0].cpu().numpy())
        sim_log["command_vel"].append(cmd_vel[0].cpu().numpy())
        
        # 1. Velocity Tracking Error
        # Get measured velocity (Base frame)
        # robot.data.root_lin_vel_b is base linear velocity in base frame
        measured_vel = robot.data.root_lin_vel_b
        
        # cmd_vel is usually (num_envs, 3) [vx, vy, w] or similar. 
        # measured_vel is (num_envs, 3) [vx, vy, vz]
        # We compare linear parts.
        # Note: cmd_vel might include angular velocity.
        # Let's assume cmd_vel[:, :2] corresponds to measured_vel[:, :2] (xy plane)
        
        # Check shapes
        if cmd_vel.shape[1] >= 2:
            target_v = cmd_vel[:, :2]
            current_v = measured_vel[:, :2]
            error = torch.norm(target_v - current_v, dim=1)
            metrics["velocity_tracking_error"].append(error.cpu().numpy())
            metrics["command_velocity"].append(target_v.cpu().numpy())
            metrics["measured_velocity"].append(current_v.cpu().numpy())

        # 2. Power Consumption (CoT related)
        # Power = sum(|torque * joint_vel|)
        torques = robot.data.applied_torque
        joint_vels = robot.data.joint_vel
        power = torch.sum(torch.abs(torques * joint_vels), dim=1)
        metrics["power_consumption"].append(power.cpu().numpy())
        
        # 3. Torque Smoothness / Jitter
        # Metric: |d(torque)/dt|
        if prev_torque is not None:
            torque_diff = (torques - prev_torque) / dt
            smoothness = torch.mean(torch.abs(torque_diff), dim=1)
            metrics["torque_smoothness"].append(smoothness.cpu().numpy())
        prev_torque = torques.clone()
        
        # 4. Undesired Contacts
        # Check for collisions.
        # We use the contact sensor if available.
        if "contact_forces" in env.unwrapped.scene.keys():
            contact_sensor = env.unwrapped.scene["contact_forces"]
            # net_forces_w is (num_envs, num_bodies, 3)
            # We need to find the index of the base.
            # Assuming base is the first body or named "base"
            # contact_sensor.body_names is a list of body names
            
            # Find base index
            base_idx = None
            for idx, name in enumerate(contact_sensor.body_names):
                if "base" in name:
                    base_idx = idx
                    break
            
            if base_idx is not None:
                base_contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, base_idx, :], dim=1)
                # If force > threshold (e.g. 1.0 N), count as contact
                metrics["undesired_contacts"] += torch.sum(base_contact_forces > 1.0).item()
        else:
            # Fallback if no contact sensor (or named differently)
            pass

    # --- Report Generation ---
    report_lines = []
    report_lines.append("\n" + "="*50)
    report_lines.append(f"EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*50)
    
    # Convert lists to numpy arrays for calculation
    vel_errors = np.array(metrics["velocity_tracking_error"]) # (steps, num_envs)
    powers = np.array(metrics["power_consumption"]) # (steps, num_envs)
    smoothness = np.array(metrics["torque_smoothness"]) # (steps-1, num_envs)
    
    # 1. Velocity RMSE
    rmse = None
    if vel_errors.size > 0:
        rmse = np.sqrt(np.mean(vel_errors**2))
        report_lines.append(f"Velocity RMSE: {rmse:.4f} m/s")
        report_lines.append(f"  Target: < 0.05 m/s")
    
    # 2. Cost of Transport (CoT)
    # CoT = P_total / (m * g * v)
    # We calculate P_total / v as requested for relative comparison
    # We use average power and average velocity
    cot_proxy = None
    if powers.size > 0 and len(metrics["measured_velocity"]) > 0:
        avg_power = np.mean(powers)
        measured_vels = np.array(metrics["measured_velocity"])
        avg_vel = np.mean(np.linalg.norm(measured_vels, axis=2)) # Average speed
        
        if avg_vel > 1e-3:
            cot_proxy = avg_power / avg_vel
            report_lines.append(f"CoT Proxy (Power/Velocity): {cot_proxy:.4f} J/m")
        else:
            report_lines.append(f"Average Power: {avg_power:.4f} W (Velocity too low for CoT)")
            
    # 3. Torque Smoothness
    avg_smoothness = None
    if smoothness.size > 0:
        avg_smoothness = np.mean(smoothness)
        report_lines.append(f"Torque Smoothness (dTorque/dt): {avg_smoothness:.4f} Nm/s")
        
    # 4. Undesired Contacts
    report_lines.append(f"Undesired Base Contacts (Total Frames): {metrics['undesired_contacts']}")
    
    report_lines.append("="*50)
    
    # Print report
    report_str = "\n".join(report_lines)
    print(report_str)
    
    # Save report and metrics
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save to eval_output_dir (scripts/evaluation)
    # Create a subfolder for the experiment if desired, or just save in evaluation folder
    # User requested: "evaluation 폴더 안에 있었으면 좋겠어"
    
    # Create pkl directory
    pkl_output_dir = os.path.join(eval_output_dir, "pkl")
    os.makedirs(pkl_output_dir, exist_ok=True)
        
    # Save Sim Log for Sim2Real (Contains raw time-series data)
    sim_log_file_path = os.path.join(pkl_output_dir, f"sim_log_{agent_cfg.experiment_name}_{date_str}.pkl")
    import pickle
    with open(sim_log_file_path, "wb") as f:
        pickle.dump(sim_log, f)

    # Save Sim Log to CSV (New)
    csv_output_dir = os.path.join(eval_output_dir, "csv")
    os.makedirs(csv_output_dir, exist_ok=True)
    sim_log_csv_path = os.path.join(csv_output_dir, f"sim_log_{agent_cfg.experiment_name}_{date_str}.csv")
    
    
    with open(sim_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['timestamp']
        header += [f'cmd_vel_{i}' for i in range(3)]
        header += [f'base_lin_vel_{i}' for i in range(3)]
        header += [f'dof_pos_{i}' for i in range(12)]
        header += [f'dof_vel_{i}' for i in range(12)]
        header += [f'dof_torque_{i}' for i in range(12)]
        writer.writerow(header)
        
        # Helper function for formatting
        def fmt(val):
            if isinstance(val, (int, float, np.float32, np.float64)):
                return round(float(val), 5)
            return val

        # Rows
        num_rows = len(sim_log['timestamp'])
        for i in range(num_rows):
            row = [fmt(sim_log['timestamp'][i])]
            # Handle potential numpy arrays or lists
            row += [fmt(x) for x in sim_log['command_vel'][i]]
            row += [fmt(x) for x in sim_log['base_lin_vel'][i]]
            row += [fmt(x) for x in sim_log['dof_pos'][i]]
            row += [fmt(x) for x in sim_log['dof_vel'][i]]
            row += [fmt(x) for x in sim_log['dof_torque'][i]]
            writer.writerow(row)

    metrics_file_path = os.path.join(pkl_output_dir, f"evaluation_metrics_{agent_cfg.experiment_name}_{date_str}.pkl")
    with open(metrics_file_path, "wb") as f:
        pickle.dump(metrics, f)
        
    # Save to CSV (Summary)
    csv_filename = f"evaluation_results_{agent_cfg.experiment_name}.csv"
    csv_path = os.path.join(eval_output_dir, csv_filename)
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['Date', 'Checkpoint', 'Velocity_RMSE', 'CoT_Proxy', 'Torque_Smoothness', 'Undesired_Contacts']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Checkpoint': os.path.relpath(resume_path, os.getcwd()),
            'Velocity_RMSE': f"{rmse:.4f}" if rmse is not None else "N/A",
            'CoT_Proxy': f"{cot_proxy:.4f}" if cot_proxy is not None else "N/A",
            'Torque_Smoothness': f"{avg_smoothness:.4f}" if avg_smoothness is not None else "N/A",
            'Undesired_Contacts': metrics['undesired_contacts']
        })
    
    # --- MinIO Upload ---
    if args_cli.minio_endpoint:
        # Import local utility
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from minio_utils import MinioClientWrapper
        
        # Convert lists to numpy arrays
        for key in sim_log:
            sim_log[key] = np.array(sim_log[key])
            
        # Add metadata
        sim_log["metadata"] = {
            "experiment_name": agent_cfg.experiment_name,
            "date": date_str,
            "dt": dt,
            "num_envs": env.num_envs,
            "checkpoint": resume_path
        }

        minio_client = MinioClientWrapper(
            args_cli.minio_endpoint,
            args_cli.minio_access_key,
            args_cli.minio_secret_key,
            args_cli.minio_bucket
        )
        minio_client.upload_log(agent_cfg.experiment_name, date_str, sim_log)
    else:
        print("[INFO] MinIO endpoint not specified. Skipping upload. (Local logs saved)")
        
    print(f"[INFO] Sim Log saved to: {sim_log_file_path}")
    print(f"[INFO] Evaluation metrics saved to: {metrics_file_path}")
    print(f"[INFO] Evaluation results appended to: {csv_path}")
    print(f"[INFO] Evaluation complete.")
    
    # Close the simulator
    os._exit(0)

if __name__ == "__main__":
    # run the main function
    main()
