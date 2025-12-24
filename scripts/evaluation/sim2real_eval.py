# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Sim-to-Real Evaluator Script
----------------------------
This script compares simulation logs (local .pkl) with real-world robot logs (ROS2 .mcap).
It calculates Sim-to-Real gaps in terms of velocity tracking, torque prediction, and energy efficiency.

Usage:
    isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file <PATH_TO_PKL> --real_bag <PATH_TO_MCAP>

Dependencies:
    pip install mcap mcap-ros2-support scipy matplotlib pandas
"""

import argparse
import os
import sys
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d

# Import local utility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from minio_utils import MinioClientWrapper

# Optional dependencies
try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    print("[WARN] 'mcap' libraries are missing. Real-world data loading will fail.")
    print("Install via: pip install mcap mcap-ros2-support")


class MinioLogLoader:
    """Loads simulation logs from MinIO storage."""
   
    def __init__(self, endpoint, access_key, secret_key, bucket):
        self.client = MinioClientWrapper(endpoint, access_key, secret_key, bucket)

    def load(self, experiment_name, run_date=None):
        return self.client.download_latest_log(experiment_name, run_date)

class LocalSimLogLoader:
    """Loads simulation logs from local file system."""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Sim log file not found: {self.file_path}")
            
        print(f"[INFO] Loading Sim Log: {self.file_path}")
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        return data


class RosBagLoader:
    """Loads real-world robot logs from ROS2 MCAP files."""
    
    def __init__(self, bag_path):
        self.bag_path = bag_path

    def load(self):
        """Parses standard ROS2 topics from the bag file."""
        if not os.path.exists(self.bag_path):
            raise FileNotFoundError(f"Bag file not found: {self.bag_path}")

        print(f"[INFO] Parsing ROS Bag: {self.bag_path}")
        
        data = {
            "timestamp": [],
            "base_lin_vel": [],
            "dof_pos": [],
            "dof_vel": [],
            "dof_torque": [],
            "command_vel": []
        }
        
        # Temporary storage for synchronization
        # We assume messages are written in blocks with same timestamp or close enough
        # We trigger a data point when we receive JointState
        last_cmd = [0.0, 0.0, 0.0] # vx, vy, wz
        last_odom_vel = [0.0, 0.0, 0.0] # vx, vy, vz
        
        with open(self.bag_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            start_time = None
            
            # Topics to read
            topics = ["/joint_states", "/odom", "/cmd_vel"]
            
            for schema, channel, message in reader.iter_decoded_messages(topics=topics):
                topic = channel.topic
                
                # 1. Command Velocity
                if topic == "/cmd_vel":
                    last_cmd = [message.linear.x, message.linear.y, message.angular.z]
                    
                # 2. Odometry (Base Velocity)
                elif topic == "/odom":
                    last_odom_vel = [
                        message.twist.twist.linear.x,
                        message.twist.twist.linear.y,
                        message.twist.twist.linear.z
                    ]
                    
                # 3. Joint States (Trigger)
                elif topic == "/joint_states":
                    # Extract timestamp
                    t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                    
                    if start_time is None:
                        start_time = t
                    
                    # Joint Data
                    # Ensure order is correct (FL, FR, RL, RR) or just take as is if logger preserves order
                    # go2_logger writes in 0-11 order
                    q = message.position
                    dq = message.velocity
                    tau = message.effort
                    
                    if len(q) != 12:
                        continue

                    data["timestamp"].append(t - start_time)
                    data["dof_pos"].append(q)
                    data["dof_vel"].append(dq)
                    data["dof_torque"].append(tau)
                    data["base_lin_vel"].append(last_odom_vel)
                    data["command_vel"].append(last_cmd)

        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
            
        print(f"[INFO] Loaded {len(data['timestamp'])} frames from Real Log.")
        return data


class SimToRealEvaluator:
    def __init__(self, sim_data, real_data):
        self.sim = sim_data
        self.real = real_data
        self.aligned = {}

    def align_data(self):
        """Resamples Real data to match Simulation timestamps."""
        print("[INFO] Aligning data (Resampling Real -> Sim)...")
        
        sim_t = self.sim["timestamp"]
        real_t = self.real["timestamp"]
        
        if len(real_t) == 0:
            raise ValueError("Real data is empty.")

        # Create interpolation functions for Real data
        # We use 'nearest' or 'linear' interpolation
        # Axis 0 is time
        
        self.aligned["timestamp"] = sim_t
        
        # Interpolate DOF Torque
        f_torque = interp1d(real_t, self.real["dof_torque"], axis=0, kind='linear', fill_value="extrapolate")
        self.aligned["real_dof_torque"] = f_torque(sim_t)
        
        # Interpolate Base Velocity
        f_vel = interp1d(real_t, self.real["base_lin_vel"], axis=0, kind='linear', fill_value="extrapolate")
        self.aligned["real_base_lin_vel"] = f_vel(sim_t)
        
        # Interpolate DOF Pos/Vel
        f_pos = interp1d(real_t, self.real["dof_pos"], axis=0, kind='linear', fill_value="extrapolate")
        self.aligned["real_dof_pos"] = f_pos(sim_t)
        
        f_dvel = interp1d(real_t, self.real["dof_vel"], axis=0, kind='linear', fill_value="extrapolate")
        self.aligned["real_dof_vel"] = f_dvel(sim_t)

    def calculate_gaps(self):
        """Calculates Sim-to-Real metrics."""
        print("[INFO] Calculating Gaps...")
        
        # Constants for Unitree Go2
        ROBOT_MASS = 15.0  # kg (approx for Go2 Pro/Edu)
        GRAVITY = 9.81
        
        # 1. Velocity Tracking Gap (Sim vs Real)
        # Sim is usually the "Reference" or "Ideal", Real is "Actual"
        # But here we want to see how different Real is from Sim for the SAME command.
        # Assuming Sim followed command perfectly, Sim is the reference.
        
        sim_vel_norm = np.linalg.norm(self.sim["base_lin_vel"][:, :2], axis=1) # XY speed
        real_vel_norm = np.linalg.norm(self.aligned["real_base_lin_vel"][:, :2], axis=1)
        
        vel_rmse = np.sqrt(np.mean((sim_vel_norm - real_vel_norm)**2))
        
        # 2. Torque Reality Gap
        # RMSE between Sim Torque and Real Torque
        # Shape: (Steps, 12)
        torque_diff = self.sim["dof_torque"] - self.aligned["real_dof_torque"]
        torque_rmse = np.sqrt(np.mean(torque_diff**2))
        
        # 3. CoT (Dimensionless)
        # CoT = Power / (Mass * Gravity * Velocity)
        
        # Sim Power
        sim_power = np.sum(np.abs(self.sim["dof_torque"] * self.sim["dof_vel"]), axis=1)
        sim_mech_power_mean = np.mean(sim_power)
        sim_vel_mean = np.mean(sim_vel_norm) + 1e-6
        sim_cot = sim_mech_power_mean / (ROBOT_MASS * GRAVITY * sim_vel_mean)
        
        # Real Power (using aligned data)
        real_power = np.sum(np.abs(self.aligned["real_dof_torque"] * self.aligned["real_dof_vel"]), axis=1)
        real_mech_power_mean = np.mean(real_power)
        real_vel_mean = np.mean(real_vel_norm) + 1e-6
        real_cot = real_mech_power_mean / (ROBOT_MASS * GRAVITY * real_vel_mean)
        
        cot_ratio = real_cot / (sim_cot + 1e-6)
        
        # 4. Torque Smoothness (Jitter)
        # Metric: Mean Absolute Derivative of Torque
        # Sim
        sim_torque_diff = np.diff(self.sim["dof_torque"], axis=0)
        sim_smoothness = np.mean(np.abs(sim_torque_diff))
        
        # Real
        real_torque_diff = np.diff(self.aligned["real_dof_torque"], axis=0)
        real_smoothness = np.mean(np.abs(real_torque_diff))
        
        return {
            "velocity_gap_rmse": vel_rmse,
            "torque_gap_rmse": torque_rmse,
            "sim_cot": sim_cot,
            "real_cot": real_cot,
            "cot_ratio": cot_ratio,
            "sim_torque_smoothness": sim_smoothness,
            "real_torque_smoothness": real_smoothness
        }

    def generate_report(self, metrics, output_dir="scripts/evaluation/result"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Ground Truths
        GT_COT = 0.4
        
        report_lines = []
        report_lines.append("="*50)
        report_lines.append(f"SIM-TO-REAL EVALUATION REPORT")
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*50)
        report_lines.append(f"1. Velocity Tracking Gap (RMSE): {metrics['velocity_gap_rmse']:.4f} m/s")
        report_lines.append(f"   (Lower is better, indicates Real matches Sim dynamics)")
        report_lines.append(f"2. Torque Reality Gap (RMSE):    {metrics['torque_gap_rmse']:.4f} Nm")
        report_lines.append(f"   (Key indicator of actuator model accuracy)")
        report_lines.append(f"3. Cost of Transport (CoT) [Dimensionless]")
        report_lines.append(f"   Ground Truth (Ideal): {GT_COT}")
        report_lines.append(f"   Sim CoT:  {metrics['sim_cot']:.4f}")
        report_lines.append(f"   Real CoT: {metrics['real_cot']:.4f}")
        report_lines.append(f"   Sim-to-Real Ratio: {metrics['cot_ratio']:.4f} (Target: 1.0)")
        report_lines.append(f"4. Torque Smoothness (Jitter)")
        report_lines.append(f"   Sim Smoothness:  {metrics['sim_torque_smoothness']:.4f}")
        report_lines.append(f"   Real Smoothness: {metrics['real_torque_smoothness']:.4f}")
        report_lines.append("="*50)
        
        report_str = "\n".join(report_lines)
        print(report_str)
        
        # Save to CSV (Daily Log)
        today_str = datetime.now().strftime('%Y-%m-%d')
        csv_filename = os.path.join(output_dir, f"sim2real_report_{today_str}.csv")
        
        csv_data = {
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Velocity_Tracking_Gap_RMSE": metrics['velocity_gap_rmse'],
            "Torque_Reality_Gap_RMSE": metrics['torque_gap_rmse'],
            "Sim_CoT": metrics['sim_cot'],
            "Real_CoT": metrics['real_cot'],
            "Sim_Real_CoT_Ratio": metrics['cot_ratio'],
            "Sim_Torque_Smoothness": metrics['sim_torque_smoothness'],
            "Real_Torque_Smoothness": metrics['real_torque_smoothness']
        }
        
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_data)
            
        print(f"[INFO] Report appended to {csv_filename}")


def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real Evaluator")
    parser.add_argument("--sim_id", type=str, required=True, help="Experiment name used in Sim (MinIO search key)")
    parser.add_argument("--sim_file", type=str, required=True, help="Path to local Simulation log (.pkl)")
    parser.add_argument("--real_bag", type=str, required=True, help="Path to Real-world ROS2 bag (.mcap)")
    parser.add_argument("--minio_endpoint", type=str, default=None)
    parser.add_argument("--minio_access", type=str, default=None)
    parser.add_argument("--minio_secret", type=str, default=None)
    parser.add_argument("--minio_bucket", type=str, default=None)
    
    args = parser.parse_args()
    
    # 1. Load Sim Data
    # minio_loader = MinioLogLoader(args.minio_endpoint, args.minio_access, args.minio_secret, args.minio_bucket)
    # try:
    #     sim_data = minio_loader.load(args.sim_id)
    # except Exception as e:
    #     print(f"[ERROR] Failed to load Sim data: {e}")
    #     return
    
    loader = LocalSimLogLoader(args.sim_file)
    try:
        sim_data = loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load Sim data: {e}")
        return

    # 2. Load Real Data
    bag_loader = RosBagLoader(args.real_bag)
    try:
        real_data = bag_loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load Real data: {e}")
        return

    # 3. Evaluate
    evaluator = SimToRealEvaluator(sim_data, real_data)
    evaluator.align_data()
    metrics = evaluator.calculate_gaps()
    evaluator.generate_report(metrics)

if __name__ == "__main__":
    main()
