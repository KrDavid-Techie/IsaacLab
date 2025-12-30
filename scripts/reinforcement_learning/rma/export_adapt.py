# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export RMA Student Policy (Teacher + Adaptation Module) to ONNX.
This script runs without launching the Isaac Sim simulation app.
"""

import argparse
import os

# Fix for OMP error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import copy
import re

# ---------------------------------------------------------
# Helper Classes
# ---------------------------------------------------------

class Normalizer(nn.Module):
    """Simple Normalizer module compatible with RSL-RL EmpiricalNormalization."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, x):
        return (x - self.mean) / self.std

class RMAExportModule(nn.Module):
    """Combined RMA Module for Export."""
    def __init__(self, adaptation_module, teacher_actor, normalizer=None):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.teacher_actor = teacher_actor
        if normalizer:
            self.normalizer = normalizer
        else:
            self.normalizer = nn.Identity()
            
    def forward(self, base_obs, proprio_hist):
        # Predict Extrinsics
        pred_extrinsics = self.adaptation_module(proprio_hist)
        # Concatenate Base + Predicted
        student_obs = torch.cat([base_obs, pred_extrinsics], dim=-1)
        # Normalize
        student_obs = self.normalizer(student_obs)
        # Policy Inference
        return self.teacher_actor(student_obs)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def get_log_root(run_name, folder_name="rsl_rl"):
    """Helper to find log root."""
    # Check custom paths first
    if os.path.exists(os.path.join("logs", "ram", run_name)):
        return os.path.join("logs", "ram")
    elif os.path.exists(os.path.join("logs", "rma", run_name)):
        return os.path.join("logs", "rma")
    else:
        return os.path.join("logs", folder_name)

def get_checkpoint_path(run_dir, load_run, load_checkpoint):
    """Find the checkpoint path."""
    if load_run == ".*":
        # Find latest run
        runs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        runs.sort()
        if not runs:
            raise ValueError(f"No runs found in {run_dir}")
        run_path = os.path.join(run_dir, runs[-1])
    else:
        run_path = os.path.join(run_dir, load_run)
        
    if load_checkpoint == "model_.*.pt":
        # Find latest checkpoint
        files = [f for f in os.listdir(run_path) if f.startswith("model_") and f.endswith(".pt")]
        # Sort by number
        files.sort(key=lambda x: int(re.search(r"model_(\d+).pt", x).group(1)))
        if not files:
            raise ValueError(f"No checkpoints found in {run_path}")
        ckpt_file = files[-1]
    else:
        ckpt_file = load_checkpoint
        
    return os.path.join(run_path, ckpt_file)

def build_mlp_from_weights(state_dict, prefix="actor."):
    """Reconstruct an MLP from state dict weights."""
    layers = []
    
    # Find all weight keys
    weight_keys = [k for k in state_dict.keys() if k.startswith(prefix) and k.endswith(".weight")]
    # Sort by layer index
    weight_keys.sort(key=lambda x: int(x.split(".")[1]))
    
    for i, key in enumerate(weight_keys):
        weight = state_dict[key]
        bias_key = key.replace(".weight", ".bias")
        bias = state_dict[bias_key]
        
        out_features, in_features = weight.shape
        linear = nn.Linear(in_features, out_features)
        linear.weight.data = weight
        linear.bias.data = bias
        layers.append(linear)
        
        # Add activation if not last layer
        if i < len(weight_keys) - 1:
            layers.append(nn.ELU())
            
    return nn.Sequential(*layers)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export RMA Student Policy to ONNX.")
    parser.add_argument("--run_phase2", type=str, default=None, help="Name of the Adaptation Module run (Phase 2).")
    parser.add_argument("--ckpt_phase2", type=str, default="adaptation_module.pt", help="Checkpoint file for Adaptation Module. Can be a full path.")
    parser.add_argument("--run_phase1", type=str, default=None, help="Name of the Teacher Policy run (Phase 1).")
    parser.add_argument("--ckpt_phase1", type=str, default=None, help="Checkpoint file for Teacher Policy. Can be a full path.")
    parser.add_argument("--out_dir", type=str, default="exported_models", help="Directory to save the exported model.")
    parser.add_argument("--filename", type=str, default="rma_policy.onnx", help="Name of the exported ONNX file.")
    parser.add_argument("--export_teacher", action="store_true", default=False, help="Export only the Teacher Policy (Phase 1).")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 0. Locate Phase 2 Run (if needed)
    # ---------------------------------------------------------
    run_path_p2 = None
    
    if not args.export_teacher:
        if args.ckpt_phase2 and os.path.isfile(args.ckpt_phase2):
            run_path_p2 = os.path.dirname(args.ckpt_phase2)
            print(f"[INFO] Using direct path for Phase 2: {run_path_p2}")
        elif args.run_phase2:
            log_root_p2 = get_log_root(args.run_phase2, folder_name="rma")
            run_path_p2 = os.path.join(log_root_p2, args.run_phase2)
            
            # If direct path doesn't exist, try searching one level deep (e.g. logs/rma/<experiment>/<timestamp>)
            if not os.path.exists(run_path_p2) and os.path.exists(log_root_p2):
                for subdir in os.listdir(log_root_p2):
                    candidate = os.path.join(log_root_p2, subdir, args.run_phase2)
                    if os.path.isdir(candidate):
                        run_path_p2 = candidate
                        break

            if os.path.exists(run_path_p2):
                # Check if this is an experiment folder (contains subdirs) or a run folder
                # We assume a run folder has the checkpoint or info file, or no subdirs that look like timestamps
                has_ckpt = os.path.exists(os.path.join(run_path_p2, args.ckpt_phase2))
                has_info = os.path.exists(os.path.join(run_path_p2, "teacher_policy_info.txt"))
                
                if not (has_ckpt or has_info):
                    subdirs = [d for d in os.listdir(run_path_p2) if os.path.isdir(os.path.join(run_path_p2, d))]
                    subdirs.sort()
                    if subdirs:
                        run_path_p2 = os.path.join(run_path_p2, subdirs[-1])
        else:
            raise ValueError("Must provide --run_phase2 or a valid file path in --ckpt_phase2.")
            
        print(f"[INFO] Phase 2 Run Directory: {run_path_p2}")

    # ---------------------------------------------------------
    # 1. Load Teacher Policy (Phase 1)
    # ---------------------------------------------------------
    resume_path_p1 = None
    
    # Check if direct path provided
    if args.ckpt_phase1:
        # Check if it is a file
        if os.path.isfile(args.ckpt_phase1):
            resume_path_p1 = args.ckpt_phase1
        elif not args.run_phase1:
            # If run_phase1 is NOT provided, then ckpt_phase1 MUST be a valid path.
            # If it's not a file, we should raise an error here to be helpful.
            raise FileNotFoundError(f"Provided checkpoint file does not exist: {args.ckpt_phase1}")
    
    # Try to read from info file if not provided directly and we have Phase 2 run path
    if not resume_path_p1 and run_path_p2:
        info_file = os.path.join(run_path_p2, "teacher_policy_info.txt")
        if os.path.exists(info_file):
            with open(info_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip().startswith("Teacher Policy Path:"):
                        parts = line.split(":", 1)
                        if len(parts) > 1 and parts[1].strip():
                            resume_path_p1 = parts[1].strip()
                        elif i + 1 < len(lines):
                            resume_path_p1 = lines[i+1].strip()
                        
                        if resume_path_p1:
                            print(f"[INFO] Found Teacher Policy path in info file: {resume_path_p1}")
                            break
    
    # Try to find from run name
    if not resume_path_p1 and args.run_phase1:
        log_root_p1 = get_log_root(args.run_phase1)
        ckpt = args.ckpt_phase1 if args.ckpt_phase1 else "model_.*.pt"
        resume_path_p1 = get_checkpoint_path(
            os.path.join(log_root_p1, args.run_phase1), 
            ".*", 
            ckpt
        )
        
    if not resume_path_p1:
        raise ValueError("Could not determine Teacher Policy path. Provide --ckpt_phase1 (path) or --run_phase1.")

    print(f"[INFO]: Loading Teacher Policy from: {resume_path_p1}")
    
    # Load Teacher Checkpoint
    teacher_ckpt = torch.load(resume_path_p1, map_location="cpu")
    model_state_dict = teacher_ckpt['model_state_dict']
    
    # Reconstruct Teacher Actor MLP
    print("[INFO] Reconstructing Teacher Actor...")
    teacher_actor = build_mlp_from_weights(model_state_dict, prefix="actor.")
    teacher_actor.eval()
    
    # Load Normalizer if exists
    normalizer = None
    if 'obs_norm_state_dict' in teacher_ckpt:
        print("[INFO] Found observation normalizer in checkpoint.")
        norm_state = teacher_ckpt['obs_norm_state_dict']
        # RSL-RL EmpiricalNormalization stores 'mean', 'var', 'count'
        # We need mean and std = sqrt(var)
        mean = norm_state['mean']
        var = norm_state['var']
        std = torch.sqrt(var + 1e-8) # Add epsilon for numerical stability
        normalizer = Normalizer(mean, std)
        normalizer.eval()
    else:
        print("[WARNING] No observation normalizer found. Using Identity.")

    # ---------------------------------------------------------
    # 2. Export Teacher Only (if requested)
    # ---------------------------------------------------------
    if args.export_teacher:
        print("[INFO] Exporting Teacher Policy Only...")
        
        # Create Export Module (Teacher + Normalizer)
        class TeacherExportModule(nn.Module):
            def __init__(self, actor, normalizer=None):
                super().__init__()
                self.actor = actor
                self.normalizer = normalizer if normalizer else nn.Identity()
                
            def forward(self, obs):
                return self.actor(self.normalizer(obs))
                
        export_module = TeacherExportModule(teacher_actor, normalizer)
        export_module.eval()
        
        # Infer input dim
        input_dim = teacher_actor[0].in_features
        print(f"[INFO] Inferred Input Dim: {input_dim}")
        
        dummy_input = torch.zeros(1, input_dim)
        
        # Output directory
        out_dir = args.out_dir
        if run_path_p2:
             out_dir = os.path.join(run_path_p2, args.out_dir)
        else:
             # If no phase 2 run, save in phase 1 run dir or current dir
             p1_dir = os.path.dirname(resume_path_p1)
             out_dir = os.path.join(p1_dir, args.out_dir)
             
        os.makedirs(out_dir, exist_ok=True)
        onnx_path = os.path.join(out_dir, "teacher_policy.onnx")
        
        print(f"[INFO] Exporting to: {onnx_path}")
        
        torch.onnx.export(
            export_module,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=18,
            verbose=False,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "actions": {0: "batch_size"},
            }
        )
        print("[INFO] Teacher Export Complete!")
        return

    # ---------------------------------------------------------
    # 3. Load Adaptation Module (Phase 2)
    # ---------------------------------------------------------
    if args.ckpt_phase2 and os.path.isfile(args.ckpt_phase2):
        adapt_ckpt_path = args.ckpt_phase2
    else:
        adapt_ckpt_path = os.path.join(run_path_p2, args.ckpt_phase2)
        
    print(f"[INFO]: Loading Adaptation Module from: {adapt_ckpt_path}")
    
    adapt_state_dict = torch.load(adapt_ckpt_path, map_location="cpu")
    
    # Infer dimensions
    # First layer weight: (128, proprio_dim)
    proprio_dim = adapt_state_dict['0.weight'].shape[1]
    # Last layer weight: (privileged_dim, 128)
    privileged_dim = adapt_state_dict['4.weight'].shape[0]
    
    print(f"[INFO] Inferred Proprio Dim: {proprio_dim}")
    print(f"[INFO] Inferred Privileged Dim: {privileged_dim}")
    
    adaptation_module = nn.Sequential(
        nn.Linear(proprio_dim, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, privileged_dim)
    )
    adaptation_module.load_state_dict(adapt_state_dict)
    adaptation_module.eval()

    # ---------------------------------------------------------
    # 3. Export to ONNX
    # ---------------------------------------------------------
    
    # Create Export Module
    rma_module = RMAExportModule(adaptation_module, teacher_actor, normalizer)
    rma_module.eval()
    
    # Determine Base Obs Dim
    # Teacher input dim = base_obs_dim + privileged_dim
    teacher_input_dim = teacher_actor[0].in_features
    base_obs_dim = teacher_input_dim - privileged_dim
    
    print(f"[INFO] Inferred Base Obs Dim: {base_obs_dim}")
    
    # Dummy inputs
    dummy_base_obs = torch.zeros(1, base_obs_dim)
    dummy_proprio_hist = torch.zeros(1, proprio_dim)
    
    # Output directory
    out_path = os.path.join(run_path_p2, args.out_dir)
    os.makedirs(out_path, exist_ok=True)
    onnx_path = os.path.join(out_path, args.filename)
    
    print(f"[INFO] Exporting to: {onnx_path}")
    
    torch.onnx.export(
        rma_module,
        (dummy_base_obs, dummy_proprio_hist),
        onnx_path,
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=["base_obs", "proprio_hist"],
        output_names=["actions"],
        dynamic_axes={
            "base_obs": {0: "batch_size"},
            "proprio_hist": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
    )
    
    print("[INFO] Export Complete!")

if __name__ == "__main__":
    main()
