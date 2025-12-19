# Rapid Motor Adaptation (RMA) Implementation in IsaacLab

This directory contains scripts for implementing Rapid Motor Adaptation (RMA) for the Unitree Go2 robot. RMA enables the robot to adapt to varying physical parameters (like friction, payload) by estimating them from proprioceptive history.

## Architecture

The implementation follows a two-phase training process:

### Phase 1: Teacher Policy Training
The teacher policy is trained using **Privileged Information** (ground truth physics parameters) that is available in simulation but not in the real world.

- **Task:** `Isaac-Velocity-Rough-Unitree-Go2-RMA-v0`
- **Observations:**
  - `policy`: Proprioception + Privileged Info (Extrinsics)
  - `privileged`: Explicit privileged parameters (e.g., body mass, friction)
  - `proprio`: History of proprioceptive states (joint pos, vel, actions)
- **Script:** Standard `train.py` (using RSL-RL)

### Phase 2: Adaptation Module Training
The adaptation module is trained to estimate the **Privileged Information** (Extrinsics) solely from the **Proprioception History**.

- **Input:** Proprioception History (`proprio` group)
- **Target:** Privileged Information (`privileged` group)
- **Teacher:** Frozen policy from Phase 1
- **Script:** `train_adapt.py`

### Phase 3: Deployment (Student Policy)
The student policy combines the frozen Teacher Policy and the trained Adaptation Module.

- **Flow:** `Proprio History` -> **Adaptation Module** -> `Predicted Extrinsics` + `Base Obs` -> **Teacher Policy** -> `Action`
- **Script:** `play_adapt.py`

## Usage

### 1. Train Teacher Policy (Phase 1)
Train the base policy with access to privileged information.

```bash
# Run from the root of the repository
isaaclab.bat -p source/isaaclab/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --headless
```

### 2. Train Adaptation Module (Phase 2)
Train the adaptation module to predict extrinsics.

```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/train_adapt.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --load_run unitree_go2_rough_rma_phase1 ^
    --load_checkpoint model_100.pt ^
    --headless
```
*Note: This will save results to `logs/rma/<experiment_name>_phase2`.*

### 3. Play / Evaluate (Student Policy)
Run the combined policy in the simulator.

```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/play_adapt.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --num_envs 32 ^
    --run_phase2 unitree_go2_rough_rma_phase2
```
*Note: The script automatically locates the corresponding Teacher Policy using `teacher_policy_info.txt` generated in Phase 2.*

### 4. Export to ONNX
Export the combined Student Policy (Adaptation + Teacher) to ONNX format for deployment.

```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/export_adapt.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --run_phase2 unitree_go2_rough_rma_phase2
```
*Note: The exported model will be saved in `logs/rma/<experiment_name>_phase2/exported_models/rma_policy.onnx`.*
*Inputs: `base_obs`, `proprio_hist` -> Output: `actions`*

## File Structure

- `train_adapt.py`: Script for training the adaptation module (Supervised Learning).
- `play_adapt.py`: Inference script for the full RMA pipeline.
- `export_adapt.py`: Script to export the combined policy to ONNX.
- `velocity_env_cfg.py` (in `isaaclab_tasks`): Defines the `LocomotionVelocityRoughEnvCfg_RMA` class and observation groups.

## Configuration Details

The environment configuration (`velocity_env_cfg.py`) defines three key observation groups:
1. **`policy`**: Inputs to the RL agent. Includes `base_mass` and `friction` (Extrinsics).
2. **`privileged`**: Targets for the adaptation module. Identical to the Extrinsics in `policy`.
3. **`proprio`**: Inputs to the adaptation module. Includes history of joint positions, velocities, and actions.
