# Real Robot Deployment for Unitree Go2

This directory contains information and scripts for deploying Isaac Lab trained policies to a real Unitree Go2 robot using `unitree_sdk2`.

## 1. Model Export

To export your trained policy to ONNX format, run the `play.py` script with your trained run. This script automatically exports the policy to `logs/rsl_rl/<experiment_name>/exported/policy.onnx`.

```bash
# Example command to play and export
./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 1
```

The exported ONNX model includes the observation normalizer, so you can feed raw observations directly.

## 2. Input/Output Structure

The policy expects an observation vector `obs` and outputs an action vector `actions`.

### Observation Space (Input)

The observation vector is a concatenation of the following terms (in order):

1.  **Base Linear Velocity** (3 dims): $(v_x, v_y, v_z)$ in base frame.
    *   *Note*: Real robots often don't have perfect state estimation for this. You may need to use an estimator or set to 0 if training with domain randomization/blind policy.
2.  **Base Angular Velocity** (3 dims): $(\omega_x, \omega_y, \omega_z)$ in base frame (from IMU gyroscope).
3.  **Projected Gravity** (3 dims): Gravity vector projected into base frame $(g_x, g_y, g_z)$.
    *   Computed from IMU orientation (quaternion/Euler).
4.  **Velocity Commands** (3 dims): Desired commands $(v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd})$.
5.  **Joint Positions** (12 dims): Joint positions relative to default configuration $(q - q_{default})$.
    *   Order: FL_hip, FL_thigh, FL_calf, FR_hip, ... (Check specific robot config).
6.  **Joint Velocities** (12 dims): Joint velocities $\dot{q}$.
7.  **Last Actions** (12 dims): The action vector from the previous step.
8.  **Height Scan** (Optional, ~187 dims): Height map around the robot.
    *   *Note*: For sim-to-real without a depth camera pipeline, this is usually removed (blind policy) or zeroed out.

### Action Space (Output)

The output `actions` (12 dims) corresponds to joint position targets (deltas).

*   **Control Mode**: PD Control
*   **Target Position**: $q_{target} = q_{default} + \text{action} \times \text{scale}$
*   **Scale**: Typically 0.25 (defined in `rough_env_cfg.py`).
*   **Kp/Kd**: Fixed gains (e.g., Kp=25, Kd=0.5).

## 3. Deployment with unitree_sdk2

The `deploy_unitree.py` script demonstrates how to interface with the robot using `unitree_sdk2` (Python bindings).

### Prerequisites
*   Unitree Go2 Robot (or simulation).
*   `unitree_sdk2` python bindings installed.
*   `onnxruntime` installed (`pip install onnxruntime`).
*   `numpy`.

### Usage

1.  Copy `policy.onnx` to this directory.
2.  Run the deployment script:
    ```bash
    python3 deploy_unitree.py <network_interface>
    # e.g., python3 deploy_unitree.py eth0
    ```

## 4. ROS 2 Integration (Optional)

If you prefer ROS 2, you can use `unitree_ros` packages.
*   Create a ROS 2 node that subscribes to `/lowstate` (or similar topic from `unitree_ros_to_real`).
*   Construct the observation vector.
*   Run ONNX inference.
*   Publish to `/lowcmd`.

Refer to `unitree_ros/unitree_legged_control` for C++ examples of low-level control.
