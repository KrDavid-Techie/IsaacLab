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
    *   **Mapping**: The script automatically handles the reordering between Unitree SDK order (FR, FL, RR, RL) and Isaac Lab order (FL, FR, RL, RR).
6.  **Joint Velocities** (12 dims): Joint velocities $\dot{q}$.
7.  **Last Actions** (12 dims): The action vector from the previous step.
8.  **Height Scan** (Optional, ~187 dims): Height map around the robot.
    *   *Note*: The `deploy_unitree.py` script estimates the robot's base height using Forward Kinematics (assuming flat ground) and populates this field with `-estimated_height`. This enables running policies trained with height scans on a blind robot on flat terrain.

### Action Space (Output)

The output `actions` (12 dims) corresponds to joint position targets (deltas).

*   **Control Mode**: PD Control
*   **Target Position**: $q_{target} = q_{default} + \text{action} \times \text{scale}$
*   **Scale**: Typically 0.25 (defined in `rough_env_cfg.py`).
*   **Kp/Kd**: Fixed gains (e.g., Kp=25, Kd=0.5).

## 3. Deployment Methods

### Option A: ROS 2 Package (Recommended)

The `go2_rl_deploy` directory contains a ROS 2 package that wraps the deployment logic into a ROS 2 node.

#### Features
*   **Safety**: Sends `StandDown` command on startup and exit.
*   **Soft Start**: Gradually interpolates from current lying position to standing position over 3 seconds.
*   **ROS 2 Integration**: Subscribes to `/cmd_vel` for control commands.
*   **Auto-Stand**: If no command is received or command is zero, the robot holds its standing position without running the policy inference (saving compute and ensuring stability).

#### Prerequisites
*   ROS 2 (Humble or newer recommended)
*   `unitree_sdk2py` installed
*   `onnxruntime`, `numpy`, `scipy`

#### Build & Run

1.  **Build the package**:
    ```bash
    cd real_bot
    colcon build --packages-select go2_rl_deploy
    source install/setup.bash
    ```

2.  **Run the node**:
    ```bash
    ros2 run go2_rl_deploy deploy_node --ros-args -p policy_path:=/path/to/policy.onnx -p network_interface:=enp2s0
    ```
    *   `policy_path`: Absolute path to your exported ONNX file.
    *   `network_interface`: Network interface connected to the robot (e.g., `eth0`, `enp2s0`). If not provided, it tries to auto-detect from `CYCLONEDDS_URI`.

3.  **Control**:
    Publish `geometry_msgs/Twist` to `/cmd_vel` to move the robot.
    ```bash
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
    ```

### Option B: Standalone Script

You can also run the standalone script `deploy_unitree.py` if you don't want to build a ROS 2 package.

```bash
python3 deploy_unitree.py
```
*   Ensure `unitree_sdk2py` and `onnxruntime` are installed.
*   Edit the script to set `POLICY_PATH` and `network_interface` manually if needed.

## 4. Safety & Troubleshooting

*   **Soft Start**: The robot will take 3 seconds to slowly stand up. Do not send commands during this time.
*   **E-Stop**: Always have the remote controller ready to press the emergency stop button.
*   **Suspension Test**: For the first run, suspend the robot so its feet don't touch the ground to verify leg movements.
*   **Interface**: If the robot doesn't respond, check your network interface name (`ifconfig` or `ip a`) and ensure you can ping the robot.

