import time
import sys
import os
import socket
import platform
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R


# ROS 2 Imports
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
except ImportError:
    print("Warning: ROS 2 (rclpy) not found. Command velocity will be fixed.")
    rclpy = None

# Unitree SDK Imports
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.low_level import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.go2.low_level import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd__init
except ImportError:
    print("Error: unitree_sdk2py not found. Please install the Unitree SDK 2 Python bindings.")
    sys.exit(1)

# --- Configuration ---
POLICY_PATH = ".\\sota3-2025-12-08_13-55-31\\exported\\policy.onnx"
# POLICY_PATH = "exported/policy.onnx" # Path to the ONNX model
ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5
DT = 0.02  # Control cycle (50Hz)
NUM_ACTIONS = 12
HEIGHT_SCAN_SIZE = 187 # From env.yaml (1.6x1.0m grid with 0.1m resolution -> 17x11 points)

# Default Joint Positions (Isaac Lab Order: FL, FR, RL, RR)
# FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf 
# RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5   # RR
], dtype=np.float32)

# Joint Mapping Indices
# Unitree SDK Order: FR, FL, RR, RL
# Isaac Lab Order:   FL, FR, RL, RR

# Unitree (FR, FL, RR, RL) -> Isaac (FL, FR, RL, RR)
# FR(0-2)   -> Isaac(3-5)
# FL(3-5)   -> Isaac(0-2)
# RR(6-8)   -> Isaac(9-11)
# RL(9-11)  -> Isaac(6-8)
UNITREE_TO_ISAAC_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] # for swapping pairs

# Isaac (FL, FR, RL, RR) -> Unitree (FR, FL, RR, RL)
# FL(0-2)   -> Unitree(3-5)
# FR(3-5)   -> Unitree(0-2)
# RL(6-8)   -> Unitree(9-11)
# RR(9-11)  -> Unitree(6-8)
ISAAC_TO_UNITREE_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] # for swapping pairs

def get_network_interface(input_str):
    """
    Resolves an IP address to a network interface name on Linux.
    If the input is not a valid IP or we are not on Linux, returns the input as is.
    """
    try:
        # Check if input is a valid IP
        socket.inet_aton(input_str)
        is_ip = True
    except socket.error:
        is_ip = False

    if is_ip and platform.system() == 'Linux':
        try:
            import fcntl
            import struct
            
            # Iterate over interfaces to find the one with this IP
            if os.path.exists('/sys/class/net'):
                interfaces = os.listdir('/sys/class/net')
                for iface in interfaces:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        # SIOCGIFADDR = 0x8915
                        if_ip = socket.inet_ntoa(fcntl.ioctl(
                            s.fileno(),
                            0x8915,
                            struct.pack('256s', iface[:15].encode('utf-8'))
                        )[20:24])
                        if if_ip == input_str:
                            print(f"Resolved IP {input_str} to interface {iface}")
                            return iface
                    except Exception:
                        continue
        except ImportError:
            pass # fcntl not available
        except Exception as e:
            print(f"Warning: Failed to resolve IP to interface: {e}")
    
    return input_str

class RobotController:
    def __init__(self, network_interface):
        print(f"Initializing RobotController on {network_interface}...")
        
        # Initialize SDK
        ChannelFactoryInitialize(0, network_interface)
        self.pub = ChannelFactoryInitialize(0).createPublisher("rt/lowcmd", unitree_go_msg_dds__LowCmd_)
        self.sub = ChannelFactoryInitialize(0).createSubscriber("rt/lowstate", unitree_go_msg_dds__LowState_)
        
        # Initialize LowCmd
        self.cmd_msg = unitree_go_msg_dds__LowCmd__init()
        self.cmd_msg.head[0] = 0xFE
        self.cmd_msg.head[1] = 0xEF
        self.cmd_msg.levelFlag = 0xFF
        self.cmd_msg.gpio = 0
        for i in range(20):
            self.cmd_msg.motorCmd[i].mode = 0x01    # Servo Low-cmd mode
            self.cmd_msg.motorCmd[i].q = 0.0        # Will be set later
            self.cmd_msg.motorCmd[i].dq = 0.0       # Will be set later
            self.cmd_msg.motorCmd[i].Kp = 0.0       # Will be set later
            self.cmd_msg.motorCmd[i].Kd = 0.0       # Will be set later
            self.cmd_msg.motorCmd[i].tau = 0.0      # Will be set later

        # Load Policy
        try:
            self.policy_session = ort.InferenceSession(POLICY_PATH)
            self.input_name = self.policy_session.get_inputs()[0].name
            self.input_shape = self.policy_session.get_inputs()[0].shape
            self.expected_obs_dim = self.input_shape[1]
            print(f"Policy loaded from {POLICY_PATH}. Expected obs dim: {self.expected_obs_dim}")
        except Exception as e:
            print(f"Error loading policy: {e}")
            sys.exit(1)

        # State variables
        self.last_actions = np.zeros(NUM_ACTIONS, dtype=np.float32) # Initialize last actions to zeros
        self.target_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32) # vx, vy, wz

        # --- ROS 2 Setup ---
        if rclpy:
            rclpy.init()
            self.ros_node = Node('rl_policy_runner')
            self.ros_sub = self.ros_node.create_subscription(
                Twist,
                'cmd_vel',
                self.cmd_vel_callback,
                10
            )
            # ROS 2 스핀을 별도 스레드에서 실행 (메인 제어 루프 방해 금지)
            self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
            self.ros_thread.start()
            print("ROS 2 Node initialized. Subscribed to /cmd_vel")
        else:
            print("ROS 2 not initialized. Using fixed target velocity.")

        print("Waiting for LowState...")
        while True:
            if self.sub.getData() is not None:
                print("Received LowState.")
                break
            time.sleep(0.1)


    def cmd_vel_callback(self, msg):
            """ROS 2 cmd_vel callback Function"""
            vx = msg.linear.x
            vy = msg.linear.y
            wz = msg.angular.z
            # 필요하다면 여기서 속도 제한(clipping)이나 스케일링을 수행하세요.
            self.target_vel = np.array([vx, vy, wz], dtype=np.float32)
            # print(f"Cmd Vel Updated: {self.target_vel}")


    def get_gravity_vector(self, quaternion):
        # Unitree Quat: [w, x, y, z] -> Scipy expects [x, y, z, w]
        r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        # Projected Gravity: R^T * [0, 0, -1]
        return r.apply([0, 0, -1], inverse=True).astype(np.float32)

    def estimate_base_height(self, q):
        # Estimate robot base height from ground using Forward Kinematics
        # Assumes feet are on the ground.
        # q: Joint positions in Isaac Lab order (FL, FR, RL, RR)
        # Constants for Go2
        L1 = 0.213 # Thigh length
        L2 = 0.213 # Calf length
        
        heights = []
        for i in range(4):
            # FL, FR, RL, RR
            idx = i * 3
            theta1 = q[idx]     # Hip Roll
            theta2 = q[idx+1]   # Thigh Pitch
            theta3 = q[idx+2]   # Calf Pitch
            
            # Y offset of thigh joint in hip frame
            # FL(0), RL(2) -> Left (+), FR(1), RR(3) -> Right (-)
            y_t = 0.0955 if (i % 2 == 0) else -0.0955
            
            # Height calculation (Z distance from foot to base)
            # h = L1*cos(th1)*cos(th2) + L2*cos(th1)*cos(th2+th3) - yt*sin(th1)
            h = L1 * np.cos(theta1) * np.cos(theta2) + L2 * np.cos(theta1) * np.cos(theta2 + theta3) - y_t * np.sin(theta1)

            heights.append(h)
            
        return np.mean(heights)

    def get_observation(self, state):
        # 1. Base Linear Velocity (Estimated)
        # Real robot usually doesn't provide accurate base velocity without VIO/Lidar.
        # We use 0.0 or state estimator if available. For robustness, often 0.0 is used in blind deployment.
        base_lin_vel = np.zeros(3, dtype=np.float32) # Initialize to zeros
        
        # 2. Base Angular Velocity (Gyro)
        base_ang_vel = np.array(state.imu.gyroscope, dtype=np.float32) # Initialize to IMU's gyroscope
        
        # 3. Projected Gravity
        projected_gravity = self.get_gravity_vector(state.imu.quaternion) # Initialize to IMU's quaternion
        
        # 4. Commands (Velocity) vx, vy, wz
        commands = self.target_vel
        
        # Read Raw (Unitree Order: FR, FL, RR, RL)
        q_raw = np.array([m.q for m in state.motorState[:12]], dtype=np.float32)# 5. Joint Positions 
        dq_raw = np.array([m.dq for m in state.motorState[:12]], dtype=np.float32) # 6. Joint Velocities
        
        # Convert to Isaac Order (FL, FR, RL, RR)
        q_isaac = q_raw[UNITREE_TO_ISAAC_IDX]
        dq_isaac = dq_raw[UNITREE_TO_ISAAC_IDX]
        
        # Relative Joint Positions
        joint_pos_rel = q_isaac - DEFAULT_JOINT_POS
        
        # 7. Last Actions
        last_actions = self.last_actions
        
        # 8. Height Scan
        # Estimate base height and use it to populate height scan (assuming flat ground)
        # Height Scan = z_terrain - z_base ~= -estimated_height
        est_height = self.estimate_base_height(q_isaac)
        height_scan = np.full(HEIGHT_SCAN_SIZE, -est_height, dtype=np.float32)
        
        # Concatenate
        obs = np.concatenate([
            base_lin_vel,      # 3
            base_ang_vel,      # 3
            projected_gravity, # 3
            commands,          # 3
            joint_pos_rel,     # 12
            dq_isaac,          # 12
            last_actions,      # 12
            height_scan        # 187
        ])
        
        # Verify dimension
        if obs.shape[0] != self.expected_obs_dim:
            # If mismatch, try to pad or trim (fallback)
            # print(f"Warning: Obs dim mismatch. Expected {self.expected_obs_dim}, got {obs.shape[0]}")
            if obs.shape[0] < self.expected_obs_dim:
                obs = np.concatenate([obs, np.zeros(self.expected_obs_dim - obs.shape[0], dtype=np.float32)])
            else:
                obs = obs[:self.expected_obs_dim]
                
        return obs.astype(np.float32)

    def run(self):
        print("Starting control loop...")
        try:
            while True:
                start_time = time.time()
                
                # 1. Receive State
                state = self.sub.getData()
                if state is not None:
                    # 2. Get Observation
                    obs = self.get_observation(state)
                    
                    # 3. Inference
                    # ONNX Runtime expects list of inputs, usually named 'obs'
                    ort_inputs = {self.input_name: obs.reshape(1, -1)}
                    action_raw = self.policy_session.run(None, ort_inputs)[0][0]
                    
                    self.last_actions = action_raw
                    
                    # 4. Process Action (Sim to Real)
                    # Target q = Default + Action * Scale
                    q_target_isaac = DEFAULT_JOINT_POS + (action_raw * ACTION_SCALE)
                    
                    # Convert Isaac Order (FL, FR, RL, RR) -> Unitree Order (FR, FL, RR, RL)
                    q_target_unitree = q_target_isaac[ISAAC_TO_UNITREE_IDX]
                    
                    # 5. Send Command
                    for i in range(12):
                        self.cmd_msg.motorCmd[i].q = q_target_unitree[i]
                        self.cmd_msg.motorCmd[i].dq = 0.0
                        self.cmd_msg.motorCmd[i].Kp = KP
                        self.cmd_msg.motorCmd[i].Kd = KD
                        self.cmd_msg.motorCmd[i].tau = 0.0
                    
                    self.cmd_msg.crc = CRC().Crc(self.cmd_msg)
                    self.pub.write(self.cmd_msg)
                
                # Maintain Frequency
                elapsed = time.time() - start_time
                if elapsed < DT:
                    time.sleep(DT - elapsed)
                    
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python deploy_unitree.py <network_interface || ip>")
        print("Example: python deploy_unitree.py eth0")
        print("Example: python deploy_unitree.py 192.168.123.10")
        sys.exit(1)
        
    net_iface = get_network_interface(sys.argv[1])
    controller = RobotController(net_iface) 
    # Set initial command (e.g., walk forward slowly)
    # controller.target_vel = np.array([0.3, 0.0, 0.0], dtype=np.float32) 
    controller.run()