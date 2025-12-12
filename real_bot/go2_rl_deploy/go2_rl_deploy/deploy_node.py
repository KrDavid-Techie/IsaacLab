import time
import sys
import os
import socket
import platform
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
import threading
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Unitree SDK Imports
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import LowCmd_ as unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.default import LowState_ as unitree_go_msg_dds__LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as unitree_go_msg_dds__LowCmd__init
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError as e:
    print(f"Error: unitree_sdk2py not found or failed to import. {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Configuration Constants ---
ACTION_SCALE = 0.15  # Reduced from 0.25 to reduce excessive movement
KP = 20.0  # Reduced from 25.0 for softer movement
KD = 0.8   # Increased from 0.5 for better damping
DT = 0.02  # Control cycle (50Hz)
NUM_ACTIONS = 12
HEIGHT_SCAN_SIZE = 187
CMD_TIMEOUT = 0.5  # Command timeout in seconds
SOFT_START_DURATION = 3.0  # Increased from 3.5s for slower startup

# Default Joint Positions (Isaac Lab Order: FL, FR, RL, RR)
# Modified to stand taller (smaller thigh angle, smaller calf angle magnitude)
DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,   # FL (hip, thigh, calf)
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5   # RR
], dtype=np.float32)


# Indices mapping
# Unitree (FR, FL, RR, RL) <-> Isaac (FL, FR, RL, RR)
UNITREE_TO_ISAAC_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
ISAAC_TO_UNITREE_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

def get_interface_from_cyclonedds_env():
    """
    CYCLONEDDS_URI 환경 변수에서 네트워크 인터페이스 이름을 추출합니다.
    """
    uri = os.environ.get('CYCLONEDDS_URI')
    if not uri:
        return None
    
    try:
        if os.path.isfile(uri):
            tree = ET.parse(uri)
            root = tree.getroot()
        elif uri.strip().startswith('<'):
            root = ET.fromstring(uri)
        else:
            return None

        for interface in root.iter('NetworkInterface'):
            name = interface.get('name')
            if name:
                return name
    except Exception as e:
        print(f"Warning: Failed to parse CYCLONEDDS_URI: {e}")
        return None
    return None

def get_network_interface(input_str):
    try:
        socket.inet_aton(input_str)
        is_ip = True
    except socket.error:
        is_ip = False

    if is_ip and platform.system() == 'Linux':
        try:
            import fcntl
            import struct
            if os.path.exists('/sys/class/net'):
                interfaces = os.listdir('/sys/class/net')
                for iface in interfaces:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        if_ip = socket.inet_ntoa(fcntl.ioctl(
                            s.fileno(),
                            0x8915,
                            struct.pack('256s', iface[:15].encode('utf-8'))
                        )[20:24])
                        if if_ip == input_str:
                            return iface
                    except Exception:
                        continue
        except ImportError:
            pass
    return input_str

class RobotController(Node):
    def __init__(self):
        super().__init__('go2_rl_deploy_node')
        
        # Parameters
        self.declare_parameter('network_interface', 'enp2s0')
        self.declare_parameter('policy_path', r'.\sota3-2025-12-08_13-55-31\exported\policy.onnx')
        
        net_iface_param = self.get_parameter('network_interface').get_parameter_value().string_value
        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        
        # Network Interface Detection
        if net_iface_param:
            self.network_interface = net_iface_param
        else:
            detected_iface = get_interface_from_cyclonedds_env()
            if detected_iface:
                self.network_interface = detected_iface
                self.get_logger().info(f"Auto-detected network interface: {self.network_interface}")
            else:
                self.network_interface = 'enp2s0'
                self.get_logger().warn(f"Could not detect interface. Defaulting to {self.network_interface}")

        self.get_logger().info(f"Initializing RobotController on {self.network_interface}...")
        
        # Initialize SDK Channel (Required for both SportClient and LowLevel)
        net_iface = get_network_interface(self.network_interface)
        ChannelFactoryInitialize(0, net_iface)

        # --- Step 1: StandDown (Safety) ---
        self.get_logger().info("Step 1: Sending StandDown command...")
        try:
            sport_client = SportClient()
            sport_client.SetTimeout(5.0)
            sport_client.Init()
            sport_client.StandDown()
            self.get_logger().info("Waiting 3 seconds for robot to lie down...")
            time.sleep(3.0)

        except Exception as e:
            self.get_logger().warn(f"Failed to send StandDown (Robot might be already in low level?): {e}")

        # --- Step 3: Joint Mapping ---
        # Isaac Lab (FL, FR, RL, RR) -> Unitree (FR, FL, RR, RL)
        self.default_joint_pos_mapped = DEFAULT_JOINT_POS[ISAAC_TO_UNITREE_IDX]
        self.get_logger().info(f"Step 3: Joint mapping initialized.")

        # Initialize Low Level Communication
        self.pub = ChannelPublisher("rt/lowcmd", unitree_go_msg_dds__LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", unitree_go_msg_dds__LowState_)
        self.sub.Init()
        
        # Initialize LowCmd Message
        self.cmd_msg = unitree_go_msg_dds__LowCmd__init()
        self.cmd_msg.head[0] = 0xFE
        self.cmd_msg.head[1] = 0xEF
        self.cmd_msg.level_flag = 0xFF
        self.cmd_msg.gpio = 0
        for i in range(20):
            self.cmd_msg.motor_cmd[i].mode = 0x01 # Servo Mode
            self.cmd_msg.motor_cmd[i].q = 0.0
            self.cmd_msg.motor_cmd[i].dq = 0.0
            self.cmd_msg.motor_cmd[i].kp = 0.0
            self.cmd_msg.motor_cmd[i].kd = 0.0
            self.cmd_msg.motor_cmd[i].tau = 0.0

        # Load Policy
        self.get_logger().info(f"Loading policy from {self.policy_path}...")
        try:
            self.policy_session = ort.InferenceSession(self.policy_path)
            self.input_name = self.policy_session.get_inputs()[0].name
            self.input_shape = self.policy_session.get_inputs()[0].shape
            self.expected_obs_dim = self.input_shape[1]
            self.get_logger().info(f"Policy loaded. Expected obs dim: {self.expected_obs_dim}")
        except Exception as e:
            self.get_logger().error(f"Error loading policy: {e}")
            sys.exit(1)

        # State variables
        self.last_actions = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32) # vx, vy, wz
        self.last_cmd_time = time.time()

        # ROS 2 Subscription
        self.create_subscription(
            Twist, 
            'cmd_vel', 
            self.cmd_vel_callback, 
            10
        )
        
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def cmd_vel_callback(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z
        self.target_vel = np.array([vx, vy, wz], dtype=np.float32)
        self.last_cmd_time = time.time()

    def get_gravity_vector(self, quaternion):
        r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        return r.apply([0, 0, -1], inverse=True).astype(np.float32)

    def estimate_base_height(self, q):
        L1 = 0.213
        L2 = 0.213
        heights = []
        for i in range(4):
            idx = i * 3
            theta1 = q[idx]
            theta2 = q[idx+1]
            theta3 = q[idx+2]
            y_t = 0.0955 if (i % 2 == 0) else -0.0955
            h = L1 * np.cos(theta1) * np.cos(theta2) + L2 * np.cos(theta1) * np.cos(theta2 + theta3) - y_t * np.sin(theta1)
            heights.append(h)
        return np.mean(heights)

    def get_observation(self, state):
        base_lin_vel = np.zeros(3, dtype=np.float32)
        base_ang_vel = np.array(state.imu_state.gyroscope, dtype=np.float32)
        projected_gravity = self.get_gravity_vector(state.imu_state.quaternion)
        commands = self.target_vel 
        
        q_raw = np.array([m.q for m in state.motor_state[:12]], dtype=np.float32)
        dq_raw = np.array([m.dq for m in state.motor_state[:12]], dtype=np.float32)
        
        q_isaac = q_raw[UNITREE_TO_ISAAC_IDX]
        dq_isaac = dq_raw[UNITREE_TO_ISAAC_IDX]
        
        joint_pos_rel = q_isaac - DEFAULT_JOINT_POS
        last_actions = self.last_actions
        
        est_height = self.estimate_base_height(q_isaac)
        height_scan = np.full(HEIGHT_SCAN_SIZE, -est_height, dtype=np.float32)
        
        obs = np.concatenate([
            base_lin_vel, base_ang_vel, projected_gravity, commands,
            joint_pos_rel, dq_isaac, last_actions, height_scan
        ])
        
        if obs.shape[0] != self.expected_obs_dim:
            if obs.shape[0] < self.expected_obs_dim:
                obs = np.concatenate([obs, np.zeros(self.expected_obs_dim - obs.shape[0], dtype=np.float32)])
            else:
                obs = obs[:self.expected_obs_dim]
                
        return obs.astype(np.float32)

    def control_loop(self):
        self.get_logger().info("Waiting for LowState to initialize Soft Start...")
        while self.running and self.sub.Read() is None:
            time.sleep(0.1)
        
        # 1. 초기 상태 읽기 (Soft Start 시작점)
        first_state = self.sub.Read()
        current_joints = np.array([m.q for m in first_state.motor_state[:12]], dtype=np.float32)
        
        self.get_logger().info(f"Step 4: Starting Soft Start ({SOFT_START_DURATION}s)...")
        
        # 2. Soft Start 루프 (Step 4)
        steps = int(SOFT_START_DURATION / DT)
        for t_idx in range(steps):
            if not self.running: break
            
            t = t_idx / steps # 0.0 to 1.0
            
            # 목표: 현재 자세(엎드림) -> Default 자세(서있음)로 보간
            target_q = (1 - t) * current_joints + t * self.default_joint_pos_mapped
            
            for i in range(12):
                self.cmd_msg.motor_cmd[i].q = target_q[i]
                self.cmd_msg.motor_cmd[i].dq = 0.0
                self.cmd_msg.motor_cmd[i].kp = KP * t  # Kp를 0에서 목표값까지 서서히 증가
                self.cmd_msg.motor_cmd[i].kd = KD
                self.cmd_msg.motor_cmd[i].tau = 0.0
            
            self.cmd_msg.crc = CRC().Crc(self.cmd_msg)
            self.pub.Write(self.cmd_msg)
            time.sleep(DT)

        # 3. RL 루프 진입
        self.get_logger().info("Soft start complete. Switching to RL policy.")
            
        try:
            while self.running:
                start_time = time.time()
                
                # Check for command timeout
                if time.time() - self.last_cmd_time > CMD_TIMEOUT:
                    self.target_vel[:] = 0.0
                
                state = self.sub.Read()
                if state is not None:
                    # If target velocity is zero, enter Stand Mode (bypass RL)
                    if np.linalg.norm(self.target_vel) < 0.01:
                        action_raw = np.zeros(NUM_ACTIONS, dtype=np.float32)
                        self.last_actions = action_raw
                    else:
                        obs = self.get_observation(state)
                        ort_inputs = {self.input_name: obs.reshape(1, -1)}
                        action_raw = self.policy_session.run(None, ort_inputs)[0][0]
                        self.last_actions = action_raw
                    
                    q_target_isaac = DEFAULT_JOINT_POS + (action_raw * ACTION_SCALE)
                    q_target_unitree = q_target_isaac[ISAAC_TO_UNITREE_IDX]
                    
                    for i in range(12):
                        self.cmd_msg.motor_cmd[i].q = q_target_unitree[i]
                        self.cmd_msg.motor_cmd[i].dq = 0.0
                        self.cmd_msg.motor_cmd[i].kp = KP
                        self.cmd_msg.motor_cmd[i].kd = KD
                        self.cmd_msg.motor_cmd[i].tau = 0.0
                    
                    self.cmd_msg.crc = CRC().Crc(self.cmd_msg)
                    self.pub.Write(self.cmd_msg)
                
                elapsed = time.time() - start_time
                if elapsed < DT:
                    time.sleep(DT - elapsed)
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")

    def destroy_node(self):
        self.running = False
        if self.control_thread.is_alive():
            self.control_thread.join()
        
        self.get_logger().info("Shutting down... Sending StandDown command.")
        try:
            sport_client = SportClient()
            sport_client.SetTimeout(5.0)
            sport_client.Init()
            sport_client.StandDown()
            time.sleep(1.0)
        except Exception as e:
            self.get_logger().warn(f"Failed to send StandDown on exit: {e}")
            
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
