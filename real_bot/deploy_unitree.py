import time
import sys
import math
import numpy as np
import onnxruntime as ort

# NOTE: You need to install unitree_sdk2 python bindings
# pip install unitree-sdk2py (or similar, check Unitree documentation)
try:
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState
    from unitree_sdk2py.idl.go2.constants import unitree_go_msg_dds__LowCmd_Constants as LowCmdConstants
    from unitree_sdk2py.utils.crc import crc32
except ImportError:
    print("Error: unitree_sdk2py not found. Please install the Unitree SDK 2 Python bindings.")
    print("This script is a template and requires the actual SDK to run on the robot.")
    # Mocking for demonstration purposes if SDK is missing
    class Mock: pass
    ChannelFactory = Mock()
    ChannelFactory.Instance = lambda: Mock()
    ChannelFactory.Instance().Init = lambda x, y: None
    ChannelPublisher = lambda x, y: Mock()
    ChannelSubscriber = lambda x, y: Mock()
    LowCmd = lambda: Mock()
    LowState = lambda: Mock()
    crc32 = lambda x: 0

# Configuration
POLICY_PATH = "policy.onnx"
NUM_ACTIONS = 12
NUM_OBS = 48 # Example size, check your specific config (without height scan)
DECIMATION = 4
DT = 0.005
CONTROL_DT = DT * DECIMATION # 0.02s (50Hz)

# Robot Constants (Go2)
# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
# NOTE: Check your specific robot's joint order in Isaac Lab config!
DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5   # RR
], dtype=np.float32)

ACTION_SCALE = 0.25
KP = 20.0
KD = 0.5

class RobotController:
    def __init__(self, network_interface):
        self.low_state = None
        self.low_cmd = LowCmd()
        
        # Initialize SDK
        ChannelFactory.Instance().Init(0, network_interface)
        
        self.sub = ChannelSubscriber("rt/lowstate", LowState)
        self.sub.Init(self.low_state_handler, 10)
        
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd)
        self.pub.Init()
        
        # Load Policy
        self.ort_session = ort.InferenceSession(POLICY_PATH)
        
        # State variables
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32) # Initial set of Last action for observation
        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32) # vx, vy, wz
        
        print("Controller Initialized.")

    def low_state_handler(self, msg):
        self.low_state = msg

    def get_observation(self):
        if self.low_state is None:
            return None
            
        # 1. Base Linear Velocity (Estimated or 0)
        # Real robots usually don't have this directly without an estimator.
        # For robust policies, we often set this to 0 or use a VAE.
        base_lin_vel = np.zeros(3, dtype=np.float32) 
        
        # 2. Base Angular Velocity (IMU)
        imu = self.low_state.imu_state
        base_ang_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]], dtype=np.float32)
        
        # 3. Projected Gravity
        # Convert quaternion to gravity vector
        q = imu.quaternion # w, x, y, z
        # Simple gravity projection (assuming z-up)
        # R(q)^T * [0, 0, -1]
        # ... implementation of quaternion rotation ...
        # Placeholder:
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32) 
        
        # 4. Commands
        commands = self.command
        
        # 5. Joint Positions 
        joint_pos_raw = np.array([m.q for m in self.low_state.motor_state[:12]], dtype=np.float32)
        joint_pos_rel = joint_pos_raw - DEFAULT_JOINT_POS # Relative
        
        # 6. Joint Velocities
        joint_vel = np.array([m.dq for m in self.low_state.motor_state[:12]], dtype=np.float32)
        
        # 7. Last Actions
        last_actions = self.last_action
        
        # Concatenate
        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            commands,
            joint_pos_rel,
            joint_vel,
            last_actions
        ])
        
        # Add height scan if policy expects it (zeros for blind)
        # obs = np.concatenate([obs, np.zeros(187)])
        
        return obs.astype(np.float32)

    def run(self):
        print("Starting control loop...")
        while True:
            start_time = time.time()
            
            obs = self.get_observation()
            if obs is not None:
                # Inference
                # ONNX Runtime expects list of inputs
                ort_inputs = {self.ort_session.get_inputs()[0].name: obs.reshape(1, -1)}
                actions = self.ort_session.run(None, ort_inputs)[0][0]
                
                self.last_action = actions
                
                # Process Actions
                target_pos = DEFAULT_JOINT_POS + actions * ACTION_SCALE
                
                # Send Command
                self.send_command(target_pos)
            
            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def send_command(self, target_pos):
        # Fill LowCmd
        # Header, etc.
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01 # Servo mode
            self.low_cmd.motor_cmd[i].q = target_pos[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = KP
            self.low_cmd.motor_cmd[i].kd = KD
            self.low_cmd.motor_cmd[i].tau = 0.0
            
        # CRC (if needed by python bindings, usually handled or helper provided)
        # self.low_cmd.crc = crc32(self.low_cmd) 
        
        self.pub.Write(self.low_cmd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deploy_unitree.py <network_interface>")
        sys.exit(1)
        
    controller = RobotController(sys.argv[1])
    controller.run()
