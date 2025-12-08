import numpy as np
import onnxruntime as ort
import os

# --- Configuration ---
# 테스트할 모델 경로 (필요시 수정하세요)
MODEL_PATH = "C:\\Users\\User\\github\\IsaacLab\\real_bot\\sota-stair-2025-12-03_18-06-16\\exported\\policy.onnx"

# deploy_unitree.py와 동일한 설정
ACTION_SCALE = 0.25
HEIGHT_SCAN_SIZE = 187
DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5   # RR
], dtype=np.float32)

def estimate_base_height(q):
    """
    deploy_unitree.py에서 가져온 로봇 높이 추정 함수
    """
    L1 = 0.213 # Thigh length
    L2 = 0.213 # Calf length
    
    heights = []
    for i in range(4):
        idx = i * 3
        theta1 = q[idx]     # Hip Roll
        theta2 = q[idx+1]   # Thigh Pitch
        theta3 = q[idx+2]   # Calf Pitch
        
        y_t = 0.0955 if (i % 2 == 0) else -0.0955
        
        h = L1 * np.cos(theta1) * np.cos(theta2) + L2 * np.cos(theta1) * np.cos(theta2 + theta3) - y_t * np.sin(theta1)
        heights.append(h)
            
    return np.mean(heights)

def create_mock_observation(command, expected_dim, last_actions=None, current_joint_pos=None, current_joint_vel=None):
    """
    가상의 관측값(Observation) 생성
    command: [vx, vy, wz]
    last_actions: 이전 스텝의 행동
    current_joint_pos: 현재 관절 위치 (Isaac Order)
    current_joint_vel: 현재 관절 속도 (Isaac Order)
    """
    # 1. Base Linear Velocity (Estimated) - 보통 0으로 가정
    base_lin_vel = np.zeros(3, dtype=np.float32)
    
    # 2. Base Angular Velocity (Gyro) - 정지 상태 가정
    base_ang_vel = np.zeros(3, dtype=np.float32)
    
    # 3. Projected Gravity - 똑바로 서 있는 상태 가정 ([0, 0, -1])
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    
    # 4. Commands (User Input)
    commands = np.array(command, dtype=np.float32)
    
    # 5. Joint Positions (Relative)
    if current_joint_pos is None:
        current_joint_pos = DEFAULT_JOINT_POS
    joint_pos_rel = current_joint_pos - DEFAULT_JOINT_POS
    
    # 6. Joint Velocities
    if current_joint_vel is None:
        dq_isaac = np.zeros(12, dtype=np.float32)
    else:
        dq_isaac = current_joint_vel
    
    # 7. Last Actions
    if last_actions is None:
        last_actions = np.zeros(12, dtype=np.float32)
    
    # 8. Height Scan - 평지 가정
    # 현재 관절 위치를 기반으로 높이 추정
    est_height = estimate_base_height(current_joint_pos)
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
    
    # Padding if needed (deploy_unitree.py logic)
    if obs.shape[0] < expected_dim:
        obs = np.concatenate([obs, np.zeros(expected_dim - obs.shape[0], dtype=np.float32)])
    else:
        obs = obs[:expected_dim]
        
    return obs.astype(np.float32)

def simulate_walking(vx, vy, wz, duration=10.0, log_interval=0.5):
    print(f"\n--- Simulating Walking: vx={vx}, vy={vy}, wz={wz} for {duration}s ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    try:
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        expected_obs_dim = input_shape[1]
        
        dt = 0.02  # 50Hz
        num_steps = int(duration / dt)
        log_steps = int(log_interval / dt)
        
        # 초기 상태
        last_actions = np.zeros(12, dtype=np.float32)
        current_joint_pos = DEFAULT_JOINT_POS.copy()
        current_joint_vel = np.zeros(12, dtype=np.float32)
        
        print(f"{'Time(s)':<10} | {'Target Joints (FL, FR, RL, RR)':<85} | {'Action Max':<10}")
        print("-" * 110)

        for step in range(num_steps + 1):
            # 1. Observation 생성 (현재 관절 상태 반영)
            obs = create_mock_observation([vx, vy, wz], expected_obs_dim, last_actions, current_joint_pos, current_joint_vel)
            
            # 2. Inference
            ort_inputs = {input_name: obs.reshape(1, -1)}
            action_raw = session.run(None, ort_inputs)[0][0]
            
            # 3. Update State (Physics Simulation Mock)
            # 모델이 출력한 Action을 목표 위치로 변환
            target_joint_pos = DEFAULT_JOINT_POS + (action_raw * ACTION_SCALE)
            
            # 가상의 모터: 목표 위치로 즉시 이동한다고 가정 (Perfect Tracking)
            # 실제로는 지연이 있지만, 여기서는 단순화를 위해 바로 반영
            next_joint_pos = target_joint_pos
            
            # 속도 계산: (다음 위치 - 현재 위치) / dt
            next_joint_vel = (next_joint_pos - current_joint_pos) / dt
            
            # 상태 업데이트
            current_joint_pos = next_joint_pos
            current_joint_vel = next_joint_vel
            last_actions = action_raw
            
            # 4. Log (log_interval 마다)
            if step % log_steps == 0:
                time_sec = step * dt
                
                # 포맷팅: 소수점 3자리
                joints_str = np.array2string(target_joint_pos, formatter={'float_kind':lambda x: "%.3f" % x}, separator=', ', max_line_width=200)
                # 대괄호 제거
                joints_str = joints_str.replace('[', '').replace(']', '')
                
                print(f"{time_sec:<10.1f} | {joints_str:<85} | {np.max(np.abs(action_raw)):.4f}")

    except Exception as e:
        print(f"Error running model: {e}")

if __name__ == "__main__":
    # Test Case 2: 앞으로 걷기 (0.5 m/s)
    simulate_walking(0.5, 0.0, 0.0, duration=10.0, log_interval=0.5)
