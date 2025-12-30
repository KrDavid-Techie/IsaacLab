# RMA 모델 배포 가이드 (Deployment Guide)

이 문서는 `export_adapt.py`를 통해 추출된 ONNX 모델(`rma_policy.onnx`)을 실제 로봇이나 다른 환경에 배포할 때 필요한 입력(Input)과 출력(Output) 명세를 설명합니다.

## 1. 모델 개요 (Model Overview)

추출된 ONNX 모델은 **적응 모듈(Adaptation Module)**과 **교사 정책(Teacher Policy)**이 결합된 형태입니다.
사용자는 별도의 적응 과정을 구현할 필요 없이, **현재 로봇의 상태(Base Obs)**와 **과거 움직임 기록(Proprio History)**만 입력하면 됩니다.

- **보행 모델 파일명**: `teacher_policy.onnx`
- **적응 모델 파일명**: `rma_policy.onnx`
- **입력 정규화(Normalization)**: 모델 내부에 포함되어 있습니다. (Raw 데이터를 입력하면 됩니다.)

## 2. 입출력 인터페이스 (I/O Interface)

### 입력 (Inputs)

| 이름 (Name) | 타입 (Type) | 차원 (Shape) | 설명 (Description) |
| :--- | :--- | :--- | :--- |
| **`base_obs`** | Float32 | `(Batch, Base_Dim)` | 현재 시점의 로봇 관측 데이터 (Extrinsics 제외) |
| **`proprio_hist`** | Float32 | `(Batch, Proprio_Dim)` | 최근 $T$ 스텝 동안의 고유 수용 감각 기록 |

* `Batch`: 1 (단일 로봇 제어 시)
* `Base_Dim`: 학습 환경의 `policy` 그룹 차원에서 `privileged` 차원을 뺀 값 (예: 48)
* `Proprio_Dim`: $T \times D_{proprio}$ (예: 50 steps $\times$ 36 dims)

### 출력 (Outputs)

| 이름 (Name) | 타입 (Type) | 차원 (Shape) | 설명 (Description) |
| :--- | :--- | :--- | :--- |
| **`actions`** | Float32 | `(Batch, 12)` | 12개 관절에 대한 모터 명령 (Scaled Action) |

---

## 3. 데이터 구성 상세 (Data Composition)

### 3.1. `base_obs` 구성
`base_obs`는 학습 시 사용된 `policy` 관측 그룹에서 **특권 정보(Privileged Info, 예: 마찰계수, 질량)**를 제외한 앞부분입니다.
일반적으로 다음과 같은 순서로 구성됩니다 (정확한 순서는 `velocity_env_cfg.py`의 `policy` 그룹 정의를 따름):

1.  **Base Linear Velocity** (3): 로봇 본체의 선형 속도 ($v_x, v_y, v_z$) - *로봇 좌표계*
2.  **Base Angular Velocity** (3): 로봇 본체의 각속도 ($\omega_x, \omega_y, \omega_z$) - *로봇 좌표계*
3.  **Projected Gravity** (3): 중력 벡터의 로봇 좌표계 투영 ($g_x, g_y, g_z$)
4.  **Velocity Commands** (3): 사용자의 조종 명령 ($v^{cmd}_x, v^{cmd}_y, \omega^{cmd}_z$)
5.  **Joint Positions** (12): 관절 위치 ($q - q_{default}$)
6.  **Joint Velocities** (12): 관절 속도 ($\dot{q}$)
7.  **Previous Actions** (12): 직전 스텝의 행동 ($a_{t-1}$)
8.  **(Optional) Height Scan**: 지형 높이 정보가 포함될 수 있음.

**주의:** `base_obs`에는 **절대** 마찰계수나 로봇 질량 같은 물리 파라미터가 포함되어서는 안 됩니다.

### 3.2. `proprio_hist` 구성
적응 모듈이 환경을 추론하기 위해 사용하는 과거 데이터입니다.
보통 최근 $T$ 스텝(예: 50 steps, 1초 분량) 동안의 데이터를 평탄화(Flatten)하여 입력합니다.

각 스텝의 데이터($D_{proprio}$)는 보통 다음을 포함합니다:
1.  **Joint Positions** (12)
2.  **Joint Velocities** (12)
3.  **Previous Actions** (12)

즉, `proprio_hist` 벡터는 $[obs_{t-T}, obs_{t-T+1}, ..., obs_{t-1}]$ 순서로 연결된 형태입니다.

---

## 4. 추론 루프 구현 예시 (Python)

```python
import onnxruntime as ort
import numpy as np

# 1. 모델 로드
ort_session = ort.InferenceSession("rma_policy.onnx")

# 2. 버퍼 초기화
history_len = 50
proprio_dim_per_step = 36  # 12 pos + 12 vel + 12 action
proprio_buffer = np.zeros((history_len, proprio_dim_per_step), dtype=np.float32)

while True:
    # --- 센서 데이터 수집 ---
    # base_lin_vel, base_ang_vel, gravity, commands, joint_pos, joint_vel, prev_actions 가져오기
    
    # --- 1. Base Obs 구성 ---
    # (순서는 학습 설정과 정확히 일치해야 함)
    current_base_obs = np.concatenate([
        base_lin_vel, base_ang_vel, gravity, commands, 
        joint_pos - default_joint_pos, joint_vel, prev_actions
    ]).astype(np.float32)
    
    # --- 2. Proprio History 업데이트 ---
    current_proprio = np.concatenate([
        joint_pos - default_joint_pos, joint_vel, prev_actions
    ]).astype(np.float32)
    
    # 버퍼 시프트 (오래된 데이터 삭제, 새 데이터 추가)
    proprio_buffer[:-1] = proprio_buffer[1:]
    proprio_buffer[-1] = current_proprio
    
    # Flatten
    proprio_hist_input = proprio_buffer.flatten()
    
    # --- 3. 추론 (Inference) ---
    inputs = {
        "base_obs": current_base_obs[None, :],      # (1, Base_Dim)
        "proprio_hist": proprio_hist_input[None, :] # (1, Proprio_Dim)
    }
    
    actions = ort_session.run(None, inputs)[0] # (1, 12)
    
    # --- 4. 로봇 제어 ---
    # actions[0]를 모터 드라이버로 전송 (보통 P gain * (action * scale + default - current) + D gain * ...)
    # IsaacLab/RSL-RL은 보통 PD 제어의 target position offset으로 action을 사용함
    # target_pos = action * action_scale + default_pos
```

---

## 5. 참고 사항

- **Action Scale**: 모델이 출력하는 `actions`는 보통 정규화되거나 스케일링된 값입니다. 실제 로봇에 적용할 때는 학습 시 사용한 `action_scale`(예: 0.25)을 곱하고 `default_joint_pos`를 더해야 할 수 있습니다.
- **Loop Rate**: 학습 환경의 `dt`(제어 주기)와 실제 로봇의 제어 주기를 맞춰야 합니다. (예: 50Hz = 0.02s)
