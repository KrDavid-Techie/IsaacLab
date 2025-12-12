# Unitree Go2를 위한 실제 로봇 배포 가이드

이 문서는 Isaac Lab에서 강화학습으로 훈련된 정책(Policy)을 실제 Unitree Go2 로봇에 배포하는 전체 과정을 단계별로 상세히 설명합니다.

## 전체 프로세스 개요

1.  **시뮬레이션 훈련**: Isaac Lab에서 로봇 정책 훈련.
2.  **모델 내보내기 (Export)**: 훈련된 모델을 ONNX 형식으로 변환.
3.  **환경 설정**: 실제 로봇(또는 제어 PC)에 필요한 라이브러리 설치.
4.  **배포 및 실행**: 변환된 모델을 로봇에 로드하고 제어 루프 실행.

---

## 1단계: 모델 훈련 및 내보내기 (Training & Export)

Isaac Lab에서 훈련이 완료되면, `play.py` 스크립트를 사용하여 훈련된 정책을 확인하고 ONNX 파일로 추출합니다.

1.  **훈련된 모델 확인**:
    `logs/rsl_rl/` 디렉토리 아래에 훈련된 실험 폴더가 있는지 확인합니다.

2.  **ONNX 추출 실행**:
    다음 명령어를 실행하면 시뮬레이션이 재생되면서 동시에 `policy.onnx` 파일이 생성됩니다.
    ```bash
    # 예시: Isaac-Velocity-Rough-Unitree-Go2-v0 태스크의 최신 런을 로드
    ./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 1
    ```

3.  **파일 확인**:
    `logs/rsl_rl/<실험명>/exported/policy.onnx` 경로에 파일이 생성되었는지 확인합니다.
    *   이 ONNX 모델은 입력 정규화(Normalization) 레이어를 포함하고 있어, 로봇의 센서 데이터를 별도 가공 없이(스케일링 없이) 바로 입력으로 사용할 수 있습니다.

## 2단계: 배포 방법 (Deployment Methods)

### 옵션 A: ROS 2 패키지 (권장)

`go2_rl_deploy` 디렉토리는 배포 로직을 ROS 2 노드로 래핑한 패키지를 포함합니다.

#### 주요 기능
*   **안전 기능 (Safety)**: 시작 및 종료 시 `StandDown` 명령을 전송하여 로봇을 안전하게 눕힙니다.
*   **소프트 스타트 (Soft Start)**: 현재 누워있는 자세에서 3초에 걸쳐 서서히 일어서는 자세로 보간합니다.
*   **ROS 2 통합**: `/cmd_vel` 토픽을 구독하여 제어 명령을 받습니다.
*   **자동 대기 (Auto-Stand)**: 명령이 없거나 0인 경우, 정책 추론을 건너뛰고 제자리 서기 자세를 유지하여 안정성을 높입니다.

#### 필수 조건
*   ROS 2 (Humble 이상 권장)
*   `unitree_sdk2py` 설치됨
*   `onnxruntime`, `numpy`, `scipy`

#### 빌드 및 실행

1.  **패키지 빌드**:
    ```bash
    cd real_bot
    colcon build --packages-select go2_rl_deploy
    source install/setup.bash
    ```

2.  **노드 실행**:
    ```bash
    ros2 run go2_rl_deploy deploy_node --ros-args -p policy_path:=/path/to/policy.onnx -p network_interface:=enp2s0
    ```
    *   `policy_path`: 내보낸 ONNX 파일의 절대 경로.
    *   `network_interface`: 로봇과 연결된 네트워크 인터페이스 (예: `eth0`, `enp2s0`). 지정하지 않으면 `CYCLONEDDS_URI`에서 자동 감지를 시도합니다.

3.  **제어**:
    `/cmd_vel` 토픽에 `geometry_msgs/Twist` 메시지를 발행하여 로봇을 움직입니다.
    ```bash
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
    ```

### 옵션 B: 독립형 스크립트 (Standalone Script)

ROS 2 패키지를 빌드하고 싶지 않다면 독립형 스크립트 `deploy_unitree.py`를 실행할 수도 있습니다.

```bash
python3 deploy_unitree.py
```
*   `unitree_sdk2py`와 `onnxruntime`이 설치되어 있어야 합니다.
*   필요한 경우 스크립트 내의 `POLICY_PATH`와 `network_interface`를 수동으로 수정하세요.

## 3단계: 안전 및 문제 해결 (Safety & Troubleshooting)

*   **소프트 스타트**: 로봇이 천천히 일어서는 데 3초가 걸립니다. 이 시간 동안은 명령을 보내지 마세요.
*   **비상 정지 (E-Stop)**: 항상 리모컨의 비상 정지 버튼을 누를 준비를 하세요.
*   **공중 테스트**: 첫 실행 시에는 로봇을 매달아 발이 바닥에 닿지 않게 한 상태에서 다리 움직임을 확인하세요.
*   **인터페이스**: 로봇이 응답하지 않으면 네트워크 인터페이스 이름(`ifconfig` 또는 `ip a`)을 확인하고 로봇에 핑(ping)이 되는지 확인하세요.

*   **엉뚱한 동작**: 관절 순서(Joint Order)가 Isaac Lab 설정과 실제 로봇 SDK 설정이 일치하는지 확인하세요. (FL -> FR -> RL -> RR 순서 등)
