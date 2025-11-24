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

## 2단계: 배포 환경 준비 (Environment Setup)

실제 로봇(Unitree Go2)의 내부 PC 또는 로봇과 연결된 외부 PC(Jetson 등)에서 다음 준비를 수행합니다.

1.  **필수 라이브러리 설치**:
    *   **Python 3.8+**
    *   **NumPy**: 수치 연산용
    *   **ONNX Runtime**: 모델 추론용
        ```bash
        pip install numpy onnxruntime
        ```
    *   **Unitree SDK 2 (Python)**: 로봇 통신용
        *   Unitree 공식 문서 또는 제공된 SDK 패키지를 통해 `unitree_sdk2py`를 설치해야 합니다.
        *   일반적으로 `pip install unitree-sdk2py` 또는 소스 빌드가 필요할 수 있습니다.

2.  **파일 복사**:
    *   생성된 `policy.onnx` 파일.
    *   이 폴더의 `deploy_unitree.py` 스크립트.
    *   위 두 파일을 로봇 제어 PC의 동일한 디렉토리에 위치시킵니다.

## 3단계: 배포 스크립트 실행 및 제어 루프 (Execution)

`deploy_unitree.py`는 다음과 같은 순서로 작동합니다.

### 3.1. 초기화
*   **통신 연결**: `ChannelFactory`를 통해 로봇의 네트워크 인터페이스(예: `eth0`, `wlan0`)와 연결합니다.
*   **모델 로드**: `onnxruntime`을 통해 `policy.onnx`를 메모리에 로드합니다.
*   **Pub/Sub 설정**:
    *   Subscriber: `rt/lowstate` (로봇의 현재 상태 수신)
    *   Publisher: `rt/lowcmd` (로봇에게 모터 명령 전송)

### 3.2. 제어 루프 (Control Loop)
스크립트는 약 50Hz(0.02초 간격)로 다음 과정을 반복합니다.

1.  **상태 수신 (Observation)**:
    *   `lowstate` 토픽에서 모터 위치(q), 속도(dq), IMU(자이로, 쿼터니언) 정보를 읽어옵니다.
    *   **입력 벡터 구성**: 훈련 때와 동일한 순서로 데이터를 나열합니다.
        1.  `Base Lin Vel`: (보통 0으로 설정하거나 추정값 사용)
        2.  `Base Ang Vel`: IMU 자이로스코프 값
        3.  `Projected Gravity`: IMU 쿼터니언으로 계산한 중력 벡터
        4.  `Commands`: 사용자가 원하는 목표 속도 (예: 전진 0.5m/s)
        5.  `Joint Pos`: (현재 각도 - 기본 각도)
        6.  `Joint Vel`: 현재 관절 속도
        7.  `Last Actions`: 직전 단계의 행동 값

2.  **모델 추론 (Inference)**:
    *   구성된 입력 벡터를 ONNX 모델에 넣고 실행합니다.
    *   결과값으로 `actions` (12차원 벡터)를 얻습니다.

3.  **명령 변환 및 전송 (Action to Command)**:
    *   모델이 출력한 `actions`는 **위치 변화량(Delta)**입니다.
    *   실제 목표 위치 계산: `Target_Pos = Default_Pos + (Action * Scale)`
        *   *Scale은 보통 0.25입니다.*
    *   계산된 목표 위치를 `LowCmd` 메시지에 담아 `rt/lowcmd` 토픽으로 전송합니다.
    *   이때 Kp(강성), Kd(감쇠) 게인 값도 함께 전송하여 PD 제어를 수행합니다.

## 4단계: 안전 주의사항 (Safety First)

실제 로봇을 구동할 때는 항상 안전에 유의해야 합니다.

1.  **매달아 놓고 테스트**: 처음 실행 시에는 로봇을 갠트리나 줄에 매달아 발이 바닥에 닿지 않게 한 상태에서 다리가 정상적으로 움직이는지 확인하세요.
2.  **비상 정지(E-Stop)**: 로봇 리모컨의 비상 정지 버튼 위치를 항상 숙지하고, 이상 동작 시 즉시 누를 준비를 하세요.
3.  **제로 명령 테스트**: 처음에는 `Commands`를 0으로 설정하여 제자리에서 균형을 잡는지 먼저 확인합니다.

## 문제 해결 (Troubleshooting)

*   **로봇이 움직이지 않음**: 네트워크 인터페이스 이름(`eth0` 등)이 맞는지 확인하고, 로봇이 `Low Level` 제어 모드인지 확인하세요.
*   **다리가 심하게 떨림**: PD 게인(Kp, Kd) 값이 너무 높거나, `ACTION_SCALE`이 훈련 설정과 다른지 확인하세요.
*   **엉뚱한 동작**: 관절 순서(Joint Order)가 Isaac Lab 설정과 실제 로봇 SDK 설정이 일치하는지 확인하세요. (FL -> FR -> RL -> RR 순서 등)
