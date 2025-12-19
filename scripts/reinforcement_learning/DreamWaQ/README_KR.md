# Isaac Lab에서의 DreamWaQ 구현

이 저장소는 NVIDIA Isaac Lab을 사용하여 Unitree Go2 로봇을 위한 **DreamWaQ** 강화학습 파이프라인을 구현한 것입니다. 견고한 보행을 위해 Context Encoder Network (CENet)를 사용하는 정책을 학습시키며, 이를 위해 커스텀 RSL-RL 러너를 활용합니다.

## 개요 (Overview)

DreamWaQ는 관측 기록(history of observations)을 사용하여 잠재적인 환경 매개변수와 속도 명령을 추정하는 학습 기반 보행 제어기입니다. 이 구현체는 DreamWaQ 아키텍처를 Isaac Lab 프레임워크에 통합하고, PPO 알고리즘을 위해 `rsl_rl`을 활용합니다.

## 디렉토리 구조 (Directory Structure)

```
DreamWaQ/
├── agent_cfg.py           # RSL-RL 에이전트 설정 (PPO 설정)
├── env_cfg.py             # Isaac Lab 환경 설정 (관측, 보상)
├── train.py               # 학습 스크립트
├── play.py                # 추론/시각화 스크립트
└── rsl_rl_dreamwaq/       # 커스텀 RSL-RL 구현체
    ├── modules.py         # 신경망 아키텍처 (Actor, Critic, CENet)
    ├── on_policy_runner.py# 커스텀 러너 루프
    ├── ppo.py             # 커스텀 PPO 알고리즘
    └── storage.py         # 롤아웃 저장소 (Rollout Storage)
```

## 환경 상세 (Environment Details)

- **작업 이름 (Task Name)**: `Isaac-Velocity-Rough-Go2-DreamWaQ-v0`
- **로봇**: Unitree Go2
- **지형**: 거친 지형 (다양한 높이, 계단, 경사로)

### 관측 공간 (Observation Space)
정책 입력은 다음 요소들이 연결된 벡터로 구성됩니다:
1.  **고유수용감각 (Proprioception)**: 베이스 각속도, 투영된 중력 벡터, 관절 위치, 관절 속도, 이전 행동.
2.  **잠재/컨텍스트 (Latent/Context)**: 관측 기록으로부터 Context Encoder Network (CENet)가 생성한 값.

### 행동 공간 (Action Space)
- **관절 위치 목표 (Joint Position Targets)**: 12 자유도 (다리당 3개).

## 신경망 아키텍처 (`modules.py`)

`ActorCritic_DWAQ` 클래스는 다음 컴포넌트들을 구현합니다:
1.  **Context Encoder (CENet)**:
    -   입력: 관측 기록 (`obs_history`).
    -   출력: 환경 컨텍스트와 속도 추정을 나타내는 잠재 코드 (평균 및 분산).
2.  **Actor**:
    -   입력: 현재 관측값 + 잠재 코드.
    -   출력: 행동 분포의 평균.
3.  **Critic**:
    -   입력: Critic 관측값 (특권 정보, privileged information).
    -   출력: 가치(Value) 추정값.
4.  **Decoder** (보조 작업):
    -   의미 있는 표현 학습을 보장하기 위해 잠재 코드로부터 관측값을 복원합니다.

## 사용법 (Usage)

### 사전 요구사항
Isaac Lab이 설치되어 있고 환경이 구성되어 있어야 합니다.

### 학습 (Training)
정책 학습을 시작하려면 다음 명령어를 실행하세요:

```bash
# Isaac Lab 저장소 루트에서 실행
./isaaclab.bat -p scripts/reinforcement_learning/DreamWaQ/train.py --task Isaac-Velocity-Rough-Go2-DreamWaQ-v0 --headless
```

**인자 (Arguments):**
- `--headless`: GUI 없이 실행 (학습 속도 향상).
- `--video`: 학습 중 비디오 녹화.
- `--num_envs` : (기본값: 10).
- `--max_iterations`: 학습 반복 횟수 (기본값: 100).

### 추론 (Inference/Play)
학습된 정책을 시각화하려면 다음 명령어를 실행하세요:

```bash
./isaaclab.bat -p scripts/reinforcement_learning/DreamWaQ/play.py --task Isaac-Velocity-Rough-Go2-DreamWaQ-v0
```

**인자 (Arguments):**
- `--num_envs`: 생성할 로봇(환경)의 개수 (기본값: 10).
- `--use_pretrained_checkpoint`: Nucleus에 게시된 사전 학습된 체크포인트 사용 (가능한 경우).

## 설정 (Configuration)

- **환경 (Environment)**: `env_cfg.py`를 수정하여 관측 스케일, 보상 가중치, 지형 난이도 등을 변경합니다.
- **에이전트 (Agent)**: `agent_cfg.py`를 수정하여 PPO 하이퍼파라미터(학습률, 배치 크기 등)를 조정합니다.
- **네트워크 (Network)**: `rsl_rl_dreamwaq/modules.py`를 수정하여 레이어 크기나 활성화 함수를 변경합니다.

## 참고 사항 (Notes)
- 이 구현체는 Context Encoder가 요구하는 특정 데이터 흐름을 처리하기 위해 커스텀 `OnPolicyRunner`를 사용합니다.
- `rsl-rl-lib`이 설치되어 있고 호환되는 버전인지 확인하세요 (`train.py`에서 확인됨).
