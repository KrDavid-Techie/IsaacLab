# IsaacLab RMA (Rapid Motor Adaptation) 구현

이 디렉토리는 Unitree Go2 로봇을 위한 Rapid Motor Adaptation (RMA) 구현 스크립트를 포함하고 있습니다. RMA는 로봇이 고유 수용 감각(proprioception)의 기록(history)을 통해 마찰이나 적재 중량과 같은 물리적 매개변수를 추정함으로써, 다양한 환경 변화에 적응할 수 있도록 합니다.

## 아키텍처 (Architecture)

구현은 2단계의 학습 과정으로 이루어집니다:

### 1단계: 교사 정책 학습 (Phase 1: Teacher Policy Training)
시뮬레이션에서만 접근 가능한 **특권 정보(Privileged Information)**(실제 물리 파라미터 값 등)를 사용하여 교사 정책을 학습합니다.

- **태스크(Task):** `Isaac-Velocity-Rough-Unitree-Go2-RMA-v0`
- **관측(Observations):**
  - `policy`: 고유 수용 감각 + 특권 정보 (Extrinsics)
  - `privileged`: 명시적인 특권 파라미터 (예: 차체 질량, 마찰 계수)
  - `proprio`: 고유 수용 감각의 과거 기록 (관절 위치, 속도, 행동)
- **스크립트:** 표준 `train.py` (RSL-RL 사용)

### 2단계: 적응 모듈 학습 (Phase 2: Adaptation Module Training)
**고유 수용 감각 기록(Proprioception History)**만을 입력으로 받아 **특권 정보(Extrinsics)**를 추정하도록 적응 모듈을 학습합니다. (지도 학습)

- **입력(Input):** 고유 수용 감각 기록 (`proprio` 그룹)
- **목표(Target):** 특권 정보 (`privileged` 그룹)
- **교사(Teacher):** 1단계에서 학습된 후 동결(Frozen)된 정책
- **스크립트:** `train_adapt.py`

### 3단계: 배포 및 실행 (Phase 3: Deployment / Student Policy)
동결된 교사 정책과 학습된 적응 모듈을 결합하여 학생 정책(Student Policy)을 구성합니다.

- **흐름(Flow):** `Proprio History` -> **적응 모듈** -> `예측된 Extrinsics` + `기본 관측` -> **교사 정책** -> `행동(Action)`
- **스크립트:** `play_adapt.py`

## 사용 방법 (Usage)

### 1. 교사 정책 학습 (Phase 1)
특권 정보를 사용하여 기본 정책을 학습합니다.

```bash
# 저장소 루트에서 실행
isaaclab.bat -p source/isaaclab/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --headless
```

### 2. 적응 모듈 학습 (Phase 2)
Extrinsics를 예측하기 위해 적응 모듈을 학습합니다.

```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/train_adapt.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --load_run unitree_go2_rough_rma_phase1 ^
    --load_checkpoint model_100.pt ^
    --headless
```
*참고: 결과는 `logs/rma/<실험명>_phase2`에 저장됩니다.*

### 3. 실행 및 평가 (Student Policy)
결합된 정책을 시뮬레이터에서 실행합니다.

```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/play_adapt.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --num_envs 32 ^
    --run_phase2 unitree_go2_rough_rma_phase2
```
*참고: 스크립트는 Phase 2에서 생성된 `teacher_policy_info.txt`를 사용하여 해당하는 교사 정책을 자동으로 찾습니다.*

### 4. ONNX 내보내기 (Export to ONNX)
배포를 위해 결합된 학생 정책(적응 모듈 + 교사 정책)을 ONNX 형식으로 내보냅니다.

```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/export_adapt.py ^
    --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 ^
    --run_phase2 unitree_go2_rough_rma_phase2
```
*참고: 내보낸 모델은 `logs/rma/<실험명>_phase2/exported_models/rma_policy.onnx`에 저장됩니다.*
*입력: `base_obs`, `proprio_hist` -> 출력: `actions`*

### 5. RMA 추론 프로세스 (Inference Loop)

시뮬레이션의 매 스텝(`while simulation_app.is_running()`)마다 다음 과정이 반복됩니다.

#### 1. 관측 데이터 수집 (Get Inputs)
환경으로부터 현재 상태 정보를 가져옵니다.
*   **`proprio_hist` (`obs["proprio"]`)**: 로봇의 관절 위치, 속도, 이전 행동들의 **과거 기록(History)**입니다. 이것이 적응 모듈(Adaptation Module)의 입력이 됩니다.
*   **`policy_obs_gt` (`obs["policy"]`)**: 현재 시뮬레이터가 알고 있는 **참값(Ground Truth)** 정보입니다. 여기에는 실제 물리 파라미터(Extrinsics)가 포함되어 있지만, **학생 정책(Student Policy)은 이를 직접 보지 않고 예측값을 사용해야 하므로 분리 작업이 필요합니다.**

#### 2. 환경 정보 예측 (Predict Extrinsics)
*   **적응 모듈(Adaptation Module)**이 `proprio_hist`를 입력받아 **`pred_extrinsics`**를 출력합니다.
*   즉, "로봇이 지난 0.5초 동안 이렇게 움직였으니, 지금 바닥의 마찰력은 이 정도일 것이다"라고 추측하는 과정입니다.

#### 3. 학생 관측 데이터 구성 (Construct Student Observation)
교사 정책(Teacher Policy)에게 전달할 최종 관측 데이터를 만듭니다.
*   `policy_obs_gt`에서 참값 Extrinsics(뒷부분)를 잘라내고, 순수 로봇 상태 정보(`base_obs`)만 남깁니다.
*   여기에 방금 예측한 `pred_extrinsics`를 붙여서 **`student_obs_raw`**를 만듭니다.
    *   `Student Obs` = `Base Obs` + `Predicted Extrinsics`

#### 4. 정규화 (Normalization)
*   교사 정책이 학습될 때 데이터의 평균과 분산을 맞춰주는 정규화(Normalization)를 사용했다면, 학생 데이터(`student_obs_raw`)에도 똑같은 기준(`runner.obs_normalizer`)을 적용하여 **`student_obs`**로 변환합니다.

#### 5. 행동 결정 (Policy Inference)
*   최종적으로 만들어진 `student_obs`를 **교사 정책(Teacher Policy)**에 입력합니다.
*   교사 정책은 이 정보를 바탕으로 로봇이 취해야 할 **`actions`**(관절 모터 명령)을 출력합니다.

#### 6. 환경 적용 (Step Environment)
*   계산된 `actions`를 시뮬레이터(`env.step(actions)`)에 전달하여 로봇을 실제로 움직입니다.

---

### 요약 다이어그램

```mermaid
graph TD
    A[환경 (Environment)] -->|Proprio History| B(적응 모듈 Adaptation Module)
    A -->|Base Obs| C{데이터 결합}
    B -->|예측된 Extrinsics| C
    C -->|Student Obs| D[정규화 (Normalization)]
    D -->|Normalized Obs| E(교사 정책 Teacher Policy)
    E -->|Action| A
```

이 과정을 통해 로봇은 실제 환경 정보를 직접 알지 못해도, 자신의 움직임 이력을 통해 환경을 추정하고 적응하여 걸을 수 있게 됩니다.


## 파일 구조 (File Structure)

- `train_adapt.py`: 적응 모듈 학습을 위한 스크립트 (지도 학습).
- `play_adapt.py`: 전체 RMA 파이프라인 추론(Inference)을 위한 스크립트.
- `export_adapt.py`: 결합된 정책을 ONNX로 내보내는 스크립트.
- `velocity_env_cfg.py` (`isaaclab_tasks` 내부): `LocomotionVelocityRoughEnvCfg_RMA` 클래스와 관측 그룹을 정의.

## 설정 상세 (Configuration Details)

환경 설정 파일(`velocity_env_cfg.py`)은 세 가지 주요 관측 그룹을 정의합니다:
1. **`policy`**: RL 에이전트의 입력. `base_mass`와 `friction` (Extrinsics)을 포함합니다.
2. **`privileged`**: 적응 모듈의 예측 목표(Target). `policy`의 Extrinsics와 동일합니다.
3. **`proprio`**: 적응 모듈의 입력. 관절 위치, 속도, 이전 행동의 기록(History)을 포함합니다.


