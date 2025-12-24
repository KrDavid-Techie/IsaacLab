# Isaac Lab Evaluation Pipeline / Isaac Lab 평가 파이프라인

This directory contains a suite of tools for evaluating RL policies trained in Isaac Lab and quantifying the Sim-to-Real gap using real-world data from Unitree Go2 robots.

이 디렉토리는 Isaac Lab에서 학습된 RL 정책을 평가하고, Unitree Go2 로봇의 실제 주행 데이터와 비교하여 Sim-to-Real 격차를 정량화하기 위한 도구 모음입니다.

---

## 🚀 Workflow / 워크플로우

The evaluation process consists of three main steps:
평가 프로세스는 크게 세 단계로 구성됩니다:

1.  **Real-world Data Collection (실제 데이터 수집)**:
    *   Run the policy on the real robot.
    *   Use `go2_logger` to record sensor data into ROS2 MCAP files.
    *   실제 로봇에서 정책을 실행합니다.
    *   `go2_logger`를 사용하여 센서 데이터를 ROS2 MCAP 파일로 기록합니다.

2.  **Simulation Data Collection (시뮬레이션 데이터 수집)**:
    *   Run `sim_eval.py` to execute the policy in Isaac Sim.
    *   This generates a `.pkl` file containing time-series data of the simulation.
    *   `sim_eval.py`를 실행하여 Isaac Sim에서 정책을 실행합니다.
    *   이 과정에서 시뮬레이션의 시계열 데이터가 담긴 `.pkl` 파일이 생성됩니다.

3.  **Sim-to-Real Comparison (Sim-to-Real 비교 분석)**:
    *   Run `sim2real_eval.py` to compare the Real MCAP and Sim PKL files.
    *   Generates a CSV report with key performance metrics.
    *   `sim2real_eval.py`를 실행하여 실제 MCAP 파일과 시뮬레이션 PKL 파일을 비교합니다.
    *   주요 성능 지표가 포함된 CSV 리포트를 생성합니다.

---

## 📂 Components / 구성 요소

### 1. Real-world Logger (`go2_logger`)
*   **Path**: `scripts/evaluation/go2_logger/`
*   **Description**: A ROS2 node that subscribes to Unitree Go2 topics (`/sport/modestate`, `/lowstate`) and records them.
*   **설명**: Unitree Go2의 토픽들을 구독하고 기록하는 ROS2 노드입니다.
*   **Usage / 사용법**:
    ```bash
    # On the robot or ROS2 environment
    ros2 run go2_logger logger_node
    ```
*   **Output**: `.mcap` files in `logs/` directory.

### 2. Simulation Evaluator (`sim_eval.py`)
*   **Description**: Loads a trained checkpoint, runs the simulation (headless by default), and saves detailed logs.
*   **설명**: 학습된 체크포인트를 로드하여 시뮬레이션을 실행(기본값: Headless)하고 상세 로그를 저장합니다.
*   **Usage / 사용법**:
    ```bash
    # Run evaluation for 20 seconds
    ./isaaclab.bat -p scripts/evaluation/sim_eval.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 1 --evaluation_time 20.0
    ```
*   **Key Arguments**:
    *   `--headless`: Run without GUI (Default: True).
    *   `--evaluation_time`: Duration of the run in seconds.
*   **Output**: `scripts/evaluation/result/pkl/sim_log_{timestamp}.pkl`

### 3. Sim-to-Real Comparator (`sim2real_eval.py`)
*   **Description**: Aligns timestamps between Sim and Real data, calculates error metrics, and appends results to a daily CSV report.
*   **설명**: 시뮬레이션과 실제 데이터의 타임스탬프를 정렬하고, 오차 지표를 계산하여 일일 CSV 리포트에 추가합니다.
*   **Usage / 사용법**:
    ```bash
    ./isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file path/to/sim_log.pkl --real_bag path/to/real_log.mcap --sim_id "Experiment_Name"
    ```
*   **Output**: `scripts/evaluation/result/sim2real_report_{YYYY-MM-DD}.csv`

---

## 📊 Metrics / 평가 지표

The following metrics are calculated to evaluate the policy performance and Sim-to-Real gap.
정책 성능과 Sim-to-Real 격차를 평가하기 위해 다음 지표들이 계산됩니다.

| Metric (지표) | Description (설명) | Ideal/GT (목표값) |
| :--- | :--- | :--- |
| **Velocity Tracking Error (RMSE)** | Root Mean Square Error between command velocity and actual velocity. <br> 명령 속도와 실제 속도 간의 평균 제곱근 오차. | **Low (낮을수록 좋음)** |
| **Torque Reality Gap (RMSE)** | Difference between simulated torque and real actuator torque for the same motion. <br> 동일 동작 수행 시 시뮬레이션 토크와 실제 액추에이터 토크 간의 차이. | **Low (낮을수록 좋음)** |
| **Cost of Transport (CoT)** | Dimensionless measure of energy efficiency: $P / (mgv)$. <br> 에너지 효율성을 나타내는 무차원 지표. | **~0.4** (Unitree Go2 GT) |
| **Torque Smoothness (Jitter)** | Mean absolute derivative of torque over time. Indicates control stability. <br> 시간당 토크 변화량의 평균. 제어 안정성을 나타냄. | **Low (낮을수록 좋음)** |

---

## 📁 Directory Structure / 디렉토리 구조

```
scripts/evaluation/
├── go2_logger/          # ROS2 Package for real robot logging
├── result/              # Output directory
│   ├── pkl/             # Simulation raw data (.pkl)
│   └── sim2real_report_*.csv  # Daily evaluation reports
├── sim_eval.py          # Simulation inference script
├── sim2real_eval.py     # Comparison & Analysis script
└── README.md            # This file
```
