# Isaac Lab Evaluation Pipeline / Isaac Lab 평가 파이프라인

This directory contains a suite of tools for evaluating RL policies trained in Isaac Lab and quantifying the Sim-to-Real gap using real-world data from Unitree Go2 robots.

이 디렉토리는 Isaac Lab에서 학습된 RL 정책을 평가하고, Unitree Go2 로봇의 실제 주행 데이터와 비교하여 Sim-to-Real 격차를 정량화하기 위한 도구 모음입니다.

---

## 🚀 Workflow / 워크플로우

The evaluation process consists of three main steps:
평가 프로세스는 크게 세 단계로 구성됩니다:

1.  **Real-world Data Collection (실제 데이터 수집)**:
    *   Run the policy on the real robot.
    *   Use `go2_logger` to record sensor data.
    *   **Outputs**: `.csv` (human-readable) & `.mcap` (ROS2 standard) files.
    *   실제 로봇에서 정책을 실행합니다.
    *   `go2_logger`를 사용하여 센서 데이터를 기록합니다.
    *   **출력**: `.csv` (분석용) 및 `.mcap` (ROS2 표준) 파일이 생성됩니다.

2.  **Simulation Data Collection (시뮬레이션 데이터 수집)**:
    *   Run `sim_eval.py` to execute the policy in Isaac Sim.
    *   **Outputs**: `.csv` & `.pkl` files containing time-series data of the simulation.
    *   `sim_eval.py`를 실행하여 Isaac Sim에서 정책을 실행합니다.
    *   **출력**: 시뮬레이션의 시계열 데이터가 담긴 `.csv` 및 `.pkl` 파일이 생성됩니다.

3.  **Sim-to-Real Comparison (Sim-to-Real 비교 분석)**:
    *   Run `sim2real_eval.py` to compare the Real and Sim logs.
    *   Supports both CSV and MCAP/PKL formats.
    *   Generates a CSV report with key performance metrics.
    *   `sim2real_eval.py`를 실행하여 실제 로그와 시뮬레이션 로그를 비교합니다.
    *   CSV 및 MCAP/PKL 형식을 모두 지원합니다.
    *   주요 성능 지표가 포함된 CSV 리포트를 생성합니다.

---

## 📂 Components / 구성 요소

### 0. End to End Pipeline(`eval_pipeline.py`)
*   **Path**: `scripts/evaluation/eval_pipeline.py`
*   **Description**: An orchestration script that automates the analysis workflow. It sequentially executes real-world data evaluation (real_eval), simulation data evaluation (sim_eval), and Sim-to-Real comparison (sim2real_eval) using provided log files.
*   **설명**: 실제 로봇 주행 로그와 기존 시뮬레이션 로그를 입력받아, real_eval(실환경 데이터 분석), sim_eval(시뮬레이션 데이터 분석), sim2real_eval(Sim2Real 비교) 과정을 순차적으로 자동 실행하는 통합 파이프라인입니다.
*   **Usage / 사용법**:
    ```bash
    # Using CSV files (Recommended)
    isaaclab.bat -p scripts\evaluation\eval_pipeline.py --sim_log scripts\evaluation\result\csv\sim_log_unitree_go2_rough_2025-12-26_14-04-27.csv --real_log scripts\evaluation\go2_logger\logs\real_log_2025-12-29.csv
    ```
*   **Output**:
    *   `sim2real_reports\sim2real_report_YYYY-MM-DD.csv`

### 1. Real-world Logger (`go2_logger`)
*   **Path**: `scripts/evaluation/go2_logger/`
*   **Description**: A ROS2 node that subscribes to Unitree Go2 topics (`/sport/modestate`, `/lowstate`, `/wireless_remote`) and records them.
*   **설명**: Unitree Go2의 토픽들을 구독하고 기록하는 ROS2 노드입니다. 리모트 컨트롤러 입력도 지원합니다.
*   **Usage / 사용법**:
    ```bash
    # On the robot or ROS2 environment
    ros2 run go2_logger logger_node
    ```
*   **Output**:
    *   `logs/real_log_{YYYY-MM-DD}.csv`: Daily appended log (CSV).
    *   `logs/real_log_{HHMMSS}/`: ROS2 MCAP bag file.

### 2. Real-world Evaluator (`real_eval.py`)
*   **Path**: `scripts/evaluation/real_eval.py`
*   **Description**: Analyzes real-world robot logs to evaluate performance metrics (Velocity Tracking, Stability, Energy) based on internal ground truth.
*   **설명**: 실제 로봇 주행 로그(.csv, .mcap)를 분석하여 속도 추종성, 주행 안정성, 에너지 효율성 등의 성능 지표를 평가합니다. (시뮬레이션 비교 없음)
*   **Usage / 사용법**:
    ```bash
    isaaclab.bat -p scripts/evaluation/real_eval.py --real_log scripts/evaluation/go2_logger/logs/real_log_2025-12-29.csv
    ```
*   **Output**:
<img src="docs\real_eval.png"> 
    *   Console Output: Performance Report (Velocity RMSE, Roll/Pitch Bias, CoT, Jitter).

### 3. Simulation Evaluator (`sim_eval.py`)
*   **Description**: Loads a trained checkpoint, runs the simulation (headless by default), and saves detailed logs.
*   **설명**: 학습된 체크포인트를 로드하여 시뮬레이션을 실행(기본값: Headless)하고 상세 로그를 저장합니다.
*   **Usage / 사용법**:
    ```bash
    # Run evaluation for 20 seconds
    isaaclab.bat -p scripts/evaluation/sim_eval.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 1 --evaluation_time 20.0
    ```
*   **Key Arguments**:
    *   `--headless`: Run without GUI (Default: True).
    *   `--evaluation_time`: Duration of the run in seconds.
*   **Output**:
<img src="docs\sim_eval.png"> 
    *   `scripts/evaluation/result/csv/sim_log_{timestamp}.csv`
    *   `scripts/evaluation/result/pkl/sim_log_{timestamp}.pkl`

### 4. Sim-to-Real Comparator (`sim2real_eval.py`)
*   **Description**: Aligns timestamps between Sim and Real data, calculates error metrics, and appends results to a daily CSV report.
*   **설명**: 시뮬레이션과 실제 데이터의 타임스탬프를 정렬하고, 오차 지표를 계산하여 일일 CSV 리포트에 추가합니다.
*   **Usage / 사용법**:
    ```bash
    # Using CSV files (Recommended)
    isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file scripts/evaluation/result/csv/sim_log_....csv --real_log scripts/evaluation/go2_logger/logs/real_log_....csv
    
    # Using Legacy formats (PKL + MCAP)
    isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file path/to/sim_log.pkl --real_log path/to/real_log.mcap
    ```
*   **Output**:
<img src="docs\sim2real_eval.png"> 
    * `scripts/evaluation/result/sim2real_report_{YYYY-MM-DD}.csv`

---

## 📊 Metrics / 평가 지표

The following metrics are calculated to evaluate the policy performance and Sim-to-Real gap.
정책 성능과 Sim-to-Real 격차를 평가하기 위해 다음 지표들이 계산됩니다.

| Metric (지표) | Description (설명) | Note (비고) |
| :--- | :--- | :--- |
| **Velocity Tracking Error (RMSE)** | Root Mean Square Error between command velocity and actual velocity. <br> 명령 속도와 실제 속도 간의 평균 제곱근 오차. | Lower is better. |
| **Torque Reality Gap (RMSE)** | Difference between simulated torque and real actuator torque for the same motion. <br> 동일 동작 수행 시 시뮬레이션 토크와 실제 액추에이터 토크 간의 차이. | Lower is better. |
| **Sim CoT (Mech)** | Cost of Transport in Simulation (Mechanical Work only). <br> 시뮬레이션 상의 이동 비용 (기계적 일률 기준). | $P_{mech} / (mgv)$ |
| **Real CoT (Mech)** | Cost of Transport in Real World (Mechanical Work only). <br> 실제 로봇의 이동 비용 (기계적 일률 기준). | Used for Sim-to-Real comparison. |
| **Real CoT (Elec)** | Cost of Transport in Real World (Total Electrical Power). <br> 실제 로봇의 이동 비용 (배터리 총 소모량 기준). | Includes computer/sensor power. Higher than Mech. |
| **Torque Smoothness (Jitter)** | Mean absolute derivative of torque over time. Indicates control stability. <br> 시간당 토크 변화량의 평균. 제어 안정성을 나타냄. | Lower is better. |

> **Note**: CoT values are set to 0.0 if the robot's velocity is near zero (< 0.01 m/s) to avoid division by zero errors.
> **참고**: 로봇의 속도가 0에 가까울 경우 (< 0.01 m/s), 0으로 나누는 오류를 방지하기 위해 CoT 값은 0.0으로 설정됩니다.

---

## 📁 Directory Structure / 디렉토리 구조

```
scripts/evaluation/
├── go2_logger/          # ROS2 Package for real robot logging
│   └── logs/            # Real-world logs (.csv, .mcap)
├── result/              # Output directory
│   ├── csv/             # Simulation raw data (.csv)
│   ├── pkl/             # Simulation raw data (.pkl)
│   └── sim2real_report_*.csv  # Daily evaluation reports
├── sim_eval.py          # Simulation inference script
├── sim2real_eval.py     # Comparison & Analysis script
└── README.md            # This file
```
