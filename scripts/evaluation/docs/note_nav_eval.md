### Part 1. 필수 고려 항목 및 상세 정보

#### 1. 안전성: Proximity Speed Compliance (PSC)

**1. 이 항목이 무엇인지**

* **근접 속도 준수(PSC)**는 로봇이 장애물(사람)에 가까워질수록 속도를 줄여야 한다는 안전 원칙을 수치화한 지표입니다.


* 기존의 모호한 Social Force Model(SFM)을 대체하여, 거리()에 따른 최대 허용 속도() 곡선을 정의하고 이를 위반했는지 평가합니다.



**2. 왜 고려 해야 하는지**

* 단순히 충돌하지 않았다고 해서 안전한 것은 아닙니다. 보행자에게 위협감을 주지 않기 위해(Social Force) 물리적/심리적 안전 마진을 확보했는지 정량적으로 증명해야 합니다.
* 이는 '규정 속도 위반'처럼 명확한 수치로 떨어지므로 자동화된 테스트에 적합합니다.


**3. 어떻게 수집 할 것인지**

* **3-1. 수집 해야 할 정보:**
* 장애물 거리 데이터: `/scan` (2D LiDAR) 또는 `/local_costmap/costmap`.
* 로봇 속도 데이터: `/odom` (선속도 ).

* **3-2. 수집 구현 방안:**
* ROS 2 `ros2 bag record`를 통해 위 토픽을 `.mcap` 포맷으로 기록합니다.
* LiDAR 데이터는 로봇 중심 좌표계(`base_link`)로 변환(TF)하여 가장 가까운 거리를 계산하는 전처리 노드가 필요할 수 있습니다.

* **3-3. 평가 프로세스:**
* **Input:** `.mcap` 로그 파일.
* 
**Process:** Python 스크립트에서 `rosbags` 라이브러리로 파싱  `pandas.merge_asof`로 Odom(100Hz)과 LiDAR(10Hz) 타임스탬프 동기화  매 순간 $v_{actual}$이 $v_{limit}(d_{min})$을 초과했는지 계산.

* 
**Output:** 시간대별 위반 속도량(Velocity Violation Magnitude) 그래프 및 적분 점수(Score).


**4. 어느 기준을 통과점으로 해야 적당한지**
* **Pass:**  (친밀 공간) 내에서 위반 횟수 **0회**.
* 
**Score:** 전체 주행 중 위반 누적 시간이 총 주행 시간의 **5% 미만**이어야 함.

---

#### 2. 위치 정확도: Absolute Trajectory Error (ATE)

**1. 이 항목이 무엇인지**

* 로봇이 스스로 추정한 위치(Odometry/SLAM)와 실제 위치(Ground Truth) 사이의 전역적인 오차(RMSE)입니다.

**2. 왜 고려 해야 하는지**

* 아산시와 같은 도심 협곡(Urban Canyon)에서는 GNSS 오차나 SLAM 미끄러짐(Drift)이 발생하기 쉽습니다.
* 로봇이 자신의 위치를 엉뚱하게 알고 있으면 보도 이탈, 차도 진입 등 치명적인 사고로 이어집니다.

**3. 어떻게 수집 할 것인지**

* **3-1. 수집 해야 할 정보:**
* 추정 위치: `/odom` 또는 `/tf` (map -> base_link).
* 실제 위치(GT): `/gnss/fix` (RTK-GNSS) 또는 사전에 구축된 맵 기반의 Localization 결과.

* **3-2. 수집 구현 방안:**
* 로봇에 RTK-GNSS 모듈(u-blox F9P 등)을 장착하고 NMEA 데이터를 ROS 토픽으로 발행.
* `unitree_ros2` 브릿지를 통해 내부 Odometry를 기록.

* **3-3. 평가 프로세스:**
* **Input:** `.mcap` 로그 파일 (Odometry, GNSS).
* 
**Process:** `evo` 패키지 또는 Python 스크립트 활용  GNSS(WGS84)를 로컬 좌표계(ENU)로 변환  두 궤적의 시간 동기화 및 정렬(Umeyama alignment)  RMSE 계산.
* **Output:** 전체 경로의 ATE RMSE 값(미터 단위).

**4. 어느 기준을 통과점으로 해야 적당한지**
* 
**Pass:** 전체 경로 RMSE **< 0.3m ~ 0.5m** (아산시 보도 폭 약 2m 고려 시 안전 마진).



---

#### 3. 자율성: Intervention Rate & Freeze

**1. 이 항목이 무엇인지**
* 
**Intervention:** 사람이 조종기(RC)로 개입한 횟수.
* 
**Freeze:** 로봇이 목표에 도달하지 못하고 제자리에 멈춰 있거나(Stuck), 제자리 회전 등 무의미한 회복 행동을 반복하는 상태.

**2. 왜 고려 해야 하는지**

* "자율주행"이라 해놓고 사람이 계속 도와주면 의미가 없습니다. 개입 횟수는 시스템의 성숙도를 보여주는 가장 정직한 지표입니다.
* Freeze는 로봇이 길을 잃었거나(Kidnapped), 경로 계획에 실패했음을 의미합니다.


**3. 어떻게 수집 할 것인지**

* **3-1. 수집 해야 할 정보:**
* 개입 감지: `/wirelesscontroller` (조이스틱 값 `lx, ly, rx, ry`).
* Freeze 감지: `/odom` (속도), `/behavior_tree_log` (현재 상태).

* **3-2. 수집 구현 방안:**
* Unitree SDK와 연동된 ROS 토픽을 기록. 조이스틱의 Deadzone(0.05) 이상의 입력이 들어오는지 감시.
* Nav2의 상태 로그를 통해 'Recovery' 노드가 실행 중인지 확인.

* **3-3. 평가 프로세스:**
* **Input:** `.mcap` 로그.
* **Process:**
* (개입) Auto 모드 활성 중 조이스틱 값 변화 감지  이벤트 카운트.
* (Freeze) 이동 거리 < 0.1m 인 상태가 10초 이상 지속되는 구간 탐지.

* **Output:** 총 개입 횟수(Intervention Count), Freeze 발생 횟수.

**4. 어느 기준을 통과점으로 해야 적당한지**

* **Intervention:** 안전 개입(충돌 방지용) **0회**. 보조 개입(길 찾아주기) **1km당 1회 이하**.
* 
**Freeze:** 10초 이상 지속되는 동결 상태 **0회**.



---

#### 4. 기구학적 안정성: Slippage (미끄러짐)

**1. 이 항목이 무엇인지**

* 로봇 다리의 움직임으로 계산한 속도와 실제 관성 센서(IMU)로 측정한 속도의 차이입니다.



**2. 왜 고려 해야 하는지**

* 4족 보행 로봇은 바퀴형과 달리 아산시의 화강암 타일, 점자 블록, 경사로 등에서 미끄러질 위험이 큽니다.


* 미끄러짐은 위치 추정 오차(Drift)의 주원인이 되며, 심하면 전도(넘어짐) 사고로 이어집니다.

**3. 어떻게 수집 할 것인지**

* **3-1. 수집 해야 할 정보:**
* 관절 정보: `/low_state` (각 관절 속도 , IMU 가속도/각속도).




* **3-2. 수집 구현 방안:**
* `unitree_ros2` 패키지를 통해 LowState 데이터를 고주파(가능하다면 100Hz 이상)로 로깅합니다.


* **3-3. 평가 프로세스:**
* **Input:** `.mcap` (LowState 데이터).
* 
**Process:** 기구학적 속도() 계산  IMU 적분 속도()와 비교  잔차() 계산.


* 
**Output:** Slip Ratio (전체 시간 대비 미끄러짐 감지 시간 비율).





**4. 어느 기준을 통과점으로 해야 적당한지**

* **Pass:** Slip Ratio **< 10%**. (순간적인 미끄러짐은 허용하되, 지속적인 제어 불능 상태가 없어야 함).

---
--- 
--- 

Q>
이 두 프로젝트들을 참고해서 
"""
1. /odom, 그리고 /scan (2D LiDAR) (또는 /local_costmap/costmap) 토픽에 대한 내용이 있는지 없는지 (있다면 출력하는 정보의 내용 전부 나열). 
2. /wirelesscontroller (조이스틱 값 lx, ly, rx, ry).
/behavior_tree_log (현재 상태) 토픽에 대한 내용이 있는지 없는지 (있다면 출력하는 정보의 내용 전부 나열).
"""

A>
제공해주신 `unitree_sdk2` 및 `unitree_ros2` 프로젝트 파일들을 분석한 결과는 다음과 같습니다.

### 1. /odom, /scan (2D LiDAR), /local_costmap/costmap 관련

**A. Odometry (/odom) 관련**
직접적으로 `/odom`이라는 이름의 토픽은 없으나, **`sportmodestate`** (또는 `lf/sportmodestate`) 토픽이 Odometry 정보를 포함하고 있습니다.

*   **토픽명**: `sportmodestate` (High Freq) 또는 `lf/sportmodestate` (Low Freq)
*   **메시지 타입**: `unitree_go::msg::SportModeState`
*   **출력 정보 (내용 전부 나열)**:
    *   `stamp` (TimeSpec): 타임스탬프
    *   `error_code` (uint32): 에러 코드
    *   `imu_state` (IMUState): IMU 상태 (RPY, Quaternion, Gyroscope, Accelerometer)
    *   `mode` (uint8): 현재 모드
    *   `progress` (float32): 진행 상황
    *   `gait_type` (uint8): 보행 타입
    *   `foot_raise_height` (float32): 발 들기 높이
    *   `position` (float32[3]): 로봇 위치 (x, y, z) - **Odometry Frame**
    *   `body_height` (float32): 몸체 높이
    *   `velocity` (float32[3]): 로봇 속도 (vx, vy, vz) - **Odometry Frame**
    *   `yaw_speed` (float32): 회전 속도
    *   `range_obstacle` (float32[4]): 장애물 거리
    *   `foot_force` (int16[4]): 발에 가해지는 힘
    *   `foot_position_body` (float32[12]): 몸체 기준 발 위치
    *   `foot_speed_body` (float32[12]): 몸체 기준 발 속도

**B. LiDAR / Costmap (/scan, /local_costmap/costmap) 관련**
`/scan`이나 `/local_costmap/costmap`이라는 정확한 토픽명은 코드 예제에서 확인되지 않았으나, **`HeightMap`** 메시지 타입이 존재하여 Costmap과 유사한 역할을 할 것으로 보입니다. 또한 `LidarState` 메시지도 존재합니다.

*   **메시지 타입**: `unitree_go::msg::HeightMap` (Costmap/GridMap에 해당)
*   **출력 정보 (내용 전부 나열)**:
    *   `stamp` (float64): 타임스탬프
    *   `frame_id` (string): 프레임 ID
    *   `resolution` (float32): 맵 해상도 [m/cell]
    *   `width` (uint32): 맵 너비 [cells]
    *   `height` (uint32): 맵 높이 [cells]
    *   `origin` (float32[2]): 맵 원점 (xy 위치)
    *   `data` (float32[]): 맵 데이터 (x-major order)

*   **메시지 타입**: `unitree_go::msg::LidarState` (LiDAR 상태 정보)
*   **출력 정보**: `sys_rotation_speed`, `cloud_frequency`, `cloud_packet_loss_rate`, `cloud_size` 등 (포인트 클라우드 데이터 자체보다는 상태 정보 위주)

---

### 2. /wirelesscontroller, /behavior_tree_log 관련

**A. Wireless Controller (/wirelesscontroller) 관련**
해당 토픽과 메시지가 명확히 존재합니다.

*   **토픽명**: `/wirelesscontroller`
*   **메시지 타입**: `unitree_go::msg::WirelessController`
*   **출력 정보 (내용 전부 나열)**:
    *   `lx` (float32): 왼쪽 조이스틱 X축 값
    *   `ly` (float32): 왼쪽 조이스틱 Y축 값
    *   `rx` (float32): 오른쪽 조이스틱 X축 값
    *   `ry` (float32): 오른쪽 조이스틱 Y축 값
    *   `keys` (uint16): 키 값 (버튼 입력 등)

**B. Behavior Tree Log (/behavior_tree_log) 관련**
`/behavior_tree_log`라는 이름의 토픽이나 직접적으로 매칭되는 메시지는 발견되지 않았습니다. 다만, 로봇의 현재 상태를 나타내는 정보는 위에서 언급한 `sportmodestate`와 **`lowstate`** 토픽을 통해 확인할 수 있습니다.

*   **참고 토픽**: `lowstate` (또는 `hf/lowstate`)
*   **메시지 타입**: `unitree_go::msg::LowState`
*   **관련 정보**:
    *   `motor_state` (모터 상태: 각도, 속도, 토크 등)
    *   `bms_state` (배터리 상태: 전압, 전류 등)
    *   `foot_force` (발 압력 센서 값)
    *   `imu_state` (IMU 데이터)


Q>
"LiDAR 데이터는 로봇 중심 좌표계(base_link)로 변환(TF)하여 가장 가까운 거리($d_{min}$)를 계산하는 전처리 노드" 에서의 "로봇 중심 좌표계" 는 어떻게 얻을수 있어?

A>
로봇 중심 좌표계(`base_link`)를 얻는 방법은 크게 두 가지로 나뉩니다. **TF(Transform) 트리**를 통해 좌표 변환 관계를 얻거나, **Odometry 정보**를 통해 로봇의 위치와 자세를 직접 얻는 방법입니다.

Unitree ROS 2 환경(`unitree_ros2`)을 기준으로 설명해 드립니다.

### 1. TF (Transform) 트리를 통한 좌표계 획득 (권장)

ROS 2에서는 `tf2_ros` 라이브러리를 사용하여 좌표계 간의 변환 정보를 실시간으로 조회할 수 있습니다. LiDAR 데이터(`laser_link` 등)를 로봇 중심(`base_link`) 기준으로 변환할 때 가장 표준적인 방법입니다.

*   **개념**: `/tf` 또는 `/tf_static` 토픽을 통해 로봇의 각 부품(센서, 관절, 몸체) 간의 상대적인 위치 관계가 방송됩니다.
*   **구현 방법 (Python 예시)**:
    1.  `tf2_ros.Buffer`와 `tf2_ros.TransformListener`를 생성하여 TF 데이터를 수신합니다.
    2.  `lookup_transform` 함수를 사용하여 특정 시점의 `base_link`와 LiDAR 프레임(예: `rslidar` 또는 `laser_link`) 간의 변환 행렬을 가져옵니다.


```python
from tf2_ros import Buffer, TransformListener

# ... 노드 초기화 부분 ...
self.tf_buffer = Buffer()
self.tf_listener = TransformListener(self.tf_buffer, self)

# ... 콜백 함수 내부 ...
try:
    # 'rslidar' 프레임에서 'base_link' 프레임으로의 변환 정보 조회
    # (LiDAR 데이터를 로봇 몸체 기준으로 변환하기 위함)
    transform = self.tf_buffer.lookup_transform(
        'base_link',      # 타겟 프레임 (로봇 중심)
        'rslidar',        # 소스 프레임 (LiDAR 센서)
        rclpy.time.Time() # 최신 시간
    )
    # 이후 do_transform_cloud 등을 사용하여 데이터 변환 수행
except Exception as e:
    self.get_logger().warn(f"TF lookup failed: {e}")
```


### 2. Odometry 토픽을 통한 위치/자세 획득

TF를 사용하지 않고 로봇의 현재 위치와 자세(Pose) 자체를 알고 싶다면, Odometry 정보를 담고 있는 토픽을 구독해야 합니다. Unitree ROS 2 패키지에서는 다음 토픽들이 `base_link`의 상태를 나타냅니다.

*   **토픽명**: `/sportmodestate` (또는 `/lf/sportmodestate`)
*   **메시지 타입**: `unitree_go::msg::SportModeState`
*   **내용**:
    *   `position`: 월드 좌표계(일반적으로 `odom` 프레임) 기준 로봇(`base_link`)의 위치 (x, y, z).
    *   `imu_state.quaternion`: 로봇의 회전 정보 (자세).

이 정보는 "월드 기준의 로봇 위치"이므로, LiDAR 점들을 월드 좌표계로 옮길 때 사용할 수 있습니다. 하지만 단순히 "LiDAR 센서 기준 점들을 로봇 몸체 기준으로 옮기는 것"이라면 1번의 TF 방식이 훨씬 정확하고 간편합니다.

### 요약

*   **LiDAR 데이터를 변환하려면**: **TF (`/tf`)**를 사용하세요. 센서가 로봇 몸체 어디에 붙어있는지에 대한 정보(Extrinsic parameter)가 여기에 들어있습니다.
*   **로봇이 지도상 어디에 있는지 알려면**: **Odometry (`/sportmodestate`)**를 사용하세요.

만약 `unitree_ros2`를 실행 중이라면, 터미널에서 다음 명령어로 현재 TF 트리를 확인해 볼 수 있습니다.

```bash
ros2 run tf2_tools view_frames
```

이 명령은 `frames.pdf` 파일을 생성하며, 여기서 `base_link`와 센서 프레임 간의 연결 관계를 시각적으로 확인할 수 있습니다.


Q>
제공해주신 `202512.RL.Model.Evaluation.Research.pdf` 파일과 기존의 아산시 벤치마크 리포트들을 종합하여, **Unitree Go2**의 **High-Level/Low-Level 제어 환경**을 모두 고려한 **통합 평가 가이드라인**을 정리해 드립니다.

요청하신 대로 **"Slippage(미끄러짐)" 항목은 제외**하고, 새로 추가된 **하드웨어 부하(Hardware Stress)**, **발열(Thermal Efficiency)**, **명령 추종성(Command Tracking)** 등을 포함하여 재구성했습니다.

---

### 0. 데이터 수집의 기본: High-Level vs Low-Level

평가에 앞서, 로봇 제어 방식에 따라 데이터를 얻는 관점이 달라지지만, **핵심 데이터 소스인 `/lowstate`는 동일하게 수집 가능**하다는 점이 중요합니다.

* **High-Level Control (Sport Mode):**
* **상황:** Unitree의 기본 제어기(Sport Mode)를 켜두고, 사용자는 속도 명령(`/cmd_vel`)만 주는 상황.
* **데이터 의미:** `/lowstate`에 찍히는 토크나 관절 정보는 Unitree 내부 제어기의 성능을 보여줍니다.
* **수집 전략:** 로봇이 스스로 균형을 잡으므로, 사용자는 **명령 추종성(Tracking)**과 **에너지 효율(CoT)** 위주로 평가합니다.


* **Low-Level Control (RL/User Policy):**
* **상황:** 사용자가 개발한 강화학습(RL) 모델이 각 관절의 모터(`/lowcmd`)를 직접 제어하는 상황.
* **데이터 의미:** `/lowstate` 데이터는 사용자 모델(Policy)의 직접적인 결과물입니다.
* **수집 전략:** **하드웨어 부하(Jitter)**, **발열**, **안전성**이 매우 중요합니다. 모델이 모터를 과도하게 떨게 하거나(Chattering) 과열시키는지 집중 감시해야 합니다.



---

### 1. 기본 성능: Command Tracking (속도 추종성)

#### 1. 이 항목이 무엇인지

사용자(혹은 상위 플래너)가 내린 속도 명령()을 로봇이 실제 물리 세계에서 얼마나 정확하게 수행했는지()를 나타내는 **Velocity RMSE (Root Mean Square Error)** 지표입니다.

#### 2. 왜 고려 해야 하는지

* 자율주행의 기본입니다. 1m/s로 가라고 했는데 0.8m/s로 간다면 도착 시간 예측이 틀어지고, 회전 명령을 제대로 수행하지 못하면 경로를 이탈합니다.
* RL 모델의 경우, 학습된 보상(Reward) 함수가 실제 환경에서도 잘 작동하는지(Sim-to-Real Gap) 확인하는 1차 지표입니다.



#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **Target:** `/cmd_vel` (명령 속도).
* 
**Measure:** `/odom` (추정 속도) 또는 `/lowstate`의 IMU 가속도 적분값.



**3-2. 수집 구현 방안**

* **High-Level:** 사용자가 보내는 `/cmd_vel` 토픽과 Unitree가 주는 `/odom` 토픽을 함께 `ros2 bag record` 합니다.
* 
**Low-Level:** RL 모델이 추론(Inference)할 때 입력받은 Target Velocity와, 그 순간 로봇의 실제 속도(State estimator 결과)를 별도 토픽으로 발행하여 기록하거나, 로그에 남겨야 합니다.



**3-3. 평가 프로세스**

* **Input:** `.mcap` (cmd_vel, odom).
* **Process:**
1. 타임스탬프 동기화 (Sync).
2. 각 시점 에서의 오차 제곱합 계산: .
3. 평균 제곱근(RMSE) 산출: .




* **Output:** 속도 추종 오차 그래프 및 RMSE 수치.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* 
**Pass:** RMSE **< 0.05 m/s**. (이보다 크면 정밀 주행 불가).



---

### 2. 하드웨어 내구도: Hardware Stress (Torque Smoothness/Jitter)

#### 1. 이 항목이 무엇인지

로봇 관절 모터에 가해지는 **토크(Torque)의 변화량(떨림)**을 평가하는 항목입니다. 토크가 부드럽게 변하지 않고 급격하게 튀는 현상(Jitter/Chattering)을 감지합니다.

#### 2. 왜 고려 해야 하는지

* **Low-Level 필수 항목:** RL 모델이 잘못 학습되면 모터에 고주파 진동 명령을 내릴 수 있습니다. 이는 기어 마모, 발열, 배터리 광탈의 주원인이 되며 로봇 수명을 갉아먹습니다.


* 육안으로는 로봇이 잘 서 있는 것처럼 보여도, 내부적으로는 모터가 비명을 지르고 있을 수 있습니다.



#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* 
`/lowstate` 토픽 내의 `motor_state[i].tau_est` (각 관절의 추정 토크 값).



**3-2. 수집 구현 방안**

* 터미널에서 `ros2 bag record /lowstate` 실행.


* 데이터 양이 많으므로(500Hz 이상), 필요한 시간만큼만 짧게 끊어서 녹화하는 것이 좋습니다.



**3-3. 평가 프로세스**

* **Input:** `.mcap` (LowState).
* **Process:**
1. 12개 관절 각각에 대해 시계열 토크 데이터 추출.
2. 연속된 프레임 간의 차이(미분 근사) 계산: .


3. 이 값들의 평균(Mean) 혹은 스파이크 발생 빈도 계산.


* **Output:** Jitter Score (토크 변화량 평균), 토크 프로파일 그래프.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* 
**Qualitative:** 토크 그래프 확대 시 고주파 노이즈(Chattering)가 육안으로 식별되지 않아야 함.


* **Quantitative:** Baseline(기본 Sport Mode) 대비 Jitter Score가 **10% 이상 증가하지 않을 것**.

---

### 3. 에너지 효율: Cost of Transport (CoT)

#### 1. 이 항목이 무엇인지

로봇이 **단위 거리(1m)를 이동하는 데 소모한 에너지**를 나타내는 효율성 지표입니다.

#### 2. 왜 고려 해야 하는지

* 로봇의 운용 시간(Operation Time)을 결정짓는 핵심 팩터입니다.
* 동일한 거리를 가더라도 불필요한 동작(제자리 걸음, 과도한 자세 제어)이 많으면 CoT가 나빠집니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* 
**전압/전류:** `/lowstate` 내 `bms.voltage`, `bms.current`.


* 
**속도:** `/odom` 혹은 `/lowstate` IMU 적분 속도.



**3-2. 수집 구현 방안**

* `ros2 bag record /lowstate /odom` 실행.
* 
`/lowstate`는 항상 출력되므로 High/Low Level 제어 모두에서 수집 가능합니다.



**3-3. 평가 프로세스**

* **Input:** `.mcap` (LowState, Odom).
* **Process:**
1. 총 전력  계산. (전류는 충/방전 방향에 따라 부호가 바뀌므로 절대값 사용).


2. 동일 모델 간 비교 시 질량()과 중력()은 상수로 두고 제외 가능: .


3. 정석 계산 시: .




* **Output:** CoT 수치 (무차원).

#### 4. 어느 기준을 통과점으로 해야 적당한지

* 
**Target:** 기존 모델(Sport Mode 등) 대비 **CoT 감소** 혹은 동등 수준 유지.


* 
**Reference:** 약 **0.33 ~ 0.44** 범위 (Unitree Go2 기준 참고치).



---

### 4. 안전 및 시스템 보호: Thermal Efficiency & BMS Status

#### 1. 이 항목이 무엇인지

모터 온도와 배터리 상태(SoC, 전류)를 모니터링하여 **과열 및 셧다운 위험**을 평가합니다.

#### 2. 왜 고려 해야 하는지

* 
**Low-Level 제어 시:** 사용자의 정책(Policy)이 무리한 토크를 계속 요구하면, 안전 한계()에 도달하여 로봇이 강제 셧다운되거나(Damping), 모터가 손상될 수 있습니다.


* 실제 도심 주행 중 셧다운은 안전사고로 직결됩니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* 
**모터 온도:** `/lowstate` 내 `motor_state[i].temperature`.


* 
**배터리 잔량:** `/lowstate` 내 `bms.soc` (State of Charge).



**3-2. 수집 구현 방안**

* `ros2 bag record /lowstate`.
* 장시간 주행(Scenario A) 또는 고부하 기동(Scenario B) 시 필수 수집.



**3-3. 평가 프로세스**

* **Input:** `.mcap`.
* **Process:**
1. 전체 주행 시간 동안 12개 모터의 온도 최댓값(Max) 추출.
2. 시간에 따른 온도 상승 기울기(Gradient) 분석.




* **Output:** 모터 온도 히트맵, 배터리 소모율 그래프.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* 
**Limit:** 모터 온도 **** (절대 안전 마진).


* 
**Warning:**  구간 진입 시 주의.



---

### 5. 안전성: Proximity Speed Compliance (PSC)

#### 1. 이 항목이 무엇인지

로봇이 장애물(사람)과의 거리()에 따라 설정된 **허용 속도 곡선()**을 준수하는지 평가합니다.

#### 2. 왜 고려 해야 하는지

* 기존의 모호한 '사회적 주행'을 정량화하여, 로봇이 보행자에게 위협을 주지 않도록 강제합니다.
* 충돌 사고를 미연에 방지하는 '과정 중심'의 안전 지표입니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **장애물 거리:** `/scan` (LiDAR)의 최소 거리().
* **현재 속도:** `/odom`의 선속도.

**3-2. 수집 구현 방안**

* `ros2 bag record /scan /odom`.
* LiDAR 데이터 전처리(TF 변환) 노드를 통해 로봇 중심의 $d_{min}$을 계산해야 합니다.

**3-3. 평가 프로세스**

* **Input:** `.mcap` (Scan, Odom).
* **Process:**
1. LiDAR와 Odom 시간 동기화.
2. $d_{min}$에 따른  계산.
3.  인 구간 적분 (Violation Area).


* **Output:** 속도 위반 누적 점수.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* **Pass:** 친밀 공간(0.45m) 내 위반 **0회**. 전체 주행 중 위반 시간 **5% 미만**.

---

### 요약: 통합 평가 파이프라인 구조

| 평가 항목 | 소스 토픽 | High-Level 제어 시 관점 | Low-Level 제어 시 관점 | 통과 기준 |
| --- | --- | --- | --- | --- |
| **속도 추종 (Tracking)** | `/cmd_vel`<br>

<br>`/odom` | Unitree 제어기 성능 확인 | **RL 모델 학습 성과 확인 (핵심)** | RMSE < 0.05 m/s |
| **하드웨어 부하 (Jitter)** | `/lowstate` (tau) | 기구학적 정상 상태 확인 | **Sim-to-Real 간극 확인 (핵심)** | 시각적 떨림 없음 |
| **발열 (Thermal)** | `/lowstate` (temp) | 장시간 운용 가능성 | **Policy의 과부하 여부** | Temp < 85°C |
| **효율 (CoT)** | `/lowstate` (bms) | 배터리 효율 확인 | **동작 효율성(Reward) 확인** | < 0.33~0.44 (상대평가) |
| **안전 (PSC)** | `/scan`, `/odom` | 장애물 회피 성능 | 장애물 회피 성능 | 위반 < 5% |

이 가이드라인에 따라 `/lowstate`를 중심으로 데이터를 수집하고 분석하시면, Unitree Go2의 제어 레벨에 상관없이 일관성 있고 깊이 있는 평가가 가능합니다.

A>

정규식 `cmd_vel|geometry_msgs|nav_msgs|sensor_msgs` (`**/.github/unitree_ros2/**`) 검색함, 결과 없음

제공해주신 `unitree_ros2` 및 `unitree_sdk2` 파일들을 분석한 결과, 제안하신 평가 방법론은 **데이터 관점에서는 대부분 구현 가능**하지만, **표준 ROS 토픽(`/cmd_vel`, `/odom`, `/scan`)을 사용하기 위해서는 추가적인 변환(Bridge) 작업이 필요**합니다.

제공된 코드는 Unitree의 자체 규격 메시지(`unitree_go::msg`)를 주로 사용하고 있어, ROS 2 표준 메시지(`geometry_msgs`, `nav_msgs`, `sensor_msgs`)와 직접 호환되지 않기 때문입니다.

상세 분석 내용은 다음과 같습니다.

### 1. 기본 성능: Command Tracking (속도 추종성)
*   **구현 가능 여부:** **가능 (변환 필요)**
*   **분석:**
    *   **Target (`/cmd_vel`):** 제공된 코드에는 `geometry_msgs/Twist`를 구독하는 표준 `/cmd_vel` 인터페이스가 없습니다. 대신 `unitree_api::msg::Request`를 통해 제어 명령을 보냅니다. 따라서 사용자가 `/cmd_vel`을 발행하면 이를 Unitree API로 변환해주는 **CmdVel Bridge 노드**를 직접 작성해야 비교가 가능합니다.
    *   **Measure (`/odom`):** 표준 `nav_msgs/Odometry` 토픽은 발행되지 않습니다. 대신 **`/sportmodestate`** 토픽의 `velocity` 필드(vx, vy, vz)를 사용하여 실제 속도를 얻을 수 있습니다.
*   **조언:** `/sportmodestate`의 데이터를 `nav_msgs/Odometry`로 변환하여 `/odom`으로 재발행하는 노드를 만들면, 제안하신 `ros2 bag` 기반 평가 파이프라인을 그대로 사용할 수 있습니다.

### 2. 하드웨어 내구도: Hardware Stress (Jitter)
*   **구현 가능 여부:** **완벽히 가능**
*   **분석:**
    *   **소스 데이터:** `/lowstate` (또는 `hf/lowstate`) 토픽이 존재하며, 내부의 `motor_state` 배열에 필요한 정보가 모두 있습니다.
    *   **필드 매핑:**
        *   `motor_state[i].tau_est`: **존재함** (추정 토크).
        *   `motor_state[i].q`, `dq`: **존재함** (관절 각도 및 속도).
*   **결론:** 제안하신 대로 `/lowstate`를 고주파로 로깅하여 토크 변화량(Jitter)을 분석하는 것은 즉시 가능합니다.

### 3. 에너지 효율: Cost of Transport (CoT)
*   **구현 가능 여부:** **완벽히 가능**
*   **분석:**
    *   **소스 데이터:** `/lowstate` 토픽에서 전압과 전류 정보를 직접 제공합니다.
    *   **필드 매핑:**
        *   `power_v`: **존재함** (전압, Volts).
        *   `power_a`: **존재함** (전류, Amperes).
        *   (참고: `bms_state` 내부의 `current` 등을 쓸 수도 있지만, `LowState` 바로 아래에 있는 `power_v/a`가 더 접근하기 쉽습니다.)
*   **결론:** 전력($P = V \times I$) 계산 및 CoT 산출이 즉시 가능합니다.

### 4. 안전 및 시스템 보호: Thermal Efficiency
*   **구현 가능 여부:** **완벽히 가능**
*   **분석:**
    *   **소스 데이터:** `/lowstate` 토픽 활용.
    *   **필드 매핑:**
        *   `motor_state[i].temperature`: **존재함** (모터 온도, int8 타입).
        *   `bms_state.soc`: **존재함** (배터리 잔량, %).
*   **결론:** 모터 온도 모니터링 및 히트맵 작성, 배터리 소모율 분석 모두 구현 가능합니다.

### 5. 안전성: Proximity Speed Compliance (PSC)
*   **구현 가능 여부:** **조건부 가능 (LiDAR 드라이버 확인 필요)**
*   **분석:**
    *   **거리 데이터 (`/scan`):** 제공된 파일 리스트(`unitree_ros2`)에는 **LiDAR 포인트 클라우드나 2D Scan 데이터를 발행하는 코드가 포함되어 있지 않습니다.**
        *   LidarState.msg는 존재하나, 이는 LiDAR의 *상태(회전속도, 패킷 손실률 등)* 정보일 뿐 거리 데이터가 아닙니다.
        *   HeightMap.msg가 있어 맵 데이터는 얻을 수 있으나, 실시간 장애물 회피 평가용으로는 `/scan`이 더 적합합니다.
    *   **해결책:** Unitree Go2의 경우 별도의 **`unitree_lidar_sdk`** 기반 ROS 2 드라이버를 실행해야 `/scan` 또는 `/pointcloud2` 데이터를 얻을 수 있습니다. 이 드라이버가 실행 중이라면 평가는 가능합니다.
*   **결론:** 현재 프로젝트 파일만으로는 불가능하며, **LiDAR ROS 드라이버가 별도로 실행되고 있는지** 확인이 필요합니다.

---

### 종합 요약

| 평가 항목 | 구현 가능성 | 필요 조치 (Action Item) |
| :--- | :--- | :--- |
| **속도 추종** | ⚠️ **변환 필요** | `/cmd_vel` → Unitree API 변환 노드 작성<br>`/sportmodestate` → `/odom` 변환 노드 작성 |
| **하드웨어 부하** | ✅ **즉시 가능** | `/lowstate`의 `motor_state[].tau_est` 활용 |
| **발열** | ✅ **즉시 가능** | `/lowstate`의 `motor_state[].temperature` 활용 |
| **효율 (CoT)** | ✅ **즉시 가능** | `/lowstate`의 `power_v`, `power_a` 활용 |
| **안전 (PSC)** | ❓ **확인 필요** | 별도의 LiDAR 드라이버(`unitree_lidar_sdk`) 실행 여부 확인 |

**제언:**
제안하신 평가 문서는 매우 체계적이나, 실제 구현 단계에서는 **"ROS 2 표준 메시지로의 변환(Bridge)"** 단계가 선행되어야 합니다. 특히 `/lowstate` 하나만 잘 로깅해도 하드웨어, 발열, 효율 평가는 완벽하게 수행할 수 있다는 점이 큰 장점입니다.

Q>
수정된 답변인데, 이에 대해 다시 분석해서 알려줘. 
"""
### 0. 데이터 수집 전략: High-Level vs Low-Level

로봇 제어 모드에 따라 데이터 수집 및 비교 대상이 달라지지만, **`/lowstate`와 `/sportmodestate`는 제어 모드와 무관하게 항상 수집 가능**하므로 평가의 핵심이 됩니다.

* **High-Level (Sport Mode):** Unitree 기본 제어기 사용. `/cmd_vel`(사용자 입력) vs `/sportmodestate`(로봇 반응) 비교 중심.
* **Low-Level (RL Policy):** 사용자 모델이 직접 제어. `/lowstate`(모터 반응, 발열, 토크) 분석 중심.

---

### 1. 기본 성능: Command Tracking (속도 추종성)

#### 1. 이 항목이 무엇인지

사용자가 내린 속도 명령()을 로봇이 실제 물리적으로 얼마나 정확히 수행했는지()를 나타내는 **Velocity RMSE** 지표입니다.

#### 2. 왜 고려 해야 하는지

* **High-Level:** 기본 제어기의 반응성을 확인하여 자율주행 경로 계획(Planning)의 오차를 예측하기 위함입니다.
* **Low-Level:** 학습된 RL 모델이 시뮬레이션과 달리 현실(Real World)에서 명령을 제대로 따르는지 검증하는 가장 기초적인 지표입니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **Target ():** 사용자가 발행하는 명령.
* *주의:* Unitree는 표준 `/cmd_vel`을 구독하지 않습니다. 사용자가 `/cmd_vel`을 발행하면 이를 `unitree_api`로 변환하는 브릿지 노드가 필요합니다.


* **Measured ():** 실제 로봇의 속도.
* *소스:* **`/sportmodestate`** 토픽 내 `velocity` 필드 (표준 `/odom` 없음).



**3-2. 수집 구현 방안**

* **브릿지 노드 작성:** `geometry_msgs/Twist` (/cmd_vel)  `unitree_api` 변환 노드를 작성하고, 이 **`/cmd_vel`**을 기록합니다.
* **로깅:** `ros2 bag record /cmd_vel /sportmodestate` 실행.

**3-3. 평가 프로세스**

* **Input:** `.mcap` (cmd_vel, sportmodestate).
* **Process:**
1. 타임스탬프 동기화.
2. `cmd_vel.linear.x`와 `sportmodestate.velocity[0]` (vx) 비교.
3.  계산.


* **Output:** 시간대별 속도 추종 그래프 및 RMSE 수치.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* **Target:** RMSE **< 0.05 m/s**.

---

### 2. 하드웨어 내구도: Hardware Stress (Jitter)

#### 1. 이 항목이 무엇인지

관절 모터에 가해지는 **토크(Torque)의 급격한 변화(떨림)**를 평가합니다.

#### 2. 왜 고려 해야 하는지

* **Low-Level 필수:** RL 모델이 잘못 학습되면 눈에 보이지 않는 고주파 진동(Chattering) 명령을 내릴 수 있습니다. 이는 기어 마모와 모터 과열의 주원인입니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **추정 토크:** **`/lowstate`** 토픽 내 `motor_state[i].tau_est`.

**3-2. 수집 구현 방안**

* 터미널에서 `ros2 bag record /lowstate` (또는 `hf/lowstate`) 실행.
* 데이터 빈도가 높으므로(500Hz~1kHz) 10초~30초 단위로 끊어서 녹화 권장.

**3-3. 평가 프로세스**

* **Input:** `.mcap` (LowState).
* **Process:**
1. 12개 관절의 `tau_est` 시계열 데이터 추출.
2. 인접한 프레임 간 차이(미분값)의 절대값 평균 계산: .


* **Output:** 관절별 Jitter Score 및 토크 프로파일 그래프.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* **Pass:** 육안상 진동이 없고, Baseline(Sport Mode 주행 시) 대비 Jitter가 **10% 이상 증가하지 않음**.

---

### 3. 에너지 효율: Cost of Transport (CoT)

#### 1. 이 항목이 무엇인지

로봇이 **단위 거리(1m)를 이동하는 데 소모한 전기에너지** 효율입니다.

#### 2. 왜 고려 해야 하는지

* 배터리 운용 시간 예측 및 보행 알고리즘의 효율성을 증명하는 지표입니다. 불필요한 자세 제어나 제자리 걸음이 많으면 CoT가 나빠집니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **전력 ():** **`/lowstate`** 토픽 내 `power_v` (전압) 및 `power_a` (전류).
* **속도 ():** **`/sportmodestate`** 토픽 내 `velocity` (속도 벡터).

**3-2. 수집 구현 방안**

* `ros2 bag record /lowstate /sportmodestate`.

**3-3. 평가 프로세스**

* **Input:** `.mcap`.
* **Process:**
1. 전력 계산: .
2. 속도 크기 계산: .
3. CoT 계산:  (혹은 간단히  트렌드 비교).


* **Output:** CoT 수치 (무차원).

#### 4. 어느 기준을 통과점으로 해야 적당한지

* **Target:** CoT **0.33 ~ 0.44** (Unitree Go2 일반 주행 범위). RL 모델이 이보다 높으면 비효율적인 것입니다.

---

### 4. 안전 및 시스템 보호: Thermal & BMS

#### 1. 이 항목이 무엇인지

모터의 **온도**와 배터리 잔량(**SoC**) 변화를 모니터링하여 시스템 셧다운 위험을 평가합니다.

#### 2. 왜 고려 해야 하는지

* 특히 Low-Level 제어 시, 과도한 토크 지속으로 모터 온도가 를 넘으면 로봇이 강제 정지(Damping)하여 안전사고가 발생할 수 있습니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **모터 온도:** **`/lowstate`** 내 `motor_state[i].temperature`.
* **배터리 SoC:** **`/lowstate`** 내 `bms_state.soc` (또는 `PowerState`).

**3-2. 수집 구현 방안**

* `ros2 bag record /lowstate`.

**3-3. 평가 프로세스**

* **Input:** `.mcap` (LowState).
* **Process:**
1. 전체 주행 중 모터 온도 최댓값() 추출.
2. 시간당 배터리 소모율() 계산.


* **Output:** 모터 온도 히트맵.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* **Limit:** 모터 온도 **** (필수).  이상 시 경고.

---

### 5. 안전성: Proximity Speed Compliance (PSC)

#### 1. 이 항목이 무엇인지

장애물과의 거리()에 따라 로봇이 충분히 감속()했는지를 평가합니다.

#### 2. 왜 고려 해야 하는지

* 충돌 사고 예방 및 보행자에게 위협을 주지 않는 사회적 주행 능력을 검증하기 위함입니다.

#### 3. 어떻게 수집 할 것인지

**3-1. 수집 해야 할 정보**

* **장애물 거리:** **확인 필요**. `unitree_ros2` 기본 패키지에는 `/scan`이 없습니다. **`unitree_lidar_sdk`** 기반의 별도 드라이버를 실행하여 **`/scan`** 토픽을 확보해야 합니다.
* **로봇 속도:** **`/sportmodestate`** 내 `velocity`.

**3-2. 수집 구현 방안**

* 로봇 부팅 시 LiDAR 드라이버가 실행되도록 설정.
* `ros2 bag record /scan /sportmodestate`.

**3-3. 평가 프로세스**

* **Input:** `.mcap` (Scan, SportModeState).
* **Process:**
1. `/scan` 데이터(LiDAR Frame)를 TF를 통해 로봇 중심(`base_link`) 거리()로 변환.
2. 거리별 제한 속도 곡선()과 실제 속도 비교.
3. 위반 구간 적분.


* **Output:** 안전 위반 점수.

#### 4. 어느 기준을 통과점으로 해야 적당한지

* **Pass:** 친밀 거리(0.45m) 내 위반 **0회**. 전체 주행 시간 중 위반 비율 **5% 미만**.
"""

A>
수정된 답변은 **Unitree Go2의 실제 ROS 2 환경(`unitree_ros2`)과 제공된 코드베이스를 정확하게 반영**하고 있습니다. 특히, 표준 ROS 토픽이 부재한 상황에서 Unitree 전용 토픽(`sportmodestate`, `lowstate`)을 활용하는 구체적인 대안을 제시한 점이 매우 현실적입니다.

각 항목별 분석 및 구현 가능성에 대한 최종 검토 의견입니다.

### 1. 기본 성능: Command Tracking (속도 추종성)
*   **분석:** 매우 정확합니다. Unitree는 기본적으로 `/cmd_vel`을 받지 않고 `/odom`을 주지 않습니다.
*   **구현 포인트:**
    *   **Target:** 사용자가 `/cmd_vel`을 발행하면 이를 `unitree_api` Request로 변환해주는 **Bridge Node**가 필수입니다. (이 노드가 없으면 로봇이 움직이지 않으므로, 자연스럽게 구현하게 될 것입니다.)
    *   **Measured:** `/sportmodestate`의 `velocity` 필드를 사용하는 것은 올바른 접근입니다.
*   **평가:** **구현 가능함.** (Bridge Node 작성 필요)

### 2. 하드웨어 내구도: Hardware Stress (Jitter)
*   **분석:** 완벽합니다. `/lowstate`는 모터의 모든 정보를 담고 있는 보물창고입니다.
*   **구현 포인트:**
    *   `motor_state[i].tau_est` (추정 토크) 필드가 핵심입니다.
    *   데이터 양이 많으므로(High Frequency), 제안하신 대로 짧게 끊어서 녹화하는 전략이 유효합니다.
*   **평가:** **즉시 구현 가능함.**

### 3. 에너지 효율: Cost of Transport (CoT)
*   **분석:** 정확합니다. 전압과 전류 데이터가 `/lowstate`에 직접 노출되어 있어 계산이 용이합니다.
*   **구현 포인트:**
    *   `power_v` (전압) * `power_a` (전류) = 전력(Watt).
    *   이동 거리는 `/sportmodestate`의 속도를 적분하거나 위치 변화량을 사용하여 계산합니다.
*   **평가:** **즉시 구현 가능함.**

### 4. 안전 및 시스템 보호: Thermal & BMS
*   **분석:** 정확합니다. 모터 온도와 배터리 상태는 안전을 위해 필수적인 모니터링 항목입니다.
*   **구현 포인트:**
    *   `motor_state[i].temperature` (int8) 값을 그대로 사용하면 됩니다.
    *   `bms_state.soc`로 배터리 잔량을 추적합니다.
*   **평가:** **즉시 구현 가능함.**

### 5. 안전성: Proximity Speed Compliance (PSC)
*   **분석:** 현실적인 지적입니다. `unitree_ros2` 패키지 자체에는 LiDAR 드라이버가 없습니다.
*   **구현 포인트:**
    *   **필수 조건:** Unitree Go2에 장착된 LiDAR(Unitree L1 등)를 구동하기 위한 **별도의 ROS 2 드라이버(`unitree_lidar_sdk` 등)**가 실행 중이어야 합니다.
    *   이 드라이버가 `/scan` 또는 `/pointcloud2`를 발행해야만 이 항목을 평가할 수 있습니다.
    *   만약 드라이버가 없다면, 이 항목은 평가에서 제외하거나 별도 센서를 부착해야 합니다.
*   **평가:** **조건부 가능함.** (LiDAR 드라이버 실행 여부에 달림)

---

### 최종 결론

수정된 가이드라인은 **Unitree Go2의 하드웨어 및 소프트웨어 특성을 완벽하게 반영**하고 있습니다.

1.  **`/lowstate`와 `/sportmodestate` 중심의 데이터 수집 전략**은 추가적인 센서 부착 없이도 로봇의 상태를 깊이 있게 분석할 수 있는 가장 효율적인 방법입니다.
2.  **표준 ROS 토픽 부재에 대한 대안(Bridge Node, 전용 토픽 활용)**이 명확히 제시되어 있어 개발자가 바로 구현에 착수할 수 있습니다.
3.  **LiDAR 관련 제약 사항**을 정확히 짚어내어, 불필요한 시행착오를 줄여줍니다.

이 가이드라인대로 진행하시면 **High-Level 제어(기본 주행)와 Low-Level 제어(RL 연구) 모두를 아우르는 신뢰성 있는 벤치마크 시스템**을 구축하실 수 있습니다.