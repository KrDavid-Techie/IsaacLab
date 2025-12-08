# RSL_RL
## Train
### Flat
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --max_iterations 4000
```
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --max_iterations 4000 --resume --load_run "2025-11-14_09-41-57"
# 추가학습
```
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --max_iterations 4000 --load_run "2025-11-14_09-41-57"
# 기반학습
```
---
---
---
---
---
### Rough

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 1 --max_iterations 100
# Visualized 
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 1500
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 500 --resume --load_run "2025-11-24_19-17-49" 
# 기반 학습
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 1000 --load_run "2025-12-03_14-11-02"
# 추가 학습
```
---
---
---
---
---
### Stair

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Stair-Unitree-Go2-v0 --num_envs 4096 --max_iterations 1000
```
---
---
---
---
---
---------------
---
---
---
---
---
## Play
### Flat
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Unitree-Go2-v0  --num_envs 10 --checkpoint checkpoint_path --log_joints --log_commands --log_interval 20
```
---
---
---
---
---
### Rough
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 20 --checkpoint checkpoint_path 
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 20 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\sota_2025-11-06_11-43-35\model_999.pt
#False
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 20 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\sota2-2025-12-03_14-11-02\model_1499.pt --video --video_length 800
#False
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 20 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-12-04_11-48-06\model_999.pt
#False
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 20 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-12-04_18-29-47\model_4999.pt
#False
```
---
---
---
---
---
### Stair
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Stair-Unitree-Go2-v0  --num_envs 10 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_stair\2025-12-03_18-06-16\model_4999.pt
#False
```
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Stair-Unitree-Go2-v0  --num_envs 10 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_stair\2025-12-04_10-18-28\model_999.pt
#False
```
---
---
---
---
---
# Note 
```bash
isaaclab.bat -p -m tensorboard.main --logdir=logs
```
---------------
# [INFO] Observation Manager: <ObservationManager> contains 1 groups.
+------------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (3099,)) |
+-----------+--------------------------------+---------------+
|   Index   | Name                           |     Shape     |
+-----------+--------------------------------+---------------+
|     0     | base_lin_vel                   |      (3,)     |
|     1     | base_ang_vel                   |      (3,)     |
|     2     | projected_gravity              |      (3,)     |
|     3     | velocity_commands              |      (3,)     |
|     4     | joint_pos                      |     (12,)     |
|     5     | joint_vel                      |     (12,)     |
|     6     | actions                        |     (12,)     |
|     7     | height_scan                    |     (187,)    |
|     8     | lidar_scan                     |    (2864,)    |
+-----------+--------------------------------+---------------+