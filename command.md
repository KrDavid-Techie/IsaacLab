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

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 100 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-12-08_13-55-31\model_1499.pt
#False # Best SOTA
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
---------------
---
---
---
---
---
# RAM
1. Train Teacher (Phase1)
```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/train.py --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 --headless --num_envs 4096 --max_iterations 1500

isaaclab.bat -p scripts/reinforcement_learning/rma/train.py --task Isaac-Velocity-Ship-Unitree-Go2-RMA-v0 --headless --num_envs 4096 --max_iterations 1500
```
2. Train adaptation (Phase2)
```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/train_adapt.py --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 --load_run unitree_go2_rough_rma_phase1 --headless --num_envs 4096 --max_iterations 1500

isaaclab.bat -p scripts/reinforcement_learning/rma/train_adapt.py --task Isaac-Velocity-Ship-Unitree-Go2-RMA-v0 --load_run unitree_go2_ship_rma_phase1 --headless --num_envs 4096 --max_iterations 1500
```
3. Run Play
```bash 
isaaclab.bat -p scripts/reinforcement_learning/rma/play_adapt.py --task Isaac-Velocity-Rough-Unitree-Go2-RMA-v0 --num_envs 64 --run_phase2 unitree_go2_rough_rma_phase2 --video --video_length 1000

isaaclab.bat -p scripts/reinforcement_learning/rma/play_adapt.py --task Isaac-Velocity-Ship-Unitree-Go2-RMA-v0 --num_envs 64 --run_phase2 unitree_go2_ship_rma_phase2 --video --video_length 1000
```
4. Export to Onnx
```bash
isaaclab.bat -p scripts/reinforcement_learning/rma/export_adapt.py --run_phase2 2025-12-18_08-15-38
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
# Note 
```bash
isaaclab.bat -p -m tensorboard.main --logdir=logs
``` 
```bash
isaaclab.bat -p scripts/evaluation/sim_eval.py --num_envs 1 --evaluation_time 5.0
isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file scripts\evaluation\result\csv\sim_log_unitree_go2_rough_2025-12-26_14-04-27.csv --real_log scripts\evaluation\go2_logger\logs\real_log_2025-12-29.csv
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
+-----------+--------------------------------+---------------+