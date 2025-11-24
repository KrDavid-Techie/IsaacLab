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

### Rough

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 20 --max_iterations 100
# Visualized 
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 1000
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 500 --resume --load_run "2025-11-06_11-43-35" 
# 추가 학습
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 1000 --load_run "2025-11-20_08-15-24" 
# 기반학습
```

---------------

## Play
### Flat
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Unitree-Go2-v0  --num_envs 10 --checkpoint checkpoint_path --log_joints --log_commands --log_interval 20
```

### Rough
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 20 --checkpoint checkpoint_path 
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 10 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-11-06_11-43-35\model_999.pt #False
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 10 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-11-06_11-43-35\model_999.pt --log_joints  --log_interval 40 #False
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 10 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-11-17_17-32-35\model_1099.pt #False
```

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --num_envs 10 --checkpoint C:\Users\User\github\IsaacLab\logs\rsl_rl\unitree_go2_rough\2025-11-17_17-32-35\model_1099.pt --log_joints --log_commands --log_interval 20 #False
```


---
---
---
---
---

# SKRL
## Train
```bash
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --max_iterations 500 --headless
```

---------------

## Play
```bash
isaaclab.bat -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 8 --checkpoint C:\Users\User\github\IsaacLab\logs\skrl\unitree_go2_flat\2025-11-03_16-20-44_ppo_torch\checkpoints\best_agent.pt --video --video_length 200

```
---
---
---
---
---

# Unitree_rl_lab
## Train
```bash
python scripts/rsl_rl/train.py --headless --max_iterations 20000 --num_envs 4096 --task Unitree-Go2-Velocity
```

---------------

## Play
```bash
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --num_envs 1 --checkpoint [] --video --video_length 800 --log_joints --log_commands
```

---
---
---
---
---

# Note 
```bash
C:\Users\User\github\IsaacLab\source\isaaclab\isaaclab\envs\mdp\observations.py
C:\Users\User\github\IsaacLab\source\isaaclab\isaaclab\managers\manager_term_cfg.py
C:\Users\User\github\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\velocity_env_cfg.py
```
```bash
C:\Users\User\github\unitree_rl_lab\source\unitree_rl_lab\unitree_rl_lab\tasks\locomotion\robots\go2\velocity_env_cfg.py
```
```bash
isaaclab.bat -p -m tensorboard.main --logdir=logs
```
---------------