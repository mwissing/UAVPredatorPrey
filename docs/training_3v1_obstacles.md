# 3v1 Obstacles Training Notes

This document records the reproducible training path used for `3v1-obstacles-v0`.

## Current Status

The environment trains successfully with obstacle observations and a catch-distance curriculum.

Observed qualitative behavior after the `0.3` stage:

- The prey is still caught relatively quickly.
- The prey does not yet actively use obstacles as cover.
- Predator behavior is useful but somewhat aggressive near arena boundaries.
- Obstacle collision metrics are stable enough for this curriculum stage.

## Important Code State

The obstacle environment expects:

- `catch_distance = 0.3` for the current final curriculum stage.
- `num_obstacles = 4`.
- Predator observation space `84`.
- Prey observation space `34`.
- Centralized state space `118`.

Obstacle vectors are expressed in each agent's body frame. This keeps the transferred empty-arena policy compatible while giving agents egocentric obstacle information.

## Training Workflow

Run commands from the repository root:

```powershell
cd C:\RL\UAVPredatorPrey
$env:CONDA_PREFIX="C:\RL\env_isaaclab\Scripts"
```

Use Isaac Lab's launcher, not plain `python`, because the working environment is `env_isaaclab`:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=3v1-obstacles-v0 --algorithm=MAPPO --headless
```

## Curriculum Stages

### Stage 0: Transfer From Empty Arena

Use the transfer helper to expand an empty-arena `3v1` checkpoint to the obstacle observation/state dimensions:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\transfer_3v1_empty_to_obstacles.py logs\skrl\uav_3v1_direct\2026-03-16_15-19-16_mappo_torch\checkpoints\agent_72000.pt --output logs\skrl\uav_3v1_obstacles_direct\transfer_checkpoints\empty_2026-03-16_15-19-16_agent_72000_to_obstacles.pt
```

This checkpoint was the best transfer source found during testing.

### Stage 1: Train With `catch_distance = 0.5`

Set `catch_distance = 0.5`, then train from the transferred checkpoint:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=3v1-obstacles-v0 --algorithm=MAPPO --headless --checkpoint logs\skrl\uav_3v1_obstacles_direct\transfer_checkpoints\empty_2026-03-16_15-19-16_agent_72000_to_obstacles.pt
```

Result run:

```text
logs\skrl\uav_3v1_obstacles_direct\2026-05-01_20-21-12_mappo_torch
```

Use:

```text
checkpoints\best_agent.pt
```

Expected final/tail behavior:

```text
catch_rate:              about 0.92
predator_oob_rate:       about 0.04
prey_oob_rate:           about 0.04-0.05
obstacle_collision_rate: about 0.004-0.005
```

### Stage 2: Train With `catch_distance = 0.4`

Set `catch_distance = 0.4`, then continue from the Stage 1 `best_agent.pt`:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=3v1-obstacles-v0 --algorithm=MAPPO --headless --checkpoint logs\skrl\uav_3v1_obstacles_direct\2026-05-01_20-21-12_mappo_torch\checkpoints\best_agent.pt
```

Result run:

```text
logs\skrl\uav_3v1_obstacles_direct\2026-05-02_10-26-26_mappo_torch
```

Use:

```text
checkpoints\best_agent.pt
```

The best checkpoint was written around `432k` timesteps. It was preferable to the final `480k` checkpoint.

Expected behavior near best checkpoint:

```text
catch_rate:              about 0.86
predator_oob_rate:       about 0.07-0.08
prey_oob_rate:           about 0.08
obstacle_collision_rate: about 0.006
```

### Stage 3: Train With `catch_distance = 0.3`

Set `catch_distance = 0.3`, then continue from the Stage 2 `best_agent.pt`:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=3v1-obstacles-v0 --algorithm=MAPPO --headless --checkpoint logs\skrl\uav_3v1_obstacles_direct\2026-05-02_10-26-26_mappo_torch\checkpoints\best_agent.pt
```

Result run:

```text
logs\skrl\uav_3v1_obstacles_direct\2026-05-02_23-49-22_mappo_torch
```

Use:

```text
checkpoints\best_agent.pt
```

The best checkpoint was written around `432k` timesteps. It was slightly better balanced than `agent_480000.pt`.

Expected behavior near best checkpoint:

```text
catch_rate:              about 0.78
predator_oob_rate:       about 0.15
prey_oob_rate:           about 0.09
obstacle_collision_rate: about 0.008
closest_approach:        about 0.292
```

## Evaluation

Visual check:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\play.py --task=3v1-obstacles-v0 --algorithm=MAPPO --num_envs=10 --checkpoint logs\skrl\uav_3v1_obstacles_direct\2026-05-02_23-49-22_mappo_torch\checkpoints\best_agent.pt
```

If no checkpoint is passed, `play.py` will load the latest matching checkpoint automatically. Passing the checkpoint explicitly is safer when comparing curriculum stages.

## What To Commit

Commit code and documentation:

```text
source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/uavpredatorprey_3v1/uav_3v1_env.py
source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/uavpredatorprey_3v1_obstacles/uav_3v1_obstacles_env.py
source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/uavpredatorprey_3v1_obstacles/uav_3v1_obstacles_env_cfg.py
scripts/transfer_3v1_empty_to_obstacles.py
docs/training_3v1_obstacles.md
```

Do not commit `logs/` by default. The repository `.gitignore` excludes logs, which is usually correct.

If you want reproducible pretrained artifacts in the repository, add only selected `best_agent.pt` files via Git LFS and document exactly which curriculum stage each file belongs to. Do not add full run directories.

## Recommended Next Steps

1. Improve prey obstacle usage.
2. Add obstacle-cover metrics, such as line-of-sight blocked by obstacle or prey-nearest-obstacle distance during chase.
3. Reward prey for keeping an obstacle between itself and the nearest predator.
4. Penalize predator line-of-sight interception through obstacle regions if obstacle avoidance should matter more.
5. Consider a `catch_distance = 0.35` bridge if future `0.3` runs become unstable.
6. Add an evaluation script that runs deterministic episodes and reports catch rate, OOB rates, obstacle metrics, episode length, and closest approach without doing training.
