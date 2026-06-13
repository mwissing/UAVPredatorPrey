# 3v1 Obstacles Training Notes

This document records the reproducible training path used for `3v1-obstacles-v0`.

## Current Status

The environment trains successfully with obstacle observations and a catch-distance curriculum.

Observed qualitative behavior after the `0.3` stage:

- The prey is still caught relatively quickly.
- The prey does not yet actively use obstacles as cover.
- Predator behavior is useful but somewhat aggressive near arena boundaries.
- Obstacle collision metrics are stable enough for this curriculum stage.

The next iteration adds explicit cover shaping while keeping observation/action shapes unchanged, so existing
checkpoints remain loadable. The cover-seeking reward prefers a safe ring around obstacles instead of rewarding
scraping the obstacle boundary.

## Important Code State

The obstacle environment expects:

- `catch_distance = 0.3` for the current final curriculum stage.
- `num_obstacles = 4`.
- Obstacles spawn in a reachable annulus from `1.3m` to `2.8m`.
- Obstacles keep at least `0.9m` distance from initial drone spawn positions.
- The first two obstacles are biased toward prey-predator approach lines with jitter, so cover is available early.
- Predator observation space `84`.
- Prey observation space `34`.
- Centralized state space `118`.

Predator obstacle vectors are expressed in each drone's body frame. The prey keeps the same extra 4D shape, but now
receives a body-frame shadow-target vector and score. This keeps transferred obstacle-policy checkpoints loadable while
making the prey's obstacle signal more directly useful for cover seeking.

The cover-shaped version also rewards the prey when an obstacle lies on the line segment between the prey and the closest predator. This is logged via:

```text
Reward/prey_cover
Reward/prey_cover_seek
Reward/prey_shadow
Reward/predator_cover_penalty
Metrics/prey_cover_rate
Metrics/prey_cover_score_episode
Metrics/prey_cover_when_threatened
Metrics/prey_shadow_target_score
Metrics/obstacle_spawn_min_agent_distance
Metrics/obstacle_spawn_mean_agent_distance
Metrics/obstacle_spawn_min_separation
Metrics/obstacle_spawn_mean_separation
```

Cover reward is gated off when the prey is inside the obstacle collision radius. Obstacle collision/proximity penalties
are intentionally stronger in this stage because cover-biased layouts put obstacles directly into approach corridors.

## Training Workflow

Run commands from the repository root:

```powershell
cd C:\RL\UAVPredatorPrey
$env:CONDA_PREFIX="C:\RL\env_isaaclab\Scripts"
$env:VIRTUAL_ENV="C:\RL\env_isaaclab"
$env:Path="C:\RL\env_isaaclab\Scripts;$env:Path"
```

Use Isaac Lab's launcher after setting the environment variables above, because the working environment is
`C:\RL\env_isaaclab`:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=3v1-obstacles-v0 --algorithm=MAPPO --headless
```

Before a training run, use the shape smoke test to verify MARL observation/action/state dimensions and the new
diagnostic log keys:

```powershell
C:\RL\env_isaaclab\Scripts\python.exe scripts\check_3v1_obstacles_shapes.py --task=3v1-obstacles-v0 --num_envs=8 --steps=2 --headless
```

If Isaac Sim is not available in the current shell, run the static contract fallback. This checks the configured action,
observation, state, and diagnostic-key contract, but it does not replace the runtime tensor check above:

```powershell
py scripts\check_3v1_obstacles_shapes.py --contract-only
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

### Stage 4: Cover-Shaped Fine-Tuning

After visual inspection, the prey was still caught quickly and did not actively use obstacles. Keep
`catch_distance = 0.3`, then fine-tune from the Stage 3 `best_agent.pt` with the cover-shaped reward,
reachable obstacle spawning, and initial spawn separation from drones:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=3v1-obstacles-v0 --algorithm=MAPPO --headless --checkpoint logs\skrl\uav_3v1_obstacles_direct\2026-05-02_23-49-22_mappo_torch\checkpoints\best_agent.pt
```

During this stage, watch these metrics:

```text
Metrics/prey_cover_rate
Metrics/prey_cover_when_threatened
Metrics/prey_cover_score_episode
Metrics/prey_shadow_target_score
Info / Metrics/catch_rate
Info / Metrics/predator_oob_rate
Info / Metrics/obstacle_collision_rate
```

Desired direction:

```text
prey_cover_rate:              up
prey_cover_when_threatened:   up
catch_rate:                   may drop initially, then should recover
predator_oob_rate:            should stay below about 0.2
obstacle_collision_rate:      should stay below about 0.02
```

If obstacle collision rate rises far above `0.02`, stop the run and increase obstacle-agent spawn separation or
move obstacles farther outward. A previous attempt with wider central spawning caused excessive obstacle contact
without increasing prey cover usage.

If prey cover usage remains low, prefer cover-biased obstacle spawning over simply increasing cover reward scale. Purely
random central obstacles previously increased obstacle contact more than strategy.

For cover-biased layouts, high obstacle collision rates early in fine-tuning are expected because the old policy learned
to fly through those newly relevant obstacle lines. The important sign is that collision rate trends downward while cover
usage stays above the original random-layout baseline.

## Evaluation

Visual check:

```powershell
C:\RL\IsaacLab\isaaclab.bat -p scripts\skrl\play.py --task=3v1-obstacles-v0 --algorithm=MAPPO --num_envs=10 --checkpoint logs\skrl\uav_3v1_obstacles_direct\2026-05-02_23-49-22_mappo_torch\checkpoints\best_agent.pt
```

If no checkpoint is passed, `play.py` will load the latest matching checkpoint automatically. Passing the checkpoint explicitly is safer when comparing curriculum stages.

## What To Commit

Commit code and documentation for this iteration:

```text
source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/uavpredatorprey_3v1_obstacles/uav_3v1_obstacles_env.py
source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/uavpredatorprey_3v1_obstacles/uav_3v1_obstacles_env_cfg.py
scripts/check_3v1_obstacles_shapes.py
scripts/reset_checkpoint.py
docs/training_3v1_obstacles.md
```

Do not commit `logs/` by default. The repository `.gitignore` excludes logs, which is usually correct.

If you want reproducible pretrained artifacts in the repository, add only selected `best_agent.pt` files via Git LFS and document exactly which curriculum stage each file belongs to. Do not add full run directories.

## Recommended Next Steps

1. Run the cover-shaped fine-tuning stage.
2. Compare cover metrics against the original `0.3` run.
3. Add an evaluation script that runs deterministic episodes and reports catch rate, OOB rates, obstacle metrics, episode length, closest approach, and cover usage without doing training.
4. If prey cover usage rises but catch rate collapses, reduce `prey_cover_reward_scale` or add a `0.35` bridge.
5. If prey still ignores obstacles, increase `prey_cover_seek_reward_scale` or spawn some obstacles even closer to the center.
