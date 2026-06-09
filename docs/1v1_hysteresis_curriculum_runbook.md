# 1v1 Hysteresis Curriculum Runbook

This note records the first stable 1v1 self-play curriculum setup. Its purpose
is reproducibility: which task, scripts, checkpoints, and decision rules were
used to get the current predator/prey behaviors.

## Task

- `1v1-survival-soft-oob-v0`
- Agent config: `skrl_mappo_finetune_cfg_entry_point`
- Algorithm: `MAPPO`

## Core Idea

Use a short outer-loop scheduler instead of one long uncontrolled self-play run.
Each phase trains for a fixed number of iterations, evaluates deterministically,
then chooses which side should learn next.

Decision rule:

- `catch_rate < 0.40`: train predator, freeze prey.
- `catch_rate > 0.60`: train prey, freeze predator.
- `0.40 <= catch_rate <= 0.60`: train both.
- Safety guards can override this if OOB or soft-arena metrics are too high.

This keeps the game near the edge of competence instead of letting one side
dominate for a long time.

## Scripts

- `scripts/skrl/train.py`
  - Supports `--freeze-agents predator|prey`.
  - Frozen agents use a no-op optimizer during the run.
- `scripts/skrl/evaluate.py`
  - Deterministic evaluation via mean actions.
  - Writes JSON summaries via `--json`.
- `scripts/skrl/hysteresis_curriculum.py`
  - Runs train/eval phases and chooses the next phase.
  - Optional old-opponent sampling via `--opponent-pool`.
- `scripts/compose_checkpoint.py`
  - Composes checkpoints from separate predator/prey sources.
- `scripts/reset_checkpoint.py`
  - Optional optimizer/log-std reset for clean phase starts.

## Useful Checkpoints

Predator examples:

- `C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-07_22-41-19_mappo_finetune_torch_finetune\checkpoints\best_agent.pt`
  - Repaired chase/intercept predator.
- `C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-08_19-01-14_mappo_finetune_torch_finetune\checkpoints\agent_48000.pt`
  - Strong intercept predator. Visually waits/trails and cuts into the prey path.

Prey examples:

- `C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-08_17-36-59_mappo_finetune_torch_finetune\checkpoints\agent_1200.pt`
  - Moderate evasive prey, deterministic catch rate about `0.66`.
- `C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-08_17-36-59_mappo_finetune_torch_finetune\checkpoints\agent_1800.pt`
  - Balanced evasive prey, deterministic catch rate about `0.39`.
- `C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-08_23-25-12_mappo_finetune_torch_finetune\checkpoints\agent_12000.pt`
  - Strong irregular/random-looking evasion prey.
- `C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-09_02-16-19_mappo_finetune_torch_finetune\checkpoints\agent_12000.pt`
  - Strong orbit/direction-change prey from pooled hysteresis training.

## First Hysteresis Run

Final checkpoint:

```text
C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-08_23-25-12_mappo_finetune_torch_finetune\checkpoints\agent_12000.pt
```

Scheduler log:

```text
C:\RL\UAVPredatorPrey\logs\curriculum\2026-06-08_19-39-44_hysteresis
```

Qualitative behavior:

- Prey used irregular, random-looking motion to break predator timing.
- Predator attempted anticipation/interception, but the final state was
  prey-favored.

## Pooled Hysteresis Run

Command shape:

```powershell
& "C:\RL\env_isaaclab\Scripts\python.exe" `
  "C:\RL\UAVPredatorPrey\scripts\skrl\hysteresis_curriculum.py" `
  --checkpoint "<start-checkpoint.pt>" `
  --opponent-pool "C:\RL\UAVPredatorPrey\scripts\skrl\opponent_pool_1v1.example.json" `
  --pool-prob 0.25 `
  --total-iterations 20000 `
  --phase-iterations 500 `
  --low 0.40 `
  --high 0.60 `
  --balanced-action both `
  --eval-num-envs 512 `
  --eval-episodes 512 `
  --train-num-envs 4096 `
  --seed 42
```

Final checkpoint:

```text
C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-06-09_02-16-19_mappo_finetune_torch_finetune\checkpoints\agent_12000.pt
```

Scheduler log:

```text
C:\RL\UAVPredatorPrey\logs\curriculum\2026-06-08_23-41-07_hysteresis
```

Final deterministic evaluation:

- `catch_rate`: about `0.223`
- `episode_length`: about `445`
- `predator_oob_rate`: about `0.029`
- `prey_oob_rate`: about `0.016`
- `predator_soft_arena_outside`: about `0.016`
- `prey_soft_arena_outside`: about `0.058`

Qualitative behavior:

- Prey primarily uses fast orbit escape.
- Prey occasionally changes direction or briefly jitters to break timing.
- Predator appears to anticipate and force prey movement, but the final state is
  still prey-favored.

## Visual Evaluation

```powershell
& "C:\RL\IsaacLab\isaaclab.bat" `
  -p "C:\RL\UAVPredatorPrey\scripts\skrl\evaluate.py" `
  --task 1v1-survival-soft-oob-v0 `
  --agent skrl_mappo_finetune_cfg_entry_point `
  --algorithm MAPPO `
  --num_envs 1 `
  --episodes 4 `
  --seed 42 `
  --checkpoint "<checkpoint.pt>"
```

## Current Next Step

The final pooled checkpoint is useful, safe, and prey-favored. It should be kept
as a prey pool member. For more 1v1 work, the next phase would train the
predator against this strong prey or continue pooled hysteresis with a slightly
higher chance of prey-pool sampling for predator phases.

The next curriculum expansion should likely be `2v1`, not obstacles yet.
