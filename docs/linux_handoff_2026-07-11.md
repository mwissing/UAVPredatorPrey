# Linux Migration Handoff - 2026-07-11

## Purpose

This document transfers the current reproducible 3v1 training state from the
Windows workstation to Linux. Code travels through Git. Checkpoints, pool
policies, TensorBoard evidence, and evaluation summaries travel in the external
artifact bundle described below.

## Git State

- Repository: `https://github.com/mwissing/UAVPredatorPrey.git`
- Branch: `dev-3v1-obstacles`
- Source commit: `094d8c7db43844fa614b6d08f006a31f50386f3f`
- Commit title: `Add pursuit diagnostics and low-progress training task`
- Verification before export: `45 passed`
- Four old `scripts/skrl/opponent_pool_*.json` files remained untracked and are
  deliberately excluded from the migration.

The source workstation used a parent-level `C:\RL\AGENTS.md`. A copy is added
to the repository root for Linux so Codex receives the same RL learning and
engineering instructions after cloning the repository.

## Source Runtime

- Isaac Lab: `2.3.2`
- Isaac Lab commit: `f4aa17f87e2e5db5484f0b5974918573e8918ce2`
- Isaac Sim: `5.1.0`
- Python: `3.11.9`
- PyTorch: `2.7.0+cu128`
- skrl: `1.4.3`
- Source GPU: NVIDIA GeForce RTX 5090

Install a clean Linux environment. Do not copy the Windows virtual environment.
Match Isaac Sim, Isaac Lab, Python, PyTorch, and skrl as closely as practical
before judging behavioral parity.

## Active Model And Task

- Preset: `3v1-attention-critic-prey-attention-large-gru`
- Task: `3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0`
- Algorithm: recurrent MAPPO
- Actor: predator and prey entity attention with GRU memory
- Critic: recurrent centralized entity-attention value model
- Actor hidden size: `256`
- Actor attention size: `128`
- Actor GRU hidden size: `256`
- Critic hidden size: `128`
- Critic attention size: `64`
- Critic GRU hidden size: `64`
- Sequence length / rollout length: `24`
- Learning rate: `1e-5`
- PPO ratio clip: `0.2`
- KL threshold: `0.05`
- Entropy coefficient: `0.005`
- Seed: `42`

The late-stage task changes only the predator distance-progress reward scale
from the older shaping value to `2.0`. It keeps the observation, action, model,
and checkpoint contracts compatible with the regular random-spawn task.

## Current Anchor

Portable checkpoint after importing the bundle:

```text
.pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt
```

Windows source:

```text
C:\RL\UAVPredatorPrey\logs\skrl\uav_3v1_direct\2026-07-11_16-15-56_mappo_attention_critic_prey_attention_large_gru_torch_attention_critic_prey_attention_large_gru\checkpoints\agent_115200.pt
```

Balanced deterministic evaluation over 512 environments/episodes:

| Metric | Value |
| --- | ---: |
| catch rate | 0.677734 |
| predator OOB rate | 0.000000 |
| prey OOB rate | 0.000000 |
| mean episode length | 318.441 |
| closest approach | 0.314487 m |
| predator closing speed | 0.269214 m/s |
| predator lateral relative speed | 4.021025 m/s |
| predator RMS speed | 3.769160 m/s |
| predator action near-limit fraction | 0.176806 |

Relative to the progress-scale-10 baseline, this checkpoint caught more prey
while using slightly lower RMS speed and fewer near-limit actions. That supports
keeping progress scale `2.0` for late-stage training.

## Opponent Pool

Portable pool after importing the bundle:

```text
.pretrained_checkpoints/linux_migration_2026-07-11/pool/opponent_pool_portable.json
```

The pool contains four predator and four prey entries. The export copies only
the eight active `checkpoint` files and rewrites those fields to repository-
relative POSIX paths. Historical Windows paths inside diagnostic metadata are
retained for provenance but are not used to load pool policies.

The source pool is:

```text
C:\RL\UAVPredatorPrey\logs\curriculum\2026-07-10_19-21-51_3v1_attention_critic_prey_attention_large_gru_hysteresis\opponent_pool_auto.json
```

## Artifact Bundle Layout

The USB export is `D:\uavpredatorprey_linux_migration_2026-07-11`:

```text
uavpredatorprey_linux_migration_2026-07-11/
  README.md
  MANIFEST.json
  SHA256SUMS.txt
  current/
    agent_115200.pt
  pool/
    opponent_pool_portable.json
    opponent_pool_windows_original.json
    checkpoints/
      predator_00.pt ... predator_03.pt
      prey_00.pt ... prey_03.pt
  provenance/
    current_run/
    curriculum_2026-07-10_19-21-51/
    evaluations/
    git/
```

## Restore On Linux

```bash
git clone https://github.com/mwissing/UAVPredatorPrey.git
cd UAVPredatorPrey
git checkout dev-3v1-obstacles
git rev-parse HEAD

export ISAACLAB_ROOT="$HOME/IsaacLab"

mkdir -p .pretrained_checkpoints
cp -a /media/$USER/<USB_LABEL>/uavpredatorprey_linux_migration_2026-07-11 \
  .pretrained_checkpoints/linux_migration_2026-07-11

cd .pretrained_checkpoints/linux_migration_2026-07-11
sha256sum -c SHA256SUMS.txt
cd ../..
```

Commit `094d8c7` already contains the repository-root `AGENTS.md`. After this
handoff document is committed and pushed, Linux should check out that newer
handoff commit. The artifact manifest records `094d8c7` as the code state used
during export.

## Linux Validation Order

1. Install the matching Isaac Sim and Isaac Lab versions.
2. Install this repository as an Isaac Lab extension.
3. Run the unit tests.
4. Run a one-environment visual play/evaluation to catch rendering or physics
   differences.
5. Run a 128-environment deterministic evaluation.
6. Run a short 100-300 iteration training smoke test.
7. Only then start a long hysteresis curriculum.

Unit tests:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pytest tests -q
```

Deterministic parity evaluation:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/skrl/evaluate.py \
  --headless \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0 \
  --agent skrl_mappo_attention_critic_prey_attention_large_gru_cfg_entry_point \
  --algorithm MAPPO \
  --num_envs 512 \
  --episodes 512 \
  --seed 42 \
  --checkpoint ".pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt"
```

Do not require bit-identical metrics across operating systems. Investigate a
large behavioral shift, recurrent-state failure, OOB increase, or a catch-rate
change well beyond normal seed/evaluation variance before continuing.

## Proposed Next Curriculum Run

Start this only after parity evaluation succeeds:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/skrl/hysteresis_curriculum.py \
  --checkpoint ".pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt" \
  --preset 3v1-attention-critic-prey-attention-large-gru \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0 \
  --total-iterations 38400 \
  --predator-phase-iterations 9600 \
  --prey-phase-iterations 4800 \
  --both-phase-iterations 2400 \
  --learning-rate 1e-5 \
  --kl-threshold 0.05 \
  --low 0.40 \
  --high 0.60 \
  --max-predator-oob 0.12 \
  --max-prey-oob 0.12 \
  --pool-prob 0.0 \
  --per-env-pool-prob 0.50 \
  --opponent-pool ".pretrained_checkpoints/linux_migration_2026-07-11/pool/opponent_pool_portable.json" \
  --per-env-pool-max-policies 4 \
  --auto-pool \
  --pool-sampling pfsp \
  --cross-play \
  --cross-play-max-opponents 4 \
  --train-num-envs 8192 \
  --eval-num-envs 512 \
  --eval-episodes 512 \
  --seed 42 \
  --record-phase-video \
  --phase-video-every 1 \
  --phase-video-length 500 \
  --phase-video-num-envs 1 \
  --phase-video-camera-eye 8 -8 12 \
  --phase-video-camera-target 0 0 6
```

If 8192 environments regress throughput or memory behavior on the Linux setup,
benchmark 4096 and 8192 with the existing scaling script before the long run.

## Current Interpretation

- The latest predator anchor is above the hysteresis `high=0.60` threshold
  against the current prey, with no OOB events in deterministic evaluation.
- The prior curriculum still exposed a hard prey pool policy, so cross-play
  robustness remains the meaningful bottleneck rather than basic flight.
- The next long run should use the portable elite/recent pool and conditional
  exposure logic, not discard the pool or return to high progress shaping.
- Preserve balanced-per-environment evaluation and recurrent state handling;
  both fixed earlier sources of misleading results.
