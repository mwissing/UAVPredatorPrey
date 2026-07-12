# 3v1 Hysteresis Curriculum On Linux

This runbook starts, validates, and monitors the current 3v1 MAPPO hysteresis
curriculum in the pinned Isaac Lab Docker runtime. It is the operational guide
for the migrated recurrent checkpoint; the older
[`1v1_hysteresis_curriculum_runbook.md`](1v1_hysteresis_curriculum_runbook.md)
is historical Windows/1v1 documentation.

## Validated Contract

| Setting | Value |
| --- | --- |
| Task | `3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0` |
| Agent | `skrl_mappo_attention_critic_prey_attention_large_gru_cfg_entry_point` |
| Algorithm | MAPPO |
| Start checkpoint | `.pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt` |
| Opponent pool | `.pretrained_checkpoints/linux_migration_2026-07-11/pool/opponent_pool_portable.json` |
| Seed | `42` |
| Hysteresis band | catch rate `0.40` to `0.60` |

The runtime versions and migration checksums are enforced by
`scripts/linux/verify_container_runtime.sh`.

## Curriculum Decision Rule

After every deterministic evaluation, the outer-loop scheduler chooses one
training phase:

- catch rate below `0.40`: train the predator and freeze the prey;
- catch rate above `0.60`: train the prey and freeze the predator;
- catch rate from `0.40` through `0.60`: train both roles;
- excessive OOB or soft-arena metrics override the catch-rate decision with a
  safety-repair phase.

One MAPPO update consumes the configured rollout length. For the current agent,
`100` curriculum iterations produce `2400` trainer timesteps because the
rollout length is `24`.

## 1. Start And Enter The Container

Run these commands on the host:

```bash
cd "$HOME/RL/IsaacLab"

python3 docker/container.py start base \
  --suffix uav51 \
  --files "$HOME/RL/UAVPredatorPrey/docker/isaaclab.override.yaml" \
  --env-files "$HOME/RL/UAVPredatorPrey/docker/.env"

python3 docker/container.py enter base \
  --suffix uav51 \
  --files "$HOME/RL/UAVPredatorPrey/docker/isaaclab.override.yaml" \
  --env-files "$HOME/RL/UAVPredatorPrey/docker/.env"
```

Inside the container:

```bash
cd /workspace/UAVPredatorPrey
export TERM=xterm-256color
```

The explicit `TERM` export is harmless on new containers and fixes older
running containers that inherited `TERM=dumb`.

X11 forwarding is only for interactive GUI play. Batch Isaac processes must
not inherit `DISPLAY` or `XAUTHORITY`: the validated host produced a Kit
`XOpenDisplay` segmentation fault when a headless evaluator inherited them.
Prefix every evaluation, training, certification, cross-play, and offscreen
video invocation with `env -u DISPLAY -u XAUTHORITY` as shown below. This
leaves X11 available in the shell for later interactive use.

## 2. Verify Before Training

```bash
./scripts/linux/verify_container_runtime.sh
/workspace/isaaclab/isaaclab.sh -p -m pytest tests -q
```

For a clean reproducibility certificate, also run:

```bash
env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/certify_baseline.py
```

Do not begin a long run if runtime verification, tests, checkpoint loading, or
the Windows parity verdict fails.

## 3. Evaluation-Only Scheduler Check

Every scheduler invocation must have a new output directory. The scheduler is
not an implicit resume mechanism and rejects non-empty directories.

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_low_progress_eval_only"
OUTPUT_DIR="/workspace/artifacts/isaac/curriculum_smoke/${RUN_ID}"

env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/hysteresis_curriculum.py \
  --checkpoint ".pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt" \
  --preset 3v1-attention-critic-prey-attention-large-gru \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0 \
  --total-iterations 0 \
  --eval-num-envs 128 \
  --eval-episodes 128 \
  --seed 42 \
  --output-dir "$OUTPUT_DIR"
```

Expected files:

```text
run_config.json
phase_000_start_eval.json
final_checkpoint.txt
```

The known seed-42 128-episode anchor has catch rate `0.6484375` with zero
predator and prey OOB. It should therefore request a prey phase when training is
enabled.

## 4. One-Phase Training Smoke

This isolates the scheduler, freeze logic, checkpoint discovery, and post-phase
evaluation. It intentionally disables pool sampling and video.

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_one_phase_100_seed42"
OUTPUT_DIR="/workspace/artifacts/isaac/curriculum_smoke/${RUN_ID}"

env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/hysteresis_curriculum.py \
  --checkpoint ".pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt" \
  --preset 3v1-attention-critic-prey-attention-large-gru \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0 \
  --total-iterations 100 \
  --predator-phase-iterations 100 \
  --prey-phase-iterations 100 \
  --both-phase-iterations 100 \
  --learning-rate 1e-5 \
  --kl-threshold 0.05 \
  --low 0.40 \
  --high 0.60 \
  --max-predator-oob 0.12 \
  --max-prey-oob 0.12 \
  --pool-prob 0.0 \
  --per-env-pool-prob 0.0 \
  --no-recent-pool \
  --train-num-envs 4096 \
  --eval-num-envs 128 \
  --eval-episodes 128 \
  --seed 42 \
  --no-phase-video-markers \
  --output-dir "$OUTPUT_DIR"
```

A successful smoke has:

- exit status `0`;
- one decision record in `history.jsonl` with
  `actual_phase_iterations: 100`;
- a new `agent_2400.pt` checkpoint under the host-mounted SKRL log tree;
- `phase_001_after_eval.json` with exactly 128 completed episodes;
- no frozen-agent fingerprint error;
- `final_checkpoint.txt` pointing to the new checkpoint.

For the expected prey phase, the predator policy/value/preprocessor state must
remain unchanged while the prey state changes.

## 5. Pool And Cross-Play Integration Smoke

This is recommended before a long Isaac league run, but it is not a blocker for
starting the separate JAX simulator. It exercises per-environment opponent
mixing, the evolving pool file, one cross-play opponent, and PFSP updates.

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_pool_crossplay_smoke"
OUTPUT_DIR="/workspace/artifacts/isaac/curriculum_smoke/${RUN_ID}"

env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/hysteresis_curriculum.py \
  --checkpoint ".pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt" \
  --preset 3v1-attention-critic-prey-attention-large-gru \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0 \
  --total-iterations 100 \
  --predator-phase-iterations 100 \
  --prey-phase-iterations 100 \
  --both-phase-iterations 100 \
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
  --cross-play-max-opponents 1 \
  --cross-play-num-envs 64 \
  --cross-play-episodes 64 \
  --train-num-envs 4096 \
  --eval-num-envs 128 \
  --eval-episodes 128 \
  --seed 42 \
  --no-phase-video-markers \
  --output-dir "$OUTPUT_DIR"
```

In addition to the one-phase outputs, expect `opponent_pool_auto.json`, composed
checkpoint evidence when required, and cross-play evaluation JSON.

This short smoke deliberately evaluates only one cross-play opponent. The
scheduler default requires at least two cross-play results before they may
influence a phase decision, so this command validates composition, evaluation,
PFSP metadata, and persistence but cannot test cross-play-driven phase
selection. Use `--cross-play-max-opponents 2` or more for that test. The long
curriculum below uses `4`.

## 6. Planned Long Curriculum

The migrated handoff proposes the following phase sizes after the pool
integration smoke passes:

```text
total:      38400 iterations
predator:    9600 iterations per predator phase
prey:        4800 iterations per prey phase
both:        2400 iterations per joint phase
training:    8192 environments, benchmark against 4096 first
evaluation:   512 environments and 512 completed episodes
```

The complete proposed command is:

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_3v1_pool_hysteresis"
OUTPUT_DIR="/workspace/artifacts/isaac/curriculum/${RUN_ID}"

env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/hysteresis_curriculum.py \
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
  --phase-video-camera-target 0 0 6 \
  --no-phase-video-markers \
  --output-dir "$OUTPUT_DIR"
```

Drone markers are disabled by default and must remain explicitly disabled in
documented commands. Benchmark `4096` against `8192` training environments
before committing to this long run.

## Outputs And Provenance

Each new scheduler directory contains some or all of:

```text
run_config.json                 exact argv, resolved arguments, Git state, and input hashes
history.jsonl                   phase decisions, metrics, pool events, and iteration counts
phase_000_start_eval.json       initial deterministic evaluation
phase_NNN_after_eval.json       deterministic evaluation after each phase
opponent_pool_auto.json         evolving pool when auto-pool or cross-play is enabled
composed/                       temporary/current-pair checkpoint compositions when needed
videos/                         optional phase videos
final_checkpoint.txt            final checkpoint after graceful scheduler completion
```

Training checkpoints and TensorBoard events live under:

```text
/workspace/UAVPredatorPrey/logs/skrl/uav_3v1_direct/
```

That container path is mounted from `$HOME/RL/artifacts/isaac/logs` on the
host. Evaluation and curriculum summaries are mounted under
`$HOME/RL/artifacts/isaac/`.

To retain console output without placing a file in the scheduler directory
before startup, enable pipeline failure propagation and write a sibling log:

```bash
set -o pipefail
# append `2>&1 | tee "${OUTPUT_DIR}.console.log"` to the scheduler command
STATUS=${PIPESTATUS[0]}
printf 'status=%s\noutput=%s\n' "$STATUS" "$OUTPUT_DIR" | \
  tee "${OUTPUT_DIR}.status.txt"
```

Capture `PIPESTATUS[0]` immediately after the pipeline. The sibling console log
contains scheduler output but not the shell's exit status; the separate status
file preserves it.

## Stopping And Recovery

- `Ctrl-C` stops the active process; wait for the subprocess to exit.
- `final_checkpoint.txt` is written only after graceful scheduler completion.
- The scheduler does not restore its phase counter, retry state, or RNG state.
- Never restart into the same output directory.
- For manual recovery, select the latest valid training checkpoint, preserve the
  interrupted directory as evidence, and begin a new scheduler invocation with
  a new output directory. If an evolving pool was used, pass the preserved
  `opponent_pool_auto.json` as the new `--opponent-pool`.

This is a new run from recovered weights and pool state, not a bit-exact resume.

### Extract A Recovery Checkpoint Safely

The pinned container does not provide `jq` or a `python3` executable on
`PATH`. Use the bundled Isaac Python binary directly to read the completed
phase JSON:

```bash
INT_DIR=/workspace/artifacts/isaac/curriculum_smoke/<interrupted_run>
ISAAC_PYTHON="${ISAACSIM_PATH:-/workspace/isaaclab/_isaac_sim}/kit/python/bin/python3"

RECOVERY_CKPT="$(
  "$ISAAC_PYTHON" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
    "$INT_DIR/phase_001_after_eval.json"
)"

printf 'recovery_checkpoint=%s\n' "$RECOVERY_CKPT"
test -n "$RECOVERY_CKPT"
test -f "$RECOVERY_CKPT"
sha256sum "$RECOVERY_CKPT"
```

Stop if any check fails. Never invoke the scheduler with an empty checkpoint:
an empty path resolves to the repository directory and later fails with
`IsADirectoryError`. After the checks pass, start the intended scheduler
command with `--checkpoint "$RECOVERY_CKPT"` and a fresh `--output-dir`.

## Verified Linux Smoke Evidence

The July 12 Linux artifact proved the following sequence:

| Phase | Decision | Catch before | Catch after | Predator OOB after | Prey OOB after |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | prey | 0.648438 | 0.601562 | 0.000000 | 0.000000 |
| 2 | prey | 0.601562 | 0.562500 | 0.000000 | 0.007812 |
| 3 | both | 0.562500 | 0.578125 | 0.000000 | 0.000000 |

Checkpoint fingerprints confirmed that the predator remained exactly unchanged
during both prey phases and that both roles changed during the joint phase.

The pasted terminal command described a 100-iteration run, but this artifact
contains 300 iterations. The old scheduler did not persist argv, so the cause
cannot be reconstructed. `run_config.json` and the non-empty-directory guard
were added afterward specifically to prevent this ambiguity.

A later clean run at Git commit `b60f5aa` validated the complete pool path over
three 100-update prey phases: exact predator freezing, 50% per-environment
pool mixing, three checkpoint compositions, three 64-episode cross-play
evaluations, PFSP metadata updates, recent-pool rotation, and three 1280x720
videos with explicit `--no-video-markers`. The run finalized normally after
300 updates.

The interruption procedure was also validated. `Ctrl-C` during phase 2 returned
status `130`; the completed phase-1 evaluation and checkpoint remained valid,
and the interrupted directory correctly had no `final_checkpoint.txt`. A fresh
10-update scheduler invocation started from that recorded checkpoint hash and
returned status `0` with a new final checkpoint. This validates manual weight
recovery, not bit-exact scheduler resume.
