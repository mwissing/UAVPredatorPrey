# Predator-Prey Curriculum Learning Plan

This plan is the anchor for future training decisions. The goal is not to make
one run look better, but to build a policy stack whose behavior remains
explainable as task difficulty increases.

## Research-Guided Principles

1. Train one missing capability at a time.
   Do not combine harder opponents, obstacles, cover behavior, and self-play
   until the simpler prerequisite is measurable.

2. Use promotion gates, not intuition.
   A phase is complete only when evaluation metrics and videos agree.

3. Keep the opponent distribution controlled.
   If both sides change too quickly, the learning problem becomes nonstationary
   and the TensorBoard reward can become misleading.

4. Prefer capability curricula over reward chasing.
   If a skill is missing, simplify the environment until that skill has a clean
   signal, then transfer it upward.

5. Separate training reward from evaluation truth.
   The evaluation dashboard should track catch rate, OOB rates, episode length,
   closest approach, obstacle collisions, cover/shadow metrics, and videos.

## Promotion Gates

A checkpoint can move to the next phase only if all relevant gates hold over at
least 512 deterministic evaluation episodes and a short visual inspection:

- Predator flight gate: predator_oob_rate <= 0.10.
- Prey flight gate: prey_oob_rate <= 0.10.
- Stability gate: no obvious crash, height collapse, or boundary farming.
- Survival gate: episode_length improves without prey_oob_rate increasing.
- Obstacle gate: obstacle_collision_rate decreases without catch collapse.
- Transfer gate: previous easier task does not regress badly after fine-tuning.

## Phase 0: Diagnostics And Reproducibility

Goal: make experiments comparable.

- Use `evaluate.py` for every checkpoint comparison.
- Save JSON summaries in `logs/evaluations`.
- Compare at least current task plus one easier transfer task.
- Keep commits aligned with experiment boundaries.

Pass condition:

- We can answer what changed, which checkpoint was evaluated, and which metrics
  improved or regressed.

## Phase 1: Low-Level Flight Feasibility

Goal: prove that each vehicle can fly inside the arena without the game logic
dominating learning.

Tasks:

- Single UAV hover/arena checks if needed.
- Scripted prey escape diagnostics.
- Action scaling, height, OOB, and boundary sanity checks.

Pass condition:

- The prey can move fast enough to avoid capture in principle, and bounded
  flight is physically feasible.

## Phase 2: 1v1 Bounded Survival

Goal: teach the prey to survive without obstacles and without 3v1 pressure.

Primary task:

- `1v1-survival-easy-v0`

Why this phase exists:

- Our 3v1 survival experiments did not clearly improve prey behavior.
- Scripted escape avoided capture but went OOB, so the missing skill is bounded
  evasion, not raw speed.

Metrics to watch:

- catch_rate
- prey_oob_rate
- episode_length
- prey_distance_progress
- prey_boundary_progress
- prey_boundary_pressure

Pass condition:

- Prey reduces catch rate or increases episode length while keeping
  prey_oob_rate <= 0.10.

## Phase 3: 1v1 Competitive Fine-Tuning

Goal: make both agents competent without destabilizing prey survival.

Setup:

- Start from the Phase-2 checkpoint.
- Alternate short predator and prey updates only if one side dominates.
- Keep an older opponent checkpoint available for evaluation.

Pass condition:

- The prey remains bounded, and the predator can still close distance without
  relying on prey OOB.

## Phase 4: 3v1 No-Obstacle Survival

Goal: scale from one predator to coordinated pressure.

Setup:

- Transfer from the best 1v1 checkpoint only after Phase 2 passes.
- Start with weaker or more distant predators if needed.
- Increase pressure gradually.

Pass condition:

- 3v1 survival improves without returning to high prey OOB or predator OOB.

## Phase 5: Obstacles As Safety Constraints

Goal: add obstacle avoidance before using obstacles as cover.

Setup:

- Start with bridge/mid obstacle difficulty.
- Penalize collisions and proximity, but avoid rewards that make the predator
  refuse to chase.

Pass condition:

- Obstacle collision rate drops while catch behavior remains meaningful.

## Phase 6: Cover Emerges Or Is Lightly Shaped

Goal: let the prey exploit obstacles only after bounded survival works.

Setup:

- First evaluate whether cover use emerges from survival plus obstacles.
- Add cover/shadow shaping only if the policy ignores useful geometry.
- Avoid freezing the prey when the goal is to improve prey cover behavior.

Pass condition:

- prey_cover_gain_episode becomes nonnegative or cover rate improves without
  OOB/collision regressions.

## Phase 7: League And Robustness

Goal: avoid overfitting to one opponent snapshot or one obstacle distribution.

Setup:

- Evaluate against a small opponent checkpoint pool.
- Randomize initial distance, obstacle layouts, and seeds.
- Keep a champion checkpoint only if it passes all earlier transfer gates.

Pass condition:

- The policy performs acceptably across easy, bridge, mid, full, and cover
  variants, not just the training task.

## Default Next Action

If a phase stalls, do not immediately tune PPO hyperparameters. First inspect:

- observation correctness
- action scaling
- reward magnitudes
- termination reasons
- per-agent OOB/collision metrics
- deterministic videos
- whether the task is still too hard for the current capability
