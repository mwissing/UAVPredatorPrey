# Experiment Log

Dated record of experiments, results, and decisions. Each entry gives the
question, the result, the decision, and where the evidence lives. Paths under
`artifacts/` are relative to `$HOME/RL/` (outside this repository). Since
2026-10-07, every experiment directory contains a `PLAN.md` written before
training (design and decision rule) and a `RESULTS.md`.

Unless stated otherwise, evaluations are deterministic, 1024 environments and
episodes, seed 42, so all matchups share the same initial states. Catch rates
are those of the predator (higher = better predator, lower = better prey).

## 2026-10-04: July 19 diagnosis and return to the archived recipe

- Question: why did the catch rate drop after the July 19 joint update?
- Result (256 episodes, paired 2x2 role swap): the drop came from the prey
  alone (prey effect 0.34, predator effect 0.004). Against the five reference
  prey, the scratch-lineage predator caught 0.098 of episodes, against 0.854
  for the Windows anchor.
- Decisions: reproduce the Windows anchor recipe (FP32, LR 1e-5, log-std floor
  -2, KL 0.05). Restore `low-progress-v0` to the legacy reward and move the
  6 m/s speed margin to its own task id. Add the catch-rate band guard. The
  archived-recipe smoke reproduced the July 12 smoke bit-exactly (`ec2d939`).
- Evidence: `artifacts/isaac/diagnostics/2026-10-04_phase5_role_swap_and_windows_reference/`.

## 2026-10-05 and 10-06: Archived-recipe long run

- Part 1 (prey, predator, prey phases): catch 0.285, 0.621, 0.432. The band
  guard carried the other role's pool share into the switched phase (fixed in
  `29c282f`).
- Part 2 (8 joint phases): catch 0.49-0.60, OOB at most 1%. The predator's
  reference mean fell to 0.596 (Windows anchor 0.854), while the prey conceded
  0.252.
- Diagnosis: forgetting. PFSP pushed beaten prey to the weight floor, top-4
  truncation dropped them, and weights were refreshed only after single-role
  phases.
- Evidence: `artifacts/isaac/curriculum/2026-10-05_*`,
  `artifacts/isaac/diagnostics/reference_matrix/`.

## 2026-10-06: Predator rehearsal, toward_center, and anchor v2

- Rehearsal (predator-only, 2400 updates, 75% pool of the five reference
  prey): reference mean 0.596 -> 0.828, but 0.535 -> 0.402 against its own
  prey (a trade-off).
- `toward_center` scheduler (`1d907ca`, no joint phases), 6 x 1200 predator
  updates: reference mean 0.873 (Windows anchor 0.854). Frozen as anchor v2.
- Evidence: `artifacts/isaac/experiments/2026-10-06_predator_rehearsal_pool/`,
  [anchor v2 doc](anchor_v2_2026-10-06.md).

## 2026-10-07: League, escape analysis, and horizon test

- 9x9 league (Bradley-Terry fit): anchor v2 is strongest in both roles
  (predator +0.93, prey +0.47 logits relative to the Windows anchor prey). The
  game is strongly non-transitive.
- Horizon test (anchor v2 pair): catch 0.486 / 0.724 / 0.854 at 10 / 20 / 30 s
  with a constant hazard of about 0.06/s, so the policies are behaviorally
  length-independent.
- Evidence: `artifacts/isaac/diagnostics/league_2026-10-07/RESULTS.md`.

## 2026-10-07: Time-limit bootstrapping A/B (pre-registered)

- Question: should time-outs bootstrap with `gamma * V(s)` instead of acting
  as terminal states?
- Design: prey-only from anchor v2, 2400 updates, the flag as the only
  difference.
- Result: own matchup 0.367 (bootstrap) vs 0.387, which is equal within the
  ±0.03 margin. Beyond the training horizon the bootstrapped prey held up
  better: 0.597 vs 0.633 at 20 s and 0.739 vs 0.789 at 30 s. Without
  bootstrapping, the critic's explained variance collapsed from about 0.93 to
  0.21 at synchronized time-outs.
- Decision: adopt. It is the preset default since `837e0aa`, and
  `--no-time-limit-bootstrap` reproduces the archived recipe bit-exactly
  (verified, `0afa89e`).
- Evidence: `artifacts/isaac/experiments/2026-10-07_time_limit_bootstrap_ab/`.

## 2026-10-07: Value normalizer measurement

- Question: the critics' running return normalizers have absorbed about 1e11
  samples. Do their statistics still match the return scale?
- Result: raw return std is about 45 (prey) and 31 (predator) against
  normalizer std 14,300 and 634, so the normalized targets are squashed by
  about 300x and 20x. The critics still explain 72-87% of the return
  variance.
- Decision: not a bottleneck. Keep the statistics fixed in comparisons,
  because recalibration would change the critic's effective learning rate.
  The new diagnostics (`77ca941`) were shown not to change learning: all 134
  logged training scalars of the smoke were reproduced bit-exactly.
- Evidence: `artifacts/isaac/experiments/2026-10-07_value_normalizer_calibration/`.

## 2026-10-07: Predator pursuit-shaping A/B (pre-registered)

- Question: does the per-drone distance-progress reward keep the predators in
  pack pursuit?
- Design: anchor v2 predator against the arm B prey from the bootstrap A/B,
  predator-only, 2400 updates, shaping 2.0 vs 0.
- Result: 0.533 vs 0.555 against the training prey, held-out mean 0.787 vs
  0.742, and unchanged geometry.
- Decision: keep 2.0.
- Evidence: `artifacts/isaac/experiments/2026-10-07_predator_progress_shaping_ab/`.

## 2026-10-07: Predator exploration floor (pre-registered)

- Question: the predator's action std sat at its floor (e^-2) for whole runs.
  Does a wider floor (-1.5, predator only, `34134a3`) find a different
  strategy?
- Result: 0.578 vs 0.533 against the training prey, but the held-out mean fell
  from 0.787 to 0.719, and the geometry was unchanged.
- Decision: keep -2.0. More noise produced faster specialization, not a new
  strategy.
- Strategy lens corrected (`artifacts/isaac/diagnostics/boundary_analysis.py`):
  in 3D three predators can only corner the prey against the arena wall. The
  prey lives near the wall (median 3.9 m from the center, wall at 5 m), and
  about half of all catches happen within 1 m of it. Enclosure on the wall
  occurs in about 19% of the last second before catches vs 13% in escapes,
  and it was unchanged by both predator interventions.
- Evidence: `artifacts/isaac/experiments/2026-10-07_predator_exploration_floor/`.

## 2026-10-07: Prey agility curriculum (pre-registered)

- Question: does a more agile prey (thrust-to-weight 2.5 instead of 2.2,
  rewards unchanged) make cornering against the wall more frequent?
- Design: stage 1 adapted the prey at 2.5. The gate passed: the anchor v2
  predator still caught 0.198 of episodes, and prey OOB was 0.002. Stage 2
  trained the predator against the adapted prey only.
- Result: the predator improved strongly against the agile prey
  (0.198 -> 0.476), but wall enclosure in the last second before catches fell
  (0.193 -> 0.156). At T/W 2.2 enclosure stayed at the control's level (0.182
  vs 0.190).
- Decision: no coordination gain; this form of option 2 does not create the
  cornering strategy.
- Exploratory: the predators learned better individual interception. They
  closed the distance far better (closest approach in escapes 0.64 -> 0.45 m)
  and steered more toward the prey's heading (lead index 0.13 -> 0.23,
  `artifacts/isaac/diagnostics/pursuit_geometry.py`) without flying faster.
  At T/W 2.2 the stage-2 predator caught the arm B prey more often (0.621)
  than the control trained against it (0.533). This is confounded by the
  missing opponent pool and the shared prey lineage. Without rehearsal the
  reference-prey mean fell to 0.570.
- Evidence: `artifacts/isaac/experiments/2026-10-07_prey_agility_curriculum/`.

## 2026-10-07: Training-seed noise floor

- Question: how much do results move when the same training run is repeated
  with another seed?
- Design: arm A of the shaping A/B (seed 42) retrained with seeds 43 and 44;
  identical configuration and evaluation.
- Result: SD of a single run is 0.027 on the own matchup (seeds: 0.533, 0.567,
  0.587), 0.003 on the reference mean, 0.015 on the held-out mean, 0.016 on
  wall enclosure, and 0.009 on the lead index.
- Re-check: all pre-registered decisions stand. The held-out losses of the
  shaping and exploration-floor treatments are real (z -2.0 and -3.1). The
  exploration floor's own-matchup gain is within noise (z +1.2). The agility
  transfer gain is borderline (z +2.3) and confounded.
- Lessons: seed 42 was the lowest of the three on the own matchup, so compare
  against multi-seed control means. A/B margins must be at least 0.08 on the
  own matchup (or use 3 seeds per arm). Forgetting of the old reference prey is
  systematic (0.878 -> 0.803 in all seeds, SD 0.003), so the 50% pool with
  PFSP weight floor 0.1 does not protect them.
- Evidence: `artifacts/isaac/experiments/2026-10-07_training_seed_noise/`.

## 2026-10-07/08: Rehearsal pre-test and anchor v3 (pre-registered, overnight)

- Pre-test: one predator arm with a 75% pool and PFSP floor 0.3 removed the
  forgetting (reference mean 0.882 vs 0.803), but cost 0.16 against the
  current opponent (0.402 vs 0.562). By the rule, the run used a 60% pool with
  floor 0.3.
- Anchor v3 run: `toward_center`, 12 x 1200 updates from the anchor v2
  predator plus the bootstrap-trained arm B prey (9 predator, 3 prey phases).
- Result: predator 0.367 -> 0.562 against the start prey; held-out mean
  0.691 -> 0.812; reference mean 0.869 (guardrail 0.868 held). The prey
  conceded less on every measure (0.312 to the anchor v2 predator, reference
  0.164, held-out 0.295), but missed its strict criterion (0.287).
  League: anchor v3 is strongest in both roles (predator +1.13, prey +1.09
  logits; anchor v2 +0.81 / +0.50).
- Decision: accept anchor v3 as the new reference. It is frozen as
  `.pretrained_checkpoints/anchor_v3_2026-10-08`, and the bundled copy
  reproduced the evaluation bit-exactly ([doc](anchor_v3_2026-10-08.md)).
- Strategy against the same prey: as strong as a single training phase (0.562
  vs the 3-seed mean 0.562), with much less forgetting, less wall enclosure
  (0.112 vs about 0.17), and slightly more lead pursuit. Coordination still
  did not emerge.
- Evidence: `artifacts/isaac/experiments/2026-10-07_anchor_v3_overnight/`,
  `artifacts/isaac/diagnostics/league_2026-10-08/`.

