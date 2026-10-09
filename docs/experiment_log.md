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


## 2026-10-08: Obstacles (Stage 8): implementation and first training

- Implementation: four vertical pillars with analytic collisions (touching a
  pillar is a crash), per-drone body-frame pillar features, and
  zero-initialized obstacle branches in both actors and the critic. Expanded
  checkpoints are function-preserving (`scripts/expand_checkpoint_obstacles.py`).
  Commits 2fd33e6, 93b8602, 9e28bbf (obstacle-branch LR multiplier, 30 in the
  obstacle config, d2f5fb2), and 5a82556 (soft keep-out cost like the soft
  arena wall). Checks: the obstacle-free path is bit-exact, anchor v3 with
  parked pillars reproduces 0.4609375, and the live feature errors are about
  1e-7.
- Obstacle-blind start (anchor v3 with four pillars): catch 0.107, predator
  crash 0.583, prey crash 0.485, episodes 2.2 s.
- First training: after three restarts (sequential safety repair too slow,
  the branch too slow at LR 1e-5, crashes too sparse a signal), a long joint
  run of 15,000 updates at 4096 envs (8192 runs out of GPU memory).
- Result: per second of flight, crash hazards fell 37% (predator) and 35%
  (prey), and catch reached 0.52. Per episode, crashes stay at 0.50 / 0.42, so
  the criterion (< 0.15) was missed. Zeroing the trained obstacle branch does
  not raise crashes in either role, so the crash reductions are blind. The
  prey does use its pillar observations, but to evade. Its branch takes it
  among the pillars, where it is caught less (catch hazard -23%) but crashes
  more (+46%). This is the first pillar-related strategy. The predator does
  not use its branch and chases straight through pillars. There is no
  trapping, hiding, or luring. With pillars parked 50 m away, the trained branch drives the prey
  out of the arena (OOB 0.90; branch zeroed 0.002).
- Predator diagnostic (pre-registered, 600 predator-only updates each, against
  the start prey): exploration floor -1.0 (0.541), crash cost x3 (0.531), and
  x10 (0.549) do not beat the control (0.555). Both levers together do
  (0.482, no catch loss), and a pre-registered replicate with training seed
  43 confirmed it (0.486 vs control 0.564). Zeroing the branch changes
  crashes by at most 0.03 in every arm and seed, so even this gain is not
  yet clearly perception-driven.
- Lessons: the incentives barely reward avoidance. For the prey, a crash
  costs the same as being caught; for the predator, one lost drone costs a
  third of a catch (the team reward is a mean, and only the catch bonus is
  scaled by team size). A higher penalty does not buy exploration. Absent
  entities need a mask, not a far-away position. Ablate a new pathway (zero
  its output) before crediting it with a behavior change.
- Next: crash price plus exploration for both roles, started 18:39 with user
  approval (`artifacts/isaac/experiments/2026-10-08_obstacles_crash_price_exploration/`,
  pre-registered against this run as control). Later: a Lagrangian crash
  constraint and a presence mask for absent pillars.
- Evidence: `artifacts/isaac/experiments/2026-10-08_obstacles_first_training/`
  (PLAN.md, RESULTS.md, predator_diag/, curve/, final_eval/, videos/).

## 2026-10-08: Crash price plus exploration (pre-registered, evening)

- Question: does a higher crash price for both roles, plus more predator
  exploration, lower crash hazards against fixed opponents compared with the
  first long run? And do the agents then use their pillar observations?
- Design: the long run repeated (same start, seed, and 15,000 updates) with
  three overrides:
  - predator crash cost -600 per drone, so one lost drone costs the team one
    catch;
  - predator log-std floor -1.0;
  - prey crash cost -600, three times as bad as being caught.
- Result, predator: crash hazard against the start prey 0.57x the control
  (0.183 vs 0.321/s), with the catch rate held. Zeroing its obstacle branch
  raises its crash hazard by 42%: this is the first perception-based
  avoidance. Its pursuit also improved: lead index 0.395 (control 0.283),
  wall enclosure before catches 0.165 (0.082).
- Result, prey: crash hazard 0.42x, but the start predator catches it 2.9x
  faster. It halved its speed and left the pillar zone, which is blind
  avoidance. Its guard failed.
- Decision: adopt the predator settings for obstacle training, but not the
  prey's -600 (its price is under test in `prey_diag/`). In open space,
  parked pillars now break the predator's branch (OOB 0.775), so a presence
  mask is needed.
- Lesson: a crash price teaches perception only where no blind shortcut
  exists. The prey could simply slow down; the predator could not, because
  slowing down costs catches.
- Prey dose-response (pre-registered, prey-only, 1200 updates against the
  frozen new predator): crash prices of -200, -350, and -600 act as a speed
  dial (4.5 / 3.8 / 3.2 m/s). Crashes go down, catches go up, and the
  prey's obstacle branch plays no role. The pre-registered rule picks -600,
  but the long run shows that price makes the prey passive, so -350 is
  recommended as the compromise (user decision).
- Evidence: `artifacts/isaac/experiments/2026-10-08_obstacles_crash_price_exploration/`.

## 2026-10-09: Prey skill stage against scripted ghost pursuers (pre-registered)

- Idea (user): learn the flying skill first, then the game. Implementation:
  "ghost" predators (918c401). They are kinematic, scripted lead pursuers
  (leads 0, 0.25, and 0.5 s), speed and acceleration limited, with no
  colliders. They never crash and fly through pillars, so pillars give the
  prey no cover and slowing down gets it caught. Off by default; with the
  flag off, existing evaluations reproduce bit-exactly.
- Design: prey-only, 9600 updates from the pre-treatment prey. G trained
  against 3.5 m/s ghosts (speed set by a pre-registered calibration). The
  control K trained against the frozen predator from the 2026-10-08 evening
  run.
- Result: G passes all four criteria; K fails the perception criterion.
  - G: crash hazard against the start predator 0.102/s (from 0.272).
    Zeroing the obstacle branch doubles it (ratio 2.08). Median speed
    5.2 m/s; catch hazard 0.115/s (from 0.134).
  - K: 0.160/s, ratio 0.92 (blind), 4.27 m/s.
  - G's perception emerged between 3840 and 5760 updates. It also evades the
    trained predator far better (catch hazard 0.067 vs 0.175/s).
- Caveat: G's prey increasingly circles the pillar-free outer belt. Its time
  in the pillar zone against the start predator fell 62% -> 43%. A second
  version places pillars up to the wall and draws ghost speeds from
  3.5-6.5 m/s per episode (b250977), with a layout-only control.
- Lesson: perception-based avoidance appears only when no blind shortcut is
  left. Earlier shortcuts were slowing down, weaving at a crash cost, and
  moving outward; here the belt still was one.
- Evidence: `artifacts/isaac/experiments/2026-10-09_prey_ghost_skill_stage/`;
  v2 in `artifacts/isaac/experiments/2026-10-09_prey_ghost_stage_v2/`.
- v2 result (pre-registered: neither arm uses perception):
  - With pillars up to the wall and ghost speeds of 3.5-6.5 m/s, the prey
    became a fast, excellent evader (catch hazard 0.054/s, 5.5 m/s) and no
    longer uses the belt. But its crashes barely fell (0.240/s, zeroing
    ratio 0.99).
  - The layout-only control reduced crashes blindly.
  - Explanation: for the prey, a crash costs the same as being caught, so
    under fast ghosts it escapes at speed and accepts crashes.
  - v3 (user's suggestion, running): the same with a prey crash cost of
    -300 (1.5x being caught).
- v3 result (pre-registered: no perception):
  - Against ghosts, the higher crash price works as intended: 38% fewer
    crashes, 2.3x more catches, so the prey no longer takes the crash when
    about to be caught.
  - But no perception emerged (ratio 1.05), and against the start predator
    crashes fell only 7%.
  - Reading: perception appeared only in v1, where evasion was easy. v1
    had slow ghosts and the pillar-free belt, so the two causes are
    confounded.
  - Correction (user's point, confirmed by the overnight cycle 0): without
    the belt, v1's prey crashes about twice as often (0.196 vs 0.102/s),
    and its ablation ratio drops from 2.08 to 1.28. Much of its safety came
    from the belt, but some genuine perception transfers. Next candidate: a
    curriculum from easy to hard evasion.
  - The overnight self-play starts from v1's prey (pre-registered rule).
