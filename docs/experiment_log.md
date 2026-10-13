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

## 2026-10-09/10: Overnight self-play with ghost rehearsal (pre-registered)

- Design: the predator from the crash-price-plus-exploration run plus v1's
  ghost-trained prey, in the full layout (pillars up to the wall). 6 cycles
  of 2400 joint updates plus 600 prey ghost updates (rehearsal). Prey crash
  -300; predator -600 with log-std floor -1.0.
- Result: criterion 1 fails (the prey loses perception); speed and balance
  pass.
  - Own matchup: catch rate 0.16 -> 0.49; predator crash hazard
    0.173 -> 0.085/s; prey 0.307 -> 0.223/s.
  - The predator avoids pillars by perception: ablation ratio 1.41 against
    the start prey, +52% crashes without its branch in its own matchup.
    Its interception (lead index 0.06 -> 0.27) and wall enclosure
    (0.10 -> 0.17) also improved.
  - The prey lost its perception in the first cycle (ratio 1.28 -> 1.00 ->
    0.71). It weaves among pillars again (77% of the time in the pillar
    zone), and its branch serves evasion, not avoidance.
- Reading: in self-play, the prey's incentives favor evasion over safety
  even at a crash cost of 1.5x a catch. Fast-ghost rehearsal did not protect
  the skill.
- Missing 2x2 cell: slow ghosts in the full layout. It would separate the
  ghost-speed effect from the belt effect.
- Evidence: `artifacts/isaac/experiments/2026-10-09_overnight_selfplay_ghost_rehearsal/`
  (RESULTS.md, summary.txt, videos/final_camera_D/).

## 2026-10-10: Prey ghost stage v4, slow ghosts in the full layout (pre-registered)

- Fills the 2x2 of ghost speed x layout. G4 is G2 with a constant ghost
  speed of 3.5 m/s; pillars go up to the wall, so there is no belt.
- Result: it formally passes all four criteria, but the perception signal
  is marginal.
  - Crash hazard against the start predator 0.272 -> 0.174/s (G2: 0.240).
  - Catch hazard 0.046/s; speed 5.0 m/s.
  - Ablation ratio 1.24 in the standard layout (1.10 one checkpoint
    earlier) and 0.93 in its own full layout.
- Reading: slow ghosts lower crashes and let perception grow slowly. v1's
  strong effect (2.08) needed the belt as well. The ablation ratio also
  depends on pillar density.
- Next candidates: a longer slow-ghost stage, or more pillars in the full
  layout, before a slow-to-fast curriculum.
- Evidence: `artifacts/isaac/experiments/2026-10-10_prey_ghost_stage_v4_slow_full_layout/`.

## 2026-10-10: Obstacle reference predator frozen

- The overnight run's predator becomes the obstacle reference predator
  ([doc](obstacle_predator_2026-10-10.md)). Read-only bundle
  `.pretrained_checkpoints/obstacle_predator_2026-10-10/`, checkpoint sha256
  `470c11e6...`. The bundled copy reproduces its evaluation bit-exactly.
- Ablation ratio 1.41 against the start prey and 1.52 in its own matchup.
  Crash hazard 0.100/s against the start prey; lead index 0.27; wall
  enclosure 0.165. The prey role of the checkpoint is not a reference.

## 2026-10-10: Presence mask for absent pillars

- Problem: absent pillars are parked 50 m away, and trained obstacle branches
  react to those out-of-range features. In open space the reference pair left
  the arena (predator OOB 0.733, prey 0.725, catch 0.052).
- Fix (72b877a): the agent also passes raw states, and the models treat a
  pillar as present only if its raw surface-distance feature is below 5.0
  (25 m). Absent pillars get no attention; without any present pillar the
  obstacle term is exactly zero.
- Checks:
  - with all pillars present, a stored evaluation reproduces bit-exactly;
  - in open space the masked pair equals the pair with the obstacle branch
    zeroed bit-exactly (catch 0.746, predator OOB 0.007, prey OOB 0.001);
  - 4 new unit tests.

## 2026-10-10: Prey ghost stage v5, slow ghosts continued (pre-registered)

- G4 continued for another 9600 updates with identical settings.
- Result: the pre-registered decision is a plateau.
  - Crash hazard against the start predator stays at 0.17-0.20/s.
  - The ablation ratio fluctuates around 1.2 in the standard layout (last
    two checkpoints 1.18 / 1.22) and about 1.0 in the full layout.
  - The prey stays fast (4.8 m/s) and evasive (catch hazard 0.064/s).
- Reading: four pillars spread up to the wall are met too rarely to drive
  perception-based avoidance. Next candidate: higher pillar density, using
  the presence mask to expand checkpoints from 4 to 8 pillar slots without
  changing their function.
- Evidence: `artifacts/isaac/experiments/2026-10-10_prey_ghost_stage_v5_slow_continued/`.

## 2026-10-10: Crashed-predator placeholder (found by the user)

- Bug: a crashed predator appears in the prey's observation as a placeholder
  10 m straight ahead with zero relative velocity (likewise for crashed
  teammates in predator observations). Prey that never met it in training
  leave the arena.
- Causal check, evaluation only on identical episodes: an opt-in mask
  (7c816d8, off by default; anchor v3 and the obstacle reference stay
  bit-exact) cuts exits after predator crashes.
  - v1 ghost prey vs the start predator: 78% -> 1.6%.
  - Start prey vs the run-T predator: 62% -> 0%.
  - Overnight prey: 3% -> 0%.
- Rescoring the ghost-stage prey with the mask:
  - The perception conclusions hold (v1 G ratio 2.54; v2-v5 at most about
    1.2).
  - Crash hazards against the crash-prone start predator were biased low.
    v1 G crashes as often as its control (0.147/s each) and is caught more.
    The ghost-only prey (v2-v5) crash 2-3x as often once predators crash.
- Next: use the masks in evaluations against crash-prone opponents
  (re-baseline), and decide whether new training enables them.
- Evidence: `artifacts/isaac/experiments/2026-10-10_crashed_predator_mask_check/`.

## 2026-10-10: Prey ghost stage v6, 8 pillars (pre-registered, masked protocol)

- Design: G5's prey, expanded to 8 pillar slots (c8518bc), trained for 9600
  updates against slow ghosts with 8 pillars up to the wall. Evaluations from
  now on use both crashed-drone masks.
- Pre-registered decision: density alone does not create perception.
  Criterion 1 fails: the ablation ratio in the dense layout is 1.20 / 1.06
  at the last two checkpoints. The other four criteria pass.
- On the standard 4-pillar yardstick, G6 is the best prey so far:
  - crash hazard against the start predator 0.324 -> 0.118/s;
  - ablation ratio 1.13 -> 2.11, growing steadily;
  - catch hazard unchanged (0.072/s); speed 4.5 m/s.
  - For comparison, v1 G: 0.147 / 2.54 / 0.217; K: 0.147 / 1.14 / 0.066.
- Open question: in the dense layout itself crashes fell by a third
  (0.584 -> 0.389/s) but are hardly sensitive to the pillar observations.
- Crash analysis:
  - The dense crashes are not forced. 94% are head-on flights into the hit
    pillar at about 4.7 m/s; few happen with a pursuer close or in clutter.
  - The branch prevents half of these crashes with 4 pillars and almost
    none with 8.
  - Next hypothesis: attention dilution over 8 pillars. Test: record the
    attention weights live.
- Attention probe (evaluate.py unchanged plus a recording hook; the runs
  reproduce exactly):
  - The prey's obstacle attention is one-hot (saturated softmax, entropy 0).
  - The attended pillar is unrelated to danger: the later-hit pillar at
    chance level, the nearest pillar below chance. It switches 3-4 times per
    second.
  - The perception bottleneck is therefore architectural, not dilution.
    Candidate fixes: max-pooling over pillars, a distance prior, or
    normalized multi-head attention.
- Evidence: `artifacts/isaac/experiments/2026-10-10_prey_ghost_stage_v6_dense8/`.

## 2026-10-10: Multi-head obstacle attention, prey ghost stage v7 (pre-registered)

- Direction chosen by the user: the most advanced long-term solution, a
  modern multi-head attention block (option 3 of the probe's candidates).
- Code (cbb8753): `ObstacleMultiHeadAttention`, enabled with
  `+agent.models.policy.obstacle_attention_heads=4`. It is added next to the
  single-head path with a zero-initialized output:
  - 4 heads, layer-normalized pillar tokens;
  - cosine scores times a learnable temperature (start 5, cap 30), so the
    scores stay bounded;
  - the presence mask.
- Checks, all bit-exact live: block off (reference pair) and block on (G6)
  reproduce their evaluations. A 48-update smoke run shows the prey's block
  learning while the frozen predator's block stays exactly zero.
- Untrained block on G6 (step-0 probe): soft attention (entropy 0.95-0.99).
  One head already favors the later-hit pillar (weight 0.205 vs chance
  0.125), from its random initialization alone.
- Design: G6's prey with the block continues for 9600 updates in the dense
  ghost stage, seed 45. There is no parallel control arm, because two runs do
  not fit in GPU memory (about 15 GB each). The block's share is measured
  within the run by zeroing only the block.
- Decision criteria in the plan:
  - dense ablation ratio >= 1.2 at the last two checkpoints;
  - block-only ratio >= 1.2;
  - dense crash hazard <= 0.8x G6;
  - guards on the standard yardstick and on speed.
- Evidence: `artifacts/isaac/experiments/2026-10-10_prey_ghost_stage_v7_mha/`.
- Result: **the block does not fix dense perception.** Criteria 1, 3, 4,
  and 5 fail; 2 (block-only ratio 1.20) and 6 (speed) pass. The result
  splits in two:
  - Mechanism, success: two heads put 0.49 and 0.39 of their weight on the
    pillar about to be hit (chance 0.125; untrained block 0.205). The old
    one-hot path attends away from it (0.003); the paths have co-adapted.
  - Behavior, failure: the training return peaked at 1920 updates (-52.8)
    and then fell steadily to -82.7, while G6's run improved under the same
    setup. The prey moved into the pillar zone (79% of the time vs 57%) and
    flew faster (mean speed in training 4.9 -> 6.0 m/s).
  - At 9600 updates, against the start predator on the 4-pillar yardstick:
    crash hazard 0.118 -> 0.201/s, catch hazard 0.072 -> 0.235/s.
- Candidate causes (untested):
  - the block learns 30x faster than the trunk;
  - the critic has no block and may misjudge the interior flight;
  - continued training from G6 would degrade anyway.
- The pre-registered next step was to test the perception pathway in
  isolation. It is deferred, because the probe shows the pathway works. The
  open problem is the training dynamics (v8).

## 2026-10-10: Crash costs become obstacle-task defaults

- User decision: in the obstacle tasks, a prey crash or arena exit costs
  -300, more than being caught (-200). The prey should not take a crash to
  escape a catch.
- Until now, -300 was only a per-run override (v3, overnight self-play);
  v4-v7 ran at the default -200.
- The predator's -600 per drone, passed by every obstacle run since
  2026-10-08, is now the default as well. The team reward is the mean over
  3 drones, so a lost drone costs the team -200, as much as a catch earns.
- Obstacle-free tasks keep -200 for both roles, which the anchor and
  reference reproductions need. `tests/test_obstacle_crash_cost.py` checks
  all 16 task configs. Commits cdbe9c3 and 556faa0, merged at 20:16 after
  v7 had finished.
- The exploration floor (-1.0) and the crashed-drone masks remain explicit
  run flags.

## 2026-10-10/11: Prey ghost stage v8, block x crash cost (pre-registered)

- A 2x2 from G6, all arms with seed 45 like B7:
  - block yes/no;
  - prey crash -300/-200.
- B7 (v7) is one arm. B8 (block, -300), C8 (no block, -300), and C7 (no
  block, -200) run one after another, 20:16 to about 06:30.
- Questions:
  - Does the block cause B7's degradation? (B7 vs C7)
  - Does the block help at the new cost? (B8 vs C8; this decides self-play
    with or without the block)
  - What does the crash cost change?
- Rule: end values are the mean of the last two checkpoints. The block
  helps if dense crashes are <= 0.85x, the dense ratio is >= 1.2, and the
  4-pillar catch hazard is <= 1.2x. It hurts if dense crashes are >= 1.15x,
  or a 4-pillar hazard is > 1.2x.
- Evidence: `artifacts/isaac/experiments/2026-10-10_prey_ghost_stage_v8_block_x_crash_cost/`.
- Result: **the block hurts in both pairs, so self-play goes without the
  block, and B7's degradation comes from the block.** End values against the
  start predator:
  - -300, B8 vs C8: dense crash 0.532 vs 0.347/s, 4-pillar crash 0.126 vs
    0.059, 4-pillar catch 0.294 vs 0.101.
  - -200, B7 vs C7: 4-pillar crash 0.219 vs 0.097, 4-pillar catch 0.227 vs
    0.065.
- Training: the no-block arms improve smoothly (C8 return -86 -> -53). B8
  declined to -153 and then switched abruptly to flying at about 8 m/s. That
  works against ghosts and fails against real predators. The block's
  attention still found the dangerous pillar (best head 0.56, 82% argmax),
  so the module works, but grafting it onto a trained network at a 30x
  learning rate destabilizes PPO.
- Crash cost -300 vs -200 without the block (C8 vs C7):
  - crashes 0.57-0.68x, catches 1.2-1.5x;
  - perception much stronger: dense ratio 1.43 vs 0.97, 4-pillar ratio
    3.59 vs 1.97.
- **C8 is the best prey so far:**
  - dense perception at last: ratio 1.41 and 1.46 at the last two
    checkpoints;
  - 4-pillar yardstick crash hazard 0.057/s (G6 0.118, v1 G 0.147);
  - the price is more catches (4-pillar 0.101 vs 0.072/s).

## 2026-10-11: Self-play in the dense layout from C8 (pre-registered)

- Setup:
  - start: the reference predator (8-slot) and v8's C8 prey, no block;
  - crashed-drone masks in training; prey crash -300, predator -600;
  - 5 cycles of 2400 joint plus 1200 prey rehearsal updates (3.5 m/s
    ghosts, a third of the prey's updates).
- Pre-registered decision: criteria 1-3 fail, 4 passes (mean of cycles 4
  and 5). By the rule, the prey loses perception under self-play; next, mix
  ghost and real-predator episodes within each rollout.
- Nuances:
  - Speed fails only because of a carried-over threshold (3.66 -> 3.69 m/s,
    no slowdown; note in the plan).
  - The predator gained dense perception: ratio 1.09 -> 1.18 (peak 1.36),
    crash hazard -29%, and +41% crashes in its own matchup when its branch
    is zeroed.
  - The prey kept part of its perception against the fixed opponent (1.46
    -> 1.14; 4-pillar 3.52 -> 2.67) instead of collapsing as in the earlier
    self-play (0.71). It does not use it against its real opponents: in the
    own matchup, zeroing its branch changes crashes by -7%.
  - Its own-matchup pillar crashes grew 25%; 93% are head-on with no
    pursuer within 1.5 m in 77%, so they are avoidable.
- Strategy:
  - predators crash less (0.47 -> 0.29 per episode) and chase around
    pillars;
  - the prey moves outward, uses pillars as cover more often (occluded
    escapes 17% -> 24%), and is caught more often near the wall.
- Evidence: `artifacts/isaac/experiments/2026-10-11_selfplay_dense_c8/`.

## 2026-10-11/12: Mixed ghost rollouts (pre-registered)

- The dense self-play's pre-registered next step. Code 252779f:
  - `env.ghost_env_fraction` gives a share of the envs 3.5 m/s ghost
    predators (here a third);
  - `RecurrentMAPPO.set_training_envs` trains the predator only on the
    real-predator envs: advantage normalization, value-normalizer
    statistics, minibatches;
  - training metrics per env group.
- Checks:
  - no ghosts and all ghosts reproduce their evaluations bit-exactly;
  - with half ghosts, ghost predators never crash and stay at 3.5 m/s;
  - smoke run: the predator trains on 2731 of 4096 envs.
- Design: the same start pair as the phase-wise run. One continuous run of
  28,800 updates, evaluated every 3600. The decision is at 18,000 updates,
  the phase-wise budget.
- The prey must reach a ratio >= 1.2, dense and 4-pillar, and use its
  perception in the own matchup (ablation ratio >= 1.1; phase-wise 0.93).
  Guards: predator own-matchup ratio >= 1.2, catch rate 0.3-0.7, speed
  >= 3.3 m/s (0.9x the start value).
- Evidence: `artifacts/isaac/experiments/2026-10-11_mixed_ghost_rollouts/`.
- Result (18,000 updates): criteria 1 and 4 pass; 2, 3, and 5 fail. By the
  rule, mixing alone does not protect the prey's perception against real
  opponents.
- Against the fixed opponent, mixing protects the prey's perception much
  better than phases (ratio 1.66 / 4.01 vs 1.14 / 2.67). Own-matchup
  crashes fall by more than half for both roles.
- In the own matchup, however, self-play settled into a cautious
  equilibrium (point 0 -> 8):
  - the prey flies slowly along the wall (median radius 3.6 -> 4.6 m; speed
    3.66 -> 2.69 m/s; time in the pillar zone 62% -> 29%);
  - the predators keep their distance (2.3 -> 4.1 m) and catch it at the
    wall (90% of catches at r > 4 m).
  So the prey's perception matters little against its real opponents
  (own-matchup ablation 1.10 -> 1.01 -> 0.76).
- Points 6-8 deepen the passivity. Against the start predator, perception
  declines (dense 1.56 -> 1.24) and catches rise.
- Next: a design decision on incentives and geometry, without
  strategy-encoding rewards. Options: remove the pillar-poor wall ring
  (layout) or lower the prey crash cost (for example -250).

## 2026-10-12: Mixed self-play with pillars up to the wall (pre-registered)

- User decision: remove the pillar-poor wall ring behind the cautious
  equilibrium, keep mixed rollouts.
- Code d933b12: task `...-obstacles10-v0`, 10 pillar axes over the whole
  5 m arena disk (density 0.127 vs 0.120 per m^2). Checks:
  - the expanded start pair with 8 active pillars at radius 4.6 reproduces
    the 8-pillar evaluation bit-exactly;
  - the axes reach 5.0 m, with the outer ring slightly denser than a
    uniform disk.
- Design: the same start pair and mixed setup as the night run; 18,000
  updates, decision at the end.
- The wall ring caused the equilibrium if, at point 5:
  - the prey's own-matchup ablation ratio is >= 1.1 (mixed: 1.01);
  - its own-matchup speed is >= 0.9x its start speed (mixed: 0.79x);
  - its ratio against the fixed opponent is >= 1.2.
  Guards: catch rate 0.3-0.7, predator own-matchup ratio >= 1.2.
  Otherwise, next: the crash-cost lever (prey -250).
- Evidence: `artifacts/isaac/experiments/2026-10-12_mixed_wall_pillars10/`.
- Result: criteria 1, 4, and 5 pass; 2 and 3 fail. By the rule, the
  cautious equilibrium is not only a layout effect; next is the crash-cost
  lever (prey -250).
- What the wall pillars fixed: no pillar-free refuge, so both roles now use
  perception against their real opponents (own-matchup ratios: prey 0.88
  -> 1.14, predator 0.98 -> 1.24). The predators intercept more and catch
  more (catch rate 0.50 -> 0.66).
- What they did not fix: the prey still slows down (3.53 -> 2.55 m/s,
  0.72x), keeps its distance (nearest predator 2.3 -> 4.0 m), and is caught
  at the wall (85%). At that speed the 3.5 m/s ghosts catch it more often
  (0.033 -> 0.159/s).
- The slowdown appears in both layouts (0.79x with 8 pillars, 0.72x here),
  so it is driven by the incentives: under crash -300 vs caught -200, fewer
  crashes for more catches pays.

## 2026-10-12: Predator skill stage against the waiting prey, then self-play (pre-registered)

- The user saw it in the video: the prey waits at one side and circles, and
  the predators crash often. The user judged the predators not good enough.
- The episode anatomy of the wall-pillar run's final own matchup agrees:
  - most episodes (63%) end within 2 s, mostly with a catch;
  - in the 14% that last 5 s or more, the predators lose 0.88 drones to
    pillars, fly at 1.1 m/s, and stay 6.6 m away;
  - 52% of those end by time-out.
- Part 1, the predator stage: only the predator trains (9600 updates)
  against the frozen waiting prey, in the 10-pillar layout. It passes if:
  - time-outs fall to <= 0.5x;
  - the catch rate rises by >= 0.05;
  - predator losses stay <= 1.2x;
  - the predator's own-matchup ratio is >= 1.2.
- Part 2, chained only if part 1 passes: mixed self-play from the stage's
  final pair, 28,800 updates, decision at 18,000. Criteria as in the
  wall-pillar run: the prey must keep flying (>= 0.9x its start speed) and
  use perception (own ratio >= 1.1, fixed-opponent ratio >= 1.2). If the
  prey still turns passive, next is the prepared crash-cost test (prey
  -250).
- Evidence: `artifacts/isaac/experiments/2026-10-12_predator_stage_wall10/`
  and `.../2026-10-12_selfplay_after_predator_stage/`.
- Result of part 1 (gamma 0.99): **does not pass.** Time-outs fell only
  0.070 -> 0.051, the catch rate rose only 0.661 -> 0.672, losses went
  0.27 -> 0.23, and the predator ratio is 1.20. In long episodes the
  predators stay slow (1.1 m/s) and far (6 m). The chain stopped.
- Explanation: the horizon. At 50 Hz, gamma 0.99 gives about 2 s; a catch
  4 s ahead counts 13%. The pursuit shaping is range-independent here, so it
  is not the cause.

## 2026-10-12/13: Horizon test, gamma 0.995 / 0.998 (pre-registered decision tree)

- The predator stage is repeated with gamma 0.995 (about 4 s horizon). Its
  design and criteria are identical; only gamma changes.
- Pre-registered tree:
  - pass: mixed self-play with gamma 0.995;
  - fail: a stage with gamma 0.998 (about 10 s); if it passes, self-play
    with gamma 0.998; if it fails, stop.
- The three stages form a dose-response test of the horizon hypothesis.
- Evidence: `artifacts/isaac/experiments/2026-10-12_predator_stage_gamma995/`
  (PLAN.md with the tree) and the branch folders
  `2026-10-13_predator_stage_gamma998`, `2026-10-13_selfplay_gamma995`, and
  `2026-10-13_selfplay_gamma998`.
- Result, gamma 0.995: **does not pass.** Time-outs 0.070 -> 0.063, catch
  rate 0.661 -> 0.669. The predator's pillar perception grew (own ratio
  1.24 -> 1.41), but in long episodes it still flies at about 1.2 m/s about
  6 m away. By the tree, the gamma 0.998 stage started at 23:56.
- Result, gamma 0.998: **does not pass.** Time-outs 0.054, catch rate
  0.690 (+0.029). The chain stopped (branch 2b).
- Dose-response over gamma 0.99 / 0.995 / 0.998: no effect on the long
  game, so **the horizon is not the bottleneck.**
- At every horizon the long episodes show the pillar shield: the line to the
  prey is blocked 53-56% of the time, the predators do not approach
  (closing speed about 0, 6.3-6.7 m away), and 70-84% of their losses happen
  with the line blocked. They pursue only directly, never around a pillar.
- Probable contributing cause: the Euclidean distance-progress shaping
  penalizes detours. Levers (user decision): no predator progress shaping
  (config A/B), path-distance progress (code), or a blocked-line curriculum.

## 2026-10-13: Predator stage without progress shaping, then self-play (pre-registered)

- User approval: "ok do continue as recommended".
- Test of the shaping-trap hypothesis: the same stage as gamma 0.998, but
  `predator_distance_progress_reward_scale` 0 instead of 2.0 (proximity was
  already 0). The predator learns from catches, crash costs, and the
  generic flight terms only.
- Criteria as in the earlier stages. Mechanism reported: long-episode
  closing speed, distance, and blocked-line losses.
- If it passes: mixed self-play without progress shaping (gamma 0.998,
  18,000 updates, criteria of the wall-pillar run), started automatically.
  If it fails: stop, and decide together.
- Evidence: `artifacts/isaac/experiments/2026-10-13_predator_stage_noprogress/`
  and `.../2026-10-13_selfplay_noprogress/`.
- Result: **does not pass** (time-outs 0.060, catch rate 0.688). The chain
  stopped. Neither the horizon nor the progress shaping is the bottleneck.
- Shield geometry in long episodes: the prey circles its shield pillar
  (within 1 m 35-38% of the time, about 0.8 rad/s). Two predators flank
  that pillar (> 120 deg apart) in only 1-2% of frames, and in 0-4% of
  long-episode catches. The missing skill is a pincer around the pillar, a
  multi-agent coordination problem.
- Next (user decision): an end-game curriculum with shield start
  situations, without a pincer reward.
