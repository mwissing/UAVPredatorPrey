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

## 2026-10-13: Shield starts, an end-game curriculum for the predators (pre-registered)

- User approval: "ok do that".
- Code ea86aac: `shield_start_fraction`. That share of episodes starts with:
  - the predators at a random distance (3-6 m) and an independent bearing
    around the prey, so no formation is prescribed;
  - a pillar right next to the prey (surface gap 0.3-0.8 m).
  Only the start states change, not the reward. Shield starts off is
  bit-exact; the geometry was checked live; 4 tests.
- In pure shield starts the start pair gives catch 0.326, prey crash 0.549,
  time-out 0.114.
- Stage: identical to the first predator stage (gamma 0.99, progress
  shaping 2.0), with a third of the episodes as shield starts. The same
  four criteria (regular starts). The pincer share and the shield-start
  catch rate are reported. If it passes, mixed self-play with shield starts
  follows automatically (18,000 updates).
- Evidence: `artifacts/isaac/experiments/2026-10-13_predator_stage_shield/`
  and `.../2026-10-13_selfplay_shield/`.
- Result: **does not pass.** Regular starts: time-outs 0.075, catch rate
  0.636 (start 0.661). Shield starts: catch 0.326 -> 0.306, pincer share
  unchanged at 2%. The predators crash less (hazard -28%, ratio 1.40) but
  do not learn to flank or approach.
- Five predator stages, one pattern: the predators get safer, not more
  effective at the end game.
- Hypothesis: a skill trap. With -600 per drone (-200 for the team, as much
  as a catch), approaching a pillar-adjacent prey has negative expected
  value as long as their catch chance there is low, so they never practice
  it.
- Candidate next steps (user decision): a predator crash-cost curriculum
  (lower cost in shield starts, then restore), or accept the end-game
  exploit and move on.

## 2026-10-14: Predator crash-cost curriculum (pre-registered)

- User approval: "ok continue with option 1".
- A test of the skill-trap hypothesis.
- Phase A: the shield-start stage, but with predator crash -300 per drone
  (-100 for the team) for practice, 9600 updates. It passes if time-outs
  <= 0.5x and the catch rate gains >= +0.05 (losses reported).
- Phase B, only if A passes: back to -600 for 3840 updates. It passes on
  the four original stage criteria.
- If A and B pass, mixed self-play with shield starts follows automatically.
  If A fails, the hypothesis is not supported, and the recommendation is to
  accept the end-game exploit and move on.
- Evidence: `artifacts/isaac/experiments/2026-10-14_predator_crash_curriculum/`.
- Result, phase A (crash -300): **does not pass** (time-outs 0.070 ->
  0.052, needed <= 0.035), although the catch rate rose 0.661 -> 0.729. The
  chain stopped; phase B did not run.
- Partly supported:
  - a cheaper crash made the predators engage more (closing speed in short
    episodes 0.09 -> 0.36 m/s) and catch more (shield starts 0.326 ->
    0.459, time-outs there halved);
  - they crash more (hazard 0.097 -> 0.123).
- Not broken: in the remaining long episodes the predators still hover
  about 7 m away without a pincer, and half end by time-out.
- After six predator stages, the pillar-shield exploit is robust to horizon,
  shaping, curriculum, and crash cost. Breaking it would likely need
  multi-agent coordination learning (a separate step).
- Options: accept the exploit and move on (the pre-registered
  consequence), a lower predator crash cost as a compromise, or a
  coordination research step.

## 2026-10-15: Predator crash cost -400 (user decision)

- The obstacle default for the predator becomes -400 per drone (302ca61):
  -133 for the team, two thirds of a catch.
- Reason: at -600 the predators stopped engaging a prey next to a pillar;
  at -300 they caught more (0.729) but crashed more.
- Obstacle-free tasks keep -200. The obstacle runs from 2026-10-08 to
  2026-10-14 used -600.

## 2026-10-15: Step 0, can a scripted pincer catch the waiting prey? (pre-registered)

- User approval: "ok start with step 0. after that the role embeddings".
- Code: c968c8c (`ghost_mode` "encircle" / "direct", scripted pursuers that
  respect pillars and can crash) and a553cbc (safe approach speed). The
  default lead ghosts are bit-exact. Evaluation only, against the frozen
  waiting prey.
- Pre-registered decision: **inconclusive.** The encircle controller loses
  too many drones to pillars (0.44, then 0.35 per episode; limit 0.27).
- Robust across both controllers:
  - the scripted encirclement creates pincers (35-38%) but catches only
    11-12%;
  - in 85% of episodes it drives the prey into a pillar instead;
  - scripted direct pursuit catches 0.35-0.47;
  - the learned predators catch 0.66 (0.73 at -300), so they are the best
    catchers.
- Persistent pressure removes the time-outs (0.2-2.9% vs 7%) through prey
  crashes. Those pay the predators nothing: their reward counts only
  catches, so the game is not zero-sum.
- So the waiting exploit is an incentive gap in the outcome definition,
  not a missing coordination skill. The premise behind role embeddings is
  not supported.
- Next (user decision): count a forced prey crash (predator within 1.5 m)
  as a predator success, or accept the exploit. Role embeddings are paused.
- Evidence: `artifacts/isaac/experiments/2026-10-15_scripted_pincer_control/`.

## 2026-10-15/16: Forced-crash outcome, predator stage with a control arm (pre-registered)

- User approval: "go ahead".
- Code c070ceb: `predator_forced_crash_bonus`. A forced prey crash (a prey
  exit without a catch while the nearest active predator is within 1.5 m)
  earns the team as much as a catch: the nearest predator gets bonus x P.
  It is 200 in the obstacle tasks and 0 otherwise. 3 tests; smoke run
  checked.
- Test: two arms from the same start against the frozen waiting prey,
  predator-only, 9600 updates each, predator crash -400:
  - f: bonus 200;
  - c: bonus 0, the control.
- Arm f passes if:
  - time-outs <= 0.5x;
  - success (catch + forced crash) gains >= +0.05;
  - losses stay <= 1.2x;
  - own predator ratio >= 1.2;
  - attribution: time-outs f <= 0.7x c.
  A shadowing check reports the predator's distance at forced crashes.
- If it passes, mixed self-play with the new outcome follows automatically
  (18,000 updates).
- Evidence: `artifacts/isaac/experiments/2026-10-15_predator_stage_forced_crash/`.
- Result: **does not pass.** Arm f (bonus) and arm c (control) are
  practically identical:
  - time-outs 0.040 / 0.040 (start 0.070);
  - catch 0.719 / 0.716 (start 0.661);
  - forced crashes 0.064 / 0.052.
- The improvement comes from the lower crash cost (-400), not from the
  bonus. In long episodes the predators still hover about 6.6 m away. No
  shadowing.
- Eight interventions in total; only the predator crash cost moved the
  result: catch 0.66 -> 0.72-0.73, time-outs 0.070 -> 0.04-0.05. The
  pillar-shield strategy holds in about 4% of episodes.
- Recommendation: accept the exploit and consolidate with a final, longer
  reference run (user decision).

## 2026-10-16: Role vectors for the predator team, two seeds (pre-registered)

- User approval: "i want you to test role embedding over the day while i am
  working (till 16-17)"; earlier "ok then the id version".
- Code bc55fd9: `predator_role_embedding` adds a learned vector per
  predator index to the own-state embedding.
  - Only its deviation from the team mean is used.
  - It starts at zero and learns in the fast group (x30).
  - 7 tests. Evaluation with the flag on and off is bit-identical; a role
    checkpoint fails loudly without the flag.
- Why: the shared predator network is permutation-equivariant, so it cannot
  learn conventions such as "drone 0 flanks left". Agent-ID conditioning is
  the standard fix. Step 0 argued against the premise; the user asked for
  the test anyway.
- Design: predator-only against the frozen waiting prey, the setup of
  forced-crash arm f (crash -400, bonus 200), 9600 updates per arm:
  - r: roles, seed 70, controlled by arm f (seed 70, done);
  - s: roles, seed 71;
  - x: no roles, seed 71, the control for s; it runs last.
- Decision per seed (point 5 vs the matched control):
  1. manipulation check: per-index side or lead spread >= 0.15 (the
     index-blind controls are <= 0.048);
  2. time-outs <= 0.7x;
  3. pincer share in long episodes >= 0.10 (controls 0.01);
  4-6. guards: success, losses, perception.
- The role-vector norms per point show whether the learning signal pushes
  the roles apart (the own embedding's norm is about 59).
- Evidence: `artifacts/isaac/experiments/2026-10-16_predator_role_embedding/`.
- Result: **the roles did not differentiate** (criterion 1 fails in both
  seeds, so by the plan the test is inconclusive about coordination).
  - The role vectors stayed at the gradient-noise level: norms 0.5-0.8,
    1.0-1.4% of the own embedding.
  - The functional probe (how differently drones act on identical
    observations) stayed at about 2%. In seed 70 it shrank from 0.027 to
    0.019.
  - Per-index spreads match the index-blind controls (side <= 0.056, lead
    <= 0.018).
  - Outcomes match the controls:

    | | time-outs | pincer | catch |
    |---|---|---|---|
    | r (roles, seed 70) | 0.042 | 0.02 | 0.722 |
    | f (control, seed 70) | 0.040 | 0.01 | 0.719 |
    | s (roles, seed 71) | 0.042 | 0.01 | 0.734 |
    | x (control, seed 71) | 0.050 | 0.02 | 0.708 |
- Seed variation, the first replicate of a predator stage (f vs x): catch
  0.011, time-outs 0.010, success 0.017. Single-run differences of 0.01-0.02
  are noise.
- Interpretation: given the capacity to specialize, the predators did not.
  The symmetric team is a stable solution of the fine-tuning objective.
  Together with step 0, the evidence says symmetry is not what limits them.
- Prepared, not run: an imposed-role arm. Random role vectors at norm 10
  give a lead spread of 0.164 at a start cost of -0.024 catch
  (`option_random_init/`).
- Recommendation unchanged: consolidate and accept the exploit (user
  decision).

## 2026-10-16: Exploration boost, are the predators stuck in a local optimum? (pre-registered)

- User question: were we stuck in a local optimum the whole time, and
  simply never got out of the waiting strategy? Approval: "yes do that
  test".
- Context:
  - All nine interventions were local searches from one checkpoint (PPO, LR
    1e-5, KL early stopping), so they show only that the plateau survives
    local changes.
  - Against a local optimum: the only known alternative, scripted pursuers
    that always press, earns the same outcome return. Outcome return is 200
    x success - 133 x drones lost: scripted 116.5, learned at point 5
    108.6-116.7 over four runs.
  - For one: the forced-crash bonus is almost never experienced, because in
    long episodes the predators stay about 6.6 m away.
- Design: one predator-only arm from the shared start, at arm f's setup
  (seed 70):
  - 4800 updates with the predator's action-noise floor raised from std
    0.37 to 0.74;
  - then the noise reset to the start's value and 4800 updates at the
    standard level.
  Arms f (seed 70) and x (seed 71) are the controls.
- Decision at 9600 updates:
  1. time-outs <= 0.020 (the four learned point-5 runs give 0.040-0.050);
  2. outcome return >= 108.6 (the lowest control);
  3. manipulation check: training forced-crash rate in the boost phase
     >= 1.25x the controls' 0.072.
- Interpretation:
  - 1 and 2 pass: an exploration-limited local optimum.
  - 1 fails and 3 passes: the plateau is real.
- Evidence: `artifacts/isaac/experiments/2026-10-16_predator_exploration_boost/`.
- Result: **inconclusive by the plan** (criteria 1 and 3 fail, 2 passes).
  - At e5 the predators sit exactly on the controls' plateau: time-outs
    0.049 (controls 0.040-0.050), outcome return 116.1 (116.7 / 108.6),
    long-game distance 6.7 m, pincer 0.01.
  - The manipulation check missed narrowly: the boost raised the
    experienced forced crashes by 19% (bar 25%).
  - More noise made the flight sloppier, not more pressing: closest approach
    0.89 vs 0.84 m, training catch 0.62 vs 0.67, crash rate 0.30 vs 0.27.
- The training path was clearly perturbed: the policy lagged the controls
  during the boost, and its obstacle ablation ratio was 1.02 at the switch.
  Yet it converged to the same end point, which is consistent with a robust
  attractor but not a proof.
- Combined with the scripted always-press strategy (same return) and the
  capped upside (about +0.05 success), a substantial missed optimum is
  unlikely; a small end-game one cannot be excluded. Per-episode (parameter)
  perturbations would be the coherent next probe, if ever wanted.
- Recommendation unchanged: consolidate (user decision).

## 2026-10-16/17: Obstacle reference self-play, closing Stage 8 (pre-registered)

- User decisions: "ok so exploit accepted"; "ok lets do step 1".
- Design: the prepared forced-crash part-2 design, run as the reference.
  - Start: arm f's final pair (the f5 predator plus the waiting prey).
  - Mixed self-play: both roles train, a third of the envs have 3.5 m/s
    ghosts.
  - 10 pillars, seed 60, 18,000 updates.
  - Current defaults: prey crash -300, predator crash -400, forced-crash
    bonus 200.
  - The config differs from the wall-pillar run only in these defaults.
- Acceptance gate at points 4 and 5 ("competent and stable"):
  1. both roles use perception (own-matchup ablation ratio >= 1.1);
  2. own catch within 0.3-0.7;
  3. no forgetting against the fixed anchor-v3 opponents (catch within 0.05
     of point 0 in the bad direction);
  4. own catch changes by <= 0.05 from point 4 to 5.
- If all pass, the final pair becomes the obstacle reference (anchor v4).
- Evidence: `artifacts/isaac/experiments/2026-10-16_obstacle_reference_selfplay/`.
- Result: **accepted; all six criteria pass**. The final pair is anchor v4
  ([doc](anchor_v4_2026-10-17.md); read-only bundle
  `.pretrained_checkpoints/anchor_v4_2026-10-17/`, bit-exact).
  - Own ablation ratios: prey 1.03 / 1.25, predator 1.20 / 1.23.
  - Own catch 0.678 / 0.680.
  - No forgetting: predator vs the fixed prey 0.149 (start 0.146); the fixed
    predator vs our prey 0.147 (start 0.167).
- Self-play partly undid the predator stage: time-outs 0.040 -> 0.062,
  predators lost 0.29 -> 0.37 per episode, and the prey slowed to 2.55 m/s.
- The prey got worse against ghosts (catch hazard 0.159 -> 0.274/s).
- No coordination emerged.
- Blind-spot probes give the same values as for f5 and point 0, so the
  defects belong to the architecture.

## 2026-10-16/17: Stage 8 blind-spot diagnostics and review

- User request: integrate an external review and "do everything that is
  necessary, so we get a full analysis".
- Tooling, `artifacts/isaac/diagnostics/2026-10-16_stage8_blind_spots/`:
  - `probe_record.py` wraps evaluate.py; a deterministic run with it is
    bit-identical to a plain evaluation. It records attention weights,
    critic values and features.
  - Relation probes and a value control.
- Measured on three pairs (stage start, f5, anchor v4):

  | measurement | value | chance or reference |
  |---|---|---|
  | predators attend the prey's pillar in standoffs | 0.4-0.7% | 10% |
  | prey in a predator's attention context | 45-49% of steps | |
  | hugging prey attends its own pillar | 0.2-1% | |
  | critic EV in long episodes | -0.14 to 0.00 | 0.38-0.43 short |
  | critic features: prey-pillar distance R^2 | 0.08-0.22 | 0.9 for agent distances |
  | line of sight blocked: set encoder / attended pillar / GRU | 0.83 / 0.57 / 0.55 | |
  | yaw rate, median | 117-125 deg/s | |

  The yaw matters because relative positions are in the body frame,
  velocities in the world frame, and the heading is not observed.
- Review: [`stage8_review_2026-10-17.md`](stage8_review_2026-10-17.md).
  - Verified: most claims, plus four corrections of our earlier
    conclusions.
  - One citation not found (Gavin & Bronz, ICUAS 2026).
- Next (user decision): representation and credit fixes, then a
  from-scratch run (3 seeds) against anchor v4, then Stage 9.

## 2026-10-17: Credit horizon, predator stage with 96-step rollouts and lambda 0.99 (pre-registered)

- User approval: "continue as suggested", the cheap credit test before the
  architecture work.
- Correction found while designing it: GAE runs inside each 24-step rollout
  and bootstraps there.
  - The direct credit window therefore averages about 12 steps (0.24 s)
    whatever lambda is, so a lambda-only A/B would barely change anything.
  - Arm h changes the rollout to 96 steps (1.92 s), lambda to 0.99, and the
    minibatches to 8. The minibatch size and gradient steps per env step
    stay as in arm f.
  - It runs 2400 updates, the same env steps as arm f; seed 70.
- Decision at h5:
  1. time-outs <= 0.028;
  2. long-episode nearest-predator distance <= 5.5 m (all six earlier runs
     5.96-6.57);
  3. guards on success, losses and perception.
  A blind-spot probe of h5 reports the critic's explained variance.
- Evidence: `artifacts/isaac/experiments/2026-10-17_predator_credit_horizon/`.

- Result: **direct credit up to about 2 s does not change the long game**
  (criteria 1 and 2 fail; guards hold; 16:08-20:11).
  - h5: time-outs 0.049 (f5 0.040, x5 0.050); long-game nearest predator
    6.21 m (f5 6.15).
  - Success 0.778, outcome return 116.9 (f5 116.7).
- Along the way arm h lost fewer drones (0.26-0.27 vs 0.32-0.35 per episode
  at points 1-3). Longer credit sharpens crash avoidance, which the
  predators can perceive, but not the end game, which they cannot.
- The h5 probe shows the same blind spots: critic EV -0.11 in long
  episodes, prey's pillar attended in 0.3% of standoff steps.
- By the rule: representation first. The relation graft (arm g) started
  automatically at 20:15.

## 2026-10-17: Relation graft, predator stage with pillar-prey relation branches (pre-registered)

- User approval: "yes", chained after the credit-horizon test.
- Code eace487:
  - `predator_obstacle_relation` (actor): every present pillar becomes a
    token with the prey (pillar offset and distance, prey offset, prey
    minus pillar, axis distance to the drone -> prey segment), with masked
    max + mean pooling;
  - `obstacle_relation` (critic): one token per pillar with everyone's
    rotation-invariant view of it, in a fixed order;
  - both zero-initialized in the fast group; evaluation is bit-identical
    with the flags on or off; 7 tests.
- Design: arm g is forced-crash arm f plus both flags (seed 70, 9600
  updates, rollouts 24, lambda 0.95). Controls f and x.
- Decision at g5:
  0. manipulation check from the blind-spot probe: policy line of sight
     >= 0.70 or prey-pillar R^2 >= 0.4, and critic prey-pillar R^2 >= 0.4;
  1. time-outs <= 0.028;
  2. long-game distance <= 5.5 m;
  3. guards.
- Evidence: `artifacts/isaac/experiments/2026-10-17_predator_relation_graft/`.

## 2026-10-17/18: Night chain after the relation graft (pre-registered, conditional)

- User: "pls continue with tasks during the night".
- `artifacts/isaac/experiments/night_chain_2026-10-18.sh` waits for the
  graft test, then branches. Each follow-up has its own PLAN.md.
  - If the graft verdict is "the missing relations were the bottleneck":
    1. the seed-71 replication (arm q,
       `2026-10-18_relation_graft_seed71/`);
    2. if that replicates, the graft self-play
       (`2026-10-18_relation_graft_selfplay/`, the reference recipe with the
       graft, 18,000 updates). Its gate is the reference gate plus
       time-outs <= 0.040 (anchor v4: 0.062).
  - Otherwise: a continuation of the graft to 19,200 updates
    (`2026-10-18_relation_graft_continuation/`), with the same criteria at
    c5.

## 2026-10-18 00:20: Relation graft crashed (CUDA out of memory); fixed and rerun

- The first graft run crashed at 21:36 after 1920 updates. The relation
  tokens' stored activations and copies added about 5 GB, and the training
  process reached 29.6 GB.
  - My mistake: its smoke run used 256 envs, not the full 4096.
  - The night chain v1 then fell through (no verdict, so the continuation
    started without a g5 checkpoint and failed). The GPU was idle from
    21:39 to 00:18.
- Fix (171fd3b): the actor's token encoder is 128 units wide, encodings are
  recomputed in the backward pass, and the pooling masks in place.
  Full-size smoke peak: 21.2 GB vs 20.2 GB without the branches.
- The drivers now stop without analysis if training fails. The chain v2
  stops on any technical failure.
- The crashed run is archived in `crashed_run_1/`, with its probe of g1
  renamed from the mislabelled "g5". At g1 the critic's long-episode
  explained variance was 0.19 (f5 -0.14); an observation, not a criterion.
- Also merged into dev-3v1-obstacles (fast-forward from branch
  stage8b-fixes), all default-off and bit-exact (training smoke identical
  to the old code):
  - body-frame observations;
  - per-drone PPO ratios;
  - the zero-at-level upright term;
  - the prey's relation branch;
  - the draft from-scratch protocol.
- The rerun of the graft test (same plan and criteria) started at 00:18.
  The chain v2 follows with the replication or the continuation.
- Rerun result (00:18-04:07, status 0): **inconclusive about the
  representation**. The policy-side manipulation check fails; criteria 1
  and 2 fail; the guards hold.
  - The probe shows where the representation stops:

    | relation probe | encoder (GRU input) | GRU output |
    |---|---|---|
    | policy: line of sight | 0.74 (earlier 0.55) | 0.56 (unchanged) |
    | critic: prey-pillar distance R^2 | 0.93 (earlier 0.08-0.15) | 0.71 |

  - The branches deliver the relations. The predators' pretrained GRU and
    actor head, at LR 1e-5, do not route them to the actions in 9600
    updates.
  - Best predator stage so far: catch 0.754 and success 0.810 (f5 0.719 /
    0.783); time-outs 0.032 (f5 0.040, x5 0.050, bar 0.028); long episodes
    7%.
  - Critic explained variance in long episodes 0.20 (f5 -0.14), so the gain
    most likely comes through the critic.
  - By the chain, the continuation to 19,200 updates started at 04:11.
- Continuation result (04:11-08:01, status 0): the pre-registered verdict is
  "the branches did not learn the relations". The policy-side check missed
  narrowly: GRU line of sight 0.670 vs the 0.70 bar, up from 0.558. The
  outcome changed sharply.

  | c5 vs | c5 | g5 | f5 |
  |---|---|---|---|
  | time-out share | **0.008** | 0.032 | 0.040 |
  | catch | **0.840** | 0.754 | 0.719 |
  | success | 0.885 | | |
  | losses / episode | 0.24 | | |
  | outcome return | 143.2 | 117.3 | 116.7 |
  | long episodes | 3.5% | | |

- Not attributable yet: no stage without the graft was ever trained beyond
  9600 updates, so training time against the frozen prey may explain part
  or all of it.
- The matched control (arm f continued to 19,200 updates) is the next
  step, pending the user's go. GPU idle since 08:04.

## 2026-10-18: Matched control, arm f continued to 19,200 updates without the graft (pre-registered)

- User approval: "yes start it".
- Arm k continues forced-crash arm f from f5, with its optimizer state, for
  9600 more updates. It is exactly arm c's protocol without the relation
  flags; the configs differ only in those two flags. Started 10:00.
- Attribution at k5 vs c5:
  - k5 time-outs > 0.028 and an outcome-return gap c5 - k5 >= 10: the graft
    is the cause;
  - k5 <= 0.028 and a gap < 10: training time explains it;
  - otherwise both contribute.
- Evidence: `artifacts/isaac/experiments/2026-10-18_control_continuation/`.
- Result (10:00-13:21, status 0): the pre-registered verdict is "the graft
  is the cause: without it the waiting game survives the longer training".

  | 19,200 updates | k5 (no graft) | c5 (graft) |
  |---|---|---|
  | time-out share | 0.047 | **0.008** |
  | catch | 0.751 | **0.840** |
  | drones lost / episode | 0.366 | **0.253** |
  | pillar-crash rate | 0.268 | **0.173** |
  | outcome return | 112.7 | **143.2** |

  - The no-graft arm stays at 114 +- 3 for all 19,200 updates. c leads k at
    every continuation checkpoint, by 11.2 to 30.5.
  - Every k5 probe is at f5's level: policy GRU line of sight 0.56, critic
    prey-pillar R^2 0.10, critic EV in long episodes -0.17. Training time
    does not move the blind spots; the graft does.
  - One seed per arm, frozen prey.
- Follow-up ablation (pre-registered 14:25, evaluation only,
  `artifacts/isaac/diagnostics/2026-10-18_c5_branch_ablation/`). c5's
  actor branches were zeroed one at a time:
  - relation branch off: outcome return 76.2, below k5; pillar crashes
    0.362, time-outs 0.032;
  - old one-hot attention off: 138.8, crashes unchanged.
  - So the c5 policy flies and hunts through its relation branch, which has
    replaced the old pillar attention. The g5 reading "most likely through
    the critic" is superseded.
  - The no-graft lineage hardly uses pillar perception: the crash hazard
    ratio with its pillar branch zeroed fell from 1.24 (point 0) to 0.97
    (k5).
- Next, the user decides between:
  - the seed-71 replication of graft plus continuation (recommended first,
    about 8 h; the plan needs an amendment);
  - the graft self-play from c5;
  - the from-scratch protocol.

## 2026-10-18: Seed-71 replication of the relation graft, with the continuation (pre-registered amendment)

- User approval: "yes we can continue as recommended".
- The night chain's seed-71 plan replicated only the 9600-update graft stage.
  At seed 70 the effect appeared only in the continuation (g5 117.3, c5
  143.2, k5 112.7). The amendment of 14:47, made before any data, therefore
  adds the continuation:
  - arm q is arm g with seed 71, 9600 updates (q1-q5);
  - then it continues from q5 with its optimizer state for 9600 more updates
    (q6-q10), as c continued g5;
  - code 171fd3b.
  - At q10: the analyses, the probe, and the branch ablation (relation
    branch off, old attention off).
- Decision at q10:
  1. time-outs <= 0.028;
  2. outcome return q10 - x5 >= 10;
  3. guards against x5 (success, losses, own ratio).
  The policy probe and the long-game distance are reported only, because c5
  missed both while its effect was large.
- Attribution:
  - A gap of >= 20 counts as the graft (k drifted -4 over the same
    continuation).
  - At 10-20, the matched control y starts automatically: x5 continued to
    19,200 updates, with k's rule (`2026-10-18_control_continuation_seed71/`).
- The decision logic was tested on mock data (seed 70 replayed as q). The
  three cases replicate, y-trigger and not replicated came out as intended,
  and y's attribution reproduced k's verdict.
- The graft self-play does not start automatically.
- Started 14:51.
- Evidence: `artifacts/isaac/experiments/2026-10-18_relation_graft_seed71/`.

## 2026-10-18: Relation-graft self-play from c5, chained after the replication (pre-registered amendment)

- User approval: "yes schedule 1. if it passes".
- The amendment of 15:49, made before any data, to the prepared self-play
  plan (`2026-10-18_relation_graft_selfplay/`):
  - **Condition.** It starts only if the seed-71 replication's verdict is
    "the effect replicates with seed 71". If the conditional control y runs,
    the self-play waits for it. Any other verdict: decide together.
  - **Start.** c5's pair (the c5 predator plus the waiting prey, optimizers
    dropped) instead of g5's. c5 is the predator whose result was attributed
    to the graft; g5 had not yet broken the waiting game.
  - **Unchanged.** Seed 60, 18,000 updates, a third of the envs with ghosts,
    the obstacle defaults, both masks, both relation flags (the prey's
    critic gets the branch, its actor does not), the reference gate, and
    criterion 5: time-outs at point 5 <= 0.040.
  - **Evaluations after the training.** On this GPU (RTX 5090, 32.6 GB) the
    predator-only graft training plateaus at 26.2 GB, and self-play adds the
    prey's update. The first graft run ran out of memory when an evaluation
    started next to its training. The evaluations are deterministic, so no
    number changes. The run writes a GPU memory log.
- `artifacts/isaac/experiments/chain_2026-10-18_selfplay_after_replication.sh`
  does this, without touching the running replication driver. It was tested
  on mock folders: replicates, replicates after y, not replicated and partly
  replicated each took the intended path. It also checks that no other
  Isaac job is running.
- Waiting since 15:53. Expected: the self-play from about 23:00 to 09:30, or
  from about 02:30 if y runs first.
- Replication result (stage q 14:51-18:52, stage qc 18:55-22:56, both
  status 0): the pre-registered verdict is "the effect replicates with seed
  71; attributed to the graft as at seed 70 (gap >= 20)".

  | 19,200 updates | q10 (graft, seed 71) | x5 (no graft, seed 71) | c5 (graft, seed 70) |
  |---|---|---|---|
  | time-out share | **0.020** | 0.050 | 0.008 |
  | catch | 0.832 | 0.708 | 0.840 |
  | outcome return | **137.2** | 108.6 | 143.2 |

  - The gap is +28.6 (seed 70: +30.5), so y did not run. Seed 71 rose from
    the start: q5 was already at 125.3.
  - Ablation: relation branch off gives 32.9 (3.65 times the gap; c5 2.2);
    old attention off gives 139.3.
  - The critic's EV in long episodes is 0.20 (k5 -0.17).
  - The policy probe misses the 0.70 bar again (0.633), while the ablation
    shows strong reliance.
  - No pincers (2% of frames).
  - Measurement gap: three curve evaluations (q2, q4) ran out of GPU memory
    next to the 26.8 GB training. They failed silently: Isaac's scripts exit
    with 0 after the exception. The decision is not affected; `fill_gaps.sh`
    reruns them after the self-play.
- By the chain, the graft self-play started at 23:01.

## 2026-10-19: Pillar-only graft, do fewer crashes alone break the waiting game? (pre-registered)

- User: "if predator crash less and by that no waiting game. maybe we just
  need to train longer? maybe ... test with longer horizon again? pls
  whatever you think schedule something for the day."
- **Training longer** without the graft is answered by the matched control:
  k crashed more (0.228 -> 0.268) and kept the waiting game.
- **A longer horizon** is deferred. With the graft, time-outs against the
  frozen prey are already 0.008-0.020. It matters if the self-play brings
  the waiting game back.
- **The user's mechanism** (fewer crashes -> no waiting game) is testable.
  The graft's actor branch carries all-pillar perception and the
  pillar-prey relation.
- Code 92c2909: `predator_obstacle_relation_pillar_only`.
  - It zeroes the prey-dependent token features (prey offset, prey minus
    pillar, segment distance), so the branch is a plain set encoder over
    the pillars, with the same shapes.
  - A marker buffer makes such a checkpoint fail to load without the flag.
  - Off by default and bit-exact; 239 tests pass.
- Design: arm n is arm g/c with the flag. Seed 70, shared start, 9600
  updates plus a continuation of 9600 from n5. The critic keeps its full
  branch.
  - Each stage is evaluated after its training (memory).
  - The driver also checks that the flag reaches the model: first the
    config, then the marker in the first checkpoint.
- Decision at n10, against c5 (full) and k5 (none):
  - time-outs <= 0.028 and outcome return >= 133.2: all-pillar perception
    is enough;
  - time-outs <= 0.028, lower return: partly;
  - time-outs > 0.028 and crashes <= 0.22: the waiting game needs the
    relation;
  - otherwise: the prey tokens carry the avoidance too.
- The decision tree was tested on mock data (all four branches).
- The driver waits for the self-play and the seed-71 gap fill (about 08:45),
  then runs about 8.5 h.
- Evidence: `artifacts/isaac/experiments/2026-10-19_relation_graft_pillar_only/`.
- Self-play result (training 23:05-07:38, status 0; evaluations after it,
  done 07:56): the pre-registered verdict is "the graft's advantage survives
  self-play and the pair passes the reference gate: anchor v5 candidate
  (freeze after the user's review)". All seven criteria pass.

  | own matchup | point 0 (c5 pair) | point 5 | anchor v4 (no graft) |
  |---|---|---|---|
  | time-outs | 0.008 | **0.005** | 0.062 |
  | catch | 0.840 | 0.564 | 0.680 |
  | prey crashes / episode | 0.144 | **0.418** | 0.247 |
  | predators lost / episode | 0.24 | 0.27 | 0.37 |
  | long episodes: nearest predator | 5.82 m | **2.01 m** | 6.57 m |
  | own ratio prey / predator | 0.80 / 1.65 | 1.15 / 2.83 | 1.25 / 1.23 |

  - The prey did not bring the waiting game back. It flees faster (3.9 m/s)
    and higher through the pillar zone and hides less (line of sight
    blocked 0.079 of the frames; anchor v4 0.217).
  - It crashes in 42% of the episodes: head-on at 4.7 m/s, half of them
    forced (a predator within 1.5 m).
  - The prey's perception is the weak side: its own ratio passes by 0.002
    (mean 1.102), the 4-pillar yardstick fell 2.44 -> 0.67, and it is
    weaker against ghosts. This is the expected asymmetry: only the
    predators have the relation branch.
  - Review options (user):
    - freeze the pair as anchor v5 with the caveat;
    - freeze only the predator and give the prey its relation branch first;
    - fold both into the from-scratch protocol.
- Seed-71 gap fill (07:57, GPU free): q2 117.4 and q4 127.3; all three
  evaluations succeeded. The decision is unchanged.
- Pillar-only result (stage n 07:59-11:41, stage nc 11:48-15:32, status 0;
  every evaluation after training succeeded; the flag marker was present in
  both stages): the pre-registered verdict is "perception lowers crashes,
  but breaking the waiting game needs the pillar-prey relation".

  | 19,200 updates, seed 70 | n10 (pillars only) | c5 (full) | k5 (none) |
  |---|---|---|---|
  | pillar-crash rate | 0.217 | 0.173 | 0.268 |
  | time-out share | **0.033** | 0.008 | 0.047 |
  | outcome return | 122.9 | 143.2 | 112.7 |

  - All-pillar perception explains part of the gain: about half the crash
    reduction and +10 in outcome, mostly fewer drone losses. The curve
    levels off at about 123.
  - The waiting game stays. The user's hypothesis (fewer crashes ->
    no waiting game) holds only in part.
  - Ablation: the pillar-only branch off costs 15.5 points (crashes 0.217
    -> 0.282), with time-outs unchanged. The full branch off cost 67 (c5).
  - The critic had the full branch and values the long game (EV 0.20), so
    the full graft's gain runs through the actor's prey-relative tokens.
  - From scratch: keep the full relation tokens; the old one-hot attention
    can go.

## 2026-10-19: Graft reference predator frozen

- User: "yes the chasing looks pretty good in replay" (the recommendation:
  freeze only the predator).
- The predator role of the graft self-play's final pair (point 5) is the
  reference predator for the 10-pillar task with the relation graft. The
  prey role is not a reference: it crashes in 42% of the episodes.
- Read-only bundle `.pretrained_checkpoints/graft_predator_2026-10-19/`
  (agent_432000.pt, sha256 002f1cd7...). The bundled copy reproduces
  `point5_own.json` bit-exactly (23 of 23 metrics).
- Loading needs both masks and both relation flags. Doc:
  [graft_predator_2026-10-19.md](graft_predator_2026-10-19.md).

## 2026-10-19: Symmetric graft self-play (pre-registered)

- User approval: "you can start with the recommended symmetric graft
  self-play".
- Question: once the prey also has its relation branch, can it hold cover
  safely and bring the waiting game back, or does it become a safer evader?
- Design: the graft self-play with one change,
  `+agent.models.policy.prey_obstacle_relation=true` (92af76c,
  zero-initialized). The same start (c5's pair), seed 60, recipe and
  evaluations after training.
  - Checks: point 0 must reproduce the asymmetric run's point 0
    bit-exactly; the prey's branch is checked in the first checkpoint.
  - A full-size memory smoke (120 updates at 4096 envs) passed: status 0,
    peak 20.5 GB.
- Decision at points 4 and 5:
  - the reference gate;
  - plus 5: time-outs <= 0.040;
  - plus 6: prey crashes <= 0.33 per episode (asymmetric 0.418).
  Outcomes: all pass -> anchor v5 pair candidate; 5 fails -> the prey brings
  the waiting game back; 6 fails -> the branch does not make the prey safer.
  The decision logic was tested on mock data.
- Started 16:26; about 9 h.
- Evidence: `artifacts/isaac/experiments/2026-10-19_symmetric_graft_selfplay/`.
- Symmetric result (training 16:30-02:01, status 0; evaluations after it,
  all 54 succeeded; point 0 bit-exact with the asymmetric run; the prey
  branch learned): the pre-registered verdict is "with the relation branch
  the prey brings the waiting game back (time-outs > 0.040); decide
  together". Criteria 5 (time-outs 0.126) and 3b (forgetting) fail; the
  others pass, including 6 (prey crashes 0.285).
- **The mechanism is not the old waiting game.**

  | long episodes | symmetric | asymmetric | anchor v4 | k5 |
  |---|---|---|---|---|
  | share | **35%** | 5% | 11% | 9.5% |
  | nearest predator | **1.76 m** | 2.01 m | 6.57 m | 6.52 m |
  | two predators within 4 m of the prey's pillar | **0.74** | 0.49 | 0.10 | 0.06 |
  | pincer | 0.13 | 0.08 | 0.03 | 0.01 |

  - In the time-out episodes, the line of sight to the prey is blocked in
    only 3% of the frames. The prey dodges engaged predators around
    pillars at close range (1.9 rad/s), a "pillar dance". Coordination is
    now the open gap.
  - Both roles use their perception heavily (own ratios: prey 6.4,
    predator 3.8).
  - The predators lose 0.20 drones per episode; the prey runs at 4.8 m/s.
  - Forgetting: the obstacle-blind fixed predator catches the prey in 0.32
    of the episodes (point 0: 0.17).
  - Lesson: time-outs alone no longer identify the strategy. Gates should
    pair them with the long-game mechanism.

## 2026-10-20: Coordination in the pillar dance, horizon vs imposed roles vs control (pre-registered)

- User approval: "yes three-arm test. start with 3 then 2 then 1 pls".
- Three predator-only arms against the frozen symmetric prey (the symmetric
  self-play's final pair). Seed 70, 9600 updates each, run in the user's
  order:
  - H: discount 0.995 (the user's longer-horizon idea);
  - R: imposed random role vectors, norm 10, from the option prepared on
    2026-10-16;
  - C: the control.
  - Evaluations after each arm's training. The driver checks each arm's
    flag in its config.
- Decision per lever at point 5 against C:
  - time-outs <= 0.7x C's and outcome return >= C's + 10: converts;
  - one of the two: partly;
  - guards: drone losses <= 1.2x, own ratio >= 1.0.
  Reported: the dance (two predators at the pillar, pincers), the long game,
  R's role spread and role-norm decay, and C against point 0.
- The decision logic was tested on mock arms.
- Started 07:04; about 12 h.
- Evidence: `artifacts/isaac/experiments/2026-10-20_dance_coordination_test/`.
- Result (H 07:07-10:51, R 10:58-14:43, C 14:50-18:39, all status 0; every
  evaluation succeeded): the pre-registered verdict is "H: partly; R: the
  lever does not convert the dance".

  | point 5 | H (discount 0.995) | R (roles) | C (control) | point 0 |
  |---|---|---|---|---|
  | time-outs | **0.064** | 0.086 | 0.086 | 0.126 |
  | outcome return | **133.4** | 117.6 | 117.9 | 108.4 |
  | drones lost | **0.105** | 0.162 | 0.163 | 0.205 |

  - H: +15.5 over C, mostly from safety (losses -36%, pillar crashes 0.091
    vs 0.134). Time-outs are 26% below C (bar 30%), so "partly". H leads C
    at every checkpoint.
  - R: the role vectors kept their size (norms about 9-10), but behavior
    re-symmetrized: the lead spread fell from 0.077 (r0) to 0.010. The
    result equals C.
  - C against point 0: time-outs 0.126 -> 0.086, so more training alone
    gains against the frozen prey.
  - Pincers stay at 13-16% in every arm; coordination is still open.
  - Next candidates: discount 0.995 as the self-play default, and
    credit-based coordination levers (per-drone counterfactual advantages,
    separate actor weights).

## 2026-10-20: Consolidation self-play, symmetric branches + discount 0.995 + rehearsal (pre-registered)

- User: "i would to consolidate rune (option 1)". The user also asked to see
  arm H in the 3D replay: H and the control are on the dashboard now
  (304cd97).
- **Fix first (5a2fa0c):** per-env pool opponents received no raw states, so
  their masks and relation branches were switched off. They now get raw
  states, as RecurrentMAPPO.act passes them. A test fails on the old code;
  240 tests pass. Earlier pool runs used open-space models without these
  paths and are unaffected.
- **Design:**
  - The start is arm H's final pair (the discount-0.995 predator plus the
    symmetric prey).
  - Five cycles: a joint phase of 3000 updates (both roles, a third of the
    envs with ghosts), then a prey rehearsal phase of 600 updates.
  - In the rehearsal phase, half of the envs face fixed predators: the
    obstacle-blind anchor v3 (weight 2), anchor v4 and the graft predator.
  - Discount 0.995, all relation flags; 18,000 updates.
  - Evaluations after each cycle.
  - Full-size smoke runs of both phase types passed; the pool loaded all
    three policies.
- **Gate at points 4 and 5:**
  - perception, balance and stability as before;
  - no forgetting as absolute bars: the fixed predator vs the prey <= 0.217,
    the predator vs the fixed prey >= 0.111;
  - no waiting game by mechanism: long episodes with the nearest predator
    <= 3 m and the line blocked <= 0.30;
  - prey crashes <= 0.33.
  All pass -> anchor v5 pair candidate. Tested on stand-in data: the
  symmetric run fails exactly 3b.
- Started 19:28; about 10 h.
- Evidence: `artifacts/isaac/experiments/2026-10-20_consolidation_selfplay/`.
- Result (19:28-04:33, all ten phases status 0, every evaluation
  succeeded): the pre-registered verdict is "criteria 3b fail; decide
  together". Seven of eight criteria pass.
  - The rehearsal works but falls just short. The old obstacle-blind
    predator catches the prey in 0.316 -> 0.282 -> 0.263 -> 0.256 -> 0.268
    -> 0.221 of the episodes. The mean of points 4 and 5 is 0.245, against a
    bar of 0.217.
  - The pair at point 5: catch 0.663, outcome return 130.3, prey crashes
    0.231 (symmetric run 0.285), predators lost 0.13. Both roles' ablation
    ratios are 5-10.
  - The game is the pillar dance: long episodes 29%, nearest predator
    1.81 m, line blocked 0.10, two predators at the pillar 0.88, pincers
    0.10, prey at 5.2 m/s.
  - User decision:
    - freeze it as anchor v5 now, with the 3b caveat documented;
    - or two more cycles under the same gate, with a commitment to freeze
      either way.

## 2026-10-21: Anchor v5 frozen

- User: "ok do as recommended", i.e. freeze the consolidated pair now and
  document the 3b miss.
- Anchor v5 is the reference pair with symmetric perception (relation
  branches in both actors and both critics) and discount 0.995. It replaces
  anchor v4 as the current reference pair.
- Read-only bundle `.pretrained_checkpoints/anchor_v5_2026-10-21/`
  (agent_14400.pt, sha256 cc17264a..., 84 files with checksums). The bundled
  copy reproduces `point5_own.json` bit-exactly (23 of 23 metrics).
- Doc: [anchor_v5_2026-10-21.md](anchor_v5_2026-10-21.md). It covers the
  required flags, the lineage, the evaluation against anchor v4, and the
  accepted limit: the obstacle-blind anchor v3 predator catches this prey in
  22-27% of the episodes (bar 21.7%).
- The dashboard now shows anchor v5 as the hero and as a reference.

## 2026-10-21: Does flanking pay in the pillar dance? (pre-registered diagnostic)

- User: "ok continue with 1." (coordination). Before building a credit
  lever, a check on existing trajectories: does a pincer raise the chance of
  success? Pre-registered at 18:05.
  - Engaged orbit frames: two predators within 4 m of the prey's pillar.
  - Success (catch or forced crash) within 1 s, pincer vs one-sided,
    standardized by the nearest-predator distance.
  - 90% bootstrap over episodes.
- Result on anchor v5: RR 1.34 [1.13, 1.58], i.e. 0.238 vs 0.178. The
  pre-registered reading is **"weak"** (bar 1.5).
  - Point 4: 1.18 [0.98, 1.41].
  - The symmetric self-play: 1.24 [1.04, 1.49].
  - The dance arms against the frozen symmetric prey: 0.92-1.10 (no
    effect).
- A rough ceiling: if every engaged moment were a pincer, anchor v5's
  time-outs would fall from about 0.10 to about 0.07. That is a small target
  for a coordination lever. Part of the remaining time-outs is likely the
  intended balance with the more agile prey.
- Evidence: `artifacts/isaac/diagnostics/2026-10-21_flanking_value/`.

## 2026-10-21: Separate decision heads per drone, the cheap coordination test (pre-registered)

- User: "1." (the cheap coordination test, after the flanking diagnostic).
- Code f2368f6: `predator_drone_heads` replaces the shared actor head with one
  head per drone index on top of the shared perception and GRU.
  - A drone's head learns only from its own actions (agent-specific
    parameters, Fu et al. 2022).
  - `split_actor_head` converts a trained checkpoint into exact copies,
    which act as before up to float rounding.
  - The two layouts refuse to load into each other. Off by default; 244
    tests pass.
- Design: predator-only against the frozen anchor v5 prey, seed 70, 9600
  updates, discount 0.995.
  - Arm S: drone heads plus per-drone PPO ratios, from the converted start.
  - Arm C: the control.
  - Evaluations after training. A full-size smoke of arm S passed (both
    flags in the config).
- Decision at point 5, S against C:
  1. specialization: per-index lead or side spread >= 0.15;
  2. pincer share >= 1.5x C;
  3. outcome: time-outs <= 0.7x and return >= +10 (converts), one of them
     (partly);
  4. guards.
  The flanking ceiling is small, so "no outcome gain" is expected even if 1
  and 2 pass.
- Started 18:14; about 8 h.
- Evidence: `artifacts/isaac/experiments/2026-10-21_drone_heads_test/`.
- Result (S 18:17-22:19, C 22:27-02:11, both status 0; every evaluation
  succeeded): the pre-registered verdict is "the heads do not specialize;
  pincers do not rise; outcome: no outcome gain". S ends below C.

  | point 5 | S (heads + per-drone ratios) | C (control) | start |
  |---|---|---|---|
  | lead / side spread | 0.020 / 0.012 | 0.023 / 0.005 | 0.011 / 0.008 |
  | pincer share | 0.119 | 0.094 | 0.102 |
  | time-outs | 0.091 | **0.053** | 0.104 |
  | outcome return | 135.5 | **148.5** | 130.3 |

  - The heads diverged in parameters (each 3.6% from the start, 3.8% apart)
    but not in behavior. That is drift from gradient noise, not roles.
  - S updated about three times less than C (KL 0.0018 vs 0.0050, clip
    fraction 0.011 vs 0.036, entropy -0.37 vs -0.64). Each head learns from
    a third of the data.
  - Pincers stop paying once the predators train on the frozen prey (RR 1.0;
    1.34 at the start). The control alone halves the time-outs: exploitation.
  - Together with the imposed roles and the flanking diagnostic: coordination
    is not the bottleneck of the full-information dance. Not adopted.

## 2026-10-22: Stage 8b from scratch, seed 81 (pre-registered)

- **User:** "pls start with stage 8b. i want to have a exact plan how you could retrain from scratch". On the plan
  (c700e9e): "yes i agree" to recommendations D1-D4.
  - D1: anchor v5's configuration unchanged.
  - D2: an LR pilot on seed 81.
  - D3: own history and ghosts as the only training opponents.
  - D4: seed 81, a review, then seeds 82 and 83.
- **Plan:** [stage8b_from_scratch_protocol.md](stage8b_from_scratch_protocol.md). Facts found while writing it:
  - Parked pillars poison the input normalizer: (50, 50) m enters the running statistics, and real pillars would later
    be clipped at −5. So pillars are active from the first update.
  - `hysteresis_curriculum.py` cannot pass the relation flags (no Hydra passthrough).
  - With relation branches, joint training takes 30-32 min per 1000 updates.
  - The July flight's LR was 1e-4, read from its checkpoints' optimizer state.
- **Recipe:**
  1. Flight among all 10 pillars from random init: chase rewards off, catches off, LR 1e-4. Gate: drones lost ≤ 0.10,
     prey lost ≤ 0.05, length ≥ 490; at most 6000 updates.
  2. An LR pilot: 1e-4 / 3e-5 / 1e-5, two paired 1200-update blocks each.
  3. The game: 24,000 updates in 1200-update blocks.
     - Joint blocks with ghosts by default.
     - A band guard trains only the predator below catch 0.25 and only the prey above 0.75, against own past
       checkpoints.
     - An interim stop at 12,000 updates.
  4. Anchor v5's consolidation (18,000 updates, LR 1e-5).
  5. The final gate: perception, balance, a duel with anchor v5 (D = R(ours vs v5's prey) − R(v5 vs our prey) ≥ −10),
     stability, the waiting-game mechanism, safety.
- **Code ceeaefd:**
  - `scripts/recipes/stage8b/`: `block.sh`, `run_seed.py` (resumable; runs a frozen copy), `recipe.py` (the decisions);
  - 11 tests; 255 pass.
- **Full-size smoke** (seed 80, 30-update blocks, `smoke/`): every check passed.
  - Two same-seed flight blocks gave 13,210 of 13,210 training scalars bit-identical.
  - Two amendments came out of it:
    - losses against ghosts count crashes (a crashing prey had looked like a perfect evader);
    - evaluation passes the training LR multiplier (the preset's ×30 broke loading the optimizer state).
- **Run:** seed 81 started 17:27; about 30 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-22_stage8b_from_scratch/`.
- **Result** (2026-10-22 17:27 → 10-23 20:01, every status 0, every evaluation succeeded): the pre-registered
  verdict is "fail: 2 balance, 3 duel with anchor v5, 5 no waiting game".

  | point c5 | seed 81 | anchor v5 |
  |---|---|---|
  | own catch | 0.033 (c4 0.012) | 0.663 |
  | time-outs | 0.94 | 0.10 |
  | long episodes: nearest predator / line blocked | 3.1 m / 0.26 | 1.8 m / 0.10 |
  | duel with v5: R(ours vs v5's prey) / R(v5 vs our prey) | 15.6 / 51.8 (v5's own: 130) | — |

  - Flight passed after 4500 updates. The pilot chose 1e-4. The 1e-5 arm scored higher, but its predators lost 0.83
    drones per episode, which the arm-relative loss guard cannot flag (a gap in the rule).
  - In the game, the band guard ran 14 of 18 blocks predator-only. The predators learned each frozen prey (catch up to
    0.82); every joint or prey block let the prey escape again.
  - The consolidation (no guard) ended at catch 0.01-0.03.
  - Mechanism: a standoff paid by the upright term (the review's item 6, B7).
    - The team earns +0.072 per step in standoffs, mostly the upright term, worth 18.3 at discount 0.995. The critic's
      standoff value is 17.7.
    - Catches are out of reach against a prey that is more agile, has 5× stronger distance shaping and has a ghost
      curriculum.
    - The pursuit skill decayed: interception of a hovering prey fell from 0.99 to 0.07.
  - The networks trained from scratch represent the game well:
    - closing speed in the GRU output 0.66 (v5 0.39);
    - line of sight 0.69 (v5 0.64);
    - soft entity attention.
  - Evidence: `RESULTS.md`, `seed81/`.

## 2026-10-23: Upright mechanism test, are the from-scratch predators paid to wait? (pre-registered)

- **User:** "yes do the mechanism test as recommended".
- **Hypothesis:** seed 81's standoff is paid by the predators' upright term.
  - +0.09 per step, worth 18.3 at discount 0.995, which is the critic's standoff value.
  - Catches are rare against the stronger prey, so waiting beats pressing at the risk of −400 per drone.
- **Design:**
  - from seed 81's final pair, two paired joint arms, 4 × 1200 updates each (LR 1e-4, ghosts in a third of the envs,
    the recipe's flags);
  - **U:** `env.predator_upright_zero_at_level=true`. In a smoke run its upright term averaged −0.006 per step against
    +0.096 in the control;
  - **C:** the control.
  - Code 345dc85: `block.sh` takes ablation overrides.
- **Decision at point 4:**
  1. U's own catch ≥ C's + 0.10;
  2. U's long-episode nearest predator ≤ 2.5 m and below C's;
  3. guard: U's drones lost ≤ 0.37.
- **Start:** 20:42; about 6 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-23_upright_mechanism_test/`.
- **Result** (U 20:42-23:31, C 23:31-02:04, every status 0): the pre-registered verdict is "the upright stream is not
  what holds the predators back".

  | point 4 | U (upright zero at level) | C (control) | start (s81 c5) |
  |---|---|---|---|
  | own catch | 0.002 | 0.002 | 0.033 |
  | long episodes: nearest predator | 2.93 m | 2.64 m | 3.11 m |
  | long episodes: closing speed | 0.77 m/s | 0.05 m/s | 1.13 m/s |
  | prey at its pillar: angular speed | 3.2 rad/s (dance) | 0.13 rad/s (hiding) | 1.9 rad/s |
  | drones lost | 0.059 | 0.018 | 0.040 |

  - The stream shapes the style, not the outcome. Without it the predators keep moving and the prey dances; with it
    the prey hides and the predators wait. Neither arm catches, even under stochastic training actions.
  - So the paid wait in seed 81 was a symptom. While the predators cannot catch this prey, pressing earns about 0 and
    risks −133 per drone, with or without the bonus.
  - The bottleneck is the arms race: in seed 81, predator-only blocks raised the catch rate against a frozen prey to
    0.82 within 6000 updates, and one block of prey training wiped that out.
  - Next (user decision):
    - a slower-learning prey (two time scales; skrl takes the learning rate per role);
    - or a prey-agility curriculum toward thrust-to-weight 2.2;
    - or Stage 9 on anchor v5.

## 2026-10-24: League continuation of seed 81, training against the opponents' whole history (pre-registered)

- **User:** "but we are still training against a pool or? ... cause then they learn in a zeno behaviour. i wouldn't
  make the prey learning slower. i would just let the training go longer until the predator learned all prey
  strategies or the 'average' prey strategy"; then "yes".
- **Finding (true):** in seed 81 both roles learned almost only against the latest opponent.
  - Joint blocks have no pool: 56% of the predators' updates, 83% of the prey's.
  - The single-role pools ("the last 4 points") held copies of frozen roles. Hashed weights show 25 distinct predators
    but only 12 distinct prey; at g13 the pool was four copies of the current prey.
  - Past prey were about 13% of the predators' experience, never more than 3 distinct ones.
- **Design (fictitious play):** from seed 81's final pair, alternating phases of 1200-update blocks.
  - The training role faces 8 opponents sampled uniformly from the other role's distinct past versions in 75% of the
    envs, and the latest version in the rest.
  - The phase lasts until the role beats that average: predators catch ≥ 0.5, prey caught ≤ 0.5; at most 6 blocks.
  - Same LR (1e-4) for both roles.
  - Each phase starts with the training role's own optimizer state. train.py does not save a frozen role's optimizer,
    and a fresh Adam made the smoke run's prey crash in half of its episodes after 30 updates.
  - Stop: at equilibrium (two cycles with one-block phases) or at 48,000 updates. Point 0 adds a cross-play matrix of
    seed-81 versions.
- **Decision:**
  1. the final predators catch ≥ 0.4 of every archived prey version, on average;
  2. own catch within 0.3-0.7 at the last two cycle ends;
  3. duel with anchor v5 ≥ −10;
  4. guards.
- **Code:** 39c3067 (`run_league.py`, league functions, `BLOCK_POOL_PROB`). 259 tests pass; two smoke runs.
- **Start:** 09:48; 12-22 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-24_league_continuation_seed81/`.
- **Result** (09:48-16:07, every status 0): the pre-registered verdict is "fail: criteria 2" (balance). The league
  stopped at its "equilibrium" rule after 3 cycles, at 10,800 of 48,000 updates.

  | | L0 (s81 c5) | final (L3) | bar |
  |---|---|---|---|
  | predators against every archived prey, mean catch | 0.082 (12 prey) | 0.449 (15); 0.532 on the same 12 | ≥ 0.4 pass |
  | own catch | 0.033 | 0.024 (L2), 0.069 (L3) | 0.3-0.7 fail |
  | duel with anchor v5 | −36.3 | +16.2 (mean L2, L3) | ≥ −10 pass |
  | drones lost / prey lost | 0.040 / 0.032 | 0.187 / 0.100 | guards pass |
  | interception of a hovering prey | 0.072 | 0.996 | |

  - **The history stopped most of the forgetting.** Point 0's matrix showed the cycling: each seed-81 predator catches
    the prey it trained on, and little else. The final predators beat the old prey 6.5 times as often as c5 did, and
    interception came back and stayed. An opponent not drawn into a phase's pool can still be lost (c5 prey: 0.74
    against L2, 0.39 against L3).
  - **The prey has one answer to every predator: orbiting a pillar** (0.64 of long-episode frames within 1 m of a
    pillar's surface).
    - A panel of 8 predators catches the final prey at 0.047, and anchor v5's predators score R 18.5 against it.
    - The final predators catch the three league prey at 0.20, 0.08 and 0.07.
    - Engagement, not flanking geometry, is missing. Two predators are within 4 m of the prey's pillar in 0.44 of the
      orbit frames (v5: 0.88). When they are, they flank as often as v5's (0.14 against 0.12).
    - When they commit, they crash: closing at 2.7 m/s from 2.0 m, the pillar in the line in 42% of the losses.
  - **The stop rule was not an equilibrium test.**
    - The prey's bar (caught ≤ 0.5) passed every phase with 0.02-0.04.
    - The predators' gate left out the latest prey.
    - The best responses were still about 0.40 apart: predators against the prey history 0.449, prey against the
      predator panel 0.047.
  - Without ghost envs, the prey's survival against ghosts fell from 0.82 to 0.12. Its safety now rests on pillars.
  - Next (user decision):
    - an exploiter test (the final predators against the frozen final prey only, ≤ 6 blocks, about 4.5 h): is the
      orbit breakable at all?
    - or league v2 directly, with a best-response-gap stop, the latest opponent in every gate and loss-weighted
      pools.
  - Evidence: `RESULTS.md`, `league/`.

## 2026-10-24: Exploiter test, can focused predators break the final prey's pillar orbit? (pre-registered)

- **User:** "yes start the test".
- **Question:** the league's final prey orbits a pillar, and no predator catches it (own catch 0.069). In each league
  phase the predators met the latest prey in only a quarter of their envs, for 1200 updates. Is the orbit breakable at
  all with focused training?
- **Design:**
  - from the league's final pair: the predators with the Adam state of their last training block, against the frozen
    final prey in every env (pool share 0, so train.py skips the pool);
  - blocks of 1200 updates with the league's predator settings (LR 1e-4, KL 0.05, discount 0.995, no ghosts);
  - after each block, a gate: catch against the final prey, 512 episodes;
  - stop at a gate ≥ 0.5 with drones lost ≤ 0.37, or after 6 blocks;
  - end: point E (full set, 1024 episodes), the catch against all 15 archived prey versions, the strategy analyses.
- **Decision at E** (the exploiter against the final prey):
  1. catch ≥ 0.4 and drones lost ≤ 0.37: "the orbit is breakable by focused training". Next proposal: league v2 with
     exploiters, loss-weighted pools, the latest opponent in every gate and a stop on the best-response gap;
  2. catch < 0.15: "focused training does not break the orbit". Next proposal: an exploration lever (hard start states)
     or a rule decision;
  3. otherwise "partial"; the gate curve decides the reading.
- **Code:** the recipe backend at da3e56a (frozen copy), driver `run_exploiter.py` in the experiment directory. A dry
  run against a fake backend checked the stop rule, the guard and resuming.
- **Start:** 19:03; at most 4.5 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-24_exploiter_test_seed81/`.
- **Result** (19:03-19:45; the first gate passed, so one block of 1200 updates): the pre-registered verdict is "the
  orbit is breakable by focused training".
  - Catch against the final prey 0.069 → 0.603 (bar ≥ 0.4), drones lost 0.174 (guard ≤ 0.37). The prey was bit-exact
    the league's (anchor v5's predators score R 18.49 against it, as before).
  - **Mechanism: engagement from one side, not pincers.**
    - Two predators within 4 m of the orbit pillar 0.44 → 0.70.
    - Distance to the prey 3.9 → 2.9 m, closing speed 0.49 → 1.12 m/s.
    - One-sided catch chance within 1 s 0.009 → 0.144. Pincers convert worse (0.040) and stay rare (0.14 → 0.17).
    - Long-episode losses rise from 0.20 to 0.26 per episode; drones lost per episode stay at 0.17.
    - This corrects the league's reading that a second side is needed.
  - **Retention: the exploiter forgot the consolidation-era prey.**
    - Orbit prey (L1-L3) 0.12 → 0.41; consolidation prey (c1-c5) 0.61 → 0.10; all 15 versions 0.449 → 0.296.
    - Each predator beats one family, neither both (non-transitivity).
  - The league failed by dilution, not by capability. The main pair's exploitability is at least 0.53 (0.603 − 0.069).
  - Next (user decision): a two-counter test (≤ 3.2 h). From the exploiter: 25% final prey, 75% the 8 prey versions
    it catches least, weighted (1 − catch)² and re-weighted after each block; gate on all 15. Can one predator hold
    the orbit and the consolidation counters at once (both families ≥ 0.4)?
  - Evidence: `RESULTS.md`, `exploiter/`.

## 2026-10-25: Two-counter test, can one predator hold the orbit and the consolidation counters at once? (pre-registered)

- **User:** "yes", to the exploiter test's proposal.
- **Question:**
  - The exploiter broke the final prey's pillar orbit (0.07 → 0.60) and forgot the consolidation-era prey
    (0.61 → 0.10).
  - The league's final predators hold the opposite pair, so each predator beats one prey family.
  - Can one predator hold both counters if it trains against the prey it loses to?
- **Design:**
  - **Start:** the exploiter, with its Adam state.
  - **Opponents:** the frozen final prey in 25% of the envs. In the other 75%, the 8 of the other 14 archived prey
    versions it catches least, weighted (1 − catch)², as in AlphaStar's prioritized fictitious self-play.
    - The weights are recomputed after each block from the gate; the first block's come from the exploiter's
      retention measurement.
    - The first pool is g8, c5, g2, c1, c4, c3, c2, L1prey. About 46% of the envs face consolidation prey, 33% orbit
      prey and 21% g2 or g8.
  - **Training:** blocks of 1200 updates with the league's predator settings, at most 4.
  - **Gate after each block:** catch and drones lost against all 15 prey versions, 512 episodes each.
  - **Stop:** both families' mean catch ≥ 0.4 (orbit L1-L3, consolidation c1-c5), with drones lost ≤ 0.37 in each.
  - **End:** point T (the full set, 1024 episodes) and the strategy analyses against the exploiter, the league's final
    and anchor v5.
- **Decision:**
  - Both families ≥ 0.4, with drones lost ≤ 0.37 in both and at T: "one predator holds both counters". Next proposal:
    league v2 on single main agents with prioritized pools and exploiters.
  - Otherwise "the counters did not combine". Next proposal: league v2 with a predator population.
  - The last block's change gives the reading: trade-off, still rising (a budget question) or stalled.
- **Guard detail:** the guard is per family because g8 costs the exploiter 2.4 drones per episode. A mean over all 15
  prey versions would be 0.39 at the start because of g8 alone.
- **Code:** the recipe backend, unchanged since da3e56a (frozen copy), driver `run_two_counter.py` in the experiment
  directory. A dry run against a fake backend checked the pools, the stop rule, the block checks, resuming and the
  decision.
- **Start:** 2026-10-25 00:29; at most about 3.2 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-25_two_counter_test_seed81/`.
- **Result** (00:29-02:01; the second gate passed, so 2 of the 4 blocks were used): the pre-registered verdict is "one
  predator holds both counters".
  - Mean catch at the second gate: orbit family 0.578, consolidation family 0.414 (bar ≥ 0.4). Drones lost 0.14 and
    0.15 per episode, and 0.142 at T (guard ≤ 0.37). The prey stayed the league's final prey (fingerprint verified;
    anchor v5's predators score R 18.49 against it, as before).
  - **Both families rose in both blocks:** orbit 0.41 → 0.53 → 0.58, consolidation 0.10 → 0.34 → 0.41, all 15 versions
    0.30 → 0.42 → 0.47. There was no trade-off, so the exploiter's forgetting came from training against a single
    opponent, not from conflicting counters.
  - **The consolidation pass is thin:** 1.4 standard errors above the bar (SE 0.010 over 5 × 512 episodes), and two
    thirds of the league final's level (0.61).
  - **Against the two specialists:**
    - In catch, T is below their per-prey best (0.472 against 0.518): the league's final predators still catch c2-c4
      at 0.73-0.81 (T 0.41-0.50).
    - In outcome return, T is above it (R 83.8 against 74.1 over all 15), because it loses half the league final's
      drones (0.11 against 0.24 per episode).
  - **Own matchup (point T, 1024 episodes):**
    - Catch 0.704 (E 0.603), time-outs 0.187, R against anchor v5's prey 53.6, duel with v5 +35.2 (E +13.0).
    - It is the exploiter's one-sided pressure, refined: one-sided catch chance within 1 s 0.180 (E 0.144).
    - Pincers convert better (relative risk within 2 s 0.56 → 0.82) but remain rare (0.16). No roles.
  - **What PFSP got wrong:**
    - Weighted by catch, hardness gave g2 and g8 29% of the second block's envs, and their catch never moved
      (0.07, 0.03).
    - g2 ends 77% of its episodes by crashing on its own, with no predator forcing it.
    - g8 lures predators into pillars (the exploiter lost 2.4 drones per episode against it). T declines: 0.11 drones
      lost, the best outcome return of the three predators (R −6.5).
    - Drones lost fell against every family (all 15: 0.39 → 0.11).
  - Next (user decision): league v2 on single main agents:
    - PFSP pools for both roles;
    - exploiters as league members;
    - the latest opponent in every gate;
    - a stop on the gap between the best responses;
    - pool rules from this run: drop self-crashing versions, and a version whose catch has not moved for two blocks
      falls back to the uniform weight.
  - Evidence: `RESULTS.md`, `two_counter/`.

## 2026-10-25: League v2, do two main agents with prioritized pools and exploiters approach an equilibrium? (pre-registered)

- **User:** "do the scale run", to the two-counter test's proposal.
- **Question:**
  - The two-counter test showed that one predator can hold the counters to every prey family when it trains against
    the prey it loses to. But the prey stayed frozen.
  - Both main agents now train in turn against the other role's whole history, with prioritized pools, and exploiters
    of both roles join the league.
  - Does the main pair approach an equilibrium, i.e. a small gap between the two best responses?
- **Design:**
  - **Start:** T (the two-counter predators) and the league's final prey, each with its Adam state.
  - **Archives:** 30 predator versions (the league's 28, E and T) and 15 prey versions. Every new main version and
    every exploiter joins them.
  - **Cycle, at most 4:**
    1. a prey block, then a predator block, 1200 updates each. Each faces the latest opponent in 25% of the envs; in
       75%, a pool of the 8 versions with the largest weights, (1 − catch)² for the predators and catch² for the prey;
    2. gates against every archived opponent version (512 episodes);
    3. one-block exploiters from each main agent, against only the frozen other one;
    4. a cycle point.
  - **Pool rules from the two-counter test:**
    - self-crashing prey versions are dropped (g2);
    - a version without progress over the role's last two phases is capped at weight 0.25 (g8 from the start).
  - **Gap:** BR_pred − BR_prey, the predator exploiter's catch against the main prey minus the main predator's catch
    against the prey exploiter. It is a NashConv with one-block best responses.
- **Stop:** the first cycle with a gap ≤ 0.2, or after 4 cycles.
- **Decision:**
  - **Pass:** gap ≤ 0.2, with the guards at the final point (drones lost ≤ 0.37, prey lost ≤ 0.33, duel with v5
    ≥ −10). Verdict: "the league approaches an equilibrium". Next: the anchor gate for the final pair.
  - **Otherwise:** "not converged within 4 cycles", with a reading:
    - shrinking (the gap fell by at least 0.1): next, more cycles;
    - not shrinking: next, a population of main agents;
    - and which side is more exploitable.
- **Code:** the recipe backend, unchanged since da3e56a (frozen copy), driver `run_league_v2.py` in the experiment
  directory. A dry run against a fake backend checked:
  - the pools and their two rules, and the composed start checkpoints;
  - the block checks and the gap stop;
  - resuming after an interruption, and the decision.
- **Start:** 2026-10-25 08:49; at most about 10.5 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-25_league_v2_seed81/`.
- **Result** (08:49-18:35, all 4 cycles; the gap stop never fired): the pre-registered verdict is "not converged within
  4 cycles". The reading is "not shrinking" (gap 0.846 → 0.691 → 0.918 → 0.971); at the last cycle the main predator
  is the more exploitable side. All guards pass at the final point: drones lost 0.027, prey lost 0.035, duel with v5
  −5.9.
  - **Exploiters:** a one-block exploiter always found a counter. In every cycle at least one moved the catch by
    ≥ 0.69.
    - Cycle 2's prey exploiter got worse than its start (0.195 against Y2's 0.102), so with the best known
      responses the gap was ≥ 0.78 in every cycle.
  - **The main agents cycle:** each version beats one or two of the other role's versions, rock-paper-scissors at the
    level of policies.
    - T catches L3prey at 0.69 and the Y versions at ≤ 0.16.
    - P3 catches Y1 and Y3 (0.74, 0.90) and Y2 and Y4 at ≤ 0.16; P4 the reverse (0.97 / 0.97 against ≤ 0.07).
  - **The pools explain the forgetting:**
    - Each phase gained against its pool and lost against the rest, within one block. The predators' last phase
      gained +0.43 in catch inside its pool and lost 0.45 outside it (Y3 0.90 → 0.02, c4 0.64 → 0.01).
    - Hard top-8 prioritization drops what the learner beats, which then comes back after it is forgotten:
      the predators against the consolidation prey went 0.41 → 0.62 → 0.29 → 0.56 → 0.02.
  - **g8 is catchable:** with 13% of the envs after its no-progress cap lapsed, P4 caught it at 0.85 (every earlier
    main predator ≤ 0.04).
  - **Mixtures (in-sample meta-game over measured matchups):**
    - A mixture of the 5 main predators catches every archived prey version but g2 at ≥ 0.305; the best single one
      reaches only 0.043.
    - A mixture of the 5 main prey is caught by every one of the 34 predator versions at ≤ 0.383; the best single
      one, at up to 0.693.
    - Fictitious play converges in the average, not in the last iterate. The gap measured last iterates.
  - **Final pair:**
    - Own catch 0.966 in 2.4 s, the predators' best case since they trained last.
    - A hovering prey is caught only 0.763 of the time (T 0.984).
    - The duel with v5 fell from +92 (C2) to −6, because v5's predators beat the newer prey (R 68.4 against Y4).
    - PPO was healthy (KL ≤ 0.0095, explained variance 0.48-0.78).
  - **Caveat:** in cycle 2, training (sampled actions) and gates (mean action) disagreed. XY2 improved in training
    and got worse at the gate.
  - **Next (user decision, pre-registered "a population of main agents"):** a PSRO pilot.
    1. Fill the league v2 population's cross-play matrix (about 40 matchups, about 15 min) and solve the meta-Nash
       mixture.
    2. Train one exploiter per role against the other's mixture (about 1 h). Compare the mixture's gap with the
       single agents' 0.78-0.97.
  - Evidence: `RESULTS.md`, `analysis/`, `league_v2/`.

## 2026-10-25: Is league v2's opponent pool already the best mix? (pre-registered)

- **User:** "i thought with the pool we already keep the best mix (how i understand psro). pls test that".
- **Question:**
  - PSRO plays a mix over its population that is computed by solving the game on the table of every version against
    every version, and new versions train against that mix.
  - League v2's pool is also a mix: the latest opponent at 25% and the 8 versions the learner currently loses to most
    at 75%.
  - Is the pool as hard to beat as the computed best mix?
  - A good mix stays hard to beat even for a learner trained one block only against it.
- **Known before the run** (league v2's evaluations): every pool was beaten after one block of training against it.
  - The predators' catch against the four prey pools rose to 0.58, 0.30, 0.55 and 0.79.
  - The four predator pools caught the prey that trained against them at 0.06, 0.05, 0.05 and 0.08.
  - Round 4: P4 catches its pool at 0.791 and the pool catches Y4 at 0.084, so the pool gap is 0.708.
- **Design:**
  - **The best mixes**, computed before the run from league v2's evaluations (regret matching+ over each role's 5
    main versions, against every archived opponent with complete data):
    - predators: P1 0.36, P4 0.33, P3 0.18, T 0.08, P2 0.05 (every archived prey but g2 caught at ≥ 0.301);
    - prey: L3prey 0.49, Y1 0.31, Y4 0.20 (caught at ≤ 0.386 by every archived predator).
  - **Two challengers with the starts and budget of P4 and Y4:**
    - BP: P3 + one block against the prey mix;
    - BY: Y3 + one block against the predator mix;
    - block seeds 81901 and 81902.
  - **Evaluations:** each challenger against every member of both mixes, the best mix's and the round-4 pool's
    (512 episodes, mean actions).
- **Decision:** best-mix gap = BP against the prey mix − the predator mix against BY.
  - ≤ 0.508 (the pool gap − 0.2): "the pool is not the best mix". Next: the PSRO pilot.
  - Below 0.908: "no clear difference".
  - Otherwise: "the pool resists better".
  - Reported without a bar: the transfer between the mixes. P4 catches the best prey mix at 0.381 (P3: 0.386). The
    best predator mix catches Y4 at 0.401 (Y3: 0.301).
- **Code:** the recipe backend, unchanged since da3e56a (frozen copy), and the driver `run_mix_test.py`. A dry run
  against a fake backend checked:
  - the composed starts and the pool shares (0.51 with 2 policies, 0.64 with 4);
  - the block checks and the evaluation names;
  - resuming and the decision.
- **Start:** 2026-10-25 19:23; about 1.4 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-25_pool_vs_best_mix_seed81/`.
- **Result** (19:23-20:31): by the pre-registered rule "the pool is not the best mix" (best-mix gap 0.423, bar
  ≤ 0.508). But only the predator half is valid.
  - **Predators (valid):**
    - One block against the best prey mix lifted the catch from 0.386 to 0.642 (outcome return 49 → 94).
    - Against the pool, the league's round-4 predators went from 0.157 to 0.791 (9 → 145).
    - The block mostly learned one counter, against the mix's largest member: start prey 0.26 → 0.70; round-4 prey
      only 0.14 → 0.23.
    - No transfer: P4 against the best mix 0.381 (P3 0.386); the challenger against the pool 0.189 (P3 0.157).
  - **Prey (invalid):**
    - The challenger did not learn: pillar crashes in training 0.41 → 0.43, return −17 → −57. The league's Y4, from
      the same start, cut its crashes 0.30 → 0.08.
    - Its lower caught rate in the evaluation (0.301 → 0.218) is crashing on its own (0.039 → 0.368). Counting
      crashes as losses, it was lost more often than its start (0.347 → 0.649).
  - **The same blind spot in league v2:**
    - Counting prey crashes as losses, league v2's gap was 0.50 → 0.76 → 0.80 → 0.98, against 0.85 → 0.69 → 0.92
      → 0.97 by catch.
    - The cycle-1 prey exploiter escaped mostly by crashing (unforced 0.42).
    - The verdict "not converged" stands (addendum in league v2's `RESULTS.md`).
  - **Next (user decision):**
    1. Redo the prey half with a new seed and a rule that counts crashes as losses (about 35 min).
    2. Then the PSRO pilot.
  - Evidence: `RESULTS.md`, `summary.txt`, `mix_test/`.

## 2026-10-26: The pool-vs-best-mix test's prey half, run again (pre-registered)

- **User:** "yes do 1 and 2" (1: this redo, 2: the PSRO pilot, which follows in its own entry).
- **Why:** the mix test's prey challenger BY failed to learn.
  - It crashed into pillars in about 42% of its training games from start to end, and on its own in 37% of the
    evaluation episodes.
  - The catch-only measure counted those crashes as escapes.
- **The measure from now on: prey lost = caught + crashed** (pillar or arena).
  - Unforced crashes = prey lost − predator success.
  - Counting crashes, the round-4 pool gap is 0.838 (P4 against its pool) − 0.117 (the pool against Y4) = 0.721.
  - BP, the mix test's predator challenger, loses the best prey mix's prey at 0.728.
- **Design:**
  - BY2 = Y3 (with its Adam state) + one prey block against the unchanged best predator mix (P1 0.36 frozen; pool P4
    0.33, P3 0.18, T 0.08, P2 0.05; share 0.64).
  - Block seed 81903.
  - BY2 is evaluated against the 11 predator versions of both mixes (512 episodes, mean actions).
- **Decision:**
  - **Validity first:** BY2's unforced crashes against the best mix must be ≤ 0.10 (Y3 0.039, BY 0.368).
    Otherwise "the prey challenger failed again", and the PSRO pilot replaces the proposed Y4 arm.
  - **Then the best-mix gap counting crashes** = 0.728 − BY2's lost against the best mix:
    - ≤ 0.521: "the pool is not the best mix";
    - below 0.921: "no clear difference".
  - **Reported without a bar:**
    - the prey half alone (BY2 against the best mix next to Y4's 0.117, margin 0.1);
    - training health next to BY and Y4;
    - the catch-only gap.
- **Code:**
  - The recipe backend is unchanged (frozen copy); the driver is `run_prey_redo.py`.
  - A dry run against a fake backend checked the start (P1 + P1 + Y3), seed 81903, the pool (0.64, 4 policies),
    the 11 evaluation names, resuming, and the three verdicts.
- **Start:** 2026-10-26 15:59; about 35 min.
- **Evidence:** `artifacts/isaac/experiments/2026-10-26_prey_half_redo_seed81/`.
- **Result** (15:59-16:31): **"the prey challenger failed again"**. The prey half of the question stays open.
  - BY2 crashed on its own in 0.30 of its games against the best predator mix (bar ≤ 0.10; Y3 0.04). Counting
    crashes, it was lost in 0.77 (Y3 0.35); against P4, which almost never beats Y3 (0.07), it was lost in 0.63.
  - Training: pillar crashes 0.32 → 0.42 against every member of the mix, and the prey's return fell −4 → −81. The
    entropy still fell (−0.56 → −1.18), so the noise-free policy itself crashes more.
  - Every league v2 prey block cut its caught rate and lengthened its games in training. In evaluation, one of them
    also crashed on its own: XY1, against P1 alone, at 0.42. P1 is among the opponents of all three failures; Y2,
    with P1 at 25%, did not fail.
  - Not a reward trick (caught −200, crash −300 for the prey).
  - Not the start checkpoint: compose copies the prey's whole block from Y3, as in the league's Y4 block.
  - The cause is open. The PSRO pilot (running since 16:31) has the same risk; its validity check marks such an
    iteration. If its prey learners break too, the next diagnostic is one prey block from Y4 against the same mix.
  - Evidence: `RESULTS.md`, `summary.txt`, `redo/`.

## 2026-10-26: PSRO pilot, does training against the computed best mix close the gap? (pre-registered)

- **User:** "yes do 1 and 2" (2: the PSRO pilot, after the prey half's redo).
- **Question:**
  - League v2 trained its newest versions against a pool and played the newest version. One block of training beat
    those versions far: the gap counting crashes was 0.50 → 0.76 → 0.80 → 0.98.
  - PSRO plays the mix that is solved from the table of every version against every version, and trains new versions
    against that mix.
  - Does the gap of PSRO's mix fall over iterations?
- **Payoff:** prey lost = caught + crashed (pillar or arena), 512 episodes per cell (seed 42, mean actions). Earlier
  evaluations of a pair are reused (pairs measured twice agree exactly).
- **Starting population:**
  - predators T, P1-P4, XP1-XP4, BP;
  - prey L3prey, Y1-Y4, XY1-XY4, BY, BY2 (if step 1 produced it).
  - The older archive is left out (about 4 h of evaluations).
- **One iteration (at most 4):**
  - Solve the table with regret matching+. The mixes are rounded to hundredths, with at most 9 members.
  - Train one block per role from the newest version with weight ≥ 0.05 in its own mix (Adam state kept) against
    the other role's mix. The largest member is the frozen opponent, the rest the pool.
  - Block seeds 81911/81912, 81921/81922, …
  - Add the new versions (PP_k, PY_k) and fill the table.
- **The gap of iteration k** = max(PP_k against q_k, best old predator against q_k) − min(p_k against PY_k, p_k
  against its hardest old prey).
- **Validity of an iteration:** PY_k's unforced crashes ≤ 0.10, and PP_k's drones lost ≤ 0.37 per episode against
  the mix.
- **Stop:** a valid gap ≤ 0.2, or 4 iterations.
- **Decision:**
  - "PSRO converges for one-block learners" (a valid gap ≤ 0.2);
  - "PSRO narrows the gap" (last valid ≤ first valid − 0.2);
  - "no narrowing in the pilot" (otherwise, with ≥ 2 valid iterations);
  - "undecided".
- **Reported:**
  - the gap series next to league v2's;
  - the gains per side;
  - whether the new versions enter the next mix;
  - training health;
  - iteration 0 (the mix tests' challengers);
  - the final mixes.
- **Code:**
  - The recipe backend is unchanged (frozen copy); the driver is `run_psro_pilot.py`.
  - A dry run against a fake backend checked the population and table reuse, the solver (it reproduces league v2's
    5 × 5 mix), the starts (a new version with no weight is skipped), the composes and pool shares, the block checks,
    the stop rule, an invalid iteration, resuming, a single-member mix (share 0, no pool) and the verdicts.
- **Start:** automatically after step 1 ends (about 16:25); about 5 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-26_psro_pilot_seed81/`.
- **Result** (16:31-21:08, all 4 iterations): **"no narrowing in the pilot"**. Valid gaps 0.390 (iteration 1) and
  0.425 (iteration 4); iterations 2 (0.129) and 3 (0.450) were not valid.
  - Gains per round, predator / prey: +0.19 / +0.20, +0.00 / +0.12, +0.34 / +0.11, +0.21 / +0.21. Round 3's sound
    predator learner alone beat the mix by 0.34, so that round was not converged either.
  - Game value of the population 0.47 → 0.48 → 0.40 → 0.44 → 0.42 (final). New versions entered every next mix
    (except the broken PP2). Final mixes: predators PP4 35, P4 35, PP1 30; prey PY4 39, Y2 37, Y1 24.
  - Two escape families: only P4 and XP2 (partly XP4) catch the Y2 family, and they catch little else. P4 and Y2 are
    in every mix. All four pilot predators gave Y2 up in training (catch 0.31 → 0.06, 0.23 → 0.04, 0.26 → 0.06,
    0.33 → 0.17) while learning the newest prey.
  - Learners: PP2 broke (1.20 drones lost per game, drone crashes against Y2 0.14 → 0.87; trained against PY1 54 /
    Y2 46). PY2 and PY3 crashed on their own (0.19, 0.15) at a similar rate against every predator but improved the
    prey's result and their own return; PY4 brought it back to 0.04 with the gains kept. The crash check rejects
    this risky style as well as real breakage.
  - League v2's newest versions were beaten by more (0.76-0.98 in rounds 2-4), but a mix is expected to be harder to
    beat than a single version. PSRO's mixes did not get harder to beat.
  - Next (proposals): can one predator learn both families (PP4, three blocks against Y2 50 / PY4 50, about 2 h);
    more PSRO iterations (about 7 h for six); the final mix against the older archive (about 30 min).
  - Evidence: `RESULTS.md`, `summary.txt`, `psro/`.

## 2026-10-27: One predator, two escape families: can one predator learn to catch both? (pre-registered)

- **User:** "you can continue as recommended. i am at work now for 9-10h. try to fill this time if possible". Step 1
  of a day chain (`artifacts/isaac/experiments/day_chain_2026-10-27.sh`); steps 2 and 3 are the next two entries.
- **Question:** in the PSRO pilot no predator caught both escape families, and every new predator gave the Y2 family
  up within its one block. Is that a limit of the learner, or of one block of training?
- **Two arms, three blocks each** (1,200 updates, LR 1e-4, KL 0.05, Adam state kept), against Y2 50% (the frozen
  prey) and PY4 50% (the per-env pool):
  - arm A from PP4 (prey lost: Y2 0.17, PY4 0.74); it has to learn Y2;
  - arm B from P4 (Y2 0.98, PY4 0.06); it has to learn PY4.
  - Order A1, B1, A2, B2, A3, B3; block seeds 81401-81403 and 81411-81413.
- **Gate after each block:** the new predator against the pilot's 15 prey versions, 512 games each (seed 42, mean
  actions). Videos of A3 and B3 against Y2 and PY4 at the end.
- **Bar:** prey lost ≥ 0.50 against Y2 and against PY4 at one gate, drones lost ≤ 0.37 against the pair. The best mix
  of the existing predators reaches 0.48 against this pair, the best single predator 0.18.
- **Decision:**
  - "one predator can catch both families" (an arm clears the bar);
  - otherwise "the families did not combine in three blocks", with each arm read as broke, trade-off, still rising
    or stalled.
- **Reported:** family means, the value against PSRO's final prey mix and the gain over its best response (0.427),
  training health per opponent, the pilot's table solved again with the six new predators.
- **Code:** the recipe backend unchanged (frozen copy), driver `run_two_families.py`. A dry run against a fake backend
  checked the starts, seeds, pool, gates, videos, resuming, the readings and decision.json.
- **Start:** 07:20; about 4 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-27_one_predator_two_families_seed81/`.
- **Result** (07:15-10:49, every status 0): **"one predator can catch both families"**.
  - **Arm A cleared the bar at gate 1 and held it.** Prey lost against Y2 / PY4: 0.904 / 0.943 (A1), 0.992 / 0.996
    (A2), 1.000 / 0.973 (A3). Drones lost against the pair: 0.18, 0.07, 0.15.
  - **Arm B "stalled":** PY4 0.064 → 0.156 → 0.180 → 0.117, with Y2 ≥ 0.996.
  - **Transfer:** the Y2 counter carried over to XY2 and XY3 (A3: 1.00, 0.99) but not to Y4. A3 wins 0.25 against Y4
    and loses 0.59 drones per game there: A1 kept away from Y4, A2 and A3 chase it and crash. A3 wins ≥ 0.82 against 12
    of the 15 prey; Y4, Y1 (0.34) and L3prey (0.36) escape it.
  - **Training:**
    - A1 caught Y2 0.41 → 0.82 (the pilot's PP4 block: 0.33 → 0.17).
    - Arm B's catch against PY4 stayed flat at about 0.11-0.16 after the first tenth of each block, with games of 446
      of 500 steps. The first-tenth values are inflated because short games finish first.
    - PY4 had been trained against a predator mix with P4 at 37%.
    - Arm B forgot Y4, which was not one of its opponents: 1.00 → 0.48.
  - **No learner broke:** six blocks on an exact 50/50 split, drones lost 0.07-0.18. This weakens the lead that
    near-even splits break learners.
  - **As a best response to PSRO's final prey mix,** A1 would gain +0.36 (A3: +0.40), more than any pilot learner
    (0.00-0.34).
  - **The pilot's table with the six new predators:** predators PP1 38, P4 32, A3 29, XP4 1; prey Y4 42, Y1 30, PY4 28;
    lo 0.480, hi 0.482 (pilot: 0.422, 0.427). This is a new cycle: P4 beats Y4, PP1 beats Y1, A3 beats PY4.
  - **Why A1 succeeded where the pilot's blocks failed is open.** The start, Y2's share, the other opponents and the
    seed all differ, and the closest pilot case (PP2) broke. The continuation's PP5 (from PP4, against PY4 39 / Y2 37 /
    Y1 24) is a partial test.
    - Addendum 12:15: PP5 also learned Y2 in one block (0.17 → 0.62), after learning Y1 first (0.30 → 0.94). Two of two
      blocks from PP4 learned Y2, and the pilot's four blocks from other starts did not. So the start is the more
      likely factor; the third prey delayed Y2 but did not prevent it.
  - **Next (proposals):**
    - A3 against the re-solved prey mix (Y4, Y1, PY4), about 2 h;
    - PSRO with A1-A3 in the population;
    - a readout of whether A3 tells Y2 from PY4 early in a game.
  - **Evidence:** `RESULTS.md`, `summary.txt`, `two_families/` (with the videos of A3 and B3).

## 2026-10-27: PSRO pilot's final mix against the older archive (pre-registered)

- **User:** as above; step 2 of the day chain, evaluations only.
- **Question:** the pilot's final mixes (predators PP4 35, P4 35, PP1 30; prey PY4 39, Y2 37, Y1 24) hold the
  population to lo 0.422 and hi 0.427. Does a version outside the population beat them?
- **Archive** (league v2's start archive): prey g0, g1, g2, g8, g14, g18, g20, c1-c5, L1prey, L2prey; predators g0-g13,
  g15-g20, c1-c5, L1pred-L3pred, E.
- **Cells:** each mix member against each archive version, 512 games. P4 against the archive prey and Y1, Y2 against
  the archive predators are reused from league v2's gates; 57 new evaluations.
- **Decision:** "the final mix holds against the archive" if lo_ext ≥ 0.422 − 0.05 and hi_ext ≤ 0.427 + 0.05;
  otherwise "the archive beats the final mix", with the versions.
- **Code:** driver `run_archive_check.py`; a dry run checked the archive labels, the 57 + 72 cells (the reused ones
  equal league v2's records), resuming and the decision.
- **Start:** after step 1 (about 11:45); about 25 min.
- **Evidence:** `artifacts/isaac/experiments/2026-10-27_psro_archive_check_seed81/`.
- **Result** (10:49-11:06, all 57 evaluations): **"the archive beats the final mix"**, on the prey side only.
  - **The predator mix against the archive prey:** c5 0.115, c1 0.186, c4 0.201, c2 0.234, c3 0.242, L1prey 0.245,
    L2prey 0.367 (bar ≥ 0.372; population lo 0.422).
    - No member handles c1-c5: P4 ≤ 0.08, PP1 ≤ 0.22, PP4 0.21-0.46.
    - P1 and P3, in the population, still beat the seven: 0.42-0.71 and 0.38-0.71 (league v2's gates). T (0.41) and
      the 10-24 league's predators (0.61) caught them too (catch only).
    - The lure prey g8 does not beat the mix (P4: 0.91).
  - **The prey mix holds:** the best archive predator, g17, reaches 0.399 (hi 0.427). The best archive predators are
    Y2 specialists (Y2 0.70-0.85, PY4 ≤ 0.11).
  - **Meaning:**
    - The pilot's value of 0.42 belongs to the restricted game. Against the full prey history, the predator mix's worst
      case is 0.12.
    - The newer predators lost the counter (mean over the seven: P1 0.60, P4 0.06), as arm B did with Y4. A table
      without the seven prey gives P1 and P3 no weight, so the mix hides that loss.
    - The PSRO continuation inherits this blind spot: its population excludes the archive, as planned.
  - **Next:**
    - the extended table, queued as step 4 (next entry);
    - PSRO from the extended population (a proposal).
  - **Evidence:** `RESULTS.md`, `summary.txt`, `archive/`.

## 2026-10-27: PSRO continuation, does the gap shrink in iterations 5-8? (pre-registered)

- **User:** as above; step 3 of the day chain. Option 2 of the pilot's proposals, with four iterations instead of six.
- **Question:** the pilot's valid gaps were 0.39 and 0.43. Four iterations are few for PSRO; does the gap fall in four
  more under the same rules?
- **Design:** the pilot's rules and driver unchanged, with iterations up to 8. It starts from the pilot's state
  (population, 210-cell table) and replays iterations 1-4 from their records; iteration 5's mixes are the pilot's
  final mixes, its starts PP4 and PY4. Block seeds 81951/81952, …, 81981/81982. The archive and the two-families
  predators stay out of the population.
- **Stop:** a valid gap ≤ 0.2, or iteration 8.
- **Decision** (on the valid gaps of iterations 1-8):
  - "PSRO converges for one-block learners" (a valid gap ≤ 0.2);
  - "PSRO narrows the gap" (last valid ≤ 0.39 − 0.2);
  - "no narrowing in eight iterations".
- **Code:** driver `run_psro_continuation.py`; a dry run checked the replay (no training of iterations 1-4, the
  pilot's records unchanged, solve 5 = the pilot's final solve), the starts, pools and seeds of iteration 5, an
  interruption and resume, the stop and the verdict.
- **Start:** after step 2 (about 12:10); about 66 min per iteration, at most about 4.6 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-27_psro_continuation_seed81/`.
- **Result** (11:06-15:27, every status 0, all 132 evaluations): **"no narrowing in eight iterations"**.
  - **Gaps:** valid 0.390 (1), 0.425 (4), 0.661 (5), 0.773 (6) and 0.749 (8). Iteration 7 was not valid: PY7 crashed
    on its own in 0.111 of games against its mix (limit 0.10); its gap was 0.603.
  - **The prey lineage got stronger every round.** PY8 is lost in 0.12-0.19 of games against each of the eight
    computed predator mixes, which held the hardest prey of their time to 0.39-0.60. Each prey from PY6 on beats every
    mix up to its round by at least 0.2.
  - **The predator lineage reached a generalist and lost it.**
    - PP7 beat all 18 prey in the table, at least 0.56 against each, with at most 0.32 drones lost per game.
    - That includes the prey that escape A3: Y4 0.96, Y1 0.79, L3prey 0.56 (A3: 0.25, 0.34, 0.36).
    - PY8 escaped it in one block (0.14).
    - PP8, trained against L3prey 65 / PY7 35, lost 10 of PP7's 18 counters (drops of at least 0.3).
  - **Forgetting showed up as pillar crashes both times.**
    - PP6 (trained against 66% PY5) and PP8 (two prey) chase faster and steeper than their starts.
    - They lose up to 0.93 and 1.39 drones per game against the prey left out (PP7: at most 0.32).
    - The validity check looks only at the training mix (0.16, 0.15), so it missed this.
  - **Game value:** 0.42 → 0.60 in round 8 (PP7 86% of the predator mix) → 0.49 final.
    - Final predator mix: PP8 45, PP5 27, PP7 21, XP2 7.
    - Final prey mix: PY8 48, XY2 29, PY4 22, XY3 1.
    - Every new version entered the next mix.
  - **Meaning:**
    - Here PSRO's gap mixes three things: weak mixes, stronger learners and forgetting. The table of each new version
      against every older mix separates them.
    - The predators' bottleneck is forgetting, not capacity.
    - PSRO trains each new version against the computed mix only, two to four opponents. That is the narrow kind of
      training that causes the forgetting.
  - **Next (proposals):**
    - PP7 against the whole prey history, with the archive seven: 2-3 blocks, gated on the worst case and on drones
      lost over all prey;
    - PSRO with a history share in each new version's training;
    - a safety check against the whole table for every new version.
  - **Evidence:** `RESULTS.md`, `summary.txt`, `psro/`. Videos (step 5 of the day chain, not planned) go to
    `videos_after_run/`.

## 2026-10-27: The table extended with the archive prey that beat PSRO's mix (pre-registered)

- **User:** as above. This is step 4 of the day chain, added at 11:20 after the archive check. Evaluations only.
- **Question:** seven old prey beat the pilot's final predator mix: c1-c5, L1prey and L2prey (the mix wins 0.12-0.37
  against them). Can the predators we have re-mix to cover them?
  - P1 and P3, in the population, beat the seven in 0.38-0.71 of games.
  - But they are weak against the Y2 family and the PY lineage.
- **Table:**
  - rows: the population's predators, the continuation's PP5 and later (only together with their PY pair, and only if
    the continuation's table holds all their cells), A1-A3, B1-B3 and L3pred;
  - columns: the population's prey, PY5 and later, and the seven.
  - About 170 new evaluations. The rest are reused from the continuation's table, the two-families gates, the
    archive check and league v2's gates.
- **Spot check:** P1 against c1 is evaluated again. If it differs from league v2's record by more than 0.005, no cell
  is taken from league v2's gates.
- **Decision:**
  - "the predators we have cover the archive prey" if the extended lo ≥ the restricted lo − 0.05. Both games are
    solved over the same rows, as in the pilot.
  - otherwise "the archive prey need a new counter".
- **Code:** driver `run_extended_table.py`. A dry run against a fake backend checked the versions (a dangling PSRO pair
  is left out), every reuse source, both spot-check paths, resuming and the decision.
- **Start:** after step 3. `day_chain_2026-10-27_step4.sh` waits for "day chain done". About 55 min.
- **Evidence:** `artifacts/isaac/experiments/2026-10-27_extended_table_seed81/`.
- **Result** (15:28-16:20, all 171 evaluations): **"the archive prey need a new counter"** (lo 0.521 → 0.374; bar
  ≥ 0.471).
  - **Reuse:** the spot check matched exactly (P1 against c1: 0.570 and 0.178 drones lost, as recorded), so league
    v2's cells were reused.
    - 650 cells (25 predators × 26 prey): 342 from the continuation, 90 from the two-families gates, 15 from the
      archive check, 32 from league v2's gates and 171 new.
  - **Restricted game:** predators PP8 48, A2 29, PP5 14, PP6 5, PP7 3, A3 1; lo 0.521. With A1-A3, B1-B3 and L3pred
    the value rises from the continuation's 0.49.
  - **Extended game:** predators P3 47, PP8 30, A3 19, P1 3, A2 1; prey c5 47, PY8 23, PY7 12, PY4 12, XY2 6; lo 0.374,
    hi 0.376.
  - **The predators split into two groups, each blind to the other's prey:**
    - The old ones beat the seven but not the newer prey. P1 wins 0.42-0.71 against the seven and averages 0.30 over
      the 19 population prey. P3 has 0.39-0.71 and 0.42, L3pred 0.75-0.82 against c2-c4 and 0.30.
    - The new ones are the reverse: PP7 0.83 over the population but 0.40 against the seven (c5 0.20), A3 0.72 and
      0.40.
    - No single predator wins more than 0.16 against all 26 prey.
  - **c5 is the hardest:** the restricted mix wins only 0.17 against it, and only P1 (0.65) and P3 (0.59) beat it
    clearly.
  - **PP7 does not crash against the seven** (0.14-0.29 drones lost per game); it does not catch them either.
  - **Meaning:**
    - The honest number for our predator side is 0.37, the most it can guarantee against every prey version we have.
    - The counters exist, but in different networks. The forgetting runs through the whole lineage, as in the
      continuation.
  - **Next (proposals):**
    - a generalist trained against all 26 prey from PP7, with a second arm from P1 to test the start, gated on the
      worst case over all prey;
    - or PSRO from the extended table with a history share.
  - **Evidence:** `RESULTS.md`, `summary.txt`, `extended/`.

## 2026-10-27: A generalist predator, trained against the whole prey history (pre-registered)

- **User:** "ok go on as recommended", the day report's recommendation.
- **Question:** no mix of our 25 predators covers all 26 prey (best 0.374).
  - The old predators beat the archive prey, and the new ones beat the newer prey.
  - Narrow training made the predators forget (PSRO continuation).
  - If a predator trains against all 26 prey at once, does it keep its counters and learn the missing ones?
- **Design:**
  - **Two arms** of three 1200-update blocks each (LR 1e-4, KL 0.05), with the Adam state kept:
    - A from PP7 (GA1-GA3, seeds 81621-81623);
    - B from P1 (GB1-GB3, seeds 81631-81633).
  - **Every block faces all 26 prey:**
    - half of the games are spread evenly (each prey gets at least 1/52);
    - half go to each prey in proportion to 1 − prey lost at the last gate (gate 0 is the extended table);
    - the largest share is the frozen "latest", and the other 25 are pool policies (`BLOCK_POOL_MAX`, commit
      7c13fea).
  - **A gate after each block** against the 26 prey (512 games, seed 42). Each arm's last gate also covers seven
    held-out archive prey: g0, g1, g2, g8, g14, g18 and g20.
- **Decision:**
  - "one predator covers the whole prey history" if a gate has a worst case ≥ 0.40 over the 26 prey, with drones
    lost ≤ 0.37 against each;
  - otherwise "no gate reaches the bar".
  - Flags per arm:
    - "kept": no prey more than 0.2 below the start at the last gate;
    - "grew": the worst case rose by at least 0.1.
  - Also reported: the extended table re-solved with the six new predators.
- **Code:** driver `run_generalist.py`, and `summarize.py`.
  - A dry run against a fake backend checked the versions, shares, pools, seeds, gates, resuming, the smoke path and
    the decision.
  - A 24-update GPU smoke block checked the 25-policy pool: memory and speed.
- **Start:** after the smoke test; about 4-5 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-27_generalist_predator_seed81/`.
- **Result** (16:58-22:59, six blocks and 170 evaluations; the held-out baseline 23:00-23:05): **"no gate reaches the
  bar"** (arm A: kept and grew; arm B: neither).
  - **Arm A (PP7 → GA1-GA3):**
    - worst case 0.141 → 0.332 → 0.311 → 0.393, always against PY8. GA3 misses the bar by 0.007, within the ±0.02
      noise of one cell, so by the rule it does not pass;
    - the seven 0.40 → 0.61 → 0.59 → 0.66; the 19 newer prey 0.83 → 0.90;
    - the largest drop of any prey from PP7 to GA3 is 0.03 (BY2), so nothing was forgotten;
    - drones lost at most 0.31 at GA3. GA2 briefly lost 0.48 per game against XY3, a prey it already beat and that
      got 3% of the training games: its pillar crashes there rose from 0.19 to 0.33 of games;
    - PY8 (0.39) and c5 (0.48) are the only prey still below 0.5.
  - **Arm B (P1 → GB1-GB3):**
    - the 19 newer prey 0.30 → 0.80, about one family per block (GB2 the Y and XY prey, GB3 the PY lineage);
    - the seven 0.60 → 0.37 (c5 0.65 → 0.24, c1 0.57 → 0.24), although each still got 2.8-4.9% of the games;
    - the worst case 0.074 → 0.162 (PY8 never moved).
  - **Seven held-out archive prey** (g0, g1, g2, g8, g14, g18, g20; never trained against). The starts' rows were added
    after the plan (`held_out_baseline.py`, 14 cells):
    - GA3 0.75-1.00 against PP7's 0.51-1.00: better against five, equal against two;
    - GB3 learned g8 (0.07 → 0.70) but lost g18 (0.70 → 0.44);
    - g2's numbers are mostly the prey's own crashes. Against g8, GA3 still loses 0.47 drones per game.
  - **Re-solved table (31 × 26):** lo 0.374 → 0.440, hi 0.443.
    - Predator mix: GA3 91, PP8 9 (PP8 beats PY8 0.92). Prey mix: c5 57, PY8 43.
    - The mix's drones lost are at most 0.33 per game.
  - **Changes from the plan:**
    - the training curves start at the second tenth, because the synchronized reset inflates the first;
    - the held-out baseline was added;
    - the blocks took 48-52 min, not 40-45.
  - **Meaning:**
    - The predators' forgetting came from what they trained against, not from a lack of capacity: GA3 holds the
      consolidation counter and the counters to the newer prey at once (the seven 0.66, the 19 newer prey 0.90).
    - The start matters: from P1, the same training swapped the archive prey for the newer ones.
    - GA3 is our best predator. The best single predator before won only 0.16 against its hardest prey.
  - **Next (proposals):**
    - a prey trained against GA3: one block each from PY8 and c5. Can a new prey escape it within one block, as PY8
      escaped PP7?
    - then whole-history training on both sides;
    - or one more block for arm A (GA4).
  - **Evidence:** `RESULTS.md`, `summary.txt`, `generalist/`, `held_out_baseline/`.

## 2026-10-28: Can a prey escape GA3 within one block? (pre-registered)

- **User:** "ok do as recommended", the generalist test's recommendation.
- **Question:** GA3 wins at least 0.393 against each of the 26 prey we have. But PY8 escaped the previous generalist,
  PP7, within one block (0.662 → 0.141), and three of league v2's four one-block prey exploiters beat their predator
  clearly. Can a prey also escape GA3 within one block, or is GA3 hard to beat?
- **Design:**
  - **Two prey blocks** of 1200 updates (LR 1e-4, KL 0.05) against GA3 alone (frozen, no pool), with the Adam state
    kept:
    - EY8 from PY8 (seed 81641);
    - EY5 from c5 (seed 81642).
    - They are the two prey GA3 does worst against, and the two prey of the best prey mix.
  - **A gate after each block:** the new prey against all 31 predators (the extended table's 25 and GA1-GB3), 512
    games, seed 42, GA3 first. The starts' rows are reused from the extended table and the generalist test.
- **Decision** (GA3's prey lost against each new prey):
  - "GA3 is escaped" if it is ≤ 0.20 against either;
  - "GA3 holds" if it is at most 0.10 below its result against the start for both (PY8 0.393, c5 0.477);
  - otherwise "GA3 is weakened, not escaped".
  - Also reported: each new prey against the 31 predators next to its start, the training curves, and the table
    re-solved with the two new prey (31 × 28; the predator side's guarantee is 0.440 without them).
- **Code:** driver `run_prey_vs_ga3.py`, and `summarize.py`.
  - The 31 predators are rebuilt with the extended table's own code.
  - A dry run against a fake backend checked the versions, starts, seeds, pool settings, gate order, resuming and all
    three outcomes.
  - A 24-update GPU smoke block from c5 passed: its optimizer state loads, the predator stays frozen, 23 steps/s.
- **Start:** about 07:10; about 1.1 h.
- **Evidence:** `artifacts/isaac/experiments/2026-10-28_prey_vs_ga3_seed81/`.
- **Result (2026-10-28, run 07:00-08:01; `RESULTS.md`):** **"GA3 is escaped".**
  - **GA3 against the new prey:**
    - PY8 0.393 → EY8 **0.166**;
    - c5 0.477 → EY5 **0.207**, just above 0.20 but far below the 0.377 that "holds" needed.
    - Both are evasion: GA3's catch rate fell from 0.355 to 0.117 and from 0.451 to 0.131. The prey's own crashes rose
      a little (0.031 → 0.045, 0.016 → 0.074) and count against them.
  - **EY8's escape is narrow.**
    - GA1-GA3 are each 0.11-0.23 lower against it.
    - Its average over the 31 predators is unchanged (0.222 → 0.217).
    - PP8 still wins 0.861 against it (0.14 drones lost per game).
  - **EY5 escaped every predator.**
    - The predators that had caught c5 lost most: P1 0.645 → 0.270, P3 0.588 → 0.287, GB1 0.559 → 0.283.
    - It got worse against those that never had (XP2 0.06 → 0.32), so its average is flat (0.247 → 0.239).
    - The best predator against it is XP2 at 0.324, losing 1.50 drones per game; within the drone limit, B3 at 0.307.
  - **Re-solved table (31 × 28):** the predator guarantee falls from 0.440 to 0.292, and the prey mix is 83% EY5.
    - The new predator mix leans on XP2 and XP4, which lose up to 2-2.75 drones per game against some prey.
    - With only the 10 predators within the drone limit (exploratory): 0.413 → 0.268.
  - **Training (noisy actions):** most of the escape came in the first ~200 updates. EY5's block swung between being
    caught and crashing. Both prey flew slower by the end (EY8 4.2 → 3.8 m/s, EY5 5.8 → 5.2 m/s).
  - **Changes from the plan:**
    - none in the design or the decision;
    - the plan's header time (07:05) was a planned time; the file was finished at 07:00:20, before the launch;
    - added afterwards: the drone-limited re-solve, the noisy-action reference, the speed numbers, and five videos
      (not analyzed).
  - **Meaning:** whole-history training fixes forgetting but not exploitability. A frozen predator is still beaten by
    the next one-block prey. Both sides need whole-history training, in turn.
  - **Next (proposal, not run):** two rounds of whole-history training on both sides, about 4.5 h; or GA4 alone, about
    1.2 h.
  - **Evidence:** `RESULTS.md`, `summary.txt`, `escape/`, `videos_after_run/`.
