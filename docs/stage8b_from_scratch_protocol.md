# Stage 8b: from-scratch training, the exact plan (2026-10-22, for approval)

This plan says exactly how the predator-prey pair for the 10-pillar game would be trained from random initialization:
- stages and settings;
- budgets and gates;
- seeds and compute;
- the scripts to write;
- the decisions left to the user.

It replaces the draft of 2026-10-18 (in the git history). Nothing has run.

## 1. Questions

Anchor v5 ([doc](anchor_v5_2026-10-21.md)) is the end of a fine-tune lineage:
- the Windows anchor;
- the archived recipe, which gave anchor v3;
- grafted obstacle branches, which gave anchor v4;
- grafted relation branches, which gave anchor v5.

Every step started from the previous network. Only a from-scratch run can answer three questions:

1. **Recipe.** Can one script, at one commit, train a pair from random initialization to anchor v5's level? No
   from-scratch recipe exists for the chase, let alone the pillar game. The dashboard recipe marks step 3 as the "gap".
2. **Equilibrium.** Is the pillar dance (time-outs 0.104, pincers rare) what this game and architecture converge to,
   or a product of the lineage's history? Three independent seeds answer that.
3. **Representation.** Do branches trained from the first update represent the game better than grafted ones? The
   blind-spot probes measure that.

Out of scope here:
- ablations of single fixes: body frame, zero-at-level upright term, per-drone ratios, removing the one-hot attention,
  correlated noise;
- coordination;
- CTBR.

They need a working recipe first (decision D1).

## 2. What the plan rests on

| fact | evidence | source (under `artifacts/isaac/`) |
|---|---|---|
| Flight must come first | Joint training of the full game from random init: predator OOB 1.0 after 2000 updates (v2) | `lineages/2026-07-17_19-47-44_scratch_joint_bootstrap_v2/` |
| ...and in the target arena | Flight learned at z = 1 is perfect on its own task, but gives predator OOB 1.0 in the elevated arena (v3) | `lineages/2026-07-17_20-43-59_scratch_flight_bootstrap_v3/` |
| Two-stage flight works | 750 updates with a fixed spawn, then 2000 with a random spawn: 8192 envs, 52.6 min. LR 1e-4, read from the checkpoints' optimizer state. Target task: predator OOB 0.0039, episodes 494.6 of 500 steps. A fresh GRU policy learned to fly on the bootstrap task in 300 updates. | `lineages/2026-07-17_21-28-34_scratch_two_stage_flight_v4/`, `evaluations/isaac_flight_bootstrap_gru_fresh300_seed42*.json` |
| The chase on top of it failed | 50,000 updates (15.4 GPU h), all with drifted settings: LR 5e-5, AMP, log-std floor −3, 500-1000-update phases, a 6 m/s speed margin. The predator, trained for 49,000 updates, caught the five reference prey in 0.098 of episodes (Windows anchor 0.854). The prey reached Windows level after 9,500 updates. | `diagnostics/2026-10-04_phase5_role_swap_and_windows_reference/` |
| The recipe behind anchors v2/v3 | FP32, LR 1e-5, KL 0.05, floor −2, long phases, opponent pools: 64,800 updates in 22 GPU h. It started from the Windows anchor, not from scratch. | [experiment log](experiment_log.md), 10-05 to 10-08 |
| The pillar game was never trained from scratch | Every obstacle network descends from anchor v3 through grafts | experiment log 10-08 onward |
| Parked pillars corrupt the input normalizer | See the explanation below the table | `RunningStandardScaler`; the presence mask, 72b877a |
| Throughput: obstacles10, 4096 envs, relation branches | Minutes per 1000 updates: predator-only 23-25; joint with ghosts in a third of the envs 30-32; prey-only 20.5. The consolidation's 18,000 updates took 9 h 11 min, including its evaluations. | `experiments/2026-10-18_*`, `2026-10-19_symmetric_graft_selfplay`, `2026-10-20_consolidation_selfplay` |
| GPU memory | 27.0 GB peak. 8192 envs ran out of memory. An evaluation next to a training run ran out of memory too. | `2026-10-21_drone_heads_test` |
| Training is bit-reproducible | Reruns at a fixed seed, env count and container reproduced every training scalar | `docs/3v1_hysteresis_linux_runbook.md` |
| `hysteresis_curriculum.py` cannot run this model | It passes no Hydra overrides, so relation flags, masks and the discount are lost; evaluating a relation checkpoint would fail. Since 10-08, all obstacle work has used drivers that call `train.py` directly. | code |
| No scripted evader exists for training | The scripted prey in `evaluate.py` is for evaluation only, holds z = 1 m and ignores pillars. Ghosts script predators, i.e. they serve prey training. | code |

**The normalizer trap.**
- Parked pillars sit at (50, 50) m. Their features, about 10-14 per component, enter the running mean and variance of
  every pillar column.
- After an open-space phase, real pillar features would be standardized to about −26 and clipped at −5.
- The large accumulated count lets those statistics adapt only slowly.
- So in the 10-slot model, pillars must be active from the first update, or the flight must use the obstacle-free model
  and be expanded later.

Anchor v5 reference values, own matchup, point 5:
- catch 0.663, success 0.736;
- time-outs 0.104;
- prey lost 0.231 per episode, predator drones lost 0.13 per episode;
- ablation ratio, prey / predator: 7.5 / 8.4 at point 4, 5.3 / 7.4 at point 5;
- long episodes: nearest predator 1.8 m, line of sight blocked 0.10.

## 3. Configuration (the same in every stage)

This is anchor v5's configuration, changed only where from-scratch training requires it.

| setting | value | why |
|---|---|---|
| task | `3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-obstacles10-v0`: 10 pillars over the 5 m disk, 10 s episodes, 50 Hz policy, 4096 envs | v5 |
| shapes | predator obs 210 = 3 × [own 12, prey 6, teammates 12, pillars 10 × 4]; prey obs 70; critic state 280; actions 12 (3 × 4) and 4 | v5 |
| model | preset `skrl_mappo_attention_critic_prey_attention_large_gru_obstacles10_cfg_entry_point`. Both masks, relation branches in both actors and both critics, the one-hot pillar attention kept, one shared decision head. | v5; the flags must match those of the opponents in the duels |
| frames, rewards | `observation_frame=mixed`, positive upright term, obstacle defaults (prey crash −300, predator crash −400, forced-crash bonus 200) | v5 (see D1) |
| PPO | rollouts 24, 3 epochs × 2 minibatches, discount 0.995, lambda 0.95, ratio clip 0.2, entropy 0.005, time-limit bootstrap on, one joint ratio per team | v5 |
| exploration | initial log std −0.5; floor −1.0 for the predators, −2.0 for the prey | v5 |
| obstacle LR multiplier | **1.0** (`agent.agent.obstacle_lr_multiplier=1.0`) | The ×30 group exists for zero-initialized grafts on a trained network. From scratch, it would give the pillar branches 3e-3 at a base LR of 1e-4. |
| precision | FP32, never AMP | July lesson |
| evaluation | deterministic, masked, seed 42; 1024 episodes (512 at the light game-stage points) | v5 |

Flags for every training block:

```
--task <task> --agent <preset> --algorithm MAPPO --num_envs 4096 --seed <block seed> [--checkpoint <previous block>]
--max_iterations <updates> --learning-rate <lr> --kl-threshold <kl>
agent.agent.obstacle_lr_multiplier=1.0 agent.agent.discount_factor=0.995
agent.agent.experiment.checkpoint_interval=<24 x updates>
env.prey_oob_penalty=-300.0 env.predator_oob_penalty=-400.0 env.predator_forced_crash_bonus=200.0
+agent.models.policy.predator_min_log_std=-1.0
+agent.models.policy.prey_mask_inactive_predators=true +agent.models.policy.predator_mask_inactive_teammates=true
+agent.models.policy.predator_obstacle_relation=true +agent.models.value.obstacle_relation=true
+agent.models.policy.prey_obstacle_relation=true
```

Stage-specific additions (Hydra overrides always go before `--freeze-agents`, which would swallow them):

| mode | additions |
|---|---|
| flight | `env.predator_distance_progress_reward_scale=0.0 env.predator_catch_bonus=0.0 env.predator_forced_crash_bonus=0.0 env.prey_evasion_reward_scale=0.0 env.prey_caught_penalty=0.0 env.prey_distance_progress_reward_scale=0.0 env.catch_distance=0.0` |
| joint | `env.ghost_env_fraction=0.3333 env.ghost_max_speed=3.5` |
| predator | `--freeze-agents prey --per-env-opponent-pool <pool.json> --per-env-pool-prob 0.5 --per-env-pool-seed <block seed>` |
| prey | `--freeze-agents predator --per-env-opponent-pool <pool.json> --per-env-pool-prob 0.5 --per-env-pool-seed <block seed>` |

Flight mode:
- It zeroes the same chase, evasion and catch terms as the flight-bootstrap task. Crash prices, the keep-out and soft-wall
  costs, the upright term and the prey's alive bonus stay on.
- `catch_distance=0` disables catches, so episodes end by time-out or loss.

## 4. Stages

| stage | updates | mode | LR / KL threshold | ghosts | opponents |
|---|---|---|---|---|---|
| F, flight | blocks of 1500, at most 6000, promoted at the gate | both roles train | 1e-4 (the July flight's LR) / 0 (the YAML default) | none | none |
| G0, LR pilot (seed 1 only) | 2 × 1200 per arm | joint | 1e-4, 3e-5, 1e-5 / 0.05 | a third, 3.5 m/s | none |
| G, game | 20 × 1200 = 24,000 (G0's 2400 included) | band guard | chosen in G0 / 0.05 | a third, joint blocks only | own past checkpoints, single-role blocks only |
| C, consolidation | 5 × (3000 joint + 600 prey) = 18,000 | anchor v5 recipe | 1e-5 / 0.05 | a third, joint phases only | own past predators, prey phases only |

### Stage 0: preparation (no long training)

**Code** (details in section 8): the host driver, the container block script, a helper module with unit tests, the
summary script, and a `--gamma` option for the probe.

No environment or model change is needed: every setting above uses an existing key.

**Smoke runs** at full size, about 45 min of GPU time. They must pass before any long run.

| | what | pass if |
|---|---|---|
| S1 | a flight block from random init, 30 updates | `params/env.yaml` and `params/agent.yaml` hold every setting of section 3. The checkpoint has one optimizer parameter group per role (the multiplier is 1). Losses are finite. The memory peak is below 30 GB. |
| S2 | S1 again with the same seed | every training scalar matches S1 bit-exactly (otherwise reproducibility becomes statistical, across 3 seeds) |
| S3 | from S1: a joint block with ghosts, a predator block with a two-entry own pool, a prey block with a pool, 30 updates each | the pool loads all policies; the frozen-role fingerprint check passes; each block loads its predecessor's checkpoint |
| S4 | the light and full evaluation points on S3's checkpoint | every JSON exists (compositions with anchors v5 and v3, the ablations, ghosts, the flight-settings evaluation); `summarize.py` writes `decision.json` |
| S5 | a dry run of the driver with a stub instead of `docker` | the band guard, the pool files, the gate checks and resume-after-interrupt follow the plan |

### Stage F: flight in the pillar arena

- **Start:** random init (no `--checkpoint`); both roles train; all 10 pillars are active from the first update.
- **Why pillars from the start:**
  - the normalizer trap (section 2);
  - no pillar-blind flight habits;
  - DualCL's result was obtained by training in its pillar scenario itself (Stage 8 review, claim 8).
- **Budget:**
  - Blocks of 1500 updates. After each block, the flight gate runs.
  - The stage is promoted at the first pass, after at most 4 blocks (6000 updates).
  - The July two-stage flight needed 2750 updates at 8192 envs.
- **Flight gate** (flight settings, 1024 episodes):
  - drones lost per episode (sum over the three predators) ≤ 0.10;
  - prey lost (arena exit or pillar) ≤ 0.05;
  - mean episode length ≥ 490 steps.
- **Also evaluated, not gated:**
  - the same pair on the game task (catches on), which becomes point 0 of stage G;
  - pillar crash rates and minimum heights.
- **Failure rule:** no pass after 6000 updates means stop and report. The pre-registered fallback F' needs the user's
  approval. It consists of:
  1. the verified July two-stage flight with the obstacle-free model;
  2. `scripts/expand_checkpoint_obstacles.py --from-obstacles 0 --num-obstacles 10` (zero-initialized branches; the new
     scaler columns get mean 0 and variance 1);
  3. then stage G.

### Stage G0: learning-rate pilot (seed 1 only)

**The LR question.**
- The only from-scratch LR on record is 1e-4, in flight. Every fine-tune ran at 1e-5. The July chase at 5e-5 is
  confounded with AMP, the floor −3 and short phases.
- The policy change per update grows roughly with the square of the step size. So 1e-4 moves the policy about 100× as
  far as 1e-5 per update, in KL.
- That speed is needed from scratch, but it can destabilize a game.

**Pilot design.**
- **Arms:** three from F's final pair: LR 1e-4, 3e-5 and 1e-5.
  - KL threshold 0.05, joint blocks with ghosts, 2 × 1200 updates.
  - Identical block seeds, so the arms are paired.
- **Measured at 2400 updates** (1024 episodes):
  - predator interception: success against the frozen F prey, which hovers and does not evade;
  - prey survival against 3.5 m/s ghosts: 1 − caught − crashed (amendment 1);
  - the KL early-stop epoch fraction (TensorBoard);
  - non-finite losses;
  - drones lost in the own matchup.
- **Stable arm:**
  - finite losses;
  - a mean KL early-stop epoch fraction in block 2 ≤ 0.5;
  - drones lost ≤ 1.5× the 1e-5 arm's + 0.05.
- **Rule:**
  - score = interception + ghost survival;
  - choose the highest stable LR whose score is at least the next lower LR's − 0.03.
- **Continuation:** the chosen arm becomes seed 1's stage G (its 2400 updates are blocks 1-2). Seeds 2 and 3 use the
  chosen LR without a pilot.
- **No stable arm:** stop and report.

### Stage G: the game

- **Start:** F's final pair. For seed 1 this is the chosen pilot arm.
- **Blocks:** 20 blocks of 1200 updates. Blocks 1 and 2 are joint; from block 3 the band guard decides each block's
  mode.
- **Band guard:** after each block, the own-matchup catch rate (512 episodes) sets the next block:

  | catch | next block |
  |---|---|
  | below 0.25 | predator block: prey frozen; half of the envs face own past prey |
  | above 0.75 | prey block: predator frozen; half of the envs face own past predators; no ghosts |
  | otherwise | joint block, a third of the envs with ghosts |

  - The pools hold the frozen role from the last 4 points of stage G, with equal weights. The current checkpoint is
    excluded automatically.
  - Per-env pools work only with exactly one frozen role, so joint blocks have none.
  - Why a guard: in July the prey reached Windows level after 9,500 updates, while the predators were still weak after
    49,000. A role that loses almost every episode gets almost no signal from the outcome. Joint blocks are the default
    because they are the anchor v5 recipe. Single-role blocks only correct an imbalance.
- **Points:** a light point after every block; full points after blocks 10 and 20.
- **Interim rule at block 10** (12,000 updates), a pre-registered stop. It passes if:
  - the predator intercepts the F prey in ≥ 0.5 of episodes;
  - the prey's loss hazard against ghosts (caught or crashed) is ≤ 0.7× its value at the start of stage G
    (amendment 1).

  Otherwise, stop and diagnose before changing anything, in the order of AGENTS.md: observations, action scaling,
  rewards, terminations, log-probabilities, values, advantages.

### Stage C: consolidation (the anchor v5 recipe)

- **Start and phases:** from G's final pair. Five cycles of a joint phase (3000 updates, a third of the envs with
  ghosts) and a prey rehearsal phase (600 updates).
- **Settings:** 18,000 updates; LR 1e-5, KL 0.05; checkpoints chained (they keep the optimizer states).
- **Prey rehearsal:**
  - half of the envs face own past predators, with equal weights: from G blocks 10 and 20, and from every completed C
    cycle;
  - the other half face the current, frozen predator.
  - The anchors stay out of training (D3).
- **Points:** after each cycle. Points 1-3 are light; points 0 (the end of G), 4 and 5 are full.

### Final gate (per seed, points 4 and 5 of stage C)

1. **Perception:** the own-matchup ablation ratio, ≥ 1.1 for both roles (mean of points 4 and 5). The ratio is the crash
   hazard with all obstacle outputs zeroed divided by the intact hazard.
2. **Balance:** own catch within 0.3-0.7 at points 4 and 5.
3. **Duel with anchor v5** (mean of points 4 and 5):
   - D = R(our predators vs v5's prey) − R(v5's predators vs our prey);
   - R = 200 × success − 400/3 × drones lost per episode.
   - D ≥ −10 passes; D > +10 means stronger than v5. The bar is about 3 standard errors of the two-point mean.
   - Neither side ever trained against the other, so the duel favors neither side's familiarity. A comparison against
     v5's own matchup would favor v5, whose predator was trained against its own prey.
4. **Stability:** own catch changes by ≤ 0.05 from point 4 to point 5.
5. **No waiting game,** judged by its mechanism in the long episodes at point 5:
   - nearest predator, median ≤ 3.0 m;
   - line to the prey blocked ≤ 0.30.
6. **Safety** (point 5): prey lost ≤ 0.33 per episode; predator drones lost ≤ 0.37 per episode (anchor v4's level).

**Verdicts.**
- Per seed: pass (all six criteria), pass and stronger (D > +10), or fail with the failing criteria named.
- Recipe: reproducible if at least 2 of the 3 seeds pass.
- Anchor v6 candidate: the passing seed with the highest league rating. It is frozen only after the user's review.

**Reported without a gate:**
- the duel with anchor v3 (obstacle-blind);
- prey against ghosts;
- time-outs, pincers, prey speed and catches at the wall;
- the probes, compared with v5:
  - line of sight at the GRU output (v5 0.64);
  - closing speed (v5 0.39; a raw-input MLP reaches 0.75);
  - critic explained variance in long episodes, at discount 0.995;
  - the prey's weight in the entity attention (v5 0.67).

## 5. Seeds and order

- **Training seeds:** 81, 82 and 83.
  - A block's seed is seed × 1000 + its block number: F 1-4, G 101-120, C joint 201-205, C prey 301-305.
  - The per-env pool seed equals the block seed. Evaluations use seed 42.
- **Seed 1 (81)** runs F, G0, G and C as one chain, about 30 h.
- **Seeds 2 and 3 (82, 83)** run after seed 1's review, chained, about 2 × 27 h.
- **If the review changes the recipe** (for example the G budget, because seed 1's duel curve was still rising):
  - write it as an amendment before seeds 2 and 3 start;
  - seed 1 then no longer counts toward the recipe verdict, so a third seed of the amended recipe is needed.

## 6. Evaluation

**Light point** (about 7 min, 512 episodes):
- the own matchup (it feeds the band guard);
- the predator against the F prey;
- the prey against ghosts;
- the duel with v5 in both directions.

**Full point** (about 15 min, 1024 episodes):
- the own matchup with trajectories;
- the duels with v5 and v3;
- the own-matchup ablations of both roles;
- the prey against ghosts;
- the predator against the F prey.

**At the end of each seed:**
- the strategy analyses on the own trajectories: anatomy, long game, shield geometry, forced distance, roles, flanking
  value;
- the probe at discount 0.995;
- videos from the elevated camera (eye 10 −10 12.5, target 0 0 5). Seed 1 also gets clips at the end of F, at G 12k,
  at G 24k and at the end of C, showing the path from random init to the game.

**After all seeds:**
- a 6 × 6 league: predators and prey from the three seeds and anchors v5, v4 and v3; 36 cells, about 1 h;
- a Bradley-Terry fit (`diagnostics/league_2026-10-08/fit_bradley_terry.py`).

`summarize.py` refreshes `decision.json` after every point, so the dashboard shows each gate as it develops.

## 7. Compute and schedule

Measured throughput at 4096 envs: joint about 31 min per 1000 updates, predator-only 24, prey-only 20.5.

| part | updates | training | evaluation | total |
|---|---|---|---|---|
| F | 1500-6000 | 0.8-3.1 h | 0.1-0.3 h | 1-3.5 h |
| G0, the two extra arms (seed 1 only) | 2 × 2400 | 2.5 h | 0.5 h | 3 h |
| G | 24,000 | 10-12.5 h (depends on the mode mix) | 2.6 h | 13-15 h |
| C | 18,000 | 8.6 h | 0.9 h | 9.5 h |
| analyses, probe, videos | | | 0.7 h | 0.7 h |
| **one seed** | **about 45,000** | | | **24-29 h (seed 1 with the pilot: 27-32 h)** |
| three seeds plus the final comparison | | | | **about 80-90 h of GPU time** |

Calendar if approved today:
1. Stage 0: about 3 h of implementation plus about 45 min of smoke runs.
2. Seed 1: about 30 h, started the same evening.
3. Review.
4. Seeds 2 and 3: about 54 h.
5. Final comparison: about 3 h.

All of it fits in 4-5 days. No other GPU job runs during training. Each long run gets a completion waiter, so results
are written up as soon as they land.

## 8. Scripts and reproducibility

**In the repo:**
- `scripts/recipes/stage8b/block.sh` runs inside the container. It runs one block with the fixed flags of section 3.
  - modes: flight, joint, predator, prey;
  - arguments: output directory, tag, mode, checkpoint or none, updates, seed, LR, KL, pool file;
  - it writes a status file.
- `scripts/recipes/stage8b/run_seed.py` is the host driver, standard library only. It is in Python rather than bash
  because the pipeline branches (gates, band guard, pools) and must resume after an interruption:
  - It runs one seed through F, G0 (with `--pilot`), G, C and the final evaluation.
  - It records each finished block, so after a power-off or crash it continues from the last finished block. Training
    is deterministic, so a rerun block reproduces itself.
  - It checks each run's `params/*.yaml` against the plan and stops on a mismatch, on a status other than 0, on a
    missing checkpoint or on a missing JSON. Isaac scripts exit with 0 even after an exception.
  - It refuses to start if `source/`, `scripts/skrl/` or `scripts/recipes/` have uncommitted changes. It runs a copy of
    itself and of `block.sh` placed in the experiment directory, so a later edit cannot change a running seed.
  - It ends with a `done` line, which the waiter watches.
- `scripts/recipes/stage8b/recipe.py` holds the decisions as pure functions: block seeds, the band guard, pool files,
  the F, G0 and interim gates, and the final gate.
- `tests/test_stage8b_recipe.py` tests those functions on synthetic evaluation JSONs.

**In `artifacts/isaac/experiments/<date>_stage8b_from_scratch/`:**
- `PLAN.md`: the pre-registration, with its timestamp and the commit;
- per-seed directories;
- `summarize.py` and `decision.json`;
- copies of the analysis scripts.

**In the diagnostics:** `probe_analyze.py` gets `--gamma`. Today it fixes 0.99, which undervalues a 0.995 critic.

**Reproduction:** check out the plan's commit, `docker start isaac-lab-base-uav51`, then on the host:

```
python3 scripts/recipes/stage8b/run_seed.py --seed 81 --pilot --out <experiment>/seed81
python3 scripts/recipes/stage8b/run_seed.py --seed 82 --lr <G0's choice> --out <experiment>/seed82
```

The container image `uavpredatorprey-isaaclab:f4aa17f-sim5.1.0-skrl1.4.3` is pinned. Given the seed, the env count and
the image, each run is reproducible exactly (smoke check S2).

## 9. Risks and fallbacks

| risk | sign | response |
|---|---|---|
| Flight does not learn among pillars | flight gate fails after 6000 updates | stop; fallback F' (verified flight, then expansion), with approval |
| The predators lag behind, as in July | catch below 0.25 again and again; the interim rule | the band guard gives predator blocks; the interim rule at 12,000 updates stops a real failure |
| LR 1e-4 destabilizes the game | non-finite losses, KL early stops, drone losses | the G0 pilot; the KL threshold 0.05 |
| Too small a budget | the duel curve still rising at the end | report; an amendment before seeds 2 and 3 (section 5) |
| The cautious equilibrium or the waiting game returns | criteria 5 and 6; prey speed and catches at the wall | a result, not a bug. It would weaken the claim that missing representations caused the waiting game. |
| The normalizer statistics lag across stages | stage F's samples dominate early in stage G | they are diluted to about 10-20% of all samples by the end of G; no action |
| A frozen role's optimizer state is not saved | its Adam moments restart after each single-role block | known; minor at these step sizes |
| Interruption | a power-off (one lost run on 10-04) | resume from the last finished block |
| Memory | evaluations never run next to training | — |
| Not bit-exact | smoke check S2 fails | reproducibility becomes statistical, across the three seeds |

## 10. Decisions for the user

**D1. Configuration.**
- **Recommended: anchor v5's configuration unchanged.**
  - The curriculum is the unknown: flight among pillars, the chase from flight and the pillar game have never been
    trained from scratch.
  - Each of the alternatives below would stack a second untested change on the run and blur a failure.
  - The duel and the probes stay directly comparable with v5.
- **Alternative: the body frame (A3) now.**
  - It is the one measured defect that only a from-scratch run can fix. The drones yaw about 160 deg/s, and the
    closing-speed probe reaches 0.39 from the features against 0.75 from the raw input.
  - It probably matters most for Stage 9, where the predators must track a hidden prey from their own motion.
  - Recommendation: test it right after the recipe works, as a 2-seed ablation against the three seeds' spread.
- **Also possible: the zero-at-level upright term (B7).** It removes a stream worth about 13 return points at discount
  0.995, but it is also the predators' only positive term in the flight stage.

**D2. LR pilot.** Recommended: yes, on seed 1 only (+3 h). Choosing an LR without data risks a day of training too slow
or unstable.

**D3. Training opponents.** Recommended: own history and ghosts only, with the anchors used only for evaluation. A
recipe that needs the anchors would be neither independent nor reproducible without them, and training against v5
would bias the duel.

**D4. Order.** Recommended: seed 1, then a review, then seeds 2 and 3. The alternative is all three chained, about 3.5
unattended days.

**D5. CTBR.** Not bundled, unchanged: one new difficulty per stage.

## Amendments

Both were found in the smoke run on 2026-10-22, before any real training.

1. **Losses against ghosts count crashes.** Ghosts never crash, so against them the prey ends an episode in one of
   three ways: caught, crashed (pillar or arena) or timed out.
   - The draft scored only catches, so a prey that crashed in every episode looked like a perfect evader. The smoke
     run's untrained pilot prey did exactly that.
   - The pilot's ghost survival is therefore 1 − caught − crashed. The interim rule uses the prey's loss hazard
     against ghosts, counting both.
2. **Loading uses the training multiplier.** Evaluations, the probe and videos also pass
   `agent.agent.obstacle_lr_multiplier=1.0`.
   - Otherwise the preset's ×30 rebuilds an optimizer with two parameter groups, and loading the recipe's
     single-group optimizer state fails.
   - Composed checkpoints carry no optimizer state, so the flag changes nothing there.
