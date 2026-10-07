# MARL Roadmap for UAV Predator-Prey

This is the long-term plan for the project and changes slowly. The dated status
snapshot below records where the project stands. Results and decisions of
individual experiments are in [`experiment_log.md`](experiment_log.md), and the
commands for training and evaluation are in the
[runbook](3v1_hysteresis_linux_runbook.md).

## North Star: Visual Occluded Pursuit

The long-term goal is a predator team that catches an agile prey using only
on-board perception, even when the prey is temporarily hidden behind
obstacles. The end problem is therefore multi-agent visual pursuit under
partial observability:

- predators receive camera-based or local sensor observations instead of
  perfect prey coordinates,
- obstacles can occlude the prey,
- predators must maintain a belief over where the hidden prey is likely to be,
- predators must predict likely prey motion and intercept paths,
- predators must coordinate without relying on perfect global state, and can
  share what they see (communication or a shared belief),
- all agents must satisfy flight stability and collision-safety constraints.

Decision (2026-10-07): the policies should use an interface that real flight
controllers accept, so that deployment on real quadrotors stays possible. The
action interface moves to collective thrust plus body rates; see
[`ctbr_action_interface_plan.md`](ctbr_action_interface_plan.md).

The progression adds one new difficulty per step:

1. full-state 3v1 pursuit in an empty arena (done: anchor v2),
2. obstacle-aware full-state pursuit,
3. line-of-sight masked state observations,
4. belief over the hidden prey: memory, an explicit estimator, and belief
   sharing between predators,
5. camera or depth observations, reached by distilling the full-state policy
   into a perception policy (teacher-student),
6. optional: predictive latent world models (JEPA-style) for representation
   learning.

The closest current external reference is Chen et al., "Multi-UAV
Pursuit-Evasion with Online Planning in Unknown Environments by Deep
Reinforcement Learning" (OPEN). It is the main domain comparator for
obstacle-aware, partially observable, 3D UAV pursuit-evasion:

- MAPPO trained in a GPU-parallel UAV simulator,
- collective thrust and body-rate commands as the policy output,
- attention-based observation encoder,
- LSTM evader-prediction module for partial observability,
- adaptive environment generator for curriculum and generalization,
- two-stage reward refinement for smoother deployment behavior,
- zero-shot transfer to real quadrotors after dynamics calibration.

Project implication: our self-play, opponent-pool, and cross-play stack is a
different strength than OPEN, but the missing OPEN-style pieces are clear:
evader prediction, adaptive scenario generation, obstacle/occlusion
generalization, and deployment-oriented reward/control refinement.

JEPA-style world models are not an immediate step. They are a candidate
research module once partial observability and visual perception become the
bottleneck. The likely role is representation learning or auxiliary prediction
in latent space, not replacing MAPPO/self-play as the control training backbone.

## Design Rules

Each rule was learned from a concrete failure or near miss in this project.

- **One new difficulty per stage.** A stage counts as done only when its gate
  passes, so that failures stay attributable.
- **Evaluation before architecture.** Every claim is measured against frozen
  references and held-out opponents with deterministic, paired evaluation
  (shared initial states, 1024 episodes per matchup).
- **Metrics before mechanisms.** Define the behavior metric before the
  intervention that is supposed to produce it. On 2026-10-07 an intuitive
  encirclement metric turned out to be misleading in 3D (see Metrics).
- **Pre-registered A/Bs.** Write the decision rule before training. Margins
  must exceed the training-seed noise, not only the evaluation noise. Measured
  2026-10-07 (predator, 2400 updates): own matchup SD 0.027 per run, so use a
  margin of at least 0.08 or 3 seeds per arm, and compare against multi-seed
  control means. Reference mean 0.01, held-out mean 0.045, wall enclosure
  0.045.
- **No strategy-encoding rewards.** Coordination such as cornering or
  encirclement must emerge from the task, physics, opponents, and curriculum.
  Reward terms that pay for a specific strategy are out of scope.
- **Length-independent policies.** No time-to-go observation. Time-outs are
  truncations and are bootstrapped (preset default since 2026-10-07).
- **The prey is more agile on purpose** (thrust-to-weight 2.2 vs 1.9 for the
  predators), so that plain pursuit is not enough.
- **Reproducibility.** Archived results must stay bit-exactly reproducible.
  The runbook smoke checks this after every relevant change.
- **Do not add all SOTA components at once.** A new component must address a
  measured bottleneck.

## Status Snapshot (2026-10-08)

| Item | Status |
|---|---|
| Evaluation anchors (Stage 0) | done: reference set, held-out opponents, 9x9 league with Bradley-Terry ratings, deterministic paired evaluation that stays bit-identical under GPU load |
| Attention actors, entity-attention critic, prey attention, GRU memory (Stages 1-4) | implemented and in use; ablations against the simpler variants are still open |
| Spawn robustness (Stage 5) | random spawn in the spherical arena; no initial-velocity randomization yet |
| League hardening (Stage 6) | pools, cross-play, PFSP weights, per-env mixing, rehearsal; forgetting and non-transitivity are measured but unsolved; no exploiters |
| Locked obstacle-free baseline (Stage 7) | anchor v3 ([doc](anchor_v3_2026-10-08.md)), accepted 2026-10-08 by a pre-registered rule; training-seed replicates missing |
| Obstacles, partial observability, perception (Stages 8-10) | not started |
| Correctness gate | all items closed except three open ones (see Correctness Gate) |

Key findings:

- **Anchor v3 is the strongest pair** in a 10x10 league of historical
  checkpoints (predator +1.13, prey +1.09 logits relative to the Windows
  anchor prey; anchor v2 +0.81 / +0.50). Its predator catches the five
  reference prey in 0.869 of episodes on average and the held-out prey in
  0.812, and its prey concedes 0.164 to the reference predators (1024 episodes
  per matchup).
- **The game is strongly non-transitive.** Many matchups deviate from a
  single-rating model by more than 10 standard errors, so a single opponent is
  a misleading benchmark.
- **Time-outs were treated as terminal.** This specification error is fixed in
  837e0aa. The bootstrapped prey held up better beyond the 10 s training
  horizon (30 s catch rate 0.739 vs 0.789), and the critic no longer collapses
  at time-outs.
- **The value normalizer is frozen at an old return scale** (about 300x too
  wide for the prey), but the critics still explain 72-87% of the return
  variance. It is documented and must stay fixed in comparisons.
- **Coordination is rare and has not responded to three levers.** In 3D,
  three predators can only corner the prey against the arena wall. Enclosure
  on the wall occurs in about 15-20% of the last second before catches (11-13%
  in escapes). Removing the pursuit shaping, widening predator exploration,
  and training against a more agile prey (T/W 2.5) changed performance and
  generalization, but not this geometry. Against the agile prey the predators
  learned better individual interception (steering toward the prey's heading)
  instead.
- **Self-play forgets unless rehearsal is strong enough.** With a 50% pool
  and weight floor 0.1, all three seeds lost against the old reference prey
  (0.878 -> 0.803). A 75% pool prevents it but costs about 0.16 against the
  current opponent. The 60% pool with floor 0.3 used for anchor v3 kept the
  reference prey (0.869) at no measurable cost.

Current bottlenecks, in order:

1. Emergent coordination (cornering against boundaries) is rare.
2. Forgetting and non-transitivity in self-play.
3. Small effects need several seeds: one predator run varies by SD 0.027 on
   its own matchup.

No longer bottlenecks: basic flight and pursuit capability, evaluation
correctness, and reproducibility.

## Target Architecture

### Actor

- Shared per-predator parameters. Today they live inside one team agent: one
  skrl agent controls all three predators with shared per-drone weights.
  Decentralized execution under partial observability needs a per-UAV
  interface in which each drone acts on its own local observation.
- Entity or graph attention over visible objects: own UAV state, prey state,
  teammate predator states, and later obstacles, arena features, and
  line-of-sight masks.
- Prey actor with entity attention over predator entities (own prey state as
  the controlled agent; later obstacles and line-of-sight masks as additional
  entities).
- Recurrent memory where partial observability is real; GRU first. A GRU also
  lets a policy infer elapsed time from its hidden state, which is why
  time-out handling mattered even without a time observation.
- Tanh-squashed Gaussian continuous-action head.
- Action interface: currently normalized collective thrust plus body moments,
  so the policy stabilizes attitude itself. The candidate alternative is
  collective thrust plus body rates with a low-level rate controller (see
  Decision Point Before Stage 8).

### Critic

- Centralized critic for MAPPO-style training.
- Global entity encoder over all agents and environment objects.
- Attention or graph network over predators, prey, obstacles, and
  arena/boundary features.
- Optional temporal memory when observations become history-dependent.
- Scalar value head. Read critic quality from explained variance and the raw
  return scale, not from the normalized value loss.

### Training

- MAPPO with the recurrent wrapper as the stable baseline.
- For resumed-run learning-rate ablations, apply the override after checkpoint
  loading so the new optimizer LR is used while Adam's learned moments are
  preserved. Keep the baseline YAML unchanged until the ablation wins.
- Alternating self-play with freeze phases, driven by the hysteresis scheduler
  (band guard, `toward_center` instead of joint phases).
- League / opponent-pool training with per-env mixing during freeze phases,
  PFSP-style weights, and rehearsal shares large enough to prevent forgetting.
- Curriculum: no obstacles, 1v1, 2v1, 3v1 (done), then simple obstacles, then
  complex obstacles and occlusion.
- Reward shaping lifecycle:
  - use dense helper rewards early when sparse task rewards are too hard,
  - explicitly ablate or anneal helper rewards once the behavior is learnable,
  - treat proximity, cover-seeking, boundary-progress, and similar terms as
    scaffolding rather than permanent objectives unless diagnostics prove they
    still improve final behavior,
  - watch for shaping rewards creating local optima such as hovering near the
    prey, clustering, boundary surfing, or avoiding risk instead of solving the
    task,
  - never add terms that pay for a specific coordination strategy.
- Domain randomization: spawn positions, initial velocities, UAV dynamics,
  action noise, observation noise, sensor dropout / line-of-sight masking.

## Stages

### Stages 0-4: Foundations (done)

Evaluation anchors, shared predator attention (2v1, then 3v1), the
entity-attention centralized critic, prey-side entity attention, and the GRU
actor/critic with recurrent MAPPO are implemented and in use.

Open ablations, needed for rigorous claims rather than for progress:

- entity-attention critic vs flat centralized critic,
- prey attention vs flat prey actor,
- GRU vs feedforward attention policy under full observability.

Controlled ablation rule: when comparing these variants, keep matched the
learning rate and scheduler, entropy scale, initial log standard deviation,
log standard deviation min/max, rollout length and number of environments,
reward scales, and termination rules.

### Stage 5: Spawn and Initial-State Robustness (partial)

Goal: prevent opening-book strategies that only work from one spawn geometry.
Random spawn in the elevated spherical arena (with minimum separations) is the
active training task. Still missing:

1. small initial horizontal velocity noise,
2. larger velocity and yaw/angular-velocity noise once position
   randomization is stable.

Constraints: minimum predator-prey distance (no trivial catches), minimum
predator-predator distance, margin from the soft arena boundary, height near
the target height, and an optional cap on initial speed early on.

Watch: catch rate must not be dominated by instant spawn catches, OOB must stay
low, episode length must not collapse, and policies must still work on the
original ring spawn.

### Stage 6: League Hardening (partial)

Current mechanism: single-role phases mix the latest frozen opponent and pool
opponents across environments. Auto-pool promotion writes new candidates into
the run-local pool, cross-play updates pool weights, and PFSP-style sampling
emphasizes difficult opponents.

Lessons from 2026-10-05 to 2026-10-07:

- Beaten opponents fall to the PFSP weight floor, or are truncated by the
  top-k limit, and are then forgotten. Rehearsal needs a meaningful share
  (a 75% pool restored the reference mean from 0.596 to 0.828).
- Pool weights refreshed only after single-role phases go stale during joint
  phases. `toward_center` avoids joint phases altogether.
- The measured non-transitivity means a scalar rating is only an average.
  Pool composition should cover the cycles, not only the hardest opponents.
- Pools may only contain GRU-compatible checkpoints. Old feedforward
  checkpoints remain useful for diagnostic migration tests, not as league
  opponents.

Next upgrades:

- minimum-quality gates before promotion (role-specific catch threshold,
  maximum own OOB / soft-arena usage, no exploit via opponent OOB),
- stale-entry pruning based on cross-play usefulness,
- a game-theoretic pool: meta-strategy weights from the empirical payoff
  matrix (PSRO-style), with Nash averaging as the rating,
- conceptual roles: main agents, exploiters that only target the main agent's
  weaknesses, and historical reference policies.

### Stage 7: Locked Obstacle-Free Baseline (done for anchor v2)

Required evidence, with its status for anchor v2:

- one named checkpoint pair, exact training command and pool file: done
  (read-only bundle with manifest, hashes, and provenance),
- latest-vs-latest evaluation and a predator-vs-prey cross-play matrix: done
  (9x9 league, 1024 episodes per matchup),
- representative videos and behavior notes (pursuit, cornering, evasion,
  failure modes): done (escape and wall analyses),
- training-seed replicates of the final training step: missing.

Gate: clean catch rate high enough to show predator competence, predator and
prey OOB below 5-10%, low forced prey OOB, no catastrophic hole in cross-play,
and videos showing pursuit and evasion rather than only instant catches.

### Decision Point Before Stage 8

1. **Training-seed noise floor (done 2026-10-07).** The own-matchup SD is
   0.027 per run; the reference mean is very stable (SD 0.003). Forgetting of
   the old reference prey under a 50% pool with weight floor 0.1 is
   systematic.
2. **Action interface (decided 2026-10-07: CTBR).** Body moments make the
   policy do attitude control itself: the predators are tilted beyond 45 deg
   about 31% of the time. Body rates with a rate controller are the standard
   for agile quadrotor RL and sim-to-real (OPEN, Swift), and they are what an
   MPC layer would build on. The target is a well-motorized FPV quad. Every
   realism change invalidates all checkpoints, so CTBR, FPV-class dynamics,
   latency, and an optional motor model come together as one deployment
   migration, after the behavior research in the current simulation
   ([`ctbr_action_interface_plan.md`](ctbr_action_interface_plan.md)).
3. **Coordination (answered 2026-10-07).** The prey agility curriculum did
   not create cornering in the empty arena, so look for coordination where
   obstacles create traps. Agility training still improved the predators'
   individual interception. Randomizing prey agility per environment, combined
   with a rehearsal pool, is a candidate ingredient for anchor v3.
4. **Anchor v3 (done 2026-10-08).** The overnight run with bootstrapped
   time-outs and a 60% rehearsal pool (floor 0.3) was accepted; see the
   [anchor v3 doc](anchor_v3_2026-10-08.md).

### Stage 8: Obstacles as Entities

Goal: extend the entity set from agents to agents plus obstacles under full
observability.

- Observations: visible obstacle relative position, obstacle radius/type,
  optionally line-of-sight or occlusion flags, attention over all visible
  obstacles (or nearest-k).
- Expectation: obstacles create natural traps (corners, gaps, occluders) that
  the empty sphere lacks, so coordination may emerge more readily.
- Metrics: catches near obstacles, enclosure using obstacles and walls,
  predator-obstacle and predator-predator collisions.

### Stage 9: Partial Observability, Belief, and Communication

Memory becomes necessary when the prey can be hidden. Use it when agents can
disappear behind obstacles, when velocity or intent must be inferred from
history, when sensors become noisy, or when line-of-sight masks remove current
state information.

- **Memory:** the GRU is the baseline. Compare other sequence models only
  after the GRU shows a bottleneck (see Research Parking Lot).
- **Belief estimator:** estimate compact task-relevant hidden state first (prey
  position, velocity, optional heading/acceleration, and uncertainty), not a
  full scene.
  - Baselines first: last-seen position, constant-velocity extrapolation, and
    uncertainty growing with time since the last sighting.
  - Inputs: own state history, visible prey-state history, visibility history,
    own action history, obstacle or occlusion indicators.
  - Outputs: deterministic `[p_prey, v_prey]`, or mean and log standard
    deviation for an uncertainty-aware belief.
  - Training target in simulation: the ground-truth prey state even when the
    policy observation is masked; supervised MSE first, then negative
    log-likelihood if the network predicts uncertainty.
  - Policy input: current visible observation, estimated prey position and
    velocity, uncertainty, and a visibility flag.
  - Later: prey-position heatmaps or dynamic occupancy maps; point-cloud
    completion only if scene reconstruction becomes necessary for control.
- **Shared belief:** when one predator sees the prey and the others do not,
  the team needs the information. Start with a consensus-based distributed
  estimator over the predators' local estimates (a control-theory baseline),
  then compare learned communication (targeted message passing).
- Success criteria: occluded-pursuit catch rate, collision rate,
  generalization to longer occlusions, and inference latency.

### Stage 10: Perception by Teacher-Student Distillation

Rendering cameras for thousands of parallel environments is not practical, so
perception policies are trained by distillation rather than from scratch:

- teacher: the full-state (privileged) policy from the previous stages,
- student: the same task with line-of-sight-limited or camera/depth inputs,
  trained to imitate the teacher on the student's own state distribution
  (DAgger), then fine-tuned with RL,
- the critic may keep privileged state (asymmetric actor-critic),
- the rendering budget sets the number of student environments,
- JEPA-style predictive representations are an optional research module at
  this stage.

### Deployment Track (only if real flight becomes a goal)

- Body-rate action interface with a low-level rate controller.
- Dynamics calibration and randomization: mass, thrust coefficients, motor
  delay, drag, latency, and sensor noise.
- Reward refinement for smooth, feasible commands (OPEN's second stage).
- Safety: a collision-avoidance safety filter (control barrier functions) or
  constrained PPO, evaluated as part of the cross-play protocol.
- Sim-to-sim checks (Crazyflow/JAX and Isaac Lab) before any real flight.

## Research Parking Lot

These components are interesting, but they wait for a measured bottleneck.

- **Sequence memory beyond GRU** (GTrXL / Transformer-XL, Mamba / Mamba-2,
  xLSTM): treat them as memory backbones, not replacements for PPO/MAPPO. Test
  only after a GRU baseline under occlusion shows that history matters. First
  comparison: feedforward attention policy vs GRU vs one modern sequence
  model. Most relevant uses: occluded prey tracking, long-horizon belief over
  hidden prey motion, and camera/depth token histories. Success: better
  occluded-pursuit success, fewer collisions, better generalization to longer
  occlusions, acceptable inference latency.
- **SPO (Simple Policy Optimization):** an update-rule ablation against the
  locked MAPPO baseline with the same checkpoint, seed, opponent pool, rollout
  budget, and evaluation. Judge it by cross-play robustness, collapse
  avoidance, OOB behavior, and latest-opponent recovery, not only training
  reward.
- **HAPPO/HATRPO:** only if agents become truly heterogeneous.
- **Full AlphaStar-style league:** automated population scoring, pruning, and
  role assignment, after the Stage 6 upgrades.
- **JEPA-style world models:** see the North Star.
- **MPC-structured actors:** keep MAPPO/self-play as the outer loop and
  replace only the action head:

  ```text
  observation -> neural actor -> MPC parameters/references -> MPC -> action
  ```

  The most relevant variant is an MA-AC-MPC-style actor. The network outputs
  physically meaningful MPC parameters (state-tracking and control-effort
  weights, local state or control references), and a short-horizon MPC
  produces the final command while enforcing dynamics, action limits,
  smoothness, and safety constraints. Raw MPC state must bypass observation
  normalization. Questions to answer before implementation:
  - Does the actor output references, cost weights, or both?
  - Is the MPC a differentiable actor layer or an environment-side wrapper?
  - Is the policy distribution over final actions or over MPC parameters?
  - Which raw state does the MPC need, and how does it stay consistent between
    Crazyflow/JAX and Isaac Lab?
  - Does it still work when both roles co-adapt in self-play?

  First ablation: the direct Gaussian actor vs the same architecture with an
  MPC-structured head, with rewards, observations, self-play schedule, pool
  evaluation, spawn distribution, steps, and seeds fixed. Success: equal or
  better latest-vs-latest performance, no worse worst-case cross-play,
  smoother actions, lower OOB/collision/saturation rates, and better
  Crazyflow/JAX to Isaac Lab transfer. This builds naturally on the body-rate
  interface.

## Correctness Gate (RL Core)

| # | Item | Status |
|---:|---|---|
| 1 | Exact frozen-role semantics (no loss, optimizer, or normalizer updates; GRU state still advances) | done; fingerprint check in every frozen run |
| 2 | Balanced vectorized evaluation (fixed episode quota per environment) | done; `episode_accounting: balanced_per_env` |
| 3 | Actor/environment action contract (tanh-squashed actions, stored pre-tanh values) | done |
| 4 | Episode-level arena safety metrics | done |
| 5 | Predator crash-cost semantics (one-time cost per crash) | done |
| 6 | RL-core regression suite | done; 162 tests (2026-10-07) |
| 7 | Recurrent state in evaluation and play | done; the July bug made evaluation nearly memoryless |
| 8 | Time-limit semantics | done; bootstrapped by default (837e0aa), and `--no-time-limit-bootstrap` reproduces the archived recipe bit-exactly |
| 9 | Value-target diagnostics | done; raw return scale, explained variance, and normalizer std in TensorBoard (77ca941) |
| 10 | Value normalizer policy | open; statistics frozen at an old scale. Keep them fixed; a PopArt-style recalibration needs its own A/B |
| 11 | Evaluation policy vs training policy | open; deterministic evaluation of a stochastic policy (prey std about 0.45) tests a different policy. Add stochastic evaluation as a second view |
| 12 | Training-seed noise floor | done for the predator role (3 seeds, 2026-10-07); the prey role is not yet measured |

Acceptance rule: no long league run unless the regression suite passes and the
runbook smoke reproduces bit-exactly. Behavior-changing fixes go through
controlled A/Bs. Record throughput before and after changes to the update path.

## Metrics

- **Outcome:** catch rate and OOB rates from deterministic, paired evaluation
  (1024 episodes, seed 42), and the 10/20/30 s horizon test (constant catch
  hazard = length-independent behavior).
- **Robustness:** the five reference opponents, held-out opponents never used
  in training, the league with Bradley-Terry ratings, and its non-transitive
  residuals.
- **Strategy** (`artifacts/isaac/diagnostics/boundary_analysis.py`): enclosure
  on the arena wall (the largest predator gap in the wall's tangent plane is
  below 180 deg), catches within 1 m of the wall, near misses, and the
  catch-time distribution. In free 3D space three predators can never
  surround the prey, because any three directions lie in a common hemisphere.
  The horizontal angular gap of `escape_analysis.py` is therefore only a
  legacy view. Also watch for role collapse (all predators doing the same
  thing), a predator that becomes useless, and dependence on slot order.
- **Training health:** explained variance, raw return scale, normalizer std,
  KL and clip fraction, and action std relative to its floor.
- **Videos** for qualitative checks of anything the numbers suggest.

## Reading Map

### Core MARL and MAPPO

- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
  - Why centralized training helps with non-stationarity.
  - https://arxiv.org/abs/1706.02275
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
  - Why MAPPO is a strong baseline and which implementation details matter.
  - https://arxiv.org/abs/2103.01955

### RL Correctness

- Pardo et al., "Time Limits in Reinforcement Learning"
  - Time-outs as truncation vs termination; the basis of the 2026-10-07 fix.
  - https://arxiv.org/abs/1712.00378
- van Hasselt et al., "Learning values across many orders of magnitude"
  - PopArt value normalization that preserves outputs when statistics change.
  - https://arxiv.org/abs/1602.07714

### Attention and Graph Structure

- Iqbal and Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement Learning"
  - Directly relevant to attention-based centralized critics.
  - https://arxiv.org/abs/1810.02912
- Velickovic et al., "Graph Attention Networks"
  - General graph-attention foundation.
  - https://arxiv.org/abs/1710.10903
- Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem"
  - Transformer-style MARL, useful for understanding a more ambitious end state.
  - https://arxiv.org/abs/2205.14953

### Heterogeneous MARL

- Kuba et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning"
  - HATRPO/HAPPO motivation for heterogeneous agents.
  - https://arxiv.org/abs/2109.11251

### Self-Play, League Training, and Autocurricula

- Baker et al., "Emergent Tool Use From Multi-Agent Autocurricula"
  - Why competitive self-play creates staged behaviors and counter-behaviors.
  - https://arxiv.org/abs/1909.07528
- Berner et al., "Dota 2 with Large Scale Deep Reinforcement Learning"
  - Large-scale self-play and continual training.
  - https://arxiv.org/abs/1912.06680
- Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning"
  - League-style training and exploiters as a robust self-play template.
  - https://www.nature.com/articles/s41586-019-1724-z
- Jaderberg et al., "Population Based Training of Neural Networks"
  - Population and hyperparameter scheduling ideas.
  - https://arxiv.org/abs/1711.09846
- Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"
  - PSRO: opponent mixtures from the empirical payoff matrix.
  - https://arxiv.org/abs/1711.00832
- Balduzzi et al., "Re-evaluating Evaluation"
  - Nash averaging as a rating that is robust to non-transitive cycles.
  - https://arxiv.org/abs/1806.02643
- Balduzzi et al., "Open-ended Learning in Symmetric Zero-sum Games"
  - Why non-transitive games need population diversity, not only a stronger agent.
  - https://arxiv.org/abs/1901.08106

### Partial Observability, Memory, and Communication

- Parisotto et al., "Stabilizing Transformers for Reinforcement Learning"
  - GTrXL as a recurrent/Transformer memory architecture for RL.
  - https://arxiv.org/abs/1910.06764
- Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
  - Segment-level recurrence concept.
  - https://arxiv.org/abs/1901.02860
- Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  - Research direction for long sequence memory.
  - https://arxiv.org/abs/2312.00752
- Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
  - Mamba-2 / SSD direction.
  - https://arxiv.org/abs/2405.21060
- Beck et al., "xLSTM: Extended Long Short-Term Memory"
  - Modern LSTM-family sequence model; a research comparison to GRU and Mamba-style memory.
  - https://arxiv.org/abs/2405.04517
- Mustafa et al., "Context-aware Mamba-based Reinforcement Learning for social robot navigation"
  - Example of Mamba used in robot navigation RL.
  - https://arxiv.org/abs/2408.02661
- Liu et al., "RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation"
  - Example of Mamba in robot manipulation and VLA-style policy modeling.
  - https://arxiv.org/abs/2406.04339
- Liu et al., "TrackingMiM: Efficient Mamba-in-Mamba Serialization for Real-time UAV Object Tracking"
  - UAV-relevant Mamba example for real-time visual tracking rather than control.
  - https://arxiv.org/abs/2507.01535
- Olfati-Saber, "Distributed Kalman Filtering for Sensor Networks" (IEEE CDC 2007)
  - Consensus-based distributed estimation; the baseline for a shared prey belief.
- Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning"
  - Learned communication protocols.
  - https://arxiv.org/abs/1605.06676
- Das et al., "TarMAC: Targeted Multi-Agent Communication"
  - Attention-based targeted message passing.
  - https://arxiv.org/abs/1810.11187

### Privileged Learning and Perception

- Chen et al., "Learning by Cheating"
  - Distilling a privileged teacher into a sensor-based student.
  - https://arxiv.org/abs/1912.12294
- Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain"
  - Teacher-student training for robust robot control.
  - https://arxiv.org/abs/2010.11251
- Kaufmann et al., "Champion-level drone racing using deep reinforcement learning"
  - Swift: state-based RL with body-rate commands, deployed with perception.
  - https://www.nature.com/articles/s41586-023-06419-4

### Quadrotor Control Interface and Safety

- Kaufmann, Bauersfeld, and Scaramuzza, "A Benchmark Comparison of Learned Control Policies for Agile Quadrotor Flight"
  - Compares control abstractions (rotor thrusts, body rates, velocities) for learned policies.
  - https://arxiv.org/abs/2202.10796
- Ames et al., "Control Barrier Functions: Theory and Applications"
  - Safety filters for collision avoidance.
  - https://arxiv.org/abs/1903.11199

### UAV Pursuit-Evasion and Online Planning

- Chen et al., "Multi-UAV Pursuit-Evasion with Online Planning in Unknown
  Environments by Deep Reinforcement Learning"
  - Closest current domain comparator for this project.
  - Combines MAPPO, calibrated UAV dynamics, collective thrust/body-rate
    actions, an attention-based observation encoder, an LSTM evader-prediction
    module, adaptive environment generation, and reward refinement for
    zero-shot real-world deployment.
  - Project connection: its strongest lesson is not "replace MAPPO"; it is to
    add explicit prey-belief prediction, hard-scenario generation, and
    deployment-oriented action/reward refinement around MAPPO.
  - https://arxiv.org/abs/2409.15866
  - https://sites.google.com/view/pursuit-evasion-rl
- Isaacs, "Differential Games" (1965)
  - Classical pursuit-evasion theory; capture conditions for faster evaders.

### Model-Based Control and MPC-Structured Actors

- Llanes et al., "Merging model-based control with multi-agent reinforcement
  learning for multi-agent cooperative teaming strategies"
  - Direct reference for an MA-AC-MPC-style actor where a neural cost network
    outputs MPC parameters and the MPC layer returns feasible low-level
    actions.
  - Project connection: late-stage comparison against the current direct
    Gaussian normalized-wrench actor while keeping the self-play and opponent
    pool stack intact.
  - https://arxiv.org/abs/2606.06011

## Near-Term Sequence (2026-10-07)

1. Prey agility curriculum: done; no coordination gain, better individual
   interception (see the experiment log).
2. Training-seed noise floor: done (see the experiment log).
3. Keep the current simulation for behavior research. CTBR and FPV-class
   dynamics come later as one deployment migration (see the plan).
4. Anchor v3: done and frozen (2026-10-08).
5. Add obstacles as entities (Stage 8) with obstacle-aware strategy metrics.
6. Add line-of-sight masking with the simple belief baselines, then the GRU,
   the estimator, and belief sharing (Stage 9).
7. Distill toward perception (Stage 10).
8. Take research branches only against a measured bottleneck.

## History

- July 2026: GRU evaluation and play did not commit recurrent hidden states
  between inference steps, so deterministic evaluation behaved almost
  memoryless. After the fix, `agent_576000.pt` went from a prey OOB rate of
  about 0.955 to 0.012, matching the training rollouts.
- July 2026: the correctness audit from commit `ceef347` closed gate items
  1-6 before long league runs were allowed again.
- Earlier status sections and near-term rules (before 2026-10-04) are
  superseded by the snapshot above. See this file's Git history.
