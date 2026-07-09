# MARL SOTA Roadmap for UAV Predator-Prey

This note records the long-term architecture and training direction for the UAV
predator-prey project. It is a roadmap, not a commitment to implement every
component immediately.

The practical goal is to move from controlled 1v1 and 2v1 experiments toward a
modern, scalable multi-agent robotics setup that can handle 3v1, obstacles,
partial observability, and robust self-play.

## North Star: Visual Occluded Pursuit

The long-term end goal is a predator team that can catch a prey using only
on-board perception, even when the prey is temporarily hidden behind obstacles.

This means the final problem is not just full-state continuous-control MARL. It
is multi-agent visual pursuit under partial observability:

- predators receive camera-based or local sensor observations instead of perfect
  prey coordinates,
- obstacles can occlude the prey,
- predators must maintain a belief over where the hidden prey is likely to be,
- predators must predict likely prey motion and intercept paths,
- predators must coordinate without relying on perfect global state,
- all agents must still satisfy flight stability and collision-safety
  constraints.

The roadmap should therefore progress in controlled steps:

1. full-state 1v1 / 2v1 / 3v1 pursuit,
2. obstacle-aware full-state pursuit,
3. line-of-sight masked state observations,
4. recurrent belief-based policies,
5. camera- or depth-based observations,
6. JEPA-style or other world-model representation learning for predictive
   latent scene understanding.

The closest current external reference for this direction is Chen et al.,
"Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep
Reinforcement Learning" (OPEN). It should be treated as the main domain
comparator for obstacle-aware, partially observable, 3D UAV pursuit-evasion:

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

JEPA-style world models are not the immediate next step, but they are a
candidate research module once partial observability and visual perception
become the bottleneck. The likely role is representation learning or auxiliary
prediction in latent space, not replacing MAPPO/self-play as the control
training backbone.

## Target End State

### Actor

- Shared per-predator parameters.
- Entity or graph attention over visible objects:
  - own UAV state,
  - prey state,
  - teammate predator states,
  - later: obstacles, arena features, line-of-sight masks.
- Prey actor should also use entity attention over predator entities:
  - own prey state as the controlled agent state,
  - predator relative states as entities,
  - later: obstacles and line-of-sight masks as additional entities.
- Recurrent memory only when partial observability is real:
  - GRU as the first practical option,
  - GTrXL / Transformer-XL style memory as a later option,
  - Mamba / Mamba-2 style state-space sequence models only as a
    research-level extension,
  - xLSTM as another modern recurrent-memory research option.
- Gaussian continuous-action head.
- Optional late-stage model-based actor:
  - keep MAPPO/self-play/league training as the outer learning loop,
  - replace only the low-level action head with an MPC-structured action
    layer after strong direct-action baselines exist,
  - let the neural actor output physically meaningful MPC cost parameters or
    local references rather than raw body-rate/thrust commands,
  - use MPC to enforce short-horizon dynamics, action limits, smoothness, and
    safety constraints,
  - compare against the direct Gaussian body-rate/thrust actor under the same
    self-play, cross-play, and robustness evaluation protocol.

### Critic

- Centralized critic for MAPPO-style training.
- Global entity encoder over all agents and environment objects.
- Attention or graph network over:
  - predators,
  - prey,
  - obstacles,
  - arena/boundary features.
- Optional temporal memory when observations become history-dependent.
- Scalar value head.

### Training

- MAPPO as the stable baseline.
- HAPPO/HATRPO only if agents become truly heterogeneous.
- SPO / Simple Policy Optimization as a later algorithm-update ablation:
  - test only after the current MAPPO self-play loop is stable enough for a
    fair comparison,
  - keep the same checkpoint, seed, opponent pool, rollout budget, and
    evaluation protocol as the MAPPO baseline,
  - judge it by cross-play robustness, collapse avoidance, OOB behavior, and
    latest-opponent recovery, not only by training reward or one current-pair
    catch rate,
  - treat it as an optimizer/update-rule experiment, not as a replacement for
    the league, pooling, GRU, attention, or environment work.
- Alternating self-play with freeze phases.
- League / opponent-pool training to avoid cyclic forgetting.
- Per-env opponent-pool mixing during single-agent freeze phases:
  - most envs use the latest frozen opponent,
  - a controlled fraction of envs use historical pool opponents,
  - this is meant to reduce overfitting to one frozen opponent behavior inside
    a phase.
- PFSP-style opponent sampling as a league upgrade:
  - sample old opponents by measured difficulty / win-rate,
  - prefer opponents that are neither trivial nor impossible,
  - keep this as a sampling rule before building full automated population
    management.
- Curriculum:
  1. no obstacles,
  2. 1v1,
  3. 2v1,
  4. 3v1,
  5. simple obstacles,
  6. complex obstacles / occlusion.
- Reward shaping lifecycle:
  - use dense helper rewards early when sparse task rewards are too hard,
  - explicitly ablate or anneal helper rewards once the behavior is learnable,
  - treat proximity, cover-seeking, boundary-progress, and similar terms as
    scaffolding rather than permanent objectives unless diagnostics prove they
    still improve final behavior,
  - watch for shaping rewards creating local optima such as hovering near the
    prey, clustering, boundary surfing, or avoiding risk instead of solving the
    task.
- Domain randomization:
  - spawn positions,
  - initial velocities,
  - UAV dynamics,
  - action noise,
  - observation noise,
  - sensor dropout / line-of-sight masking.

## Current Implementation Status

### Already Implemented

- MAPPO training through skrl.
- Alternating freeze-based self-play.
- Hysteresis curriculum scheduler.
- Opponent-pool / league-style checkpoint sampling.
- Per-env opponent-pool mixing for frozen opponents during single-agent phases.
- Auto-pool promotion into run-local `opponent_pool_auto.json`.
- Cross-play evaluation against pool entries.
- PFSP-style weighted opponent sampling.
- Soft OOB arena penalties.
- 1v1 and 2v1 survival curriculum tasks.
- First shared per-predator attention actor for predator teams.
- Prey-side entity-attention actor over predator entities.
- Gaussian action policy.
- Centralized MAPPO state input.
- Experimental entity-attention centralized critic config.
- Larger symmetric predator/prey entity-attention actor configuration.
- GRU recurrent actor/critic configuration for the large entity-attention
  setup.
- Recurrent MAPPO wrapper for GRU rollouts and sequence-based PPO updates.
- Recurrent-state reset/commit handling in evaluation and play scripts.
- Random-spawn 3v1 task variant with minimum-separation constraints.

### Partially Implemented

- Shared predator actor:
  - implemented inside the current team-agent interface,
  - not yet a fully separate per-UAV external MARL interface.
- Entity attention:
  - currently in both predator and prey policies,
  - currently over prey + teammate entities for predators,
  - currently over predator entities for prey,
  - not yet over obstacles or line-of-sight filtered entities.
- Centralized critic:
  - entity-attention critic implementation exists,
  - currently used in the attention-critic configurations,
  - still needs systematic ablations against flat centralized critics.
- League training:
  - implemented as explicit checkpoint pools,
  - supports run-local auto-pool promotion,
  - supports cross-play-based weight updates,
  - supports PFSP-style weighted sampling,
  - supports per-env opponent mixing during single-agent frozen-opponent phases,
  - not yet a full AlphaStar-style league with main agents, exploiters,
    automatic role assignment, and population-level promotion rules.
- Domain randomization:
  - random-spawn 3v1 task exists,
  - not yet a staged mixed-to-random curriculum,
  - not yet systematic initial velocity randomization,
  - not yet systematic dynamics/sensor randomization.
- Recurrent memory:
  - GRU actor/critic path exists,
  - GRU evaluation/play state handling has been fixed and validated,
  - currently being tested as the active 3v1 league baseline,
  - still needs robust multi-seed league evidence and ablation against the
    feedforward large attention baseline.

### Missing

- Neural belief-state estimator for occluded prey position, velocity, and
  uncertainty.
- Obstacle-aware attention.
- Partial observability / line-of-sight masking.
- Locked 3v1 GRU league baseline with seed/cross-play robustness evidence.
- MPC-structured actor or local controller interface as a late-stage
  architecture experiment.
- HAPPO/HATRPO experiments.
- SPO / Simple Policy Optimization ablation against the locked MAPPO baseline.
- Full automated league population scoring, pruning, and role assignment.
- Initial velocity randomization curriculum.

## Current Active Status: 3v1 Large GRU League

Current focus:

- task: `3v1-survival-soft-oob-teammate-vel-random-spawn-v0`
- agent: `skrl_mappo_attention_critic_prey_attention_large_gru_cfg_entry_point`
- architecture:
  - shared per-predator entity-attention actor,
  - prey entity-attention actor,
  - entity-attention centralized critic,
  - GRU memory in actor/critic,
  - Gaussian continuous-action heads.
- reward configuration:
  - proximity reward disabled to reduce predator clustering,
  - catch and distance-progress remain the main predator pursuit signals,
  - prey has alive/evasion/distance and stability penalties.
- training setup:
  - MAPPO,
  - hysteresis freeze phases,
  - per-env opponent-pool mixing during freeze phases,
  - run-local auto-pool promotion,
  - elite/recent opponent pool tracking,
  - cross-play and PFSP-style weight updates,
  - longer predator recovery phases when the latest prey becomes hard.

Important implementation note:

- GRU evaluation/play originally did not commit recurrent hidden states between
  inference steps.
- This caused deterministic eval to behave almost memoryless and produced
  misleading prey-OOB collapses.
- The eval/play recurrent-state path now:
  - keeps the new hidden state for continuing envs,
  - zeros hidden state for terminated/truncated envs.
- After the fix, `agent_576000.pt` from the joint GRU run changed from
  `prey_oob_rate ~= 0.955` in eval to `prey_oob_rate ~= 0.012`, matching the
  TensorBoard rollout behavior much more closely.

Current GRU league status:

- The GRU stack is no longer only an implementation experiment; it is the
  active obstacle-free 3v1 league candidate.
- Random-spawn and lifted-sphere arena training are active.
- Proximity reward is disabled; catch and distance-progress are the main
  predator pursuit signals.
- GRU-only pool entries should be used for league training. Old feedforward
  checkpoints remain useful for diagnostic migration tests, not as normal GRU
  league opponents.
- Recent curriculum runs showed that the old elite prey pool can be solved
  while the latest prey remains difficult. This means latest-vs-latest and
  pool cross-play must both be tracked.
- A long frozen-prey predator recovery diagnostic from the current hard matchup
  reached roughly `clean_catch_rate ~= 0.996` with near-zero OOB. This indicates
  that the predator architecture/reward can solve the latest prey when given
  enough uninterrupted gradient time.
- Current bottleneck: curriculum/league dynamics and phase scheduling, not
  basic predator capability.
- Current near-term scheduler direction:
  - use longer predator phases for hard latest-prey matchups,
  - avoid switching to `both` too early when predator promotion plateaus,
  - keep pool exposure modest when the old pool is already solved,
  - keep safe non-promoted policies in a recent pool so useful recovery
    checkpoints are not lost.

## Recommended Implementation Order

### Stage 0: Establish Evaluation Anchors

Goal: make self-play progress interpretable before judging architecture or
reward changes.

Reason:

- Self-play catch-rate alone is ambiguous:
  - a flat catch-rate can mean both teams stagnate,
  - or both teams improve at the same speed,
  - or one side exploits a temporary weakness of the other side.
- Frozen reference opponents make progress measurable across time.

Evaluation protocol:

- evaluate current predator against the current prey,
- evaluate current predator against older frozen prey checkpoints,
- evaluate older frozen predator checkpoints against the current prey,
- keep a small cross-play matrix for important checkpoints,
- optionally add scripted diagnostic opponents for measurement only, not as the
  main training target.

Gate:

- do not treat a run as a new reference baseline unless it has been evaluated
  against at least the current opponent and a small frozen checkpoint set.

### Controlled Ablation Rule

Goal: make architecture comparisons defensible.

When comparing flat MLP, predator attention, prey attention, and
entity-attention critics, keep optimizer and exploration settings matched unless
the experiment is explicitly about hyperparameters.

Control at least:

- learning rate and scheduler,
- entropy scale,
- initial log standard deviation,
- log standard deviation min/max,
- rollout length and number of environments,
- reward scales and termination rules.

### Stage 1: Stabilize 2v1 Attention Self-Play

Goal: prove that the shared-attention actor can learn stable coordinated
predator behavior and robust prey evasions.

Status: completed as a stepping stone. The project has moved to 3v1.

Metrics:

- predator catch-rate cycles should become less extreme over pool runs,
- predator and prey OOB should remain low,
- predator soft-arena usage should remain low,
- prey soft-arena usage should remain low,
- visual behavior should show role flexibility rather than fixed slot roles.

### Stage 2: Add an Entity-Attention Centralized Critic

This was the first serious architecture upgrade beyond a flat centralized
critic.

Status: implemented and used in the active attention-critic configurations.
Still needs clean ablations against the flat centralized critic.

Reason:

- The actor now has a relational inductive bias.
- The critic is still a flat MLP over the centralized state.
- A critic that understands agent-agent and agent-boundary relations should
  produce better value estimates and cleaner advantages.

Expected benefit:

- more stable MAPPO updates,
- better scaling from 2v1 to 3v1,
- easier obstacle extension.

### Stage 3: Validate 3v1 With Shared Attention

Goal: check whether the shared actor actually scales better than the old flat
team policy.

Status: in progress. 3v1 behavior exists, but the active target is now a robust
large GRU league baseline under random spawn.

Key failure modes to watch:

- all predators collapse into the same role,
- one predator becomes useless,
- policies rely on slot order,
- catch-rate oscillates without strategic improvement.

### Stage 4: Add Prey-Side Entity Attention

Goal: give the prey the same relational inductive bias that the predator team
already has, instead of relying on the older flat MLP fallback.

Status: implemented in the current prey-attention and large-GRU configurations.
Needs ablation against the earlier flat-prey baseline if the improvement must be
claimed rigorously.

Priority:

- add this after stable 3v1 without obstacles,
- do this before obstacle-aware attention if the prey remains a limiting factor,
- use matched optimizer / exploration settings when comparing against the flat
  prey baseline.

Reason:

- The prey observation is a set-like collection of predator states.
- A flat MLP can overfit to predator slot order.
- A prey attention actor can learn which predator is currently most dangerous,
  which predator is cutting off the path, and which predator can be ignored.

Expected benefit:

- better scaling from 2v1 to 3v1,
- more robust evasive behavior across randomized spawn orders,
- cleaner extension to obstacle and line-of-sight entities later.

Implementation direction:

- keep the prey as one skrl agent with a 4D action output,
- encode the prey's own state,
- encode each predator relative state as an entity,
- attend from prey state to predator entities,
- feed own-state embedding + attention context into the Gaussian action head.

### Stage 5: Spawn and Initial-State Robustness Curriculum

Goal: prevent both sides from learning opening-book strategies that only work
from the current symmetric spawn geometry.

Status: random-spawn task exists and is the active 3v1 training task. A staged
mixed-to-random curriculum and initial velocity curriculum are still missing.

Reason:

- If predators always spawn in fixed ring-like positions around the prey, the
  policies can memorize common first moves.
- Robust pursuit/evasion should work from different relative geometries, not
  only from the current curriculum start state.
- This should be added before obstacles so failures remain attributable to
  start-state distribution rather than obstacle reasoning.

Recommended progression:

1. current ring spawn as the baseline,
2. mixed spawn:
   - mostly current ring spawn,
   - some wider predator ring spawns,
   - some random arena spawns,
3. full random arena spawn with minimum separation constraints,
4. small initial horizontal velocity noise,
5. larger velocity and yaw/angular-velocity noise after position
   randomization is stable.

Required constraints:

- minimum predator-prey distance to avoid immediate trivial catches,
- minimum predator-predator distance to avoid spawn collisions,
- margin from soft arena boundary,
- height near target height,
- optional cap on initial speed during early curriculum stages.

Metrics to watch:

- catch-rate should not be dominated by instant spawn catches,
- predator/prey OOB should remain low,
- episode length should not collapse due to bad initial states,
- policies should still work on the original ring spawn evaluation.

### Stage 6: League Hardening After Per-Env Pooling

Goal: turn per-env pooling from a useful anti-overfitting mechanism into a
reliable training distribution.

Status: in progress. Per-env pooling, auto-promotion, cross-play, PFSP weights,
and run-local pools exist. The current GRU league is being built from a fresh
GRU-compatible pool.

Current mechanism:

- single-agent phases can mix latest frozen opponent and pool opponents across
  envs,
- auto-pool promotion writes new candidates into the run-local pool,
- cross-play updates pool weights,
- PFSP-style sampling can emphasize difficult opponents.

Next upgrades:

- start some diagnostic runs from an empty pool to verify the league can build
  itself,
- keep explicit evaluation against:
  - latest opponent,
  - current run-local pool,
  - selected historical reference opponents,
- add minimum-quality gates before promotion:
  - role-specific win/catch threshold,
  - max own OOB / soft-arena usage,
  - max opponent exploit via OOB / crash,
- add stale-entry pruning based on cross-play usefulness,
- separate conceptual roles later:
  - main agents,
  - exploiters,
  - historical reference policies.

Practical default for the current GRU league:

- keep phase-level pool composition disabled while validating per-env mixing,
- use per-env pool mixing only during single-agent freeze phases,
- use a GRU-only pool,
- start lower while the pool is tiny and increase once several compatible
  predator/prey entries exist,
- compare latest-vs-latest, pool cross-play, and videos before changing pool
  probability.

### Stage 7: Lock A 3v1 GRU Baseline

Goal: freeze one defensible obstacle-free 3v1 baseline before adding obstacles
or perception.

Required evidence:

- one named checkpoint or checkpoint pair,
- exact training command and pool file,
- 5-seed deterministic eval,
- latest-vs-latest eval,
- predator-vs-prey pool cross-play matrix,
- representative videos,
- brief behavior notes:
  - pursuit behavior,
  - intercept behavior,
  - prey evasive behavior,
  - failure modes.

Suggested baseline gate:

```text
clean_catch_rate:        high enough to show predator competence
predator_oob_rate:       below 5-10%
prey_oob_rate:           below 5-10%
forced_prey_oob_rate:    low
worst-case cross-play:   no catastrophic opponent hole
videos:                  visible pursuit/intercept/evasion, not only instant catches
```

### Stage 8: Add Obstacles as Entities

Goal: extend the entity set from agents only to agents + obstacles.

Observation direction:

- visible obstacle relative position,
- obstacle radius / type,
- optionally line-of-sight or occlusion flags,
- nearest-k or attention over all visible obstacles.

### Stage 9: Add Partial Observability and Memory

Memory is now available as a GRU baseline earlier than originally planned. The
next use case should be partial observability, where memory solves a real
missing-information problem rather than acting only as extra capacity.

Use memory when:

- agents can disappear behind obstacles,
- velocity or intent must be inferred from history,
- sensor observations become noisy,
- line-of-sight masks remove current state information.

Initial memory choice:

- GRU first. This is now implemented and should be the baseline memory model.

Research memory choices:

- GTrXL / Transformer-XL,
- Mamba / Mamba-2 style state-space sequence models,
- xLSTM-style modern recurrent memory.

Modern sequence-memory options:

- Treat Mamba, Mamba-2, or xLSTM as memory/backbone modules, not as
  replacements for PPO/MAPPO.
- Test them only after a GRU memory baseline shows that history is useful.
- Most relevant use case in this project:
  - occluded prey tracking,
  - long-horizon belief over hidden prey motion,
  - camera/depth/lidar token histories,
  - temporal fusion of own state, teammate states, and visible object tokens.
- First comparison should be:
  1. feedforward entity-attention policy,
  2. GRU entity-attention policy,
  3. one modern sequence-memory variant: Mamba, Mamba-2, or xLSTM.
- Success criteria:
  - better occluded-pursuit success rate,
  - lower collision rate,
  - better generalization to longer occlusions,
  - no unacceptable inference-latency increase.
- Do not add it for full-state predator-prey without occlusion; in that setting
  it is likely extra complexity rather than the bottleneck.

Neural belief-state estimator option:

- Use this after line-of-sight masking or obstacle occlusion makes the prey
  state genuinely hidden.
- Estimate compact task-relevant hidden state first, not a full hidden point
  cloud:
  - prey position,
  - prey velocity,
  - optional heading/acceleration,
  - uncertainty over the estimate.
- Possible input:
  - own state history,
  - visible prey-state history,
  - visibility / line-of-sight history,
  - own action history,
  - obstacle or occlusion indicators.
- Possible output:
  - deterministic estimate `[p_prey, v_prey]`,
  - or mean and log standard deviation for an uncertainty-aware belief.
- Training target in simulation:
  - use ground-truth prey state even when the policy observation is masked,
  - start with supervised MSE for position/velocity,
  - upgrade to negative log likelihood if the network predicts uncertainty.
- Policy input after estimator:
  - current visible observation,
  - estimated prey position and velocity,
  - uncertainty,
  - visibility flag.
- Later extensions:
  - prey-position probability heatmap,
  - dynamic occupancy map,
  - voxel / point-cloud completion only if visual scene reconstruction becomes
    necessary for control.
- First baseline before a neural estimator:
  - last-seen prey position,
  - constant-velocity extrapolation,
  - uncertainty increasing with time since last seen.

### Stage 10: Investigate MPC-Structured Actors

Goal: test whether model-based local control improves physical robustness
without replacing the self-play and league stack.

This is a final-stage research topic, not a near-term replacement for the
active GRU league. It should only be tested after there is a strong direct
body-rate/thrust baseline under random spawns, obstacles, and cross-play.

Core idea:

```text
observation -> neural actor -> MPC parameters/references -> MPC -> action
```

The most relevant variant is an MA-AC-MPC-style actor:

- the neural policy observes the same local entity/state features as the
  direct actor,
- the network outputs structured MPC parameters such as:
  - state tracking weights,
  - control effort weights,
  - local state/reference targets,
  - control references,
- a short-horizon MPC layer produces the actual body-rate/thrust command,
- PPO/MAPPO still trains the actor/critic using the same opponent-pool and
  cross-play machinery,
- raw MPC state must bypass observation normalization so the controller sees
  physically meaningful positions, velocities, attitude, and rates.

Questions to answer before implementation:

- Does the actor output local references, MPC cost weights, or both?
- Does the MPC run as a differentiable actor layer or as a non-differentiable
  environment-side controller wrapper?
- Is the policy distribution placed over final low-level actions or over the
  MPC parameter vector?
- What raw state is required by the MPC, and how is it kept consistent between
  Crazyflow/JAX and Isaac Lab?
- Does the method still work when both predator and prey are co-adapting
  through self-play, not only against fixed scripted opponents?

First useful ablation:

```text
direct Gaussian actor
vs.
same architecture + MPC-structured action head
```

Keep fixed:

- reward terms,
- observation layout,
- self-play schedule,
- pool/cross-play evaluation,
- random spawn distribution,
- number of environment steps and seeds.

Success criteria:

- equal or better catch/survival performance under latest-vs-latest,
- no worse worst-case pool cross-play,
- smoother and more physically plausible actions,
- lower OOB/collision or control-saturation rates,
- better transfer from Crazyflow/JAX to Isaac Lab fine-tuning.

## Reading Map

### Core MARL and MAPPO

- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
  - Why centralized training helps with non-stationarity.
  - https://arxiv.org/abs/1706.02275
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
  - Why MAPPO is a strong baseline and what implementation details matter.
  - https://arxiv.org/abs/2103.01955

### Attention and Graph Structure

- Iqbal and Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement Learning"
  - Directly relevant to attention-based centralized critics.
  - https://arxiv.org/abs/1810.02912
- Velickovic et al., "Graph Attention Networks"
  - General graph-attention foundation.
  - https://arxiv.org/abs/1710.10903
- Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem"
  - Transformer-style MARL, useful for understanding the more ambitious end state.
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

### Partial Observability and Memory

- Parisotto et al., "Stabilizing Transformers for Reinforcement Learning"
  - GTrXL as a recurrent/Transformer memory architecture for RL.
  - https://arxiv.org/abs/1910.06764
- Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
  - Segment-level recurrence concept.
  - https://arxiv.org/abs/1901.02860
- Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  - Research direction for long sequence memory; not an immediate implementation target.
  - https://arxiv.org/abs/2312.00752
- Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
  - Mamba-2 / SSD direction; relevant if Mamba becomes a serious memory-backbone experiment.
  - https://arxiv.org/abs/2405.21060
- Beck et al., "xLSTM: Extended Long Short-Term Memory"
  - Modern LSTM-family sequence model; useful as a research comparison to GRU and Mamba-style memory.
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

### UAV Pursuit-Evasion and Online Planning

- Chen et al., "Multi-UAV Pursuit-Evasion with Online Planning in Unknown
  Environments by Deep Reinforcement Learning"
  - Closest current domain comparator for this project.
  - Combines MAPPO, calibrated UAV dynamics, collective thrust/body-rate
    actions, an attention-based observation encoder, an LSTM evader-prediction
    module, adaptive environment generation, and reward refinement for
    zero-shot real-world deployment.
  - Project connection: compare our current GRU self-play league against their
    prediction-and-curriculum route. Their strongest lesson is not "replace
    MAPPO"; it is to add explicit prey-belief prediction, hard-scenario
    generation, and deployment-oriented action/reward refinement around MAPPO.
  - https://arxiv.org/abs/2409.15866
  - https://sites.google.com/view/pursuit-evasion-rl

### Model-Based Control and MPC-Structured Actors

- Llanes et al., "Merging model-based control with multi-agent reinforcement
  learning for multi-agent cooperative teaming strategies"
  - Direct reference for an MA-AC-MPC-style actor where a neural cost network
    outputs MPC parameters and the MPC layer returns feasible low-level
    actions.
  - Project connection: late-stage comparison against the current direct
    Gaussian body-rate/thrust actor while keeping the self-play and opponent
    pool stack intact.
  - https://arxiv.org/abs/2606.06011

## Near-Term Rule

Do not add all SOTA components at once.

The current near-term sequence is:

1. Continue the 3v1 large-GRU league from the strongest recent predator/prey
   checkpoint pair, using a GRU-only opponent pool.
2. Keep latest-vs-latest and pool cross-play separate:
   - latest predator vs latest prey,
   - latest predator vs all prey pool entries,
   - latest prey vs all predator pool entries,
   - selected pool predator/prey cross-play pairs.
3. When the old pool is solved but the latest prey remains hard, prioritize
   latest-opponent recovery over more old-pool exposure:
   - longer predator phases,
   - modest per-env pool probability,
   - promotion-failure continuation toward predator instead of early `both`
     training.
4. Keep recent safe-but-not-elite policies as a diagnostic/training buffer, but
   avoid treating them as elite promotion evidence until cross-play confirms
   robustness.
5. If behavior is visibly good, freeze a named 3v1 GRU baseline:
   - checkpoint,
   - pool file,
   - command,
   - commit,
   - eval JSONs,
   - representative videos.
6. Run robustness diagnostics:
   - seeds 42, 43, 44, 45, 46,
   - deterministic eval,
   - pool cross-play,
   - visual review of representative successes and failures.
7. If the baseline is robust, document it as the obstacle-free reference.
8. Only then add obstacles as entities under full observability.
9. After obstacle-aware full-state pursuit works, add line-of-sight masking and
   partial observability.
10. Use the GRU baseline as the first memory model for occluded pursuit; compare
   Mamba/Mamba-2/xLSTM-style memory only after GRU shows a clear bottleneck.
11. Treat SPO / Simple Policy Optimization as an algorithm-update ablation
    only after the MAPPO GRU league baseline is locked.
12. Treat MPC-structured actors as a later research branch after the direct
    GRU/attention stack has a defensible obstacle-aware baseline.
