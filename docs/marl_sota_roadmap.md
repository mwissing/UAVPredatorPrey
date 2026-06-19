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
- Gaussian action policy.
- Centralized MAPPO state input.
- Experimental entity-attention centralized critic config.

### Partially Implemented

- Shared predator actor:
  - implemented inside the current team-agent interface,
  - not yet a fully separate per-UAV external MARL interface.
- Entity attention:
  - currently in the predator policy,
  - currently over prey + teammate entities for predators,
  - prey policy still falls back to a flat MLP,
  - not yet over obstacles or line-of-sight filtered entities.
- Centralized critic:
  - entity-attention critic implementation exists,
  - currently under validation against the flat centralized critic baseline.
- League training:
  - implemented as explicit checkpoint pools,
  - supports run-local auto-pool promotion,
  - supports cross-play-based weight updates,
  - supports PFSP-style weighted sampling,
  - supports per-env opponent mixing during single-agent frozen-opponent phases,
  - not yet a full AlphaStar-style league with main agents, exploiters,
    automatic role assignment, and population-level promotion rules.
- Domain randomization:
  - basic spawn/randomization exists,
  - not yet systematic mixed/random arena spawn curriculum,
  - not yet systematic initial velocity randomization,
  - not yet systematic dynamics/sensor randomization.

### Missing

- Prey-side entity-attention actor over predator entities.
- Recurrent actor/critic memory.
- Neural belief-state estimator for occluded prey position, velocity, and
  uncertainty.
- Obstacle-aware attention.
- Partial observability / line-of-sight masking.
- 3v1 shared-attention validation.
- HAPPO/HATRPO experiments.
- Full automated league population scoring, pruning, and role assignment.
- Mixed/random spawn curriculum.
- Initial velocity randomization curriculum.

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

Metrics:

- predator catch-rate cycles should become less extreme over pool runs,
- predator and prey OOB should remain low,
- predator soft-arena usage should remain low,
- prey soft-arena usage should remain low,
- visual behavior should show role flexibility rather than fixed slot roles.

### Stage 2: Add an Entity-Attention Centralized Critic

This is the next serious architecture upgrade.

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

Key failure modes to watch:

- all predators collapse into the same role,
- one predator becomes useless,
- policies rely on slot order,
- catch-rate oscillates without strategic improvement.

### Stage 4: Add Prey-Side Entity Attention

Goal: give the prey the same relational inductive bias that the predator team
already has, instead of using the current flat MLP fallback.

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

Practical default:

- keep phase-level pool composition disabled while validating per-env mixing,
- use per-env pool mixing only during single-agent freeze phases,
- start with about 25% pool envs and tune only after comparing videos and
  cross-play matrices.

### Stage 7: Add Obstacles as Entities

Goal: extend the entity set from agents only to agents + obstacles.

Observation direction:

- visible obstacle relative position,
- obstacle radius / type,
- optionally line-of-sight or occlusion flags,
- nearest-k or attention over all visible obstacles.

### Stage 8: Add Partial Observability and Memory

Only add memory when it solves a real missing-information problem.

Use memory when:

- agents can disappear behind obstacles,
- velocity or intent must be inferred from history,
- sensor observations become noisy,
- line-of-sight masks remove current state information.

Initial memory choice:

- GRU first.

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

## Near-Term Rule

Do not add all SOTA components at once.

The current near-term sequence after the 3v1 attention-critic per-env-pooling
run is:

1. validate per-env pooling against latest-vs-latest, pool cross-play, and
   phase videos,
2. if the run is stable, use per-env pooling as the default self-play
   curriculum mechanism,
3. run at least one empty-start-pool experiment to check whether the league can
   build useful opponents without seeded historical policies,
4. harden league logic with promotion gates, stale-entry pruning, and explicit
   cross-play matrices,
5. add mixed spawn and later random arena spawn as the next robustness
   curriculum,
6. add initial velocity randomization only after position randomization is
   stable,
7. add prey-side entity attention if prey behavior remains slot/order sensitive
   or brittle under randomized spawns,
8. add obstacles as entities,
9. add recurrent memory only when partial observability or occlusion makes the
   current state insufficient.
