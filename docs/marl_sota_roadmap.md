# MARL SOTA Roadmap for UAV Predator-Prey

This note records the long-term architecture and training direction for the UAV
predator-prey project. It is a roadmap, not a commitment to implement every
component immediately.

The practical goal is to move from controlled 1v1 and 2v1 experiments toward a
modern, scalable multi-agent robotics setup that can handle 3v1, obstacles,
partial observability, and robust self-play.

## Target End State

### Actor

- Shared per-predator parameters.
- Entity or graph attention over visible objects:
  - own UAV state,
  - prey state,
  - teammate predator states,
  - later: obstacles, arena features, line-of-sight masks.
- Recurrent memory only when partial observability is real:
  - GRU as the first practical option,
  - GTrXL / Transformer-XL style memory as a later option,
  - Mamba-style sequence models only as a research-level extension.
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
- Soft OOB arena penalties.
- 1v1 and 2v1 survival curriculum tasks.
- First shared per-predator attention actor for predator teams.
- Gaussian action policy.
- Centralized MAPPO state input.

### Partially Implemented

- Shared predator actor:
  - implemented inside the current team-agent interface,
  - not yet a fully separate per-UAV external MARL interface.
- Entity attention:
  - currently only in the predator policy,
  - currently over prey + teammate entities,
  - not yet over obstacles or line-of-sight filtered entities.
- League training:
  - implemented as explicit checkpoint pools,
  - not yet automated population management.
- Domain randomization:
  - basic spawn/randomization exists,
  - not yet systematic dynamics/sensor randomization.

### Missing

- Entity-attention centralized critic.
- Recurrent actor/critic memory.
- Obstacle-aware attention.
- Partial observability / line-of-sight masking.
- 3v1 shared-attention validation.
- HAPPO/HATRPO experiments.
- Automated league population scoring and pruning.

## Recommended Implementation Order

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

### Stage 4: Add Obstacles as Entities

Goal: extend the entity set from agents only to agents + obstacles.

Observation direction:

- visible obstacle relative position,
- obstacle radius / type,
- optionally line-of-sight or occlusion flags,
- nearest-k or attention over all visible obstacles.

### Stage 5: Add Partial Observability and Memory

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
- Mamba-style state-space sequence model.

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

## Near-Term Rule

Do not add all SOTA components at once.

The next architecture step after the current 2v1 attention pool run should be:

1. evaluate whether shared-attention actor + pool stabilizes 2v1,
2. if yes, add entity-attention centralized critic,
3. then validate 3v1,
4. then add obstacles as entities,
5. then add memory only when partial observability is introduced.
