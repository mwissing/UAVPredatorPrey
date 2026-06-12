# Papers to Read

This file tracks papers relevant to the UAV predator-prey MARL roadmap.

Use it as a working reading list: after reading a paper, add short notes under
`My notes` in your own words. The goal is not just to collect references, but
to connect each paper to our implementation decisions.

## Reading Order

### 1. MAPPO Baseline

**The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**  
Chao Yu et al., 2021  
Link: https://arxiv.org/abs/2103.01955

Why it matters:

- Explains why MAPPO is a strong practical baseline.
- Helps interpret our current skrl MAPPO setup.
- Useful for understanding implementation details such as advantage
  normalization, value normalization, rollout length, and clipping.

Project connection:

- We currently use MAPPO as the main training algorithm.
- This paper helps decide whether problems come from architecture/curriculum
  rather than from using MAPPO itself.

Status:

- [ ] Read

My notes:

-

### 2. Centralized Training / Decentralized Execution

**Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**  
Ryan Lowe et al., 2017  
Link: https://arxiv.org/abs/1706.02275

Why it matters:

- Introduces the core centralized-training idea used by many MARL methods.
- Explains non-stationarity when multiple agents learn simultaneously.
- Useful background for predator-vs-prey self-play.

Project connection:

- Our predator/prey policies are trained in a non-stationary setting.
- The MAPPO critic uses centralized information, while actors use observations.

Status:

- [ ] Read

My notes:

-

### 3. Attention-Based Critic

**Actor-Attention-Critic for Multi-Agent Reinforcement Learning**  
Shariq Iqbal and Fei Sha, 2018  
Link: https://arxiv.org/abs/1810.02912

Why it matters:

- Directly relevant to attention in MARL.
- Uses attention in the critic to focus on relevant agents.
- Good conceptual bridge to our planned entity-attention centralized critic.

Project connection:

- We already implemented a first shared-attention predator actor.
- The likely next architecture step is an entity-attention centralized critic.

Status:

- [ ] Read

My notes:

-

### 4. Graph Attention Foundation

**Graph Attention Networks**  
Petar Velickovic et al., 2017  
Link: https://arxiv.org/abs/1710.10903

Why it matters:

- General foundation for attention over graph/entity neighborhoods.
- Helps understand agent-agent and agent-obstacle relation modeling.

Project connection:

- Future observations can be represented as entities:
  predators, prey, obstacles, arena features.
- Attention/graph layers can replace fixed slot-based flat MLPs.

Status:

- [ ] Read

My notes:

-

### 5. HAPPO / HATRPO

**Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning**  
Jakub Grudzien Kuba et al., 2021  
Link: https://arxiv.org/abs/2109.11251

Why it matters:

- Introduces HATRPO and HAPPO.
- Relevant when agents are heterogeneous and parameter sharing is no longer
  appropriate.
- Gives a more principled view of multi-agent policy updates.

Project connection:

- Not needed immediately while predators share dynamics and roles.
- Relevant later if predators/prey become structurally different agents or if
  we stop sharing predator parameters.

Status:

- [ ] Read

My notes:

-

### 6. Multi-Agent Transformer

**Multi-Agent Reinforcement Learning is a Sequence Modeling Problem**  
Muning Wen et al., 2022  
Link: https://arxiv.org/abs/2205.14953

Why it matters:

- Shows a transformer-style view of multi-agent action generation.
- Useful for understanding a more ambitious long-term architecture.

Project connection:

- Not the next implementation step.
- Helps frame why transformer-style MARL might eventually be useful for 3v1,
  many entities, and variable agent counts.

Status:

- [ ] Read

My notes:

-

### 7. Autocurricula and Emergent Strategy

**Emergent Tool Use From Multi-Agent Autocurricula**  
Bowen Baker et al., 2019  
Link: https://arxiv.org/abs/1909.07528

Why it matters:

- Explains how competitive self-play can create staged strategy/counter-strategy
  progressions.
- Useful for interpreting our predator/prey cycles.

Project connection:

- Our hysteresis curriculum and opponent pools are simple versions of an
  autocurriculum.
- Helps reason about cyclic behavior and why league play can matter.

Status:

- [ ] Read

My notes:

-

### 8. Large-Scale Self-Play

**Dota 2 with Large Scale Deep Reinforcement Learning**  
OpenAI et al., 2019  
Link: https://arxiv.org/abs/1912.06680

Why it matters:

- Large-scale PPO/self-play example.
- Useful for understanding what robust population training looks like at scale.

Project connection:

- Our setup is much smaller, but the core ideas of self-play, robustness, and
  opponent diversity are relevant.

Status:

- [ ] Read

My notes:

-

### 9. League Training

**Grandmaster level in StarCraft II using multi-agent reinforcement learning**  
Oriol Vinyals et al., 2019  
Link: https://www.nature.com/articles/s41586-019-1724-z

Why it matters:

- Strong reference for league-style training and exploiters.
- Shows why a pool of opponents is useful to avoid forgetting and overfitting to
  the current opponent.

Project connection:

- Our opponent-pool JSONs are a small, manual version of league training.
- Future work could automate pool scoring, pruning, and exploiter selection.

Status:

- [ ] Read

My notes:

-

### 10. Population-Based Training

**Population Based Training of Neural Networks**  
Max Jaderberg et al., 2017  
Link: https://arxiv.org/abs/1711.09846

Why it matters:

- Introduces population-based hyperparameter and policy evolution.
- Useful for understanding automated training schedules.

Project connection:

- Later, we could use PBT-like ideas for entropy, learning rate, pool weights,
  curriculum thresholds, or role-specific variants.

Status:

- [ ] Read

My notes:

-

## Memory and Partial Observability

These are not immediate implementation targets. Read them when we start adding
occlusion, noisy sensors, or line-of-sight constraints.

### 11. GTrXL for RL Memory

**Stabilizing Transformers for Reinforcement Learning**  
Emilio Parisotto et al., 2019  
Link: https://arxiv.org/abs/1910.06764

Why it matters:

- Transformer-style memory for RL.
- More directly RL-relevant than generic sequence modeling papers.

Project connection:

- Relevant if obstacles create partial observability and GRU is not enough.

Status:

- [ ] Read later

My notes:

-

### 12. Transformer-XL

**Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**  
Zihang Dai et al., 2019  
Link: https://arxiv.org/abs/1901.02860

Why it matters:

- Segment-level recurrence idea.
- Good background for memory beyond fixed observation windows.

Project connection:

- Useful conceptually for long-horizon pursuit/evasion with hidden state.

Status:

- [ ] Read later

My notes:

-

### 13. Mamba

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
Albert Gu and Tri Dao, 2023  
Link: https://arxiv.org/abs/2312.00752

Why it matters:

- Modern state-space sequence modeling.
- Efficient alternative family to attention for long sequences.

Project connection:

- Research-level extension, not near-term engineering work.

Status:

- [ ] Read later

My notes:

-

## Practical Rule

For the current project stage, prioritize papers in this order:

1. MAPPO.
2. Actor-Attention-Critic.
3. Graph Attention Networks.
4. Autocurricula / league training.
5. Memory papers only after partial observability is introduced.
