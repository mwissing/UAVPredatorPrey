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

## Closest UAV Pursuit-Evasion Reference

### 14. OPEN / Multi-UAV Pursuit-Evasion with Online Planning

**Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning**  
Jiayu Chen, Chao Yu, Guosheng Li, Wenhao Tang, Xinyi Yang, Botian Xu, Huazhong Yang, Yu Wang, 2024  
Link: https://arxiv.org/abs/2409.15866  
Project page: https://sites.google.com/view/pursuit-evasion-rl

Why it matters:

- This is the closest current paper to the long-term UAV predator-prey goal:
  3D multi-UAV pursuit, physical dynamics, obstacles, partial observability,
  online planning, and real quadrotor deployment.
- Uses MAPPO with collective thrust/body-rate commands, an attention-based
  observation encoder, an LSTM evader-prediction network, adaptive environment
  generation, and two-stage reward refinement.
- Gives a concrete reference point for what "domain-level SOTA" looks like
  beyond a strong full-state MAPPO baseline.

Project connection:

- Similar to our direction: MAPPO, continuous UAV control, attention over
  entities/observations, multi-UAV capture, and eventual obstacle/occlusion
  handling.
- Different from our current strength: the paper emphasizes evader prediction,
  hard-scenario generation, calibrated dynamics, and sim-to-real deployment;
  our stack emphasizes self-play, opponent pools, cross-play robustness, and
  co-adapting predator/prey policies.
- Most useful near-term lesson: before adding visual models or MPC, add
  measurable prey-prediction / belief-state diagnostics and hard-scenario
  curriculum tests once partial observability is introduced.

Status:

- [ ] Read later

My notes:

-

## Late-Stage MPC / Model-Based Control

These are not immediate implementation targets. Read them when the direct
MAPPO/self-play stack has a strong baseline and we want to test whether
model-based local control improves transfer, safety, or action smoothness.

### 15. MA-AC-MPC Actor Interface

**Merging model-based control with multi-agent reinforcement learning for multi-agent cooperative teaming strategies**  
Christian Llanes, Spencer W. Jensen, Samuel Coogan, 2026  
Link: https://arxiv.org/abs/2606.06011

Why it matters:

- Shows how a neural actor can output MPC cost/reference parameters instead of
  directly outputting low-level actions.
- Keeps MAPPO-style actor-critic training while adding a differentiable MPC
  layer for dynamically feasible actions.
- Provides a useful comparison point for an eventual
  `RL chooses local objective -> MPC executes` architecture.

Project connection:

- Relevant as a late-stage actor-head experiment, not as a replacement for the
  current self-play stack.
- The direct comparison for our project would be:
  - current Gaussian body-rate/thrust actor,
  - same self-play/curriculum/evaluation stack with an MPC-structured action
    head.
- Especially relevant if Crazyflow/JAX becomes a fast backend for testing
  structured local controllers before Isaac Lab fine-tuning.

Status:

- [ ] Read later

My notes:

-

## Practical Rule

For the current project stage, prioritize papers in this order:

1. MAPPO.
2. Actor-Attention-Critic.
3. Graph Attention Networks.
4. OPEN / Multi-UAV Pursuit-Evasion once comparing against domain-level UAV
   pursuit-evasion SOTA.
5. Autocurricula / league training.
6. Memory papers only after partial observability is introduced.
7. MPC/model-based control only after a strong direct-action baseline exists.
