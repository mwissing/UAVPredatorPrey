# Stage 8 review: why the predators cannot break the pillar shield (2026-10-17)

This review merges our experiments (2026-10-12 to 2026-10-16) with an external
review from a separate chat, adds diagnostics measured on the trained
networks, and checks the cited literature. It ends with the recommended next
steps.

Evidence:
- `artifacts/isaac/diagnostics/2026-10-16_stage8_blind_spots/` (README.md,
  `runs/*.json`);
- the experiment directories listed in
  [`experiment_log.md`](experiment_log.md).

## Short answer

The waiting strategy is not mainly an incentive problem, and probably not a
local optimum of a capable policy. **The networks cannot represent the
situation.** Measured on the predator of forced-crash arm f:

1. **The pillar the prey hides behind is invisible.**
   - Each drone's pillar attention is one-hot, and its query comes only from
     the drone's own state.
   - In standoffs it attends the prey's pillar in 0.4% of steps; chance is
     10%.
   - Line-of-sight and prey-pillar information is present in the input: a
     prey-conditioned probe recovers it at 0.81. After the attention it is
     gone: 0.57 / 0.03.
   - The module is degenerate for both roles. The prey's attention picks
     its nearest pillar in 1% of steps and attends the pillar it hugs in
     0.9% of hugging steps, mostly picking pillars about 5 m away.
   - The stage start shows the same values, so 9600 updates of predator
     training did not change them.
2. **The prey is visible only half the time.** The prey and the two
   teammates share one softmax, which is also one-hot. In 55% of steps the
   attention context holds a teammate instead of the prey.
3. **The critic is blind in the long game.** Explained variance of the
   returns:
   - 0.43 in short episodes;
   - -0.14 in long episodes;
   - 0.05 in standoffs.
   Its features encode agent distances (R^2 0.9) but barely the pillar
   relations (0.15-0.22). With lambda 0.95, long-game credit flows almost
   entirely through this critic, so flanking gets no usable learning
   signal.
4. **Frames are inconsistent.**
   - Relative positions are in the body frame, relative velocities in the
     world frame, and the heading is not observed.
   - The drones yaw 121 deg/s (median) and rotate more than 90 deg in 81% of
     drone-episodes.

All nine earlier interventions varied incentives or noise. None changed what
the networks can represent, how credit is assigned, or the training regime.
That was the blind spot of the investigation.

## What we concluded before, and what changes

| earlier conclusion | status now |
|---|---|
| "The waiting exploit is an incentive gap, not a missing coordination skill" (step 0, scripted pincer) | **Wrong.** The forced-crash bonus then changed nothing. The scripted ring was a naive encirclement, not the cut-off-plus-drive pincer the theory calls for. Step 0 did not test coordination. |
| "The learning signal does not seek roles; the symmetric team is a stable solution" (role vectors) | **Overread.** With zero init, interchangeable drones get identical expected gradients. OpenAI Five reports the same for zero-initialized new LSTM units, "gradient updates will never differentiate them". The test was weak by design. The imposed-role arm is the valid test. |
| "The horizon is not the bottleneck" (gamma 0.995 / 0.998) | **Weaker than stated.** Gamma changed, lambda did not: gamma x lambda went from 0.94 to 0.95. Long-horizon credit still ran through a critic that cannot see the geometry. A longer gamma also raises the value of the upright bonus (7 -> 37). |
| "A substantial missed optimum is unlikely" (scripted always-press earns the same return) | **Wrong alternative.** The relevant alternative is a coordinated press that the current networks cannot represent. "Missing capability" is the better framing. |
| Incentive and noise levers do not move the end game | **Still holds.** These are pre-registered, controlled, seed-replicated measurements. |

## Claims of the external review, checked

| # | claim | status | evidence |
|---|---|---|---|
| 1 | Pillar attention can attend "near me", not "next to the prey" | **verified, measured** | Code: query from own state only. One-hot (0.998); attends the shield pillar in 0.4% of standoff steps (chance 10%), the blocking pillar in 4.7% of blocked steps. Prey-conditioned probe 0.81 vs attended pillar 0.57 (line of sight) / 0.03 (prey-pillar R^2). |
| 1a | (found here) The prey's pillar attention is degenerate too | **measured** | The prey attends its nearest pillar in 1% of steps and the pillar it hugs in 0.9% of hugging steps; it mostly picks pillars ranked 5-9 by distance (72%). The 2026-10-10 probe already showed below-chance weight on the nearest pillar. |
| 1b | Prey and teammates share one softmax; the prey may crowd out teammates | **verified; the effect runs the other way too** | One-hot: prey > 0.9 in 45% of steps, a teammate in the other 55%. Each crowds out the other. |
| 2 | The critic cannot value the shield geometry | **verified, measured** | Code: 40 drone-pillar views mean-pooled without identity. EV: 0.43 (short), -0.14 (long), 0.05 (standoff). Linear representation probes, which do not depend on return noise: nearest-predator distance R^2 0.90-0.94, but the prey-pillar distance only 0.15-0.22 and line-of-sight blocking 0.18-0.19 (a prey-conditioned encoder reaches 0.81). A fresh value regressor trained on 770 episodes was too data-limited to serve as a control. |
| 2b | With lambda 0.95, rewards beyond about 17 steps reach the advantage only via the critic | **verified, and stricter** | (gamma lambda)^k = 0.9405^k fades by e at 16.8 steps. But GAE runs inside each 24-step rollout and bootstraps at its end, so the direct window averages about 12 steps (0.24 s) and never exceeds 24 (0.48 s), whatever lambda is. Raising lambda alone barely helps. The credit-horizon test (2026-10-17) uses 96-step rollouts with lambda 0.99. |
| 3 | The team learns as one 12-dimensional agent: one ratio, crashed drones included | verified (code) | One summed log-prob over 12 dimensions; dead drones are 14.0% of drone-steps (measured). |
| 3b | A shared policy without IDs gets at most half the optimum; IDs restore it | **partly** | Fu et al. 2022, Thm 4.2: a shared policy cannot learn the optimum of the XOR game; the proof's bound is 0.5. IDs reach the optimum only empirically (Bridge), not as a theorem. The bound assumes identical observations; our drones observe different positions, so it bites only in near-symmetric states. |
| 3c | The zero-init role test could not work | verified in principle | Equal expected gradients at the symmetric point, so separation comes from noise alone. Measured: roles stayed at noise level in both seeds. "Could not" is slightly too strong. |
| 4 | White noise at 50 Hz is filtered by the airframe; exploration needs temporal correlation | consistent with our data | Boost test: more noise gave sloppier flight (closest approach 0.89 vs 0.84 m), not more pressing. gSDE (CoRL 2021) and colored noise in PPO (AAAI 2024) are real. |
| 5 | Everything is fine-tuning of one lineage | verified | LR 1e-5; arm f KL 0.006 per update; obstacle branches grafted onto an obstacle-free model. OpenAI Five's from-scratch Rerun beat the surgery model >98% at about 20% of the compute. |
| 6 | Predators are paid to wait (upright term) | **partly** | The term exists (+0.064 per step in standoffs). Calm standoff steps net +0.016 per step, positive only because of it. Small next to a catch (200), but it grows with gamma. |
| 7 | Frames are mixed and yaw is unobserved | **verified, measured** | Code; yaw 121 deg/s median, 81% of drone-episodes rotate > 90 deg. Closing speed: MLP on the raw input R^2 0.77; the network's features are linearly no better than the raw input (0.44-0.46 vs 0.41). |
| 8 | DualCL: plain MAPPO about 0%, curriculum 70.3%, failure archive 93.4% in a one-pillar scenario | **verified** | 1-Central Tower; scripted potential-field evader, faster (2.4 vs 1.0); reward +10 capture, -1 collision. The "split into chasers and interceptors" quote is from the Empty scenario. |
| 9 | Pillar avoidance: local distance grid >> nearest-obstacle list (0.79 vs 0.10) | **verified** | Huang et al., ICRA 2024: 3x3 SDF grid; removing it drops success 0.79 -> 0.10 (cumulative ablation). |
| 10 | Hide-and-seek needed a full-state critic | **verified** | A masked value function "never progressed past stage 3" at the same compute. |
| 11 | Gavin & Bronz (ICUAS 2026): a learned evader exploited a boundary penalty, never countered | **not found** | No trace in search; do not cite. Related work does report evaders exploiting walls. |
| 12 | CoPPO: joint-ratio variance grows exponentially with the number of agents | unverified detail | The paper exists (NeurIPS 2021) and addresses variance from concurrent updates. The scaling claim was not checked. |

## Literature (checked 2026-10-16)

- Karnad, Isler. *Lion and Man Game in the Presence of a Circular Obstacle.*
  IROS 2009.
  - Characterizes win regions around a round obstacle.
  - The detailed theorems were not re-read.
  - https://www-users.cse.umn.edu/~isler/pub/iros09pe.pdf
- Fu et al. *Revisiting Some Common Practices in Cooperative MARL.* ICML 2022.
  - Thm 4.2, shared policy in the XOR game.
  - https://proceedings.mlr.press/v162/fu22d.html
- Chen et al. *A Dual Curriculum Learning Framework for Multi-UAV
  Pursuit-Evasion in Diverse Environments* (DualCL).
  - https://arxiv.org/abs/2312.12255
- Chen et al. *Online Planning for Multi-UAV Pursuit-Evasion in Unknown
  Environments Using Deep RL* (OPEN).
  - Collective thrust and body rates, evader prediction, adaptive
    environment generator.
  - https://arxiv.org/abs/2409.15866
- Huang et al. *Collision Avoidance and Navigation for a Quadrotor Swarm Using
  End-to-end Deep RL.* ICRA 2024.
  - https://arxiv.org/abs/2309.13285
- Hollenstein, Martius, Piater. *Colored Noise in PPO.* AAAI 2024.
  - https://arxiv.org/abs/2312.11091
- Raffin et al. *Smooth Exploration for Robotic Reinforcement Learning*
  (gSDE). CoRL 2021.
- Berner et al. *Dota 2 with Large Scale Deep RL* (OpenAI Five).
  - Rerun, and surgery with symmetry breaking.
  - https://arxiv.org/abs/1912.06680
- Baker et al. *Emergent Tool Use From Multi-Agent Autocurricula.* ICLR 2020.
  - https://arxiv.org/abs/1909.07528
- Dohare et al. *Loss of plasticity in deep continual learning.* Nature 2024.
- Ray, Achiam, Amodei. *Benchmarking Safe Exploration in Deep RL* (Safety
  Gym). 2019.
- Wu et al. *Coordinated Proximal Policy Optimization.* NeurIPS 2021.

## Recommendations

### A. Representation (measured defects; fix first)

1. **Prey-conditioned pillar encoding.** Each pillar's features should
   include its relation to the prey (prey-minus-pillar offset), or attention
   should use a query built from own state and prey. Better still: soft,
   multi-head self-attention over all entities (self, prey, teammates,
   pillars), as in OPEN and hide-and-seek.
   - Test: the line-of-sight and prey-pillar probes after the encoder (now
     0.56 / -0.03; the input supports 0.81 / 0.81).
2. **Prey and teammates in separate slots.** The prey gets its own,
   always-present embedding. The teammates get their own attention or
   pooling.
   - Test: prey information present in 100% of steps.
3. **Consistent frames.** All relative vectors in one frame: a
   gravity-aligned heading frame plus the observed heading (sin and cos of
   yaw), or the body frame throughout.
   - Test: a closing-speed probe from the features at the raw-MLP level
     (0.77).
4. **A critic that knows whose view each pillar is.** Per-entity pillar
   views pooled per entity, or attention over (entity, pillar) pairs.
   - Test: EV in long episodes and standoffs (now -0.14 / 0.05).

### B. Credit and reward hygiene

5. **Per-drone PPO ratios, crashed drones masked out.** The team reward
   stays shared.
6. **GAE lambda 0.95 -> 0.98**, as the direct test of long-horizon credit. It
   is a one-line A/B on the current setup and the only cheap item that needs
   no new architecture.
7. **Upright term with zero at level:** `5 (cos theta - 1) dt` instead of a
   positive stream.

### C. Exploration, roles, curriculum (after A and B)

8. Temporally correlated exploration (colored noise or gSDE).
9. The imposed-role arm (prepared: random centered role vectors, norm 10).
10. A failure-targeted scenario archive, as in DualCL and OPEN. The shield
    starts were a fixed third, not aimed at failures.

### Protocol

- Items 1-5 and 7 change observations or architecture, so checkpoints become
  incompatible.
- That argues for **one from-scratch training with the fixed architecture, 3
  seeds**. It would give a reproducible anchor and a real answer to the
  local-optimum question.
- The current-architecture reference pair, [anchor v4](anchor_v4_2026-10-17.md)
  (accepted 2026-10-17; own catch 0.680, time-outs 0.062), is the baseline to
  beat. The same evaluation suite and these blind-spot probes are the gate.
- Stage 9 (line-of-sight masking) waits for A1. Masking is about which pillar
  hides the prey, exactly the relation the current encoder cannot represent.
- Whether to bundle the thrust-plus-body-rate migration (CTBR) into the
  from-scratch run is a separate user decision. It adds risk and invalidates
  everything once more anyway.

## Open questions

- Answered: the blind spots look the same in the reference pair after
  18,000 updates of mixed self-play (anchor v4, accepted 2026-10-17):
  - shield-pillar attention 0.7% in standoffs;
  - prey visible 49%;
  - critic EV -0.09 in long episodes;
  - critic prey-pillar R^2 0.12-0.17.

  Three pairs from different stages agree, so the defects belong to the
  architecture and observation design. See the diagnostics README.
- How much does each fix contribute? Use ablations on the from-scratch
  protocol, with the attention, probe and EV gates as intermediate metrics.
- Cost: the lineage was built over months. A from-scratch recipe for the
  obstacle game does not exist yet and needs its own curriculum (flight,
  pursuit, pillars).
