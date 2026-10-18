# Stage 8b: from-scratch protocol (draft, 2026-10-18, for decision)

This is the follow-up to the [Stage 8 review](stage8_review_2026-10-17.md).
The review traced the pillar-shield exploit to representation and credit
defects, and recommended one from-scratch training with the fixed
architecture, 3 seeds, against [anchor v4](anchor_v4_2026-10-17.md). This
draft lays out what that would mean. Nothing here has run.

## Why from scratch

- **Lineage.** Every network since anchor v3 is a fine-tune at LR 1e-5 with
  zero-initialized grafts:
  - arm f: KL 0.006 per update, clip fraction 1.9%, action noise at its
    floor;
  - plasticity loss under repeated environment changes is documented
    (Dohare et al. 2024);
  - OpenAI Five's from-scratch Rerun beat its surgery model >98% at about
    20% of the compute.
- **The observation frame.** Body-frame observations change what every
  input means. Old weights cannot be reused.
- **Reproducibility.** There is no from-scratch recipe for the obstacle
  game. Anchor v4 cannot be reproduced from random initialization.
- **The local-optimum question.** Only independent starts can answer it.

## Configuration (all options exist; default off)

| fix (review item) | option | code | evidence so far |
|---|---|---|---|
| prey-conditioned pillar relations (A1, A2) | `+agent.models.policy.predator_obstacle_relation=true` | eace487, 171fd3b | graft (2026-10-18): line of sight in the encoder 0.74 (from 0.55), but not in the pretrained GRU output (0.56) |
| identity-aware critic (A4) | `+agent.models.value.obstacle_relation=true` | eace487, 171fd3b | graft: prey-pillar R^2 0.93 (from 0.08-0.15); critic EV in long episodes 0.20 (from -0.14) |
| consistent frames (A3) | `env.observation_frame=body` | 52dfd8d (branch stage8b-fixes) | live: matches Isaac Lab's quat_apply_inverse to 1e-6 |
| per-drone PPO ratios, crashed drones masked (B5) | `+agent.agent.per_drone_ratio=true` | 3c9a71f (branch) | smoke: trains, KL 0.003-0.014 |
| upright term zero at level (B7) | `env.predator_upright_zero_at_level=true` | 5dd0e15 (branch) | unit tests |
| long-horizon credit (B6) | `agent.agent.rollouts=96 agent.agent.lambda=0.99 agent.agent.mini_batches=8` | config only | no effect on the long game; fewer crashes mid-run |

Open items before a from-scratch run:
- Done (92af76c, branch): the prey's counterpart,
  `+agent.models.policy.prey_obstacle_relation=true`. Its pillar attention
  is equally degenerate (it attends the pillar it hugs in 0.2-1% of those
  steps), so the same tokens are built relative to the nearest live
  predator.
- **The old one-hot pillar attention stays in both actors.** From scratch,
  it could be replaced by the relation pooling; there is no option yet.
- **Correlated exploration (C8, colored noise or gSDE)** is not
  implemented.

The graft result strengthens the case for training from scratch. The new
features reach the encoder and the critic, but the pretrained recurrent core
of the policy, at LR 1e-5, does not route them to the actions within 9600
updates. The continuation (2026-10-18) tests whether more time is enough.

## Curriculum (each phase pre-registered with gates)

It follows `curriculum_learning_plan.md`: one missing capability at a time,
promotion by gates.

1. **Flight in the pillar arena** (both roles, short episodes).
   - Gates: out-of-bounds <= 0.10, pillar crash hazard at the obstacle
     reference's level, stable height.
2. **Role skills against scripted opponents.**
   - Predators against a scripted evader; prey against the scripted ghost
     pursuers (`ghost_mode`).
   - Gates: catch rate or survival thresholds, both roles' obstacle ablation
     ratio >= 1.2.
3. **Mixed self-play with rehearsal** (the anchor v3 pool settings), the
   reference recipe.
   - Gates: anchor v4's gate (perception, balance, no forgetting,
     stability), plus:
     - time-outs <= anchor v4's 0.062;
     - the blind-spot probes: line of sight decodable >= 0.70, critic EV in
       long episodes > 0.2, prey visible in 100% of steps.
   - Head-to-head and league entries against anchor v4.
4. **Three seeds** for phase 3 at least. Single-run differences of 0.01-0.02
   are noise (f vs x).

## Cost (rough)

At about 30 it/s with 4096 envs:
- phase 1: a few hours;
- phase 2: about 2 x 3 h;
- phase 3: about 8 h per seed.

That is roughly a day per seed, 2-3 days for three seeds, plus evaluations.
The anchor lineage took far longer, but it explored. This protocol reuses
known recipes.

## Decisions for the user

1. **When:** now, or after the graft evidence from the night chain. If the
   graft breaks the waiting game, the self-play with the graft already gives
   a candidate. From scratch is still the reproducible version.
2. **Bundle CTBR** (thrust plus body rates, the deployment migration)?
   - For: one retraining instead of two.
   - Against: one more change in the same run, which goes against "one new
     difficulty per stage".
   - Recommendation: not in the first from-scratch run.
3. **Scope:** replacing the old pillar attention and correlated exploration
   can wait for the first from-scratch result. The prey branch is ready.
