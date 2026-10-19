# Graft reference predator (2026-10-19)

The predator role of the graft self-play's final pair is the reference predator for the 10-pillar task with the
relation graft. The user accepted it on 2026-10-19 ("yes the chasing looks pretty good in replay"), following the
recommendation to freeze only the predator. The prey role is not a reference.

- **Read-only bundle:** `.pretrained_checkpoints/graft_predator_2026-10-19/` (gitignored). It holds README,
  MANIFEST.json, SHA256SUMS.txt, and provenance: the experiment, all evaluations, and the run's parameters.
- **Checkpoint:** `current/agent_432000.pt`, sha256 `002f1cd7...`. Evaluating the bundled copy reproduced the final
  own-matchup evaluation bit-exactly.
- **Loading needs both masks and both relation flags:**
  - `+agent.models.policy.prey_mask_inactive_predators=true`
  - `+agent.models.policy.predator_mask_inactive_teammates=true`
  - `+agent.models.policy.predator_obstacle_relation=true`
  - `+agent.models.value.obstacle_relation=true`

## How it was made

1. **The forced-crash predator stage's start** (2026-10-15, point 0): the predators from mixed self-play with
   pillars up to the wall, against the frozen waiting prey.
2. **Relation graft** (`experiments/2026-10-17_predator_relation_graft/`, arm g): zero-initialized pillar-prey
   relation branches in the actor and an identity-aware critic, 9600 predator-only updates.
3. **Continuation to 19,200 updates** (`experiments/2026-10-18_relation_graft_continuation/`, c5).
   - It broke the waiting game against the frozen prey (time-outs 0.008).
   - The matched control without the graft did not (0.047).
   - The effect replicated at seed 71.
4. **Mixed self-play from c5's pair** (`experiments/2026-10-18_relation_graft_selfplay/`): both roles train, a third
   of the envs with 3.5 m/s ghosts, seed 60, 18,000 updates. The final pair (point 5) passed the pre-registered
   reference gate.

## Evaluation

Deterministic, masked, 1024 episodes, seed 42. The fixed opponents are the obstacle-blind anchor v3 pair.

| own matchup (point 5) | value | anchor v4 |
|---|---|---|
| time-out share | 0.005 | 0.062 |
| catch / success (catch + forced crash) | 0.564 / 0.774 | 0.680 |
| predator drones lost per episode | 0.29 | 0.37 |
| own-matchup ablation ratio (points 4, 5) | 2.40, 2.83 | 1.20, 1.23 |
| long episodes: nearest predator | 2.0 m | 6.6 m |
| catch vs the fixed prey | 0.163 | 0.149 |

## Limits

- **Its training opponent was a prey without the relation branch.** That prey crashes in 42% of the episodes.
  Against a prey that can also see the pillar relations, the predator's advantage is untested; the symmetric graft
  self-play tests it.
- **One training seed** (60).
- **No coordination emerged:** pincers stay rare.
