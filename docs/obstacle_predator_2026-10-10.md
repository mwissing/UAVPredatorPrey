# Obstacle reference predator (2026-10-10)

The predator role of this checkpoint is the reference predator for the
pillar task (roadmap Stage 8). It is the first predator that avoids pillars
by perception. Removing its obstacle branch raises its crash hazard by about
40-50%. It does this while intercepting the prey and enclosing it at the
wall. The prey role in the same checkpoint is **not** a reference: that prey
lost its perception-based avoidance in self-play.

- Read-only bundle: `.pretrained_checkpoints/obstacle_predator_2026-10-10/`
  (gitignored; README, MANIFEST.json, SHA256SUMS.txt, provenance, and
  evaluation evidence).
- Checkpoint: `current/agent_14400.pt`, sha256 `470c11e6...`. Evaluating the
  bundled copy reproduced the final own-matchup evaluation bit-exactly.
- Task `3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-obstacles-v0`,
  agent `skrl_mappo_attention_critic_prey_attention_large_gru_obstacles_cfg_entry_point`.
  Overrides: `env.obstacle_placement_radius=4.6` (pillars up to the wall),
  `env.predator_oob_penalty=-600.0`, `+agent.models.policy.predator_min_log_std=-1.0`,
  and `env.prey_oob_penalty=-300.0`. Code `b250977`.

## How it was made

1. **Obstacles.** Anchor v3 was extended with zero-initialized obstacle
   branches, then trained in joint runs on 2026-10-08.
2. **Crash price plus exploration** (2026-10-08 evening, pre-registered). A
   crash cost of -600 per drone, so one lost drone costs the team one catch,
   and a log-std floor of -1.0 taught the predator perception-based
   avoidance. A higher price alone did not, even at 10x.
3. **Overnight self-play** (2026-10-09/10, pre-registered). Starting from
   v1's ghost-trained prey, in the layout with pillars up to the wall: 6
   cycles, each 2400 joint updates plus 600 prey-only ghost updates.

## Measured (deterministic, 1024 episodes, seed 42, pillars up to the wall)

| | start of the overnight run | reference |
|---|---|---|
| crash hazard vs the obstacle-blind start prey | 0.173/s | 0.100/s |
| ablation ratio vs the start prey (zeroed / with) | 1.08 | 1.41 |
| own matchup: predator crash hazard | 0.173/s | 0.085/s |
| own matchup: ablation ratio | | 1.52 |
| own matchup: catch rate | 0.162 | 0.486 |
| lead index (interception) | 0.064 | 0.267 |
| wall enclosure in the last second before catches | 0.102 | 0.165 |

Evidence: `artifacts/isaac/experiments/2026-10-08_obstacles_crash_price_exploration/`
and `artifacts/isaac/experiments/2026-10-09_overnight_selfplay_ghost_rehearsal/`
(RESULTS.md, summary.txt, videos/final_camera_D/).
