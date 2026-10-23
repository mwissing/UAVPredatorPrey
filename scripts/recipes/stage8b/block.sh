#!/usr/bin/env bash
# One training block of the Stage 8b from-scratch recipe (docs/stage8b_from_scratch_protocol.md, section 3).
# Inside the container:
#   bash block.sh <out_dir> <tag> <mode> <checkpoint|none> <updates> <seed> <lr> <kl> [pool.json]
# mode: flight | joint | predator | prey. Writes <out_dir>/blocks/<tag>.status.txt and .console.log; the run directory
# is logs/skrl/uav_3v1_direct/<timestamp>_..._s8b_<tag>. Hydra overrides come before --freeze-agents, which would
# otherwise swallow them. BLOCK_EXTRA (environment, space-separated) adds Hydra overrides for ablations of the recipe.
set -uo pipefail
OUT="${1:?usage: block.sh <out_dir> <tag> <mode> <checkpoint|none> <updates> <seed> <lr> <kl> [pool.json]}"
TAG="${2:?usage}"
MODE="${3:?usage}"
CKPT="${4:?usage}"
UPDATES="${5:?usage}"
SEED="${6:?usage}"
LR="${7:?usage}"
KL="${8:?usage}"
POOL="${9:-}"
COMMON=(agent.agent.obstacle_lr_multiplier=1.0 agent.agent.discount_factor=0.995
  agent.agent.experiment.checkpoint_interval=$((UPDATES * 24)) agent.agent.experiment.experiment_name="s8b_$TAG"
  env.prey_oob_penalty=-300.0 env.predator_oob_penalty=-400.0
  +agent.models.policy.predator_min_log_std=-1.0
  +agent.models.policy.prey_mask_inactive_predators=true +agent.models.policy.predator_mask_inactive_teammates=true
  +agent.models.policy.predator_obstacle_relation=true +agent.models.value.obstacle_relation=true
  +agent.models.policy.prey_obstacle_relation=true)
case "$MODE" in
  # The flight-bootstrap task's zeros (chase, evasion, catch) plus catches switched off; crash prices, keep-out and
  # soft-wall costs, the upright term and the prey's alive bonus stay.
  flight) EXTRA=(env.predator_forced_crash_bonus=0.0 env.predator_distance_progress_reward_scale=0.0
                 env.predator_catch_bonus=0.0 env.prey_evasion_reward_scale=0.0 env.prey_caught_penalty=0.0
                 env.prey_distance_progress_reward_scale=0.0 env.catch_distance=0.0) ;;
  joint) EXTRA=(env.predator_forced_crash_bonus=200.0 env.ghost_env_fraction=0.3333 env.ghost_max_speed=3.5) ;;
  predator|prey)
    [[ -n "$POOL" ]] || { echo "mode $MODE needs a pool file" >&2; exit 2; }
    frozen=$([[ $MODE == predator ]] && echo prey || echo predator)
    EXTRA=(env.predator_forced_crash_bonus=200.0 --freeze-agents "$frozen" --per-env-opponent-pool "$POOL"
           --per-env-pool-prob 0.5 --per-env-pool-seed "$SEED") ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
START=()
[[ "$CKPT" != none ]] && START=(--checkpoint "$CKPT")
read -r -a ABLATION <<< "${BLOCK_EXTRA:-}"
mkdir -p "$OUT/blocks"
cd /workspace/UAVPredatorPrey
export TERM=xterm-256color PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf 'started=%s\n' "$(date -Iseconds)" > "$OUT/blocks/$TAG.status.txt"
env -u DISPLAY -u XAUTHORITY /workspace/isaaclab/isaaclab.sh -p scripts/skrl/train.py --headless \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-obstacles10-v0 \
  --agent skrl_mappo_attention_critic_prey_attention_large_gru_obstacles10_cfg_entry_point \
  --algorithm MAPPO --num_envs 4096 --seed "$SEED" "${START[@]}" --max_iterations "$UPDATES" \
  --learning-rate "$LR" --kl-threshold "$KL" "${COMMON[@]}" "${ABLATION[@]}" "${EXTRA[@]}" \
  > "$OUT/blocks/$TAG.console.log" 2>&1
status=$?
printf 'finished=%s\nstatus=%s\n' "$(date -Iseconds)" "$status" >> "$OUT/blocks/$TAG.status.txt"
exit "$status"
