#!/usr/bin/env bash
# Watch a checkpoint live in Isaac Sim's window: one arena, noise-free actions (as in the gates), real time, the
# elevated camera of the videos. As in the dashboard replay, each drone is drawn as a sphere larger than the drone
# (predators red, orange and pink; the prey cyan) with a fading trail of its last 2 s, over a near-white floor. Close
# the window (or Ctrl+C) to stop. The checkpoint holds both roles; compose a pair with scripts/compose_checkpoint.py.
# Further arguments go to play.py, e.g. --no-video-markers or --trail-seconds 0.
#   scripts/watch_in_isaac.sh /home/marc/RL/artifacts/isaac/experiments/<run>/<pair>.pt [play.py options]
# Needs the container isaac-lab-base-uav51 running and a desktop session (DISPLAY). The terminal reports every 5
# simulated seconds how close the window gets to real time. If the window stays empty for more than a minute, close
# it and start again: Isaac Sim sometimes hangs while it builds the scene (2 of about 15 starts on 2026-10-28).
set -euo pipefail
CONTAINER=isaac-lab-base-uav51

host=$(realpath "$1")
case "$host" in  # the container's mounts, the more specific first
  /home/marc/RL/artifacts/isaac/logs/*) ckpt=/workspace/UAVPredatorPrey/logs/${host#/home/marc/RL/artifacts/isaac/logs/} ;;
  /home/marc/RL/artifacts/*) ckpt=/workspace/artifacts/${host#/home/marc/RL/artifacts/} ;;
  /home/marc/RL/UAVPredatorPrey/*) ckpt=/workspace/UAVPredatorPrey/${host#/home/marc/RL/UAVPredatorPrey/} ;;
  *) echo "not under a path the container mounts: $host" >&2; exit 1 ;;
esac

# The X cookie the container was started with dies with the desktop session that issued it, so hand it the current
# one. The container user has the same uid (1000), so the cookie stays readable by this user only.
cookie=$(mktemp)
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$cookie" nmerge -
docker cp -a "$cookie" "$CONTAINER":/home/ubuntu/.Xauthority-watch
rm -f "$cookie"

# Real time: Kit's frame limiters add a wait to every step, and a frame of the window costs about 11 ms, so the
# window renders every 2nd policy step (25 Hz) while physics and policy run every step (50 Hz).
kit_args="--/app/runLoops/main/rateLimitEnabled=false --/app/runLoops/present/rateLimitEnabled=false"
kit_args+=" --/app/runLoops/rendering_0/rateLimitEnabled=false"

tty=()
[ -t 0 ] && tty=(-it)  # with a terminal, Ctrl+C reaches the simulation
exec docker exec "${tty[@]}" -e DISPLAY="$DISPLAY" -e XAUTHORITY=/home/ubuntu/.Xauthority-watch \
  -w /workspace/UAVPredatorPrey "$CONTAINER" /workspace/isaaclab/isaaclab.sh -p scripts/skrl/play.py \
  --task 3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-obstacles10-v0 \
  --agent skrl_mappo_attention_critic_prey_attention_large_gru_obstacles10_cfg_entry_point --algorithm MAPPO \
  --num_envs 1 --seed 42 --checkpoint "$ckpt" --real-time --camera-eye 10 -10 12.5 --camera-target 0 0 5 \
  --floor-color 0.9 0.9 0.9 --video-markers --video-marker-height 0 --video-marker-radius 0.14 --trail-seconds 2 \
  "--kit_args=$kit_args" --render-every 2 "${@:2}" \
  +agent.models.policy.prey_mask_inactive_predators=true +agent.models.policy.predator_mask_inactive_teammates=true \
  +agent.models.policy.predator_obstacle_relation=true +agent.models.value.obstacle_relation=true \
  +agent.models.policy.prey_obstacle_relation=true agent.agent.obstacle_lr_multiplier=1.0
