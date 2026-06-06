# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministically evaluate a SKRL checkpoint and summarize episode metrics."""

"""Launch Isaac Sim Simulator first."""
import h5py  # noqa: F401
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--episodes", type=int, default=256, help="Number of completed vectorized episodes to collect.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Maximum vectorized environment steps before stopping. Defaults to a safe episode-based estimate.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--json", type=str, default=None, help="Optional path to write the evaluation summary as JSON.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import UAVPredatorPrey.tasks  # noqa: F401


PREFERRED_EPISODE_KEYS = (
    "Metrics/catch_rate",
    "Metrics/predator_oob_rate",
    "Metrics/prey_oob_rate",
    "Metrics/episode_length",
    "Metrics/episode_closest_approach",
    "Metrics/final_distance",
    "Metrics/obstacle_collision_rate",
    "Metrics/obstacle_collision_steps",
    "Metrics/prey_cover_rate",
    "Metrics/prey_cover_when_threatened",
    "Metrics/prey_cover_score_episode",
    "Metrics/prey_active_cover_rate",
    "Metrics/prey_active_cover_score",
    "Metrics/prey_initial_cover_score",
    "Metrics/prey_cover_gain_episode",
    "Metrics/prey_shadow_score_episode",
)


class WeightedMeans:
    """Accumulate scalar means with explicit weights."""

    def __init__(self) -> None:
        self._weighted_sum = defaultdict(float)
        self._weight = defaultdict(float)

    def add(self, key: str, value: float, weight: float = 1.0) -> None:
        if weight <= 0.0 or not math.isfinite(value):
            return
        self._weighted_sum[key] += value * weight
        self._weight[key] += weight

    def as_dict(self) -> dict[str, float]:
        return {
            key: self._weighted_sum[key] / self._weight[key]
            for key in sorted(self._weighted_sum)
            if self._weight[key] > 0.0
        }


def _scalar_float(value: Any) -> float | None:
    """Convert tensor/scalar-like log values to Python floats."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and hasattr(value, "mean"):
        value = value.float().mean()
    elif hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _done_mask(done_like: Any) -> torch.Tensor:
    if isinstance(done_like, dict):
        values = list(done_like.values())
        if not values:
            raise RuntimeError("Received an empty done dictionary from the environment.")
        done = values[0].bool()
        for value in values[1:]:
            done = done | value.bool()
        return done
    return done_like.bool()


def _deterministic_actions(env, outputs):
    if hasattr(env, "possible_agents"):
        return {agent: outputs[-1][agent].get("mean_actions", outputs[0][agent]) for agent in env.possible_agents}
    return outputs[-1].get("mean_actions", outputs[0])


def _collect_logs(log: dict[str, Any], stats: WeightedMeans, *, weight: float, prefixes: tuple[str, ...]) -> None:
    for key, value in log.items():
        if not key.startswith(prefixes):
            continue
        scalar = _scalar_float(value)
        if scalar is not None:
            stats.add(key, scalar, weight)


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n[INFO] Evaluation summary")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"task: {summary['task']}")
    print(f"episodes_completed: {summary['episodes_completed']}")
    print(f"vector_steps: {summary['vector_steps']}")

    episode_metrics = summary["episode_metrics"]
    if episode_metrics:
        print("\nEpisode metrics:")
        printed = set()
        for key in PREFERRED_EPISODE_KEYS:
            if key in episode_metrics:
                print(f"  {key}: {episode_metrics[key]:.6g}")
                printed.add(key)
        for key in sorted(set(episode_metrics) - printed):
            print(f"  {key}: {episode_metrics[key]:.6g}")

    step_rewards = summary["step_rewards"]
    if step_rewards:
        print("\nStep reward/log means:")
        for key, value in step_rewards.items():
            print(f"  {key}: {value:.6g}")


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Evaluate with a deterministic SKRL policy."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    num_envs = env.unwrapped.num_envs
    max_episode_length = getattr(env.unwrapped, "max_episode_length", 1)
    max_steps = args_cli.max_steps
    if max_steps is None:
        max_steps = max_episode_length * (math.ceil(args_cli.episodes / max(num_envs, 1)) + 2)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    episode_stats = WeightedMeans()
    step_reward_stats = WeightedMeans()
    completed_episodes = 0
    vector_steps = 0
    obs, _ = env.reset()

    try:
        while simulation_app.is_running() and completed_episodes < args_cli.episodes and vector_steps < max_steps:
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=vector_steps, timesteps=max_steps)
                actions = _deterministic_actions(env, outputs)
                obs, _, terminated, truncated, extras = env.step(actions)

            vector_steps += 1
            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            _collect_logs(log, step_reward_stats, weight=1.0, prefixes=("Reward/",))

            done = _done_mask(terminated) | _done_mask(truncated)
            done_count = int(done.sum().item())
            if done_count <= 0:
                continue

            remaining = args_cli.episodes - completed_episodes
            weight = float(min(done_count, remaining))
            _collect_logs(log, episode_stats, weight=weight, prefixes=("Metrics/",))
            completed_episodes += done_count
    finally:
        env.close()

    summary = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "episodes_requested": args_cli.episodes,
        "episodes_completed": min(completed_episodes, args_cli.episodes),
        "vector_steps": vector_steps,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "episode_metrics": episode_stats.as_dict(),
        "step_rewards": step_reward_stats.as_dict(),
    }
    _print_summary(summary)

    if args_cli.json:
        output_path = Path(args_cli.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote JSON summary to: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

