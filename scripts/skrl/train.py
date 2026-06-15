# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""
import h5py
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--freeze-agents",
    nargs="+",
    default=[],
    metavar="AGENT",
    help="Freeze one or more MARL agents during training, e.g. --freeze-agents predator prey.",
)
parser.add_argument(
    "--freeze-predator",
    action="store_true",
    default=False,
    help="Deprecated shortcut for --freeze-agents predator.",
)
parser.add_argument(
    "--per-env-opponent-pool",
    type=str,
    default=None,
    help=(
        "Optional opponent pool JSON used to mix frozen opponent policies across vectorized envs. "
        "Only active when exactly one agent is frozen."
    ),
)
parser.add_argument(
    "--per-env-pool-prob",
    type=float,
    default=0.0,
    help="Fraction of envs assigned to pool opponents instead of the latest frozen opponent.",
)
parser.add_argument(
    "--per-env-pool-max-policies",
    type=int,
    default=8,
    help="Maximum number of pool policies loaded for the frozen opponent role.",
)
parser.add_argument(
    "--per-env-pool-seed",
    type=int,
    default=None,
    help="Seed for per-env opponent assignment. Defaults to --seed.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import UAVPredatorPrey.tasks  # noqa: F401

if args_cli.ml_framework.startswith("torch"):
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.agents.attention_models import patch_skrl_runner

    patch_skrl_runner(Runner)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


class _FrozenAgentOptimizer(torch.optim.Optimizer):
    """No-op optimizer used to keep frozen SKRL agents out of Adam state updates."""

    def __init__(self, params):
        super().__init__(list(params), defaults={})

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                return closure()
        return None

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for param in group["params"]:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()


def _agent_parameters(agent, agent_name: str) -> list[torch.nn.Parameter]:
    """Return unique policy/value parameters for one SKRL multi-agent uid."""
    modules = []
    for attr_name in ("policies", "values"):
        mapping = getattr(agent, attr_name, {})
        if isinstance(mapping, dict):
            module = mapping.get(agent_name)
            if module is not None and module not in modules:
                modules.append(module)

    if not modules:
        models = getattr(agent, "models", {})
        agent_models = models.get(agent_name, {}) if isinstance(models, dict) else {}
        if isinstance(agent_models, dict):
            for module in agent_models.values():
                if module is not None and module not in modules:
                    modules.append(module)

    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def _freeze_agent_training(agent, agent_names: set[str]) -> None:
    """Freeze selected SKRL multi-agent policies without breaking the shared backward pass."""
    if not agent_names:
        return

    optimizers = getattr(agent, "optimizers", {})
    schedulers = getattr(agent, "schedulers", {})
    lr_schedulers_enabled = getattr(agent, "_learning_rate_scheduler", {})
    available_agents = set(optimizers.keys()) if isinstance(optimizers, dict) else set()

    for agent_name in sorted(agent_names):
        if agent_name not in available_agents:
            print(
                f"[WARNING] Cannot freeze agent '{agent_name}': optimizer not found. "
                f"Available optimizer keys: {sorted(available_agents)}"
            )
            continue

        params = _agent_parameters(agent, agent_name)
        if not params:
            print(f"[WARNING] Cannot freeze agent '{agent_name}': no policy/value parameters found.")
            continue

        for param in params:
            param.grad = None

        old_optimizer = optimizers[agent_name]
        old_optimizer.state.clear()
        optimizers[agent_name] = _FrozenAgentOptimizer(params)
        checkpoint_modules = getattr(agent, "checkpoint_modules", {})
        if isinstance(checkpoint_modules, dict) and agent_name in checkpoint_modules:
            checkpoint_modules[agent_name].pop("optimizer", None)
        print(
            f"[INFO] Agent '{agent_name}' optimizer replaced by no-op freeze optimizer; "
            "optimizer state will not be saved."
        )

        if isinstance(lr_schedulers_enabled, dict) and agent_name in lr_schedulers_enabled:
            lr_schedulers_enabled[agent_name] = None
            print(f"[INFO] Agent '{agent_name}' learning rate scheduler disabled.")
        elif isinstance(schedulers, dict) and agent_name in schedulers:
            print(f"[INFO] Agent '{agent_name}' scheduler left unused because learning rate is 0.0.")


def _load_pool_entries(path: str | None, role: str, max_policies: int) -> list[dict[str, Any]]:
    if path is None:
        return []

    pool_path = Path(path).expanduser().resolve()
    if not pool_path.exists():
        raise FileNotFoundError(f"Per-env opponent pool not found: {pool_path}")

    data = json.loads(pool_path.read_text(encoding="utf-8-sig"))
    raw_entries = data.get(role, [])
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        if isinstance(raw_entry, str):
            entry = {
                "name": Path(raw_entry).stem,
                "checkpoint": raw_entry,
                "weight": 1.0,
            }
        elif isinstance(raw_entry, dict):
            if "checkpoint" not in raw_entry:
                raise RuntimeError(f"Pool entry {role}[{index}] is missing 'checkpoint'.")
            entry = dict(raw_entry)
            entry.setdefault("name", Path(str(entry["checkpoint"])).stem)
            entry.setdefault("weight", 1.0)
        else:
            raise RuntimeError(f"Pool entry {role}[{index}] must be a string or object.")

        checkpoint = Path(str(entry["checkpoint"])).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Pool entry '{entry['name']}' checkpoint does not exist: {checkpoint}")

        entry["checkpoint"] = str(checkpoint)
        entry["weight"] = float(entry.get("weight", 1.0))
        if entry["weight"] > 0.0:
            entries.append(entry)

    entries.sort(key=lambda item: float(item.get("weight", 1.0)), reverse=True)
    if max_policies > 0:
        entries = entries[:max_policies]
    return entries


def _load_module_state(module: Any, state: Any, label: str) -> None:
    if state is None or module is None:
        return
    if hasattr(module, "load_state_dict"):
        try:
            module.load_state_dict(state)
        except Exception as exc:
            print(f"[WARNING] Could not load {label} state: {exc}")


class _PoolPolicy:
    """Frozen policy copy used only for action generation."""

    def __init__(self, *, name: str, role: str, policy: torch.nn.Module, state_preprocessor: Any | None):
        self.name = name
        self.role = role
        self.policy = policy
        self.state_preprocessor = state_preprocessor
        self.policy.eval()
        if hasattr(self.state_preprocessor, "eval"):
            self.state_preprocessor.eval()

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        states = observations
        if self.state_preprocessor is not None:
            states = self.state_preprocessor(states)
        actions, _, _ = self.policy.act({"states": states}, role="policy")
        return actions


def _build_pool_policies(agent: Any, role: str, entries: list[dict[str, Any]], device: torch.device) -> list[_PoolPolicy]:
    if not entries:
        return []

    policies = getattr(agent, "policies", {})
    state_preprocessors = getattr(agent, "_state_preprocessor", {})
    template_policy = policies.get(role) if isinstance(policies, dict) else None
    template_preprocessor = state_preprocessors.get(role) if isinstance(state_preprocessors, dict) else None
    if template_policy is None:
        raise RuntimeError(f"Cannot build per-env pool: no policy found for frozen role '{role}'.")

    pool_policies: list[_PoolPolicy] = []
    for entry in entries:
        checkpoint_path = Path(str(entry["checkpoint"]))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        role_state = checkpoint.get(role)
        if not isinstance(role_state, dict) or "policy" not in role_state:
            raise RuntimeError(f"Checkpoint does not contain role '{role}' policy: {checkpoint_path}")

        policy = copy.deepcopy(template_policy)
        policy.to(device)
        policy.load_state_dict(role_state["policy"])

        preprocessor = copy.deepcopy(template_preprocessor) if template_preprocessor is not None else None
        _load_module_state(preprocessor, role_state.get("state_preprocessor"), f"{role} state_preprocessor")
        pool_policies.append(
            _PoolPolicy(
                name=str(entry.get("name", checkpoint_path.stem)),
                role=role,
                policy=policy,
                state_preprocessor=preprocessor,
            )
        )

    return pool_policies


class _PerEnvOpponentPoolWrapper:
    """Delegate env wrapper that mixes frozen opponent policies across vectorized envs."""

    def __init__(
        self,
        env: Any,
        *,
        frozen_role: str,
        pool_policies: list[_PoolPolicy],
        pool_weights: list[float],
        pool_prob: float,
        seed: int | None,
    ):
        self._env = env
        self.frozen_role = frozen_role
        self.pool_policies = pool_policies
        self.pool_prob = max(0.0, min(1.0, float(pool_prob)))
        self._last_observations: Mapping[str, torch.Tensor] | None = None
        self._assignment: torch.Tensor | None = None
        self._logged_assignment = False

        device = getattr(env, "device", torch.device("cpu"))
        self._device = torch.device(device)
        pool_weight_tensor = torch.as_tensor(pool_weights, dtype=torch.float32)
        pool_weight_tensor = torch.clamp(pool_weight_tensor, min=0.0)
        if pool_weight_tensor.numel() == 0 or float(pool_weight_tensor.sum().item()) <= 0.0:
            raise RuntimeError("Per-env opponent pool requires at least one positive pool weight.")
        pool_weight_tensor = pool_weight_tensor / pool_weight_tensor.sum()

        latest_weight = torch.tensor([1.0 - self.pool_prob], dtype=torch.float32)
        pool_weights_scaled = pool_weight_tensor * self.pool_prob
        self._source_probs = torch.cat((latest_weight, pool_weights_scaled), dim=0)

        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample_assignment(self) -> None:
        num_envs = int(self._env.num_envs)
        self._assignment = torch.multinomial(
            self._source_probs,
            num_samples=num_envs,
            replacement=True,
            generator=self._generator,
        ).to(self._device)

        if not self._logged_assignment:
            counts = torch.bincount(self._assignment.cpu(), minlength=len(self._source_probs))
            latest_count = int(counts[0].item())
            pool_count = int(counts[1:].sum().item())
            print(
                f"[INFO] Per-env opponent pool active for frozen '{self.frozen_role}': "
                f"latest_envs={latest_count}, pool_envs={pool_count}, "
                f"pool_prob={self.pool_prob:.3f}, policies={len(self.pool_policies)}"
            )
            for index, policy in enumerate(self.pool_policies, start=1):
                print(f"[INFO]   source {index}: {policy.name}, envs={int(counts[index].item())}")
            self._logged_assignment = True

    def reset(self):
        observations, infos = self._env.reset()
        self._last_observations = observations
        self._sample_assignment()
        return observations, infos

    def step(self, actions: Mapping[str, torch.Tensor]):
        if self._last_observations is None:
            raise RuntimeError("Per-env opponent pool wrapper received step before reset.")
        if self.frozen_role not in actions:
            raise RuntimeError(f"Frozen role '{self.frozen_role}' not present in action dict.")
        if self.frozen_role not in self._last_observations:
            raise RuntimeError(f"Frozen role '{self.frozen_role}' not present in observation dict.")

        mixed_actions = dict(actions)
        frozen_actions = actions[self.frozen_role].clone()
        frozen_observations = self._last_observations[self.frozen_role]

        for source_index, policy in enumerate(self.pool_policies, start=1):
            mask = self._assignment == source_index
            if torch.any(mask):
                pool_actions = policy.act(frozen_observations)
                frozen_actions[mask] = pool_actions[mask]
        mixed_actions[self.frozen_role] = frozen_actions

        next_observations, rewards, terminated, truncated, infos = self._env.step(mixed_actions)
        self._last_observations = next_observations

        if isinstance(infos, dict):
            episode_info = infos.setdefault("episode", {})
            if isinstance(episode_info, dict):
                pool_fraction = torch.mean((self._assignment > 0).float()).to(self._device)
                episode_info[f"Pool/{self.frozen_role}_per_env_pool_fraction"] = pool_fraction

        return next_observations, rewards, terminated, truncated, infos


def _wrap_per_env_opponent_pool(env: Any, runner: Runner, freeze_agents: set[str]) -> Any:
    pool_prob = max(0.0, min(1.0, float(args_cli.per_env_pool_prob)))
    if args_cli.per_env_opponent_pool is None or pool_prob <= 0.0:
        return env
    if len(freeze_agents) != 1:
        print(
            "[WARNING] Per-env opponent pool is only active when exactly one agent is frozen. "
            f"Got frozen agents: {sorted(freeze_agents)}. Disabling per-env pool."
        )
        return env

    frozen_role = next(iter(freeze_agents))
    entries = _load_pool_entries(args_cli.per_env_opponent_pool, frozen_role, int(args_cli.per_env_pool_max_policies))
    if not entries:
        print(f"[WARNING] Per-env opponent pool has no entries for frozen role '{frozen_role}'. Disabling.")
        return env

    seed = args_cli.per_env_pool_seed if args_cli.per_env_pool_seed is not None else args_cli.seed
    pool_weights = [float(entry.get("weight", 1.0)) for entry in entries]
    pool_policies = _build_pool_policies(runner.agent, frozen_role, entries, env.device)
    return _PerEnvOpponentPoolWrapper(
        env,
        frozen_role=frozen_role,
        pool_policies=pool_policies,
        pool_weights=pool_weights,
        pool_prob=pool_prob,
        seed=seed,
    )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    freeze_agents = set(args_cli.freeze_agents)
    if args_cli.freeze_predator:
        print("[WARNING] --freeze-predator is deprecated. Use --freeze-agents predator instead.")
        freeze_agents.add("predator")
    if freeze_agents:
        print(f"[INFO] Freezing agents during training: {', '.join(sorted(freeze_agents))}")
        _freeze_agent_training(runner.agent, freeze_agents)

    runner._trainer.env = _wrap_per_env_opponent_pool(env, runner, freeze_agents)

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
