# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test tensor shapes and diagnostic logs for the 3v1 obstacle MARL task."""

from __future__ import annotations

import argparse
import ast
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_ACTIONS = {"predator": 12, "prey": 4}
EXPECTED_OBSERVATIONS = {"predator": 84, "prey": 34}
EXPECTED_STATE = 118
EXPECTED_FULL_OBSERVATIONS = {"predator": 156, "prey": 58}
EXPECTED_FULL_STATE = 214
EXPECTED_1V1_SURVIVAL_OBSERVATIONS = {"predator": 18, "prey": 18}
EXPECTED_1V1_SURVIVAL_STATE = 36
EXPECTED_2V1_SURVIVAL_ACTIONS = {"predator": 8, "prey": 4}
EXPECTED_2V1_SURVIVAL_OBSERVATIONS = {"predator": 42, "prey": 24}
EXPECTED_2V1_SURVIVAL_STATE = 66
EXPECTED_LOG_KEYS = (
    "Reward/prey_cover",
    "Reward/prey_boundary_progress",
    "Reward/prey_cover_progress",
    "Reward/prey_cover_seek",
    "Reward/prey_distance_progress",
    "Reward/prey_shadow",
    "Reward/prey_shadow_progress",
    "Reward/obstacle_proximity_pred",
    "Reward/obstacle_proximity_prey",
    "Metrics/prey_cover_score",
    "Metrics/prey_cover_score_delta",
    "Metrics/prey_boundary_progress",
    "Metrics/prey_boundary_pressure",
    "Metrics/prey_distance_progress",
    "Metrics/prey_shadow_target_score",
    "Metrics/prey_shadow_score_delta",
    "Metrics/predator_progress_gate",
    "Metrics/predator_obstacle_collision_agent_fraction",
    "Metrics/prey_spawn_cover_score",
    "Metrics/obstacle_spawn_min_agent_distance",
    "Metrics/obstacle_spawn_mean_agent_distance",
    "Metrics/obstacle_spawn_min_separation",
    "Metrics/obstacle_spawn_mean_separation",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CFG_PATH = (
    REPO_ROOT
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "uav_3v1_env_cfg.py"
)
BASE_ENV_PATH = BASE_CFG_PATH.with_name("uav_3v1_env.py")
BASE_INIT_PATH = BASE_CFG_PATH.with_name("__init__.py")
BASE_MAPPO_FINETUNE_CFG_PATH = BASE_CFG_PATH.with_name("agents") / "skrl_mappo_finetune_cfg.yaml"
BASE_MAPPO_ATTENTION_CFG_PATH = BASE_CFG_PATH.with_name("agents") / "skrl_mappo_attention_cfg.yaml"
OBSTACLE_CFG_PATH = (
    REPO_ROOT
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1_obstacles"
    / "uav_3v1_obstacles_env_cfg.py"
)
OBSTACLE_ENV_PATH = OBSTACLE_CFG_PATH.with_name("uav_3v1_obstacles_env.py")
OBSTACLE_INIT_PATH = OBSTACLE_CFG_PATH.with_name("__init__.py")
FULL_OBS_MAPPO_CFG_PATH = OBSTACLE_CFG_PATH.with_name("agents") / "skrl_mappo_full_obs_cfg.yaml"
SKRL_TRAIN_PATH = REPO_ROOT / "scripts" / "skrl" / "train.py"
SKRL_EVALUATE_PATH = REPO_ROOT / "scripts" / "skrl" / "evaluate.py"
RESET_CHECKPOINT_PATH = REPO_ROOT / "scripts" / "reset_checkpoint.py"


def _parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--contract-only", action="store_true")
    known_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Check 3v1 obstacle MARL observation/action/state tensor shapes.")
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Run a plain-Python static contract check without launching Isaac Sim.",
    )

    if known_args.contract_only:
        return parser.parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Isaac Lab runtime imports are unavailable. Activate the Isaac Lab Python environment for the full "
            "runtime tensor check, or run `py scripts\\check_3v1_obstacles_shapes.py --contract-only` for the "
            "plain-Python static contract check."
        ) from None

    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="3v1-obstacles-v0", help="Name of the task.")
    parser.add_argument("--steps", type=int, default=2, help="Number of zero-action steps to verify.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.app_launcher_class = AppLauncher
    return args_cli


def _class_literal_assignments(path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            assignments = {}
            for item in node.body:
                targets = []
                value = None
                if isinstance(item, ast.Assign):
                    targets = item.targets
                    value = item.value
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    targets = [item.target]
                    value = item.value
                if value is None:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        try:
                            assignments[target.id] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            pass
            return assignments
    raise AssertionError(f"Class {class_name} not found in {path}")


def _check_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")
    print(f"[OK] {label}: {actual}", flush=True)


def _expected_contract(task: str | None = None) -> tuple[dict[str, int], int]:
    if task and "full" in task:
        return EXPECTED_FULL_OBSERVATIONS, EXPECTED_FULL_STATE
    return EXPECTED_OBSERVATIONS, EXPECTED_STATE


def _check_contract_only() -> None:
    base_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav3v1EnvCfg")
    survival_1v1_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav1v1SurvivalEasyEnvCfg")
    soft_oob_1v1_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav1v1SurvivalSoftOobEnvCfg")
    soft_oob_2v1_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav2v1SurvivalSoftOobEnvCfg")
    soft_oob_2v1_mixed_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav2v1SurvivalSoftOobMixedSpawnEnvCfg")
    soft_oob_pred22_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav1v1SurvivalSoftOobPred22EnvCfg")
    soft_oob_pred24_cfg = _class_literal_assignments(BASE_CFG_PATH, "Uav1v1SurvivalSoftOobPred24EnvCfg")
    obstacle_cfg = _class_literal_assignments(OBSTACLE_CFG_PATH, "Uav3v1ObstaclesEnvCfg")
    full_obstacle_cfg = _class_literal_assignments(OBSTACLE_CFG_PATH, "Uav3v1ObstaclesFullObsEnvCfg")
    base_env_source = BASE_ENV_PATH.read_text(encoding="utf-8")
    base_init_source = BASE_INIT_PATH.read_text(encoding="utf-8")
    base_mappo_finetune_source = BASE_MAPPO_FINETUNE_CFG_PATH.read_text(encoding="utf-8")
    base_mappo_attention_source = BASE_MAPPO_ATTENTION_CFG_PATH.read_text(encoding="utf-8")
    obstacle_cfg_source = OBSTACLE_CFG_PATH.read_text(encoding="utf-8")
    env_source = OBSTACLE_ENV_PATH.read_text(encoding="utf-8")
    init_source = OBSTACLE_INIT_PATH.read_text(encoding="utf-8")
    full_obs_mappo_source = FULL_OBS_MAPPO_CFG_PATH.read_text(encoding="utf-8")

    _check_equal("base cfg.action_spaces", base_cfg["action_spaces"], EXPECTED_ACTIONS)
    _check_equal("1v1 survival cfg.num_predators", survival_1v1_cfg["num_predators"], 1)
    _check_equal("1v1 survival cfg.action_spaces", survival_1v1_cfg["action_spaces"], {"predator": 4, "prey": 4})
    _check_equal(
        "1v1 survival cfg.observation_spaces",
        survival_1v1_cfg["observation_spaces"],
        EXPECTED_1V1_SURVIVAL_OBSERVATIONS,
    )
    _check_equal("1v1 survival cfg.state_space", survival_1v1_cfg["state_space"], EXPECTED_1V1_SURVIVAL_STATE)
    for expected in (
        "1v1-survival-easy-v0",
        "Uav1v1SurvivalEasyEnvCfg",
        "1v1-survival-soft-oob-v0",
        "Uav1v1SurvivalSoftOobEnvCfg",
        "2v1-survival-soft-oob-v0",
        "Uav2v1SurvivalSoftOobEnvCfg",
        "2v1-survival-soft-oob-mixed-spawn-v0",
        "Uav2v1SurvivalSoftOobMixedSpawnEnvCfg",
        "skrl_mappo_attention_cfg_entry_point",
        "skrl_mappo_attention_cfg.yaml",
        "1v1-survival-soft-oob-pred22-v0",
        "Uav1v1SurvivalSoftOobPred22EnvCfg",
        "1v1-survival-soft-oob-pred24-v0",
        "Uav1v1SurvivalSoftOobPred24EnvCfg",
        "skrl_mappo_finetune_cfg_entry_point",
    ):
        if expected not in base_init_source + BASE_CFG_PATH.read_text(encoding="utf-8"):
            raise AssertionError(f"Missing 1v1 survival curriculum contract: {expected}")
    _check_equal("1v1 soft OOB cfg.soft_arena_boundary", soft_oob_1v1_cfg["soft_arena_boundary"], True)
    _check_equal("1v1 soft OOB cfg.soft_arena_penalty_scale", soft_oob_1v1_cfg["soft_arena_penalty_scale"], 200.0)
    _check_equal("1v1 soft OOB cfg.boundary_penalty_scale", soft_oob_1v1_cfg["boundary_penalty_scale"], 0.0)
    _check_equal(
        "1v1 soft OOB cfg.prey_low_altitude_penalty_scale",
        soft_oob_1v1_cfg["prey_low_altitude_penalty_scale"],
        80.0,
    )
    _check_equal(
        "1v1 soft OOB cfg.prey_low_altitude_margin",
        soft_oob_1v1_cfg["prey_low_altitude_margin"],
        1.0,
    )
    _check_equal("2v1 soft OOB cfg.num_predators", soft_oob_2v1_cfg["num_predators"], 2)
    _check_equal(
        "2v1 soft OOB cfg.action_spaces",
        soft_oob_2v1_cfg["action_spaces"],
        EXPECTED_2V1_SURVIVAL_ACTIONS,
    )
    _check_equal(
        "2v1 soft OOB cfg.observation_spaces",
        soft_oob_2v1_cfg["observation_spaces"],
        EXPECTED_2V1_SURVIVAL_OBSERVATIONS,
    )
    _check_equal("2v1 soft OOB cfg.state_space", soft_oob_2v1_cfg["state_space"], EXPECTED_2V1_SURVIVAL_STATE)
    _check_equal("2v1 mixed-spawn cfg.predator_mixed_spawn", soft_oob_2v1_mixed_cfg["predator_mixed_spawn"], True)
    _check_equal(
        "2v1 mixed-spawn cfg.predator_mixed_spawn_ring_probability",
        soft_oob_2v1_mixed_cfg["predator_mixed_spawn_ring_probability"],
        0.50,
    )
    _check_equal(
        "2v1 mixed-spawn cfg.predator_mixed_spawn_wide_probability",
        soft_oob_2v1_mixed_cfg["predator_mixed_spawn_wide_probability"],
        0.25,
    )
    _check_equal(
        "1v1 soft OOB pred22 cfg.predator_thrust_to_weight",
        soft_oob_pred22_cfg["predator_thrust_to_weight"],
        2.2,
    )
    _check_equal(
        "1v1 soft OOB pred24 cfg.predator_thrust_to_weight",
        soft_oob_pred24_cfg["predator_thrust_to_weight"],
        2.4,
    )
    for expected in (
        "for _ in range(P - 1)",
        "prey_obs_parts.extend",
        "_prev_pred_prey_distances",
        "predator_distance_progress_reward",
        "Reward/predator_distance_progress",
        "Metrics/predator_distance_progress",
        "_prey_soft_arena_outside",
        "soft_arena_boundary",
        "Reward/prey_soft_arena",
        "Metrics/prey_soft_arena_outside",
        "prey_low_altitude",
        "Reward/prey_low_altitude",
        "_prev_min_pred_prey_distance",
        "Reward/prey_boundary_progress",
        "Metrics/prey_boundary_pressure",
        "_sample_predator_spawn_xy",
        "predator_mixed_spawn",
    ):
        if expected not in base_env_source:
            raise AssertionError(f"Missing generic base-env contract for 1v1 curriculum: {expected}")
    print("[OK] 1v1 survival curriculum task", flush=True)

    for expected in (
        "learning_rate: 1.0e-04",
        "learning_rate_scheduler: null",
        "entropy_loss_scale: 0.005",
        "initial_log_std: -0.5",
        "max_log_std: 0.0",
        "min_log_std: -2.0",
        "learning_epochs: 3",
    ):
        if expected not in base_mappo_finetune_source:
            raise AssertionError(f"Missing conservative self-play fine-tune config contract: {expected}")
    print("[OK] self-play fine-tune MAPPO config", flush=True)

    for expected in (
        "SharedPredatorAttentionGaussianMixin",
        "attention_min_predators: 2",
        "attention_size: 64",
        "fallback_layers: [256, 128, 64]",
    ):
        if expected not in base_mappo_attention_source:
            raise AssertionError(f"Missing shared predator attention MAPPO config contract: {expected}")
    print("[OK] shared predator attention MAPPO config", flush=True)

    _check_equal("obstacle cfg.num_obstacles", obstacle_cfg["num_obstacles"], 4)
    _check_equal("obstacle cfg.catch_distance", obstacle_cfg["catch_distance"], 0.3)
    _check_equal("obstacle cfg.observation_spaces", obstacle_cfg["observation_spaces"], EXPECTED_OBSERVATIONS)
    _check_equal("obstacle cfg.state_space", obstacle_cfg["state_space"], EXPECTED_STATE)
    _check_equal("obstacle cfg.obstacle_observation_mode", obstacle_cfg["obstacle_observation_mode"], "nearest")
    _check_equal("obstacle cfg.predator_progress_min_gate", obstacle_cfg["predator_progress_min_gate"], 0.35)
    _check_equal("obstacle cfg.prey_cover_progress_reward_scale", obstacle_cfg["prey_cover_progress_reward_scale"], 8.0)
    _check_equal("obstacle cfg.prey_shadow_progress_reward_scale", obstacle_cfg["prey_shadow_progress_reward_scale"], 4.0)
    _check_equal(
        "full obstacle cfg.observation_spaces",
        full_obstacle_cfg["observation_spaces"],
        EXPECTED_FULL_OBSERVATIONS,
    )
    _check_equal("full obstacle cfg.state_space", full_obstacle_cfg["state_space"], EXPECTED_FULL_STATE)
    _check_equal("full obstacle cfg.obstacle_observation_mode", full_obstacle_cfg["obstacle_observation_mode"], "full")

    for key in EXPECTED_LOG_KEYS:
        if key not in env_source:
            raise AssertionError(f"Missing diagnostic log key in obstacle env source: {key}")
    print(f"[OK] obstacle env diagnostic log keys: {', '.join(EXPECTED_LOG_KEYS)}", flush=True)

    for expected in ("def _compute_full_obstacle_obs", "path_block_score", "obstacle_observation_mode == \"full\""):
        if expected not in env_source:
            raise AssertionError(f"Missing full-obstacle observation contract in obstacle env source: {expected}")
    print("[OK] full-obstacle observation contract", flush=True)

    if "skrl_mappo_full_obs_cfg.yaml" not in init_source:
        raise AssertionError("Full-observation tasks should use the larger MAPPO config")
    if "layers: [512, 256, 128]" not in full_obs_mappo_source:
        raise AssertionError("Full-observation MAPPO config should use [512, 256, 128] layers")
    print("[OK] full-observation MAPPO config", flush=True)

    for expected in (
        "3v1-obstacles-bridge-v0",
        "Uav3v1ObstaclesBridgeEnvCfg",
        "3v1-obstacles-full-bridge-v0",
        "Uav3v1ObstaclesFullObsBridgeEnvCfg",
        "3v1-cover-bridge-v0",
        "Uav3v1CoverBridgeEnvCfg",
        "3v1-cover-bridge-full-v0",
        "Uav3v1CoverBridgeFullObsEnvCfg",
    ):
        if expected not in init_source + obstacle_cfg_source:
            raise AssertionError(f"Missing bridge curriculum contract: {expected}")
    print("[OK] bridge curriculum tasks", flush=True)

    for expected in (
        "3v1-survival-easy-v0",
        "Uav3v1SurvivalEasyEnvCfg",
        "3v1-survival-v0",
        "Uav3v1SurvivalEnvCfg",
        "3v1-survival-full-v0",
        "Uav3v1SurvivalFullObsEnvCfg",
        "3v1-survival-full-easy-v0",
        "Uav3v1SurvivalFullObsEasyEnvCfg",
        "predator_spawn_radius = 3.6",
        "prey_alive_bonus = 3.0",
        "prey_evasion_reward_scale = 8.0",
        "prey_distance_progress_reward_scale = 10.0",
        "prey_boundary_progress_reward_scale = 18.0",
        "prey_cover_reward_scale = 0.0",
        "prey_shadow_reward_scale = 0.0",
    ):
        if expected not in init_source + obstacle_cfg_source:
            raise AssertionError(f"Missing survival curriculum contract: {expected}")
    print("[OK] prey survival curriculum tasks", flush=True)

    for expected in (
        "pred_collision.float().mean(dim=0)",
        "predator_progress_min_gate",
        "Metrics/predator_progress_gate",
    ):
        if expected not in env_source:
            raise AssertionError(f"Missing future-oriented reward contract: {expected}")
    print("[OK] predator reward attribution/gating contract", flush=True)

    for expected in (
        "_prev_prey_cover_score",
        "_prev_prey_shadow_score",
        "prey_cover_progress_reward",
        "prey_shadow_progress_reward",
        "prey_distance_progress_reward",
        "prey_boundary_progress_reward",
        "Metrics/prey_cover_score_delta",
        "Metrics/prey_shadow_score_delta",
        "Metrics/prey_distance_progress",
        "Metrics/prey_boundary_progress",
    ):
        if expected not in env_source:
            raise AssertionError(f"Missing prey cover progress contract: {expected}")
    print("[OK] prey cover progress shaping contract", flush=True)

    if "-cover_threat * closest_cover_score * self.cfg.predator_cover_penalty_scale * dt" not in env_source:
        raise AssertionError("Predator cover penalty should subtract from predator reward when enabled")
    print("[OK] predator cover penalty sign", flush=True)

    if "use the current physics state instead of the previous observation" not in env_source:
        raise AssertionError("Obstacle dones should refresh cached agent state before catch/OOB checks")
    if "def _refresh_agent_step_state" not in env_source or "self._refresh_agent_step_state()" not in env_source:
        raise AssertionError("Obstacle dones should use a lightweight current-step state refresh")
    dones_source = env_source.split("def _get_dones", maxsplit=1)[1].split("def _reset_idx", maxsplit=1)[0]
    if "super()._get_observations()" in dones_source:
        raise AssertionError("Obstacle dones should not rebuild full observations just to refresh termination state")
    print("[OK] current-step termination refresh", flush=True)

    if "return super()._get_dones()" not in dones_source:
        raise AssertionError("Obstacle collision should remain penalty-based and delegate termination to parent")
    print("[OK] obstacle collision remains penalty-based", flush=True)

    train_source = SKRL_TRAIN_PATH.read_text(encoding="utf-8")
    evaluate_source = SKRL_EVALUATE_PATH.read_text(encoding="utf-8")
    for expected in ('parser.add_argument("--checkpoint"', 'Runner(env, agent_cfg)', "runner.agent.load(resume_path)"):
        if expected not in train_source:
            raise AssertionError(f"SKRL training script contract changed or missing expected entry point: {expected}")
    print("[OK] SKRL training entry point preserved", flush=True)

    for label, source in (("train", train_source), ("evaluate", evaluate_source)):
        for expected in ("attention_models import patch_skrl_runner", "patch_skrl_runner(Runner)"):
            if expected not in source:
                raise AssertionError(f"Missing attention model runner patch in {label}.py: {expected}")
    print("[OK] SKRL attention model runner patch", flush=True)

    for expected in (
        "class _FrozenAgentOptimizer",
        "optimizers[agent_name] = _FrozenAgentOptimizer(params)",
        "checkpoint_modules[agent_name].pop(\"optimizer\", None)",
        "optimizer state will not be saved",
    ):
        if expected not in train_source:
            raise AssertionError(f"Missing clean freeze optimizer contract: {expected}")
    print("[OK] clean freeze optimizer contract", flush=True)

    reset_checkpoint_source = RESET_CHECKPOINT_PATH.read_text(encoding="utf-8")
    for expected in (
        "--policy-log-std",
        "log_std_parameter",
        "policy[\"log_std_parameter\"].fill_",
    ):
        if expected not in reset_checkpoint_source:
            raise AssertionError(f"Missing checkpoint log-std reset contract: {expected}")
    print("[OK] checkpoint log-std reset contract", flush=True)


def _flatdim(space: Any) -> int:
    if not hasattr(space, "shape") or space.shape is None:
        raise AssertionError(f"Unsupported non-flat space for diagnostic: {space}")
    return math.prod(space.shape)


def _check_tensor(label: str, value: Any, expected_shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected_shape:
        raise AssertionError(f"{label}: expected shape {expected_shape}, got {tuple(value.shape)}")
    if value.is_floating_point() and not value.isfinite().all():
        raise AssertionError(f"{label}: contains NaN or Inf")
    print(f"[OK] {label}: shape={tuple(value.shape)}", flush=True)


def _scalar_float(label: str, value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label}: expected scalar-like value, got {value!r}") from exc


def _check_log_min(label: str, log: Mapping[str, Any], minimum: float, tolerance: float = 1.0e-2) -> None:
    if label not in log:
        raise AssertionError(f"Missing diagnostic log key: {label}")
    value = _scalar_float(label, log[label])
    if value + tolerance < minimum:
        raise AssertionError(f"{label}: expected at least {minimum}, got {value}")
    print(f"[OK] {label}: {value:.4f} >= {minimum:.4f}", flush=True)


def _check_obs(obs: Mapping[str, Any], num_envs: int, prefix: str, expected_observations: Mapping[str, int]) -> None:
    for agent, obs_dim in expected_observations.items():
        if agent not in obs:
            raise AssertionError(f"{prefix}: missing observation for agent '{agent}'")
        _check_tensor(f"{prefix} obs[{agent}]", obs[agent], (num_envs, obs_dim))


def _zero_actions(env, sample_space) -> dict[str, Any]:
    return {
        agent: sample_space(space, env.unwrapped.device, batch_size=env.unwrapped.num_envs, fill_value=0)
        for agent, space in env.unwrapped.action_spaces.items()
    }


def _check_runtime(args_cli: argparse.Namespace) -> None:
    app_launcher = args_cli.app_launcher_class(args_cli)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs.utils.spaces import sample_space
        from isaaclab_tasks.utils import parse_env_cfg

        import UAVPredatorPrey.tasks  # noqa: F401
        expected_observations, expected_state = _expected_contract(args_cli.task)

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        _check_equal("cfg.action_spaces", env_cfg.action_spaces, EXPECTED_ACTIONS)
        _check_equal("cfg.observation_spaces", env_cfg.observation_spaces, expected_observations)
        _check_equal("cfg.state_space", env_cfg.state_space, expected_state)

        env = gym.make(args_cli.task, cfg=env_cfg)
        num_envs = env.unwrapped.num_envs

        for agent, action_dim in EXPECTED_ACTIONS.items():
            _check_equal(f"gym action_space[{agent}]", _flatdim(env.unwrapped.action_spaces[agent]), action_dim)
        for agent, obs_dim in expected_observations.items():
            _check_equal(f"gym observation_space[{agent}]", _flatdim(env.unwrapped.observation_spaces[agent]), obs_dim)
        _check_equal("gym state_space", _flatdim(env.unwrapped.state_space), expected_state)

        obs, _ = env.reset()
        _check_obs(obs, num_envs, "reset", expected_observations)
        _check_tensor("reset state", env.unwrapped.state(), (num_envs, expected_state))
        latest_log = dict(env.unwrapped.extras.get("log", {}))
        _check_log_min(
            "Metrics/obstacle_spawn_min_agent_distance",
            latest_log,
            env_cfg.obstacle_agent_min_spawn_distance,
        )
        _check_log_min(
            "Metrics/obstacle_spawn_min_separation",
            latest_log,
            env_cfg.obstacle_min_separation,
        )

        if args_cli.steps > 0:
            actions = _zero_actions(env, sample_space)
            for step in range(args_cli.steps):
                obs, rewards, terminated, truncated, extras = env.step(actions)
                latest_log = dict(extras.get("log", {}))
                _check_obs(obs, num_envs, f"step {step + 1}", expected_observations)
                _check_tensor(f"step {step + 1} state", env.unwrapped.state(), (num_envs, expected_state))
                for agent in EXPECTED_ACTIONS:
                    _check_tensor(f"step {step + 1} reward[{agent}]", rewards[agent], (num_envs,))
                    _check_tensor(f"step {step + 1} terminated[{agent}]", terminated[agent], (num_envs,))
                    _check_tensor(f"step {step + 1} truncated[{agent}]", truncated[agent], (num_envs,))

        missing_logs = [key for key in EXPECTED_LOG_KEYS if key not in latest_log]
        if missing_logs:
            raise AssertionError(f"Missing diagnostic log keys: {missing_logs}")
        print(f"[OK] diagnostic log keys present: {', '.join(EXPECTED_LOG_KEYS)}", flush=True)
        _check_log_min(
            "Metrics/obstacle_spawn_min_agent_distance",
            latest_log,
            env_cfg.obstacle_agent_min_spawn_distance,
        )
        _check_log_min(
            "Metrics/obstacle_spawn_min_separation",
            latest_log,
            env_cfg.obstacle_min_separation,
        )
        print("[OK] runtime tensor smoke test completed", flush=True)
    except BaseException:
        if env is not None:
            env.close()
        raise
    else:
        if env is not None:
            env.close()
        simulation_app.close()


def main():
    args_cli = _parse_args()
    if args_cli.contract_only:
        _check_contract_only()
    else:
        _check_runtime(args_cli)


if __name__ == "__main__":
    main()
