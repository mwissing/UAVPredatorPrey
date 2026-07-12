"""Export finite deterministic Isaac reward fixtures for the JAX reward port.

The script writes root states and evaluates the existing target task without a
physics step or policy. It records the actual Isaac readback, executed actions,
reward components, alive/event masks, and progress-memory mutation. Non-finite
PhysX states are deliberately outside this oracle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_OUTPUT = "/workspace/artifacts/transfer/reward_v0/isaac_oracle_v1.json"
TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"
ENV_ORIGIN_W_M = (12.0, -7.0, 1.0)
IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
UPSIDE_DOWN_ROLL_QUATERNION_WXYZ = (0.0, 1.0, 0.0, 0.0)
EPISODE_STEP_SEEN_BY_DONES = 100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Destination JSON path.")
    parser.add_argument("--repeats", type=int, default=3, help="Deterministic repeats per case.")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env import Uav3v1Env
from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env_cfg import (
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg,
)


PREDATOR_TERM_LOGS = {
    "upright": "Reward/predator_upright",
    "height": "Reward/predator_height",
    "lin_vel": "Reward/predator_lin_vel",
    "ang_vel": "Reward/predator_ang_vel",
    "action": "Reward/predator_action",
    "proximity": "Reward/predator_proximity",
    "distance_progress": "Reward/predator_distance_progress",
    "catch": "Reward/predator_catch",
    "assist": "Reward/predator_assist",
    "boundary": "Reward/predator_boundary",
    "oob": "Reward/predator_oob",
    "soft_arena": "Reward/predator_soft_arena",
}
PREY_TERM_LOGS = {
    "upright": "Reward/prey_upright",
    "height": "Reward/prey_height",
    "low_altitude": "Reward/prey_low_altitude",
    "lin_vel": "Reward/prey_lin_vel",
    "ang_vel": "Reward/prey_ang_vel",
    "action": "Reward/prey_action",
    "alive": "Reward/prey_alive",
    "evasion": "Reward/prey_evasion",
    "distance_progress": "Reward/prey_distance_progress",
    "boundary_progress": "Reward/prey_boundary_progress",
    "caught": "Reward/prey_caught",
    "boundary": "Reward/prey_boundary",
    "oob": "Reward/prey_oob",
    "soft_arena": "Reward/prey_soft_arena",
}


def _git_metadata(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _tensor_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _scalar_bool(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().item())


def _root_state(
    position_local_m: tuple[float, float, float],
    *,
    quaternion_wxyz: tuple[float, float, float, float] = IDENTITY_QUATERNION_WXYZ,
    linear_velocity_w_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity_w_radps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, list[float]]:
    return {
        "root_link_pos_local_m": list(position_local_m),
        "root_link_quat_wb_wxyz": list(quaternion_wxyz),
        "root_com_lin_vel_w_mps": list(linear_velocity_w_mps),
        "root_ang_vel_w_radps": list(angular_velocity_w_radps),
    }


def _zero_actions() -> dict[str, list[Any]]:
    return {
        "predator": [[0.0, 0.0, 0.0, 0.0] for _ in range(3)],
        "prey": [0.0, 0.0, 0.0, 0.0],
    }


def _request(
    root_states: list[dict[str, list[float]]],
    *,
    purpose: str,
    predator_alive_before: tuple[bool, bool, bool] = (True, True, True),
    raw_actions: dict[str, list[Any]] | None = None,
    previous_predator_minus_current_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    previous_minimum_minus_current_m: float = 0.0,
    previous_prey_radius_minus_current_m: float = 0.0,
) -> dict[str, Any]:
    if len(root_states) != 4:
        raise ValueError("root_states must contain predator 0/1/2 followed by prey")
    return {
        "purpose": purpose,
        "env_origin_w_m": list(ENV_ORIGIN_W_M),
        "root_states": root_states,
        "predator_alive_before": list(predator_alive_before),
        "raw_actions": _zero_actions() if raw_actions is None else raw_actions,
        "progress_offsets_previous_minus_current_m": {
            "predator": list(previous_predator_minus_current_m),
            "minimum": previous_minimum_minus_current_m,
            "prey_horizontal_radius": previous_prey_radius_minus_current_m,
        },
        "episode_step_seen_by_dones": EPISODE_STEP_SEEN_BY_DONES,
    }


def _neutral_states() -> list[dict[str, list[float]]]:
    # The closest physical distance is 2 m, while the prey evasion metric is
    # 1 m because its vertical component is weighted by 0.5.
    return [
        _root_state((0.0, 0.0, 8.0)),
        _root_state((3.0, 0.0, 6.0)),
        _root_state((-3.0, 0.0, 6.0)),
        _root_state((0.0, 0.0, 6.0)),
    ]


def _static_case_inputs() -> tuple[
    list[dict[str, list[float]]],
    dict[str, list[Any]],
    list[dict[str, list[float]]],
]:
    motion_states = _neutral_states()
    motion_states[0] = _root_state(
        (0.0, 0.0, 8.0),
        linear_velocity_w_mps=(1.0, -2.0, 0.5),
        angular_velocity_w_radps=(0.25, -0.5, 0.75),
    )
    motion_states[3] = _root_state(
        (0.0, 0.0, 6.0),
        linear_velocity_w_mps=(-0.5, 1.0, 0.0),
        angular_velocity_w_radps=(0.4, -0.3, 0.2),
    )
    motion_actions = _zero_actions()
    motion_actions["predator"][0] = [2.0, -2.0, 0.5, 0.0]
    motion_actions["prey"] = [-2.0, 2.0, -0.5, 0.0]

    catch_states = [
        _root_state((0.2, 0.0, 6.0)),
        _root_state((-0.2, 0.0, 6.0)),
        _root_state((2.0, 0.0, 6.0)),
        _root_state((0.0, 0.0, 6.0)),
    ]
    return motion_states, motion_actions, catch_states


def _sticky_stage_requests() -> list[dict[str, Any]]:
    stage_1_actions = _zero_actions()
    stage_1_actions["predator"][0] = [0.5, -0.5, 0.25, -0.25]
    stage_2_actions = _zero_actions()
    stage_2_actions["predator"][0] = [1.0, 1.0, 1.0, 1.0]
    return [
        {
            "stage_name": "one_predator_newly_crashes",
            "requested": _request(
                [
                    _root_state((-2.0, 0.0, 0.099)),
                    _root_state((2.0, 0.0, 6.0)),
                    _root_state((0.0, 2.0, 6.0)),
                    _root_state((0.0, -2.0, 6.0)),
                ],
                raw_actions=stage_1_actions,
                purpose="new crash event masks the predator and applies one OOB penalty",
            ),
        },
        {
            "stage_name": "dead_predator_repositioned_safely",
            "requested": _request(
                [
                    _root_state((-2.0, 0.0, 6.0)),
                    _root_state((2.0, 0.0, 6.0)),
                    _root_state((0.0, 2.0, 6.0)),
                    _root_state((0.0, -2.0, 6.0)),
                ],
                predator_alive_before=(False, True, True),
                raw_actions=stage_2_actions,
                purpose="safe position does not resurrect or repeat the event penalty",
            ),
        },
    ]


def _progress_stage_requests() -> list[dict[str, Any]]:
    return [
        {
            "stage_name": "nonzero_progress_first_read",
            "requested": _request(
                _neutral_states(),
                previous_predator_minus_current_m=(0.10, -0.10, 0.04),
                previous_minimum_minus_current_m=-0.10,
                previous_prey_radius_minus_current_m=0.25,
                purpose="nonzero history is consumed and replaced by current geometry",
            ),
        },
        {
            "stage_name": "unchanged_geometry_second_read",
            "requested": _request(
                _neutral_states(),
                purpose="carried updated history makes unchanged-state progress zero",
            ),
        },
    ]


def _clear_episode_tracking(env: Uav3v1Env) -> None:
    zero_names = (
        "_episode_catches",
        "_episode_clean_catches",
        "_episode_forced_prey_oob",
        "_episode_pred_oob",
        "_episode_prey_oob",
        "_episode_teammate_close",
        "_episode_pred_oob_by_agent",
        "_episode_pred_soft_arena_sum",
        "_episode_pred_soft_arena_max",
        "_episode_pred_soft_arena_steps",
        "_episode_prey_soft_arena_sum",
        "_episode_prey_soft_arena_max",
        "_episode_prey_soft_arena_steps",
    )
    for name in zero_names:
        getattr(env, name).zero_()
    for name in (
        "_episode_min_distance",
        "_episode_min_predator_height",
        "_episode_min_prey_height",
        "_episode_min_teammate_distance",
    ):
        getattr(env, name).fill_(100.0)
    env.extras.clear()


def _write_root_states(env: Uav3v1Env, requested: dict[str, Any]) -> None:
    env_origin = torch.tensor(
        requested["env_origin_w_m"], dtype=torch.float32, device=env.device
    )
    env._terrain.env_origins[:] = env_origin
    assets = [*env._predators, env._prey]
    for asset, state_spec in zip(assets, requested["root_states"], strict=True):
        asset.reset()
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] = (
            torch.tensor(
                state_spec["root_link_pos_local_m"],
                dtype=torch.float32,
                device=env.device,
            )
            + env_origin
        )
        root_state[:, 3:7] = torch.tensor(
            state_spec["root_link_quat_wb_wxyz"],
            dtype=torch.float32,
            device=env.device,
        )
        root_state[:, 7:10] = torch.tensor(
            state_spec["root_com_lin_vel_w_mps"],
            dtype=torch.float32,
            device=env.device,
        )
        root_state[:, 10:13] = torch.tensor(
            state_spec["root_ang_vel_w_radps"],
            dtype=torch.float32,
            device=env.device,
        )
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel)

    env.sim.forward()
    # Positive time invalidates Isaac Lab's cached root velocities after a
    # direct write without integrating the physical state.
    env.scene.update(env.sim.cfg.dt)


def _system_com_pos_w(asset) -> torch.Tensor:
    masses = asset.root_physx_view.get_masses().to(asset.device)
    total_mass = masses.sum(dim=1, keepdim=True)
    return (masses[..., None] * asset.data.body_com_pos_w).sum(dim=1) / total_mass


def _progress_memory(env: Uav3v1Env) -> dict[str, torch.Tensor]:
    return {
        "previous_predator_prey_distance_m": env._prev_pred_prey_distances.clone(),
        "previous_min_predator_prey_distance_m": env._prev_min_pred_prey_distance.clone(),
        "previous_prey_horizontal_radius_m": env._prev_prey_horiz.clone(),
    }


def _set_progress_memory(
    env: Uav3v1Env,
    requested: dict[str, Any],
    progress_override: dict[str, torch.Tensor] | None,
) -> None:
    if progress_override is not None:
        env._prev_pred_prey_distances[:] = progress_override[
            "previous_predator_prey_distance_m"
        ]
        env._prev_min_pred_prey_distance[:] = progress_override[
            "previous_min_predator_prey_distance_m"
        ]
        env._prev_prey_horiz[:] = progress_override[
            "previous_prey_horizontal_radius_m"
        ]
        return

    offsets = requested["progress_offsets_previous_minus_current_m"]
    predator_offset = torch.tensor(
        offsets["predator"], dtype=torch.float32, device=env.device
    )
    env._prev_pred_prey_distances[:] = env._current_distances + predator_offset
    env._prev_min_pred_prey_distance[:] = (
        env._active_distances.min(dim=1).values + float(offsets["minimum"])
    )
    env._prev_prey_horiz[:] = (
        env._prey_horiz + float(offsets["prey_horizontal_radius"])
    )


def _read_root_inputs(env: Uav3v1Env) -> dict[str, Any]:
    predators = env._predators
    prey = env._prey
    return {
        "env_origin_w_m": _tensor_list(env._terrain.env_origins[0]),
        "predator_root_link_pos_w_m": _tensor_list(
            torch.stack([asset.data.root_link_pos_w for asset in predators], dim=1)[0]
        ),
        "predator_root_link_quat_wb_wxyz": _tensor_list(
            torch.stack([asset.data.root_link_quat_w for asset in predators], dim=1)[0]
        ),
        "predator_root_com_lin_vel_w_mps": _tensor_list(
            torch.stack([asset.data.root_lin_vel_w for asset in predators], dim=1)[0]
        ),
        "predator_root_com_lin_vel_b_mps": _tensor_list(
            torch.stack([asset.data.root_lin_vel_b for asset in predators], dim=1)[0]
        ),
        "predator_root_ang_vel_b_radps": _tensor_list(
            torch.stack([asset.data.root_ang_vel_b for asset in predators], dim=1)[0]
        ),
        "predator_projected_gravity_b": _tensor_list(
            torch.stack([asset.data.projected_gravity_b for asset in predators], dim=1)[0]
        ),
        "predator_system_com_pos_w_m": _tensor_list(
            torch.stack([_system_com_pos_w(asset) for asset in predators], dim=1)[0]
        ),
        "prey_root_link_pos_w_m": _tensor_list(prey.data.root_link_pos_w[0]),
        "prey_root_link_quat_wb_wxyz": _tensor_list(prey.data.root_link_quat_w[0]),
        "prey_root_com_lin_vel_w_mps": _tensor_list(prey.data.root_lin_vel_w[0]),
        "prey_root_com_lin_vel_b_mps": _tensor_list(prey.data.root_lin_vel_b[0]),
        "prey_root_ang_vel_b_radps": _tensor_list(prey.data.root_ang_vel_b[0]),
        "prey_projected_gravity_b": _tensor_list(prey.data.projected_gravity_b[0]),
        "prey_system_com_pos_w_m": _tensor_list(_system_com_pos_w(prey)[0]),
    }


def _memory_lists(memory: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "previous_predator_prey_distance_m": _tensor_list(
            memory["previous_predator_prey_distance_m"][0]
        ),
        "previous_min_predator_prey_distance_m": _scalar(
            memory["previous_min_predator_prey_distance_m"][0]
        ),
        "previous_prey_horizontal_radius_m": _scalar(
            memory["previous_prey_horizontal_radius_m"][0]
        ),
    }


def _evaluate_requested_state(
    env: Uav3v1Env,
    requested: dict[str, Any],
    *,
    repeat_index: int,
    alive_override: torch.Tensor | None = None,
    progress_override: dict[str, torch.Tensor] | None = None,
    stage_index: int | None = None,
    stage_name: str | None = None,
) -> dict[str, Any]:
    _clear_episode_tracking(env)
    _write_root_states(env, requested)

    if alive_override is None:
        env._pred_alive[:] = torch.tensor(
            requested["predator_alive_before"], dtype=torch.bool, device=env.device
        )
    else:
        env._pred_alive[:] = alive_override
    predator_alive_before = env._pred_alive.clone()
    env._pred_oob[:] = ~predator_alive_before
    env._pred_newly_oob.zero_()
    env._prey_oob.zero_()
    env._caught.zero_()
    env._has_nan.zero_()

    raw_predator_actions = torch.tensor(
        requested["raw_actions"]["predator"], dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    raw_prey_actions = torch.tensor(
        requested["raw_actions"]["prey"], dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    env._pre_physics_step(
        {
            "predator": raw_predator_actions.reshape(1, -1),
            "prey": raw_prey_actions,
        }
    )
    env.episode_length_buf.fill_(int(requested["episode_step_seen_by_dones"]))
    env._intermediate_values_valid = False
    env._compute_intermediate_values()
    _set_progress_memory(env, requested, progress_override)

    progress_before = _progress_memory(env)
    root_inputs = _read_root_inputs(env)
    predator_alive_after = env._pred_alive.clone()
    predator_pos_local = env._pred_pos_rel.clone()
    prey_pos_local = env._prey_pos_rel.clone()
    raw_distances = torch.linalg.norm(
        predator_pos_local - prey_pos_local.unsqueeze(1), dim=2
    )
    weighted_delta = predator_pos_local - prey_pos_local.unsqueeze(1)
    weighted_delta[:, :, 2] *= env.cfg.prey_evasion_vertical_weight
    weighted_distances = torch.linalg.norm(weighted_delta, dim=2)
    weighted_active_distances = torch.where(
        predator_alive_after,
        weighted_distances,
        torch.full_like(weighted_distances, 100.0),
    )

    predator_flying = (
        (predator_pos_local[:, :, 2] > env.cfg.min_height)
        & predator_alive_after
    )
    predator_progress_valid = (
        predator_flying
        & (~env._caught).unsqueeze(1)
        & (~env._prey_oob).unsqueeze(1)
        & (~env._pred_oob)
    )
    prey_flying = prey_pos_local[:, 2] > env.cfg.min_height
    prey_progress_valid = (
        (~env._caught)
        & predator_alive_after.any(dim=1)
        & (~env._prey_oob)
    )
    closest_predator_index = env._active_distances.argmin(dim=1)
    predator_indices = torch.arange(env._P, device=env.device).unsqueeze(0)
    is_catcher = (
        predator_indices == closest_predator_index.unsqueeze(1)
    ) & env._caught.unsqueeze(1)
    is_assisting = (
        (~is_catcher)
        & env._caught.unsqueeze(1)
        & predator_alive_after
        & (env._active_distances < env.cfg.assist_distance)
    )

    raw_predator_progress = (
        env._prev_pred_prey_distances - env._active_distances
    )
    clipped_predator_progress = torch.clamp(
        raw_predator_progress,
        min=-env.cfg.predator_distance_progress_reward_clip,
        max=env.cfg.predator_distance_progress_reward_clip,
    )
    minimum_active_distance = env._active_distances.min(dim=1).values
    raw_prey_progress = minimum_active_distance - env._prev_min_pred_prey_distance
    clipped_prey_progress = torch.clamp(
        raw_prey_progress,
        min=-env.cfg.prey_distance_progress_reward_clip,
        max=env.cfg.prey_distance_progress_reward_clip,
    )
    raw_boundary_progress = env._prev_prey_horiz - env._prey_horiz
    clipped_boundary_progress = torch.clamp(
        raw_boundary_progress,
        min=-env.cfg.prey_boundary_progress_reward_clip,
        max=env.cfg.prey_boundary_progress_reward_clip,
    )
    boundary_progress_start = (
        env.cfg.arena_radius * env.cfg.prey_boundary_progress_start_fraction
    )
    boundary_progress_width = max(
        env.cfg.arena_radius - boundary_progress_start, 1.0e-6
    )
    boundary_pressure = torch.clamp(
        (env._prey_horiz - boundary_progress_start) / boundary_progress_width,
        min=0.0,
        max=1.0,
    )

    terminated, truncated = env._get_dones()
    rewards = env._get_rewards()
    log = env.extras["log"]
    predator_terms = {name: _scalar(log[key]) for name, key in PREDATOR_TERM_LOGS.items()}
    prey_terms = {name: _scalar(log[key]) for name, key in PREY_TERM_LOGS.items()}
    predator_reward = _scalar(rewards["predator"][0])
    prey_reward = _scalar(rewards["prey"][0])
    predator_term_sum_error = sum(predator_terms.values()) - predator_reward
    prey_term_sum_error = sum(prey_terms.values()) - prey_reward
    if abs(predator_term_sum_error) > 1.0e-4 or abs(prey_term_sum_error) > 1.0e-4:
        raise RuntimeError(
            "Logged reward terms do not reconstruct returned rewards: "
            f"predator={predator_term_sum_error}, prey={prey_term_sum_error}"
        )
    for role, terms, disabled in (
        ("predator", predator_terms, ("proximity", "assist", "boundary")),
        ("prey", prey_terms, ("boundary_progress", "boundary")),
    ):
        nonzero_disabled = {name: terms[name] for name in disabled if terms[name] != 0.0}
        if nonzero_disabled:
            raise RuntimeError(f"Disabled {role} reward terms became nonzero: {nonzero_disabled}")

    snapshot: dict[str, Any] = {
        "repeat_index": repeat_index,
        "inputs": {
            **root_inputs,
            "predator_alive_before": _tensor_list(predator_alive_before[0]),
            "raw_actions": {
                "predator": _tensor_list(raw_predator_actions[0]),
                "prey": _tensor_list(raw_prey_actions[0]),
            },
            "executed_actions": {
                "predator": _tensor_list(env._pred_actions[0]),
                "prey": _tensor_list(env._prey_actions[0]),
            },
            "progress_before": _memory_lists(progress_before),
            "episode_step_seen_by_dones": int(env.episode_length_buf[0].item()),
        },
        "intermediates": {
            "predator_root_link_pos_local_m": _tensor_list(predator_pos_local[0]),
            "prey_root_link_pos_local_m": _tensor_list(prey_pos_local[0]),
            "raw_predator_prey_distance_m": _tensor_list(raw_distances[0]),
            "current_predator_prey_distance_m": _tensor_list(env._current_distances[0]),
            "active_predator_prey_distance_m": _tensor_list(env._active_distances[0]),
            "weighted_active_predator_prey_distance_m": _tensor_list(
                weighted_active_distances[0]
            ),
            "predator_soft_arena_outside_m": _tensor_list(
                env._pred_soft_arena_outside[0]
            ),
            "prey_soft_arena_outside_m": _scalar(env._prey_soft_arena_outside[0]),
            "predator_alive_after": _tensor_list(predator_alive_after[0]),
            "predator_newly_oob": _tensor_list(env._pred_newly_oob[0]),
            "caught": _scalar_bool(env._caught[0]),
            "prey_oob": _scalar_bool(env._prey_oob[0]),
            "no_active_predators": not bool(predator_alive_after[0].any().item()),
            "predator_flying": _tensor_list(predator_flying[0]),
            "predator_progress_valid": _tensor_list(predator_progress_valid[0]),
            "prey_flying": _scalar_bool(prey_flying[0]),
            "prey_progress_valid": _scalar_bool(prey_progress_valid[0]),
            "closest_predator_index": int(closest_predator_index[0].item()),
            "is_catcher": _tensor_list(is_catcher[0]),
            "is_assisting": _tensor_list(is_assisting[0]),
            "raw_predator_distance_progress_m": _tensor_list(raw_predator_progress[0]),
            "clipped_predator_distance_progress_m": _tensor_list(
                clipped_predator_progress[0]
            ),
            "raw_prey_distance_progress_m": _scalar(raw_prey_progress[0]),
            "clipped_prey_distance_progress_m": _scalar(clipped_prey_progress[0]),
            "raw_prey_boundary_progress_m": _scalar(raw_boundary_progress[0]),
            "clipped_prey_boundary_progress_m": _scalar(
                clipped_boundary_progress[0]
            ),
            "prey_boundary_pressure": _scalar(boundary_pressure[0]),
        },
        "outputs": {
            "reward": {"predator": predator_reward, "prey": prey_reward},
            "terms": {"predator": predator_terms, "prey": prey_terms},
            "progress_after": _memory_lists(_progress_memory(env)),
            "terminated": {
                "predator": _scalar_bool(terminated["predator"][0]),
                "prey": _scalar_bool(terminated["prey"][0]),
            },
            "truncated": {
                "predator": _scalar_bool(truncated["predator"][0]),
                "prey": _scalar_bool(truncated["prey"][0]),
            },
            "term_sum_error": {
                "predator": predator_term_sum_error,
                "prey": prey_term_sum_error,
            },
        },
    }
    if stage_index is not None:
        snapshot["stage_index"] = stage_index
    if stage_name is not None:
        snapshot["stage_name"] = stage_name
    return snapshot


def _run_sequence(
    env: Uav3v1Env,
    stage_requests: list[dict[str, Any]],
    repeat_index: int,
) -> dict[str, Any]:
    alive_override: torch.Tensor | None = None
    progress_override: dict[str, torch.Tensor] | None = None
    stages = []
    for stage_index, stage in enumerate(stage_requests):
        snapshot = _evaluate_requested_state(
            env,
            stage["requested"],
            repeat_index=repeat_index,
            alive_override=alive_override,
            progress_override=progress_override,
            stage_index=stage_index,
            stage_name=stage["stage_name"],
        )
        stages.append(snapshot)
        alive_override = env._pred_alive.clone()
        progress_override = _progress_memory(env)
    return {"repeat_index": repeat_index, "stages": stages}


def _assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite number at {path}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")


def _static_cases() -> list[dict[str, Any]]:
    motion_states, motion_actions, catch_states = _static_case_inputs()
    safe_predators = [
        _root_state((-2.0, 0.0, 6.0)),
        _root_state((2.0, 0.0, 6.0)),
        _root_state((0.0, 2.0, 6.0)),
    ]
    return [
        {
            "name": "neutral_translated_origin_vertical_evasion",
            "requested": _request(
                _neutral_states(),
                purpose="baseline upright/alive and vertically weighted prey evasion",
            ),
        },
        {
            "name": "upside_down_root_link_0p101m",
            "requested": _request(
                [
                    _root_state(
                        (-2.0, 0.0, 0.101),
                        quaternion_wxyz=UPSIDE_DOWN_ROLL_QUATERNION_WXYZ,
                    ),
                    _root_state((2.0, 0.0, 6.0)),
                    _root_state((0.0, 2.0, 6.0)),
                    _root_state(
                        (0.0, -2.0, 0.101),
                        quaternion_wxyz=UPSIDE_DOWN_ROLL_QUATERNION_WXYZ,
                    ),
                ],
                purpose="root-link height, upright, height, low-altitude and soft-arena terms",
            ),
        },
        {
            "name": "velocity_and_clipped_action",
            "requested": _request(
                motion_states,
                raw_actions=motion_actions,
                purpose="linear/angular velocity and executed clipped-action penalties",
            ),
        },
        {
            "name": "progress_both_signs_and_clips",
            "requested": _request(
                _neutral_states(),
                previous_predator_minus_current_m=(0.10, -0.10, 0.04),
                previous_minimum_minus_current_m=-0.10,
                purpose="predator progress signs/clips and positive clipped prey progress",
            ),
        },
        {
            "name": "prey_progress_negative_clip",
            "requested": _request(
                _neutral_states(),
                previous_minimum_minus_current_m=0.10,
                purpose="negative clipped prey distance progress",
            ),
        },
        {
            "name": "catch_tie",
            "requested": _request(
                catch_states,
                previous_predator_minus_current_m=(0.10, 0.10, 0.10),
                previous_minimum_minus_current_m=-0.10,
                purpose="catch attribution, caught rewards, progress gates and disabled assist",
            ),
        },
        {
            "name": "outside_soft_sphere",
            "requested": _request(
                [
                    _root_state((6.0, 0.0, 6.0)),
                    _root_state((-6.0, 0.0, 6.0)),
                    _root_state((0.0, 6.0, 6.0)),
                    _root_state((0.0, -6.0, 6.0)),
                ],
                previous_prey_radius_minus_current_m=1.0,
                purpose="squared soft-arena penalties and disabled hard-boundary terms",
            ),
        },
        {
            "name": "prey_below_minimum",
            "requested": _request(
                [*safe_predators, _root_state((0.0, -2.0, 0.099))],
                previous_predator_minus_current_m=(0.10, 0.10, 0.10),
                previous_minimum_minus_current_m=-0.10,
                purpose="prey OOB/flying gates and terminal reward terms",
            ),
        },
        {
            "name": "catch_and_prey_oob",
            "requested": _request(
                [
                    _root_state((0.0, 0.0, 0.150)),
                    _root_state((2.0, 0.0, 6.0)),
                    _root_state((-2.0, 0.0, 6.0)),
                    _root_state((0.0, 0.0, 0.099)),
                ],
                previous_predator_minus_current_m=(0.10, 0.10, 0.10),
                previous_minimum_minus_current_m=-0.10,
                purpose="simultaneous catch/prey-OOB terminal rewards and progress gates",
            ),
        },
        {
            "name": "catch_plus_other_predator_crash",
            "requested": _request(
                [
                    _root_state((0.2, 0.0, 6.0)),
                    _root_state((-0.2, 0.0, 6.0)),
                    _root_state((2.0, 0.0, 0.099)),
                    _root_state((0.0, 0.0, 6.0)),
                ],
                purpose="catch bonus and a different predator's OOB event on one transition",
            ),
        },
        {
            "name": "all_predators_inactive",
            "requested": _request(
                _neutral_states(),
                predator_alive_before=(False, False, False),
                raw_actions={
                    "predator": [[1.0, 1.0, 1.0, 1.0] for _ in range(3)],
                    "prey": [0.0, 0.0, 0.0, 0.0],
                },
                previous_predator_minus_current_m=(0.10, 0.10, 0.10),
                previous_minimum_minus_current_m=-0.10,
                purpose="inactive masks, no-active terminal and 100 m prey-evasion sentinel",
            ),
        },
    ]


def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg = Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    env = Uav3v1Env(cfg=cfg)

    cases = []
    for case in _static_cases():
        cases.append(
            {
                "name": case["name"],
                "purpose": case["requested"]["purpose"],
                "requested": case["requested"],
                "repeats": [
                    _evaluate_requested_state(
                        env,
                        case["requested"],
                        repeat_index=repeat_index,
                    )
                    for repeat_index in range(args_cli.repeats)
                ],
            }
        )

    sequence_specs = (
        ("sticky_crash_event", _sticky_stage_requests()),
        ("progress_memory_update", _progress_stage_requests()),
    )
    sequences = [
        {
            "name": sequence_name,
            "stage_requests": stage_requests,
            "repeats": [
                _run_sequence(env, stage_requests, repeat_index)
                for repeat_index in range(args_cli.repeats)
            ],
        }
        for sequence_name, stage_requests in sequence_specs
    ]

    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_root = Path("/workspace/isaaclab")
    isaaclab_metadata = _git_metadata(isaaclab_root)
    if isaaclab_metadata["commit"] is None:
        isaaclab_metadata["commit"] = os.environ.get("ISAACLAB_COMMIT")
    output = {
        "schema_version": "uavpredatorprey.reward_oracle.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "producer": {
            "backend": "isaac_root_state_reward_readback",
            "task_id": TASK_ID,
            "uavpredatorprey": _git_metadata(repo_root),
            "isaaclab": isaaclab_metadata,
            "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
            "torch_version": torch.__version__,
        },
        "contract": {
            "predator_count": env._P,
            "physics_dt_s": env.sim.cfg.dt,
            "policy_dt_s": env.step_dt,
            "predator_team_reduction": "mean_over_three_per_drone_rewards",
            "position_point": "root_link",
            "position_frame": "world_with_reward_geometry_relative_to_environment_origin",
            "linear_velocity_source": "root_body_com_expressed_in_body",
            "angular_velocity_source": "root_link_expressed_in_body",
            "action_source": "clipped_normalized_action_masked_by_prior_predator_alive",
            "inactive_distance_sentinel_m": 100.0,
            "progress_offset_convention": "previous_minus_current",
            "progress_update": {
                "predator": "current_raw_root_link_distances",
                "prey_minimum": "current_minimum_active_distance",
                "prey_horizontal_radius": "current_root_link_xy_radius",
            },
            "parameters": {
                "target_height_m": env.cfg.target_height,
                "min_height_m": env.cfg.min_height,
                "catch_distance_m": env.cfg.catch_distance,
                "assist_distance_m": env.cfg.assist_distance,
                "arena_radius_m": env.cfg.arena_radius,
                "arena_center_z_m": env.cfg.arena_center_z,
                "soft_arena_radius_m": env.cfg.soft_arena_radius,
                "boundary_warn_fraction": env.cfg.boundary_warn_fraction,
                "boundary_penalty_scale": env.cfg.boundary_penalty_scale,
                "upright_reward_scale": env.cfg.upright_reward_scale,
                "height_penalty_scale": env.cfg.height_penalty_scale,
                "lin_vel_penalty": env.cfg.lin_vel_penalty,
                "predator_ang_vel_penalty": env.cfg.predator_ang_vel_penalty,
                "prey_ang_vel_penalty": env.cfg.prey_ang_vel_penalty,
                "action_penalty": env.cfg.action_penalty,
                "predator_proximity_reward_scale": env.cfg.predator_proximity_reward_scale,
                "predator_distance_progress_reward_scale": env.cfg.predator_distance_progress_reward_scale,
                "predator_distance_progress_reward_clip_m": env.cfg.predator_distance_progress_reward_clip,
                "predator_catch_bonus": env.cfg.predator_catch_bonus,
                "predator_assist_bonus": env.cfg.predator_assist_bonus,
                "predator_oob_penalty": env.cfg.predator_oob_penalty,
                "prey_alive_bonus": env.cfg.prey_alive_bonus,
                "prey_evasion_reward_scale": env.cfg.prey_evasion_reward_scale,
                "prey_evasion_vertical_weight": env.cfg.prey_evasion_vertical_weight,
                "prey_low_altitude_penalty_scale": env.cfg.prey_low_altitude_penalty_scale,
                "prey_low_altitude_margin_m": env.cfg.prey_low_altitude_margin,
                "prey_distance_progress_reward_scale": env.cfg.prey_distance_progress_reward_scale,
                "prey_distance_progress_reward_clip_m": env.cfg.prey_distance_progress_reward_clip,
                "prey_boundary_progress_reward_scale": env.cfg.prey_boundary_progress_reward_scale,
                "prey_boundary_progress_reward_clip_m": env.cfg.prey_boundary_progress_reward_clip,
                "prey_boundary_progress_start_fraction": env.cfg.prey_boundary_progress_start_fraction,
                "prey_caught_penalty": env.cfg.prey_caught_penalty,
                "prey_oob_penalty": env.cfg.prey_oob_penalty,
                "soft_arena_penalty_scale": env.cfg.soft_arena_penalty_scale,
            },
            "disabled_terms": {
                "predator": ["proximity", "assist", "boundary"],
                "prey": ["boundary_progress", "boundary"],
            },
            "terminal_transitions_receive_reward": True,
            "nan_physx_fixture": "excluded_non_finite_root_states_are_not_written_to_physx",
        },
        "cases": cases,
        "sequences": sequences,
    }
    _assert_finite_json(output)

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )
    print(
        f"[INFO] Wrote {len(cases)} reward cases and {len(sequences)} sequences "
        f"to {output_path}"
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
