"""Export deterministic Isaac termination fixtures for the JAX environment port.

The exporter writes finite root states and evaluates the target task's existing
termination code without advancing physics or invoking a policy.  It records
the root-link readback that actually reached Isaac, the sticky predator-alive
transition, individual termination causes, and the separate terminated and
truncated outputs.

NaNs are deliberately excluded: sending non-finite poses through PhysX is not
a safe or deterministic fixture mechanism.  NaN handling belongs in pure
backend unit tests, while this artifact covers the finite Isaac contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


DEFAULT_OUTPUT = "/workspace/artifacts/transfer/termination_v0/isaac_oracle_v1.json"
TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"
ENV_ORIGIN_W_M = (12.0, -7.0, 1.0)
IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
FINITE_CASE_EPISODE_STEP = 100


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

from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.episode_semantics import (
    all_predators_inactive,
)
from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env import Uav3v1Env
from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env_cfg import (
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg,
)


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


def _scalar_bool(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().item())


def _positions(
    predator_0: tuple[float, float, float],
    predator_1: tuple[float, float, float],
    predator_2: tuple[float, float, float],
    prey: tuple[float, float, float],
) -> dict[str, list[Any]]:
    return {
        "predator_root_link_pos_local_m": [
            list(predator_0),
            list(predator_1),
            list(predator_2),
        ],
        "prey_root_link_pos_local_m": list(prey),
    }


def _requested_case(
    *,
    positions: dict[str, list[Any]],
    episode_step_seen_by_dones: int,
    predator_alive_before: tuple[bool, bool, bool] = (True, True, True),
) -> dict[str, Any]:
    return {
        "env_origin_w_m": list(ENV_ORIGIN_W_M),
        **positions,
        "predator_alive_before": list(predator_alive_before),
        "episode_step_before_increment": episode_step_seen_by_dones - 1,
        "episode_step_seen_by_dones": episode_step_seen_by_dones,
    }


def _static_cases() -> list[dict[str, Any]]:
    safe_positions = _positions(
        (-2.0, 0.0, 6.0),
        (2.0, 0.0, 6.0),
        (0.0, 2.0, 6.0),
        (0.0, -2.0, 6.0),
    )
    catch_inside_positions = _positions(
        (0.299, 0.0, 6.0),
        (2.0, 0.0, 6.0),
        (-2.0, 0.0, 6.0),
        (0.0, 0.0, 6.0),
    )
    catch_boundary_positions = _positions(
        (0.300, 0.0, 6.0),
        (2.0, 0.0, 6.0),
        (-2.0, 0.0, 6.0),
        (0.0, 0.0, 6.0),
    )
    safe_predators = (
        (-2.0, 0.0, 6.0),
        (2.0, 0.0, 6.0),
        (0.0, 2.0, 6.0),
    )
    return [
        {
            "name": "safe_pre_timeout",
            "requested": _requested_case(
                positions=safe_positions,
                episode_step_seen_by_dones=498,
            ),
        },
        {
            "name": "safe_timeout",
            "requested": _requested_case(
                positions=safe_positions,
                episode_step_seen_by_dones=499,
            ),
        },
        {
            "name": "catch_inside_0p299m",
            "requested": _requested_case(
                positions=catch_inside_positions,
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "catch_exactly_0p300m",
            "requested": _requested_case(
                positions=catch_boundary_positions,
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "prey_below_0p099m",
            "requested": _requested_case(
                positions=_positions(*safe_predators, (0.0, -2.0, 0.099)),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "prey_at_min_0p100m",
            "requested": _requested_case(
                positions=_positions(*safe_predators, (0.0, -2.0, 0.100)),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "one_predator_crash",
            "requested": _requested_case(
                positions=_positions(
                    (-2.0, 0.0, 0.099),
                    (2.0, 0.0, 6.0),
                    (0.0, 2.0, 6.0),
                    (0.0, -2.0, 6.0),
                ),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "all_predators_crash",
            "requested": _requested_case(
                positions=_positions(
                    (-2.0, 0.0, 0.099),
                    (2.0, 0.0, 0.099),
                    (0.0, 2.0, 0.099),
                    (0.0, -2.0, 6.0),
                ),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "outside_soft_sphere",
            "requested": _requested_case(
                positions=_positions(
                    (5.5, 0.0, 6.0),
                    (-5.5, 0.0, 6.0),
                    (0.0, 5.5, 6.0),
                    (0.0, -5.5, 6.0),
                ),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "name": "catch_and_timeout",
            "requested": _requested_case(
                positions=catch_inside_positions,
                episode_step_seen_by_dones=499,
            ),
        },
    ]


def _sticky_stages() -> list[dict[str, Any]]:
    return [
        {
            "stage_name": "predator_0_crashes",
            "requested": _requested_case(
                positions=_positions(
                    (-2.0, 0.0, 0.099),
                    (2.0, 0.0, 6.0),
                    (0.0, 2.0, 6.0),
                    (0.0, -2.0, 6.0),
                ),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
            ),
        },
        {
            "stage_name": "predator_1_crashes_after_0_recovers_position",
            "requested": _requested_case(
                positions=_positions(
                    (-2.0, 0.0, 6.0),
                    (2.0, 0.0, 0.099),
                    (0.0, 2.0, 6.0),
                    (0.0, -2.0, 6.0),
                ),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
                predator_alive_before=(False, True, True),
            ),
        },
        {
            "stage_name": "predator_2_crashes_after_1_recovers_position",
            "requested": _requested_case(
                positions=_positions(
                    (-2.0, 0.0, 6.0),
                    (2.0, 0.0, 6.0),
                    (0.0, 2.0, 0.099),
                    (0.0, -2.0, 6.0),
                ),
                episode_step_seen_by_dones=FINITE_CASE_EPISODE_STEP,
                predator_alive_before=(False, False, True),
            ),
        },
    ]


def _write_root_state(env: Uav3v1Env, requested: dict[str, Any]) -> None:
    env_origin = torch.tensor(
        requested["env_origin_w_m"], dtype=torch.float32, device=env.device
    )
    env._terrain.env_origins[:] = env_origin
    local_positions = [
        *requested["predator_root_link_pos_local_m"],
        requested["prey_root_link_pos_local_m"],
    ]
    assets = [*env._predators, env._prey]

    for asset, local_position in zip(assets, local_positions, strict=True):
        asset.reset()
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] = (
            torch.tensor(local_position, dtype=torch.float32, device=env.device)
            + env_origin
        )
        root_state[:, 3:7] = torch.tensor(
            IDENTITY_QUATERNION_WXYZ, dtype=torch.float32, device=env.device
        )
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel)

    env.sim.forward()
    # A positive timestamp invalidates Isaac Lab's cached root/link readback
    # after direct pose writes without integrating the physical state.
    env.scene.update(env.sim.cfg.dt)


def _evaluate_requested_state(
    env: Uav3v1Env,
    requested: dict[str, Any],
    *,
    repeat_index: int,
    set_requested_alive: bool,
    stage_index: int | None = None,
    stage_name: str | None = None,
) -> dict[str, Any]:
    _write_root_state(env, requested)
    if set_requested_alive:
        env._pred_alive[:] = torch.tensor(
            requested["predator_alive_before"], dtype=torch.bool, device=env.device
        )

    predator_alive_before = env._pred_alive.clone()
    env.episode_length_buf.fill_(int(requested["episode_step_seen_by_dones"]))
    env._pred_oob[:] = ~predator_alive_before
    env._pred_newly_oob.zero_()
    env._prey_oob.zero_()
    env._caught.zero_()
    env._has_nan.zero_()
    env._intermediate_values_valid = False

    predator_root_link_pos_w = torch.stack(
        [predator.data.root_link_pos_w for predator in env._predators], dim=1
    )
    predator_root_link_quat_wb = torch.stack(
        [predator.data.root_link_quat_w for predator in env._predators], dim=1
    )
    prey_root_link_pos_w = env._prey.data.root_link_pos_w
    prey_root_link_quat_wb = env._prey.data.root_link_quat_w
    raw_distances = torch.linalg.norm(
        predator_root_link_pos_w - prey_root_link_pos_w[:, None, :], dim=2
    )

    env._compute_intermediate_values()
    terminated, truncated = env._get_dones()

    predator_pos_local = predator_root_link_pos_w - env._terrain.env_origins[:, None, :]
    prey_pos_local = prey_root_link_pos_w - env._terrain.env_origins
    predator_below_min = predator_pos_local[:, :, 2] < env.cfg.min_height
    prey_below_min = prey_pos_local[:, 2] < env.cfg.min_height
    no_active_predators = all_predators_inactive(env._pred_alive)
    time_out = env.episode_length_buf >= env.max_episode_length - 1

    snapshot: dict[str, Any] = {
        "repeat_index": repeat_index,
        "inputs": {
            "env_origin_w_m": _tensor_list(env._terrain.env_origins[0]),
            "predator_root_link_pos_w_m": _tensor_list(predator_root_link_pos_w[0]),
            "predator_root_link_quat_wb_wxyz": _tensor_list(
                predator_root_link_quat_wb[0]
            ),
            "prey_root_link_pos_w_m": _tensor_list(prey_root_link_pos_w[0]),
            "prey_root_link_quat_wb_wxyz": _tensor_list(prey_root_link_quat_wb[0]),
            "predator_alive_before": _tensor_list(predator_alive_before[0]),
            "episode_step_before_increment": int(
                requested["episode_step_before_increment"]
            ),
            "episode_step_seen_by_dones": int(env.episode_length_buf[0].item()),
            # Backward-compatible concise alias; this is the post-increment
            # counter observed by the task's ``_get_dones`` implementation.
            "episode_step": int(env.episode_length_buf[0].item()),
        },
        "outputs": {
            "predator_alive_after": _tensor_list(env._pred_alive[0]),
            "predator_newly_oob": _tensor_list(env._pred_newly_oob[0]),
            "raw_predator_prey_distance_m": _tensor_list(raw_distances[0]),
            "current_predator_prey_distance_m": _tensor_list(
                env._current_distances[0]
            ),
            "active_predator_prey_distance_m": _tensor_list(
                env._active_distances[0]
            ),
            "predator_soft_arena_outside_m": _tensor_list(
                env._pred_soft_arena_outside[0]
            ),
            "prey_soft_arena_outside_m": float(
                env._prey_soft_arena_outside[0].detach().cpu().item()
            ),
            "causes": {
                "predator_below_min": _tensor_list(predator_below_min[0]),
                "prey_below_min": _scalar_bool(prey_below_min[0]),
                "caught": _scalar_bool(env._caught[0]),
                "no_active_predators": _scalar_bool(no_active_predators[0]),
                "prey_oob": _scalar_bool(env._prey_oob[0]),
                "has_nan": _scalar_bool(env._has_nan[0]),
                "time_out": _scalar_bool(time_out[0]),
            },
            "terminated": {
                "predator": _scalar_bool(terminated["predator"][0]),
                "prey": _scalar_bool(terminated["prey"][0]),
            },
            "truncated": {
                "predator": _scalar_bool(truncated["predator"][0]),
                "prey": _scalar_bool(truncated["prey"][0]),
            },
        },
    }
    if stage_index is not None:
        snapshot["stage_index"] = stage_index
    if stage_name is not None:
        snapshot["stage_name"] = stage_name
    return snapshot


def _run_sticky_sequence(env: Uav3v1Env, repeat_index: int) -> dict[str, Any]:
    env._pred_alive.fill_(True)
    stages = []
    for stage_index, stage in enumerate(_sticky_stages()):
        stages.append(
            _evaluate_requested_state(
                env,
                stage["requested"],
                repeat_index=repeat_index,
                set_requested_alive=False,
                stage_index=stage_index,
                stage_name=stage["stage_name"],
            )
        )
    return {"repeat_index": repeat_index, "stages": stages}


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
                "requested": case["requested"],
                "repeats": [
                    _evaluate_requested_state(
                        env,
                        case["requested"],
                        repeat_index=repeat_index,
                        set_requested_alive=True,
                    )
                    for repeat_index in range(args_cli.repeats)
                ],
            }
        )

    sequences = [
        {
            "name": "sticky_sequential_predator_death",
            "initial_predator_alive": [True, True, True],
            "stage_requests": _sticky_stages(),
            "repeats": [
                _run_sticky_sequence(env, repeat_index)
                for repeat_index in range(args_cli.repeats)
            ],
        }
    ]

    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_root = Path("/workspace/isaaclab")
    isaaclab_metadata = _git_metadata(isaaclab_root)
    if isaaclab_metadata["commit"] is None:
        isaaclab_metadata["commit"] = os.environ.get("ISAACLAB_COMMIT")
    output = {
        "schema_version": "uavpredatorprey.termination_oracle.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "producer": {
            "backend": "isaac_root_state_termination_readback",
            "task_id": TASK_ID,
            "uavpredatorprey": _git_metadata(repo_root),
            "isaaclab": isaaclab_metadata,
            "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
            "torch_version": torch.__version__,
        },
        "contract": {
            "predator_count": env._P,
            "position_point": "root_link",
            "position_frame": "world_with_height_relative_to_environment_origin",
            "distance_metric": "full_3d_root_link_euclidean",
            "catch_distance_m": env.cfg.catch_distance,
            "catch_comparison": "strict_less_than",
            "min_height_m": env.cfg.min_height,
            "crash_comparison": "strict_less_than",
            "predator_death": "sticky_alive_next=alive_previous_and_not_below_min_height",
            "inactive_distance_sentinel_m": 100.0,
            "arena_center_z_m": env.cfg.arena_center_z,
            "soft_arena_radius_m": env.cfg.soft_arena_radius,
            "soft_arena_exit_terminates": False,
            "max_episode_length_steps": env.max_episode_length,
            "timeout_at_or_after_episode_step": env.max_episode_length - 1,
            "episode_step_semantics": "incremented_before_get_dones",
            "terminated_expression": "caught or no_active_predators or prey_oob or has_nan",
            "truncated_expression": "episode_step >= timeout_at_or_after_episode_step",
            "role_outputs": "identical_for_predator_and_prey",
            "terminated_and_truncated_may_coincide": True,
            "nan_detection": "root_link_positions_or_derived_predator_prey_distances",
            "nan_physx_fixture": "excluded_non_finite_root_poses_are_not_written_to_physx",
        },
        "cases": cases,
        "sequences": sequences,
    }

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )
    print(
        f"[INFO] Wrote {len(cases)} finite cases and {len(sequences)} sticky sequence "
        f"to {output_path}"
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
