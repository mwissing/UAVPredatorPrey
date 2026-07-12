"""Export externally observable reset statistics for the JAX reset port.

Torch and JAX intentionally use different random-number generators. This
oracle therefore records exact reset invariants and distribution summaries,
not paired same-seed coordinates. It never advances the physics simulation.
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


DEFAULT_OUTPUT = "/workspace/artifacts/transfer/reset_v0/isaac_oracle_v1.json"
TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
ROOT_TO_SYSTEM_COM_B_M = (0.0, 0.0, 0.0023829787)
READBACK_BOUND_TOLERANCE_M = 1.0e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--examples", type=int, default=8)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.examples < 0:
        parser.error("--examples cannot be negative")
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

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


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _statistics(value: torch.Tensor) -> dict[str, Any]:
    array = _to_numpy(value).astype(np.float64, copy=False).reshape(-1)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "quantiles": {
            f"{quantile:g}": float(np.quantile(array, quantile))
            for quantile in QUANTILES
        },
    }


def _system_com_pos_w(asset) -> torch.Tensor:
    masses = asset.root_physx_view.get_masses().to(asset.device)
    total_mass = masses.sum(dim=1, keepdim=True)
    return (masses[..., None] * asset.data.body_com_pos_w).sum(dim=1) / total_mass


def _pair_distances_xy(predator_local_m: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.linalg.norm(predator_local_m[:, 0, :2] - predator_local_m[:, 1, :2], dim=1),
            torch.linalg.norm(predator_local_m[:, 0, :2] - predator_local_m[:, 2, :2], dim=1),
            torch.linalg.norm(predator_local_m[:, 1, :2] - predator_local_m[:, 2, :2], dim=1),
        ),
        dim=1,
    )


def _read_reset(
    env: Uav3v1Env,
    *,
    seed: int,
    disk_radius_m: float,
    example_count: int,
) -> dict[str, Any]:
    env.reset(seed=seed)
    origins = env._terrain.env_origins
    predator_root_pos_w_m = torch.stack(
        [predator.data.root_pos_w for predator in env._predators], dim=1
    )
    prey_root_pos_w_m = env._prey.data.root_pos_w
    predator_local_m = predator_root_pos_w_m - origins[:, None, :]
    prey_local_m = prey_root_pos_w_m - origins
    predator_xy_radius_m = torch.linalg.norm(predator_local_m[:, :, :2], dim=2)
    prey_xy_radius_m = torch.linalg.norm(prey_local_m[:, :2], dim=1)
    predator_prey_delta_m = predator_local_m - prey_local_m[:, None, :]
    predator_prey_xy_distance_m = torch.linalg.norm(predator_prey_delta_m[:, :, :2], dim=2)
    predator_prey_distance_m = torch.linalg.norm(predator_prey_delta_m, dim=2)
    predator_pair_xy_distance_m = _pair_distances_xy(predator_local_m)
    remaining_invalid = (
        (predator_prey_xy_distance_m < env.cfg.random_spawn_min_prey_predator_distance).any(dim=1)
        | (predator_pair_xy_distance_m < env.cfg.random_spawn_min_predator_distance).any(dim=1)
    )

    predator_quaternion = torch.stack(
        [predator.data.root_quat_w for predator in env._predators], dim=1
    )
    prey_quaternion = env._prey.data.root_quat_w
    predator_linear_velocity = torch.stack(
        [predator.data.root_lin_vel_w for predator in env._predators], dim=1
    )
    predator_angular_velocity = torch.stack(
        [predator.data.root_ang_vel_b for predator in env._predators], dim=1
    )
    identity_predator = torch.zeros_like(predator_quaternion)
    identity_predator[:, :, 0] = 1.0
    identity_prey = torch.zeros_like(prey_quaternion)
    identity_prey[:, 0] = 1.0

    predator_system_com_w_m = torch.stack(
        [_system_com_pos_w(predator) for predator in env._predators], dim=1
    )
    prey_system_com_w_m = _system_com_pos_w(env._prey)
    expected_offset = torch.tensor(
        ROOT_TO_SYSTEM_COM_B_M, dtype=torch.float32, device=env.device
    )
    predator_offset_error = (
        predator_system_com_w_m - predator_root_pos_w_m - expected_offset
    )
    prey_offset_error = prey_system_com_w_m - prey_root_pos_w_m - expected_offset

    previous_distance_error = torch.abs(
        env._prev_pred_prey_distances - predator_prey_distance_m
    )
    previous_min_error = torch.abs(
        env._prev_min_pred_prey_distance - predator_prey_distance_m.min(dim=1).values
    )
    previous_prey_radius_error = torch.abs(env._prev_prey_horiz - prey_xy_radius_m)
    maximum_velocity = torch.stack(
        (
            predator_linear_velocity.abs().max(),
            predator_angular_velocity.abs().max(),
            env._prey.data.root_lin_vel_w.abs().max(),
            env._prey.data.root_ang_vel_b.abs().max(),
        )
    ).max()
    maximum_action = torch.maximum(env._pred_actions.abs().max(), env._prey_actions.abs().max())
    maximum_offset_error = torch.maximum(
        predator_offset_error.abs().max(), prey_offset_error.abs().max()
    )

    sample_count = predator_local_m.shape[0]
    examples = min(example_count, sample_count)
    return {
        "seed": seed,
        "sample_count": sample_count,
        "invariants": {
            "predator_disk_bound_violation_count": int(
                (predator_xy_radius_m > disk_radius_m + READBACK_BOUND_TOLERANCE_M).sum()
            ),
            "prey_disk_bound_violation_count": int(
                (prey_xy_radius_m > disk_radius_m + READBACK_BOUND_TOLERANCE_M).sum()
            ),
            "predator_height_bound_violation_count": int(
                ((predator_local_m[:, :, 2] < env.cfg.random_spawn_z_min - READBACK_BOUND_TOLERANCE_M)
                 | (predator_local_m[:, :, 2] >= env.cfg.random_spawn_z_max + READBACK_BOUND_TOLERANCE_M)).sum()
            ),
            "prey_height_bound_violation_count": int(
                ((prey_local_m[:, 2] < env.cfg.random_spawn_z_min - READBACK_BOUND_TOLERANCE_M)
                 | (prey_local_m[:, 2] >= env.cfg.random_spawn_z_max + READBACK_BOUND_TOLERANCE_M)).sum()
            ),
            "remaining_invalid_layout_count": int(remaining_invalid.sum()),
            "maximum_quaternion_identity_error": float(
                torch.maximum(
                    (predator_quaternion - identity_predator).abs().max(),
                    (prey_quaternion - identity_prey).abs().max(),
                )
            ),
            "maximum_absolute_root_velocity": float(maximum_velocity),
            "non_alive_predator_count": int((~env._pred_alive).sum()),
            "nonzero_episode_step_count": int((env.episode_length_buf != 0).sum()),
            "maximum_absolute_reset_action": float(maximum_action),
            "maximum_previous_distance_error_m": float(previous_distance_error.max()),
            "maximum_previous_minimum_distance_error_m": float(previous_min_error.max()),
            "maximum_previous_prey_radius_error_m": float(previous_prey_radius_error.max()),
            "maximum_root_to_system_com_offset_error_m": float(maximum_offset_error),
        },
        "statistics": {
            "predator_x_m": _statistics(predator_local_m[:, :, 0]),
            "predator_y_m": _statistics(predator_local_m[:, :, 1]),
            "predator_xy_radius_m": _statistics(predator_xy_radius_m),
            "predator_xy_radius_squared_m2": _statistics(predator_xy_radius_m.square()),
            "predator_z_m": _statistics(predator_local_m[:, :, 2]),
            "prey_x_m": _statistics(prey_local_m[:, 0]),
            "prey_y_m": _statistics(prey_local_m[:, 1]),
            "prey_xy_radius_m": _statistics(prey_xy_radius_m),
            "prey_xy_radius_squared_m2": _statistics(prey_xy_radius_m.square()),
            "prey_z_m": _statistics(prey_local_m[:, 2]),
            "predator_prey_xy_distance_m": _statistics(predator_prey_xy_distance_m),
            "predator_prey_distance_m": _statistics(predator_prey_distance_m),
            "predator_pair_xy_distance_m": _statistics(predator_pair_xy_distance_m),
        },
        "examples": {
            "env_origin_w_m": _to_numpy(origins[:examples]).tolist(),
            "predator_root_link_pos_local_m": _to_numpy(predator_local_m[:examples]).tolist(),
            "prey_root_link_pos_local_m": _to_numpy(prey_local_m[:examples]).tolist(),
            "predator_system_com_pos_local_m": _to_numpy(
                predator_system_com_w_m[:examples] - origins[:examples, None, :]
            ).tolist(),
            "prey_system_com_pos_local_m": _to_numpy(
                prey_system_com_w_m[:examples] - origins[:examples]
            ).tolist(),
            "previous_predator_prey_distance_m": _to_numpy(
                env._prev_pred_prey_distances[:examples]
            ).tolist(),
        },
    }


def _case(
    env: Uav3v1Env,
    *,
    name: str,
    radius_fraction: float,
    seeds: list[int],
    example_count: int,
) -> dict[str, Any]:
    env.cfg.random_spawn_radius_fraction = radius_fraction
    disk_radius_m = env.cfg.arena_radius * radius_fraction
    return {
        "name": name,
        "parameter_overrides": {"random_spawn_radius_fraction": radius_fraction},
        "disk_radius_m": disk_radius_m,
        "replicates": [
            _read_reset(
                env,
                seed=seed,
                disk_radius_m=disk_radius_m,
                example_count=example_count,
            )
            for seed in seeds
        ],
    }


def main() -> None:
    cfg = Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seeds[0]
    env = Uav3v1Env(cfg=cfg)
    nominal_radius_fraction = float(env.cfg.random_spawn_radius_fraction)
    cases = [
        _case(
            env,
            name="nominal_target",
            radius_fraction=nominal_radius_fraction,
            seeds=args_cli.seeds,
            example_count=args_cli.examples,
        ),
        _case(
            env,
            name="forced_fallback_zero_radius",
            radius_fraction=0.0,
            seeds=args_cli.seeds,
            example_count=args_cli.examples,
        ),
    ]

    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_root = Path("/workspace/isaaclab")
    isaaclab_metadata = _git_metadata(isaaclab_root)
    if isaaclab_metadata["commit"] is None:
        isaaclab_metadata["commit"] = os.environ.get("ISAACLAB_COMMIT")
    output = {
        "schema_version": "uavpredatorprey.reset_oracle.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "producer": {
            "backend": "isaac_reset_readback",
            "task_id": TASK_ID,
            "uavpredatorprey": _git_metadata(repo_root),
            "isaaclab": isaaclab_metadata,
            "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
            "torch_version": torch.__version__,
        },
        "contract": {
            "predator_count": env._P,
            "spawn_position_point": "root_link",
            "jax_physical_position_point": "mass_weighted_five_link_system_com",
            "quaternion_order": "wxyz",
            "quaternion_semantics": "q_WB_body_to_world",
            "xy_distribution": "theta=2*pi*U; radius=R*sqrt(U)",
            "nominal_xy_radius_m": env.cfg.arena_radius * nominal_radius_fraction,
            "z_min_m": env.cfg.random_spawn_z_min,
            "z_max_m": env.cfg.random_spawn_z_max,
            "minimum_prey_predator_xy_distance_m": env.cfg.random_spawn_min_prey_predator_distance,
            "minimum_predator_predator_xy_distance_m": env.cfg.random_spawn_min_predator_distance,
            "maximum_resample_attempts": env.cfg.random_spawn_resample_attempts,
            "fallback": "accept final layout after attempt budget",
            "isaac_world_origin_readback_bound_tolerance_m": READBACK_BOUND_TOLERANCE_M,
            "initial_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "initial_root_velocity": "zero",
            "root_link_to_system_com_b_m": list(ROOT_TO_SYSTEM_COM_B_M),
            "previous_distance_metric": "full_3d_root_link_euclidean",
            "previous_prey_radius_metric": "xy_root_link_relative_to_environment_origin",
            "rng_parity_boundary": "equal integer seeds do not imply paired Torch/JAX samples",
        },
        "summary_spec": {
            "standard_deviation": "population",
            "quantiles": list(QUANTILES),
        },
        "cases": cases,
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
        f"[INFO] Wrote {len(cases)} reset cases x {len(args_cli.seeds)} seeds "
        f"x {args_cli.num_envs} environments to {output_path}"
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
