"""Export deterministic Isaac observations for the JAX observation port.

The script writes root states and reads observations without taking a physics
step or invoking a policy. The artifact records the exact root quantities used
by the legacy task separately from whole-articulation COM diagnostics.
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


DEFAULT_OUTPUT = "/workspace/artifacts/transfer/observations_v0/isaac_oracle_v1.json"
TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Destination JSON path.")
    parser.add_argument("--repeats", type=int, default=3, help="Deterministic reads per case.")
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


def _root_inputs(asset) -> dict[str, list[float]]:
    """Read exactly the root quantities consumed by ``_get_observations``."""

    return {
        "root_link_pos_w_m": _tensor_list(asset.data.root_pos_w[0]),
        "root_link_quat_wb_wxyz": _tensor_list(asset.data.root_quat_w[0]),
        "root_com_lin_vel_w_mps": _tensor_list(asset.data.root_lin_vel_w[0]),
        "root_ang_vel_b_radps": _tensor_list(asset.data.root_ang_vel_b[0]),
        "projected_gravity_b": _tensor_list(asset.data.projected_gravity_b[0]),
    }


def _articulation_diagnostics(asset) -> dict[str, Any]:
    masses = asset.root_physx_view.get_masses()[0].to(asset.device)
    total_mass = masses.sum()
    system_com_pos_w = (masses[:, None] * asset.data.body_com_pos_w[0]).sum(dim=0) / total_mass
    system_com_lin_vel_w = (
        masses[:, None] * asset.data.body_com_lin_vel_w[0]
    ).sum(dim=0) / total_mass
    return {
        "root_body_com_pos_w_m": _tensor_list(asset.data.root_com_pos_w[0]),
        "root_body_com_lin_vel_w_mps": _tensor_list(asset.data.root_com_lin_vel_w[0]),
        "system_com_pos_w_m": _tensor_list(system_com_pos_w),
        "system_com_lin_vel_w_mps": _tensor_list(system_com_lin_vel_w),
        "root_to_system_com_w_m": _tensor_list(system_com_pos_w - asset.data.root_pos_w[0]),
    }


def _write_case_state(env: Uav3v1Env, case: dict[str, Any]) -> None:
    env._terrain.env_origins[:] = torch.tensor(
        case["env_origin_w_m"], dtype=torch.float32, device=env.device
    )
    assets = [*env._predators, env._prey]

    for asset, state_spec in zip(assets, case["root_states"], strict=True):
        asset.reset()
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(
            state_spec["root_link_pos_w_m"], dtype=torch.float32, device=env.device
        )
        root_state[:, 3:7] = torch.tensor(
            state_spec["root_link_quat_wb_wxyz"], dtype=torch.float32, device=env.device
        )
        root_state[:, 7:10] = torch.tensor(
            state_spec["root_com_lin_vel_w_mps"], dtype=torch.float32, device=env.device
        )
        root_state[:, 10:13] = torch.tensor(
            state_spec["root_ang_vel_w_radps"], dtype=torch.float32, device=env.device
        )
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel)

    env.sim.forward()
    # A positive timestamp invalidates Isaac Lab's cached body velocities after
    # write_root_velocity_to_sim without integrating the physical state.
    env.scene.update(env.sim.cfg.dt)

    requested_alive = torch.tensor(
        case["predator_alive"], dtype=torch.bool, device=env.device
    ).unsqueeze(0)
    env._pred_alive[:] = requested_alive
    env._pred_oob.zero_()
    env._pred_newly_oob.zero_()
    env._prey_oob.zero_()
    env._intermediate_values_valid = False
    env._compute_intermediate_values()
    if not torch.equal(env._pred_alive, requested_alive):
        raise RuntimeError(
            f"Case {case['name']!r} changed the requested alive mask; "
            "all fixture positions must remain above the crash threshold"
        )


def _read_case(env: Uav3v1Env, case: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    _write_case_state(env, case)
    observations = env._get_observations()
    centralized_state = env._get_states()
    assets = [*env._predators, env._prey]
    root_inputs = [_root_inputs(asset) for asset in assets]
    return {
        "repeat_index": repeat_index,
        "inputs": {
            "predators": root_inputs[: env._P],
            "prey": root_inputs[env._P],
            "env_origin_w_m": _tensor_list(env._terrain.env_origins[0]),
            "predator_alive": _tensor_list(env._pred_alive[0]),
        },
        "outputs": {
            "predator": _tensor_list(observations["predator"][0]),
            "prey": _tensor_list(observations["prey"][0]),
            "centralized_state": _tensor_list(centralized_state[0]),
        },
        "diagnostics": {
            "predators": [_articulation_diagnostics(asset) for asset in env._predators],
            "prey": _articulation_diagnostics(env._prey),
        },
    }


def _state(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    linear_velocity: tuple[float, float, float],
    angular_velocity_w: tuple[float, float, float],
) -> dict[str, list[float]]:
    return {
        "root_link_pos_w_m": list(position),
        "root_link_quat_wb_wxyz": list(quaternion),
        "root_com_lin_vel_w_mps": list(linear_velocity),
        "root_ang_vel_w_radps": list(angular_velocity_w),
    }


def _identity_states(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[dict[str, list[float]]]:
    ox, oy, oz = offset
    identity = (1.0, 0.0, 0.0, 0.0)
    return [
        _state((ox + 1.0, oy + 0.0, oz + 5.0), identity, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
        _state((ox + 3.0, oy + 4.0, oz + 6.0), identity, (4.0, 5.0, 6.0), (0.4, 0.5, 0.6)),
        _state((ox - 1.0, oy + 2.0, oz + 7.0), identity, (-1.0, -2.0, -3.0), (0.7, 0.8, 0.9)),
        _state((ox + 0.0, oy - 2.0, oz + 8.0), identity, (7.0, 8.0, 9.0), (-0.1, -0.2, -0.3)),
    ]


def _cases() -> list[dict[str, Any]]:
    translated_origin = (12.0, -7.0, 1.0)
    yaw_90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    pitch_30 = (math.cos(math.radians(15.0)), 0.0, math.sin(math.radians(15.0)), 0.0)
    roll_minus_45 = (
        math.cos(math.radians(22.5)),
        -math.sin(math.radians(22.5)),
        0.0,
        0.0,
    )
    yaw_minus_60 = (
        math.cos(math.radians(30.0)),
        0.0,
        0.0,
        -math.sin(math.radians(30.0)),
    )
    mixed_states = _identity_states()
    mixed_quaternions = (yaw_90, pitch_30, roll_minus_45, yaw_minus_60)
    mixed_states = [
        {**state, "root_link_quat_wb_wxyz": list(quaternion)}
        for state, quaternion in zip(mixed_states, mixed_quaternions, strict=True)
    ]
    return [
        {
            "name": "identity_origin_zero",
            "env_origin_w_m": [0.0, 0.0, 0.0],
            "predator_alive": [True, True, True],
            "root_states": _identity_states(),
        },
        {
            "name": "identity_translated_origin",
            "env_origin_w_m": list(translated_origin),
            "predator_alive": [True, True, True],
            "root_states": _identity_states(translated_origin),
        },
        {
            "name": "mixed_rotations",
            "env_origin_w_m": [0.0, 0.0, 0.0],
            "predator_alive": [True, True, True],
            "root_states": mixed_states,
        },
        {
            "name": "dead_middle_predator",
            "env_origin_w_m": [0.0, 0.0, 0.0],
            "predator_alive": [True, False, True],
            "root_states": _identity_states(),
        },
    ]


def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg = Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    env = Uav3v1Env(cfg=cfg)

    cases = []
    for case in _cases():
        cases.append(
            {
                "name": case["name"],
                "requested": {
                    "env_origin_w_m": case["env_origin_w_m"],
                    "predator_alive": case["predator_alive"],
                    "root_states": case["root_states"],
                },
                "repeats": [
                    _read_case(env, case, repeat_index)
                    for repeat_index in range(args_cli.repeats)
                ],
            }
        )

    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_root = Path("/workspace/isaaclab")
    isaaclab_metadata = _git_metadata(isaaclab_root)
    if isaaclab_metadata["commit"] is None:
        isaaclab_metadata["commit"] = os.environ.get("ISAACLAB_COMMIT")
    output = {
        "schema_version": "uavpredatorprey.observation_oracle.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "producer": {
            "backend": "isaac_physx_root_state_readback",
            "task_id": TASK_ID,
            "uavpredatorprey": _git_metadata(repo_root),
            "isaaclab": isaaclab_metadata,
            "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
            "torch_version": torch.__version__,
        },
        "contract": {
            "predator_count": env._P,
            "predator_observation_shape": [int(env.cfg.observation_spaces["predator"])],
            "prey_observation_shape": [int(env.cfg.observation_spaces["prey"])],
            "centralized_state_shape": [int(env.cfg.state_space)],
            "quaternion_order": "wxyz",
            "quaternion_semantics": "q_WB_body_to_world",
            "absolute_position_frame": "world_relative_to_environment_origin",
            "relative_position_frame": "observer_body",
            "relative_velocity_frame": "world",
            "gravity": "unit_world_down_projected_into_body",
            "dead_destination_position_sentinel_b_m": [2.0 * env.cfg.arena_radius, 0.0, 0.0],
            "centralized_state": "concat(predator_observation, prey_observation)",
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
    print(f"[INFO] Wrote {len(cases)} observation cases to {output_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
