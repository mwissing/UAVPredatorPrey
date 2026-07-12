"""Export deterministic Isaac traces for the JAX ``direct_wrench_v0`` backend.

This script launches the pinned Isaac runtime but does not train an agent. It
records a neutral JSON artifact containing system-center-of-mass translation,
root-link orientation, and root-link body angular velocity. The mixed state
definition is deliberate: Isaac's Crazyflie is a five-link articulation, while
the JAX milestone is a one-rigid-body approximation.
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


DEFAULT_OUTPUT = "/workspace/artifacts/transfer/direct_wrench_v0/isaac_oracle_v1.json"
TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"


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
from isaaclab.utils.math import quat_apply_inverse


PREDATOR_THRUST_TO_WEIGHT = 1.9
PREY_THRUST_TO_WEIGHT = 2.2
MOMENT_SCALE_NM = 0.01
POLICY_SUBSTEPS = 2


def _git_metadata(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
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


def _system_state(asset) -> dict[str, list[float]]:
    """Read the translation of the whole articulation and root attitude/rate."""

    masses = asset.root_physx_view.get_masses()[0].to(asset.device)
    total_mass = masses.sum()
    system_com_pos_w = (masses[:, None] * asset.data.body_com_pos_w[0]).sum(dim=0) / total_mass
    system_com_lin_vel_w = (
        masses[:, None] * asset.data.body_com_lin_vel_w[0]
    ).sum(dim=0) / total_mass
    root_link_quat_w = asset.data.root_link_quat_w[0]
    root_link_ang_vel_w = asset.data.root_link_ang_vel_w[0]
    return {
        "com_pos_w_m": _tensor_list(system_com_pos_w),
        "link_quat_wb_wxyz": _tensor_list(root_link_quat_w),
        "com_lin_vel_w_mps": _tensor_list(system_com_lin_vel_w),
        "ang_vel_b_radps": _tensor_list(quat_apply_inverse(root_link_quat_w, root_link_ang_vel_w)),
        "diagnostic_ang_vel_w_radps": _tensor_list(root_link_ang_vel_w),
    }


def _write_deterministic_state(env: Uav3v1Env, target_quat_wxyz: list[float]) -> None:
    assets = [*env._predators, env._prey]
    positions = ((-3.0, 0.0, 5.0), (0.0, -3.0, 5.0), (3.0, 0.0, 5.0), (0.0, 3.0, 5.0))

    for index, (asset, position) in enumerate(zip(assets, positions, strict=True)):
        asset.reset()
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(position, device=env.device)
        root_state[:, 3:7] = torch.tensor(
            target_quat_wxyz if index == 0 else (1.0, 0.0, 0.0, 0.0),
            device=env.device,
        )
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel)

    env._pred_alive.fill_(True)
    env._pred_oob.fill_(False)
    env._prey_oob.fill_(False)
    env._intermediate_values_valid = False
    env.sim.forward()
    # A positive data timestamp forces cached per-link velocities to be read
    # back after write_root_velocity_to_sim; a zero-dt update leaves them stale.
    env.scene.update(env.sim.cfg.dt)


def _actions(env: Uav3v1Env, target_action: list[float]) -> dict[str, torch.Tensor]:
    predator_hover = 2.0 / PREDATOR_THRUST_TO_WEIGHT - 1.0
    prey_hover = 2.0 / PREY_THRUST_TO_WEIGHT - 1.0
    predator = torch.tensor(
        [[[predator_hover, 0.0, 0.0, 0.0]] * env.cfg.num_predators],
        dtype=torch.float32,
        device=env.device,
    )
    predator[:, 0, :] = torch.tensor(target_action, dtype=torch.float32, device=env.device)
    prey = torch.tensor([[prey_hover, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)
    return {"predator": predator.reshape(1, -1), "prey": prey}


def _run_case(env: Uav3v1Env, case: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    _write_deterministic_state(env, case["initial_quat_wb_wxyz"])
    action_dict = _actions(env, case["action_norm"])
    samples = [{"physics_step": 0, "time_s": 0.0, "state": _system_state(env._predators[0])}]
    executed_action = None
    wrench = None

    for physics_step in range(1, case["physics_steps"] + 1):
        if (physics_step - 1) % POLICY_SUBSTEPS == 0:
            env._pre_physics_step(action_dict)
            executed_action = _tensor_list(env._pred_actions[0, 0])
            wrench = {
                "force_b_n": _tensor_list(env._pred_thrust[0, 0, 0]),
                "torque_b_nm": _tensor_list(env._pred_moment[0, 0, 0]),
            }

        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.cfg.dt)
        samples.append(
            {
                "physics_step": physics_step,
                "time_s": physics_step * env.sim.cfg.dt,
                "policy_step": (physics_step - 1) // POLICY_SUBSTEPS,
                "substep_in_policy": (physics_step - 1) % POLICY_SUBSTEPS,
                "state": _system_state(env._predators[0]),
            }
        )

    return {
        "repeat_index": repeat_index,
        "executed_action_norm": executed_action,
        "wrench_b": wrench,
        "samples": samples,
    }


def _asset_metadata(env: Uav3v1Env) -> dict[str, Any]:
    asset = env._predators[0]
    masses = asset.root_physx_view.get_masses()[0]
    inertias = asset.root_physx_view.get_inertias()[0]
    return {
        "usd": "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        "body_names": list(asset.body_names),
        "link_masses_kg": _tensor_list(masses),
        "total_mass_kg": float(masses.sum()),
        "link_inertia_b_kg_m2_row_major": _tensor_list(inertias),
        "default_joint_velocity_radps": _tensor_list(asset.data.default_joint_vel[0]),
        "gyroscopic_forces_enabled": True,
    }


def _cases() -> list[dict[str, Any]]:
    identity = [1.0, 0.0, 0.0, 0.0]
    tilt_y_30 = [math.cos(math.radians(15.0)), 0.0, math.sin(math.radians(15.0)), 0.0]
    predator_hover = 2.0 / PREDATOR_THRUST_TO_WEIGHT - 1.0
    return [
        {
            "name": "free_fall",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [-1.0, 0.0, 0.0, 0.0],
            "physics_steps": 10,
        },
        {
            "name": "hover",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [predator_hover, 0.0, 0.0, 0.0],
            "physics_steps": 10,
        },
        {
            "name": "maximum_thrust",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [1.0, 0.0, 0.0, 0.0],
            "physics_steps": 10,
        },
        {
            "name": "roll_moment",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [predator_hover, 0.1, 0.0, 0.0],
            "physics_steps": 4,
        },
        {
            "name": "pitch_moment",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [predator_hover, 0.0, 0.1, 0.0],
            "physics_steps": 4,
        },
        {
            "name": "yaw_moment",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [predator_hover, 0.0, 0.0, 0.1],
            "physics_steps": 4,
        },
        {
            "name": "zero_thrust_mixed_moment",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [-1.0, 0.2, -0.3, 0.4],
            "physics_steps": 2,
        },
        {
            "name": "tilted_hover_thrust",
            "initial_quat_wb_wxyz": tilt_y_30,
            "action_norm": [predator_hover, 0.0, 0.0, 0.0],
            "physics_steps": 10,
        },
        {
            "name": "tilted_mixed_wrench",
            "initial_quat_wb_wxyz": tilt_y_30,
            "action_norm": [0.25, 0.2, -0.3, 0.4],
            "physics_steps": 4,
        },
        {
            "name": "action_clipping",
            "initial_quat_wb_wxyz": identity,
            "action_norm": [2.0, -2.0, 0.5, 3.0],
            "physics_steps": 2,
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
                **case,
                "repeats": [_run_case(env, case, repeat) for repeat in range(args_cli.repeats)],
            }
        )

    repo_root = Path(__file__).resolve().parents[2]
    isaaclab_root = Path("/workspace/isaaclab")
    isaaclab_metadata = _git_metadata(isaaclab_root)
    if isaaclab_metadata["commit"] is None:
        isaaclab_metadata["commit"] = os.environ.get("ISAACLAB_COMMIT")
    output = {
        "schema_version": "uavpredatorprey.direct_wrench_oracle.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "producer": {
            "backend": "isaac_physx",
            "task_id": TASK_ID,
            "uavpredatorprey": _git_metadata(repo_root),
            "isaaclab": isaaclab_metadata,
            "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
            "torch_version": torch.__version__,
        },
        "contract": {
            "action_interface": "direct_wrench_v0",
            "action_order": ["collective_thrust", "tau_x", "tau_y", "tau_z"],
            "action_bounds": [-1.0, 1.0],
            "physics_dt_s": env.sim.cfg.dt,
            "policy_dt_s": env.sim.cfg.dt * env.cfg.decimation,
            "physics_substeps_per_policy_step": env.cfg.decimation,
            "quaternion_order": "wxyz",
            "quaternion_semantics": "q_WB_body_to_world",
            "translation_point": "mass_weighted_five_link_system_com",
            "orientation_source": "root_link",
            "angular_velocity_source": "root_link_com_expressed_in_body",
            "force_frame": "root_link_body",
            "torque_frame": "root_link_body",
        },
        "asset": _asset_metadata(env),
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
    print(f"[INFO] Wrote {len(cases)} oracle cases to {output_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
