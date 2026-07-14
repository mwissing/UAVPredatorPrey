#!/usr/bin/env python3
"""Export controlled Isaac rotor-speed and low-torque pulse traces.

The fixture isolates the five-link Crazyflie articulation from policy,
reward, termination, and reset behavior.  It is a diagnostic boundary only:
it never trains and it does not change the environment configuration.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "uavpredatorprey.rotor_pulse_oracle.v0"
ARRAYS_FILENAME = "arrays.npz"
METADATA_FILENAME = "metadata.json"
CHECKSUMS_FILENAME = "SHA256SUMS.txt"
FIXTURE_FILENAMES = frozenset((ARRAYS_FILENAME, METADATA_FILENAME, CHECKSUMS_FILENAME))
DEFAULT_OUTPUT = (
    "/workspace/artifacts/transfer/rotor_pulse_v0/free_gyro_on_static_phase_seed42"
)

TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"
TARGET_ASSET = "predator_0"
TARGET_BODY = "body"
BODY_NAMES = ("body", "m1_prop", "m2_prop", "m3_prop", "m4_prop")
JOINT_NAMES = ("m1_joint", "m2_joint", "m3_joint", "m4_joint")
ACTION_ORDER = ("collective_thrust", "tau_x", "tau_y", "tau_z")
ROTOR_DIRECTION = (1.0, -1.0, 1.0, -1.0)
ROTOR_SPEED_SCALARS_RADPS = (0.0, 50.0, -50.0, 200.0, -200.0)
STATIC_ROTOR_PHASES = (
    ("phase_pi_over_4_rad", float(np.float32(math.pi / 4.0))),
    ("phase_pi_over_2_rad", float(np.float32(math.pi / 2.0))),
)
SCHEDULES = (
    ("idle", None, None, 0),
    ("pulse_x_pos", "x", 0, 1),
    ("pulse_x_neg", "x", 0, -1),
    ("pulse_y_pos", "y", 1, 1),
    ("pulse_y_neg", "y", 1, -1),
    ("pulse_z_pos", "z", 2, 1),
    ("pulse_z_neg", "z", 2, -1),
)
NORMALIZED_COLLECTIVE = -1.0
NORMALIZED_MOMENT_MAGNITUDE = 0.02
PHYSICS_STEPS = 10
PHYSICS_DT_S = 0.01
PULSE_STEP_RANGE = (0, 2)
COAST_STEP_RANGE = (2, 10)
ISAAC_REPEATS = 3
INITIAL_ROOT_POS_W_M = (-3.0, 0.0, 6.0)
INITIAL_ROOT_POSITIONS_W_M = (
    INITIAL_ROOT_POS_W_M,
    (0.0, -3.0, 6.0),
    (3.0, 0.0, 6.0),
    (0.0, 3.0, 6.0),
)
INITIAL_ROOT_QUAT_WB_WXYZ = (1.0, 0.0, 0.0, 0.0)

ARRAY_NAMES = (
    "action.executed_norm",
    "action.force_b_n",
    "action.torque_b_nm",
    "state.system_com_pos_w_m",
    "state.root_link_quat_wb_wxyz",
    "state.system_com_lin_vel_w_mps",
    "state.root_ang_vel_b_radps",
    "state.root_ang_vel_w_radps",
    "state.all_link_ang_vel_b_radps",
    "state.all_link_ang_vel_w_radps",
    "state.joint_pos_rad",
    "state.joint_vel_radps",
)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _case_specs() -> list[dict[str, Any]]:
    """Return the fixed 7 rotor conditions x 7 schedules in canonical order."""

    conditions: list[dict[str, Any]] = []
    for rotor_scalar in ROTOR_SPEED_SCALARS_RADPS:
        conditions.append(
            {
                "condition_kind": "rotor_speed",
                "condition_id": f"speed_{rotor_scalar:+g}_radps",
                "initial_rotor_phase_scalar_rad": 0.0,
                "rotor_speed_scalar_radps": float(rotor_scalar),
                "name_prefix": f"rotor_{rotor_scalar:+g}",
            }
        )
    for condition_id, phase_scalar in STATIC_ROTOR_PHASES:
        conditions.append(
            {
                "condition_kind": "static_rotor_phase",
                "condition_id": condition_id,
                "initial_rotor_phase_scalar_rad": phase_scalar,
                "rotor_speed_scalar_radps": 0.0,
                "name_prefix": condition_id,
            }
        )

    cases: list[dict[str, Any]] = []
    for condition in conditions:
        phase_scalar = float(condition["initial_rotor_phase_scalar_rad"])
        rotor_scalar = float(condition["rotor_speed_scalar_radps"])
        requested_joint_position = [
            float(np.float32(phase_scalar * direction)) for direction in ROTOR_DIRECTION
        ]
        requested_joint_velocity = [
            float(np.float32(rotor_scalar * direction)) for direction in ROTOR_DIRECTION
        ]
        for schedule_id, axis, axis_index, torque_sign in SCHEDULES:
            case_index = len(cases)
            cases.append(
                {
                    "case_index": case_index,
                    "name": f"{condition['name_prefix']}_{schedule_id}",
                    "condition_kind": condition["condition_kind"],
                    "condition_id": condition["condition_id"],
                    "schedule_id": schedule_id,
                    "axis": axis,
                    "axis_index": axis_index,
                    "torque_sign": torque_sign,
                    "initial_rotor_phase_scalar_rad": phase_scalar,
                    "rotor_speed_scalar_radps": float(rotor_scalar),
                    "requested_initial_joint_pos_rad": requested_joint_position.copy(),
                    "requested_initial_joint_vel_radps": requested_joint_velocity.copy(),
                    "pulse_step_range": list(PULSE_STEP_RANGE if axis is not None else (0, 0)),
                    "coast_step_range": list(COAST_STEP_RANGE if axis is not None else (0, PHYSICS_STEPS)),
                }
            )
    return cases


def _action_schedule(case: Mapping[str, Any]) -> np.ndarray:
    actions = np.zeros((PHYSICS_STEPS, 4), dtype=np.float32)
    actions[:, 0] = np.float32(NORMALIZED_COLLECTIVE)
    axis_index = case["axis_index"]
    if axis_index is not None:
        actions[PULSE_STEP_RANGE[0] : PULSE_STEP_RANGE[1], 1 + int(axis_index)] = (
            np.float32(NORMALIZED_MOMENT_MAGNITUDE * int(case["torque_sign"]))
        )
    return actions


def _expected_array_shapes() -> dict[str, tuple[int, ...]]:
    case_count = len(_case_specs())
    sample_count = PHYSICS_STEPS + 1
    return {
        "action.executed_norm": (case_count, PHYSICS_STEPS, 4),
        "action.force_b_n": (case_count, PHYSICS_STEPS, 3),
        "action.torque_b_nm": (case_count, PHYSICS_STEPS, 3),
        "state.system_com_pos_w_m": (ISAAC_REPEATS, case_count, sample_count, 3),
        "state.root_link_quat_wb_wxyz": (ISAAC_REPEATS, case_count, sample_count, 4),
        "state.system_com_lin_vel_w_mps": (ISAAC_REPEATS, case_count, sample_count, 3),
        "state.root_ang_vel_b_radps": (ISAAC_REPEATS, case_count, sample_count, 3),
        "state.root_ang_vel_w_radps": (ISAAC_REPEATS, case_count, sample_count, 3),
        "state.all_link_ang_vel_b_radps": (
            ISAAC_REPEATS,
            case_count,
            sample_count,
            len(BODY_NAMES),
            3,
        ),
        "state.all_link_ang_vel_w_radps": (
            ISAAC_REPEATS,
            case_count,
            sample_count,
            len(BODY_NAMES),
            3,
        ),
        "state.joint_pos_rad": (
            ISAAC_REPEATS,
            case_count,
            sample_count,
            len(JOINT_NAMES),
        ),
        "state.joint_vel_radps": (
            ISAAC_REPEATS,
            case_count,
            sample_count,
            len(JOINT_NAMES),
        ),
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = _expected_array_shapes()
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        unexpected = sorted(set(arrays) - set(expected))
        raise ValueError(f"rotor-pulse arrays mismatch: missing={missing}, unexpected={unexpected}")
    for name, shape in expected.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"rotor-pulse array {name!r} must be a NumPy array")
        if value.shape != shape:
            raise ValueError(
                f"rotor-pulse array {name!r} shape mismatch: expected {shape}, got {value.shape}"
            )
        if value.dtype != np.float32:
            raise ValueError(f"rotor-pulse array {name!r} must have dtype float32")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"rotor-pulse array {name!r} contains a non-finite value")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repository: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _isaaclab_metadata(repository: Path) -> dict[str, Any]:
    metadata = _git_metadata(repository)
    if metadata["commit"] is None:
        metadata["commit"] = os.environ.get("ISAACLAB_COMMIT")
        metadata["commit_source"] = "ISAACLAB_COMMIT" if metadata["commit"] else None
    return metadata


def _first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (IndexError, OSError, UnicodeError):
        return None


def _nvidia_driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (IndexError, OSError, subprocess.CalledProcessError):
        return None


def _containing_git_worktree(path: Path) -> Path | None:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        top_level = subprocess.check_output(
            ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(top_level).resolve()


def _write_json_fsynced(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_fsynced(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **{name: arrays[name] for name in sorted(arrays)})
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "atomic no-replace publication requires renameat2") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number, f"fixture destination already exists: {destination}", destination
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _write_fixture(
    destination_argument: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    repository: Path,
) -> Path:
    _validate_arrays(arrays)
    destination = destination_argument.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(f"fixture destination already exists: {destination}")
    destination = destination.resolve()
    repository_resolved = repository.resolve()
    if destination == repository_resolved or repository_resolved in destination.parents:
        raise ValueError("rotor-pulse fixtures are runtime artifacts and must be outside Git")
    containing_worktree = _containing_git_worktree(destination)
    if containing_worktree is not None:
        raise ValueError(
            "rotor-pulse fixtures are runtime artifacts and must be outside Git; "
            f"destination is inside {containing_worktree}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        _write_npz_fsynced(temporary / ARRAYS_FILENAME, arrays)
        _write_json_fsynced(temporary / METADATA_FILENAME, metadata)
        checksum_lines = [
            f"{_sha256_file(temporary / filename)}  {filename}"
            for filename in (ARRAYS_FILENAME, METADATA_FILENAME)
        ]
        checksum_path = temporary / CHECKSUMS_FILENAME
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(checksum_lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if {path.name for path in temporary.iterdir()} != FIXTURE_FILENAMES:
            raise RuntimeError("rotor-pulse staging directory has unexpected files")
        _fsync_directory(temporary)
        _publish_directory_no_replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _numpy(value: Any) -> np.ndarray:
    result = value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(result)


def _asset_sample(asset: Any) -> dict[str, np.ndarray]:
    from isaaclab.utils.math import quat_apply_inverse

    masses = asset.root_physx_view.get_masses()[0].to(asset.device)
    total_mass = masses.sum()
    system_com_pos_w = (masses[:, None] * asset.data.body_com_pos_w[0]).sum(dim=0) / total_mass
    system_com_lin_vel_w = (
        masses[:, None] * asset.data.body_com_lin_vel_w[0]
    ).sum(dim=0) / total_mass
    root_quat_w = asset.data.root_link_quat_w[0]
    root_ang_vel_w = asset.data.root_link_ang_vel_w[0]
    link_quat_w = asset.data.body_link_quat_w[0]
    link_ang_vel_w = asset.data.body_ang_vel_w[0]
    return {
        "system_com_pos_w_m": _numpy(system_com_pos_w),
        "root_link_quat_wb_wxyz": _numpy(root_quat_w),
        "system_com_lin_vel_w_mps": _numpy(system_com_lin_vel_w),
        "root_ang_vel_b_radps": _numpy(quat_apply_inverse(root_quat_w, root_ang_vel_w)),
        "root_ang_vel_w_radps": _numpy(root_ang_vel_w),
        "all_link_ang_vel_b_radps": _numpy(quat_apply_inverse(link_quat_w, link_ang_vel_w)),
        "all_link_ang_vel_w_radps": _numpy(link_ang_vel_w),
        "joint_pos_rad": _numpy(asset.data.joint_pos[0]),
        "joint_vel_radps": _numpy(asset.data.joint_vel[0]),
    }


def _validate_initial_joint_sample(
    case: Mapping[str, Any], sample: Mapping[str, np.ndarray]
) -> None:
    requested_fields = (
        ("joint_pos_rad", "requested_initial_joint_pos_rad"),
        ("joint_vel_radps", "requested_initial_joint_vel_radps"),
    )
    for sample_name, case_name in requested_fields:
        requested = np.asarray(case[case_name], dtype=np.float32)
        actual = sample[sample_name]
        if not np.array_equal(actual, requested):
            raise RuntimeError(
                f"case {case['case_index']} sample-0 {sample_name} does not exactly "
                f"match {case_name}: requested={requested.tolist()}, actual={actual.tolist()}"
            )


def _write_controlled_state(
    env: Any,
    case: Mapping[str, Any],
    *,
    root_positions_w_m: tuple[tuple[float, float, float], ...] = (
        INITIAL_ROOT_POSITIONS_W_M
    ),
) -> dict[str, np.ndarray]:
    import torch

    assets = [*env._predators, env._prey]
    if len(root_positions_w_m) != len(assets) or any(
        len(position) != 3 or not all(math.isfinite(float(value)) for value in position)
        for position in root_positions_w_m
    ):
        raise ValueError(
            "root_positions_w_m must contain one finite xyz world position per asset"
        )
    for asset_index, (asset, position) in enumerate(
        zip(assets, root_positions_w_m, strict=True)
    ):
        asset.reset()
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(position, dtype=torch.float32, device=env.device)
        root_state[:, 3:7] = torch.tensor(
            INITIAL_ROOT_QUAT_WB_WXYZ, dtype=torch.float32, device=env.device
        )
        root_state[:, 7:] = 0.0
        joint_pos = torch.zeros_like(asset.data.default_joint_pos)
        joint_vel = torch.zeros_like(asset.data.default_joint_vel)
        if asset_index == 0:
            joint_pos[:] = torch.tensor(
                case["requested_initial_joint_pos_rad"],
                dtype=torch.float32,
                device=env.device,
            )
            joint_vel[:] = torch.tensor(
                case["requested_initial_joint_vel_radps"],
                dtype=torch.float32,
                device=env.device,
            )
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(joint_pos, joint_vel)

    env.episode_length_buf.zero_()
    env._pred_alive.fill_(True)
    env._pred_oob.zero_()
    env._pred_newly_oob.zero_()
    env._prey_oob.zero_()
    env._intermediate_values_valid = False
    env.sim.forward()
    env.scene.update(env.sim.cfg.dt)
    return _asset_sample(env._predators[0])


def _action_dict(env: Any, target_action: np.ndarray) -> dict[str, Any]:
    import torch

    predator = torch.zeros((1, env.cfg.num_predators, 4), dtype=torch.float32, device=env.device)
    predator[:, :, 0] = NORMALIZED_COLLECTIVE
    predator[0, 0] = torch.from_numpy(target_action).to(env.device)
    prey = torch.tensor(
        [[NORMALIZED_COLLECTIVE, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device
    )
    return {"predator": predator.reshape(1, -1), "prey": prey}


def _run_case(env: Any, case: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    initial_sample = _write_controlled_state(env, case)
    _validate_initial_joint_sample(case, initial_sample)
    samples = [initial_sample]
    actions: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    schedule = _action_schedule(case)
    for physics_step in range(PHYSICS_STEPS):
        env._pre_physics_step(_action_dict(env, schedule[physics_step]))
        actions.append(_numpy(env._pred_actions[0, 0]))
        forces.append(_numpy(env._pred_thrust[0, 0, 0]))
        torques.append(_numpy(env._pred_moment[0, 0, 0]))
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.cfg.dt)
        samples.append(_asset_sample(env._predators[0]))

    states = {
        name: np.stack([sample[name] for sample in samples]).astype(np.float32, copy=False)
        for name in samples[0]
    }
    return (
        states,
        np.stack(actions).astype(np.float32, copy=False),
        np.stack(forces).astype(np.float32, copy=False),
        np.stack(torques).astype(np.float32, copy=False),
    )


def _runtime_rigid_body_metadata(asset: Any) -> list[dict[str, Any]]:
    from pxr import PhysxSchema
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    link_paths = list(asset.root_physx_view.link_paths[0])
    if len(link_paths) != len(asset.body_names):
        raise RuntimeError("runtime link paths do not match Crazyflie body names")
    records = []
    for body_name, link_path in zip(asset.body_names, link_paths, strict=True):
        api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_path)
        if not api:
            raise RuntimeError(f"missing PhysxRigidBodyAPI at {link_path}")
        damping_attr = api.GetAngularDampingAttr()
        cap_attr = api.GetMaxAngularVelocityAttr()
        gyro_attr = api.GetEnableGyroscopicForcesAttr()
        damping = damping_attr.Get()
        cap_degps = cap_attr.Get()
        gyro = gyro_attr.Get()
        if damping is None or cap_degps is None or gyro is None:
            raise RuntimeError(f"missing angular runtime properties at {link_path}")
        records.append(
            {
                "body_name": body_name,
                "prim_path": str(link_path),
                "gyroscopic_forces_enabled": bool(gyro),
                "gyroscopic_forces_authored": gyro_attr.HasAuthoredValueOpinion(),
                "angular_damping_s_inv": float(damping),
                "angular_damping_authored": damping_attr.HasAuthoredValueOpinion(),
                "max_angular_velocity_usd_degps": float(cap_degps),
                "max_angular_velocity_radps": math.radians(float(cap_degps)),
                "max_angular_velocity_authored": cap_attr.HasAuthoredValueOpinion(),
            }
        )
    return records


def _asset_metadata(env: Any) -> dict[str, Any]:
    asset = env._predators[0]
    runtime = _runtime_rigid_body_metadata(asset)
    gyro_flags = [record["gyroscopic_forces_enabled"] for record in runtime]
    if not all(gyro_flags):
        raise RuntimeError(
            "free_gyro_on fixture requires gyroscopic forces on for every Crazyflie link"
        )
    return {
        "usd": "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        "body_names": list(asset.body_names),
        "joint_names": list(asset.joint_names),
        "root_body_index": asset.body_names.index(TARGET_BODY),
        "link_paths": [str(path) for path in asset.root_physx_view.link_paths[0]],
        "link_masses_kg": _numpy(asset.root_physx_view.get_masses()[0]),
        "total_mass_kg": float(asset.root_physx_view.get_masses()[0].sum()),
        "link_inertia_b_kg_m2_row_major": _numpy(asset.root_physx_view.get_inertias()[0]),
        "default_joint_position_rad": _numpy(asset.data.default_joint_pos[0]),
        "default_joint_velocity_radps": _numpy(asset.data.default_joint_vel[0]),
        "runtime_rigid_body_properties": runtime,
        "gyroscopic_forces_enabled_per_link": gyro_flags,
        "robot_weight_n": float(env._robot_weight),
        "gravity_w_mps2": [float(value) for value in env.sim.cfg.gravity],
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _run(args: argparse.Namespace) -> Path:
    import torch
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env import Uav3v1Env
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env_cfg import (
        Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg,
    )

    repository = Path(__file__).resolve().parents[2]
    torch.manual_seed(args.seed)
    cfg = Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    env = Uav3v1Env(cfg=cfg)
    try:
        if not math.isclose(env.sim.cfg.dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"expected physics dt {PHYSICS_DT_S}, got {env.sim.cfg.dt}")
        asset = env._predators[0]
        if tuple(asset.body_names) != BODY_NAMES or tuple(asset.joint_names) != JOINT_NAMES:
            raise RuntimeError(
                f"unexpected Crazyflie articulation: bodies={asset.body_names}, joints={asset.joint_names}"
            )

        cases = _case_specs()
        repeat_states: list[list[dict[str, np.ndarray]]] = []
        reference_actions: list[np.ndarray] = []
        reference_forces: list[np.ndarray] = []
        reference_torques: list[np.ndarray] = []
        for repeat_index in range(ISAAC_REPEATS):
            case_states: list[dict[str, np.ndarray]] = []
            for case in cases:
                states, actions, forces, torques = _run_case(env, case)
                case_states.append(states)
                if repeat_index == 0:
                    reference_actions.append(actions)
                    reference_forces.append(forces)
                    reference_torques.append(torques)
                else:
                    case_index = int(case["case_index"])
                    for name, actual, expected in (
                        ("action", actions, reference_actions[case_index]),
                        ("force", forces, reference_forces[case_index]),
                        ("torque", torques, reference_torques[case_index]),
                    ):
                        if not np.array_equal(actual, expected):
                            raise RuntimeError(
                                f"{name} changed in repeat {repeat_index}, case {case_index}"
                            )
            repeat_states.append(case_states)

        arrays: dict[str, np.ndarray] = {
            "action.executed_norm": np.stack(reference_actions).astype(np.float32, copy=False),
            "action.force_b_n": np.stack(reference_forces).astype(np.float32, copy=False),
            "action.torque_b_nm": np.stack(reference_torques).astype(np.float32, copy=False),
        }
        for state_name in repeat_states[0][0]:
            arrays[f"state.{state_name}"] = np.stack(
                [
                    np.stack([case_state[state_name] for case_state in repeat])
                    for repeat in repeat_states
                ]
            ).astype(np.float32, copy=False)
        _validate_arrays(arrays)

        array_records = [
            {
                "dtype": str(arrays[name].dtype),
                "kind": "action" if name.startswith("action.") else "state",
                "name": name,
                "shape": list(arrays[name].shape),
            }
            for name in sorted(arrays)
        ]
        task_dir = repository / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/uavpredatorprey_3v1"
        isaaclab_root = Path("/workspace/isaaclab")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "producer": {
                "backend": "isaac_physx_rotor_pulse_probe",
                "task_id": TASK_ID,
                "uavpredatorprey": _git_metadata(repository),
                "isaaclab": _isaaclab_metadata(isaaclab_root),
                "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
                "isaac_image_id": os.environ.get("UAV_ISAAC_IMAGE_ID"),
                "isaac_sim_version": os.environ.get("ISAACSIM_VERSION"),
                "isaac_sim_build": _first_line(Path("/isaac-sim/VERSION")),
                "python_version": sys.version.split()[0],
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "nvidia_driver_version": _nvidia_driver_version(),
                "exporter_sha256": _sha256_file(Path(__file__).resolve()),
                "task_environment_sha256": _sha256_file(task_dir / "uav_3v1_env.py"),
                "task_config_sha256": _sha256_file(task_dir / "uav_3v1_env_cfg.py"),
                "seed": args.seed,
            },
            "contract": {
                "action_interface": "direct_wrench_v0",
                "action_order": list(ACTION_ORDER),
                "action_bounds": [-1.0, 1.0],
                "target_asset": TARGET_ASSET,
                "target_body": TARGET_BODY,
                "normalized_collective_thrust": NORMALIZED_COLLECTIVE,
                "normalized_moment_magnitude": NORMALIZED_MOMENT_MAGNITUDE,
                "moment_scale_nm": float(env.cfg.moment_scale),
                "physics_dt_s": float(env.sim.cfg.dt),
                "sample_dt_s": float(env.sim.cfg.dt),
                "physics_steps_per_case": PHYSICS_STEPS,
                "sample_count": PHYSICS_STEPS + 1,
                "isaac_repeats": ISAAC_REPEATS,
                "case_order": "condition_outer_schedule_inner",
                "rotor_direction_m1_to_m4": list(ROTOR_DIRECTION),
                "pulse_step_range_half_open": list(PULSE_STEP_RANGE),
                "coast_step_range_half_open": list(COAST_STEP_RANGE),
                "action_timing": "action[t] is processed and applied before state sample t+1",
                "policy_reward_termination_or_environment_reset_called": False,
                "controlled_articulation_state_restore_before_each_case": True,
                "quaternion_order": "wxyz",
                "quaternion_semantics": "q_WB_body_to_world",
                "translation_point": "mass_weighted_five_link_system_com",
                "orientation_source": "root_link",
                "linear_velocity_source": "mass_weighted_five_link_system_com_world",
                "root_angular_velocity_frames": ["body", "world"],
                "all_link_angular_velocity_frames": ["individual_link_body", "world"],
                "force_frame": "root_link_body",
                "torque_frame": "root_link_body",
            },
            "initial_state": {
                "requested_root_link_pos_w_m": list(INITIAL_ROOT_POS_W_M),
                "requested_root_link_quat_wb_wxyz": list(INITIAL_ROOT_QUAT_WB_WXYZ),
                "requested_root_link_lin_vel_w_mps": [0.0, 0.0, 0.0],
                "requested_root_link_ang_vel_w_radps": [0.0, 0.0, 0.0],
                "requested_joint_pos_rad_source": "cases[].requested_initial_joint_pos_rad",
                "requested_joint_vel_radps_source": "cases[].requested_initial_joint_vel_radps",
                "physx_sample_0_readback": "state.*[:, :, 0, ...]",
            },
            "asset": _json_compatible(_asset_metadata(env)),
            "case_count": len(cases),
            "trace_count": ISAAC_REPEATS,
            "physics_steps": PHYSICS_STEPS,
            "sample_count": PHYSICS_STEPS + 1,
            "array_count": len(array_records),
            "arrays": array_records,
            "cases": cases,
        }
        output = _write_fixture(args.output, arrays, metadata, repository=repository)
        print(f"[INFO] Wrote immutable rotor-pulse fixture: {output}")
        print(f"[INFO] {len(cases)} cases x {ISAAC_REPEATS} repeats x {PHYSICS_STEPS} steps")
        return output
    finally:
        env.close()


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    try:
        _run(args)
    finally:
        app_launcher.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
