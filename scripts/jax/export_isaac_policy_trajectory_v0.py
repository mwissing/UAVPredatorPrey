#!/usr/bin/env python3
"""Export a paired open-loop Isaac trajectory from the migrated policy.

The first repeat evaluates the verified recurrent skrl checkpoint at a fixed,
explicit state and records deterministic ``tanh(mean)`` actions. Two further
repeats restore the same articulation state and replay the exact post-clipping
action tape. State is sampled after every 10 ms physics substep so the sibling
JAX project can compare the plant without policy-feedback, reward, termination,
or auto-reset confounds.

This is a diagnostic exporter. It never trains or changes the environment.
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


SCHEMA_VERSION = "uavpredatorprey.policy_trajectory_oracle.v0"
ARRAYS_FILENAME = "arrays.npz"
METADATA_FILENAME = "metadata.json"
CHECKSUMS_FILENAME = "SHA256SUMS.txt"
FIXTURE_FILENAMES = frozenset((ARRAYS_FILENAME, METADATA_FILENAME, CHECKSUMS_FILENAME))

DEFAULT_CHECKPOINT = (
    ".pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "88e7161b48323452c5d302b25a8f50d1fd5ac026fdeddd2819d64eeb2521ba98"
)
DEFAULT_OUTPUT = (
    "/workspace/artifacts/transfer/policy_trajectory_v0/"
    "windows_agent115200_controlled_seed42"
)

TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"
ASSET_ORDER = ("predator_0", "predator_1", "predator_2", "prey")
ROLE_ORDER = ("predator", "predator", "predator", "prey")
ACTION_ORDER = ("collective_thrust", "tau_x", "tau_y", "tau_z")
INITIAL_ROOT_POS_W_M = (
    (-3.0, 0.0, 6.0),
    (0.0, -3.0, 6.0),
    (3.0, 0.0, 6.0),
    (0.0, 3.0, 6.0),
)
INITIAL_ROOT_QUAT_WB_WXYZ = (1.0, 0.0, 0.0, 0.0)
POLICY_STEPS = 50
ISAAC_REPEATS = 3
PHYSICS_SUBSTEPS = 2
PHYSICS_DT_S = 0.01
PREDATOR_OBSERVATION_DIM = 90
PREY_OBSERVATION_DIM = 30
PREDATOR_ACTION_DIM = 12
PREY_ACTION_DIM = 4
PREDATOR_COUNT = 3
ACTOR_HIDDEN_DIM = 256

ARRAY_NAMES = (
    "action.executed_norm",
    "action.force_b_n",
    "action.torque_b_nm",
    "state.system_com_pos_w_m",
    "state.root_link_quat_wb_wxyz",
    "state.system_com_lin_vel_w_mps",
    "state.root_ang_vel_b_radps",
    "state.root_link_pos_w_m",
    "state.root_body_com_lin_vel_w_mps",
    "state.joint_pos_rad",
    "state.joint_vel_radps",
)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_CHECKPOINT_SHA256,
        help="required SHA-256 identity of the exact source checkpoint",
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-steps", type=int, default=POLICY_STEPS)
    parser.add_argument("--repeats", type=int, default=ISAAC_REPEATS)
    return parser


def _expected_array_shapes(policy_steps: int, repeats: int) -> dict[str, tuple[int, ...]]:
    samples = policy_steps * PHYSICS_SUBSTEPS + 1
    return {
        "action.executed_norm": (policy_steps, 4, 4),
        "action.force_b_n": (policy_steps, 4, 3),
        "action.torque_b_nm": (policy_steps, 4, 3),
        "state.system_com_pos_w_m": (repeats, samples, 4, 3),
        "state.root_link_quat_wb_wxyz": (repeats, samples, 4, 4),
        "state.system_com_lin_vel_w_mps": (repeats, samples, 4, 3),
        "state.root_ang_vel_b_radps": (repeats, samples, 4, 3),
        "state.root_link_pos_w_m": (repeats, samples, 4, 3),
        "state.root_body_com_lin_vel_w_mps": (repeats, samples, 4, 3),
        "state.joint_pos_rad": (repeats, samples, 4, 4),
        "state.joint_vel_radps": (repeats, samples, 4, 4),
    }


def _validate_arrays(
    arrays: Mapping[str, np.ndarray], *, policy_steps: int, repeats: int
) -> None:
    expected = _expected_array_shapes(policy_steps, repeats)
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        unexpected = sorted(set(arrays) - set(expected))
        raise ValueError(f"trajectory arrays mismatch: missing={missing}, unexpected={unexpected}")
    for name, shape in expected.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"trajectory array {name!r} must be a NumPy array")
        if value.shape != shape:
            raise ValueError(
                f"trajectory array {name!r} shape mismatch: expected {shape}, got {value.shape}"
            )
        if value.dtype != np.float32:
            raise ValueError(f"trajectory array {name!r} must have dtype float32")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"trajectory array {name!r} contains a non-finite value")


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
    at_fdcwd = -100
    rename_noreplace = 1
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace fixture publication requires renameat2",
            destination,
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"fixture destination already exists: {destination}",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _write_fixture(
    destination_argument: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    repository: Path,
    policy_steps: int,
    repeats: int,
) -> Path:
    _validate_arrays(arrays, policy_steps=policy_steps, repeats=repeats)
    destination = destination_argument.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(f"fixture destination already exists: {destination}")
    destination = destination.resolve()
    repository_resolved = repository.resolve()
    if destination == repository_resolved or repository_resolved in destination.parents:
        raise ValueError("trajectory fixtures are runtime artifacts and must be outside Git")
    containing_worktree = _containing_git_worktree(destination)
    if containing_worktree is not None:
        raise ValueError(
            "trajectory fixtures are runtime artifacts and must be outside Git; "
            f"destination is inside {containing_worktree}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
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
            raise RuntimeError("trajectory fixture staging directory has unexpected files")
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
    import torch

    from isaaclab.utils.math import quat_apply_inverse

    masses = asset.root_physx_view.get_masses()[0].to(asset.device)
    total_mass = masses.sum()
    system_com_pos_w = (masses[:, None] * asset.data.body_com_pos_w[0]).sum(dim=0) / total_mass
    system_com_lin_vel_w = (
        masses[:, None] * asset.data.body_com_lin_vel_w[0]
    ).sum(dim=0) / total_mass
    root_index = asset.body_names.index("body")
    root_quat_w = asset.data.root_link_quat_w[0]
    root_ang_vel_w = asset.data.root_link_ang_vel_w[0]

    return {
        "system_com_pos_w_m": _numpy(system_com_pos_w),
        "root_link_quat_wb_wxyz": _numpy(root_quat_w),
        "system_com_lin_vel_w_mps": _numpy(system_com_lin_vel_w),
        "root_ang_vel_b_radps": _numpy(quat_apply_inverse(root_quat_w, root_ang_vel_w)),
        "root_link_pos_w_m": _numpy(asset.data.root_link_pos_w[0]),
        "root_body_com_lin_vel_w_mps": _numpy(asset.data.body_com_lin_vel_w[0, root_index]),
        "joint_pos_rad": _numpy(asset.data.joint_pos[0]),
        "joint_vel_radps": _numpy(asset.data.joint_vel[0]),
    }


def _scene_sample(assets: list[Any]) -> dict[str, np.ndarray]:
    samples = [_asset_sample(asset) for asset in assets]
    return {
        name: np.stack([sample[name] for sample in samples]).astype(np.float32, copy=False)
        for name in (
            "system_com_pos_w_m",
            "root_link_quat_wb_wxyz",
            "system_com_lin_vel_w_mps",
            "root_ang_vel_b_radps",
            "root_link_pos_w_m",
            "root_body_com_lin_vel_w_mps",
            "joint_pos_rad",
            "joint_vel_radps",
        )
    }


def _write_controlled_state(env: Any, assets: list[Any]) -> None:
    import torch

    for asset, position in zip(assets, INITIAL_ROOT_POS_W_M, strict=True):
        asset.reset()
        root_state = asset.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(position, dtype=torch.float32, device=env.device)
        root_state[:, 3:7] = torch.tensor(
            INITIAL_ROOT_QUAT_WB_WXYZ, dtype=torch.float32, device=env.device
        )
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7])
        asset.write_root_velocity_to_sim(root_state[:, 7:])
        asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel)

    env.episode_length_buf.zero_()
    env._pred_alive.fill_(True)
    env._pred_oob.zero_()
    env._pred_newly_oob.zero_()
    env._prey_oob.zero_()
    env._intermediate_values_valid = False
    env.sim.forward()
    env.scene.update(env.sim.cfg.dt)


def _combined_action_and_wrench(env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    actions = torch.cat((env._pred_actions[0], env._prey_actions[0].unsqueeze(0)), dim=0)
    forces = torch.cat((env._pred_thrust[0, :, 0], env._prey_thrust[0]), dim=0)
    torques = torch.cat((env._pred_moment[0, :, 0], env._prey_moment[0]), dim=0)
    return _numpy(actions), _numpy(forces), _numpy(torques)


def _policy_action(
    env: Any,
    actors: tuple[Any, Any],
    scalers: tuple[Mapping[str, Any], Mapping[str, Any]],
    hidden: tuple[Any, Any],
    policy_oracle: Any,
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    import torch

    observations = env._get_observations()
    predator_observation = observations["predator"].detach().cpu()
    prey_observation = observations["prey"].detach().cpu()
    predator_input = policy_oracle._normalize_exact_skrl(predator_observation, scalers[0])
    prey_input = policy_oracle._normalize_exact_skrl(prey_observation, scalers[1])

    with torch.inference_mode():
        predator_mean, _, predator_extra = actors[0].compute(
            {"states": predator_input, "rnn": [hidden[0]]}
        )
        prey_mean, _, prey_extra = actors[1].compute(
            {"states": prey_input, "rnn": [hidden[1]]}
        )
        predator_action = torch.tanh(predator_mean).to(env.device)
        prey_action = torch.tanh(prey_mean).to(env.device)

    next_hidden = (predator_extra["rnn"][0], prey_extra["rnn"][0])
    return {"predator": predator_action, "prey": prey_action}, next_hidden


def _assert_no_episode_boundary(env: Any, *, repeat_index: int, policy_step: int) -> None:
    env._intermediate_values_valid = False
    env._compute_intermediate_values()
    if not bool(env._pred_alive.all()):
        raise RuntimeError(
            f"predator became inactive in repeat {repeat_index} by policy step {policy_step}"
        )
    if bool(env._prey_oob.any()):
        raise RuntimeError(f"prey went OOB in repeat {repeat_index} by policy step {policy_step}")
    if bool(env._caught.any()):
        raise RuntimeError(f"catch occurred in repeat {repeat_index} by policy step {policy_step}")


def _run_repeat(
    env: Any,
    assets: list[Any],
    *,
    repeat_index: int,
    policy_steps: int,
    action_tape: np.ndarray | None,
    reference_force: np.ndarray | None,
    reference_torque: np.ndarray | None,
    actors: tuple[Any, Any],
    scalers: tuple[Mapping[str, Any], Mapping[str, Any]],
    policy_oracle: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    import torch

    _write_controlled_state(env, assets)
    samples = [_scene_sample(assets)]
    generated_actions: list[np.ndarray] = []
    generated_forces: list[np.ndarray] = []
    generated_torques: list[np.ndarray] = []
    hidden = (
        torch.zeros(1, 1, PREDATOR_COUNT * ACTOR_HIDDEN_DIM, dtype=torch.float32),
        torch.zeros(1, 1, ACTOR_HIDDEN_DIM, dtype=torch.float32),
    )

    for policy_step in range(policy_steps):
        if action_tape is None:
            action_dict, hidden = _policy_action(
                env, actors, scalers, hidden, policy_oracle
            )
        else:
            action = torch.from_numpy(action_tape[policy_step]).to(env.device)
            action_dict = {
                "predator": action[:PREDATOR_COUNT].reshape(1, PREDATOR_ACTION_DIM),
                "prey": action[PREDATOR_COUNT].reshape(1, PREY_ACTION_DIM),
            }

        env._pre_physics_step(action_dict)
        executed, force, torque = _combined_action_and_wrench(env)
        if action_tape is None:
            generated_actions.append(executed)
            generated_forces.append(force)
            generated_torques.append(torque)
        else:
            if not np.array_equal(executed, action_tape[policy_step]):
                raise RuntimeError(f"executed action changed during replay at step {policy_step}")
            if not np.array_equal(force, reference_force[policy_step]):
                raise RuntimeError(f"body force changed during replay at step {policy_step}")
            if not np.array_equal(torque, reference_torque[policy_step]):
                raise RuntimeError(f"body torque changed during replay at step {policy_step}")

        for _ in range(PHYSICS_SUBSTEPS):
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.sim.cfg.dt)
            samples.append(_scene_sample(assets))

        _assert_no_episode_boundary(
            env, repeat_index=repeat_index, policy_step=policy_step + 1
        )

    if action_tape is None:
        action_tape = np.stack(generated_actions).astype(np.float32, copy=False)
        reference_force = np.stack(generated_forces).astype(np.float32, copy=False)
        reference_torque = np.stack(generated_torques).astype(np.float32, copy=False)

    states = {
        name: np.stack([sample[name] for sample in samples]).astype(np.float32, copy=False)
        for name in samples[0]
    }
    return states, action_tape, reference_force, reference_torque


def _asset_metadata(env: Any, assets: list[Any]) -> dict[str, Any]:
    asset = assets[0]
    masses = asset.root_physx_view.get_masses()[0]
    inertias = asset.root_physx_view.get_inertias()[0]
    return {
        "usd": "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        "body_names": list(asset.body_names),
        "joint_names": list(asset.joint_names),
        "root_body_index": asset.body_names.index("body"),
        "link_masses_kg": _numpy(masses),
        "total_mass_kg": float(masses.sum()),
        "link_inertia_b_kg_m2_row_major": _numpy(inertias),
        "default_joint_position_rad": _numpy(asset.data.default_joint_pos[0]),
        "default_joint_velocity_radps": _numpy(asset.data.default_joint_vel[0]),
        "runtime_rigid_body_properties": _runtime_rigid_body_metadata(asset),
        "gyroscopic_forces_enabled": True,
        "vehicle_count": len(assets),
        "robot_weight_n": float(env._robot_weight),
        "gravity_w_mps2": [float(value) for value in env.sim.cfg.gravity],
    }


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
        damping = api.GetAngularDampingAttr().Get()
        max_velocity_degps = api.GetMaxAngularVelocityAttr().Get()
        if damping is None or max_velocity_degps is None:
            raise RuntimeError(f"missing angular runtime properties at {link_path}")
        records.append(
            {
                "body_name": body_name,
                "angular_damping_s_inv": float(damping),
                "max_angular_velocity_usd_degps": float(max_velocity_degps),
                "max_angular_velocity_radps": math.radians(float(max_velocity_degps)),
            }
        )
    return records


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

    import export_skrl_policy_oracle_v0 as policy_oracle
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env import Uav3v1Env
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env_cfg import (
        Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg,
    )

    if args.policy_steps <= 0:
        raise ValueError("--policy-steps must be positive")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 (capture plus open-loop replay)")

    repository = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = repository / checkpoint
    checkpoint_state, checkpoint_sha256, checkpoint_size = policy_oracle._safe_load_checkpoint(
        checkpoint, args.expected_checkpoint_sha256
    )
    attention_models_path = (
        repository
        / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/"
        "uavpredatorprey_3v1/agents/attention_models.py"
    )
    attention_models = policy_oracle._load_attention_models_module(attention_models_path)
    predator_actor, prey_actor, _, _ = policy_oracle._models(
        attention_models, checkpoint_state
    )
    predator_scaler = policy_oracle._scaler_state(
        checkpoint_state,
        role="predator",
        name="state_preprocessor",
        feature_dim=PREDATOR_OBSERVATION_DIM,
    )
    prey_scaler = policy_oracle._scaler_state(
        checkpoint_state,
        role="prey",
        name="state_preprocessor",
        feature_dim=PREY_OBSERVATION_DIM,
    )

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    cfg = Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    env = Uav3v1Env(cfg=cfg)
    try:
        if not math.isclose(env.sim.cfg.dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"expected physics dt {PHYSICS_DT_S}, got {env.sim.cfg.dt}")
        if env.cfg.decimation != PHYSICS_SUBSTEPS:
            raise RuntimeError(
                f"expected {PHYSICS_SUBSTEPS} substeps, got {env.cfg.decimation}"
            )
        expected_spaces = {
            "predator": PREDATOR_OBSERVATION_DIM,
            "prey": PREY_OBSERVATION_DIM,
        }
        if dict(env.cfg.observation_spaces) != expected_spaces:
            raise RuntimeError(f"unexpected observation contract: {env.cfg.observation_spaces}")
        if dict(env.cfg.action_spaces) != {
            "predator": PREDATOR_ACTION_DIM,
            "prey": PREY_ACTION_DIM,
        }:
            raise RuntimeError(f"unexpected action contract: {env.cfg.action_spaces}")

        assets = [*env._predators, env._prey]
        repeat_states: list[dict[str, np.ndarray]] = []
        action_tape = None
        force_tape = None
        torque_tape = None
        for repeat_index in range(args.repeats):
            states, action_tape, force_tape, torque_tape = _run_repeat(
                env,
                assets,
                repeat_index=repeat_index,
                policy_steps=args.policy_steps,
                action_tape=action_tape,
                reference_force=force_tape,
                reference_torque=torque_tape,
                actors=(predator_actor, prey_actor),
                scalers=(predator_scaler, prey_scaler),
                policy_oracle=policy_oracle,
            )
            repeat_states.append(states)

        arrays: dict[str, np.ndarray] = {
            "action.executed_norm": action_tape,
            "action.force_b_n": force_tape,
            "action.torque_b_nm": torque_tape,
        }
        for state_name in repeat_states[0]:
            arrays[f"state.{state_name}"] = np.stack(
                [states[state_name] for states in repeat_states]
            ).astype(np.float32, copy=False)
        _validate_arrays(arrays, policy_steps=args.policy_steps, repeats=args.repeats)

        initial_readback = {
            name.removeprefix("state."): arrays[name][0, 0]
            for name in arrays
            if name.startswith("state.")
        }
        array_records = [
            {
                "dtype": str(arrays[name].dtype),
                "kind": "action" if name.startswith("action.") else "state",
                "name": name,
                "shape": list(arrays[name].shape),
            }
            for name in sorted(arrays)
        ]
        isaaclab_root = Path("/workspace/isaaclab")
        task_environment_path = (
            repository
            / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/"
            "uavpredatorprey_3v1/uav_3v1_env.py"
        )
        task_config_path = (
            repository
            / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/"
            "uavpredatorprey_3v1/uav_3v1_env_cfg.py"
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "producer": {
                "backend": "isaac_physx_policy_capture_and_open_loop_replay",
                "task_id": TASK_ID,
                "uavpredatorprey": _git_metadata(repository),
                "isaaclab": _isaaclab_metadata(isaaclab_root),
                "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
                "isaac_image_id": os.environ.get("UAV_ISAAC_IMAGE_ID"),
                "isaac_sim_version": os.environ.get("ISAACSIM_VERSION"),
                "isaac_sim_build": _first_line(Path("/isaac-sim/VERSION")),
                "skrl_version": str(policy_oracle.skrl.__version__),
                "python_version": sys.version.split()[0],
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "nvidia_driver_version": _nvidia_driver_version(),
                "exporter_sha256": _sha256_file(Path(__file__).resolve()),
                "attention_models_sha256": _sha256_file(attention_models_path),
                "task_environment_sha256": _sha256_file(task_environment_path),
                "task_config_sha256": _sha256_file(task_config_path),
                "seed": args.seed,
            },
            "source_checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_size,
                "load_policy": "verified bytes; torch.load(weights_only=True,map_location=cpu)",
                "architecture": "recurrent_entity_attention_v0_large_gru",
                "deterministic_action": "tanh(pre_tanh_latent_mean)",
                "policy_compute_device": "cpu",
                "policy_forward_validation": (
                    "separate checksummed skrl_policy_forward_oracle.v0 binds the same SHA-256"
                ),
            },
            "contract": {
                "action_interface": "direct_wrench_v0",
                "action_mode": "deterministic_tanh_latent_mean",
                "action_order": list(ACTION_ORDER),
                "action_bounds": [-1.0, 1.0],
                "asset_order": list(ASSET_ORDER),
                "role_order": list(ROLE_ORDER),
                "thrust_to_weight": [
                    float(env.cfg.predator_thrust_to_weight),
                    float(env.cfg.predator_thrust_to_weight),
                    float(env.cfg.predator_thrust_to_weight),
                    float(env.cfg.prey_thrust_to_weight),
                ],
                "moment_scale_nm": float(env.cfg.moment_scale),
                "physics_dt_s": float(env.sim.cfg.dt),
                "sample_dt_s": float(env.sim.cfg.dt),
                "policy_dt_s": float(env.sim.cfg.dt * env.cfg.decimation),
                "physics_substeps_per_policy_step": int(env.cfg.decimation),
                "policy_steps": args.policy_steps,
                "sample_count": args.policy_steps * PHYSICS_SUBSTEPS + 1,
                "isaac_repeats": args.repeats,
                "repeat_mode": "repeat_0_closed_loop_capture_repeats_1_plus_open_loop_replay",
                "repeat_semantics": [
                    "repeat_0_policy_capture_and_executed_tape",
                    *["exact_executed_action_open_loop_replay"] * (args.repeats - 1),
                ],
                "policy_initial_gru": "zero",
                "initial_actor_recurrent_state": "zero",
                "policy_feedback_in_jax_replay": False,
                "termination_reward_reset_called": False,
                "termination_or_reset": "none",
                "all_agents_required_alive": True,
                "all_agents_alive": True,
                "quaternion_order": "wxyz",
                "quaternion_semantics": "q_WB_body_to_world",
                "translation_point": "mass_weighted_five_link_system_com",
                "orientation_source": "root_link",
                "linear_velocity_source": "mass_weighted_five_link_system_com_world",
                "angular_velocity_source": "root_link_expressed_in_body",
                "force_frame": "root_link_body",
                "torque_frame": "root_link_body",
                "action_zero_order_hold": "action[t] applies to samples 2*t+1 and 2*t+2",
                "policy_horizon_sample_indices": {
                    str(horizon): PHYSICS_SUBSTEPS * horizon
                    for horizon in (1, 2, 5, 10, 25, 50)
                    if horizon <= args.policy_steps
                },
            },
            "initial_state": {
                "requested_root_link_pos_w_m": [list(value) for value in INITIAL_ROOT_POS_W_M],
                "requested_root_link_quat_wb_wxyz": [
                    list(INITIAL_ROOT_QUAT_WB_WXYZ) for _ in ASSET_ORDER
                ],
                "requested_root_body_com_lin_vel_w_mps": [
                    [0.0, 0.0, 0.0] for _ in ASSET_ORDER
                ],
                "requested_root_body_com_ang_vel_w_radps": [
                    [0.0, 0.0, 0.0] for _ in ASSET_ORDER
                ],
                "physx_readback_repeat_0_sample_0": _json_compatible(initial_readback),
            },
            "asset": _json_compatible(_asset_metadata(env, assets)),
            "trace_count": args.repeats,
            "policy_steps": args.policy_steps,
            "sample_count": args.policy_steps * PHYSICS_SUBSTEPS + 1,
            "array_count": len(array_records),
            "arrays": array_records,
        }
        output = _write_fixture(
            args.output,
            arrays,
            metadata,
            repository=repository,
            policy_steps=args.policy_steps,
            repeats=args.repeats,
        )
        print(f"[INFO] Wrote immutable paired trajectory fixture: {output}")
        print(f"[INFO] Source checkpoint SHA-256: {checkpoint_sha256}")
        print(
            f"[INFO] {args.policy_steps} policy steps, "
            f"{args.policy_steps * PHYSICS_SUBSTEPS + 1} samples/repeat, "
            f"{args.repeats} Isaac repeats"
        )
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
