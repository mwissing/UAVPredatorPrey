#!/usr/bin/env python3
"""Export a held-out Isaac rotor-phase sweep under an analytic action tape.

This diagnostic holds one deterministic policy action for two 10 ms physics
steps and records a one-second aggressive multi-axis trajectory at each of 16
midpoint rotor phases. It does not call policy, reward, termination, reset, or
training code. The existing rotor-pulse exporter remains the calibration
fixture; this independent schema is a held-out validation boundary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "uavpredatorprey.rotor_phase_sweep_oracle.v0"
DEFAULT_OUTPUT = (
    "/workspace/artifacts/transfer/rotor_phase_sweep_v0/"
    "midpoint16_aggressive_high_z10_seed42"
)

PHASE_COUNT = 16
POLICY_ACTION_COUNT = 50
PHYSICS_SUBSTEPS_PER_POLICY_ACTION = 2
PHYSICS_STEPS = POLICY_ACTION_COUNT * PHYSICS_SUBSTEPS_PER_POLICY_ACTION
PHYSICS_DT_S = 0.01
POLICY_DT_S = PHYSICS_DT_S * PHYSICS_SUBSTEPS_PER_POLICY_ACTION
ISAAC_REPEATS = 3
ROTOR_SPEED_SCALAR_RADPS = 200.0
ACTION_SCHEDULE_ID = "analytic_aggressive_multiaxis_v0"
CASE_ORDER = "held_out_phase_midpoint_index_ascending"
PHASE_GRID_RULE = "float32((k + 0.5) * pi / 16), k = 0, ..., 15"
ACTION_FORMULA_BY_COMPONENT = {
    "collective_thrust": "0.05 + 0.75 * sin(2*pi*(n + 0.5)/19)",
    "tau_x": "0.95 * sin(2*pi*(n + 0.5)/11)",
    "tau_y": "0.90 * cos(2*pi*(n + 0.5)/13)",
    "tau_z": "0.85 * sin(2*pi*(n + 0.5)/17 + pi/7)",
}
ACTION_CAST_ORDER = (
    "evaluate every component in float64, stack [n, 4], cast once to "
    "float32, repeat each policy row twice"
)
ACTION_TAPE_SHA256_ENCODING = "C-contiguous float32 raw bytes"
ASSET_NAMES = ("predator_0", "predator_1", "predator_2", "prey")
INITIAL_ROOT_POSITIONS_W_M = (
    (-3.0, 0.0, 10.0),
    (0.0, -3.0, 10.0),
    (3.0, 0.0, 10.0),
    (0.0, 3.0, 10.0),
)
MINIMUM_SYSTEM_COM_Z_M = 1.0


def _load_rotor_pulse_helper() -> ModuleType:
    """Load the import-safe shared Isaac fixture helpers without Isaac imports."""

    helper_path = Path(__file__).with_name("export_isaac_rotor_pulse_oracle_v0.py")
    specification = importlib.util.spec_from_file_location(
        "uavpredatorprey_rotor_pulse_export_helper", helper_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load rotor-pulse helper: {helper_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_PULSE = _load_rotor_pulse_helper()

ARRAYS_FILENAME = _PULSE.ARRAYS_FILENAME
METADATA_FILENAME = _PULSE.METADATA_FILENAME
CHECKSUMS_FILENAME = _PULSE.CHECKSUMS_FILENAME
FIXTURE_FILENAMES = _PULSE.FIXTURE_FILENAMES
TASK_ID = _PULSE.TASK_ID
TARGET_ASSET = _PULSE.TARGET_ASSET
TARGET_BODY = _PULSE.TARGET_BODY
BODY_NAMES = _PULSE.BODY_NAMES
JOINT_NAMES = _PULSE.JOINT_NAMES
ACTION_ORDER = _PULSE.ACTION_ORDER
ROTOR_DIRECTION = _PULSE.ROTOR_DIRECTION
INITIAL_ROOT_POS_W_M = INITIAL_ROOT_POSITIONS_W_M[0]
INITIAL_ROOT_QUAT_WB_WXYZ = _PULSE.INITIAL_ROOT_QUAT_WB_WXYZ
ARRAY_NAMES = _PULSE.ARRAY_NAMES


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _phase_scalars_rad() -> np.ndarray:
    indices = np.arange(PHASE_COUNT, dtype=np.float64)
    phases = (indices + 0.5) * np.pi / float(PHASE_COUNT)
    return np.ascontiguousarray(phases.astype(np.float32))


def _case_specs() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    requested_velocity = [
        float(np.float32(ROTOR_SPEED_SCALAR_RADPS * direction))
        for direction in ROTOR_DIRECTION
    ]
    for phase_index, phase_value in enumerate(_phase_scalars_rad()):
        phase = float(phase_value)
        requested_position = [
            float(np.float32(phase * direction)) for direction in ROTOR_DIRECTION
        ]
        cases.append(
            {
                "case_index": phase_index,
                "name": f"held_out_phase_midpoint_{phase_index:02d}",
                "condition_kind": "held_out_rotor_phase",
                "condition_id": f"midpoint_{phase_index:02d}",
                "phase_index": phase_index,
                "phase_grid_numerator": 2 * phase_index + 1,
                "phase_grid_denominator": 2 * PHASE_COUNT,
                "initial_rotor_phase_scalar_rad": phase,
                "rotor_speed_scalar_radps": ROTOR_SPEED_SCALAR_RADPS,
                "requested_initial_joint_pos_rad": requested_position,
                "requested_initial_joint_vel_radps": requested_velocity.copy(),
            }
        )
    return cases


def _policy_action_tape() -> np.ndarray:
    n = np.arange(POLICY_ACTION_COUNT, dtype=np.float64)
    centered_index = n + 0.5
    tape_float64 = np.stack(
        (
            0.05 + 0.75 * np.sin(2.0 * np.pi * centered_index / 19.0),
            0.95 * np.sin(2.0 * np.pi * centered_index / 11.0),
            0.90 * np.cos(2.0 * np.pi * centered_index / 13.0),
            0.85
            * np.sin(
                2.0 * np.pi * centered_index / 17.0 + np.pi / 7.0
            ),
        ),
        axis=-1,
    )
    tape = np.ascontiguousarray(tape_float64.astype(np.float32))
    if tape.shape != (POLICY_ACTION_COUNT, len(ACTION_ORDER)):
        raise AssertionError(f"unexpected policy action shape: {tape.shape}")
    if not np.all(np.isfinite(tape)) or np.any(np.abs(tape) > np.float32(1.0)):
        raise AssertionError("analytic policy action tape violates normalized bounds")
    return tape


def _physics_action_tape() -> np.ndarray:
    return np.ascontiguousarray(
        np.repeat(
            _policy_action_tape(),
            PHYSICS_SUBSTEPS_PER_POLICY_ACTION,
            axis=0,
        )
    )


def _float32_raw_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype != np.float32:
        raise TypeError("action-tape hashes require float32 arrays")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _action_schedule_contract() -> dict[str, Any]:
    policy_tape = _policy_action_tape()
    physics_tape = _physics_action_tape()
    return {
        "action_schedule_id": ACTION_SCHEDULE_ID,
        "action_schedule_formula_by_component": dict(ACTION_FORMULA_BY_COMPONENT),
        "action_schedule_index": "n = 0, ..., 49",
        "action_schedule_formula_evaluation_dtype": "float64",
        "action_schedule_storage_dtype": "float32",
        "action_schedule_cast_order": ACTION_CAST_ORDER,
        "action_tape_policy_shape": list(policy_tape.shape),
        "action_tape_physics_shape": list(physics_tape.shape),
        "action_tape_policy_sha256": _float32_raw_sha256(policy_tape),
        "action_tape_physics_sha256": _float32_raw_sha256(physics_tape),
        "action_tape_sha256_encoding": ACTION_TAPE_SHA256_ENCODING,
        "actions_identical_across_cases": True,
        "zero_order_hold_physics_steps": PHYSICS_SUBSTEPS_PER_POLICY_ACTION,
    }


def _initial_state_contract() -> dict[str, Any]:
    return {
        "requested_root_link_pos_w_m": list(INITIAL_ROOT_POS_W_M),
        "requested_root_link_pos_w_m_by_asset": {
            asset_name: list(position)
            for asset_name, position in zip(
                ASSET_NAMES, INITIAL_ROOT_POSITIONS_W_M, strict=True
            )
        },
        "requested_root_link_position_frame": "world",
        "requested_root_link_quat_wb_wxyz": list(
            INITIAL_ROOT_QUAT_WB_WXYZ
        ),
        "requested_root_link_lin_vel_w_mps": [0.0, 0.0, 0.0],
        "requested_root_link_ang_vel_w_radps": [0.0, 0.0, 0.0],
        "requested_joint_pos_rad_source": (
            "cases[].requested_initial_joint_pos_rad"
        ),
        "requested_joint_vel_radps_source": (
            "cases[].requested_initial_joint_vel_radps"
        ),
        "physx_sample_0_readback": "state.*[:, :, 0, ...]",
    }


def _expected_array_shapes() -> dict[str, tuple[int, ...]]:
    sample_count = PHYSICS_STEPS + 1
    return {
        "action.executed_norm": (PHASE_COUNT, PHYSICS_STEPS, 4),
        "action.force_b_n": (PHASE_COUNT, PHYSICS_STEPS, 3),
        "action.torque_b_nm": (PHASE_COUNT, PHYSICS_STEPS, 3),
        "state.system_com_pos_w_m": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            3,
        ),
        "state.root_link_quat_wb_wxyz": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            4,
        ),
        "state.system_com_lin_vel_w_mps": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            3,
        ),
        "state.root_ang_vel_b_radps": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            3,
        ),
        "state.root_ang_vel_w_radps": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            3,
        ),
        "state.all_link_ang_vel_b_radps": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            len(BODY_NAMES),
            3,
        ),
        "state.all_link_ang_vel_w_radps": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            len(BODY_NAMES),
            3,
        ),
        "state.joint_pos_rad": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            len(JOINT_NAMES),
        ),
        "state.joint_vel_radps": (
            ISAAC_REPEATS,
            PHASE_COUNT,
            sample_count,
            len(JOINT_NAMES),
        ),
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = _expected_array_shapes()
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        unexpected = sorted(set(arrays) - set(expected))
        raise ValueError(
            "rotor-phase-sweep arrays mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, shape in expected.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"rotor-phase-sweep array {name!r} must be a NumPy array")
        if value.shape != shape:
            raise ValueError(
                f"rotor-phase-sweep array {name!r} shape mismatch: "
                f"expected {shape}, got {value.shape}"
            )
        if value.dtype != np.float32:
            raise ValueError(f"rotor-phase-sweep array {name!r} must have dtype float32")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"rotor-phase-sweep array {name!r} contains a non-finite value")


def _minimum_system_com_height_validation(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    positions = arrays["state.system_com_pos_w_m"]
    system_com_z = positions[..., 2]
    minimum_flat_index = int(np.argmin(system_com_z))
    minimum_index = np.unravel_index(minimum_flat_index, system_com_z.shape)
    observed_minimum = float(system_com_z[minimum_index])
    record = {
        "check": "minimum_recorded_system_com_world_z",
        "state_array": "state.system_com_pos_w_m",
        "coordinate_frame": "world",
        "axis": "z",
        "unit": "m",
        "required_relation": "every recorded system COM z >= minimum_system_com_z_m",
        "minimum_system_com_z_m": MINIMUM_SYSTEM_COM_Z_M,
        "observed_minimum_system_com_z_m": observed_minimum,
        "observed_minimum_location": {
            "repeat_index": int(minimum_index[0]),
            "case_index": int(minimum_index[1]),
            "sample_index": int(minimum_index[2]),
        },
        "recorded_value_count": int(system_com_z.size),
        "passed": observed_minimum >= MINIMUM_SYSTEM_COM_Z_M,
    }
    if not record["passed"]:
        raise ValueError(
            "rotor-phase-sweep fixture publication refused: minimum recorded "
            f"system COM world z is {observed_minimum:.9g} m, below required "
            f"{MINIMUM_SYSTEM_COM_Z_M:.9g} m at "
            f"repeat={minimum_index[0]}, case={minimum_index[1]}, "
            f"sample={minimum_index[2]}"
        )
    return record


def _write_fixture(
    destination_argument: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    repository: Path,
) -> Path:
    _validate_arrays(arrays)
    height_validation = _minimum_system_com_height_validation(arrays)
    metadata_to_write = dict(metadata)
    validation = metadata_to_write.get("validation", {})
    if not isinstance(validation, Mapping):
        raise TypeError("rotor-phase-sweep metadata validation must be a mapping")
    metadata_to_write["validation"] = {
        **validation,
        "contact_free_height_guard": height_validation,
    }
    destination = destination_argument.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(f"fixture destination already exists: {destination}")
    destination = destination.resolve()
    repository_resolved = repository.resolve()
    if destination == repository_resolved or repository_resolved in destination.parents:
        raise ValueError(
            "rotor-phase-sweep fixtures are runtime artifacts and must be outside Git"
        )
    containing_worktree = _PULSE._containing_git_worktree(destination)
    if containing_worktree is not None:
        raise ValueError(
            "rotor-phase-sweep fixtures are runtime artifacts and must be outside Git; "
            f"destination is inside {containing_worktree}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _PULSE._write_npz_fsynced(temporary / ARRAYS_FILENAME, arrays)
        _PULSE._write_json_fsynced(
            temporary / METADATA_FILENAME, metadata_to_write
        )
        checksum_lines = [
            f"{_PULSE._sha256_file(temporary / filename)}  {filename}"
            for filename in (ARRAYS_FILENAME, METADATA_FILENAME)
        ]
        checksum_path = temporary / CHECKSUMS_FILENAME
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(checksum_lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if {path.name for path in temporary.iterdir()} != FIXTURE_FILENAMES:
            raise RuntimeError("rotor-phase-sweep staging directory has unexpected files")
        _PULSE._fsync_directory(temporary)
        _PULSE._publish_directory_no_replace(temporary, destination)
        _PULSE._fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _run_case(
    env: Any,
    case: Mapping[str, Any],
    physics_action_tape: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    initial_sample = _PULSE._write_controlled_state(
        env,
        case,
        root_positions_w_m=INITIAL_ROOT_POSITIONS_W_M,
    )
    _PULSE._validate_initial_joint_sample(case, initial_sample)
    samples = [initial_sample]
    actions: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    for target_action in physics_action_tape:
        env._pre_physics_step(_PULSE._action_dict(env, target_action))
        actions.append(_PULSE._numpy(env._pred_actions[0, 0]))
        forces.append(_PULSE._numpy(env._pred_thrust[0, 0, 0]))
        torques.append(_PULSE._numpy(env._pred_moment[0, 0, 0]))
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.cfg.dt)
        samples.append(_PULSE._asset_sample(env._predators[0]))

    executed_actions = np.stack(actions).astype(np.float32, copy=False)
    if not np.array_equal(executed_actions, physics_action_tape):
        raise RuntimeError(
            f"case {case['case_index']} executed action differs from analytic tape"
        )
    states = {
        name: np.stack([sample[name] for sample in samples]).astype(
            np.float32, copy=False
        )
        for name in samples[0]
    }
    return (
        states,
        executed_actions,
        np.stack(forces).astype(np.float32, copy=False),
        np.stack(torques).astype(np.float32, copy=False),
    )


def _run(args: argparse.Namespace) -> Path:
    import torch
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.uav_3v1_env import (
        Uav3v1Env,
    )
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
        if not math.isclose(
            env.sim.cfg.dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError(
                f"expected physics dt {PHYSICS_DT_S}, got {env.sim.cfg.dt}"
            )
        asset = env._predators[0]
        if (
            tuple(asset.body_names) != BODY_NAMES
            or tuple(asset.joint_names) != JOINT_NAMES
        ):
            raise RuntimeError(
                "unexpected Crazyflie articulation: "
                f"bodies={asset.body_names}, joints={asset.joint_names}"
            )

        cases = _case_specs()
        physics_action_tape = _physics_action_tape()
        repeat_states: list[list[dict[str, np.ndarray]]] = []
        reference_actions: list[np.ndarray] = []
        reference_forces: list[np.ndarray] = []
        reference_torques: list[np.ndarray] = []
        for repeat_index in range(ISAAC_REPEATS):
            case_states: list[dict[str, np.ndarray]] = []
            for case in cases:
                states, actions, forces, torques = _run_case(
                    env, case, physics_action_tape
                )
                case_states.append(states)
                case_index = int(case["case_index"])
                if repeat_index == 0:
                    if case_index > 0:
                        for name, actual, expected in (
                            ("action", actions, reference_actions[0]),
                            ("force", forces, reference_forces[0]),
                            ("torque", torques, reference_torques[0]),
                        ):
                            if not np.array_equal(actual, expected):
                                raise RuntimeError(
                                    f"{name} differs across phase case {case_index}"
                                )
                    reference_actions.append(actions)
                    reference_forces.append(forces)
                    reference_torques.append(torques)
                else:
                    for name, actual, expected in (
                        ("action", actions, reference_actions[case_index]),
                        ("force", forces, reference_forces[case_index]),
                        ("torque", torques, reference_torques[case_index]),
                    ):
                        if not np.array_equal(actual, expected):
                            raise RuntimeError(
                                f"{name} changed in repeat {repeat_index}, "
                                f"case {case_index}"
                            )
            repeat_states.append(case_states)

        arrays: dict[str, np.ndarray] = {
            "action.executed_norm": np.stack(reference_actions).astype(
                np.float32, copy=False
            ),
            "action.force_b_n": np.stack(reference_forces).astype(
                np.float32, copy=False
            ),
            "action.torque_b_nm": np.stack(reference_torques).astype(
                np.float32, copy=False
            ),
        }
        for state_name in repeat_states[0][0]:
            arrays[f"state.{state_name}"] = np.stack(
                [
                    np.stack(
                        [case_state[state_name] for case_state in repeat]
                    )
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
        task_dir = (
            repository
            / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/"
            "uavpredatorprey_3v1"
        )
        isaaclab_root = Path("/workspace/isaaclab")
        action_contract = _action_schedule_contract()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "producer": {
                "backend": "isaac_physx_rotor_phase_sweep_probe",
                "task_id": TASK_ID,
                "uavpredatorprey": _PULSE._git_metadata(repository),
                "isaaclab": _PULSE._isaaclab_metadata(isaaclab_root),
                "isaac_image": os.environ.get("UAV_ISAAC_IMAGE"),
                "isaac_image_id": os.environ.get("UAV_ISAAC_IMAGE_ID"),
                "isaac_sim_version": os.environ.get("ISAACSIM_VERSION"),
                "isaac_sim_build": _PULSE._first_line(Path("/isaac-sim/VERSION")),
                "python_version": sys.version.split()[0],
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "nvidia_driver_version": _PULSE._nvidia_driver_version(),
                "exporter_sha256": _PULSE._sha256_file(Path(__file__).resolve()),
                "rotor_pulse_helper_sha256": _PULSE._sha256_file(
                    Path(_PULSE.__file__).resolve()
                ),
                "task_environment_sha256": _PULSE._sha256_file(
                    task_dir / "uav_3v1_env.py"
                ),
                "task_config_sha256": _PULSE._sha256_file(
                    task_dir / "uav_3v1_env_cfg.py"
                ),
                "seed": args.seed,
            },
            "contract": {
                "action_interface": "direct_wrench_v0",
                "action_order": list(ACTION_ORDER),
                "action_bounds": [-1.0, 1.0],
                "target_asset": TARGET_ASSET,
                "target_body": TARGET_BODY,
                "moment_scale_nm": float(env.cfg.moment_scale),
                "physics_dt_s": float(env.sim.cfg.dt),
                "sample_dt_s": float(env.sim.cfg.dt),
                "policy_dt_s": POLICY_DT_S,
                "policy_action_count": POLICY_ACTION_COUNT,
                "physics_substeps_per_policy_action": (
                    PHYSICS_SUBSTEPS_PER_POLICY_ACTION
                ),
                "physics_steps_per_case": PHYSICS_STEPS,
                "sample_count": PHYSICS_STEPS + 1,
                "isaac_repeats": ISAAC_REPEATS,
                "case_order": CASE_ORDER,
                "phase_grid_interval_rad_half_open": [0.0, math.pi],
                "phase_grid_rule": PHASE_GRID_RULE,
                "rotor_direction_m1_to_m4": list(ROTOR_DIRECTION),
                "initial_rotor_speed_scalar_radps": ROTOR_SPEED_SCALAR_RADPS,
                **action_contract,
                "action_timing": (
                    "physics action[t] is processed and applied before state "
                    "sample t+1"
                ),
                "policy_reward_termination_or_environment_reset_called": False,
                "controlled_articulation_state_restore_before_each_case": True,
                "minimum_system_com_z_m_for_publication": (
                    MINIMUM_SYSTEM_COM_Z_M
                ),
                "quaternion_order": "wxyz",
                "quaternion_semantics": "q_WB_body_to_world",
                "translation_point": "mass_weighted_five_link_system_com",
                "orientation_source": "root_link",
                "linear_velocity_source": (
                    "mass_weighted_five_link_system_com_world"
                ),
                "root_angular_velocity_frames": ["body", "world"],
                "all_link_angular_velocity_frames": [
                    "individual_link_body",
                    "world",
                ],
                "force_frame": "root_link_body",
                "torque_frame": "root_link_body",
            },
            "initial_state": _initial_state_contract(),
            "asset": _PULSE._json_compatible(_PULSE._asset_metadata(env)),
            "case_count": len(cases),
            "trace_count": ISAAC_REPEATS,
            "policy_action_count": POLICY_ACTION_COUNT,
            "physics_steps": PHYSICS_STEPS,
            "sample_count": PHYSICS_STEPS + 1,
            "array_count": len(array_records),
            "arrays": array_records,
            "cases": cases,
        }
        output = _write_fixture(
            args.output, arrays, metadata, repository=repository
        )
        print(f"[INFO] Wrote immutable rotor-phase-sweep fixture: {output}")
        print(
            f"[INFO] {len(cases)} cases x {ISAAC_REPEATS} repeats x "
            f"{POLICY_ACTION_COUNT} policy actions x "
            f"{PHYSICS_SUBSTEPS_PER_POLICY_ACTION} physics substeps"
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
