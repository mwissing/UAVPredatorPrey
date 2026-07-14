"""Pure contract tests for the held-out Isaac rotor-phase sweep exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path("scripts/jax/export_isaac_rotor_phase_sweep_oracle_v0.py")


def _module():
    specification = importlib.util.spec_from_file_location(
        "export_isaac_rotor_phase_sweep_oracle_v0_test_module", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _arrays(module):
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in module._expected_array_shapes().items()
    }


def test_cli_defaults_use_separate_held_out_schema_and_artifact_path() -> None:
    module = _module()
    args = module._base_parser().parse_args([])

    assert module.SCHEMA_VERSION == "uavpredatorprey.rotor_phase_sweep_oracle.v0"
    assert str(args.output) == (
        "/workspace/artifacts/transfer/rotor_phase_sweep_v0/"
        "midpoint16_aggressive_high_z10_seed42"
    )
    assert args.seed == 42
    explicit = module._base_parser().parse_args(
        ["--seed", "7", "--output", "/tmp/held-out-phase"]
    )
    assert explicit.seed == 7
    assert explicit.output == Path("/tmp/held-out-phase")


def test_initial_state_contract_requests_contact_free_world_positions() -> None:
    module = _module()
    initial_state = module._initial_state_contract()

    assert module.MINIMUM_SYSTEM_COM_Z_M == 1.0
    assert initial_state["requested_root_link_pos_w_m"] == [-3.0, 0.0, 10.0]
    assert initial_state["requested_root_link_pos_w_m_by_asset"] == {
        "predator_0": [-3.0, 0.0, 10.0],
        "predator_1": [0.0, -3.0, 10.0],
        "predator_2": [3.0, 0.0, 10.0],
        "prey": [0.0, 3.0, 10.0],
    }
    assert initial_state["requested_root_link_position_frame"] == "world"


def test_phase_cases_are_exact_float32_midpoints_in_ascending_order() -> None:
    module = _module()
    cases = module._case_specs()

    assert len(cases) == 16
    assert [case["case_index"] for case in cases] == list(range(16))
    assert [case["phase_index"] for case in cases] == list(range(16))
    assert [case["condition_id"] for case in cases] == [
        f"midpoint_{index:02d}" for index in range(16)
    ]
    assert {case["condition_kind"] for case in cases} == {
        "held_out_rotor_phase"
    }
    expected_phases = (
        (np.arange(16, dtype=np.float64) + 0.5) * np.pi / 16.0
    ).astype(np.float32)
    np.testing.assert_array_equal(module._phase_scalars_rad(), expected_phases)
    assert float(expected_phases[0]) > 0.0
    assert float(expected_phases[-1]) < math.pi

    for index, case in enumerate(cases):
        phase = expected_phases[index]
        expected_position = [
            float(phase),
            float(-phase),
            float(phase),
            float(-phase),
        ]
        assert case["phase_grid_numerator"] == 2 * index + 1
        assert case["phase_grid_denominator"] == 32
        assert case["initial_rotor_phase_scalar_rad"] == float(phase)
        assert case["rotor_speed_scalar_radps"] == 200.0
        assert case["requested_initial_joint_pos_rad"] == expected_position
        assert case["requested_initial_joint_vel_radps"] == [
            200.0,
            -200.0,
            200.0,
            -200.0,
        ]


def test_analytic_action_tape_is_float32_bounded_and_held_twice() -> None:
    module = _module()
    policy = module._policy_action_tape()
    physics = module._physics_action_tape()

    assert policy.shape == (50, 4)
    assert physics.shape == (100, 4)
    assert policy.dtype == np.float32
    assert physics.dtype == np.float32
    assert policy.flags.c_contiguous
    assert physics.flags.c_contiguous
    assert np.all(np.isfinite(policy))
    assert np.max(np.abs(policy)) <= np.float32(1.0)
    np.testing.assert_array_equal(physics[0::2], policy)
    np.testing.assert_array_equal(physics[1::2], policy)

    n = np.arange(50, dtype=np.float64) + 0.5
    expected = np.stack(
        (
            0.05 + 0.75 * np.sin(2.0 * np.pi * n / 19.0),
            0.95 * np.sin(2.0 * np.pi * n / 11.0),
            0.90 * np.cos(2.0 * np.pi * n / 13.0),
            0.85 * np.sin(2.0 * np.pi * n / 17.0 + np.pi / 7.0),
        ),
        axis=-1,
    ).astype(np.float32)
    np.testing.assert_array_equal(policy, expected)


def test_action_contract_freezes_formula_cast_order_shapes_and_hashes() -> None:
    module = _module()
    contract = module._action_schedule_contract()

    assert contract["action_schedule_id"] == "analytic_aggressive_multiaxis_v0"
    assert contract["action_schedule_formula_by_component"] == {
        "collective_thrust": "0.05 + 0.75 * sin(2*pi*(n + 0.5)/19)",
        "tau_x": "0.95 * sin(2*pi*(n + 0.5)/11)",
        "tau_y": "0.90 * cos(2*pi*(n + 0.5)/13)",
        "tau_z": "0.85 * sin(2*pi*(n + 0.5)/17 + pi/7)",
    }
    assert contract["action_schedule_formula_evaluation_dtype"] == "float64"
    assert contract["action_schedule_storage_dtype"] == "float32"
    assert contract["action_schedule_cast_order"] == (
        "evaluate every component in float64, stack [n, 4], cast once to "
        "float32, repeat each policy row twice"
    )
    assert contract["action_tape_policy_shape"] == [50, 4]
    assert contract["action_tape_physics_shape"] == [100, 4]
    assert contract["action_tape_sha256_encoding"] == (
        "C-contiguous float32 raw bytes"
    )
    assert contract["actions_identical_across_cases"] is True
    assert contract["zero_order_hold_physics_steps"] == 2
    assert contract["action_tape_policy_sha256"] == module._float32_raw_sha256(
        module._policy_action_tape()
    )
    assert contract["action_tape_policy_sha256"] == (
        "b90340dd8927e42e883c42e892d7cef57af78be52464fa8c00249648e8c179f1"
    )
    assert contract["action_tape_physics_sha256"] == module._float32_raw_sha256(
        module._physics_action_tape()
    )
    assert contract["action_tape_physics_sha256"] == (
        "758f314df5c3d0a5d88d4107b71fea001bf489dce04fc0fa427363ced8248640"
    )


def test_array_contract_has_shared_actions_and_three_repeat_state_traces() -> None:
    module = _module()
    shapes = module._expected_array_shapes()

    assert set(shapes) == set(module.ARRAY_NAMES)
    assert shapes["action.executed_norm"] == (16, 100, 4)
    assert shapes["action.force_b_n"] == (16, 100, 3)
    assert shapes["action.torque_b_nm"] == (16, 100, 3)
    assert shapes["state.system_com_pos_w_m"] == (3, 16, 101, 3)
    assert shapes["state.root_link_quat_wb_wxyz"] == (3, 16, 101, 4)
    assert shapes["state.all_link_ang_vel_b_radps"] == (3, 16, 101, 5, 3)
    assert shapes["state.joint_pos_rad"] == (3, 16, 101, 4)
    assert shapes["state.joint_vel_radps"] == (3, 16, 101, 4)


def test_array_validation_rejects_members_dtype_shape_and_finiteness() -> None:
    module = _module()
    arrays = _arrays(module)
    module._validate_arrays(arrays)

    missing = dict(arrays)
    missing.pop("action.force_b_n")
    with pytest.raises(ValueError, match="missing"):
        module._validate_arrays(missing)

    unexpected = dict(arrays)
    unexpected["extra"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected"):
        module._validate_arrays(unexpected)

    wrong_shape = dict(arrays)
    wrong_shape["action.force_b_n"] = np.zeros((16, 99, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        module._validate_arrays(wrong_shape)

    wrong_dtype = dict(arrays)
    wrong_dtype["action.force_b_n"] = arrays["action.force_b_n"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        module._validate_arrays(wrong_dtype)

    non_finite = {name: value.copy() for name, value in arrays.items()}
    non_finite["state.root_ang_vel_w_radps"][0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        module._validate_arrays(non_finite)


def test_fixture_is_immutable_pickle_free_and_checksummed(tmp_path: Path) -> None:
    module = _module()
    arrays = _arrays(module)
    arrays["state.system_com_pos_w_m"][..., 2] = np.float32(10.0)
    output = tmp_path / "rotor_phase_sweep"

    written = module._write_fixture(
        output,
        arrays,
        {"schema_version": module.SCHEMA_VERSION},
        repository=Path.cwd(),
    )

    assert written == output.resolve()
    assert {path.name for path in written.iterdir()} == module.FIXTURE_FILENAMES
    with np.load(written / module.ARRAYS_FILENAME, allow_pickle=False) as archive:
        assert set(archive.files) == set(module.ARRAY_NAMES)
        assert all(archive[name].dtype == np.float32 for name in archive.files)
    manifest = (written / module.CHECKSUMS_FILENAME).read_text(encoding="utf-8")
    for filename in (module.ARRAYS_FILENAME, module.METADATA_FILENAME):
        digest = hashlib.sha256((written / filename).read_bytes()).hexdigest()
        assert f"{digest}  {filename}\n" in manifest
    metadata = json.loads(
        (written / module.METADATA_FILENAME).read_text(encoding="utf-8")
    )
    height_guard = metadata["validation"]["contact_free_height_guard"]
    assert height_guard == {
        "axis": "z",
        "check": "minimum_recorded_system_com_world_z",
        "coordinate_frame": "world",
        "minimum_system_com_z_m": 1.0,
        "observed_minimum_location": {
            "case_index": 0,
            "repeat_index": 0,
            "sample_index": 0,
        },
        "observed_minimum_system_com_z_m": 10.0,
        "passed": True,
        "recorded_value_count": 3 * 16 * 101,
        "required_relation": (
            "every recorded system COM z >= minimum_system_com_z_m"
        ),
        "state_array": "state.system_com_pos_w_m",
        "unit": "m",
    }

    with pytest.raises(FileExistsError, match="already exists"):
        module._write_fixture(
            output,
            arrays,
            {"schema_version": module.SCHEMA_VERSION},
            repository=Path.cwd(),
        )


def test_fixture_publication_rejects_any_system_com_sample_below_guard(
    tmp_path: Path,
) -> None:
    module = _module()
    arrays = _arrays(module)
    arrays["state.system_com_pos_w_m"][..., 2] = np.float32(10.0)
    arrays["state.system_com_pos_w_m"][2, 7, 53, 2] = np.float32(0.999)
    output = tmp_path / "too_low"

    with pytest.raises(
        ValueError,
        match=r"publication refused:.*below required 1 m.*repeat=2, case=7, sample=53",
    ):
        module._write_fixture(
            output,
            arrays,
            {"schema_version": module.SCHEMA_VERSION},
            repository=Path.cwd(),
        )

    assert not output.exists()
