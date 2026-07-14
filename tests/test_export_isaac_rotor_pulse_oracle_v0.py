"""Pure boundary tests for the Isaac rotor-pulse diagnostic exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path("scripts/jax/export_isaac_rotor_pulse_oracle_v0.py")


def _module():
    specification = importlib.util.spec_from_file_location(
        "export_isaac_rotor_pulse_oracle_v0_test_module", SCRIPT
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


def test_case_matrix_has_canonical_rotor_outer_schedule_inner_order() -> None:
    module = _module()
    cases = module._case_specs()

    assert len(cases) == 49
    assert [case["case_index"] for case in cases] == list(range(49))
    assert [case["schedule_id"] for case in cases[:7]] == [
        "idle",
        "pulse_x_pos",
        "pulse_x_neg",
        "pulse_y_pos",
        "pulse_y_neg",
        "pulse_z_pos",
        "pulse_z_neg",
    ]
    assert [cases[index]["rotor_speed_scalar_radps"] for index in range(0, 35, 7)] == [
        0.0,
        50.0,
        -50.0,
        200.0,
        -200.0,
    ]
    assert [cases[index]["condition_id"] for index in range(0, 49, 7)] == [
        "speed_+0_radps",
        "speed_+50_radps",
        "speed_-50_radps",
        "speed_+200_radps",
        "speed_-200_radps",
        "phase_pi_over_4_rad",
        "phase_pi_over_2_rad",
    ]
    assert {case["condition_kind"] for case in cases[:35]} == {"rotor_speed"}
    assert {case["condition_kind"] for case in cases[35:]} == {"static_rotor_phase"}
    assert [case["schedule_id"] for case in cases[35:42]] == [
        case["schedule_id"] for case in cases[:7]
    ]
    assert [case["schedule_id"] for case in cases[42:49]] == [
        case["schedule_id"] for case in cases[:7]
    ]
    assert cases[0]["axis"] is None
    assert cases[0]["axis_index"] is None
    assert cases[0]["torque_sign"] == 0
    assert cases[0]["pulse_step_range"] == [0, 0]
    assert cases[0]["coast_step_range"] == [0, 10]
    assert cases[1]["pulse_step_range"] == [0, 2]
    assert cases[1]["coast_step_range"] == [2, 10]


def test_requested_rotor_positions_and_velocities_use_m1_to_m4_direction_pattern() -> None:
    module = _module()
    cases = module._case_specs()

    for rotor_index, scalar in enumerate((0.0, 50.0, -50.0, 200.0, -200.0)):
        expected = [scalar, -scalar, scalar, -scalar]
        for case in cases[rotor_index * 7 : (rotor_index + 1) * 7]:
            assert case["initial_rotor_phase_scalar_rad"] == 0.0
            assert case["requested_initial_joint_pos_rad"] == [0.0] * 4
            assert case["requested_initial_joint_vel_radps"] == expected

    for offset, phase in ((35, math.pi / 4.0), (42, math.pi / 2.0)):
        expected_scalar = np.float32(phase)
        expected_positions = np.asarray(
            [expected_scalar, -expected_scalar, expected_scalar, -expected_scalar],
            dtype=np.float32,
        )
        for case in cases[offset : offset + 7]:
            assert case["rotor_speed_scalar_radps"] == 0.0
            assert case["initial_rotor_phase_scalar_rad"] == float(expected_scalar)
            assert case["requested_initial_joint_vel_radps"] == [0.0] * 4
            assert case["requested_initial_joint_pos_rad"] == [
                float(value) for value in expected_positions
            ]


def test_initial_joint_sample_must_exactly_match_requested_float32_state() -> None:
    module = _module()
    case = module._case_specs()[35]
    sample = {
        "joint_pos_rad": np.asarray(case["requested_initial_joint_pos_rad"], dtype=np.float32),
        "joint_vel_radps": np.asarray(case["requested_initial_joint_vel_radps"], dtype=np.float32),
    }
    module._validate_initial_joint_sample(case, sample)

    wrong = dict(sample)
    wrong["joint_pos_rad"] = sample["joint_pos_rad"].copy()
    wrong["joint_pos_rad"][0] = np.nextafter(
        wrong["joint_pos_rad"][0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(RuntimeError, match="does not exactly match"):
        module._validate_initial_joint_sample(case, wrong)


def test_controlled_state_keeps_default_formation_and_accepts_explicit_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _module()

    class Data:
        def __init__(self) -> None:
            self.default_root_state = torch.zeros((1, 13), dtype=torch.float32)
            self.default_joint_pos = torch.zeros((1, 4), dtype=torch.float32)
            self.default_joint_vel = torch.zeros((1, 4), dtype=torch.float32)

    class Asset:
        def __init__(self) -> None:
            self.data = Data()
            self.written_root_pose = None

        def reset(self) -> None:
            pass

        def write_root_pose_to_sim(self, value) -> None:
            self.written_root_pose = value.clone()

        def write_root_velocity_to_sim(self, value) -> None:
            pass

        def write_joint_state_to_sim(self, position, velocity) -> None:
            pass

    class Sim:
        class Config:
            dt = 0.01

        cfg = Config()

        def forward(self) -> None:
            pass

    class Scene:
        def update(self, dt: float) -> None:
            pass

    class Env:
        def __init__(self) -> None:
            assets = [Asset() for _ in range(4)]
            self._predators = assets[:3]
            self._prey = assets[3]
            self.device = "cpu"
            self.episode_length_buf = torch.ones(1)
            self._pred_alive = torch.zeros((1, 3), dtype=torch.bool)
            self._pred_oob = torch.ones((1, 3), dtype=torch.bool)
            self._pred_newly_oob = torch.ones((1, 3), dtype=torch.bool)
            self._prey_oob = torch.ones(1, dtype=torch.bool)
            self._intermediate_values_valid = True
            self.sim = Sim()
            self.scene = Scene()

    monkeypatch.setattr(module, "_asset_sample", lambda asset: {"asset": asset})
    case = module._case_specs()[0]
    env = Env()
    module._write_controlled_state(env, case)
    assets = [*env._predators, env._prey]
    default_actual = [asset.written_root_pose[0, :3].tolist() for asset in assets]
    assert default_actual == [list(position) for position in module.INITIAL_ROOT_POSITIONS_W_M]

    explicit = (
        (-3.0, 0.0, 10.0),
        (0.0, -3.0, 10.0),
        (3.0, 0.0, 10.0),
        (0.0, 3.0, 10.0),
    )
    module._write_controlled_state(env, case, root_positions_w_m=explicit)
    explicit_actual = [asset.written_root_pose[0, :3].tolist() for asset in assets]
    assert explicit_actual == [list(position) for position in explicit]

    with pytest.raises(ValueError, match="one finite xyz world position per asset"):
        module._write_controlled_state(env, case, root_positions_w_m=explicit[:3])


def test_schema_and_default_path_identify_phase_sweep_fixture() -> None:
    module = _module()

    assert module.SCHEMA_VERSION == "uavpredatorprey.rotor_pulse_oracle.v0"
    assert module.DEFAULT_OUTPUT == (
        "/workspace/artifacts/transfer/rotor_pulse_v0/"
        "free_gyro_on_static_phase_seed42"
    )


def test_action_schedule_applies_signed_low_pulse_then_zero_torque() -> None:
    module = _module()
    for case in module._case_specs():
        actions = module._action_schedule(case)
        assert actions.shape == (10, 4)
        assert actions.dtype == np.float32
        np.testing.assert_array_equal(actions[:, 0], np.full(10, -1.0, dtype=np.float32))

        expected_moments = np.zeros((10, 3), dtype=np.float32)
        if case["axis_index"] is not None:
            expected_moments[:2, case["axis_index"]] = np.float32(
                0.02 * case["torque_sign"]
            )
        np.testing.assert_array_equal(actions[:, 1:], expected_moments)


def test_array_shapes_keep_actions_shared_and_states_per_repeat() -> None:
    module = _module()
    shapes = module._expected_array_shapes()

    assert shapes["action.executed_norm"] == (49, 10, 4)
    assert shapes["action.force_b_n"] == (49, 10, 3)
    assert shapes["state.system_com_pos_w_m"] == (3, 49, 11, 3)
    assert shapes["state.root_link_quat_wb_wxyz"] == (3, 49, 11, 4)
    assert shapes["state.all_link_ang_vel_b_radps"] == (3, 49, 11, 5, 3)
    assert shapes["state.joint_pos_rad"] == (3, 49, 11, 4)
    assert shapes["state.joint_vel_radps"] == (3, 49, 11, 4)


def test_array_validation_is_strict_about_members_dtype_shape_and_finiteness() -> None:
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
    wrong_shape["action.force_b_n"] = np.zeros((49, 9, 3), dtype=np.float32)
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
    output = tmp_path / "rotor_pulse"

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

    with pytest.raises(FileExistsError, match="already exists"):
        module._write_fixture(
            output,
            arrays,
            {"schema_version": module.SCHEMA_VERSION},
            repository=Path.cwd(),
        )
