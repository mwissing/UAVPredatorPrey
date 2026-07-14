"""Pure boundary tests for the Isaac policy-trajectory exporter."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path("scripts/jax/export_isaac_policy_trajectory_v0.py")


def _module():
    specification = importlib.util.spec_from_file_location(
        "export_isaac_policy_trajectory_v0_test_module", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _arrays(module, *, policy_steps: int = 2, repeats: int = 2):
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in module._expected_array_shapes(policy_steps, repeats).items()
    }


def test_shapes_preserve_two_substeps_and_four_vehicle_action_tape() -> None:
    module = _module()
    shapes = module._expected_array_shapes(policy_steps=50, repeats=3)

    assert shapes["action.executed_norm"] == (50, 4, 4)
    assert shapes["state.system_com_pos_w_m"] == (3, 101, 4, 3)
    assert shapes["state.root_link_quat_wb_wxyz"] == (3, 101, 4, 4)
    assert shapes["state.joint_vel_radps"] == (3, 101, 4, 4)


def test_isaaclab_commit_falls_back_to_the_container_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    commit = "f4aa17f87e2e5db5484f0b5974918573e8918ce2"
    monkeypatch.setenv("ISAACLAB_COMMIT", commit)
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda _repository: {"commit": None, "dirty": None},
    )

    metadata = module._isaaclab_metadata(Path("/workspace/isaaclab"))

    assert metadata == {
        "commit": commit,
        "commit_source": "ISAACLAB_COMMIT",
        "dirty": None,
    }


def test_array_validation_is_strict_about_members_dtype_shape_and_finiteness() -> None:
    module = _module()
    arrays = _arrays(module)
    module._validate_arrays(arrays, policy_steps=2, repeats=2)

    missing = dict(arrays)
    missing.pop("action.force_b_n")
    with pytest.raises(ValueError, match="missing"):
        module._validate_arrays(missing, policy_steps=2, repeats=2)

    wrong_dtype = dict(arrays)
    wrong_dtype["action.force_b_n"] = wrong_dtype["action.force_b_n"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        module._validate_arrays(wrong_dtype, policy_steps=2, repeats=2)

    non_finite = {name: value.copy() for name, value in arrays.items()}
    non_finite["state.root_ang_vel_b_radps"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module._validate_arrays(non_finite, policy_steps=2, repeats=2)


def test_fixture_is_immutable_pickle_free_and_checksummed(tmp_path: Path) -> None:
    module = _module()
    arrays = _arrays(module)
    output = tmp_path / "paired_trace"
    repository = Path.cwd()

    written = module._write_fixture(
        output,
        arrays,
        {"schema_version": module.SCHEMA_VERSION},
        repository=repository,
        policy_steps=2,
        repeats=2,
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
            repository=repository,
            policy_steps=2,
            repeats=2,
        )
