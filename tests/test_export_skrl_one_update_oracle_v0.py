"""Boundary tests for the deterministic one-update oracle exporter."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path("scripts/jax/export_skrl_one_update_oracle_v0.py")


def _module():
    specification = importlib.util.spec_from_file_location(
        "export_skrl_one_update_oracle_v0_test_module", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_inputs_are_time_major_and_cover_selected_recurrent_resets() -> None:
    module = _module()
    arrays = module._input_arrays()

    assert arrays["input.predator_observation_raw"].shape == (4, 4, 90)
    assert arrays["input.centralized_state_raw"].shape == (4, 4, 120)
    assert arrays["input.predator_actor_initial_hidden"].shape == (4, 3, 256)
    assert arrays["input.environment_indices"].tolist() == [0, 2]
    done = arrays["input.terminated"] | arrays["input.truncated"]
    assert done[1, 0, 0]
    assert done[2, 2, 0]
    assert np.any(arrays["input.predator_actor_initial_hidden"] != 0.0)


def test_squashed_log_probability_is_one_joint_scalar_per_sample() -> None:
    module = _module()
    mean = module.torch.zeros((3, 12), dtype=module.torch.float32)
    log_std = module.torch.zeros((12,), dtype=module.torch.float32)
    pre_tanh = module.torch.linspace(-0.4, 0.4, 36).reshape(3, 12)

    result = module._squashed_log_prob(mean, log_std, pre_tanh)

    assert result.shape == (3, 1)
    assert bool(module.torch.isfinite(result).all())


def test_fixture_is_checksummed_pickle_free_and_no_clobber(tmp_path: Path) -> None:
    module = _module()
    arrays = module._input_arrays()
    arrays.update(
        {
            "input.pre_tanh_action": np.zeros((4, 4, 3, 4), dtype=np.float32),
            "input.sampled_action_norm": np.zeros((4, 4, 3, 4), dtype=np.float32),
            "input.old_log_prob": np.zeros((4, 4, 1), dtype=np.float32),
            "input.old_value": np.zeros((4, 4, 1), dtype=np.float32),
            "input.returns": np.zeros((4, 4, 1), dtype=np.float32),
            "input.normalized_advantage": np.zeros((4, 4, 1), dtype=np.float32),
            "expected.loss.total": np.asarray(0.0, dtype=np.float32),
            "expected.optimizer.pre_clip_gradient_norm": np.asarray(
                0.0, dtype=np.float32
            ),
            "expected.optimizer.step": np.asarray(1, dtype=np.int32),
        }
    )
    output = tmp_path / "one_update"

    written = module._write_fixture(
        output,
        arrays,
        {"schema_version": module.SCHEMA_VERSION},
        repository=Path.cwd(),
    )

    assert written == output.resolve()
    assert {path.name for path in written.iterdir()} == module.FIXTURE_FILENAMES
    with np.load(written / module.ARRAYS_FILENAME, allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
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
