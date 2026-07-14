"""Focused boundary tests for the standalone skrl forward-oracle exporter."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path("scripts/jax/export_skrl_policy_oracle_v0.py")


def _module():
    specification = importlib.util.spec_from_file_location(
        "export_skrl_policy_oracle_v0_test_module", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_fixed_inputs_cover_zero_nonzero_hidden_and_normalizer_extremes() -> None:
    module = _module()
    arrays = module._input_arrays()

    assert set(arrays).issubset(module.ARRAY_SHAPES)
    assert all(value.dtype == np.float32 for value in arrays.values())
    assert np.all(arrays["input.predator_actor_hidden"][0] == 0.0)
    assert np.any(arrays["input.predator_actor_hidden"][1:] != 0.0)
    assert np.all(arrays["input.prey_actor_hidden"][0] == 0.0)
    assert np.any(arrays["input.prey_actor_hidden"][1:] != 0.0)
    assert np.max(np.abs(arrays["input.predator_observation_raw"][-1])) == 75.0
    assert np.max(np.abs(arrays["input.centralized_state_raw"][-1])) == 90.0


def test_safe_checkpoint_loader_hashes_exact_bytes_and_uses_weights_only(
    tmp_path: Path,
) -> None:
    module = _module()
    checkpoint = tmp_path / "minimal.pt"
    torch.save({"weights": torch.arange(3, dtype=torch.float32)}, checkpoint)
    payload = checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    loaded, actual_digest, size = module._safe_load_checkpoint(checkpoint, digest)

    assert torch.equal(loaded["weights"], torch.arange(3, dtype=torch.float32))
    assert actual_digest == digest
    assert size == len(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._safe_load_checkpoint(checkpoint, "0" * 64)


def test_fixture_is_pickle_free_checksummed_and_never_replaced(
    tmp_path: Path,
) -> None:
    module = _module()
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in module.ARRAY_SHAPES.items()
    }
    output = tmp_path / "oracle"

    written = module._write_fixture(
        output,
        arrays,
        {"schema_version": module.SCHEMA_VERSION},
        repository=Path.cwd(),
    )

    assert written == output.resolve()
    assert {path.name for path in written.iterdir()} == module.FIXTURE_FILENAMES
    with np.load(written / module.ARRAYS_FILENAME, allow_pickle=False) as archive:
        assert set(archive.files) == set(module.ARRAY_SHAPES)
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
