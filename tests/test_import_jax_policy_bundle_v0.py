"""Tests for building an optimizer-reset skrl checkpoint from JAX tensors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPTS_JAX = Path(__file__).resolve().parents[1] / "scripts" / "jax"
if str(SCRIPTS_JAX) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_JAX))

import export_skrl_policy_transfer_v0 as transfer_v0
import import_jax_policy_bundle_v0 as reverse_import


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array(spec, *, transferred: bool) -> np.ndarray:
    dtype = np.dtype(spec.stored_dtype)
    fill = 0.25 if transferred else -0.125
    if spec.semantic_path.endswith(".log_std"):
        fill = -1.5 if transferred else -0.5
    elif spec.semantic_path.endswith(".running_variance"):
        fill = 1.75 if transferred else 1.25
    elif spec.semantic_path.endswith(".current_count"):
        fill = 4097.0 if transferred else 1025.0
    return np.full(spec.stored_shape, fill, dtype=dtype)


def _checkpoint(path: Path) -> tuple[Path, str]:
    checkpoint = {"predator": {}, "prey": {}}
    for role in transfer_v0.ROLES:
        for module in (
            "policy",
            "value",
            "state_preprocessor",
            "shared_state_preprocessor",
            "value_preprocessor",
        ):
            checkpoint[role][module] = {}
    for spec in transfer_v0.LEAF_SPECS:
        role, module, name = transfer_v0._split_source_path(spec.source_path)
        checkpoint[role][module][name] = torch.from_numpy(
            _array(spec, transferred=False)
        )
    checkpoint["predator"]["optimizer"] = {"state": {}, "param_groups": []}
    checkpoint["prey"]["optimizer"] = {"state": {}, "param_groups": []}
    torch.save(checkpoint, path)
    return path, _sha256(path)


def _reverse_transfer(path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    path.mkdir()
    arrays = {
        f"leaf_{index:06d}": _array(spec, transferred=True)
        for index, spec in enumerate(transfer_v0.LEAF_SPECS)
    }
    np.savez(path / transfer_v0.ARRAYS_FILENAME, **arrays)
    metadata = {
        "schema_version": reverse_import.TRANSFER_SCHEMA,
        "created_utc": "2026-07-16T00:00:00+00:00",
        "source_policy_bundle": {
            "path": "/workspace/artifacts/jax/policy",
            "identity": {"schema": "test", "files": {}},
            "annotations": {"test": "reverse"},
        },
        "producer_runtime": {
            "python_version": "3.11.15",
            "jax_version": "0.10.2",
            "jaxlib_version": "0.10.2",
            "backend": "gpu",
            "x64_enabled": False,
        },
        "transfer_contract": reverse_import._expected_transfer_contract(),
        "leaf_count": len(transfer_v0.LEAF_SPECS),
        "leaves": reverse_import._leaf_records(),
    }
    (path / transfer_v0.METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / transfer_v0.CHECKSUMS_FILENAME).write_text(
        f"{_sha256(path / transfer_v0.ARRAYS_FILENAME)}  "
        f"{transfer_v0.ARRAYS_FILENAME}\n"
        f"{_sha256(path / transfer_v0.METADATA_FILENAME)}  "
        f"{transfer_v0.METADATA_FILENAME}\n",
        encoding="ascii",
    )
    return path, arrays


def test_import_replaces_all_tensors_and_removes_both_optimizers(
    tmp_path: Path,
) -> None:
    transfer, arrays = _reverse_transfer(tmp_path / "reverse")
    template, digest = _checkpoint(tmp_path / "template.pt")

    output = reverse_import.import_jax_policy_bundle_v0(
        transfer=transfer,
        template_checkpoint=template,
        expected_template_sha256=digest,
        output=tmp_path / "checkpoint",
    )

    assert {path.name for path in output.iterdir()} == reverse_import.OUTPUT_FILENAMES
    checkpoint = torch.load(
        output / reverse_import.AGENT_FILENAME,
        map_location="cpu",
        weights_only=True,
    )
    extracted, optimizer_modules = transfer_v0._extract_and_validate_tensors(
        checkpoint
    )
    assert optimizer_modules == {"predator": "absent", "prey": "absent"}
    for name, expected in arrays.items():
        np.testing.assert_array_equal(extracted[name], expected)
    metadata = json.loads(
        (output / reverse_import.METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["conversion"]["optimizer_reset"] == (
        "optimizer_modules_removed_before_skrl_load"
    )
    assert metadata["conversion"]["template_optimizer_was_removed"] == {
        "predator": True,
        "prey": True,
    }


def test_template_digest_mismatch_precedes_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer, _ = _reverse_transfer(tmp_path / "reverse")
    template, _ = _checkpoint(tmp_path / "template.pt")

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run after a digest mismatch")

    monkeypatch.setattr(reverse_import.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checksum mismatch"):
        reverse_import.import_jax_policy_bundle_v0(
            transfer=transfer,
            template_checkpoint=template,
            expected_template_sha256="0" * 64,
            output=tmp_path / "checkpoint",
        )


def test_import_is_no_clobber(tmp_path: Path) -> None:
    transfer, _ = _reverse_transfer(tmp_path / "reverse")
    template, digest = _checkpoint(tmp_path / "template.pt")
    output = reverse_import.import_jax_policy_bundle_v0(
        transfer=transfer,
        template_checkpoint=template,
        expected_template_sha256=digest,
        output=tmp_path / "checkpoint",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        reverse_import.import_jax_policy_bundle_v0(
            transfer=transfer,
            template_checkpoint=template,
            expected_template_sha256=digest,
            output=output,
        )
