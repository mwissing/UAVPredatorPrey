from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pytest
import skrl
import torch


SCRIPTS_JAX = Path(__file__).resolve().parents[1] / "scripts" / "jax"
if str(SCRIPTS_JAX) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_JAX))

import export_skrl_policy_transfer_v0 as transfer_v0
import export_skrl_policy_transfer_v1 as transfer_v1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_fixture() -> dict:
    checkpoint: dict[str, dict] = {"predator": {}, "prey": {}}
    for role in transfer_v0.ROLES:
        for module in (
            "policy",
            "value",
            "state_preprocessor",
            "shared_state_preprocessor",
            "value_preprocessor",
        ):
            checkpoint[role][module] = {}

    for index, spec in enumerate(transfer_v0.LEAF_SPECS):
        role, module, name = transfer_v0._split_source_path(spec.source_path)
        dtype = torch.float32 if spec.stored_dtype == "float32" else torch.float64
        fill = -0.5 if spec.semantic_path.endswith(".log_std") else (index + 1) / 1000
        if spec.semantic_path.endswith(".running_variance"):
            fill = 1.0 + index / 1000
        elif spec.semantic_path.endswith(".current_count"):
            fill = 1025.0 + index
        checkpoint[role][module][name] = torch.full(
            spec.stored_shape, fill, dtype=dtype
        )

    for role in transfer_v0.ROLES:
        checkpoint[role]["optimizer"] = {"state": {}, "param_groups": []}
    return checkpoint


def _write_checkpoint(path: Path) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_fixture(), path)
    return path, _sha256(path)


def _rewrite_checksums(output: Path) -> None:
    (output / transfer_v0.CHECKSUMS_FILENAME).write_text(
        f"{_sha256(output / transfer_v0.ARRAYS_FILENAME)}  "
        f"{transfer_v0.ARRAYS_FILENAME}\n"
        f"{_sha256(output / transfer_v0.METADATA_FILENAME)}  "
        f"{transfer_v0.METADATA_FILENAME}\n",
        encoding="ascii",
    )


def test_v1_exports_explicitly_verified_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, digest = _write_checkpoint(
        tmp_path / "local_run" / "checkpoints" / "agent_7200.pt"
    )
    output = tmp_path / "transfer"
    real_load = transfer_v1.torch.load
    calls: list[tuple[tuple[int, ...], dict]] = []

    def checked_load(stream, *args, **kwargs):
        calls.append(
            (transfer_v0._file_identity(os.fstat(stream.fileno())), dict(kwargs))
        )
        return real_load(stream, *args, **kwargs)

    monkeypatch.setattr(transfer_v1.torch, "load", checked_load)
    result = transfer_v1.export_skrl_policy_transfer_v1(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=digest,
        output=output,
    )

    assert result == output
    assert calls == [
        (
            transfer_v0._file_identity(checkpoint.stat()),
            {"map_location": "cpu", "weights_only": True},
        )
    ]
    metadata = transfer_v1.verify_skrl_policy_transfer_v1(output)
    assert metadata["schema_version"] == transfer_v1.SCHEMA_VERSION
    assert metadata["source_checkpoint"] == {
        "path": checkpoint.absolute().as_posix(),
        "sha256": digest,
        "size_bytes": checkpoint.stat().st_size,
        "verification": "explicit_sha256",
    }
    assert metadata["source_runtime"] == {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "skrl_version": str(skrl.__version__),
    }
    assert metadata["transfer_contract"]["optimizer_modules"] == {
        "predator": "present_excluded",
        "prey": "present_excluded",
    }
    assert metadata["leaves"] == transfer_v1._leaf_records_v1()

    with np.load(output / transfer_v0.ARRAYS_FILENAME, allow_pickle=False) as archive:
        assert archive.files == [f"leaf_{index:06d}" for index in range(112)]


def test_v1_digest_mismatch_precedes_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path / "agent.pt")

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run before digest verification")

    monkeypatch.setattr(transfer_v1.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        transfer_v1.export_skrl_policy_transfer_v1(
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            output=tmp_path / "transfer",
        )
    assert not (tmp_path / "transfer").exists()


def test_v1_rejects_checkpoint_symlink_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, digest = _write_checkpoint(tmp_path / "real.pt")
    link = tmp_path / "link.pt"
    link.symlink_to(checkpoint)

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not follow a checkpoint symlink")

    monkeypatch.setattr(transfer_v1.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        transfer_v1.export_skrl_policy_transfer_v1(
            checkpoint=link,
            expected_checkpoint_sha256=digest,
            output=tmp_path / "transfer",
        )


def test_v1_runtime_tamper_and_v0_verifier_are_rejected(tmp_path: Path) -> None:
    checkpoint, digest = _write_checkpoint(tmp_path / "agent.pt")
    output = tmp_path / "transfer"
    transfer_v1.export_skrl_policy_transfer_v1(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=digest,
        output=output,
    )

    with pytest.raises(ValueError, match="unsupported transfer schema"):
        transfer_v0.verify_skrl_policy_transfer_v0(output)

    metadata_path = output / transfer_v0.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_runtime"]["python_version"] = "3.12.0"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_checksums(output)
    with pytest.raises(ValueError, match="Python 3.11.x"):
        transfer_v1.verify_skrl_policy_transfer_v1(output)
