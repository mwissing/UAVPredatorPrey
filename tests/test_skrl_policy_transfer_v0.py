from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPTS_JAX = Path(__file__).resolve().parents[1] / "scripts" / "jax"
if str(SCRIPTS_JAX) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_JAX))

import export_skrl_policy_transfer_v0 as transfer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_fixture() -> dict:
    checkpoint: dict[str, dict] = {"predator": {}, "prey": {}}
    for role in transfer.ROLES:
        for module in (
            "policy",
            "value",
            "state_preprocessor",
            "shared_state_preprocessor",
            "value_preprocessor",
        ):
            checkpoint[role][module] = {}

    for index, spec in enumerate(transfer.LEAF_SPECS):
        role, module, name = transfer._split_source_path(spec.source_path)
        dtype = torch.float32 if spec.stored_dtype == "float32" else torch.float64
        fill = -0.5 if spec.semantic_path.endswith(".log_std") else (index + 1) / 1000
        if spec.semantic_path.endswith(".running_variance"):
            fill = 1.0 + index / 1000
        elif spec.semantic_path.endswith(".current_count"):
            fill = 1025.0 + index
        checkpoint[role][module][name] = torch.full(
            spec.stored_shape, fill, dtype=dtype
        )

    checkpoint["predator"]["optimizer"] = {"state": {}, "param_groups": []}
    return checkpoint


def _write_bundle(
    root: Path,
    *,
    mutate=None,
) -> tuple[Path, Path]:
    checkpoint_state = _checkpoint_fixture()
    if mutate is not None:
        mutate(checkpoint_state)

    checkpoint = root / "current" / "agent_115200.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(checkpoint_state, checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)

    manifest = {
        "schema_version": 1,
        "runtime": {
            "python_version": "3.11.9",
            "torch_version": "2.7.0+cu128",
            "skrl_version": "1.4.3",
        },
        "current_checkpoint": {
            "bundle_path": "current/agent_115200.pt",
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "opponent_pool": {"entries": []},
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "SHA256SUMS.txt").write_text(
        f"{checkpoint_sha256}  current/agent_115200.pt\n"
        f"{_sha256(manifest_path)}  MANIFEST.json\n",
        encoding="utf-8",
    )
    return checkpoint, root


def _rewrite_output_checksums(output: Path) -> None:
    (output / "SHA256SUMS.txt").write_text(
        f"{_sha256(output / 'arrays.npz')}  arrays.npz\n"
        f"{_sha256(output / 'metadata.json')}  metadata.json\n",
        encoding="utf-8",
    )


def test_static_schema_has_exact_canonical_112_leaf_order() -> None:
    assert len(transfer.LEAF_SPECS) == 112
    assert transfer.LEAF_SPECS[0].semantic_path == (
        "model.predator_actor.own_encoder.hidden.kernel"
    )
    assert transfer.LEAF_SPECS[26].semantic_path == "model.predator_actor.log_std"
    assert transfer.LEAF_SPECS[27].semantic_path == (
        "model.prey_actor.own_encoder.hidden.kernel"
    )
    assert transfer.LEAF_SPECS[49].semantic_path == "model.prey_actor.log_std"
    assert transfer.LEAF_SPECS[50].semantic_path.startswith("model.predator_critic.")
    assert transfer.LEAF_SPECS[72].semantic_path.startswith("model.prey_critic.")
    assert transfer.LEAF_SPECS[94].semantic_path == (
        "normalizer.predator_actor.running_mean"
    )
    assert transfer.LEAF_SPECS[-1].semantic_path == (
        "normalizer.prey_value.current_count"
    )
    assert len({spec.semantic_path for spec in transfer.LEAF_SPECS}) == 112
    assert len({spec.source_path for spec in transfer.LEAF_SPECS}) == 112


def test_export_writes_verified_pickle_free_artifact_and_uses_safe_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "transfer"
    real_load = transfer.torch.load
    calls: list[dict] = []

    def checked_load(file, *args, **kwargs):
        calls.append(dict(kwargs))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(transfer.torch, "load", checked_load)
    result = transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle,
        checkpoint=checkpoint,
        output=output,
    )

    assert result == output
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert {entry.name for entry in output.iterdir()} == {
        "arrays.npz",
        "metadata.json",
        "SHA256SUMS.txt",
    }
    metadata = transfer.verify_skrl_policy_transfer_v0(output)
    assert set(metadata) == {
        "schema_version",
        "source_checkpoint",
        "source_runtime",
        "transfer_contract",
        "leaf_count",
        "leaves",
    }
    assert metadata["schema_version"] == transfer.SCHEMA_VERSION
    assert metadata["leaf_count"] == 112
    assert metadata["source_checkpoint"]["path"] == "current/agent_115200.pt"
    assert metadata["source_checkpoint"]["sha256"] == _sha256(checkpoint)
    assert metadata["source_runtime"] == transfer.EXPECTED_SOURCE_RUNTIME
    assert metadata["transfer_contract"] == {
        "model_contract": "recurrent_entity_attention_v0",
        "roles": ["predator", "prey"],
        "gru_gate_order": ["reset", "update", "new"],
        "model_dtype": "float32",
        "normalizer_dtype": "float64",
        "optimizer_state": "absent_reset_required_for_training",
        "optimizer_modules": {
            "predator": "present_excluded",
            "prey": "absent",
        },
    }
    assert [record["array_name"] for record in metadata["leaves"]] == [
        f"leaf_{index:06d}" for index in range(112)
    ]
    assert all(
        set(record)
        == {
            "index",
            "array_name",
            "semantic_path",
            "source_path",
            "stored_shape",
            "stored_dtype",
            "source_layout",
        }
        for record in metadata["leaves"]
    )

    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        assert archive.files == [f"leaf_{index:06d}" for index in range(112)]
        first = archive["leaf_000000"]
        assert first.shape == (256, 12)
        assert first.dtype == np.dtype("float32")
        scaler = archive["leaf_000094"]
        assert scaler.shape == (90,)
        assert scaler.dtype == np.dtype("float64")

    checksum_lines = (output / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(checksum_lines) == 2
    assert checksum_lines[0].endswith("  arrays.npz")
    assert checksum_lines[1].endswith("  metadata.json")


def test_checkpoint_hash_failure_happens_before_torch_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run before checksum verification")

    monkeypatch.setattr(transfer.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle,
            checkpoint=checkpoint,
            output=tmp_path / "transfer",
        )


def test_checkpoint_symlink_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    backing = tmp_path / "backing.pt"
    checkpoint.rename(backing)
    checkpoint.symlink_to(backing)

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not receive a symlink")

    monkeypatch.setattr(transfer.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle,
            checkpoint=checkpoint,
            output=tmp_path / "transfer",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda state: state["predator"]["policy"].pop("own_encoder.0.weight"),
            "tensor keys do not match",
        ),
        (
            lambda state: state["predator"]["policy"].__setitem__(
                "unexpected.weight", torch.ones(1)
            ),
            "tensor keys do not match",
        ),
        (
            lambda state: state["predator"]["policy"].__setitem__(
                "own_encoder.0.weight", torch.ones(255, 12)
            ),
            "has shape",
        ),
        (
            lambda state: state["predator"]["policy"].__setitem__(
                "own_encoder.0.weight", torch.ones(256, 12, dtype=torch.float64)
            ),
            "has dtype",
        ),
        (
            lambda state: state["predator"]["policy"]["own_encoder.0.weight"].fill_(
                float("nan")
            ),
            "is not finite",
        ),
        (
            lambda state: state["predator"]["policy"]["log_std_parameter"].fill_(
                0.01
            ),
            "must be within",
        ),
        (
            lambda state: state["predator"]["state_preprocessor"][
                "running_variance"
            ].zero_(),
            "must be strictly positive",
        ),
        (
            lambda state: state["prey"]["value_preprocessor"][
                "current_count"
            ].zero_(),
            "must be strictly positive",
        ),
    ),
)
def test_export_rejects_invalid_checkpoint_tensor_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle", mutate=mutation)
    with pytest.raises((TypeError, ValueError), match=message):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle,
            checkpoint=checkpoint,
            output=tmp_path / "transfer",
        )


def test_export_refuses_existing_directory_and_symlink_destinations(
    tmp_path: Path,
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle, checkpoint=checkpoint, output=existing
        )

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle, checkpoint=checkpoint, output=link
        )


def test_verifier_rejects_extra_members_and_symlink_payloads(tmp_path: Path) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    extra_output = tmp_path / "extra"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=extra_output
    )
    (extra_output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="directory members"):
        transfer.verify_skrl_policy_transfer_v0(extra_output)

    link_output = tmp_path / "link-payload"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=link_output
    )
    arrays = link_output / "arrays.npz"
    backing = link_output / "arrays.backing"
    arrays.rename(backing)
    arrays.symlink_to(backing.name)
    with pytest.raises(ValueError, match="directory members|regular file"):
        transfer.verify_skrl_policy_transfer_v0(link_output)


def test_verifier_rejects_npz_extra_member_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "transfer"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=output
    )
    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["unexpected"] = np.ones((1,), dtype=np.float32)
    np.savez(output / "arrays.npz", **arrays)
    _rewrite_output_checksums(output)

    with pytest.raises(ValueError, match="NPZ members"):
        transfer.verify_skrl_policy_transfer_v0(output)


def test_verifier_rejects_reversed_checksum_order(tmp_path: Path) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "transfer"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=output
    )
    checksums = output / "SHA256SUMS.txt"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    checksums.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum order"):
        transfer.verify_skrl_policy_transfer_v0(output)


def test_verifier_rejects_compressed_npz_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "transfer"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=output
    )
    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    np.savez_compressed(output / "arrays.npz", **arrays)
    _rewrite_output_checksums(output)

    with pytest.raises(ValueError, match="uncompressed ZIP_STORED"):
        transfer.verify_skrl_policy_transfer_v0(output)


def test_verifier_rejects_reordered_npz_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "transfer"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=output
    )
    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        arrays = [
            (name, np.array(archive[name], copy=True)) for name in archive.files
        ]
    np.savez(output / "arrays.npz", **dict(reversed(arrays)))
    _rewrite_output_checksums(output)

    with pytest.raises(ValueError, match="ordered NPY member list"):
        transfer.verify_skrl_policy_transfer_v0(output)


def test_verifier_rejects_member_changed_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "transfer"
    transfer.export_skrl_policy_transfer_v0(
        bundle_root=bundle, checkpoint=checkpoint, output=output
    )
    metadata_path = output / "metadata.json"
    real_validate = transfer._validate_metadata

    def mutate_after_validation(metadata) -> None:
        real_validate(metadata)
        metadata_path.write_bytes(metadata_path.read_bytes() + b" ")

    monkeypatch.setattr(transfer, "_validate_metadata", mutate_after_validation)
    with pytest.raises(ValueError, match="changed during verification"):
        transfer.verify_skrl_policy_transfer_v0(output)


def test_export_refuses_destination_created_during_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    output = tmp_path / "raced-transfer"
    real_publish = transfer._publish_directory_no_replace

    def create_destination_then_publish(source: Path, destination: Path) -> None:
        destination.mkdir()
        real_publish(source, destination)

    monkeypatch.setattr(
        transfer, "_publish_directory_no_replace", create_destination_then_publish
    )
    with pytest.raises(FileExistsError, match="appeared during save"):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle, checkpoint=checkpoint, output=output
        )

    assert output.is_dir()
    assert list(tmp_path.glob(f".{output.name}.tmp-*")) == []


def test_manifest_and_checksums_must_agree_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle = _write_bundle(tmp_path / "bundle")
    lines = (bundle / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    lines[0] = f"{'0' * 64}  current/agent_115200.pt"
    (bundle / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not run when manifests disagree")

    monkeypatch.setattr(transfer.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="differs between MANIFEST"):
        transfer.export_skrl_policy_transfer_v0(
            bundle_root=bundle,
            checkpoint=checkpoint,
            output=tmp_path / "transfer",
        )
