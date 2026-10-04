#!/usr/bin/env python3
"""Build an Isaac/skrl checkpoint from a neutral JAX reverse transfer.

The exact checkpoint template is SHA-256-bound before deserialization. Only
the 112 policy, critic, and normalizer tensors are replaced. Optimizer modules
are removed for both roles so Isaac fine-tuning starts with fresh Adam state
instead of silently reusing moments from the template checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import skrl
import torch

import export_skrl_policy_transfer_v0 as v0
import export_skrl_policy_transfer_v1 as v1


TRANSFER_SCHEMA = "uavpredatorprey_jax.policy_bundle_to_skrl.v0"
CHECKPOINT_ARTIFACT_SCHEMA = (
    "uavpredatorprey.jax_policy_to_skrl_checkpoint.v0"
)
AGENT_FILENAME = "agent.pt"
METADATA_FILENAME = "metadata.json"
CHECKSUMS_FILENAME = "SHA256SUMS.txt"
OUTPUT_FILENAMES = frozenset(
    (AGENT_FILENAME, METADATA_FILENAME, CHECKSUMS_FILENAME)
)
TRANSFER_FILENAMES = frozenset(
    (v0.ARRAYS_FILENAME, v0.METADATA_FILENAME, v0.CHECKSUMS_FILENAME)
)


def _leaf_records() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "array_name": f"leaf_{index:06d}",
            "semantic_path": spec.semantic_path,
            "source_path": spec.source_path,
            "stored_shape": list(spec.stored_shape),
            "stored_dtype": spec.stored_dtype,
            "source_layout": spec.source_layout,
        }
        for index, spec in enumerate(v0.LEAF_SPECS)
    ]


def _expected_transfer_contract() -> dict[str, Any]:
    return {
        "direction": "jax_policy_bundle_v0_to_skrl_checkpoint",
        "model_contract": "recurrent_entity_attention_v0",
        "roles": ["predator", "prey"],
        "gru_gate_order": ["reset", "update", "new"],
        "model_dtype": "float32",
        "normalizer_source_dtype": "float32",
        "normalizer_stored_dtype": "float64",
        "normalizer_cast": "exact_float32_to_float64_value_preserving",
        "stored_parameter_layout": "torch_source_layout",
        "optimizer_state": "absent_reset_required_for_finetuning",
    }


def _validate_transfer_metadata(metadata: Any) -> None:
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "schema_version",
        "created_utc",
        "source_policy_bundle",
        "producer_runtime",
        "transfer_contract",
        "leaf_count",
        "leaves",
    }:
        raise ValueError("JAX reverse-transfer metadata keys do not match v0")
    if metadata["schema_version"] != TRANSFER_SCHEMA:
        raise ValueError(
            f"unsupported JAX reverse-transfer schema: {metadata['schema_version']!r}"
        )
    if metadata["leaf_count"] != len(v0.LEAF_SPECS):
        raise ValueError("JAX reverse-transfer leaf_count must be 112")
    if metadata["leaves"] != _leaf_records():
        raise ValueError("JAX reverse-transfer leaves do not match static v0")
    source = metadata["source_policy_bundle"]
    if not isinstance(source, Mapping) or set(source) != {
        "path",
        "identity",
        "annotations",
    }:
        raise ValueError("source_policy_bundle keys do not match v0")
    if not isinstance(source["path"], str) or not source["path"]:
        raise ValueError("source_policy_bundle.path must be non-empty")
    if not isinstance(source["identity"], Mapping) or not isinstance(
        source["annotations"], Mapping
    ):
        raise ValueError("source policy identity and annotations must be objects")
    runtime = metadata["producer_runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python_version",
        "jax_version",
        "jaxlib_version",
        "backend",
        "x64_enabled",
    }:
        raise ValueError("JAX producer_runtime keys do not match v0")
    if metadata["transfer_contract"] != _expected_transfer_contract():
        raise ValueError("JAX reverse-transfer contract does not match v0")


def _read_transfer(
    directory_argument: Path,
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    directory = v0._require_real_directory(
        directory_argument, label="JAX reverse-transfer artifact"
    )
    if {entry.name for entry in directory.iterdir()} != TRANSFER_FILENAMES:
        raise ValueError("JAX reverse-transfer directory members do not match v0")
    for filename in TRANSFER_FILENAMES:
        member = directory / filename
        if member.is_symlink() or not member.is_file():
            raise ValueError(f"JAX reverse-transfer member must be regular: {member}")

    snapshots = {
        filename: v0._read_regular_file_snapshot(directory / filename)
        for filename in (
            v0.CHECKSUMS_FILENAME,
            v0.ARRAYS_FILENAME,
            v0.METADATA_FILENAME,
        )
    }
    checksums = v0._parse_output_checksums(
        snapshots[v0.CHECKSUMS_FILENAME][0]
    )
    for filename in (v0.ARRAYS_FILENAME, v0.METADATA_FILENAME):
        actual = v0._sha256_bytes(snapshots[filename][0])
        if actual != checksums[filename]:
            raise ValueError(
                f"JAX reverse-transfer checksum mismatch for {filename}"
            )

    metadata = v0._load_json_bytes(
        snapshots[v0.METADATA_FILENAME][0], label="JAX reverse-transfer metadata"
    )
    _validate_transfer_metadata(metadata)
    arrays_bytes = snapshots[v0.ARRAYS_FILENAME][0]
    expected_names = [f"leaf_{index:06d}" for index in range(len(v0.LEAF_SPECS))]
    v0._validate_npz_container(arrays_bytes, expected_names=expected_names)
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
        if archive.files != expected_names:
            raise ValueError("JAX reverse-transfer arrays do not match ordered v0")
        for index, spec in enumerate(v0.LEAF_SPECS):
            name = f"leaf_{index:06d}"
            v0._validate_numpy_array(archive[name], spec)
            arrays[name] = np.array(archive[name], copy=True, order="C")

    for filename, (_, identity) in snapshots.items():
        _, current_identity = v0._read_regular_file_snapshot(directory / filename)
        if current_identity != identity:
            raise ValueError(
                f"JAX reverse-transfer member changed while reading: {filename}"
            )
    return directory, arrays, dict(metadata), checksums


def _load_template_checkpoint(
    checkpoint_argument: Path, expected_sha256: str
) -> tuple[Path, Mapping[str, Any], str, int, dict[str, str]]:
    expected_digest = v0._require_digest(
        expected_sha256, label="expected template checkpoint sha256"
    )
    checkpoint = v1._canonical_checkpoint_path(checkpoint_argument)
    with v0._open_regular_file_no_follow(checkpoint) as stream:
        before = os.fstat(stream.fileno())
        digest, size = v0._sha256_stream(stream)
        after_hash = os.fstat(stream.fileno())
        if v0._file_identity(before) != v0._file_identity(after_hash):
            raise ValueError(f"template checkpoint changed while hashing: {checkpoint}")
        if digest != expected_digest:
            raise ValueError(
                "template checkpoint checksum mismatch: "
                f"expected {expected_digest}, got {digest}"
            )
        stream.seek(0)
        checkpoint_state = torch.load(
            stream, map_location="cpu", weights_only=True
        )
        after_load = os.fstat(stream.fileno())
        if v0._file_identity(before) != v0._file_identity(after_load):
            raise ValueError(f"template checkpoint changed while loading: {checkpoint}")
    _, optimizer_modules = v0._extract_and_validate_tensors(checkpoint_state)
    return checkpoint, checkpoint_state, digest, size, optimizer_modules


def _replace_tensors_and_reset_optimizers(
    checkpoint_state: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[Mapping[str, Any], dict[str, bool]]:
    for index, spec in enumerate(v0.LEAF_SPECS):
        role, module, tensor_name = v0._split_source_path(spec.source_path)
        target = checkpoint_state[role][module][tensor_name]
        expected_dtype = (
            torch.float32 if spec.stored_dtype == "float32" else torch.float64
        )
        replacement = torch.from_numpy(
            np.array(arrays[f"leaf_{index:06d}"], copy=True, order="C")
        )
        if replacement.dtype != expected_dtype or replacement.shape != target.shape:
            raise ValueError(f"replacement tensor mismatch for {spec.source_path}")
        checkpoint_state[role][module][tensor_name] = replacement

    removed: dict[str, bool] = {}
    for role in v0.ROLES:
        removed[role] = "optimizer" in checkpoint_state[role]
        checkpoint_state[role].pop("optimizer", None)

    extracted, optimizer_modules = v0._extract_and_validate_tensors(checkpoint_state)
    if optimizer_modules != {"predator": "absent", "prey": "absent"}:
        raise RuntimeError("optimizer modules were not removed from both roles")
    for name, expected in arrays.items():
        if not np.array_equal(extracted[name], expected):
            raise RuntimeError(f"checkpoint replacement changed tensor {name}")
    return checkpoint_state, removed


def _write_checkpoint_fsynced(path: Path, checkpoint_state: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        torch.save(checkpoint_state, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_fsynced(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_written_checkpoint(
    checkpoint_path: Path, arrays: Mapping[str, np.ndarray]
) -> None:
    with checkpoint_path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    extracted, optimizer_modules = v0._extract_and_validate_tensors(checkpoint)
    if optimizer_modules != {"predator": "absent", "prey": "absent"}:
        raise ValueError("written checkpoint unexpectedly contains optimizer state")
    for name, expected in arrays.items():
        if not np.array_equal(extracted[name], expected):
            raise ValueError(f"written checkpoint tensor mismatch: {name}")


def _build_output_metadata(
    *,
    transfer_path: Path,
    transfer_metadata: Mapping[str, Any],
    transfer_checksums: Mapping[str, str],
    template_path: Path,
    template_sha256: str,
    template_size: int,
    template_optimizer_modules: Mapping[str, str],
    removed_optimizer_modules: Mapping[str, bool],
    result_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_ARTIFACT_SCHEMA,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_reverse_transfer": {
            "path": str(transfer_path),
            "schema_version": transfer_metadata["schema_version"],
            "checksums": dict(transfer_checksums),
            "source_policy_bundle": transfer_metadata["source_policy_bundle"],
            "producer_runtime": transfer_metadata["producer_runtime"],
        },
        "template_checkpoint": {
            "path": str(template_path),
            "sha256": template_sha256,
            "size_bytes": template_size,
            "optimizer_modules": dict(template_optimizer_modules),
        },
        "result_checkpoint": {
            "filename": AGENT_FILENAME,
            "sha256": v0._sha256_file(result_path),
            "size_bytes": result_path.stat().st_size,
            "roles": ["predator", "prey"],
            "tensor_count": len(v0.LEAF_SPECS),
            "optimizer_modules": {"predator": "absent", "prey": "absent"},
        },
        "conversion": {
            "model_and_normalizers": "replaced_from_verified_neutral_arrays",
            "template_non_optimizer_structure": "preserved",
            "optimizer_reset": "optimizer_modules_removed_before_skrl_load",
            "template_optimizer_was_removed": dict(removed_optimizer_modules),
            "fine_tune_requirement": "initialize_fresh_optimizer_state",
        },
        "producer_runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__),
            "skrl_version": str(skrl.__version__),
        },
    }


def import_jax_policy_bundle_v0(
    *,
    transfer: Path,
    template_checkpoint: Path,
    expected_template_sha256: str,
    output: Path,
) -> Path:
    transfer_path, arrays, transfer_metadata, transfer_checksums = _read_transfer(
        transfer
    )
    (
        template_path,
        checkpoint_state,
        template_sha256,
        template_size,
        template_optimizer_modules,
    ) = _load_template_checkpoint(template_checkpoint, expected_template_sha256)
    checkpoint_state, removed = _replace_tensors_and_reset_optimizers(
        checkpoint_state, arrays
    )

    destination = Path(os.path.abspath(output.expanduser()))
    if os.path.lexists(destination):
        raise FileExistsError(f"checkpoint artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        checkpoint_path = temporary / AGENT_FILENAME
        _write_checkpoint_fsynced(checkpoint_path, checkpoint_state)
        _verify_written_checkpoint(checkpoint_path, arrays)
        metadata = _build_output_metadata(
            transfer_path=transfer_path,
            transfer_metadata=transfer_metadata,
            transfer_checksums=transfer_checksums,
            template_path=template_path,
            template_sha256=template_sha256,
            template_size=template_size,
            template_optimizer_modules=template_optimizer_modules,
            removed_optimizer_modules=removed,
            result_path=checkpoint_path,
        )
        _write_json_fsynced(temporary / METADATA_FILENAME, metadata)
        checksum_lines = [
            f"{v0._sha256_file(temporary / filename)}  {filename}"
            for filename in (AGENT_FILENAME, METADATA_FILENAME)
        ]
        v0._write_text_fsynced(
            temporary / CHECKSUMS_FILENAME, "\n".join(checksum_lines) + "\n"
        )
        if {path.name for path in temporary.iterdir()} != OUTPUT_FILENAMES:
            raise RuntimeError("checkpoint staging directory has unexpected files")
        v0._fsync_directory(temporary)
        v0._publish_directory_no_replace(temporary, destination)
        v0._fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--template-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-template-sha256",
        required=True,
        help="mandatory SHA-256 identity of the exact skrl template checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory containing agent.pt and provenance sidecars",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = import_jax_policy_bundle_v0(
            transfer=args.transfer,
            template_checkpoint=args.template_checkpoint,
            expected_template_sha256=args.expected_template_sha256,
            output=args.output,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[INFO] Wrote skrl checkpoint artifact: {output}")
    print(f"[INFO] Fine-tune checkpoint: {output / AGENT_FILENAME}")
    print("[INFO] Optimizer modules: predator=absent, prey=absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
