"""Export a locally produced skrl checkpoint with explicit SHA-256 provenance.

Transfer v1 keeps the exact 112-tensor data contract from the immutable
Windows migration exporter.  It replaces migration-bundle provenance with a
mandatory caller-supplied checkpoint digest and records the actual Linux
runtime that deserializes the checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import skrl
import torch

import export_skrl_policy_transfer_v0 as v0


SCHEMA_VERSION = "uavpredatorprey.skrl_policy_transfer.v1"
VERIFICATION_MODE = "explicit_sha256"
EXPECTED_PYTORCH_VERSION = "2.7.0+cu128"
EXPECTED_SKRL_VERSION = "1.4.3"

LEAF_SPECS = v0.LEAF_SPECS

_TOP_LEVEL_METADATA_KEYS = frozenset(
    (
        "schema_version",
        "source_checkpoint",
        "source_runtime",
        "transfer_contract",
        "leaf_count",
        "leaves",
    )
)
_SOURCE_CHECKPOINT_KEYS = frozenset(
    ("path", "sha256", "size_bytes", "verification")
)
_SOURCE_RUNTIME_KEYS = frozenset(
    ("python_version", "pytorch_version", "skrl_version")
)
_PYTHON_VERSION_PATTERN = re.compile(r"^3\.11\.(?:0|[1-9][0-9]*)$")


def export_skrl_policy_transfer_v1(
    *,
    checkpoint: str | os.PathLike[str],
    expected_checkpoint_sha256: str,
    output: str | os.PathLike[str],
) -> Path:
    """Verify and export one local checkpoint as a neutral v1 directory."""

    expected_digest = v0._require_digest(
        expected_checkpoint_sha256,
        label="expected checkpoint sha256",
    )
    checkpoint_path = _canonical_checkpoint_path(Path(checkpoint))

    with v0._open_regular_file_no_follow(checkpoint_path) as stream:
        before = os.fstat(stream.fileno())
        checkpoint_sha256, checkpoint_size = v0._sha256_stream(stream)
        after_hash = os.fstat(stream.fileno())
        if (
            v0._file_identity(before) != v0._file_identity(after_hash)
            or checkpoint_size != after_hash.st_size
        ):
            raise ValueError(f"checkpoint changed while hashing: {checkpoint_path}")
        if checkpoint_sha256 != expected_digest:
            raise ValueError(
                "checkpoint checksum mismatch: "
                f"expected {expected_digest}, got {checkpoint_sha256}"
            )

        # Reuse the same verified, O_NOFOLLOW descriptor for deserialization.
        stream.seek(0)
        loaded = torch.load(stream, map_location="cpu", weights_only=True)
        after_load = os.fstat(stream.fileno())
        if v0._file_identity(before) != v0._file_identity(after_load):
            raise ValueError(f"checkpoint changed while loading: {checkpoint_path}")

    arrays, optimizer_modules = v0._extract_and_validate_tensors(loaded)
    metadata = _build_metadata_v1(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_size,
        optimizer_modules=optimizer_modules,
    )
    return _write_transfer_directory_v1(Path(output), arrays, metadata)


def verify_skrl_policy_transfer_v1(
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Strictly verify a completed local-checkpoint transfer v1 directory."""

    transfer = v0._require_real_directory(Path(directory), label="transfer artifact")
    entries = {entry.name for entry in transfer.iterdir()}
    if entries != v0.TRANSFER_FILENAMES:
        raise ValueError(
            "transfer directory members do not match schema v1: "
            f"expected {sorted(v0.TRANSFER_FILENAMES)}, got {sorted(entries)}"
        )
    for filename in v0.TRANSFER_FILENAMES:
        member = transfer / filename
        if member.is_symlink() or not member.is_file():
            raise ValueError(f"transfer member must be a regular file: {member}")

    snapshots: dict[str, tuple[bytes, tuple[int, ...]]] = {}
    for filename in (v0.CHECKSUMS_FILENAME, *v0.PAYLOAD_FILENAMES):
        snapshots[filename] = v0._read_regular_file_snapshot(transfer / filename)

    expected_checksums = v0._parse_output_checksums(
        snapshots[v0.CHECKSUMS_FILENAME][0]
    )
    for filename in v0.PAYLOAD_FILENAMES:
        actual = v0._sha256_bytes(snapshots[filename][0])
        if actual != expected_checksums[filename]:
            raise ValueError(
                f"transfer checksum mismatch for {filename}: "
                f"expected {expected_checksums[filename]}, got {actual}"
            )

    metadata = v0._load_json_bytes(
        snapshots[v0.METADATA_FILENAME][0], label="transfer metadata"
    )
    _validate_metadata_v1(metadata)

    arrays_bytes = snapshots[v0.ARRAYS_FILENAME][0]
    expected_names = [f"leaf_{index:06d}" for index in range(len(LEAF_SPECS))]
    v0._validate_npz_container(arrays_bytes, expected_names=expected_names)
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
        if archive.files != expected_names:
            raise ValueError(
                "transfer NPZ members do not match the ordered static 112-leaf schema"
            )
        for index, spec in enumerate(LEAF_SPECS):
            v0._validate_numpy_array(archive[f"leaf_{index:06d}"], spec)

    for filename, (_, identity) in snapshots.items():
        _, current_identity = v0._read_regular_file_snapshot(transfer / filename)
        if current_identity != identity:
            raise ValueError(f"transfer member changed while verifying: {filename}")
    return metadata


def _canonical_checkpoint_path(argument: Path) -> Path:
    absolute = Path(os.path.abspath(argument.expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"checkpoint path must not contain symlinks: {current}"
            )
    if not absolute.is_file():
        raise FileNotFoundError(
            f"checkpoint must be a regular file: {absolute}"
        )
    return absolute


def _runtime_v1() -> dict[str, str]:
    runtime = {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "skrl_version": str(skrl.__version__),
    }
    _validate_runtime_v1(runtime)
    return runtime


def _leaf_records_v1() -> list[dict[str, Any]]:
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
        for index, spec in enumerate(LEAF_SPECS)
    ]


def _build_metadata_v1(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_size: int,
    optimizer_modules: Mapping[str, str],
) -> dict[str, Any]:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": {
            "path": PurePosixPath(checkpoint_path).as_posix(),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
            "verification": VERIFICATION_MODE,
        },
        "source_runtime": _runtime_v1(),
        "transfer_contract": {
            "model_contract": v0.MODEL_CONTRACT,
            "roles": list(v0.ROLES),
            "gru_gate_order": ["reset", "update", "new"],
            "model_dtype": v0.MODEL_DTYPE,
            "normalizer_dtype": v0.NORMALIZER_DTYPE,
            "optimizer_state": "absent_reset_required_for_training",
            "optimizer_modules": dict(optimizer_modules),
        },
        "leaf_count": len(LEAF_SPECS),
        "leaves": _leaf_records_v1(),
    }
    _validate_metadata_v1(metadata)
    return metadata


def _validate_metadata_v1(metadata: Any) -> None:
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != _TOP_LEVEL_METADATA_KEYS
    ):
        raise ValueError("transfer metadata keys do not match schema v1")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported transfer schema: {metadata['schema_version']!r}")

    source = metadata["source_checkpoint"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_CHECKPOINT_KEYS:
        raise ValueError("source_checkpoint keys do not match schema v1")
    _validate_checkpoint_provenance_v1(source)
    _validate_runtime_v1(metadata["source_runtime"])

    if metadata["leaf_count"] != len(LEAF_SPECS):
        raise ValueError("transfer metadata leaf_count must be 112")
    records = metadata["leaves"]
    expected_records = _leaf_records_v1()
    if not isinstance(records, list) or records != expected_records:
        raise ValueError("transfer metadata leaves do not match static schema v1")

    contract = metadata["transfer_contract"]
    if (
        not isinstance(contract, Mapping)
        or set(contract) != v0._TRANSFER_CONTRACT_KEYS
    ):
        raise ValueError("transfer_contract keys do not match schema v1")
    expected_contract = {
        "model_contract": v0.MODEL_CONTRACT,
        "roles": list(v0.ROLES),
        "gru_gate_order": ["reset", "update", "new"],
        "model_dtype": v0.MODEL_DTYPE,
        "normalizer_dtype": v0.NORMALIZER_DTYPE,
        "optimizer_state": "absent_reset_required_for_training",
    }
    for key, expected in expected_contract.items():
        if contract[key] != expected:
            raise ValueError(f"transfer_contract.{key} must be {expected!r}")
    optimizer_modules = contract["optimizer_modules"]
    if (
        not isinstance(optimizer_modules, Mapping)
        or set(optimizer_modules) != set(v0.ROLES)
    ):
        raise ValueError("optimizer_modules must contain predator and prey")
    if any(
        value not in v0._OPTIMIZER_MODULE_VALUES
        for value in optimizer_modules.values()
    ):
        raise ValueError("optimizer_modules contains an unsupported status")


def _validate_checkpoint_provenance_v1(source: Mapping[str, Any]) -> None:
    path = source["path"]
    if not isinstance(path, str) or not path:
        raise ValueError("source_checkpoint.path must be a non-empty string")
    parsed = PurePosixPath(path)
    if (
        not parsed.name
        or str(parsed) != path
        or path.startswith("//")
        or "\\" in path
        or any(part in ("", ".", "..") for part in parsed.parts)
    ):
        raise ValueError(
            "source_checkpoint.path must be a canonical absolute or relative POSIX path"
        )
    v0._require_digest(source["sha256"], label="source_checkpoint.sha256")
    if (
        not isinstance(source["size_bytes"], int)
        or isinstance(source["size_bytes"], bool)
        or source["size_bytes"] < 1
    ):
        raise ValueError("source checkpoint size_bytes must be positive")
    if source["verification"] != VERIFICATION_MODE:
        raise ValueError(
            f"source_checkpoint.verification must be {VERIFICATION_MODE!r}"
        )


def _validate_runtime_v1(runtime: Any) -> None:
    if not isinstance(runtime, Mapping) or set(runtime) != _SOURCE_RUNTIME_KEYS:
        raise ValueError("source_runtime keys do not match schema v1")
    python_version = runtime["python_version"]
    if (
        not isinstance(python_version, str)
        or _PYTHON_VERSION_PATTERN.fullmatch(python_version) is None
    ):
        raise ValueError("source_runtime.python_version must be Python 3.11.x")
    if runtime["pytorch_version"] != EXPECTED_PYTORCH_VERSION:
        raise ValueError(
            f"source_runtime.pytorch_version must be {EXPECTED_PYTORCH_VERSION!r}"
        )
    if runtime["skrl_version"] != EXPECTED_SKRL_VERSION:
        raise ValueError(
            f"source_runtime.skrl_version must be {EXPECTED_SKRL_VERSION!r}"
        )


def _write_transfer_directory_v1(
    destination_argument: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(os.path.abspath(destination_argument.expanduser()))
    if os.path.lexists(destination):
        raise FileExistsError(f"transfer destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError(f"transfer parent must be a real directory: {destination.parent}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        arrays_path = temporary / v0.ARRAYS_FILENAME
        np.savez(arrays_path, **arrays)
        v0._fsync_file(arrays_path)
        metadata_text = json.dumps(
            metadata, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        v0._write_text_fsynced(temporary / v0.METADATA_FILENAME, metadata_text)
        checksum_lines = [
            f"{v0._sha256_file(temporary / filename)}  {filename}"
            for filename in v0.PAYLOAD_FILENAMES
        ]
        v0._write_text_fsynced(
            temporary / v0.CHECKSUMS_FILENAME,
            "\n".join(checksum_lines) + "\n",
        )
        v0._fsync_directory(temporary)
        verify_skrl_policy_transfer_v1(temporary)
        v0._publish_directory_no_replace(temporary, destination)
        v0._fsync_directory(destination.parent)
        return destination
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="mandatory SHA-256 identity of the exact local checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new transfer directory; an existing path is never replaced",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = export_skrl_policy_transfer_v1(
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            output=args.output,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[INFO] Exported verified skrl policy transfer: {output}")
    print(f"[INFO] Schema: {SCHEMA_VERSION}; tensors: {len(LEAF_SPECS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
