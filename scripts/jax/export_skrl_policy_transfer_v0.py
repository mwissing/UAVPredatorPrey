"""Export the trusted skrl 3v1 checkpoint to a pickle-free transfer directory.

The migration checkpoint is a PyTorch/skrl agent checkpoint.  This exporter is
the only boundary that deserializes that file.  It verifies the checkpoint
against both migration manifests before calling ``torch.load`` and then writes
the 112 inference tensors in their original Torch names, layouts, and dtypes.

The output is deliberately source-faithful.  Linear/GRU transposition and the
construction of JAX parameter trees belong to the consumer, where those
semantic transformations can be tested independently.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
import torch


SCHEMA_VERSION = "uavpredatorprey.skrl_policy_transfer.v0"
MODEL_CONTRACT = "recurrent_entity_attention_v0"
ARRAYS_FILENAME = "arrays.npz"
METADATA_FILENAME = "metadata.json"
CHECKSUMS_FILENAME = "SHA256SUMS.txt"
MANIFEST_FILENAME = "MANIFEST.json"

TRANSFER_FILENAMES = frozenset(
    (ARRAYS_FILENAME, METADATA_FILENAME, CHECKSUMS_FILENAME)
)
PAYLOAD_FILENAMES = (ARRAYS_FILENAME, METADATA_FILENAME)
ROLES = ("predator", "prey")
MODEL_DTYPE = "float32"
NORMALIZER_DTYPE = "float64"
MIN_LOG_STD = -2.0
MAX_LOG_STD = 0.0

EXPECTED_SOURCE_RUNTIME = {
    "python_version": "3.11.9",
    "pytorch_version": "2.7.0+cu128",
    "skrl_version": "1.4.3",
}

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
    (
        "path",
        "sha256",
        "size_bytes",
        "manifest_path",
        "manifest_sha256",
        "checksums_path",
        "checksums_sha256",
    )
)
_SOURCE_RUNTIME_KEYS = frozenset(
    ("python_version", "pytorch_version", "skrl_version")
)
_TRANSFER_CONTRACT_KEYS = frozenset(
    (
        "model_contract",
        "roles",
        "gru_gate_order",
        "model_dtype",
        "normalizer_dtype",
        "optimizer_state",
        "optimizer_modules",
    )
)
_LEAF_RECORD_KEYS = frozenset(
    (
        "index",
        "array_name",
        "semantic_path",
        "source_path",
        "stored_shape",
        "stored_dtype",
        "source_layout",
    )
)
_SOURCE_LAYOUTS = frozenset(
    (
        "torch_linear_out_in",
        "torch_gru_3h_input",
        "torch_gru_3h_h",
        "vector",
        "scalar",
    )
)
_OPTIMIZER_MODULE_VALUES = frozenset(("present_excluded", "absent"))
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INPUT_CHECKSUM_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")
_OUTPUT_CHECKSUM_PATTERN = re.compile(
    rf"^([0-9a-f]{{64}})  ({re.escape(ARRAYS_FILENAME)}|"
    rf"{re.escape(METADATA_FILENAME)})$"
)


@dataclass(frozen=True)
class LeafSpec:
    semantic_path: str
    source_path: str
    stored_shape: tuple[int, ...]
    stored_dtype: str
    source_layout: str


def _linear_specs(
    semantic_prefix: str,
    source_prefix: str,
    *,
    input_dim: int,
    output_dim: int,
) -> tuple[LeafSpec, LeafSpec]:
    return (
        LeafSpec(
            f"{semantic_prefix}.kernel",
            f"{source_prefix}.weight",
            (output_dim, input_dim),
            MODEL_DTYPE,
            "torch_linear_out_in",
        ),
        LeafSpec(
            f"{semantic_prefix}.bias",
            f"{source_prefix}.bias",
            (output_dim,),
            MODEL_DTYPE,
            "vector",
        ),
    )


def _two_linear_specs(
    semantic_prefix: str,
    source_prefix: str,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> tuple[LeafSpec, ...]:
    return (
        *_linear_specs(
            f"{semantic_prefix}.hidden",
            f"{source_prefix}.0",
            input_dim=input_dim,
            output_dim=hidden_dim,
        ),
        *_linear_specs(
            f"{semantic_prefix}.output",
            f"{source_prefix}.2",
            input_dim=hidden_dim,
            output_dim=output_dim,
        ),
    )


def _attention_specs(
    semantic_prefix: str, source_prefix: str, *, size: int
) -> tuple[LeafSpec, ...]:
    result: list[LeafSpec] = []
    for name in ("query", "key", "value"):
        result.extend(
            _linear_specs(
                f"{semantic_prefix}.{name}",
                f"{source_prefix}.{name}",
                input_dim=size,
                output_dim=size,
            )
        )
    return tuple(result)


def _gru_specs(
    semantic_prefix: str,
    source_prefix: str,
    *,
    input_dim: int,
    hidden_dim: int,
) -> tuple[LeafSpec, ...]:
    packed = 3 * hidden_dim
    return (
        LeafSpec(
            f"{semantic_prefix}.input_kernel_rzn",
            f"{source_prefix}.weight_ih_l0",
            (packed, input_dim),
            MODEL_DTYPE,
            "torch_gru_3h_input",
        ),
        LeafSpec(
            f"{semantic_prefix}.recurrent_kernel_rzn",
            f"{source_prefix}.weight_hh_l0",
            (packed, hidden_dim),
            MODEL_DTYPE,
            "torch_gru_3h_h",
        ),
        LeafSpec(
            f"{semantic_prefix}.input_bias_rzn",
            f"{source_prefix}.bias_ih_l0",
            (packed,),
            MODEL_DTYPE,
            "vector",
        ),
        LeafSpec(
            f"{semantic_prefix}.recurrent_bias_rzn",
            f"{source_prefix}.bias_hh_l0",
            (packed,),
            MODEL_DTYPE,
            "vector",
        ),
    )


def _predator_actor_specs() -> tuple[LeafSpec, ...]:
    semantic = "model.predator_actor"
    source = "predator.policy"
    return (
        *_two_linear_specs(
            f"{semantic}.own_encoder",
            f"{source}.own_encoder",
            input_dim=12,
            hidden_dim=256,
            output_dim=128,
        ),
        *_two_linear_specs(
            f"{semantic}.prey_encoder",
            f"{source}.prey_encoder",
            input_dim=6,
            hidden_dim=256,
            output_dim=128,
        ),
        *_two_linear_specs(
            f"{semantic}.teammate_encoder",
            f"{source}.teammate_encoder",
            input_dim=6,
            hidden_dim=256,
            output_dim=128,
        ),
        *_attention_specs(f"{semantic}.attention", source, size=128),
        *_gru_specs(
            f"{semantic}.gru", f"{source}.gru", input_dim=256, hidden_dim=256
        ),
        *_two_linear_specs(
            f"{semantic}.head",
            f"{source}.actor_head",
            input_dim=256,
            hidden_dim=256,
            output_dim=4,
        ),
        LeafSpec(
            f"{semantic}.log_std",
            f"{source}.log_std_parameter",
            (4,),
            MODEL_DTYPE,
            "vector",
        ),
    )


def _prey_actor_specs() -> tuple[LeafSpec, ...]:
    semantic = "model.prey_actor"
    source = "prey.policy"
    return (
        *_two_linear_specs(
            f"{semantic}.own_encoder",
            f"{source}.own_encoder",
            input_dim=12,
            hidden_dim=256,
            output_dim=128,
        ),
        *_two_linear_specs(
            f"{semantic}.predator_encoder",
            f"{source}.predator_encoder",
            input_dim=6,
            hidden_dim=256,
            output_dim=128,
        ),
        *_attention_specs(f"{semantic}.attention", source, size=128),
        *_gru_specs(
            f"{semantic}.gru", f"{source}.gru", input_dim=256, hidden_dim=256
        ),
        *_two_linear_specs(
            f"{semantic}.head",
            f"{source}.actor_head",
            input_dim=256,
            hidden_dim=256,
            output_dim=4,
        ),
        LeafSpec(
            f"{semantic}.log_std",
            f"{source}.log_std_parameter",
            (4,),
            MODEL_DTYPE,
            "vector",
        ),
    )


def _critic_specs(role: str) -> tuple[LeafSpec, ...]:
    semantic = f"model.{role}_critic"
    source = f"{role}.value"
    return (
        *_two_linear_specs(
            f"{semantic}.predator_encoder",
            f"{source}.predator_encoder",
            input_dim=30,
            hidden_dim=128,
            output_dim=64,
        ),
        *_two_linear_specs(
            f"{semantic}.prey_encoder",
            f"{source}.prey_encoder",
            input_dim=30,
            hidden_dim=128,
            output_dim=64,
        ),
        *_attention_specs(f"{semantic}.attention", source, size=64),
        *_gru_specs(
            f"{semantic}.gru", f"{source}.gru", input_dim=64, hidden_dim=64
        ),
        *_two_linear_specs(
            f"{semantic}.head",
            f"{source}.value_head",
            input_dim=64,
            hidden_dim=128,
            output_dim=1,
        ),
    )


def _normalizer_specs() -> tuple[LeafSpec, ...]:
    mappings = (
        ("predator_actor", "predator.state_preprocessor", 90),
        ("prey_actor", "prey.state_preprocessor", 30),
        ("predator_critic", "predator.shared_state_preprocessor", 120),
        ("prey_critic", "prey.shared_state_preprocessor", 120),
        ("predator_value", "predator.value_preprocessor", 1),
        ("prey_value", "prey.value_preprocessor", 1),
    )
    result: list[LeafSpec] = []
    for semantic_name, source, feature_dim in mappings:
        semantic = f"normalizer.{semantic_name}"
        result.extend(
            (
                LeafSpec(
                    f"{semantic}.running_mean",
                    f"{source}.running_mean",
                    (feature_dim,),
                    NORMALIZER_DTYPE,
                    "vector",
                ),
                LeafSpec(
                    f"{semantic}.running_variance",
                    f"{source}.running_variance",
                    (feature_dim,),
                    NORMALIZER_DTYPE,
                    "vector",
                ),
                LeafSpec(
                    f"{semantic}.current_count",
                    f"{source}.current_count",
                    (),
                    NORMALIZER_DTYPE,
                    "scalar",
                ),
            )
        )
    return tuple(result)


LEAF_SPECS = (
    *_predator_actor_specs(),
    *_prey_actor_specs(),
    *_critic_specs("predator"),
    *_critic_specs("prey"),
    *_normalizer_specs(),
)

if len(LEAF_SPECS) != 112:
    raise AssertionError(f"transfer v0 must contain 112 tensors, got {len(LEAF_SPECS)}")
if len({spec.semantic_path for spec in LEAF_SPECS}) != len(LEAF_SPECS):
    raise AssertionError("transfer v0 semantic paths must be unique")
if len({spec.source_path for spec in LEAF_SPECS}) != len(LEAF_SPECS):
    raise AssertionError("transfer v0 source paths must be unique")


def export_skrl_policy_transfer_v0(
    *,
    bundle_root: str | os.PathLike[str],
    checkpoint: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> Path:
    """Verify and export one migration checkpoint as a neutral directory."""

    root = _require_real_directory(Path(bundle_root), label="migration bundle")
    checkpoint_path, logical_checkpoint_path = _require_bundle_member(
        root, Path(checkpoint), label="checkpoint"
    )
    manifest_path, _ = _require_bundle_member(
        root, root / MANIFEST_FILENAME, label="manifest"
    )
    checksums_path, _ = _require_bundle_member(
        root, root / CHECKSUMS_FILENAME, label="checksums"
    )

    checksums_bytes = _read_regular_file_no_follow(checksums_path)
    checksums_sha256 = _sha256_bytes(checksums_bytes)
    source_checksums = _parse_source_checksums(checksums_bytes)

    manifest_bytes = _read_regular_file_no_follow(manifest_path)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    expected_manifest_sha256 = source_checksums.get(MANIFEST_FILENAME)
    if expected_manifest_sha256 is None:
        raise ValueError("migration SHA256SUMS.txt does not contain MANIFEST.json")
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "migration manifest checksum mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    manifest = _load_json_bytes(manifest_bytes, label="migration manifest")
    manifest_record = _find_manifest_checkpoint_record(
        manifest, logical_checkpoint_path
    )

    expected_checkpoint_sha256 = _require_digest(
        manifest_record.get("sha256"), label="manifest checkpoint sha256"
    )
    checksum_checkpoint_sha256 = source_checksums.get(logical_checkpoint_path)
    if checksum_checkpoint_sha256 is None:
        raise ValueError(
            "migration SHA256SUMS.txt does not contain checkpoint "
            f"{logical_checkpoint_path!r}"
        )
    if checksum_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint digest differs between MANIFEST.json and "
            "SHA256SUMS.txt"
        )

    with _open_regular_file_no_follow(checkpoint_path) as stream:
        checkpoint_sha256, checkpoint_size = _sha256_stream(stream)
        if checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(
                "checkpoint checksum mismatch: "
                f"expected {expected_checkpoint_sha256}, got {checkpoint_sha256}"
            )
        manifest_size = manifest_record.get("size_bytes")
        if manifest_size is not None and manifest_size != checkpoint_size:
            raise ValueError(
                "checkpoint size differs from MANIFEST.json: "
                f"expected {manifest_size}, got {checkpoint_size}"
            )

        # Hash verification happens before deserialization, and the same open
        # O_NOFOLLOW file descriptor is reused to avoid a path substitution.
        stream.seek(0)
        loaded = torch.load(stream, map_location="cpu", weights_only=True)

    arrays, optimizer_modules = _extract_and_validate_tensors(loaded)
    source_runtime = _source_runtime_from_manifest(manifest)
    metadata = _build_metadata(
        logical_checkpoint_path=logical_checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_size,
        manifest_sha256=manifest_sha256,
        checksums_sha256=checksums_sha256,
        source_runtime=source_runtime,
        optimizer_modules=optimizer_modules,
    )
    return _write_transfer_directory(Path(output), arrays, metadata)


def verify_skrl_policy_transfer_v0(
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Strictly verify a completed transfer directory and return metadata."""

    transfer = _require_real_directory(Path(directory), label="transfer artifact")
    entries = {entry.name for entry in transfer.iterdir()}
    if entries != TRANSFER_FILENAMES:
        raise ValueError(
            "transfer directory members do not match schema v0: "
            f"expected {sorted(TRANSFER_FILENAMES)}, got {sorted(entries)}"
        )
    for filename in TRANSFER_FILENAMES:
        member = transfer / filename
        if member.is_symlink() or not member.is_file():
            raise ValueError(f"transfer member must be a regular file: {member}")

    snapshots: dict[str, tuple[bytes, tuple[int, ...]]] = {}
    for filename in (CHECKSUMS_FILENAME, *PAYLOAD_FILENAMES):
        snapshots[filename] = _read_regular_file_snapshot(transfer / filename)

    checksum_bytes = snapshots[CHECKSUMS_FILENAME][0]
    expected_checksums = _parse_output_checksums(checksum_bytes)
    for filename in PAYLOAD_FILENAMES:
        actual = _sha256_bytes(snapshots[filename][0])
        if actual != expected_checksums[filename]:
            raise ValueError(
                f"transfer checksum mismatch for {filename}: "
                f"expected {expected_checksums[filename]}, got {actual}"
            )

    metadata = _load_json_bytes(
        snapshots[METADATA_FILENAME][0],
        label="transfer metadata",
    )
    _validate_metadata(metadata)

    arrays_bytes = snapshots[ARRAYS_FILENAME][0]
    expected_names = [
        f"leaf_{index:06d}" for index in range(len(LEAF_SPECS))
    ]
    _validate_npz_container(arrays_bytes, expected_names=expected_names)
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
        if archive.files != expected_names:
            raise ValueError(
                "transfer NPZ members do not match the ordered static "
                "112-leaf schema"
            )
        for index, spec in enumerate(LEAF_SPECS):
            array = archive[f"leaf_{index:06d}"]
            _validate_numpy_array(array, spec)

    # Recheck both path identity and content after parsing.  This detects an
    # in-place write or atomic path substitution during verification while
    # ensuring all parsing above used the exact bytes covered by the checksums.
    for filename, (initial_bytes, initial_identity) in snapshots.items():
        final_digest, final_identity = _sha256_regular_file_snapshot(
            transfer / filename
        )
        if (
            final_identity != initial_identity
            or final_digest != _sha256_bytes(initial_bytes)
        ):
            raise ValueError(
                f"transfer artifact changed during verification: {filename}"
            )
    return metadata


def _validate_npz_container(
    content: bytes, *, expected_names: list[str]
) -> None:
    """Validate the schema-v0 ZIP envelope before NumPy parses its arrays."""

    expected_members = [f"{name}.npy" for name in expected_names]
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            infos = archive.infolist()
            actual_members = [info.filename for info in infos]
            if actual_members != expected_members:
                raise ValueError(
                    "transfer NPZ members must match the exact ordered NPY "
                    "member list"
                )
            for info in infos:
                if info.is_dir():
                    raise ValueError(
                        f"transfer NPZ member must be regular: {info.filename}"
                    )
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(
                        "transfer NPZ members must be uncompressed ZIP_STORED "
                        f"entries: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"transfer NPZ member must not be encrypted: {info.filename}"
                    )
                # NumPy ZIP entries normally encode only permission bits, so a
                # missing type is accepted; an explicit non-regular type is not.
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type not in (0, stat.S_IFREG):
                    raise ValueError(
                        f"transfer NPZ member must be regular: {info.filename}"
                    )
    except zipfile.BadZipFile as error:
        raise ValueError("transfer arrays.npz is not a valid ZIP archive") from error


def _extract_and_validate_tensors(
    checkpoint: Any,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    if set(checkpoint) != set(ROLES):
        raise ValueError(
            f"checkpoint roles must be exactly {list(ROLES)}, got {sorted(checkpoint)}"
        )

    expected_keys: dict[tuple[str, str], set[str]] = {}
    for spec in LEAF_SPECS:
        role, module, tensor_name = _split_source_path(spec.source_path)
        expected_keys.setdefault((role, module), set()).add(tensor_name)

    required_modules = {
        "policy",
        "value",
        "state_preprocessor",
        "shared_state_preprocessor",
        "value_preprocessor",
    }
    optimizer_modules: dict[str, str] = {}
    for role in ROLES:
        role_state = checkpoint[role]
        if not isinstance(role_state, Mapping):
            raise TypeError(f"checkpoint role {role!r} must be a mapping")
        actual_modules = set(role_state)
        missing_modules = required_modules - actual_modules
        extra_modules = actual_modules - required_modules - {"optimizer"}
        if missing_modules or extra_modules:
            raise ValueError(
                f"checkpoint role {role!r} module mismatch: "
                f"missing={sorted(missing_modules)}, extra={sorted(extra_modules)}"
            )
        if "optimizer" in role_state:
            optimizer = role_state["optimizer"]
            if not isinstance(optimizer, Mapping) or set(optimizer) != {
                "state",
                "param_groups",
            }:
                raise ValueError(
                    f"checkpoint role {role!r} optimizer must be a standard "
                    "state/param_groups mapping"
                )
            optimizer_modules[role] = "present_excluded"
        else:
            optimizer_modules[role] = "absent"

        for module in required_modules:
            module_state = role_state[module]
            if not isinstance(module_state, Mapping):
                raise TypeError(f"checkpoint {role}.{module} must be a mapping")
            expected = expected_keys[(role, module)]
            if set(module_state) != expected:
                raise ValueError(
                    f"checkpoint {role}.{module} tensor keys do not match v0: "
                    f"missing={sorted(expected - set(module_state))}, "
                    f"extra={sorted(set(module_state) - expected)}"
                )

    arrays: dict[str, np.ndarray] = {}
    for index, spec in enumerate(LEAF_SPECS):
        role, module, tensor_name = _split_source_path(spec.source_path)
        tensor = checkpoint[role][module][tensor_name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint tensor {spec.source_path} is not a Tensor")
        expected_torch_dtype = (
            torch.float32 if spec.stored_dtype == MODEL_DTYPE else torch.float64
        )
        if tuple(tensor.shape) != spec.stored_shape:
            raise ValueError(
                f"checkpoint tensor {spec.source_path} has shape "
                f"{tuple(tensor.shape)}, expected {spec.stored_shape}"
            )
        if tensor.dtype != expected_torch_dtype:
            raise ValueError(
                f"checkpoint tensor {spec.source_path} has dtype {tensor.dtype}, "
                f"expected {expected_torch_dtype}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint tensor {spec.source_path} is not finite")

        if spec.semantic_path.endswith(".log_std") and not bool(
            ((tensor >= MIN_LOG_STD) & (tensor <= MAX_LOG_STD)).all()
        ):
            raise ValueError(
                f"checkpoint tensor {spec.source_path} must be within "
                f"[{MIN_LOG_STD}, {MAX_LOG_STD}]"
            )
        if spec.semantic_path.endswith(".running_variance") and not bool(
            (tensor > 0).all()
        ):
            raise ValueError(
                f"checkpoint tensor {spec.source_path} must be strictly positive"
            )
        if spec.semantic_path.endswith(".current_count") and not bool(tensor > 0):
            raise ValueError(
                f"checkpoint tensor {spec.source_path} must be strictly positive"
            )

        arrays[f"leaf_{index:06d}"] = (
            tensor.detach().cpu().contiguous().numpy().copy()
        )
    return arrays, optimizer_modules


def _build_metadata(
    *,
    logical_checkpoint_path: str,
    checkpoint_sha256: str,
    checkpoint_size: int,
    manifest_sha256: str,
    checksums_sha256: str,
    source_runtime: Mapping[str, str],
    optimizer_modules: Mapping[str, str],
) -> dict[str, Any]:
    leaves = []
    for index, spec in enumerate(LEAF_SPECS):
        leaves.append(
            {
                "index": index,
                "array_name": f"leaf_{index:06d}",
                "semantic_path": spec.semantic_path,
                "source_path": spec.source_path,
                "stored_shape": list(spec.stored_shape),
                "stored_dtype": spec.stored_dtype,
                "source_layout": spec.source_layout,
            }
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": {
            "path": logical_checkpoint_path,
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
            "manifest_path": MANIFEST_FILENAME,
            "manifest_sha256": manifest_sha256,
            "checksums_path": CHECKSUMS_FILENAME,
            "checksums_sha256": checksums_sha256,
        },
        "source_runtime": dict(source_runtime),
        "transfer_contract": {
            "model_contract": MODEL_CONTRACT,
            "roles": list(ROLES),
            "gru_gate_order": ["reset", "update", "new"],
            "model_dtype": MODEL_DTYPE,
            "normalizer_dtype": NORMALIZER_DTYPE,
            "optimizer_state": "absent_reset_required_for_training",
            "optimizer_modules": dict(optimizer_modules),
        },
        "leaf_count": len(LEAF_SPECS),
        "leaves": leaves,
    }
    _validate_metadata(metadata)
    return metadata


def _write_transfer_directory(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(os.path.abspath(destination.expanduser()))
    if os.path.lexists(destination):
        raise FileExistsError(f"transfer destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError(
            f"transfer parent must be a real directory: {destination.parent}"
        )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
    )
    try:
        arrays_path = temporary / ARRAYS_FILENAME
        np.savez(arrays_path, **arrays)
        _fsync_file(arrays_path)

        metadata_text = (
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        _write_text_fsynced(temporary / METADATA_FILENAME, metadata_text)

        checksum_lines = [
            f"{_sha256_file(temporary / filename)}  {filename}"
            for filename in PAYLOAD_FILENAMES
        ]
        _write_text_fsynced(
            temporary / CHECKSUMS_FILENAME, "\n".join(checksum_lines) + "\n"
        )
        _fsync_directory(temporary)
        verify_skrl_policy_transfer_v0(temporary)

        _publish_directory_no_replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, Mapping) or set(metadata) != _TOP_LEVEL_METADATA_KEYS:
        raise ValueError("transfer metadata keys do not match schema v0")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported transfer schema: {metadata['schema_version']!r}"
        )
    if metadata["leaf_count"] != len(LEAF_SPECS):
        raise ValueError("transfer metadata leaf_count must be 112")
    if not isinstance(metadata["leaves"], list) or len(metadata["leaves"]) != len(
        LEAF_SPECS
    ):
        raise ValueError("transfer metadata leaves must contain 112 records")

    source = metadata["source_checkpoint"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_CHECKPOINT_KEYS:
        raise ValueError("source_checkpoint keys do not match schema v0")
    if not _is_safe_relative_posix_path(source["path"]):
        raise ValueError("source checkpoint path must be a safe relative POSIX path")
    if source["manifest_path"] != MANIFEST_FILENAME:
        raise ValueError("source manifest_path must be MANIFEST.json")
    if source["checksums_path"] != CHECKSUMS_FILENAME:
        raise ValueError("source checksums_path must be SHA256SUMS.txt")
    for name in ("sha256", "manifest_sha256", "checksums_sha256"):
        _require_digest(source[name], label=f"source_checkpoint.{name}")
    if (
        not isinstance(source["size_bytes"], int)
        or isinstance(source["size_bytes"], bool)
        or source["size_bytes"] <= 0
    ):
        raise ValueError("source checkpoint size_bytes must be positive")

    runtime = metadata["source_runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != _SOURCE_RUNTIME_KEYS:
        raise ValueError("source_runtime keys do not match schema v0")
    if dict(runtime) != EXPECTED_SOURCE_RUNTIME:
        raise ValueError(
            f"source_runtime must match the migrated v0 runtime: "
            f"{EXPECTED_SOURCE_RUNTIME}"
        )

    contract = metadata["transfer_contract"]
    if not isinstance(contract, Mapping) or set(contract) != _TRANSFER_CONTRACT_KEYS:
        raise ValueError("transfer_contract keys do not match schema v0")
    expected_contract_values = {
        "model_contract": MODEL_CONTRACT,
        "roles": list(ROLES),
        "gru_gate_order": ["reset", "update", "new"],
        "model_dtype": MODEL_DTYPE,
        "normalizer_dtype": NORMALIZER_DTYPE,
        "optimizer_state": "absent_reset_required_for_training",
    }
    for key, expected in expected_contract_values.items():
        if contract[key] != expected:
            raise ValueError(f"transfer_contract.{key} must be {expected!r}")
    optimizer_modules = contract["optimizer_modules"]
    if not isinstance(optimizer_modules, Mapping) or set(optimizer_modules) != set(
        ROLES
    ):
        raise ValueError("optimizer_modules must contain predator and prey")
    if any(value not in _OPTIMIZER_MODULE_VALUES for value in optimizer_modules.values()):
        raise ValueError("optimizer_modules contains an unsupported status")

    for index, (record, spec) in enumerate(
        zip(metadata["leaves"], LEAF_SPECS, strict=True)
    ):
        if not isinstance(record, Mapping) or set(record) != _LEAF_RECORD_KEYS:
            raise ValueError(f"transfer leaf {index} keys do not match schema v0")
        expected = {
            "index": index,
            "array_name": f"leaf_{index:06d}",
            "semantic_path": spec.semantic_path,
            "source_path": spec.source_path,
            "stored_shape": list(spec.stored_shape),
            "stored_dtype": spec.stored_dtype,
            "source_layout": spec.source_layout,
        }
        if dict(record) != expected:
            raise ValueError(f"transfer leaf {index} does not match static schema v0")
        if record["source_layout"] not in _SOURCE_LAYOUTS:
            raise ValueError(f"transfer leaf {index} has an unsupported source layout")


def _validate_numpy_array(array: np.ndarray, spec: LeafSpec) -> None:
    if tuple(array.shape) != spec.stored_shape:
        raise ValueError(
            f"transfer array {spec.semantic_path} has shape {array.shape}, "
            f"expected {spec.stored_shape}"
        )
    if str(array.dtype) != spec.stored_dtype:
        raise ValueError(
            f"transfer array {spec.semantic_path} has dtype {array.dtype}, "
            f"expected {spec.stored_dtype}"
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"transfer array {spec.semantic_path} is not finite")
    if spec.semantic_path.endswith(".log_std") and not bool(
        ((array >= MIN_LOG_STD) & (array <= MAX_LOG_STD)).all()
    ):
        raise ValueError(f"transfer array {spec.semantic_path} violates log-std bounds")
    if spec.semantic_path.endswith(".running_variance") and not bool(
        (array > 0).all()
    ):
        raise ValueError(f"transfer array {spec.semantic_path} must be positive")
    if spec.semantic_path.endswith(".current_count") and not bool(array > 0):
        raise ValueError(f"transfer array {spec.semantic_path} must be positive")


def _source_runtime_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("migration manifest runtime must be an object")
    source_runtime = {
        "python_version": runtime.get("python_version"),
        "pytorch_version": runtime.get("torch_version"),
        "skrl_version": runtime.get("skrl_version"),
    }
    if source_runtime != EXPECTED_SOURCE_RUNTIME:
        raise ValueError(
            "migration manifest runtime is incompatible with transfer v0: "
            f"expected {EXPECTED_SOURCE_RUNTIME}, got {source_runtime}"
        )
    return source_runtime


def _find_manifest_checkpoint_record(
    manifest: Any, logical_path: str
) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("migration manifest root must be an object")
    candidates: list[Mapping[str, Any]] = []
    current = manifest.get("current_checkpoint")
    if isinstance(current, Mapping):
        candidates.append(current)
    opponent_pool = manifest.get("opponent_pool")
    if isinstance(opponent_pool, Mapping):
        entries = opponent_pool.get("entries")
        if isinstance(entries, list):
            candidates.extend(entry for entry in entries if isinstance(entry, Mapping))
    matching = [entry for entry in candidates if entry.get("bundle_path") == logical_path]
    if len(matching) != 1:
        raise ValueError(
            f"migration manifest must contain exactly one checkpoint record for "
            f"{logical_path!r}; found {len(matching)}"
        )
    return matching[0]


def _parse_source_checksums(content: bytes) -> dict[str, str]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("migration SHA256SUMS.txt is not valid UTF-8") from error
    if not lines:
        raise ValueError("migration SHA256SUMS.txt is empty")
    result: dict[str, str] = {}
    for line in lines:
        match = _INPUT_CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid migration checksum line: {line!r}")
        digest, path = match.groups()
        if not _is_safe_relative_posix_path(path):
            raise ValueError(f"unsafe migration checksum path: {path!r}")
        if path in result:
            raise ValueError(f"duplicate migration checksum path: {path!r}")
        result[path] = digest
    return result


def _parse_output_checksums(content: bytes) -> dict[str, str]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("transfer SHA256SUMS.txt is not valid UTF-8") from error
    if len(lines) != len(PAYLOAD_FILENAMES):
        raise ValueError("transfer SHA256SUMS.txt must contain exactly two lines")
    result: dict[str, str] = {}
    for line, expected_filename in zip(lines, PAYLOAD_FILENAMES, strict=True):
        match = _OUTPUT_CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid transfer checksum line: {line!r}")
        digest, filename = match.groups()
        if filename != expected_filename:
            raise ValueError(
                "transfer checksum order must be arrays.npz then metadata.json"
            )
        result[filename] = digest
    return result


def _split_source_path(path: str) -> tuple[str, str, str]:
    role, module, tensor_name = path.split(".", 2)
    return role, module, tensor_name


def _require_real_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.is_symlink() or not absolute.is_dir():
        raise FileNotFoundError(f"{label} must be a real directory: {absolute}")
    return absolute


def _require_bundle_member(
    root: Path, path: Path, *, label: str
) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside migration bundle {root}") from error
    if not relative.parts:
        raise ValueError(f"{label} must name a file inside the migration bundle")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks: {current}")
    if not absolute.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {absolute}")
    logical = PurePosixPath(*relative.parts).as_posix()
    if not _is_safe_relative_posix_path(logical):
        raise ValueError(f"{label} has an unsafe relative path: {logical!r}")
    return absolute, logical


def _is_safe_relative_posix_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def _open_regular_file_no_follow(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"path must be a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_no_follow(path: Path) -> bytes:
    with _open_regular_file_no_follow(path) as stream:
        return stream.read()


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_regular_file_snapshot(path: Path) -> tuple[bytes, tuple[int, ...]]:
    with _open_regular_file_no_follow(path) as stream:
        before = os.fstat(stream.fileno())
        content = stream.read()
        after = os.fstat(stream.fileno())
    before_identity = _file_identity(before)
    after_identity = _file_identity(after)
    if before_identity != after_identity or len(content) != after.st_size:
        raise ValueError(f"regular file changed while reading: {path}")
    return content, after_identity


def _sha256_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> str:
    with _open_regular_file_no_follow(path) as stream:
        digest, _ = _sha256_stream(stream)
    return digest


def _sha256_regular_file_snapshot(path: Path) -> tuple[str, tuple[int, ...]]:
    with _open_regular_file_no_follow(path) as stream:
        before = os.fstat(stream.fileno())
        digest, size = _sha256_stream(stream)
        after = os.fstat(stream.fileno())
    before_identity = _file_identity(before)
    after_identity = _file_identity(after)
    if before_identity != after_identity or size != after.st_size:
        raise ValueError(f"regular file changed while hashing: {path}")
    return digest, after_identity


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json_bytes(content: bytes, *, label: str) -> Any:
    def reject_constant(value: str):
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        return json.loads(
            content.decode("utf-8"), parse_constant=reject_constant
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _write_text_fsynced(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    at_fdcwd = -100
    rename_noreplace = 1
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace directory publication requires renameat2",
            destination,
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"transfer destination appeared during save: {destination}",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help="Verified Linux migration bundle containing MANIFEST.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint inside --bundle-root to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New transfer directory; an existing path is never replaced.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        output = export_skrl_policy_transfer_v0(
            bundle_root=args.bundle_root,
            checkpoint=args.checkpoint,
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
