#!/usr/bin/env python3
"""Export a deterministic skrl checkpoint-forward oracle for JAX transfer.

This utility deliberately does not import the Isaac application or the task
package ``__init__``. It loads the project's actual recurrent attention model
source as an isolated Python submodule, restores a SHA-256-verified checkpoint
with PyTorch's weights-only loader, and evaluates a fixed float32 batch.

The resulting directory is a pickle-free, checksummed, no-clobber fixture. It
contains inputs and PyTorch reference outputs only; model parameters remain in
the separately verified neutral transfer artifact.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
from typing import Any, Mapping

import gymnasium
from gymnasium.spaces import Box
import numpy as np
import skrl
import torch


SCHEMA_VERSION = "uavpredatorprey.skrl_policy_forward_oracle.v0"
ARRAYS_FILENAME = "arrays.npz"
METADATA_FILENAME = "metadata.json"
CHECKSUMS_FILENAME = "SHA256SUMS.txt"
FIXTURE_FILENAMES = frozenset(
    (ARRAYS_FILENAME, METADATA_FILENAME, CHECKSUMS_FILENAME)
)

DEFAULT_CHECKPOINT = (
    ".pretrained_checkpoints/linux_migration_2026-07-11/current/"
    "agent_115200.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "88e7161b48323452c5d302b25a8f50d1fd5ac026fdeddd2819d64eeb2521ba98"
)
DEFAULT_OUTPUT = (
    "/workspace/artifacts/transfer/skrl_policy_oracle_v0/"
    "windows_agent_115200"
)

BATCH_SIZE = 4
PREDATOR_COUNT = 3
PREDATOR_OBSERVATION_DIM = 90
PREY_OBSERVATION_DIM = 30
CENTRALIZED_STATE_DIM = 120
ACTION_DIM = 4
ACTOR_HIDDEN_DIM = 256
CRITIC_HIDDEN_DIM = 64
NORMALIZER_EPSILON = 1.0e-8
NORMALIZER_CLIP_THRESHOLD = 5.0

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ARRAY_SHAPES: dict[str, tuple[int, ...]] = {
    "input.predator_observation_raw": (BATCH_SIZE, PREDATOR_OBSERVATION_DIM),
    "input.prey_observation_raw": (BATCH_SIZE, PREY_OBSERVATION_DIM),
    "input.centralized_state_raw": (BATCH_SIZE, CENTRALIZED_STATE_DIM),
    "input.predator_actor_hidden": (
        BATCH_SIZE,
        PREDATOR_COUNT,
        ACTOR_HIDDEN_DIM,
    ),
    "input.prey_actor_hidden": (BATCH_SIZE, ACTOR_HIDDEN_DIM),
    "input.predator_critic_hidden": (BATCH_SIZE, CRITIC_HIDDEN_DIM),
    "input.prey_critic_hidden": (BATCH_SIZE, CRITIC_HIDDEN_DIM),
    "expected.predator_actor_input_normalized": (
        BATCH_SIZE,
        PREDATOR_OBSERVATION_DIM,
    ),
    "expected.prey_actor_input_normalized": (
        BATCH_SIZE,
        PREY_OBSERVATION_DIM,
    ),
    "expected.predator_critic_input_normalized": (
        BATCH_SIZE,
        CENTRALIZED_STATE_DIM,
    ),
    "expected.prey_critic_input_normalized": (
        BATCH_SIZE,
        CENTRALIZED_STATE_DIM,
    ),
    "expected.predator_actor_latent_mean": (
        BATCH_SIZE,
        PREDATOR_COUNT,
        ACTION_DIM,
    ),
    "expected.predator_actor_log_std": (
        BATCH_SIZE,
        PREDATOR_COUNT,
        ACTION_DIM,
    ),
    "expected.predator_actor_deterministic_action": (
        BATCH_SIZE,
        PREDATOR_COUNT,
        ACTION_DIM,
    ),
    "expected.predator_actor_next_hidden": (
        BATCH_SIZE,
        PREDATOR_COUNT,
        ACTOR_HIDDEN_DIM,
    ),
    "expected.prey_actor_latent_mean": (BATCH_SIZE, ACTION_DIM),
    "expected.prey_actor_log_std": (BATCH_SIZE, ACTION_DIM),
    "expected.prey_actor_deterministic_action": (BATCH_SIZE, ACTION_DIM),
    "expected.prey_actor_next_hidden": (BATCH_SIZE, ACTOR_HIDDEN_DIM),
    "expected.predator_critic_value_normalized": (BATCH_SIZE, 1),
    "expected.predator_critic_value_inverse": (BATCH_SIZE, 1),
    "expected.predator_critic_next_hidden": (BATCH_SIZE, CRITIC_HIDDEN_DIM),
    "expected.prey_critic_value_normalized": (BATCH_SIZE, 1),
    "expected.prey_critic_value_inverse": (BATCH_SIZE, 1),
    "expected.prey_critic_next_hidden": (BATCH_SIZE, CRITIC_HIDDEN_DIM),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT),
        help="source skrl .pt checkpoint",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_CHECKPOINT_SHA256,
        help="required SHA-256 identity of the exact source checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="new fixture directory outside this Git repository",
    )
    return parser.parse_args(argv)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repository: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _containing_git_worktree(path: Path) -> Path | None:
    """Return the Git top-level containing an existing ancestor, if any."""

    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        top_level = subprocess.check_output(
            ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(top_level).resolve()


def _safe_load_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], str, int]:
    expected_sha256 = expected_sha256.strip().lower()
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("--expected-checkpoint-sha256 must be 64 lowercase hex characters")

    checkpoint_argument = checkpoint_path.expanduser()
    if checkpoint_argument.is_symlink():
        raise ValueError(f"checkpoint path must not be a symlink: {checkpoint_argument}")
    checkpoint = checkpoint_argument.resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint must be a regular file: {checkpoint}")

    # Loading the already-hashed bytes closes the hash/load time-of-check race.
    payload = checkpoint.read_bytes()
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    checkpoint_state = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    return checkpoint_state, actual_sha256, len(payload)


def _load_attention_models_module(attention_models_path: Path):
    """Load the actual model file without importing its Isaac task package."""

    source_path = attention_models_path.resolve(strict=True)
    agents_directory = source_path.parent
    package_name = "_uavpredatorprey_policy_oracle_agents"
    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = types.ModuleType(package_name)
    package.__file__ = str(agents_directory / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(agents_directory)]
    sys.modules[package_name] = package

    qualified_name = f"{package_name}.attention_models"
    specification = importlib.util.spec_from_file_location(
        qualified_name,
        source_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load attention model source: {source_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[qualified_name] = module
    specification.loader.exec_module(module)
    if Path(module.__file__).resolve() != source_path:
        raise RuntimeError("loaded attention model module has an unexpected origin")
    return module


def _box(feature_dim: int, *, actions: bool = False) -> Box:
    if actions:
        return Box(-1.0, 1.0, shape=(feature_dim,), dtype=np.float32)
    return Box(-np.inf, np.inf, shape=(feature_dim,), dtype=np.float32)


def _require_exact_state_dict(
    model: torch.nn.Module,
    state: Any,
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} state must be a mapping")
    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"{label} state keys do not match the exact recurrent architecture; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, template in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}.{name} must be a tensor")
        if value.shape != template.shape:
            raise ValueError(
                f"{label}.{name} shape mismatch: expected {tuple(template.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"{label}.{name} must have dtype torch.float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label}.{name} contains a non-finite value")
    log_std = state.get("log_std_parameter")
    if log_std is not None and not bool(((log_std >= -2.0) & (log_std <= 0.0)).all()):
        raise ValueError(f"{label}.log_std_parameter is outside [-2, 0]")

    model.load_state_dict(state, strict=True)
    model.eval()


def _models(module: Any, checkpoint_state: Mapping[str, Any]):
    required_roles = {"predator", "prey"}
    if not required_roles.issubset(checkpoint_state):
        raise ValueError("checkpoint must contain predator and prey role mappings")

    actor_kwargs = {
        "hidden_size": 256,
        "attention_size": 128,
        "rnn_hidden_size": 256,
        "rnn_sequence_length": 24,
        "fallback_activation": "elu",
        "attention_min_predators": 2,
        "prey_attention_min_predators": 2,
        "clip_actions": False,
        "clip_log_std": True,
        "min_log_std": -2.0,
        "max_log_std": 0.0,
        "initial_log_std": -0.5,
        "reduction": "sum",
    }
    critic_kwargs = {
        "hidden_size": 128,
        "attention_size": 64,
        "rnn_hidden_size": 64,
        "rnn_sequence_length": 24,
        "fallback_activation": "elu",
        "max_predators": 6,
        "clip_actions": False,
    }

    predator_actor = module.RecurrentPredatorPreyAttentionGaussianModel(
        _box(PREDATOR_OBSERVATION_DIM),
        _box(PREDATOR_COUNT * ACTION_DIM, actions=True),
        device="cpu",
        **actor_kwargs,
    )
    prey_actor = module.RecurrentPredatorPreyAttentionGaussianModel(
        _box(PREY_OBSERVATION_DIM),
        _box(ACTION_DIM, actions=True),
        device="cpu",
        **actor_kwargs,
    )
    predator_critic = module.RecurrentEntityAttentionCentralizedValueModel(
        _box(CENTRALIZED_STATE_DIM),
        _box(PREDATOR_COUNT * ACTION_DIM, actions=True),
        device="cpu",
        **critic_kwargs,
    )
    prey_critic = module.RecurrentEntityAttentionCentralizedValueModel(
        _box(CENTRALIZED_STATE_DIM),
        _box(ACTION_DIM, actions=True),
        device="cpu",
        **critic_kwargs,
    )

    if predator_actor.mode != "predator" or prey_actor.mode != "prey":
        raise RuntimeError("actor spaces did not select the intended entity layouts")
    if not predator_critic.uses_entity_attention or not prey_critic.uses_entity_attention:
        raise RuntimeError("critic spaces did not select the intended entity layout")

    for role, role_state, actor, critic in (
        ("predator", checkpoint_state["predator"], predator_actor, predator_critic),
        ("prey", checkpoint_state["prey"], prey_actor, prey_critic),
    ):
        if not isinstance(role_state, Mapping):
            raise TypeError(f"checkpoint role {role!r} must be a mapping")
        _require_exact_state_dict(actor, role_state.get("policy"), label=f"{role}.policy")
        _require_exact_state_dict(critic, role_state.get("value"), label=f"{role}.value")

    return predator_actor, prey_actor, predator_critic, prey_critic


def _pattern(rows: int, columns: int, *, phase: float, scale: float) -> np.ndarray:
    grid = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
    values = scale * (
        np.sin(grid * np.float32(0.173) + np.float32(phase))
        + np.float32(0.37)
        * np.cos(grid * np.float32(0.071) - np.float32(0.5 * phase))
    )
    values = np.asarray(values, dtype=np.float32)
    values[0] = np.float32(0.0)
    return values


def _input_arrays() -> dict[str, np.ndarray]:
    predator_observation = _pattern(
        BATCH_SIZE, PREDATOR_OBSERVATION_DIM, phase=0.3, scale=3.25
    )
    prey_observation = _pattern(
        BATCH_SIZE, PREY_OBSERVATION_DIM, phase=-0.7, scale=2.75
    )
    centralized_state = _pattern(
        BATCH_SIZE, CENTRALIZED_STATE_DIM, phase=1.1, scale=4.0
    )

    # The final row intentionally reaches both normalizer clipping sides.
    predator_observation[-1] = np.where(
        np.arange(PREDATOR_OBSERVATION_DIM) % 2 == 0, 75.0, -75.0
    ).astype(np.float32)
    prey_observation[-1] = np.where(
        np.arange(PREY_OBSERVATION_DIM) % 2 == 0, -65.0, 65.0
    ).astype(np.float32)
    centralized_state[-1] = np.where(
        np.arange(CENTRALIZED_STATE_DIM) % 3 == 0, 90.0, -55.0
    ).astype(np.float32)

    return {
        "input.predator_observation_raw": predator_observation,
        "input.prey_observation_raw": prey_observation,
        "input.centralized_state_raw": centralized_state,
        "input.predator_actor_hidden": _pattern(
            BATCH_SIZE,
            PREDATOR_COUNT * ACTOR_HIDDEN_DIM,
            phase=0.2,
            scale=0.16,
        ).reshape(BATCH_SIZE, PREDATOR_COUNT, ACTOR_HIDDEN_DIM),
        "input.prey_actor_hidden": _pattern(
            BATCH_SIZE, ACTOR_HIDDEN_DIM, phase=-0.4, scale=0.13
        ),
        "input.predator_critic_hidden": _pattern(
            BATCH_SIZE, CRITIC_HIDDEN_DIM, phase=0.8, scale=0.11
        ),
        "input.prey_critic_hidden": _pattern(
            BATCH_SIZE, CRITIC_HIDDEN_DIM, phase=-1.2, scale=0.09
        ),
    }


def _scaler_state(
    checkpoint_state: Mapping[str, Any],
    *,
    role: str,
    name: str,
    feature_dim: int,
) -> Mapping[str, torch.Tensor]:
    role_state = checkpoint_state[role]
    scaler = role_state.get(name)
    if not isinstance(scaler, Mapping):
        raise TypeError(f"{role}.{name} must be a mapping")
    if set(scaler) != {"running_mean", "running_variance", "current_count"}:
        raise ValueError(f"{role}.{name} has unexpected scaler fields")
    expected_shapes = {
        "running_mean": (feature_dim,),
        "running_variance": (feature_dim,),
        "current_count": (),
    }
    for field, expected_shape in expected_shapes.items():
        value = scaler[field]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{role}.{name}.{field} must be a tensor")
        if value.shape != expected_shape or value.dtype != torch.float64:
            raise ValueError(
                f"{role}.{name}.{field} must be float64 with shape {expected_shape}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{role}.{name}.{field} contains a non-finite value")
    if not bool((scaler["running_variance"] >= 0.0).all()):
        raise ValueError(f"{role}.{name}.running_variance must be non-negative")
    if not bool(scaler["current_count"] > 0.0):
        raise ValueError(f"{role}.{name}.current_count must be positive")
    return scaler


def _normalize_exact_skrl(
    samples: torch.Tensor,
    scaler: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return torch.clamp(
        (samples - scaler["running_mean"].float())
        / (torch.sqrt(scaler["running_variance"].float()) + NORMALIZER_EPSILON),
        min=-NORMALIZER_CLIP_THRESHOLD,
        max=NORMALIZER_CLIP_THRESHOLD,
    )


def _inverse_exact_skrl(
    normalized_samples: torch.Tensor,
    scaler: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return (
        torch.sqrt(scaler["running_variance"].float())
        * torch.clamp(
            normalized_samples,
            min=-NORMALIZER_CLIP_THRESHOLD,
            max=NORMALIZER_CLIP_THRESHOLD,
        )
        + scaler["running_mean"].float()
    )


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    result = tensor.detach().cpu().numpy()
    if result.dtype != np.float32:
        result = result.astype(np.float32)
    return np.ascontiguousarray(result)


def _evaluate_checkpoint(
    checkpoint_state: Mapping[str, Any],
    attention_models: Any,
) -> dict[str, np.ndarray]:
    predator_actor, prey_actor, predator_critic, prey_critic = _models(
        attention_models, checkpoint_state
    )
    arrays = _input_arrays()
    tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}

    predator_actor_scaler = _scaler_state(
        checkpoint_state,
        role="predator",
        name="state_preprocessor",
        feature_dim=PREDATOR_OBSERVATION_DIM,
    )
    prey_actor_scaler = _scaler_state(
        checkpoint_state,
        role="prey",
        name="state_preprocessor",
        feature_dim=PREY_OBSERVATION_DIM,
    )
    predator_critic_scaler = _scaler_state(
        checkpoint_state,
        role="predator",
        name="shared_state_preprocessor",
        feature_dim=CENTRALIZED_STATE_DIM,
    )
    prey_critic_scaler = _scaler_state(
        checkpoint_state,
        role="prey",
        name="shared_state_preprocessor",
        feature_dim=CENTRALIZED_STATE_DIM,
    )
    predator_value_scaler = _scaler_state(
        checkpoint_state,
        role="predator",
        name="value_preprocessor",
        feature_dim=1,
    )
    prey_value_scaler = _scaler_state(
        checkpoint_state,
        role="prey",
        name="value_preprocessor",
        feature_dim=1,
    )

    predator_actor_input = _normalize_exact_skrl(
        tensors["input.predator_observation_raw"], predator_actor_scaler
    )
    prey_actor_input = _normalize_exact_skrl(
        tensors["input.prey_observation_raw"], prey_actor_scaler
    )
    predator_critic_input = _normalize_exact_skrl(
        tensors["input.centralized_state_raw"], predator_critic_scaler
    )
    prey_critic_input = _normalize_exact_skrl(
        tensors["input.centralized_state_raw"], prey_critic_scaler
    )

    predator_hidden = tensors["input.predator_actor_hidden"].reshape(
        1, BATCH_SIZE, PREDATOR_COUNT * ACTOR_HIDDEN_DIM
    )
    prey_hidden = tensors["input.prey_actor_hidden"].reshape(
        1, BATCH_SIZE, ACTOR_HIDDEN_DIM
    )
    predator_critic_hidden = tensors["input.predator_critic_hidden"].reshape(
        1, BATCH_SIZE, CRITIC_HIDDEN_DIM
    )
    prey_critic_hidden = tensors["input.prey_critic_hidden"].reshape(
        1, BATCH_SIZE, CRITIC_HIDDEN_DIM
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        predator_mean_flat, predator_log_std_flat, predator_extra = predator_actor.compute(
            {"states": predator_actor_input, "rnn": [predator_hidden]}
        )
        prey_mean, prey_log_std_parameter, prey_extra = prey_actor.compute(
            {"states": prey_actor_input, "rnn": [prey_hidden]}
        )
        predator_value, predator_value_extra = predator_critic.compute(
            {"states": predator_critic_input, "rnn": [predator_critic_hidden]}
        )
        prey_value, prey_value_extra = prey_critic.compute(
            {"states": prey_critic_input, "rnn": [prey_critic_hidden]}
        )

        predator_mean = predator_mean_flat.reshape(
            BATCH_SIZE, PREDATOR_COUNT, ACTION_DIM
        )
        predator_log_std = predator_log_std_flat.reshape(
            1, PREDATOR_COUNT, ACTION_DIM
        ).expand(BATCH_SIZE, -1, -1)
        prey_log_std = prey_log_std_parameter.reshape(1, ACTION_DIM).expand(
            BATCH_SIZE, -1
        )

        arrays.update(
            {
                "expected.predator_actor_input_normalized": _numpy(predator_actor_input),
                "expected.prey_actor_input_normalized": _numpy(prey_actor_input),
                "expected.predator_critic_input_normalized": _numpy(predator_critic_input),
                "expected.prey_critic_input_normalized": _numpy(prey_critic_input),
                "expected.predator_actor_latent_mean": _numpy(predator_mean),
                "expected.predator_actor_log_std": _numpy(predator_log_std),
                "expected.predator_actor_deterministic_action": _numpy(
                    torch.tanh(predator_mean)
                ),
                "expected.predator_actor_next_hidden": _numpy(
                    predator_extra["rnn"][0].reshape(
                        BATCH_SIZE, PREDATOR_COUNT, ACTOR_HIDDEN_DIM
                    )
                ),
                "expected.prey_actor_latent_mean": _numpy(prey_mean),
                "expected.prey_actor_log_std": _numpy(prey_log_std),
                "expected.prey_actor_deterministic_action": _numpy(torch.tanh(prey_mean)),
                "expected.prey_actor_next_hidden": _numpy(
                    prey_extra["rnn"][0].reshape(BATCH_SIZE, ACTOR_HIDDEN_DIM)
                ),
                "expected.predator_critic_value_normalized": _numpy(predator_value),
                "expected.predator_critic_value_inverse": _numpy(
                    _inverse_exact_skrl(predator_value, predator_value_scaler)
                ),
                "expected.predator_critic_next_hidden": _numpy(
                    predator_value_extra["rnn"][0].reshape(
                        BATCH_SIZE, CRITIC_HIDDEN_DIM
                    )
                ),
                "expected.prey_critic_value_normalized": _numpy(prey_value),
                "expected.prey_critic_value_inverse": _numpy(
                    _inverse_exact_skrl(prey_value, prey_value_scaler)
                ),
                "expected.prey_critic_next_hidden": _numpy(
                    prey_value_extra["rnn"][0].reshape(
                        BATCH_SIZE, CRITIC_HIDDEN_DIM
                    )
                ),
            }
        )

    _validate_arrays(arrays)
    return arrays


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(ARRAY_SHAPES):
        missing = sorted(set(ARRAY_SHAPES) - set(arrays))
        unexpected = sorted(set(arrays) - set(ARRAY_SHAPES))
        raise ValueError(f"oracle arrays mismatch: missing={missing}, unexpected={unexpected}")
    for name, expected_shape in ARRAY_SHAPES.items():
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise TypeError(f"oracle array {name!r} must be a NumPy array")
        if array.shape != expected_shape:
            raise ValueError(
                f"oracle array {name!r} shape mismatch: "
                f"expected {expected_shape}, got {array.shape}"
            )
        if array.dtype != np.float32:
            raise ValueError(f"oracle array {name!r} must have dtype float32")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"oracle array {name!r} contains a non-finite value")


def _metadata(
    *,
    arrays: Mapping[str, np.ndarray],
    repository: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_size: int,
    attention_models_path: Path,
) -> dict[str, Any]:
    records = [
        {
            "dtype": str(arrays[name].dtype),
            "kind": "input" if name.startswith("input.") else "expected",
            "name": name,
            "shape": list(arrays[name].shape),
        }
        for name in sorted(arrays)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
            "load_policy": "torch.load(weights_only=True,map_location=cpu) from verified bytes",
        },
        "producer": {
            "backend": "pytorch_skrl_checkpoint_forward",
            "uavpredatorprey": _git_metadata(repository),
            "attention_models_path": str(attention_models_path.resolve()),
            "attention_models_sha256": _sha256_file(attention_models_path),
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "skrl_version": str(skrl.__version__),
            "gymnasium_version": str(gymnasium.__version__),
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "contract": {
            "batch_cases": [
                "zero_raw_input_and_zero_hidden",
                "patterned_raw_input_and_nonzero_hidden_a",
                "patterned_raw_input_and_nonzero_hidden_b",
                "normalizer_extreme_input_and_nonzero_hidden",
            ],
            "architecture": "recurrent_entity_attention_v0_large_gru",
            "actor_output": "pre_tanh_latent_mean",
            "deterministic_action": "tanh(pre_tanh_latent_mean)",
            "predator_hidden_layout": "batch_predator_hidden",
            "prey_hidden_layout": "batch_hidden",
            "critic_hidden_layout": "batch_hidden",
            "normalizer_forward": "clamp((x-mean.float())/(sqrt(variance.float())+1e-8),-5,5)",
            "normalizer_inverse": "sqrt(variance.float())*clamp(x,-5,5)+mean.float()",
            "normalizer_buffers_in_checkpoint": "float64_cast_to_float32_before_forward",
            "gru_gate_order": ["reset", "update", "new"],
            "gru_step_count": 1,
            "terminated_input": "absent_no_reset",
            "all_fixture_arrays_dtype": "float32",
        },
        "array_count": len(records),
        "arrays": records,
    }


def _write_json_fsynced(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_fsynced(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **{name: arrays[name] for name in sorted(arrays)})
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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
            "atomic no-replace fixture publication requires renameat2",
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
            f"fixture destination already exists: {destination}",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _write_fixture(
    destination_argument: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    repository: Path,
) -> Path:
    _validate_arrays(arrays)
    destination = destination_argument.expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(f"fixture destination already exists: {destination}")
    destination = destination.resolve()
    repository_resolved = repository.resolve()
    if destination == repository_resolved or repository_resolved in destination.parents:
        raise ValueError(
            "policy oracle fixtures are runtime artifacts and must be written outside Git"
        )
    containing_worktree = _containing_git_worktree(destination)
    if containing_worktree is not None:
        raise ValueError(
            "policy oracle fixtures are runtime artifacts and must be written outside "
            f"Git; destination is inside {containing_worktree}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _write_npz_fsynced(temporary / ARRAYS_FILENAME, arrays)
        _write_json_fsynced(temporary / METADATA_FILENAME, metadata)
        checksum_lines = [
            f"{_sha256_file(temporary / filename)}  {filename}"
            for filename in (ARRAYS_FILENAME, METADATA_FILENAME)
        ]
        checksum_path = temporary / CHECKSUMS_FILENAME
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(checksum_lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if {path.name for path in temporary.iterdir()} != FIXTURE_FILENAMES:
            raise RuntimeError("fixture staging directory contains unexpected files")
        _fsync_directory(temporary)
        _publish_directory_no_replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    checkpoint_argument = args.checkpoint
    if not checkpoint_argument.is_absolute():
        checkpoint_argument = repository / checkpoint_argument
    checkpoint_state, checkpoint_sha256, checkpoint_size = _safe_load_checkpoint(
        checkpoint_argument,
        args.expected_checkpoint_sha256,
    )

    attention_models_path = (
        repository
        / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/"
        "uavpredatorprey_3v1/agents/attention_models.py"
    )
    attention_models = _load_attention_models_module(attention_models_path)
    torch.manual_seed(0)
    arrays = _evaluate_checkpoint(checkpoint_state, attention_models)
    metadata = _metadata(
        arrays=arrays,
        repository=repository,
        checkpoint_path=checkpoint_argument,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_size,
        attention_models_path=attention_models_path,
    )
    output = _write_fixture(args.output, arrays, metadata, repository=repository)
    print(f"[INFO] Wrote {len(arrays)} arrays to immutable fixture {output}")
    print(f"[INFO] Source checkpoint SHA-256: {checkpoint_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
