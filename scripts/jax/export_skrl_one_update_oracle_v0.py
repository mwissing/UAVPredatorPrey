#!/usr/bin/env python3
"""Export one deterministic predator PPO/Adam update as a JAX oracle.

The fixture starts from the exact SHA-256-bound skrl checkpoint used by the
policy-transfer artifact. It replays one time-major recurrent predator
minibatch and records scalar losses, raw/clipped gradients, Adam moments, and
updated parameters in their original Torch layouts.

Entropy is intentionally disabled in this first oracle. This isolates the
deterministic PPO/value/optimizer boundary from framework-specific random
number generators; stochastic entropy-gradient parity belongs to a later,
separate fixture.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = "uavpredatorprey.skrl_one_update_oracle.v0"
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
    "/workspace/artifacts/transfer/skrl_one_update_oracle_v0/"
    "windows_agent_115200_predator"
)

ROLLOUT_LENGTH = 4
NUM_ENVS = 4
MINIBATCH_ENV_INDICES = (0, 2)
PREDATOR_COUNT = 3
PREDATOR_OBSERVATION_DIM = 90
CENTRALIZED_STATE_DIM = 120
ACTION_DIM = 4
ACTOR_HIDDEN_DIM = 256
CRITIC_HIDDEN_DIM = 64

RATIO_CLIP = 0.2
VALUE_CLIP = 0.2
VALUE_LOSS_SCALE = 2.0
ENTROPY_LOSS_SCALE = 0.0
LEARNING_RATE = 1.0e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
MAX_GRADIENT_NORM = 1.0
TORCH_GRADIENT_CLIP_EPSILON = 1.0e-6


def _load_sibling_module(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load sibling utility: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


_FORWARD = _load_sibling_module(
    "_uavpredatorprey_forward_oracle_helpers_v0",
    "export_skrl_policy_oracle_v0.py",
)
_TRANSFER = _load_sibling_module(
    "_uavpredatorprey_transfer_specs_v0",
    "export_skrl_policy_transfer_v0.py",
)


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
        help="new immutable fixture directory outside this Git repository",
    )
    return parser.parse_args(argv)


def _pattern(rows: int, columns: int, *, phase: float, scale: float) -> np.ndarray:
    return _FORWARD._pattern(rows, columns, phase=phase, scale=scale)


def _time_major_pattern(
    feature_dim: int, *, phase: float, scale: float
) -> np.ndarray:
    return _pattern(
        ROLLOUT_LENGTH * NUM_ENVS,
        feature_dim,
        phase=phase,
        scale=scale,
    ).reshape(ROLLOUT_LENGTH, NUM_ENVS, feature_dim)


def _input_arrays() -> dict[str, np.ndarray]:
    terminated = np.zeros((ROLLOUT_LENGTH, NUM_ENVS, 1), dtype=np.bool_)
    truncated = np.zeros_like(terminated)
    terminated[1, 0, 0] = True
    terminated[0, 1, 0] = True
    truncated[2, 2, 0] = True

    return {
        "input.predator_observation_raw": _time_major_pattern(
            PREDATOR_OBSERVATION_DIM, phase=0.31, scale=2.25
        ),
        "input.centralized_state_raw": _time_major_pattern(
            CENTRALIZED_STATE_DIM, phase=-0.77, scale=2.75
        ),
        "input.predator_actor_initial_hidden": _pattern(
            NUM_ENVS,
            PREDATOR_COUNT * ACTOR_HIDDEN_DIM,
            phase=0.19,
            scale=0.12,
        ).reshape(NUM_ENVS, PREDATOR_COUNT, ACTOR_HIDDEN_DIM),
        "input.predator_critic_initial_hidden": _pattern(
            NUM_ENVS, CRITIC_HIDDEN_DIM, phase=0.83, scale=0.09
        ),
        "input.terminated": terminated,
        "input.truncated": truncated,
        "input.environment_indices": np.asarray(
            MINIBATCH_ENV_INDICES, dtype=np.int32
        ),
    }


def _select_environments(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected = torch.index_select(tensor, 1, indices)
    return selected.reshape(-1, *selected.shape[2:])


def _squashed_log_prob(
    latent_mean: torch.Tensor,
    log_std: torch.Tensor,
    pre_tanh_action: torch.Tensor,
) -> torch.Tensor:
    distribution = torch.distributions.Normal(latent_mean, log_std.exp())
    log_jacobian = 2.0 * (
        math.log(2.0)
        - pre_tanh_action
        - F.softplus(-2.0 * pre_tanh_action)
    )
    return (distribution.log_prob(pre_tanh_action) - log_jacobian).sum(
        dim=-1, keepdim=True
    )


def _model_specs_for_predator() -> tuple[Any, ...]:
    result = tuple(
        spec
        for spec in _TRANSFER.LEAF_SPECS
        if spec.stored_dtype == "float32"
        and (
            spec.source_path.startswith("predator.policy.")
            or spec.source_path.startswith("predator.value.")
        )
    )
    if not result:
        raise RuntimeError("predator model transfer specifications are empty")
    return result


def _named_parameters_by_source(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    result = {
        **{
            f"predator.policy.{name}": parameter
            for name, parameter in actor.named_parameters()
        },
        **{
            f"predator.value.{name}": parameter
            for name, parameter in critic.named_parameters()
        },
    }
    expected = {spec.source_path for spec in _model_specs_for_predator()}
    if set(result) != expected:
        raise RuntimeError(
            "trainable predator parameter paths do not match transfer v0: "
            f"missing={sorted(expected - set(result))}, "
            f"unexpected={sorted(set(result) - expected)}"
        )
    return result


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def _scalar(value: torch.Tensor | float) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    return np.asarray(value, dtype=np.float32)


def _parameter_array_name(source_path: str, quantity: str) -> str:
    return f"expected.parameter.{source_path}.{quantity}"


def _evaluate_checkpoint(
    checkpoint_state: Mapping[str, Any], attention_models: Any
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    actor, _, critic, _ = _FORWARD._models(attention_models, checkpoint_state)
    arrays = _input_arrays()
    tensors = {
        name: torch.from_numpy(value)
        for name, value in arrays.items()
        if name != "input.environment_indices"
    }
    environment_indices = torch.from_numpy(arrays["input.environment_indices"]).long()

    actor_scaler = _FORWARD._scaler_state(
        checkpoint_state,
        role="predator",
        name="state_preprocessor",
        feature_dim=PREDATOR_OBSERVATION_DIM,
    )
    critic_scaler = _FORWARD._scaler_state(
        checkpoint_state,
        role="predator",
        name="shared_state_preprocessor",
        feature_dim=CENTRALIZED_STATE_DIM,
    )
    value_scaler = _FORWARD._scaler_state(
        checkpoint_state,
        role="predator",
        name="value_preprocessor",
        feature_dim=1,
    )

    observation_raw = tensors["input.predator_observation_raw"]
    centralized_state_raw = tensors["input.centralized_state_raw"]
    observation = _FORWARD._normalize_exact_skrl(observation_raw, actor_scaler)
    centralized_state = _FORWARD._normalize_exact_skrl(
        centralized_state_raw, critic_scaler
    )
    done = tensors["input.terminated"] | tensors["input.truncated"]
    actor_hidden = tensors["input.predator_actor_initial_hidden"].reshape(
        1, NUM_ENVS, PREDATOR_COUNT * ACTOR_HIDDEN_DIM
    )
    critic_hidden = tensors["input.predator_critic_initial_hidden"].reshape(
        1, NUM_ENVS, CRITIC_HIDDEN_DIM
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    with torch.no_grad():
        full_mean, full_log_std, _ = actor.compute(
            {
                "states": observation.reshape(-1, PREDATOR_OBSERVATION_DIM),
                "rnn": [actor_hidden],
                "terminated": done.reshape(-1, 1),
            }
        )
        full_value_normalized, _ = critic.compute(
            {
                "states": centralized_state.reshape(-1, CENTRALIZED_STATE_DIM),
                "rnn": [critic_hidden],
                "terminated": done.reshape(-1, 1),
            }
        )
        action_offset = torch.from_numpy(
            _time_major_pattern(
                PREDATOR_COUNT * ACTION_DIM,
                phase=1.37,
                scale=0.18,
            )
        ).reshape(-1, PREDATOR_COUNT * ACTION_DIM)
        pre_tanh_action = full_mean + action_offset
        old_log_prob = _squashed_log_prob(
            full_mean, full_log_std, pre_tanh_action
        )
        old_value = _FORWARD._inverse_exact_skrl(
            full_value_normalized, value_scaler
        )

        return_offset = torch.from_numpy(
            _time_major_pattern(1, phase=-1.11, scale=0.42)
        ).reshape(-1, 1)
        return_normalized = torch.clamp(
            full_value_normalized + return_offset,
            min=-4.0,
            max=4.0,
        )
        returns = _FORWARD._inverse_exact_skrl(return_normalized, value_scaler)

        raw_advantage = torch.from_numpy(
            _time_major_pattern(1, phase=0.57, scale=1.0)
        ).reshape(-1, 1)
        normalized_advantage = (
            raw_advantage - raw_advantage.mean()
        ) / (raw_advantage.std(correction=1) + 1.0e-8)

    arrays.update(
        {
            "input.predator_actor_input_normalized": _as_numpy(
                observation
            ),
            "input.predator_critic_input_normalized": _as_numpy(
                centralized_state
            ),
            "input.pre_tanh_action": _as_numpy(
                pre_tanh_action.reshape(
                    ROLLOUT_LENGTH, NUM_ENVS, PREDATOR_COUNT, ACTION_DIM
                )
            ),
            "input.sampled_action_norm": _as_numpy(
                torch.tanh(pre_tanh_action).reshape(
                    ROLLOUT_LENGTH, NUM_ENVS, PREDATOR_COUNT, ACTION_DIM
                )
            ),
            "input.old_log_prob": _as_numpy(
                old_log_prob.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1)
            ),
            "input.old_value": _as_numpy(
                old_value.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1)
            ),
            "input.returns": _as_numpy(
                returns.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1)
            ),
            "input.normalized_advantage": _as_numpy(
                normalized_advantage.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1)
            ),
        }
    )

    selected_observation = _select_environments(observation, environment_indices)
    selected_state = _select_environments(centralized_state, environment_indices)
    selected_done = _select_environments(done, environment_indices)
    selected_pre_tanh = _select_environments(
        pre_tanh_action.reshape(
            ROLLOUT_LENGTH, NUM_ENVS, PREDATOR_COUNT * ACTION_DIM
        ),
        environment_indices,
    )
    selected_old_log_prob = _select_environments(
        old_log_prob.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1),
        environment_indices,
    )
    selected_old_value_normalized = _select_environments(
        full_value_normalized.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1),
        environment_indices,
    )
    selected_return_normalized = _select_environments(
        return_normalized.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1),
        environment_indices,
    )
    selected_advantage = _select_environments(
        normalized_advantage.reshape(ROLLOUT_LENGTH, NUM_ENVS, 1),
        environment_indices,
    )
    selected_actor_hidden = torch.index_select(
        tensors["input.predator_actor_initial_hidden"],
        0,
        environment_indices,
    ).reshape(
        1,
        len(MINIBATCH_ENV_INDICES),
        PREDATOR_COUNT * ACTOR_HIDDEN_DIM,
    )
    selected_critic_hidden = torch.index_select(
        tensors["input.predator_critic_initial_hidden"],
        0,
        environment_indices,
    ).reshape(1, len(MINIBATCH_ENV_INDICES), CRITIC_HIDDEN_DIM)

    current_mean, current_log_std, _ = actor.compute(
        {
            "states": selected_observation,
            "rnn": [selected_actor_hidden],
            "terminated": selected_done,
        }
    )
    current_log_prob = _squashed_log_prob(
        current_mean, current_log_std, selected_pre_tanh
    )
    with torch.no_grad():
        behavior_mean, behavior_log_std, _ = actor.compute(
            {
                "states": selected_observation,
                "rnn": [selected_actor_hidden],
                "terminated": selected_done,
            }
        )
        behavior_log_prob = _squashed_log_prob(
            behavior_mean, behavior_log_std, selected_pre_tanh
        )
    anchored_log_prob = selected_old_log_prob.detach() + (
        current_log_prob - behavior_log_prob.detach()
    )
    log_ratio = anchored_log_prob - selected_old_log_prob.detach()
    ratio = torch.exp(log_ratio)
    surrogate = selected_advantage.detach() * ratio
    surrogate_clipped = selected_advantage.detach() * torch.clamp(
        ratio, 1.0 - RATIO_CLIP, 1.0 + RATIO_CLIP
    )
    policy_loss = -torch.minimum(surrogate, surrogate_clipped).mean()

    predicted_value, _ = critic.compute(
        {
            "states": selected_state,
            "rnn": [selected_critic_hidden],
            "terminated": selected_done,
        }
    )
    value_used = selected_old_value_normalized.detach() + torch.clamp(
        predicted_value - selected_old_value_normalized.detach(),
        min=-VALUE_CLIP,
        max=VALUE_CLIP,
    )
    value_loss = VALUE_LOSS_SCALE * F.mse_loss(
        value_used, selected_return_normalized.detach()
    )
    entropy_loss = policy_loss.new_zeros(())
    total_loss = policy_loss + value_loss + entropy_loss

    with torch.no_grad():
        diagnostic_ratio = torch.exp(log_ratio)
        approximate_kl = ((diagnostic_ratio - 1.0) - log_ratio).mean()
        ratio_deviation = torch.abs(diagnostic_ratio - 1.0)
        ratio_clip_fraction = (
            (diagnostic_ratio < 1.0 - RATIO_CLIP)
            | (diagnostic_ratio > 1.0 + RATIO_CLIP)
        ).float().mean()

    parameters_by_source = _named_parameters_by_source(actor, critic)
    parameter_specs = _model_specs_for_predator()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    raw_gradients: dict[str, torch.Tensor] = {}
    for source_path, parameter in parameters_by_source.items():
        if parameter.grad is None:
            raise RuntimeError(f"parameter has no gradient: {source_path}")
        raw_gradients[source_path] = parameter.grad.detach().clone()

    all_parameters = list(actor.parameters()) + list(critic.parameters())
    pre_clip_gradient_norm = torch.nn.utils.clip_grad_norm_(
        all_parameters, MAX_GRADIENT_NORM
    )
    gradient_scale = min(
        1.0,
        MAX_GRADIENT_NORM
        / (float(pre_clip_gradient_norm) + TORCH_GRADIENT_CLIP_EPSILON),
    )
    clipped_gradients = {
        source_path: parameter.grad.detach().clone()
        for source_path, parameter in parameters_by_source.items()
    }
    optimizer.step()
    actor.project_log_std_parameter_()

    arrays.update(
        {
            "expected.new_log_prob": _as_numpy(
                anchored_log_prob.reshape(
                    ROLLOUT_LENGTH, len(MINIBATCH_ENV_INDICES), 1
                )
            ),
            "expected.new_value_normalized": _as_numpy(
                predicted_value.reshape(
                    ROLLOUT_LENGTH, len(MINIBATCH_ENV_INDICES), 1
                )
            ),
            "expected.loss.total": _scalar(total_loss),
            "expected.loss.policy": _scalar(policy_loss),
            "expected.loss.entropy": _scalar(entropy_loss),
            "expected.loss.value": _scalar(value_loss),
            "expected.approximate_kl": _scalar(approximate_kl),
            "expected.mean_ratio_deviation": _scalar(
                ratio_deviation.mean()
            ),
            "expected.max_ratio_deviation": _scalar(
                ratio_deviation.max()
            ),
            "expected.ratio_clip_fraction": _scalar(
                ratio_clip_fraction
            ),
            "expected.optimizer.pre_clip_gradient_norm": _scalar(
                pre_clip_gradient_norm
            ),
            "expected.optimizer.gradient_scale": _scalar(gradient_scale),
            "expected.optimizer.step": np.asarray(1, dtype=np.int32),
        }
    )

    parameter_records: list[dict[str, Any]] = []
    for spec in parameter_specs:
        parameter = parameters_by_source[spec.source_path]
        optimizer_state = optimizer.state[parameter]
        quantity_tensors = {
            "gradient_raw": raw_gradients[spec.source_path],
            "gradient_clipped": clipped_gradients[spec.source_path],
            "first_moment": optimizer_state["exp_avg"],
            "second_moment": optimizer_state["exp_avg_sq"],
            "updated": parameter,
        }
        quantity_arrays: dict[str, str] = {}
        for quantity, tensor in quantity_tensors.items():
            array_name = _parameter_array_name(spec.source_path, quantity)
            arrays[array_name] = _as_numpy(tensor)
            quantity_arrays[quantity] = array_name
        parameter_records.append(
            {
                "semantic_path": spec.semantic_path,
                "source_path": spec.source_path,
                "source_layout": spec.source_layout,
                "stored_shape": list(spec.stored_shape),
                "stored_dtype": spec.stored_dtype,
                "arrays": quantity_arrays,
            }
        )

    _validate_arrays(arrays)
    return arrays, parameter_records


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "input.predator_observation_raw",
        "input.centralized_state_raw",
        "input.predator_actor_initial_hidden",
        "input.predator_critic_initial_hidden",
        "input.terminated",
        "input.truncated",
        "input.environment_indices",
        "input.pre_tanh_action",
        "input.sampled_action_norm",
        "input.old_log_prob",
        "input.old_value",
        "input.returns",
        "input.normalized_advantage",
        "expected.loss.total",
        "expected.optimizer.pre_clip_gradient_norm",
        "expected.optimizer.step",
    }
    if not required.issubset(arrays):
        raise ValueError(f"oracle arrays missing: {sorted(required - set(arrays))}")
    for name, array in arrays.items():
        if not isinstance(name, str) or not isinstance(array, np.ndarray):
            raise TypeError("oracle arrays must map string names to NumPy arrays")
        if array.dtype not in (np.dtype("float32"), np.dtype("bool"), np.dtype("int32")):
            raise TypeError(f"unsupported oracle dtype for {name}: {array.dtype}")
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"oracle array {name} contains non-finite values")


def _metadata(
    *,
    arrays: Mapping[str, np.ndarray],
    parameter_records: list[dict[str, Any]],
    repository: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_size: int,
    attention_models_path: Path,
) -> dict[str, Any]:
    array_records = [
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
            "load_policy": (
                "torch.load(weights_only=True,map_location=cpu) from verified bytes"
            ),
        },
        "producer": {
            "backend": "pytorch_skrl_predator_one_update",
            "uavpredatorprey": _FORWARD._git_metadata(repository),
            "attention_models_path": str(attention_models_path.resolve()),
            "attention_models_sha256": _FORWARD._sha256_file(
                attention_models_path
            ),
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "contract": {
            "role": "predator",
            "rollout_length": ROLLOUT_LENGTH,
            "num_envs": NUM_ENVS,
            "minibatch_environment_indices": list(MINIBATCH_ENV_INDICES),
            "time_layout": "time_major",
            "done_semantics": "reset recurrent state before t when done[t-1]",
            "policy_density": "joint_tanh_gaussian_sum_over_12_actions",
            "replay_anchor": (
                "old_log_prob + current_replay_log_prob - "
                "stop_gradient(frozen_behavior_replay_log_prob)"
            ),
            "advantage_normalization": "global_TxN_torch_std_correction_1",
            "ratio_clip": RATIO_CLIP,
            "value_clip": VALUE_CLIP,
            "value_loss_scale": VALUE_LOSS_SCALE,
            "entropy_loss_scale": ENTROPY_LOSS_SCALE,
            "entropy_ablation_reason": (
                "deterministic framework-independent first oracle"
            ),
            "optimizer": "joint_actor_critic_torch_adam",
            "learning_rate": LEARNING_RATE,
            "adam_beta1": ADAM_BETAS[0],
            "adam_beta2": ADAM_BETAS[1],
            "adam_epsilon": ADAM_EPSILON,
            "max_gradient_norm": MAX_GRADIENT_NORM,
            "optimizer_initial_state": "zero_moments_step_0",
            "post_step_projection": "actor_log_std_clamp_-3_0",
        },
        "parameter_count": len(parameter_records),
        "parameters": parameter_records,
        "array_count": len(array_records),
        "arrays": array_records,
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
        raise ValueError("one-update fixtures must be written outside Git")
    containing_worktree = _FORWARD._containing_git_worktree(destination)
    if containing_worktree is not None:
        raise ValueError(
            "one-update fixtures must be written outside Git; destination is "
            f"inside {containing_worktree}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _write_npz_fsynced(temporary / ARRAYS_FILENAME, arrays)
        _write_json_fsynced(temporary / METADATA_FILENAME, metadata)
        checksums = [
            f"{_FORWARD._sha256_file(temporary / filename)}  {filename}"
            for filename in (ARRAYS_FILENAME, METADATA_FILENAME)
        ]
        checksum_path = temporary / CHECKSUMS_FILENAME
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(checksums) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if {path.name for path in temporary.iterdir()} != FIXTURE_FILENAMES:
            raise RuntimeError("fixture staging directory contains unexpected files")
        _FORWARD._fsync_directory(temporary)
        _FORWARD._publish_directory_no_replace(temporary, destination)
        _FORWARD._fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = repository / checkpoint_path
    checkpoint_state, checkpoint_sha256, checkpoint_size = (
        _FORWARD._safe_load_checkpoint(
            checkpoint_path, args.expected_checkpoint_sha256
        )
    )
    attention_models_path = (
        repository
        / "source/UAVPredatorPrey/UAVPredatorPrey/tasks/direct/"
        "uavpredatorprey_3v1/agents/attention_models.py"
    )
    attention_models = _FORWARD._load_attention_models_module(
        attention_models_path
    )
    arrays, parameter_records = _evaluate_checkpoint(
        checkpoint_state, attention_models
    )
    metadata = _metadata(
        arrays=arrays,
        parameter_records=parameter_records,
        repository=repository,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_size,
        attention_models_path=attention_models_path,
    )
    output = _write_fixture(args.output, arrays, metadata, repository=repository)
    print(
        f"[INFO] Wrote {len(arrays)} arrays for "
        f"{len(parameter_records)} predator parameters to {output}"
    )
    print(f"[INFO] Source checkpoint SHA-256: {checkpoint_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
