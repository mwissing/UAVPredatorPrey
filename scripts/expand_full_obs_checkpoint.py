"""Warm-start the full-observation MAPPO checkpoint from a nearest-obstacle checkpoint.

The full-observation tasks deliberately change observation/state dimensions and use a
larger MLP. This helper copies the old learned subnetwork into a full-observation
template checkpoint while zeroing the new input columns for copied units. The resulting
policy starts close to the old behavior and can then learn from the richer obstacle
features.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch


OLD_OBS = {"predator": 84, "prey": 34}
NEW_OBS = {"predator": 156, "prey": 58}
OLD_STATE = 118
NEW_STATE = 214


def _load(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected checkpoint dict at {path}, got {type(checkpoint).__name__}")
    return checkpoint


def _obs_column_map(agent: str) -> torch.Tensor:
    if agent == "predator":
        mapping = list(range(72))
        for predator_idx in range(3):
            old_start = 72 + predator_idx * 4
            new_start = 72 + predator_idx * 28
            mapping.extend(range(new_start, new_start + 4))
        if len(mapping) != OLD_OBS["predator"]:
            raise RuntimeError(f"Internal predator observation map has wrong length: {len(mapping)}")
        return torch.tensor(mapping, dtype=torch.long)
    if agent == "prey":
        mapping = list(range(30))
        mapping.extend(range(30, 34))
        if len(mapping) != OLD_OBS["prey"]:
            raise RuntimeError(f"Internal prey observation map has wrong length: {len(mapping)}")
        return torch.tensor(mapping, dtype=torch.long)
    raise RuntimeError(f"Unsupported agent: {agent}")


def _state_column_map() -> torch.Tensor:
    mapping = []
    # Predator observation block in MAPPO state.
    mapping.extend(range(72))
    for predator_idx in range(3):
        new_start = 72 + predator_idx * 28
        mapping.extend(range(new_start, new_start + 4))

    # Prey observation block follows the full predator observation block.
    new_prey_offset = NEW_OBS["predator"]
    mapping.extend(range(new_prey_offset, new_prey_offset + 30))
    mapping.extend(range(new_prey_offset + 30, new_prey_offset + 34))

    if len(mapping) != OLD_STATE:
        raise RuntimeError(f"Internal state map has wrong length: {len(mapping)}")
    return torch.tensor(mapping, dtype=torch.long)


def _zero_and_copy_1d(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    result = target.clone()
    limit = min(result.shape[0], source.shape[0])
    result[:limit] = source[:limit]
    return result


def _zero_and_copy_2d(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    result = target.clone()
    rows = min(result.shape[0], source.shape[0])
    cols = min(result.shape[1], source.shape[1])
    result[:rows, :] = 0.0
    result[:rows, :cols] = source[:rows, :cols]
    return result


def _copy_first_layer(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    source_cols: int,
    target_cols: int,
    column_map: torch.Tensor,
) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2:
        return target
    if source.shape[1] != source_cols or target.shape[1] != target_cols:
        return _zero_and_copy_2d(target, source)

    result = target.clone()
    rows = min(result.shape[0], source.shape[0])
    result[:rows, :] = 0.0
    result[:rows, column_map] = source[:rows, : source.shape[1]]
    return result


def _copy_tensor(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    agent: str,
    model_name: str,
) -> torch.Tensor:
    if target.shape == source.shape:
        return source.clone()

    if target.ndim == 1 and source.ndim == 1:
        return _zero_and_copy_1d(target, source)

    if target.ndim == 2 and source.ndim == 2:
        if model_name == "policy":
            return _copy_first_layer(
                target,
                source,
                source_cols=OLD_OBS[agent],
                target_cols=NEW_OBS[agent],
                column_map=_obs_column_map(agent),
            )
        if model_name == "value":
            return _copy_first_layer(
                target,
                source,
                source_cols=OLD_STATE,
                target_cols=NEW_STATE,
                column_map=_state_column_map(),
            )
        return _zero_and_copy_2d(target, source)

    return target


def _neutral_expanded_tensor(target: torch.Tensor, key: str) -> torch.Tensor:
    lowered = key.lower()
    if "var" in lowered or "std" in lowered or "scale" in lowered:
        return torch.ones_like(target)
    return torch.zeros_like(target)


def _copy_expanded_last_dim(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    column_map: torch.Tensor,
    key: str,
) -> torch.Tensor:
    result = _neutral_expanded_tensor(target, key)
    if target.ndim == 1:
        result[column_map] = source[: column_map.numel()]
        return result
    if target.ndim == 2:
        result[:, column_map] = source[:, : column_map.numel()]
        return result
    return target


def _copy_preprocessor_tensor(target: torch.Tensor, source: torch.Tensor, *, agent: str, key: str) -> torch.Tensor:
    if target.shape == source.shape:
        return source.clone()

    if source.shape[-1] == OLD_OBS[agent] and target.shape[-1] == NEW_OBS[agent]:
        return _copy_expanded_last_dim(
            target,
            source,
            column_map=_obs_column_map(agent),
            key=key,
        )

    if source.shape[-1] == OLD_STATE and target.shape[-1] == NEW_STATE:
        return _copy_expanded_last_dim(
            target,
            source,
            column_map=_state_column_map(),
            key=key,
        )

    return target


def _copy_preprocessor(target: Any, source: Any, *, agent: str, prefix: str = "") -> tuple[Any, int]:
    if isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor):
        return _copy_preprocessor_tensor(target, source, agent=agent, key=prefix), 1

    if isinstance(target, dict) and isinstance(source, dict):
        copied = 0
        result = copy.deepcopy(target)
        for key, target_value in target.items():
            if key not in source:
                continue
            result[key], added = _copy_preprocessor(
                target_value,
                source[key],
                agent=agent,
                prefix=f"{prefix}.{key}" if prefix else str(key),
            )
            copied += added
        return result, copied

    return target, 0


def _copy_model(target_model: dict[str, Any], source_model: dict[str, Any], *, agent: str, model_name: str) -> int:
    copied = 0
    for key, target_value in list(target_model.items()):
        source_value = source_model.get(key)
        if not isinstance(target_value, torch.Tensor) or not isinstance(source_value, torch.Tensor):
            continue
        target_model[key] = _copy_tensor(target_value, source_value, agent=agent, model_name=model_name)
        copied += 1
    return copied


def expand_checkpoint(source: Path, template: Path, output: Path) -> None:
    if source.resolve() == template.resolve():
        raise RuntimeError(
            "--source must be an old nearest-obstacle checkpoint; --template must be a full-observation checkpoint."
        )
    source_checkpoint = _load(source)
    output_checkpoint = copy.deepcopy(_load(template))

    for agent in ("predator", "prey"):
        if agent not in source_checkpoint or agent not in output_checkpoint:
            raise RuntimeError(f"Both checkpoints must contain agent '{agent}'")
        source_agent = source_checkpoint[agent]
        target_agent = output_checkpoint[agent]
        if not isinstance(source_agent, dict) or not isinstance(target_agent, dict):
            raise RuntimeError(f"Agent '{agent}' block must be a dict in both checkpoints")

        for model_name in ("policy", "value"):
            source_model = source_agent.get(model_name)
            target_model = target_agent.get(model_name)
            if not isinstance(source_model, dict) or not isinstance(target_model, dict):
                raise RuntimeError(f"Agent '{agent}' model '{model_name}' must be a dict in both checkpoints")
            copied = _copy_model(target_model, source_model, agent=agent, model_name=model_name)
            print(f"{agent}.{model_name}: copied {copied} tensors")

        for key in list(target_agent.keys()):
            if key == "optimizer":
                del target_agent[key]
                print(f"{agent}: removed {key}")
            elif "preprocessor" in key and key in source_agent:
                target_agent[key], copied = _copy_preprocessor(
                    target_agent[key],
                    source_agent[key],
                    agent=agent,
                    prefix=key,
                )
                print(f"{agent}.{key}: copied/expanded {copied} tensors")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output)
    print(f"Saved expanded full-observation checkpoint: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Nearest-obstacle checkpoint to copy from.")
    parser.add_argument("--template", type=Path, required=True, help="Full-observation checkpoint providing shapes.")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path.")
    args = parser.parse_args()
    expand_checkpoint(args.source, args.template, args.output)


if __name__ == "__main__":
    main()
