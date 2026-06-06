"""Compose a SKRL MAPPO checkpoint from separate predator/prey checkpoints.

This is a lightweight offline league helper. It lets you train or evaluate
combinations such as a new prey against an older predator without modifying the
SKRL trainer or sampling different opponents inside one vectorized run.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch


AGENTS = ("predator", "prey")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected a dict checkpoint at {path}, got {type(checkpoint).__name__}")
    return checkpoint


def _reset_training_state(agent_data: dict[str, Any], *, keep_optimizers: bool, reset_preprocessors: bool) -> list[str]:
    removed = []
    for key in list(agent_data.keys()):
        if (key == "optimizer" and not keep_optimizers) or (reset_preprocessors and "preprocessor" in key):
            del agent_data[key]
            removed.append(key)
    return removed


def _tensor_shapes(agent_data: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(agent_data, dict):
        return {}
    shapes = {}
    for model_name in ("policy", "value"):
        model = agent_data.get(model_name)
        if not isinstance(model, dict):
            continue
        for key, value in model.items():
            if isinstance(value, torch.Tensor):
                shapes[f"{model_name}.{key}"] = tuple(value.shape)
    return shapes


def inspect_checkpoint(path: Path) -> None:
    checkpoint = _load_checkpoint(path)
    print(f"Checkpoint: {path}")
    for agent in checkpoint:
        agent_data = checkpoint[agent]
        if not isinstance(agent_data, dict):
            print(f"{agent}: {type(agent_data).__name__}")
            continue
        print(f"{agent}: {', '.join(agent_data.keys())}")
        for key, shape in _tensor_shapes(agent_data).items():
            print(f"  {key}: {shape}")


def _check_compatible(base_agent: dict[str, Any], source_agent: dict[str, Any], agent_name: str) -> None:
    base_shapes = _tensor_shapes(base_agent)
    source_shapes = _tensor_shapes(source_agent)
    mismatches = [
        (key, base_shapes[key], source_shapes[key])
        for key in sorted(base_shapes.keys() & source_shapes.keys())
        if base_shapes[key] != source_shapes[key]
    ]
    if mismatches:
        details = "\n".join(f"  {key}: base {base_shape} != source {source_shape}" for key, base_shape, source_shape in mismatches)
        raise RuntimeError(f"Shape mismatch while composing agent '{agent_name}':\n{details}")


def compose_checkpoint(
    output: Path,
    *,
    base: Path | None,
    predator: Path | None,
    prey: Path | None,
    keep_optimizers: bool,
    reset_preprocessors: bool,
) -> None:
    sources = {"predator": predator, "prey": prey}
    selected_sources = {agent: path for agent, path in sources.items() if path is not None}
    if not selected_sources:
        raise RuntimeError("Pass at least one source checkpoint via --predator or --prey.")

    base_path = base or next(iter(selected_sources.values()))
    checkpoint = _load_checkpoint(base_path)
    print(f"Base checkpoint: {base_path}")

    for agent_name, source_path in selected_sources.items():
        source_checkpoint = _load_checkpoint(source_path)
        if agent_name not in source_checkpoint:
            raise RuntimeError(f"Source checkpoint {source_path} does not contain agent '{agent_name}'")
        if agent_name in checkpoint:
            _check_compatible(checkpoint[agent_name], source_checkpoint[agent_name], agent_name)
        checkpoint[agent_name] = copy.deepcopy(source_checkpoint[agent_name])
        print(f"{agent_name}: copied from {source_path}")

    for agent_name in AGENTS:
        agent_data = checkpoint.get(agent_name)
        if not isinstance(agent_data, dict):
            continue
        removed = _reset_training_state(
            agent_data,
            keep_optimizers=keep_optimizers,
            reset_preprocessors=reset_preprocessors,
        )
        if removed:
            print(f"{agent_name}: removed {', '.join(removed)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved composed checkpoint: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, help="Template checkpoint. Defaults to the first selected source.")
    parser.add_argument("--predator", type=Path, help="Checkpoint providing the predator agent block.")
    parser.add_argument("--prey", type=Path, help="Checkpoint providing the prey agent block.")
    parser.add_argument("--output", type=Path, help="Output checkpoint path.")
    parser.add_argument("--inspect", type=Path, help="Only print checkpoint structure for the given path.")
    parser.add_argument(
        "--keep-optimizers",
        action="store_true",
        help="Keep optimizer states. By default they are removed so training resumes with fresh optimizers.",
    )
    parser.add_argument(
        "--reset-preprocessors",
        action="store_true",
        help="Remove RunningStandardScaler states so they recalibrate on the next run.",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_checkpoint(args.inspect)
        return

    if args.output is None:
        raise SystemExit("--output is required unless --inspect is used.")

    compose_checkpoint(
        args.output,
        base=args.base,
        predator=args.predator,
        prey=args.prey,
        keep_optimizers=args.keep_optimizers,
        reset_preprocessors=args.reset_preprocessors,
    )


if __name__ == "__main__":
    main()

