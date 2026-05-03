"""Transfer a 3v1-empty SKRL MAPPO checkpoint to 3v1-obstacles.

The obstacles environment adds observation inputs:
- predator policy: 72 -> 84 observations
- prey policy: 30 -> 34 observations
- centralized value state: 102 -> 118 inputs

This script copies all compatible learned weights and expands the first linear
layers with zero-initialized columns for the new obstacle features. With those
zeros, the transferred policy initially behaves like the empty-env policy and
can then learn obstacle awareness during fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


AGENT_DIMS = {
    "predator": {"policy_old": 72, "policy_new": 84, "value_old": 102, "value_new": 118},
    "prey": {"policy_old": 30, "policy_new": 34, "value_old": 102, "value_new": 118},
}


def _find_first_linear_weight(state_dict: dict[str, torch.Tensor], old_in_features: int) -> str:
    matches = [
        key
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == old_in_features
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one first-layer weight with {old_in_features} inputs, found {matches}"
        )
    return matches[0]


def _expand_input_layer(state_dict: dict[str, torch.Tensor], old_in_features: int, new_in_features: int) -> str:
    weight_key = _find_first_linear_weight(state_dict, old_in_features)
    old_weight = state_dict[weight_key]
    new_weight = old_weight.new_zeros((old_weight.shape[0], new_in_features))
    new_weight[:, :old_in_features] = old_weight
    state_dict[weight_key] = new_weight
    return weight_key


def _reset_training_state(agent_data: dict[str, Any], *, reset_preprocessors: bool) -> list[str]:
    removed = []
    for key in list(agent_data.keys()):
        if key == "optimizer" or (reset_preprocessors and "preprocessor" in key):
            del agent_data[key]
            removed.append(key)
    return removed


def transfer_checkpoint(
    source: Path,
    output: Path,
    *,
    reset_preprocessors: bool,
    template: Path | None = None,
    agents: set[str] | None = None,
) -> None:
    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint = torch.load(template, map_location="cpu", weights_only=False) if template else source_checkpoint
    agents = agents or set(AGENT_DIMS)

    for agent_name, dims in AGENT_DIMS.items():
        if agent_name not in source_checkpoint:
            raise RuntimeError(f"Missing agent '{agent_name}' in checkpoint")
        if agent_name not in checkpoint:
            raise RuntimeError(f"Missing agent '{agent_name}' in template checkpoint")
        if agent_name not in agents:
            agent_data = checkpoint[agent_name]
            if isinstance(agent_data, dict):
                removed = _reset_training_state(agent_data, reset_preprocessors=reset_preprocessors)
                if removed:
                    print(f"{agent_name}: kept from template, removed {', '.join(removed)}")
            continue

        agent_data = source_checkpoint[agent_name]
        if not isinstance(agent_data, dict):
            raise RuntimeError(f"Expected checkpoint['{agent_name}'] to be a dict")

        policy = agent_data.get("policy")
        value = agent_data.get("value")
        if not isinstance(policy, dict) or not isinstance(value, dict):
            raise RuntimeError(f"Expected '{agent_name}' policy/value entries to be state_dict-like dicts")

        policy_key = _expand_input_layer(policy, dims["policy_old"], dims["policy_new"])
        value_key = _expand_input_layer(value, dims["value_old"], dims["value_new"])
        removed = _reset_training_state(agent_data, reset_preprocessors=reset_preprocessors)

        print(f"{agent_name}: expanded policy {policy_key} {dims['policy_old']} -> {dims['policy_new']}")
        print(f"{agent_name}: expanded value  {value_key} {dims['value_old']} -> {dims['value_new']}")
        if removed:
            print(f"{agent_name}: removed {', '.join(removed)}")
        checkpoint[agent_name] = agent_data

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved transferred checkpoint: {output}")


def inspect_checkpoint(source: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    for agent_name, agent_data in checkpoint.items():
        if not isinstance(agent_data, dict):
            print(f"{agent_name}: {type(agent_data).__name__}")
            continue
        print(f"{agent_name}: {', '.join(agent_data.keys())}")
        for model_name in ("policy", "value"):
            model = agent_data.get(model_name)
            if isinstance(model, dict):
                shapes = {
                    key: tuple(value.shape)
                    for key, value in model.items()
                    if isinstance(value, torch.Tensor) and value.ndim >= 1
                }
                print(f"  {model_name}: {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="3v1-empty checkpoint")
    parser.add_argument("--output", type=Path, help="Output checkpoint path")
    parser.add_argument(
        "--template",
        type=Path,
        help="Obstacle checkpoint to use as a base for non-transferred agents",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=sorted(AGENT_DIMS),
        help="Agents to transfer from source (default: predator prey)",
    )
    parser.add_argument("--inspect", action="store_true", help="Only print checkpoint structure")
    parser.add_argument(
        "--keep-preprocessors",
        action="store_true",
        help="Keep RunningStandardScaler state instead of recalibrating for obstacle observations",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_checkpoint(args.source)
        return

    output = args.output or args.source.with_name(args.source.stem + "_to_obstacles.pt")
    transfer_checkpoint(
        args.source,
        output,
        reset_preprocessors=not args.keep_preprocessors,
        template=args.template,
        agents=set(args.agents) if args.agents else None,
    )


if __name__ == "__main__":
    main()
