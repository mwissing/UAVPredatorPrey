"""Expand an obstacle-free 3v1 checkpoint to the pillar-obstacle observation layout.

The recurrent attention networks accept the old weights unchanged: their
obstacle_* parameters are created fresh with a zero output projection, so the
expanded policy and critic initially behave exactly like the original (see
tests/test_obstacles.py). Only the observation normalizers change size. Each
per-drone block (every predator block and the prey block) gains
num_obstacles * obstacle_dim features with mean 0 and variance 1, because the
environment already scales those features. Optimizer states are dropped since
the parameter set changes.

Example (inside the Isaac container):
  isaaclab.sh -p scripts/expand_checkpoint_obstacles.py \\
    --input .pretrained_checkpoints/anchor_v3_2026-10-08/current/agent_28800.pt \\
    --output /workspace/artifacts/.../anchor_v3_obstacles4.pt --num-obstacles 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def expand_blocks(values: torch.Tensor, block_sizes: list[int], extra: int, fill: float) -> torch.Tensor:
    """Insert `extra` entries of `fill` after each block of a flat per-feature tensor."""
    if sum(block_sizes) != values.shape[0]:
        raise ValueError(f"blocks {block_sizes} do not cover {values.shape[0]} features")
    parts, start = [], 0
    for size in block_sizes:
        parts.append(values[start : start + size])
        parts.append(torch.full((extra,), fill, dtype=values.dtype, device=values.device))
        start += size
    return torch.cat(parts)


def expand_scaler(state: dict[str, torch.Tensor], block_sizes: list[int], extra: int) -> dict[str, torch.Tensor]:
    return {
        "running_mean": expand_blocks(state["running_mean"], block_sizes, extra, 0.0),
        "running_variance": expand_blocks(state["running_variance"], block_sizes, extra, 1.0),
        "current_count": state["current_count"].clone(),
    }


def expand_checkpoint(
    checkpoint: dict[str, Any], *, num_predators: int, num_obstacles: int, obstacle_dim: int = 4
) -> dict[str, Any]:
    extra = num_obstacles * obstacle_dim
    predator_obs = checkpoint["predator"]["state_preprocessor"]["running_mean"].shape[0]
    prey_obs = checkpoint["prey"]["state_preprocessor"]["running_mean"].shape[0]
    if predator_obs % num_predators:
        raise ValueError(f"predator observation size {predator_obs} is not divisible by {num_predators} drones")
    per_predator = predator_obs // num_predators
    blocks = {
        "predator": [per_predator] * num_predators,
        "prey": [prey_obs],
        "shared": [per_predator] * num_predators + [prey_obs],
    }
    expanded: dict[str, Any] = {}
    for role, agent in checkpoint.items():
        agent = dict(agent)
        agent.pop("optimizer", None)
        agent["state_preprocessor"] = expand_scaler(agent["state_preprocessor"], blocks[role], extra)
        agent["shared_state_preprocessor"] = expand_scaler(agent["shared_state_preprocessor"], blocks["shared"], extra)
        expanded[role] = agent
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-obstacles", type=int, required=True)
    parser.add_argument("--obstacle-dim", type=int, default=4)
    parser.add_argument("--num-predators", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    expanded = expand_checkpoint(
        checkpoint, num_predators=args.num_predators, num_obstacles=args.num_obstacles, obstacle_dim=args.obstacle_dim
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded, args.output)
    for role, agent in expanded.items():
        sizes = {k: tuple(agent[k]["running_mean"].shape) for k in ("state_preprocessor", "shared_state_preprocessor")}
        print(f"{role}: {sizes}, optimizer dropped")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
