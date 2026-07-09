"""Pure tensor helpers for 3v1 crash rewards and episode termination."""

from __future__ import annotations

import torch


def predator_crash_event_penalty(
    newly_oob: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Return one per-predator penalty for newly inactive drones only."""
    return newly_oob.transpose(0, 1).float() * float(penalty)


def all_predators_inactive(predator_alive: torch.Tensor) -> torch.Tensor:
    """Return the environment mask for episodes with no active predator."""
    return ~predator_alive.any(dim=1)
