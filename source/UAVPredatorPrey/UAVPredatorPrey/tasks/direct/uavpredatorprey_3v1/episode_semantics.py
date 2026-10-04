"""Pure tensor helpers for 3v1 rewards and episode termination."""

from __future__ import annotations

import torch


def shifted_squared_speed_penalty(
    speed_squared: torch.Tensor,
    *,
    free_speed: float,
    penalty_scale: float,
    dt: float,
) -> torch.Tensor:
    """Penalize squared speed only above a free-speed margin in m/s."""
    if free_speed < 0.0:
        raise ValueError("free_speed must be non-negative")
    penalized_speed_squared = torch.clamp(speed_squared - float(free_speed) ** 2, min=0.0)
    return penalized_speed_squared * float(penalty_scale) * float(dt)


def predator_crash_event_penalty(
    newly_oob: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Return one per-predator penalty for newly inactive drones only."""
    return newly_oob.transpose(0, 1).float() * float(penalty)


def all_predators_inactive(predator_alive: torch.Tensor) -> torch.Tensor:
    """Return the environment mask for episodes with no active predator."""
    return ~predator_alive.any(dim=1)
