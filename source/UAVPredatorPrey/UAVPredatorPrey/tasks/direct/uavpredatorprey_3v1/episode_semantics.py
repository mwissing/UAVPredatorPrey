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


def upright_reward(gravity_z: torch.Tensor, scale: float, dt: float, zero_at_level: bool = False) -> torch.Tensor:
    """Per-step upright term from the body-frame gravity z component (-1 when level, so -gravity_z = cos(tilt)).

    The default pays scale x dt per step for a level drone, so it doubles as an alive bonus that a catch ends: in calm
    standoffs it is what makes waiting slightly positive for the predators (Stage 8 review). With zero_at_level the
    term is scale x dt x (cos(tilt) - 1): the same tilt gradient, but a level drone earns nothing for staying airborne.
    """
    upright = -gravity_z
    if zero_at_level:
        upright = upright - 1.0
    return upright * scale * dt


def predator_crash_event_penalty(
    newly_oob: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Return one per-predator penalty for newly inactive drones only."""
    return newly_oob.transpose(0, 1).float() * float(penalty)


def all_predators_inactive(predator_alive: torch.Tensor) -> torch.Tensor:
    """Return the environment mask for episodes with no active predator."""
    return ~predator_alive.any(dim=1)


def forced_crash_bonus(
    prey_oob: torch.Tensor,
    caught: torch.Tensor,
    active_distances: torch.Tensor,
    *,
    bonus: float,
    assist_distance: float,
) -> torch.Tensor:
    """(P, N) per-drone reward for a forced prey crash, attributed like a catch.

    A forced crash is a prey exit (pillar crash or arena boundary) without a catch while the nearest active
    predator is within assist_distance, the same definition as the forced_prey_oob metric. That predator gets
    bonus * P, so after the team mean the team earns bonus, as the catch bonus is scaled.

    Args:
        prey_oob, caught: (N,) bool for this step.
        active_distances: (N, P) predator-prey distances, inactive predators far away.
    """
    num_predators = active_distances.shape[1]
    nearest_distance, nearest = active_distances.min(dim=1)
    forced = prey_oob & ~caught & (nearest_distance < assist_distance)
    index = torch.arange(num_predators, device=active_distances.device).unsqueeze(1)
    forcer = (nearest.unsqueeze(0) == index) & forced.unsqueeze(0)
    return forcer.float() * float(bonus) * num_predators
