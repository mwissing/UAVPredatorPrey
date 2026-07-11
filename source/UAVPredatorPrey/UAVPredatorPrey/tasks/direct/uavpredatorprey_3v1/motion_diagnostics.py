"""Pure tensor helpers for 3v1 pursuit-motion diagnostics."""

from __future__ import annotations

import math

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    count = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / count


def _masked_rms(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_masked_mean(torch.square(values), mask))


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, -torch.inf))
    return torch.where(mask.any(), masked.amax(), torch.zeros((), dtype=values.dtype, device=values.device))


def _masked_fraction(condition: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask_f = mask.to(dtype=dtype)
    count = mask_f.sum().clamp(min=1.0)
    return (condition & mask).to(dtype=dtype).sum() / count


def compute_motion_diagnostics(
    *,
    predator_positions: torch.Tensor,
    prey_positions: torch.Tensor,
    predator_velocities: torch.Tensor,
    prey_velocities: torch.Tensor,
    predator_alive: torch.Tensor,
    raw_distance_progress: torch.Tensor,
    progress_valid: torch.Tensor,
    predator_upright: torch.Tensor,
    progress_clip: float,
) -> dict[str, torch.Tensor]:
    """Summarize speed, pursuit direction, progress shaping, and tilt without changing rewards."""

    to_prey = prey_positions.unsqueeze(1) - predator_positions
    line_of_sight = to_prey / torch.linalg.vector_norm(to_prey, dim=-1, keepdim=True).clamp(min=1.0e-6)
    relative_velocity = predator_velocities - prey_velocities.unsqueeze(1)
    closing_speed = torch.sum(relative_velocity * line_of_sight, dim=-1)
    lateral_velocity = relative_velocity - closing_speed.unsqueeze(-1) * line_of_sight

    predator_speed = torch.linalg.vector_norm(predator_velocities, dim=-1)
    prey_speed = torch.linalg.vector_norm(prey_velocities, dim=-1)
    lateral_speed = torch.linalg.vector_norm(lateral_velocity, dim=-1)
    alive = predator_alive.to(dtype=torch.bool)
    valid_progress = progress_valid.to(dtype=torch.bool)
    dtype = predator_speed.dtype
    clip_threshold = max(float(progress_clip) - 1.0e-6, 0.0)

    return {
        "predator_speed_mean": _masked_mean(predator_speed, alive),
        "predator_speed_rms": _masked_rms(predator_speed, alive),
        "predator_speed_max": _masked_max(predator_speed, alive),
        "predator_speed_above_4mps_fraction": _masked_fraction(predator_speed > 4.0, alive, dtype),
        "predator_speed_above_6mps_fraction": _masked_fraction(predator_speed > 6.0, alive, dtype),
        "predator_closing_speed_mean": _masked_mean(closing_speed, alive),
        "predator_closing_speed_positive_fraction": _masked_fraction(closing_speed > 0.0, alive, dtype),
        "predator_closing_speed_negative_fraction": _masked_fraction(closing_speed < 0.0, alive, dtype),
        "predator_lateral_relative_speed_mean": _masked_mean(lateral_speed, alive),
        "predator_progress_positive_fraction": _masked_fraction(raw_distance_progress > 0.0, valid_progress, dtype),
        "predator_progress_negative_fraction": _masked_fraction(raw_distance_progress < 0.0, valid_progress, dtype),
        "predator_progress_clip_fraction": _masked_fraction(
            torch.abs(raw_distance_progress) >= clip_threshold,
            valid_progress,
            dtype,
        ),
        "predator_upright_mean": _masked_mean(predator_upright, alive),
        "predator_tilt_over_45deg_fraction": _masked_fraction(
            predator_upright < math.cos(math.pi / 4.0),
            alive,
            dtype,
        ),
        "prey_speed_mean": prey_speed.mean(),
        "prey_speed_rms": torch.sqrt(torch.square(prey_speed).mean()),
        "prey_speed_max": prey_speed.amax(),
        "prey_speed_above_4mps_fraction": (prey_speed > 4.0).to(dtype=dtype).mean(),
        "prey_speed_above_6mps_fraction": (prey_speed > 6.0).to(dtype=dtype).mean(),
    }
