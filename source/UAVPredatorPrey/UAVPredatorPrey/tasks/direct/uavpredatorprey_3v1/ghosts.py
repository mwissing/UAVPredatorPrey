"""Scripted "ghost" pursuers for training the prey on its own.

A ghost is a kinematic predator: it ignores its policy's actions, follows a speed- and
acceleration-limited lead pursuit of the prey, never crashes, and flies through pillars. Against
ghosts, pillars give the prey no cover and slowing down gets it caught, so the only way to survive
is to fly fast without hitting pillars. Kept free of Isaac Lab imports so the math can be unit-tested
on the CPU.
"""

from __future__ import annotations

import torch


def ghost_pursuit_velocity(
    position: torch.Tensor,
    velocity: torch.Tensor,
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    lead_times: torch.Tensor,
    *,
    max_speed: float,
    max_accel: float,
    dt: float,
    min_aim_height: float,
) -> torch.Tensor:
    """Next velocity of each ghost.

    Each ghost aims at the target's predicted position after its lead time and accelerates toward
    full speed in that direction, with the velocity change per step limited to max_accel * dt.

    Args:
        position: (N, P, 3) ghost positions relative to the env origin [m].
        velocity: (N, P, 3) current ghost velocities [m/s].
        target_position: (N, 3) prey position relative to the env origin [m].
        target_velocity: (N, 3) prey velocity [m/s].
        lead_times: (P,) per-ghost lead times [s].
        min_aim_height: lowest aim point [m], so ghosts never chase a diving prey into the ground.

    Returns:
        (N, P, 3) velocities with norm at most max_speed.
    """
    aim = target_position[:, None, :] + lead_times[None, :, None] * target_velocity[:, None, :]
    aim[..., 2] = aim[..., 2].clamp(min=min_aim_height)
    offset = aim - position
    distance = torch.linalg.norm(offset, dim=-1, keepdim=True)
    # Full speed toward the aim point, slowing down within one step of it so the ghost does not
    # oscillate around a lead point it has already reached.
    desired = offset / distance.clamp(min=1e-6) * max_speed * torch.clamp(distance / (max_speed * dt), max=1.0)
    change = desired - velocity
    change_norm = torch.linalg.norm(change, dim=-1, keepdim=True)
    change = change * torch.clamp(max_accel * dt / change_norm.clamp(min=1e-9), max=1.0)
    new_velocity = velocity + change
    speed = torch.linalg.norm(new_velocity, dim=-1, keepdim=True)
    return new_velocity * torch.clamp(max_speed / speed.clamp(min=1e-9), max=1.0)
