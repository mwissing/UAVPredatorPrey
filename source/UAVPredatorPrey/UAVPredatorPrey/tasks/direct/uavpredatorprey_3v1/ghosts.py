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
    max_speed: float | torch.Tensor,
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
        max_speed: speed limit [m/s], a scalar or an (N, 1, 1) tensor of per-env limits.
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


def scripted_pursuit_velocity(
    position: torch.Tensor,
    velocity: torch.Tensor,
    alive: torch.Tensor,
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    obstacle_xy: torch.Tensor,
    *,
    mode: str,
    max_speed: float,
    max_accel: float,
    dt: float,
    min_aim_height: float,
    obstacle_radius: float,
    ring_radius: float = 2.5,
    spread_tolerance: float = 0.8,
    avoid_range: float = 1.0,
) -> torch.Tensor:
    """Next velocity of scripted predators that respect pillars (a diagnostic, not a training opponent).

    Unlike the lead-pursuit ghosts above, these pursuers treat pillars as obstacles and can crash. Two modes:
    - "direct": every predator flies at the prey (the control);
    - "encircle": the live predators take evenly spaced slots on a circle around the prey, assigned in the cyclic
      order of their current bearings so they do not cross. The circle shrinks from ring_radius to the prey only
      as the slots are reached: the radius scales with the largest bearing error relative to spread_tolerance
      [rad]. This is a stateless encirclement law.
    Both add a repulsive plus tangential (vortex) field around pillars within avoid_range of their surface; the
    tangential part turns toward the goal's side, so a predator slides around a pillar instead of stalling in front
    of it. Dead predators stop.

    Args:
        position, velocity: (N, P, 3) relative to the env origin [m, m/s]; alive: (N, P) bool.
        target_position, target_velocity: (N, 3) prey state; obstacle_xy: (N, K, 2) pillar axes (parked ones are far).
    Returns:
        (N, P, 3) velocities, speed <= max_speed, change per step <= max_accel * dt.
    """
    n, num_predators, _ = position.shape
    rel = position[..., :2] - target_position[:, None, :2]                       # (N, P, 2) predator from prey
    if mode == "direct":
        aim_xy = target_position[:, None, :2].expand(n, num_predators, 2)
    elif mode == "encircle":
        bearing = torch.atan2(rel[..., 1], rel[..., 0])                          # (N, P)
        # Dead predators sort last; the live ones keep their cyclic order around the prey.
        key = torch.where(alive, bearing, torch.full_like(bearing, 10.0))
        order = torch.argsort(key, dim=1)
        live = alive.sum(dim=1, keepdim=True).clamp(min=1)                       # (N, 1)
        rank = torch.argsort(order, dim=1)                                       # position of each predator in the order
        spacing = 2.0 * torch.pi / live                                          # (N, 1)
        slot = rank.float() * spacing                                            # (N, P) relative slot angles
        # Rotate the whole formation to the circular mean of (bearing - slot) over live predators: least travel.
        offset = torch.where(alive, bearing - slot, torch.zeros_like(bearing))
        phase = torch.atan2((torch.sin(offset) * alive).sum(1), (torch.cos(offset) * alive).sum(1))[:, None]
        goal_bearing = phase + slot
        error = torch.remainder(bearing - goal_bearing + torch.pi, 2.0 * torch.pi) - torch.pi
        worst = torch.where(alive, error.abs(), torch.zeros_like(error)).amax(dim=1, keepdim=True)
        radius = ring_radius * torch.clamp(worst / spread_tolerance, max=1.0)    # (N, 1): close in once spread
        aim_xy = target_position[:, None, :2] + radius[..., None] * torch.stack(
            (torch.cos(goal_bearing), torch.sin(goal_bearing)), dim=-1
        )
    else:
        raise ValueError(f"unknown scripted pursuit mode {mode!r}")
    aim = torch.cat((aim_xy, target_position[:, None, 2:3].expand(n, num_predators, 1).clamp(min=min_aim_height)), dim=-1)
    aim = aim + target_velocity[:, None, :] * dt                                # follow the prey's motion
    offset = aim - position
    distance = torch.linalg.norm(offset, dim=-1, keepdim=True)
    desired = offset / distance.clamp(min=1e-6) * max_speed * torch.clamp(distance / (max_speed * dt), max=1.0)

    # Pillar avoidance (horizontal). It starts avoid_range plus the course-change distance v^2 / (2 a) before a
    # surface, because a kinematic pursuer at full speed cannot turn any faster. Within that reach, the part of the
    # desired velocity that points into the pillar is removed and a sideways (vortex) push toward the goal's side
    # is added, so a pursuer slides around a pillar instead of stalling in front of it or touching it.
    away = position[:, :, None, :2] - obstacle_xy[:, None, :, :]                # (N, P, K, 2)
    centre_distance = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
    normal = away / centre_distance
    surface = centre_distance - obstacle_radius
    reach = avoid_range + max_speed**2 / (2.0 * max_accel)
    weight = torch.clamp((reach - surface) / reach, min=0.0, max=1.0)           # (N, P, K, 1)
    tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
    goal_dir = offset[:, :, None, :2] / distance[:, :, None, :].clamp(min=1e-6)
    side = torch.sign((tangent * goal_dir).sum(-1, keepdim=True))
    side = torch.where(side == 0, torch.ones_like(side), side)
    desired_xy = desired[..., :2]
    inward = torch.clamp((desired_xy[:, :, None, :] * normal).sum(-1, keepdim=True), max=0.0)  # (N, P, K, 1)
    desired_xy = desired_xy - (weight * inward * normal).sum(dim=2)
    desired_xy = desired_xy + (weight * side * tangent).sum(dim=2) * max_speed
    desired = torch.cat((desired_xy, desired[..., 2:]), dim=-1)
    speed = torch.linalg.norm(desired, dim=-1, keepdim=True)
    desired = desired * torch.clamp(max_speed / speed.clamp(min=1e-9), max=1.0)

    change = desired - velocity
    change_norm = torch.linalg.norm(change, dim=-1, keepdim=True)
    change = change * torch.clamp(max_accel * dt / change_norm.clamp(min=1e-9), max=1.0)
    new_velocity = velocity + change
    speed = torch.linalg.norm(new_velocity, dim=-1, keepdim=True)
    new_velocity = new_velocity * torch.clamp(max_speed / speed.clamp(min=1e-9), max=1.0)
    return new_velocity * alive[..., None]
