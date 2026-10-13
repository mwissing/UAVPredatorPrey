"""Vertical pillar obstacles for the 3v1 arena: placement, distances, collisions.

Pillars are infinite vertical cylinders, so all geometry is horizontal (x, y).
Collisions are detected analytically, not by physics: the rendered cylinders
have no collision shapes. Kept free of Isaac Lab imports so the math can be
unit-tested on the CPU.
"""

from __future__ import annotations

import torch

# An obstacle that could not be placed is parked far outside the arena: it is
# then never near a drone, never collides, and its observation stays constant.
PARKED_DISTANCE = 50.0


def sample_obstacle_xy(
    drone_xy: torch.Tensor,
    num_obstacles: int,
    *,
    placement_radius: float,
    obstacle_radius: float,
    min_axis_separation: float,
    spawn_clearance: float,
    candidates: int = 64,
    fixed_xy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pillar axes per environment by rejection.

    Args:
        drone_xy: (N, D, 2) horizontal spawn positions of all drones.
        placement_radius: maximum horizontal distance of an axis from the arena center.
        obstacle_radius: pillar radius.
        min_axis_separation: minimum distance between two pillar axes.
        spawn_clearance: minimum gap between a pillar surface and any spawning drone.
        fixed_xy: optional (N, 2) axis for the first pillar (NaN rows: none), for example a shield pillar next to
            the prey; the other pillars keep their separation from it. None leaves the sampling unchanged.

    Returns:
        (N, K, 2) axis positions and (N,) number of pillars that had to be parked.
    """
    n = drone_xy.shape[0]
    device = drone_xy.device
    placed = torch.full((n, num_obstacles, 2), PARKED_DISTANCE, device=device)
    parked = torch.zeros(n, dtype=torch.long, device=device)
    for k in range(num_obstacles):
        # Uniform over the disk: radius ~ sqrt(U).
        radius = placement_radius * torch.sqrt(torch.rand(n, candidates, device=device))
        angle = 2.0 * torch.pi * torch.rand(n, candidates, device=device)
        cand = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)  # (N, M, 2)
        drone_gap = torch.linalg.norm(cand[:, :, None, :] - drone_xy[:, None, :, :], dim=-1)  # (N, M, D)
        valid = (drone_gap >= obstacle_radius + spawn_clearance).all(dim=-1)
        if k > 0:
            axis_gap = torch.linalg.norm(cand[:, :, None, :] - placed[:, None, :k, :], dim=-1)  # (N, M, k)
            valid &= (axis_gap >= min_axis_separation).all(dim=-1)
        has_valid = valid.any(dim=1)
        first = torch.argmax(valid.to(torch.int8), dim=1)
        chosen = cand[torch.arange(n, device=device), first]
        placed[:, k] = torch.where(has_valid[:, None], chosen, placed[:, k])
        if k == 0 and fixed_xy is not None:
            has_fixed = torch.isfinite(fixed_xy).all(dim=-1)
            placed[:, 0] = torch.where(has_fixed[:, None], fixed_xy, placed[:, 0])
            parked += (~has_valid & ~has_fixed).long()
        else:
            parked += (~has_valid).long()
    return placed, parked


def shield_predator_xy(
    prey_xy: torch.Tensor,
    pred_xy: torch.Tensor,
    *,
    distance_range: tuple[float, float],
    spawn_radius: float,
    min_separation: float,
    candidates: int = 64,
) -> torch.Tensor:
    """Predator starts around the prey for shield starts: random distance in distance_range, random bearing.

    The bearings are independent, so the predators sometimes start on one side and sometimes spread out; the start
    does not prescribe a formation. Args: prey_xy (N, 2); pred_xy (N, P, 2) the regular starts, kept for a predator
    without a valid candidate. Returns (N, P, 2) starts inside spawn_radius and min_separation apart.
    """
    n, num_predators, _ = pred_xy.shape
    device = pred_xy.device
    out = pred_xy.clone()
    rows = torch.arange(n, device=device)
    for i in range(num_predators):
        distance = distance_range[0] + (distance_range[1] - distance_range[0]) * torch.rand(n, candidates, device=device)
        angle = 2.0 * torch.pi * torch.rand(n, candidates, device=device)
        cand = prey_xy[:, None, :] + distance[..., None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        valid = torch.linalg.norm(cand, dim=-1) <= spawn_radius
        if i > 0:
            gap = torch.linalg.norm(cand[:, :, None, :] - out[:, None, :i, :], dim=-1)  # (N, M, i)
            valid &= (gap >= min_separation).all(dim=-1)
        has_valid = valid.any(dim=1)
        chosen = cand[rows, torch.argmax(valid.to(torch.int8), dim=1)]
        out[:, i] = torch.where(has_valid[:, None], chosen, out[:, i])
    return out


def shield_pillar_xy(
    prey_xy: torch.Tensor,
    pred_xy: torch.Tensor,
    *,
    gap_range: tuple[float, float],
    obstacle_radius: float,
    placement_radius: float,
    spawn_clearance: float,
    candidates: int = 64,
) -> torch.Tensor:
    """Axis of a shield pillar right next to the prey: surface gap in gap_range, on a random side.

    Args: prey_xy (N, 2), pred_xy (N, P, 2) actual starts. Returns (N, 2) axes inside placement_radius and clear of
    every predator by spawn_clearance; NaN rows where no candidate fits (those envs keep the regular layout).
    """
    n = prey_xy.shape[0]
    device = prey_xy.device
    gap = gap_range[0] + (gap_range[1] - gap_range[0]) * torch.rand(n, candidates, device=device)
    angle = 2.0 * torch.pi * torch.rand(n, candidates, device=device)
    direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    cand = prey_xy[:, None, :] + (obstacle_radius + gap)[..., None] * direction  # (N, M, 2)
    valid = torch.linalg.norm(cand, dim=-1) <= placement_radius
    pred_gap = torch.linalg.norm(cand[:, :, None, :] - pred_xy[:, None, :, :], dim=-1)  # (N, M, P)
    valid &= (pred_gap >= obstacle_radius + spawn_clearance).all(dim=-1)
    has_valid = valid.any(dim=1)
    chosen = cand[torch.arange(n, device=device), torch.argmax(valid.to(torch.int8), dim=1)]
    return torch.where(has_valid[:, None], chosen, torch.full_like(chosen, float("nan")))


def surface_distance(points_xy: torch.Tensor, obstacle_xy: torch.Tensor, obstacle_radius: float) -> torch.Tensor:
    """Horizontal distance from points to pillar surfaces (negative inside a pillar).

    Args:
        points_xy: (N, D, 2) drone positions.
        obstacle_xy: (N, K, 2) pillar axes.

    Returns:
        (N, D, K) distances.
    """
    return torch.linalg.norm(points_xy[:, :, None, :] - obstacle_xy[:, None, :, :], dim=-1) - obstacle_radius


def collides(points_xy: torch.Tensor, obstacle_xy: torch.Tensor, obstacle_radius: float, margin: float) -> torch.Tensor:
    """(N, D) mask of drones whose body (radius `margin`) touches any pillar."""
    if obstacle_xy.shape[1] == 0:
        return torch.zeros(points_xy.shape[:2], dtype=torch.bool, device=points_xy.device)
    return (surface_distance(points_xy, obstacle_xy, obstacle_radius) < margin).any(dim=-1)


def axis_offsets_world(points: torch.Tensor, obstacle_xy: torch.Tensor) -> torch.Tensor:
    """World-frame vectors from drones to the nearest axis point at the drone's height.

    Args:
        points: (N, D, 3) drone positions.
        obstacle_xy: (N, K, 2) pillar axes.

    Returns:
        (N, D, K, 3) vectors with zero z component.
    """
    offsets = torch.zeros(*points.shape[:2], obstacle_xy.shape[1], 3, device=points.device, dtype=points.dtype)
    offsets[..., :2] = obstacle_xy[:, None, :, :] - points[:, :, None, :2]
    return offsets


def soft_clearance_penalty(clearance: torch.Tensor, margin: float, scale: float, dt: float) -> torch.Tensor:
    """Quadratic keep-out cost inside a band of width `margin` around pillar surfaces.

    Mirrors the soft arena-wall penalty: a crash alone is a sparse signal, while
    this cost grows smoothly as a drone approaches a pillar, so small steering
    corrections are rewarded before a collision happens. Returns a non-positive
    reward per step.
    """
    return -torch.square(torch.clamp(margin - clearance, min=0.0)) * scale * dt
