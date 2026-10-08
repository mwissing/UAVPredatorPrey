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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pillar axes per environment by rejection.

    Args:
        drone_xy: (N, D, 2) horizontal spawn positions of all drones.
        placement_radius: maximum horizontal distance of an axis from the arena center.
        obstacle_radius: pillar radius.
        min_axis_separation: minimum distance between two pillar axes.
        spawn_clearance: minimum gap between a pillar surface and any spawning drone.

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
        parked += (~has_valid).long()
    return placed, parked


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
