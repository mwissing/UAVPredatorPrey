"""Shield starts (end-game curriculum): predators around the prey and a pillar right next to it."""

from __future__ import annotations

import torch

from obstacles import sample_obstacle_xy, shield_pillar_xy, shield_predator_xy, surface_distance

R, PLACEMENT, CLEARANCE, SEPARATION = 0.4, 5.0, 0.5, 1.6
N, P, K = 64, 3, 10


def _drones(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    prey = (torch.rand(N, 2, generator=generator) * 2 - 1) * 2.5
    pred = (torch.rand(N, P, 2, generator=generator) * 2 - 1) * 4.0
    return prey, pred


def _sample(drone_xy: torch.Tensor, fixed: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return sample_obstacle_xy(drone_xy, K, placement_radius=PLACEMENT, obstacle_radius=R, min_axis_separation=SEPARATION,
                              spawn_clearance=CLEARANCE, fixed_xy=fixed)


def test_without_a_fixed_pillar_the_placement_is_unchanged() -> None:
    prey, pred = _drones(0)
    drone_xy = torch.cat((pred, prey[:, None]), dim=1)
    regular, parked = _sample(drone_xy, None)
    no_fixed, parked_nan = _sample(drone_xy, torch.full((N, 2), float("nan")))
    assert torch.equal(regular, no_fixed) and torch.equal(parked, parked_nan)


def test_fixed_pillar_takes_slot_0_and_the_others_keep_their_separation() -> None:
    prey, pred = _drones(1)
    drone_xy = torch.cat((pred, prey[:, None]), dim=1)
    torch.manual_seed(3)
    fixed = shield_pillar_xy(prey, pred, gap_range=(0.3, 0.8), obstacle_radius=R, placement_radius=PLACEMENT,
                             spawn_clearance=CLEARANCE)
    fixed[::2] = float("nan")                                       # every other env keeps the regular layout
    placed, _ = _sample(drone_xy, fixed)
    has = torch.isfinite(fixed).all(dim=-1)
    assert has.any() and torch.equal(placed[has, 0], fixed[has])
    real = placed[has][:, 1:]                                     # the other pillars of shield envs
    ok = torch.linalg.norm(real, dim=-1) < 40.0                    # placed, not parked
    gap = torch.linalg.norm(real - placed[has, 0][:, None], dim=-1)
    assert (gap[ok] >= SEPARATION - 1e-6).all()
    others = surface_distance(drone_xy[has], real, R)              # (n, D, K-1): drones vs the other pillars
    assert (others.transpose(1, 2)[ok] >= CLEARANCE - 1e-6).all()


def test_shield_pillar_sits_next_to_the_prey_clear_of_the_predators() -> None:
    prey, pred = _drones(2)
    torch.manual_seed(4)
    axis = shield_pillar_xy(prey, pred, gap_range=(0.3, 0.8), obstacle_radius=R, placement_radius=PLACEMENT,
                            spawn_clearance=CLEARANCE)
    has = torch.isfinite(axis).all(dim=-1)
    assert has.float().mean() > 0.9
    gap = torch.linalg.norm(prey[has] - axis[has], dim=-1) - R
    assert ((gap >= 0.3 - 1e-6) & (gap <= 0.8 + 1e-6)).all()
    assert (torch.linalg.norm(axis[has], dim=-1) <= PLACEMENT + 1e-6).all()
    assert (torch.linalg.norm(pred[has] - axis[has][:, None], dim=-1) - R >= CLEARANCE - 1e-6).all()
    # Predators packed around the prey leave no free side: no shield pillar, NaN.
    ring = prey[:1, None] + 0.9 * torch.stack((torch.cos(torch.arange(8) * 0.785), torch.sin(torch.arange(8) * 0.785)), -1)
    blocked = shield_pillar_xy(prey[:1], ring, gap_range=(0.3, 0.8), obstacle_radius=R, placement_radius=PLACEMENT,
                               spawn_clearance=CLEARANCE)
    assert torch.isnan(blocked).all()


def test_predators_start_around_the_prey_inside_the_arena() -> None:
    prey, pred = _drones(3)
    torch.manual_seed(5)
    around = shield_predator_xy(prey, pred, distance_range=(3.0, 6.0), spawn_radius=4.25, min_separation=0.8)
    distance = torch.linalg.norm(around - prey[:, None], dim=-1)
    moved = ~torch.isclose(around, pred).all(dim=-1)              # fallback keeps the regular start
    assert moved.float().mean() > 0.9
    assert ((distance[moved] >= 3.0 - 1e-6) & (distance[moved] <= 6.0 + 1e-6)).all()
    assert (torch.linalg.norm(around[moved], dim=-1) <= 4.25 + 1e-6).all()
    pair = torch.linalg.norm(around[:, :, None] - around[:, None, :], dim=-1)
    all_moved = moved.all(dim=1)
    off_diagonal = ~torch.eye(P, dtype=torch.bool)
    assert (pair[all_moved][:, off_diagonal] >= 0.8 - 1e-6).all()
