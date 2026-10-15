"""Scripted pursuers that respect pillars: the positive control for the pillar shield (ghosts.scripted_pursuit_velocity)."""

from __future__ import annotations

import math

import torch

from ghosts import scripted_pursuit_velocity

DT, VMAX, AMAX, R = 0.02, 6.0, 12.0, 0.4
FAR = torch.full((1, 1, 2), 50.0)                    # one parked pillar


def _step(position, velocity, alive, prey, mode, pillars=FAR):
    return scripted_pursuit_velocity(position, velocity, alive, prey, torch.zeros_like(prey), pillars, mode=mode,
                                     max_speed=VMAX, max_accel=AMAX, dt=DT, min_aim_height=0.6, obstacle_radius=R)


def test_direct_pursuit_heads_for_the_prey_within_the_limits() -> None:
    position = torch.tensor([[[3.0, 0.0, 2.0], [0.0, -4.0, 2.0], [-2.0, 2.0, 2.0]]])
    velocity = torch.zeros(1, 3, 3)
    prey = torch.tensor([[0.0, 0.0, 2.0]])
    new = _step(position, velocity, torch.ones(1, 3, dtype=torch.bool), prey, "direct")
    to_prey = prey[:, None] - position
    cos = (new * to_prey).sum(-1) / (new.norm(dim=-1) * to_prey.norm(dim=-1))
    assert (cos > 0.999).all()
    assert (new.norm(dim=-1) <= AMAX * DT + 1e-6).all()          # from rest: one step of acceleration


def test_encircle_spreads_a_one_sided_pack_and_closes_in_once_spread() -> None:
    prey = torch.tensor([[0.0, 0.0, 2.0]])
    alive = torch.ones(1, 3, dtype=torch.bool)
    pack = torch.tensor([[[3.0, -0.3, 2.0], [3.0, 0.0, 2.0], [3.0, 0.3, 2.0]]])   # all on one side
    new = _step(pack, torch.zeros(1, 3, 3), alive, prey, "encircle")
    heading = torch.atan2(new[0, :, 1], new[0, :, 0])
    assert heading.max() - heading.min() > 1.0                    # the outer two fan out to different sides
    spread = torch.tensor([[[3.0 * math.cos(a), 3.0 * math.sin(a), 2.0] for a in (0.0, 2.094, 4.189)]])
    new = _step(spread, torch.zeros(1, 3, 3), alive, prey, "encircle")
    inward = -(new[0, :, :2] * spread[0, :, :2]).sum(-1) / (new[0, :, :2].norm(dim=-1) * 3.0)
    assert (inward > 0.99).all()                                  # spread out: everyone closes in on the prey


def test_pursuer_goes_around_a_pillar_without_touching_it() -> None:
    for start_y in (0.05, 0.0):                                   # slightly off-axis and exactly head-on
        _around_pillar(start_y)


def _around_pillar(start_y: float) -> None:
    position = torch.tensor([[[-3.0, start_y, 2.0]]])
    velocity = torch.zeros(1, 1, 3)
    prey = torch.tensor([[3.0, 0.0, 2.0]])
    pillar = torch.tensor([[[0.0, 0.0]]])                         # right on the straight line
    alive = torch.ones(1, 1, dtype=torch.bool)
    closest = float("inf")
    for _ in range(400):
        velocity = _step(position, velocity, alive, prey, "direct", pillar)
        position = position + velocity * DT
        closest = min(closest, float(position[0, 0, :2].norm()) - R)
    assert closest > 0.06                                         # never inside the crash margin
    assert float((position[0, 0] - prey[0]).norm()) < 0.5         # and it gets there


def test_dead_pursuers_stop() -> None:
    position = torch.tensor([[[3.0, 0.0, 2.0], [0.0, 3.0, 2.0], [-3.0, 0.0, 2.0]]])
    velocity = torch.ones(1, 3, 3)
    alive = torch.tensor([[True, False, True]])
    for mode in ("direct", "encircle"):
        new = _step(position, velocity, alive, torch.tensor([[0.0, 0.0, 2.0]]), mode)
        assert torch.equal(new[0, 1], torch.zeros(3))
