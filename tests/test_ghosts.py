from __future__ import annotations

import torch

from ghosts import ghost_pursuit_velocity

DT = 0.02
LIMITS = {"max_speed": 4.5, "max_accel": 12.0, "dt": DT, "min_aim_height": 0.6}


def _step(position, velocity, target_position, target_velocity, lead):
    new_velocity = ghost_pursuit_velocity(position, velocity, target_position, target_velocity, lead, **LIMITS)
    return position + new_velocity * DT, new_velocity


def test_shapes_and_limits_hold_every_step():
    torch.manual_seed(0)
    n, p = 5, 3
    position = torch.rand(n, p, 3) * 6.0
    # Start inside the speed limit, as in the env (ghosts start at rest). The final speed clamp is a
    # projection onto the speed ball, so it never breaks the acceleration limit from inside the ball.
    velocity = torch.randn(n, p, 3)
    velocity = velocity / torch.linalg.norm(velocity, dim=-1, keepdim=True) * torch.rand(n, p, 1) * LIMITS["max_speed"]
    lead = torch.tensor([0.0, 0.25, 0.5])
    for _ in range(200):
        target_position = torch.rand(n, 3) * 6.0 + 1.0
        target_velocity = torch.randn(n, 3) * 4.0
        new_velocity = ghost_pursuit_velocity(position, velocity, target_position, target_velocity, lead, **LIMITS)
        assert new_velocity.shape == (n, p, 3)
        assert torch.all(torch.linalg.norm(new_velocity, dim=-1) <= LIMITS["max_speed"] + 1e-5)
        assert torch.all(torch.linalg.norm(new_velocity - velocity, dim=-1) <= LIMITS["max_accel"] * DT + 1e-5)
        position, velocity = position + new_velocity * DT, new_velocity


def test_reaches_full_speed_toward_a_far_stationary_target():
    position = torch.tensor([[[0.0, 0.0, 1.0]]])
    velocity = torch.zeros(1, 1, 3)
    target = torch.tensor([[20.0, 0.0, 1.0]])
    for _ in range(30):  # 0.6 s; full speed needs 4.5 / 12 = 0.375 s
        position, velocity = _step(position, velocity, target, torch.zeros(1, 3), torch.zeros(1))
    assert torch.allclose(velocity, torch.tensor([[[4.5, 0.0, 0.0]]]), atol=1e-4)


def test_lead_aims_ahead_of_a_moving_target():
    # From rest, the first velocity change points straight at the aim point.
    position = torch.zeros(1, 2, 3)
    target_position = torch.tensor([[4.0, 0.0, 1.0]])
    target_velocity = torch.tensor([[0.0, 4.0, 0.0]])
    lead = torch.tensor([0.0, 0.5])
    velocity = ghost_pursuit_velocity(position, torch.zeros(1, 2, 3), target_position, target_velocity, lead, **LIMITS)
    pure = velocity[0, 0] / torch.linalg.norm(velocity[0, 0])
    leading = velocity[0, 1] / torch.linalg.norm(velocity[0, 1])
    assert torch.allclose(pure, torch.tensor([4.0, 0.0, 1.0]) / torch.tensor([4.0, 0.0, 1.0]).norm(), atol=1e-5)
    assert torch.allclose(leading, torch.tensor([4.0, 2.0, 1.0]) / torch.tensor([4.0, 2.0, 1.0]).norm(), atol=1e-5)


def test_aim_point_never_below_the_minimum_height():
    position = torch.tensor([[[0.0, 0.0, 0.6]]])
    target_position = torch.tensor([[3.0, 0.0, 0.2]])  # diving prey below the aim floor
    velocity = ghost_pursuit_velocity(
        position, torch.zeros(1, 1, 3), target_position, torch.zeros(1, 3), torch.zeros(1), **LIMITS
    )
    assert abs(velocity[0, 0, 2].item()) < 1e-6
    assert velocity[0, 0, 0].item() > 0.0


def test_settles_on_a_reached_aim_point_without_oscillating():
    position = torch.zeros(1, 1, 3)
    velocity = torch.zeros(1, 1, 3)
    target = torch.tensor([[2.0, 0.0, 1.0]])
    for _ in range(300):
        position, velocity = _step(position, velocity, target, torch.zeros(1, 3), torch.zeros(1))
    assert torch.linalg.norm(position[0, 0] - target[0]).item() < 0.05
    assert torch.linalg.norm(velocity).item() < 0.5


def test_catches_a_slower_target_on_a_straight_line():
    position = torch.zeros(1, 1, 3)
    velocity = torch.zeros(1, 1, 3)
    target_position = torch.tensor([[3.0, 0.0, 1.0]])
    target_velocity = torch.tensor([[0.0, 3.0, 0.0]])
    caught = False
    for _ in range(500):  # 10 s
        position, velocity = _step(position, velocity, target_position, target_velocity, torch.zeros(1))
        target_position = target_position + target_velocity * DT
        if torch.linalg.norm(position[0, 0] - target_position[0]).item() < 0.3:
            caught = True
            break
    assert caught


def test_per_env_speed_limits():
    n, p = 4, 3
    caps = torch.tensor([3.5, 4.5, 5.5, 6.5]).view(n, 1, 1)
    limits = {**LIMITS, "max_speed": caps}
    position = torch.zeros(n, p, 3)
    velocity = torch.zeros(n, p, 3)
    target_position = torch.tensor([[30.0, 0.0, 1.0]]).repeat(n, 1)
    lead = torch.tensor([0.0, 0.25, 0.5])
    for _ in range(60):
        velocity = ghost_pursuit_velocity(position, velocity, target_position, torch.zeros(n, 3), lead, **limits)
        position = position + velocity * DT
    speed = torch.linalg.norm(velocity, dim=-1)
    assert torch.allclose(speed, caps.view(n, 1).expand(n, p), atol=1e-4)
