"""Per-drone PPO ratios for the predator team (Stage 8 review): one ratio and clip per drone, crashed drones out."""

from __future__ import annotations

import pytest
import torch

from recurrent_mappo import _live_drones, _per_drone_policy_loss

CLIP = 0.2


def _joint_loss(next_dims, old_dims, advantages):
    ratio = torch.exp((next_dims - old_dims).sum(-1, keepdim=True))
    return -torch.min(advantages * ratio, advantages * torch.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()


def _batch(seed: int, drones: int = 3, scale: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(64, 4 * drones, generator=g)
    new = old + scale * torch.randn(64, 4 * drones, generator=g)
    advantages = torch.randn(64, 1, generator=g)
    return new, old, advantages


def test_one_drone_reduces_to_standard_ppo() -> None:
    new, old, advantages = _batch(0, drones=1)
    loss, _, _ = _per_drone_policy_loss(new, old, advantages, torch.ones(64, 1, dtype=torch.bool), CLIP)
    assert torch.allclose(loss, _joint_loss(new, old, advantages), atol=1e-6)


def test_crashed_drones_do_not_count() -> None:
    new, old, advantages = _batch(1)
    alive = torch.ones(64, 3, dtype=torch.bool)
    alive[:, 2] = False
    loss, kl, _ = _per_drone_policy_loss(new, old, advantages, alive, CLIP)
    noisy = new.clone()
    noisy[:, 8:] += torch.randn(64, 4, generator=torch.Generator().manual_seed(2))   # the dead drone's dimensions
    loss2, kl2, _ = _per_drone_policy_loss(noisy, old, advantages, alive, CLIP)
    assert torch.allclose(loss, loss2) and torch.allclose(kl, kl2)
    new = new.clone().requires_grad_(True)
    _per_drone_policy_loss(new, old, advantages, alive, CLIP)[0].backward()
    assert new.grad[:, 8:].abs().max() == 0 and new.grad[:, :8].abs().max() > 0


def test_one_drone_s_large_change_clips_only_that_drone() -> None:
    new, old, advantages = _batch(3, scale=0.0)
    advantages = advantages.abs() + 0.1                       # positive advantages: ratios above 1 + clip are capped
    new = new.clone()
    new[:, 0] += 1.0                                          # drone 0 moves far, drones 1 and 2 not at all
    new[:, 4] += 0.05                                         # drone 1 moves a little
    new = new.requires_grad_(True)
    alive = torch.ones(64, 3, dtype=torch.bool)
    _, _, clipped = _per_drone_policy_loss(new, old, advantages, alive, CLIP)
    assert clipped == pytest.approx(1 / 3)                    # only drone 0's ratio leaves the clip range
    _per_drone_policy_loss(new, old, advantages, alive, CLIP)[0].backward()
    assert new.grad[:, 0].abs().max() == 0                    # clipped: no gradient for drone 0
    assert new.grad[:, 4].abs().max() > 0                     # drone 1 still learns
    joint = new.detach().clone().requires_grad_(True)
    _joint_loss(joint, old, advantages).backward()
    assert joint.grad.abs().max() == 0                        # a joint ratio clips the whole team


def test_live_drones_come_from_the_projected_gravity() -> None:
    raw = torch.zeros(2, 3 * 70)
    raw.view(2, 3, 70)[:, :2, 6:9] = torch.tensor([0.0, 0.0, -1.0])   # drones 0 and 1 alive, drone 2 crashed
    assert _live_drones(raw, 3).tolist() == [[True, True, False]] * 2
