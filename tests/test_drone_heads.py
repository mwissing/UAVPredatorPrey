"""Separate decision heads per predator drone (predator_drone_heads): converted from a shared head they act as before,
each head moves only its own drone, and checkpoints of the two layouts do not load into each other."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import gymnasium
import pytest
import torch

AGENTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "agents"
)
PACKAGE_NAME = "_drone_heads_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

K, P, BLOCK = 4, 3, 30 + 4 * 4          # pillars, predators, per-drone block (own 12, prey 6, teammates 12, pillars)


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _predator(drone_heads: bool):
    torch.manual_seed(1)
    return attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(P * BLOCK), _box(P * 4), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=K, predator_mask_inactive_teammates=True, predator_drone_heads=drone_heads,
    )


def _states(batch: int) -> torch.Tensor:
    blocks = torch.randn(batch, P, BLOCK, generator=torch.Generator().manual_seed(4))
    blocks[:, :, 30:].reshape(batch, P, K, 4)[..., 3] = 0.3                 # pillars present
    return blocks.reshape(batch, -1)


def _converted():
    shared, heads = _predator(False), _predator(True)
    heads.load_state_dict(attention_models.split_actor_head(shared.state_dict(), P))
    return shared, heads


def test_split_heads_act_like_the_shared_head() -> None:
    shared, heads = _converted()
    states = _states(5)
    inputs = {"states": states, "raw_states": states}
    # Same arithmetic per drone; three small matmuls may round differently from one large one.
    assert torch.allclose(shared.compute(inputs)[0], heads.compute(inputs)[0], atol=1e-6)


def test_each_head_moves_only_its_own_drone() -> None:
    _, heads = _converted()
    states = _states(5)
    inputs = {"states": states, "raw_states": states}
    before = heads.compute(inputs)[0].detach()
    with torch.no_grad():
        for parameter in heads.drone_heads[1].parameters():
            parameter.add_(0.5)
    after = heads.compute(inputs)[0].detach()
    changed = (after - before).abs().view(5, P, 4).amax(dim=(0, 2))
    assert changed[1] > 1e-3 and changed[0] == 0 and changed[2] == 0


def test_the_two_head_layouts_do_not_load_into_each_other() -> None:
    shared, heads = _converted()
    with pytest.raises(RuntimeError):
        _predator(False).load_state_dict(heads.state_dict())
    with pytest.raises(RuntimeError):
        _predator(True).load_state_dict(shared.state_dict())


def test_the_prey_keeps_its_single_head() -> None:
    prey = attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(12 + 6 * P + 4 * K), _box(4), device="cpu", hidden_size=32, attention_size=16, num_obstacles=K,
        predator_drone_heads=True,
    )
    assert hasattr(prey, "actor_head") and not hasattr(prey, "drone_heads")
