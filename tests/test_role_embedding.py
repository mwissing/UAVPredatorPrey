"""Per-index role vectors for the predator team: opt-in, exact at zero, and only their differences matter."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import gymnasium
import pytest
import torch

from recurrent_mappo import _obstacle_param_groups

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
PACKAGE_NAME = "_role_embedding_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

K, OBSTACLE_DIM, PREDATORS = 4, 4, 3
BLOCK = 30 + K * OBSTACLE_DIM  # own 12, prey 6, two teammates 6 each, K pillars


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _actor(roles: bool, team: bool = True):
    torch.manual_seed(1)
    observations, actions = (PREDATORS * BLOCK, PREDATORS * 4) if team else (BLOCK, 4)
    model = attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(observations), _box(actions), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=K, predator_mask_inactive_teammates=True, predator_role_embedding=roles,
    )
    torch.nn.init.normal_(model.obstacle_out.weight, std=0.5)  # a trained obstacle branch
    return model


def _states(batch: int, seed: int, identical_drones: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randn(batch, 1 if identical_drones else PREDATORS, BLOCK, generator=generator)
    blocks[:, :, 30:].view(batch, -1, K, OBSTACLE_DIM)[..., 3] = 0.5  # all pillars present
    return blocks.expand(batch, PREDATORS, BLOCK).reshape(batch, -1)


def _actions(model, states: torch.Tensor) -> torch.Tensor:
    """(B, drones, 4) mean actions."""
    return model.compute({"states": states, "raw_states": states})[0].view(states.shape[0], PREDATORS, 4)


def _with_random_roles(model) -> None:
    with torch.no_grad():
        model.role_embedding.copy_(torch.randn(model.role_embedding.shape, generator=torch.Generator().manual_seed(7)))


def test_zero_roles_leave_a_loaded_network_unchanged() -> None:
    old, new = _actor(roles=False), _actor(roles=True)
    new.load_state_dict(old.state_dict())  # the role vectors are allowed to be missing and stay zero
    states = _states(6, seed=0)
    for inputs in ({"states": states}, {"states": states, "raw_states": states}):
        assert torch.equal(old.compute(inputs)[0], new.compute(inputs)[0])


def test_a_role_checkpoint_does_not_load_without_the_flag() -> None:
    with pytest.raises(RuntimeError, match="role_embedding"):
        _actor(roles=False).load_state_dict(_actor(roles=True).state_dict())


def test_roles_let_identical_drones_act_differently() -> None:
    model = _actor(roles=True)
    states = _states(5, seed=1, identical_drones=True)
    symmetric = _actions(model, states)
    assert torch.equal(symmetric[:, 0], symmetric[:, 1]) and torch.equal(symmetric[:, 1], symmetric[:, 2])
    _with_random_roles(model)
    broken = _actions(model, states)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        assert (broken[:, i] - broken[:, j]).abs().amax() > 1e-3


def test_a_shared_shift_of_all_role_vectors_changes_nothing() -> None:
    model = _actor(roles=True)
    _with_random_roles(model)
    states = _states(5, seed=2)
    before = _actions(model, states)
    with torch.no_grad():
        model.role_embedding.add_(torch.randn(16, generator=torch.Generator().manual_seed(8)))
    assert torch.allclose(_actions(model, states), before, atol=1e-6)


def test_the_role_gradient_is_nonzero_and_centered() -> None:
    model = _actor(roles=True)
    _actions(model, _states(8, seed=3)).square().sum().backward()
    grad = model.role_embedding.grad
    assert grad.abs().amax() > 0
    assert torch.allclose(grad.sum(dim=0), torch.zeros(16), atol=1e-6)


def test_roles_learn_in_the_fast_group() -> None:
    model = _actor(roles=True)
    base, fast = _obstacle_param_groups([model], 1e-5, 30.0)
    assert any(p is model.role_embedding for p in fast["params"])
    assert not any(p is model.role_embedding for p in base["params"])
    assert fast["lr"] == pytest.approx(3e-4)


def test_the_prey_ignores_the_flag() -> None:
    assert not hasattr(_actor(roles=True, team=False), "role_embedding")
