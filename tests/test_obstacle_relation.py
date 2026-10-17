"""Pillar-prey relation branches (Stage 8 review): opt-in, exact at zero, and they carry what the old paths cannot."""

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
PACKAGE_NAME = "_obstacle_relation_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

K, P, BLOCK = 4, 3, 30 + 4 * 4          # pillars, predators, per-drone block (own 12, prey 6, teammates 12, pillars)
PREY_OBS = 12 + 6 * P
STATE = P * BLOCK + PREY_OBS + 4 * K
PRESENT, PARKED = 0.3, 9.9               # raw surface distance / scale: in the arena vs parked far away


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _actor(relation: bool, team: bool = True):
    torch.manual_seed(1)
    observations, actions = (P * BLOCK, P * 4) if team else (BLOCK, 4)
    return attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(observations), _box(actions), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=K, predator_mask_inactive_teammates=True, predator_obstacle_relation=relation,
    )


def _critic(relation: bool):
    torch.manual_seed(2)
    return attention_models.RecurrentEntityAttentionCentralizedValueModel(
        _box(STATE), _box(1), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24, num_obstacles=K,
        obstacle_relation=relation,
    )


def _actor_states(batch: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randn(batch, P, BLOCK, generator=generator)
    blocks[:, :, 30:].view(batch, P, K, 4)[..., 3] = PRESENT
    return blocks.reshape(batch, -1)


def _critic_states(batch: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(batch, STATE, generator=generator)
    blocks = states[:, : P * BLOCK].view(batch, P, BLOCK)
    blocks[:, :, 6:9] = torch.nn.functional.normalize(blocks[:, :, 6:9], dim=-1)        # live drones: unit gravity
    blocks[:, :, 30:].reshape(batch, P, K, 4)[..., 3] = PRESENT
    blocks[:, :, 30:] = blocks[:, :, 30:].reshape(batch, P, K, 4).reshape(batch, P, -1)
    states[:, P * BLOCK + PREY_OBS :].view(batch, K, 4)[..., 3] = PRESENT
    return states


def _randomize(module) -> None:
    torch.manual_seed(5)
    torch.nn.init.normal_(module.obstacle_out_rel.weight, std=0.5)
    torch.nn.init.normal_(module.obstacle_out_rel.bias, std=0.5)


def test_segment_distance_marks_the_blocking_pillar() -> None:
    target = torch.tensor([[4.0, 0.0]]).expand(4, 2)
    axes = torch.tensor([[2.0, 0.1], [2.0, 2.0], [-1.0, 0.0], [5.0, 0.0]])
    distance = attention_models._segment_axis_distance(axes, target).squeeze(-1)
    assert torch.allclose(distance, torch.tensor([0.1, 2.0, 1.0, 1.0]), atol=1e-6)


def test_zero_relation_branches_leave_loaded_networks_unchanged() -> None:
    old, new = _actor(False), _actor(True)
    new.load_state_dict(old.state_dict())
    states = _actor_states(5, seed=0)
    for inputs in ({"states": states}, {"states": states, "raw_states": states}):
        assert torch.equal(old.compute(inputs)[0], new.compute(inputs)[0])
    old_c, new_c = _critic(False), _critic(True)
    new_c.load_state_dict(old_c.state_dict())
    shared = _critic_states(5, seed=1)
    inputs = {"states": shared, "raw_states": shared}
    assert torch.equal(old_c.compute(inputs)[0], new_c.compute(inputs)[0])


def test_the_actor_branch_ignores_parked_pillars_and_sees_present_ones() -> None:
    model = _actor(True)
    _randomize(model)
    states = _actor_states(4, seed=2)
    pillars = states.view(4, P, BLOCK)[:, :, 30:].reshape(4, P, K, 4)
    pillars[:, :, 3, 3] = PARKED
    out = lambda s: model.compute({"states": s, "raw_states": s})[0]
    moved_parked = states.clone()
    moved_parked.view(4, P, BLOCK)[:, :, 30:].reshape(4, P, K, 4)[:, :, 3, :3] += 2.0
    moved_present = states.clone()
    moved_present.view(4, P, BLOCK)[:, :, 30:].reshape(4, P, K, 4)[:, :, 0, :3] += 2.0
    assert torch.equal(out(states), out(moved_parked))
    assert (out(states) - out(moved_present)).abs().amax() > 1e-4


def test_the_prey_and_missing_raw_states_get_no_branch() -> None:
    assert not hasattr(_actor(True, team=False), "obstacle_rel_encoder")
    old, new = _actor(False), _actor(True)
    new.load_state_dict(old.state_dict(), strict=False)
    _randomize(new)
    states = _actor_states(3, seed=3)
    assert torch.equal(old.compute({"states": states})[0], new.compute({"states": states})[0])


def test_the_critic_branch_knows_whose_view_a_pillar_is() -> None:
    # Swap the prey's view of pillar 0 with predator 1's view of it: the old summary pools the same multiset of views
    # and cannot tell the two states apart; the relation branch sees that now the prey, not the predator, is close.
    model = _critic(True)
    states = _critic_states(4, seed=4)
    prey_view = states[:, P * BLOCK + PREY_OBS :].view(4, K, 4)
    prey_view[:, 0, 3] = 0.02
    predator_view = states[:, : P * BLOCK].view(4, P, BLOCK)[:, 1, 30:].reshape(4, K, 4)
    predator_view[:, 0, 3] = 0.8
    states[:, : P * BLOCK].view(4, P, BLOCK)[:, 1, 30:] = predator_view.reshape(4, -1)
    swapped = states.clone()
    a = swapped[:, P * BLOCK + PREY_OBS :].view(4, K, 4)[:, 0].clone()
    b = swapped[:, : P * BLOCK].view(4, P, BLOCK)[:, 1, 30:34].clone()
    swapped[:, P * BLOCK + PREY_OBS :].view(4, K, 4)[:, 0] = b
    swapped[:, : P * BLOCK].view(4, P, BLOCK)[:, 1, 30:34] = a
    value = lambda s: model.compute({"states": s, "raw_states": s})[0]
    assert torch.allclose(value(states), value(swapped), atol=1e-6)       # zero branch: indistinguishable
    _randomize(model)
    assert (value(states) - value(swapped)).abs().amax() > 1e-4


def test_a_crashed_predator_contributes_nothing_but_its_flag() -> None:
    model = _critic(True)
    _randomize(model)
    states = _critic_states(2, seed=5)
    dead = states.clone()
    dead[:, : P * BLOCK].view(2, P, BLOCK)[:, 2, :] = 0.0                       # crashed: zeroed own block
    other = dead.clone()
    other[:, P * BLOCK + 12 + 12 : P * BLOCK + 12 + 15] += 3.0                 # the prey's view of the dead predator
    value = lambda s: model.compute({"states": s, "raw_states": s})[0]
    # The relation tokens mask the dead predator's prey distance; only the old entity path may still see the change.
    tokens_changed = model._obstacle_relation(dead) - model._obstacle_relation(other)
    assert tokens_changed.abs().amax() < 1e-6
    assert (value(states) - value(dead)).abs().amax() > 1e-6


def test_the_relation_branches_learn_in_the_fast_group_and_count_as_pillar_input() -> None:
    actor, critic = _actor(True), _critic(True)
    base, fast = _obstacle_param_groups([actor, critic], 1e-5, 30.0)
    new = [p for m in (actor, critic) for n, p in m.named_parameters() if n.startswith(("obstacle_rel", "obstacle_out_rel"))]
    assert new and all(any(p is q for q in fast["params"]) for p in new)
    # zero_obstacle_branch.py zeroes every key that starts with obstacle_out: the ablation removes these branches too.
    assert all(n.startswith("obstacle_out") for n, _ in actor.named_parameters() if n.startswith("obstacle_out_rel"))
    assert fast["lr"] == pytest.approx(3e-4)
