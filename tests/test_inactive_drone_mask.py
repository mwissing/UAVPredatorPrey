from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import gymnasium
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
PACKAGE_NAME = "_inactive_mask_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

# What the env writes for a crashed drone: 2 x arena_radius straight ahead, zero relative velocity.
PLACEHOLDER = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _actor(observations: int, actions: int, **flags):
    torch.manual_seed(1)
    return attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(observations), _box(actions), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24, **flags
    )


def _mean(model, states: torch.Tensor, raw: bool) -> torch.Tensor:
    inputs = {"states": states, "raw_states": states} if raw else {"states": states}
    return model.compute(inputs)[0]


def test_mask_is_off_by_default_and_a_no_op_without_crashed_drones() -> None:
    generator = torch.Generator().manual_seed(0)
    for observations, actions, flag in ((30, 4, "prey_mask_inactive_predators"), (90, 12, "predator_mask_inactive_teammates")):
        states = torch.randn(5, observations, generator=generator)
        default, masked = _actor(observations, actions), _actor(observations, actions, **{flag: True})
        masked.load_state_dict(default.state_dict())
        assert not default.mask_inactive and masked.mask_inactive
        assert torch.equal(_mean(default, states, raw=False), _mean(default, states, raw=True))
        assert torch.equal(_mean(default, states, raw=False), _mean(masked, states, raw=True))


def test_masked_prey_ignores_a_crashed_predator() -> None:
    """With predator 2 crashed, the masked 3-predator prey acts like a 2-predator prey with the same weights."""
    generator = torch.Generator().manual_seed(1)
    three = _actor(30, 4, prey_mask_inactive_predators=True)
    two = _actor(24, 4)
    two.load_state_dict(three.state_dict())
    own, predators = torch.randn(5, 12, generator=generator), torch.randn(5, 2, 6, generator=generator)
    with_crash = torch.cat((own, predators.reshape(5, 12), PLACEHOLDER.repeat(5, 1)), dim=1)
    without = torch.cat((own, predators.reshape(5, 12)), dim=1)
    assert torch.allclose(_mean(three, with_crash, raw=True), _mean(two, without, raw=False), atol=1e-6)
    # Unmasked, the placeholder acts like a real predator 10 m ahead and changes the output.
    unmasked = _actor(30, 4)
    unmasked.load_state_dict(three.state_dict())
    assert not torch.allclose(_mean(unmasked, with_crash, raw=True), _mean(two, without, raw=False), atol=1e-6)


def test_masked_predators_ignore_a_crashed_teammate() -> None:
    """With predator 2 crashed, drones 0 and 1 of the masked team act like a 2-predator team."""
    generator = torch.Generator().manual_seed(2)
    three = _actor(90, 12, predator_mask_inactive_teammates=True)
    two = _actor(48, 8)
    two.load_state_dict(three.state_dict())
    own, prey, mutual = (torch.randn(5, 2, 12, generator=generator), torch.randn(5, 2, 6, generator=generator),
                         torch.randn(5, 2, 6, generator=generator))
    # Team of 3: drone i sees its live teammate (the other of 0/1) first, then the crashed drone 2.
    live_blocks = torch.cat((own, prey, mutual, PLACEHOLDER.repeat(5, 2, 1)), dim=2)  # (5, 2, 30)
    crashed_block = torch.cat((torch.zeros(5, 1, 18), torch.randn(5, 1, 12, generator=generator)), dim=2)
    with_crash = torch.cat((live_blocks, crashed_block), dim=1).reshape(5, 90)
    without = torch.cat((own, prey, mutual), dim=2).reshape(5, 48)
    assert torch.allclose(_mean(three, with_crash, raw=True)[:, :8], _mean(two, without, raw=False), atol=1e-6)
