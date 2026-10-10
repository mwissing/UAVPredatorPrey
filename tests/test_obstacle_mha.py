from __future__ import annotations

import importlib
import math
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
PACKAGE_NAME = "_obstacle_mha_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

K, OBSTACLE_DIM = 4, 4
PRESENT, PARKED = 0.5, 9.9  # raw surface distance / feature scale: in the arena vs parked 50 m away


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _actor(observations: int, actions: int, heads: int = 0):
    torch.manual_seed(1)
    return attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(observations), _box(actions), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=K, obstacle_attention_heads=heads,
    )


def _states(batch: int, blocks: int, distances: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random observations with K pillars per block whose raw surface distances are `distances` (blocks, K)."""
    states = torch.randn(batch, blocks, 30 + K * OBSTACLE_DIM, generator=generator)
    states[:, :, 30:].view(batch, blocks, K, OBSTACLE_DIM)[..., 3] = distances
    return states.reshape(batch, -1)


def _cases():
    return ((90 + 3 * K * OBSTACLE_DIM, 12, 3), (30 + K * OBSTACLE_DIM, 4, 1))  # predator team, prey


def _with_trained_mha(model) -> None:
    torch.manual_seed(9)
    torch.nn.init.normal_(model.obstacle_out.weight, std=0.5)
    torch.nn.init.normal_(model.obstacle_mha.out.weight, std=0.5)
    torch.nn.init.normal_(model.obstacle_mha.out.bias, std=0.5)


def test_adding_the_block_leaves_a_trained_network_unchanged() -> None:
    generator = torch.Generator().manual_seed(0)
    for observations, actions, blocks in _cases():
        old = _actor(observations, actions)
        torch.nn.init.normal_(old.obstacle_out.weight, std=0.5)  # a trained single-head branch
        new = _actor(observations, actions, heads=4)
        new.load_state_dict(old.state_dict())
        states = _states(5, blocks, torch.full((blocks, K), PRESENT), generator)
        for inputs in ({"states": states}, {"states": states, "raw_states": states}):
            assert torch.equal(old.compute(inputs)[0], new.compute(inputs)[0])


def test_block_ignores_absent_pillars_and_adds_nothing_without_any() -> None:
    generator = torch.Generator().manual_seed(1)
    for observations, actions, blocks in _cases():
        model = _actor(observations, actions, heads=4)
        _with_trained_mha(model)
        distances = torch.tensor([PRESENT, PRESENT, PARKED, PARKED]).repeat(blocks, 1)
        first = _states(4, blocks, distances, generator)
        second = first.clone().view(4, blocks, -1)
        second[:, :, 30:].view(4, blocks, K, OBSTACLE_DIM)[:, :, 2:, :3] += 3.0
        second = second.reshape(4, -1)
        out = lambda s: model.compute({"states": s, "raw_states": s})[0]
        assert torch.equal(out(first), out(second))
        # Without any present pillar the whole obstacle term is zero: same as zeroed output projections.
        parked = _states(4, blocks, torch.full((blocks, K), PARKED), generator)
        zeroed = _actor(observations, actions, heads=4)
        zeroed.load_state_dict(model.state_dict())
        with torch.no_grad():
            for module in (zeroed.obstacle_out, zeroed.obstacle_mha.out):
                module.weight.zero_()
                module.bias.zero_()
        assert torch.equal(out(parked), zeroed.compute({"states": parked, "raw_states": parked})[0])


def test_attention_is_soft_at_initialization() -> None:
    generator = torch.Generator().manual_seed(2)
    block = attention_models.ObstacleMultiHeadAttention(OBSTACLE_DIM, 32, 16, 4, "elu")
    own = torch.randn(256, 16, generator=generator)
    tokens = block.token_norm(block.encoder(torch.randn(256 * K, OBSTACLE_DIM, generator=generator))).view(256, K, -1)
    weights = block.weights(own, tokens, None)                              # (B, heads, K)
    entropy = -(weights * weights.clamp(min=1e-12).log()).sum(-1) / math.log(K)
    assert entropy.mean() > 0.5
    assert weights.max(dim=-1).values.mean() < 0.8  # far from the one-hot (1.0) seen in the single-head branch


def test_gradient_reaches_queries_keys_and_temperature() -> None:
    generator = torch.Generator().manual_seed(3)
    model = _actor(30 + K * OBSTACLE_DIM, 4, heads=4)
    _with_trained_mha(model)
    states = _states(8, 1, torch.full((1, K), PRESENT), generator)
    model.compute({"states": states, "raw_states": states})[0].sum().backward()
    for parameter in (model.obstacle_mha.query.weight, model.obstacle_mha.key.weight, model.obstacle_mha.log_temperature):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_temperature_is_capped() -> None:
    block = attention_models.ObstacleMultiHeadAttention(OBSTACLE_DIM, 32, 16, 4, "elu")
    with torch.no_grad():
        block.log_temperature.fill_(10.0)  # e^10 if unclamped
    own, tokens = torch.randn(2, 16), torch.randn(2, K, 16)
    with torch.no_grad():
        query = torch.nn.functional.normalize(block.query(block.query_norm(own)).view(2, 4, 1, 4), dim=-1)
        key = torch.nn.functional.normalize(block.key(tokens).view(2, K, 4, 4).transpose(1, 2), dim=-1)
        expected = torch.softmax(block.MAX_TEMPERATURE * (query * key).sum(-1), dim=-1)
    assert torch.allclose(block.weights(own, tokens, None), expected, atol=1e-6)
