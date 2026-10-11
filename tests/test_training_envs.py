"""A role can be trained on a subset of envs (mixed ghost rollouts: the predator trains only where it is real)."""

from __future__ import annotations

import pytest
import torch

from recurrent_mappo import RecurrentMAPPO, _compute_gae, _env_minibatches
from skrl.resources.preprocessors.torch import RunningStandardScaler

T, N = 6, 8
REAL = torch.tensor([False, False, False, True, True, True, True, True])  # the first 3 envs have ghosts


def _rollout(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randn(T, N, 1, generator=generator)
    rewards[:, ~REAL] += 50.0  # ghost envs: a very different return scale that must not leak into the stats
    dones = torch.zeros(T, N, 1, dtype=torch.bool)
    dones[3, 1] = dones[4, 5] = True
    return {"rewards": rewards, "dones": dones, "values": torch.randn(T, N, 1, generator=generator),
            "last_values": torch.randn(N, 1, generator=generator)}


def test_gae_without_envs_is_unchanged() -> None:
    r = _rollout()
    returns, advantages = _compute_gae(**r, discount_factor=0.99, lambda_coefficient=0.95)
    raw = returns - r["values"]
    assert torch.allclose(advantages, (raw - raw.mean()) / (raw.std() + 1e-8))


def test_gae_normalizes_with_the_training_envs_only() -> None:
    r = _rollout()
    envs = REAL.nonzero().reshape(-1)
    returns_all, _ = _compute_gae(**r, discount_factor=0.99, lambda_coefficient=0.95)
    returns, advantages = _compute_gae(**r, discount_factor=0.99, lambda_coefficient=0.95, envs=envs)
    assert torch.equal(returns, returns_all)  # returns do not depend on the selection
    selected = advantages[:, envs]
    assert abs(float(selected.mean())) < 1e-5
    assert abs(float(selected.std()) - 1.0) < 1e-4


def test_minibatches_cover_exactly_the_training_envs() -> None:
    envs = REAL.nonzero().reshape(-1)
    torch.manual_seed(0)
    batches = _env_minibatches(N, 2, envs, "cpu")
    assert len(batches) == 2
    assert sorted(torch.cat(batches).tolist()) == envs.tolist()  # each training env once, no ghost env
    assert sorted(torch.cat(_env_minibatches(N, 2, None, "cpu")).tolist()) == list(range(N))


def test_value_scaler_statistics_come_from_the_selected_envs() -> None:
    values = _rollout()["rewards"]
    envs = REAL.nonzero().reshape(-1)
    masked, reference = RunningStandardScaler(1, device="cpu"), RunningStandardScaler(1, device="cpu")
    masked(values[:, envs], train=True)
    normalized = masked(values)  # train=False: normalizes every env without updating the statistics
    reference(values[:, envs], train=True)
    assert torch.equal(masked.running_mean, reference.running_mean)
    assert torch.equal(masked.running_variance, reference.running_variance)
    assert torch.allclose(normalized, reference(values))


def _agent(rnn: bool = True) -> RecurrentMAPPO:
    agent = RecurrentMAPPO.__new__(RecurrentMAPPO)
    agent.possible_agents = ["predator", "prey"]
    agent.device = "cpu"
    agent._rnn = rnn
    agent._training_envs = {}
    agent.memories = {uid: type("Memory", (), {"num_envs": N})() for uid in agent.possible_agents}
    return agent


def test_set_training_envs_stores_indices_and_validates() -> None:
    agent = _agent()
    agent.set_training_envs("predator", REAL)
    assert agent._training_envs["predator"].tolist() == [3, 4, 5, 6, 7]
    assert "prey" not in agent._training_envs
    with pytest.raises(ValueError):
        agent.set_training_envs("ghost", REAL)
    with pytest.raises(ValueError):
        agent.set_training_envs("predator", REAL[:-1])
    with pytest.raises(ValueError):
        agent.set_training_envs("predator", torch.zeros(N, dtype=torch.bool))
    with pytest.raises(NotImplementedError):
        _agent(rnn=False).set_training_envs("predator", REAL)
