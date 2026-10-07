from __future__ import annotations

import math

import torch

from recurrent_mappo import _compute_gae, _explained_variance


def test_gae_bootstraps_only_across_nonterminal_steps() -> None:
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.tensor([[[0.5]], [[0.25]], [[1.0]]])
    dones = torch.tensor([[[False]], [[True]], [[False]]])
    last_values = torch.tensor([[4.0]])

    returns, normalized_advantages = _compute_gae(
        rewards,
        dones,
        values,
        last_values,
        discount_factor=0.9,
        lambda_coefficient=0.8,
    )

    expected_advantages = torch.tensor([[[1.985]], [[1.75]], [[5.6]]])
    expected_returns = expected_advantages + values

    assert torch.allclose(returns, expected_returns)
    assert torch.allclose(normalized_advantages.mean(), torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(normalized_advantages.std(), torch.tensor(1.0), atol=1.0e-6)


def test_gae_terminal_final_step_ignores_last_value() -> None:
    returns, _ = _compute_gae(
        rewards=torch.tensor([[[2.0]], [[3.0]]]),
        dones=torch.tensor([[[True]], [[True]]]),
        values=torch.tensor([[[0.75]], [[1.0]]]),
        last_values=torch.tensor([[100.0]]),
        discount_factor=0.99,
        lambda_coefficient=0.95,
    )

    assert torch.allclose(returns, torch.tensor([[[2.0]], [[3.0]]]))


def test_explained_variance_reference_points() -> None:
    returns = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[6.0]]])

    assert _explained_variance(returns, returns.clone()) == 1.0
    assert abs(_explained_variance(returns, torch.full_like(returns, 3.0))) < 1.0e-6
    # Predicting the negated deviation doubles the error: 1 - Var(2(r - mean)) / Var(r) = -3.
    assert abs(_explained_variance(returns, 6.0 - returns) + 3.0) < 1.0e-5
    # Constant returns carry no variance to explain.
    assert math.isnan(_explained_variance(torch.ones(4, 1, 1), torch.zeros(4, 1, 1)))
