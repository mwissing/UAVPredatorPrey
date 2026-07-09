from __future__ import annotations

import torch

from recurrent_mappo import _compute_gae


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
