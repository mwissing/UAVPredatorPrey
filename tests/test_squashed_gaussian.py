from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.models.torch import Model
from squashed_gaussian import TanhGaussianMixin


class TinySquashedPolicy(TanhGaussianMixin, Model):
    def __init__(self) -> None:
        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        Model.__init__(self, observation_space, action_space, device="cpu")
        TanhGaussianMixin.__init__(self, min_log_std=-5.0, max_log_std=2.0)
        self.mean = nn.Parameter(torch.tensor([2.0, -3.0]))
        self.log_std = nn.Parameter(torch.tensor([-0.5, -0.25]))

    def compute(self, inputs, role=""):
        batch_size = inputs["states"].shape[0]
        return self.mean.expand(batch_size, -1), self.log_std, {}


def test_actions_and_deterministic_means_are_bounded() -> None:
    policy = TinySquashedPolicy()
    actions, _, outputs = policy.act({"states": torch.zeros(32, 2)})

    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.allclose(outputs["mean_actions"][0], torch.tanh(policy.mean))


def test_corrected_log_probability_matches_change_of_variables() -> None:
    policy = TinySquashedPolicy()
    states = torch.zeros(4, 2)
    actions, log_prob, outputs = policy.act({"states": states})
    pre_tanh = outputs["pre_tanh_actions"]

    base = torch.distributions.Normal(policy.mean.expand_as(pre_tanh), policy.log_std.exp())
    correction = 2.0 * (
        torch.log(torch.tensor(2.0))
        - pre_tanh
        - torch.nn.functional.softplus(-2.0 * pre_tanh)
    )
    expected = (base.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)

    assert torch.allclose(log_prob, expected, atol=1.0e-6)
    assert torch.allclose(actions, torch.tanh(pre_tanh))


def test_stored_pre_tanh_action_reproduces_old_log_probability() -> None:
    policy = TinySquashedPolicy()
    states = torch.zeros(8, 2)
    actions, old_log_prob, outputs = policy.act({"states": states})

    _, replayed_log_prob, _ = policy.act(
        {
            "states": states,
            "taken_actions": actions,
            "pre_tanh_actions": outputs["pre_tanh_actions"],
        }
    )

    assert torch.allclose(old_log_prob, replayed_log_prob, atol=1.0e-6)


def test_squashed_entropy_estimate_is_finite_and_differentiable() -> None:
    policy = TinySquashedPolicy()
    policy.act({"states": torch.zeros(16, 2)})
    entropy = policy.get_entropy().mean()

    assert torch.isfinite(entropy)
    entropy.backward()
    assert policy.log_std.grad is not None
    assert torch.isfinite(policy.log_std.grad).all()
