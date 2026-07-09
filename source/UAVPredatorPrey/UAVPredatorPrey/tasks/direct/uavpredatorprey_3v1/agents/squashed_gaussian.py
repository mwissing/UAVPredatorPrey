"""Tanh-squashed Gaussian policy mixin with PPO-compatible log probabilities."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import GaussianMixin


class TanhGaussianMixin(GaussianMixin):
    """Bound Gaussian samples with tanh and correct their action-space density."""

    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        role: str = "",
        inverse_epsilon: float = 1.0e-6,
    ) -> None:
        # Tanh already enforces the action bounds. Post-sample clipping would
        # break the one-to-one action/log-probability contract again.
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )
        self._sg_inverse_epsilon = float(inverse_epsilon)
        self._sg_entropy = None

    @staticmethod
    def _log_abs_det_jacobian(pre_tanh_actions: torch.Tensor) -> torch.Tensor:
        # Stable equivalent of log(1 - tanh(u)^2).
        return 2.0 * (math.log(2.0) - pre_tanh_actions - F.softplus(-2.0 * pre_tanh_actions))

    def _log_prob_per_dimension(self, pre_tanh_actions: torch.Tensor) -> torch.Tensor:
        return self._g_distribution.log_prob(pre_tanh_actions) - self._log_abs_det_jacobian(pre_tanh_actions)

    def _inverse_tanh(self, actions: torch.Tensor) -> torch.Tensor:
        bounded = actions.clamp(
            min=-1.0 + self._sg_inverse_epsilon,
            max=1.0 - self._sg_inverse_epsilon,
        )
        return torch.atanh(bounded)

    def act(self, inputs: Mapping[str, torch.Tensor | Any], role: str = ""):
        latent_mean, log_std, outputs = self.compute(inputs, role)
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        self._g_log_std = log_std
        self._g_num_samples = latent_mean.shape[0]
        self._g_distribution = Normal(latent_mean, log_std.exp())

        sampled_pre_tanh = self._g_distribution.rsample()
        actions = torch.tanh(sampled_pre_tanh)

        evaluated_pre_tanh = inputs.get("pre_tanh_actions")
        if evaluated_pre_tanh is None:
            taken_actions = inputs.get("taken_actions")
            evaluated_pre_tanh = (
                sampled_pre_tanh
                if taken_actions is None
                else self._inverse_tanh(taken_actions)
            )

        log_prob_per_dimension = self._log_prob_per_dimension(evaluated_pre_tanh)
        log_prob = log_prob_per_dimension
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        sampled_log_prob = self._log_prob_per_dimension(sampled_pre_tanh)
        self._sg_entropy = -sampled_log_prob
        outputs["mean_actions"] = torch.tanh(latent_mean)
        outputs["pre_tanh_actions"] = sampled_pre_tanh
        outputs["latent_mean_actions"] = latent_mean
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        if self._sg_entropy is None:
            return torch.tensor(0.0, device=self.device)
        return self._sg_entropy
