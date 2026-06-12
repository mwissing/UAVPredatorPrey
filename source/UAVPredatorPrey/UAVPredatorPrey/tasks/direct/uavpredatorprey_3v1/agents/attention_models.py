# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom skrl model instantiators for shared predator attention policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


def _mlp(input_dim: int, layers: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    modules: list[nn.Module] = []
    current_dim = input_dim
    for layer_dim in layers:
        modules.append(nn.Linear(current_dim, layer_dim))
        modules.append(_activation(activation))
        current_dim = layer_dim
    modules.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*modules)


def _infer_predator_layout(num_observations: int, num_actions: int, min_predators: int) -> tuple[int, int] | None:
    if num_actions % 4 != 0:
        return None

    num_predators = num_actions // 4
    if num_predators < min_predators:
        return None

    per_predator_obs = 12 + 6 + 3 * (num_predators - 1)
    if num_observations != num_predators * per_predator_obs:
        return None
    return num_predators, per_predator_obs


class SharedPredatorAttentionGaussianModel(GaussianMixin, Model):
    """Gaussian policy with shared per-predator actor and entity attention.

    The model keeps the current environment contract: the predator team is still
    one skrl agent with a stacked observation and a stacked action. Internally,
    however, each predator is processed by the same actor network. If the input
    shape does not match the predator-team layout, the model falls back to a
    standard flat MLP. This lets the same YAML policy class serve predator and
    prey agents.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -2.0,
        max_log_std: float = 0.0,
        initial_log_std: float = -0.5,
        reduction: str = "sum",
        hidden_size: int = 128,
        attention_size: int = 64,
        fallback_layers: Sequence[int] = (256, 128, 64),
        fallback_activation: str = "elu",
        attention_min_predators: int = 2,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        layout = _infer_predator_layout(self.num_observations, self.num_actions, attention_min_predators)
        self.uses_predator_attention = layout is not None

        if self.uses_predator_attention:
            self.num_predators, self.per_predator_obs = layout
            self.own_encoder = _mlp(12, (hidden_size,), attention_size, fallback_activation)
            self.prey_encoder = _mlp(6, (hidden_size,), attention_size, fallback_activation)
            self.teammate_encoder = _mlp(3, (hidden_size,), attention_size, fallback_activation)
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.actor_head = _mlp(attention_size * 2, (hidden_size,), 4, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        else:
            self.num_predators = 0
            self.per_predator_obs = 0
            self.net = _mlp(self.num_observations, fallback_layers, self.num_actions, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if not self.uses_predator_attention:
            return self.net(states), self.log_std_parameter, {}

        batch_size = states.shape[0]
        predator_obs = states.view(batch_size, self.num_predators, self.per_predator_obs)

        own = predator_obs[:, :, :12]
        prey = predator_obs[:, :, 12:18]
        teammates = predator_obs[:, :, 18:].view(batch_size, self.num_predators, self.num_predators - 1, 3)

        own_emb = self.own_encoder(own.reshape(batch_size * self.num_predators, 12))
        own_emb = own_emb.view(batch_size, self.num_predators, -1)

        prey_emb = self.prey_encoder(prey.reshape(batch_size * self.num_predators, 6))
        prey_emb = prey_emb.view(batch_size, self.num_predators, 1, -1)

        teammate_emb = self.teammate_encoder(teammates.reshape(-1, 3))
        teammate_emb = teammate_emb.view(batch_size, self.num_predators, self.num_predators - 1, -1)

        entities = torch.cat((prey_emb, teammate_emb), dim=2)
        query = self.query(own_emb).unsqueeze(2)
        key = self.key(entities)
        value = self.value(entities)

        weights = torch.softmax((query * key).sum(dim=-1) / math.sqrt(key.shape[-1]), dim=-1)
        context = (weights.unsqueeze(-1) * value).sum(dim=2)

        actions = self.actor_head(torch.cat((own_emb, context), dim=-1))
        actions = actions.reshape(batch_size, self.num_predators * 4)
        log_std = self.log_std_parameter.repeat(self.num_predators)

        return actions, log_std, {}


def shared_predator_attention_gaussian_model(
    observation_space,
    action_space,
    device=None,
    return_source: bool = False,
    **kwargs: Any,
):
    """skrl Runner-compatible model instantiator."""

    if return_source:
        return (
            "SharedPredatorAttentionGaussianModel("
            "shared per-predator actor; attention over prey and teammates; flat MLP fallback)"
        )
    return SharedPredatorAttentionGaussianModel(observation_space, action_space, device=device, **kwargs)


def patch_skrl_runner(Runner) -> None:
    """Register custom model names with skrl's YAML runner."""

    if getattr(Runner, "_uav_attention_models_patched", False):
        return

    original_component = Runner._component

    def _component(self, name: str):
        if name.lower() == "sharedpredatorattentiongaussianmixin":
            return shared_predator_attention_gaussian_model
        return original_component(self, name)

    Runner._component = _component
    Runner._uav_attention_models_patched = True
