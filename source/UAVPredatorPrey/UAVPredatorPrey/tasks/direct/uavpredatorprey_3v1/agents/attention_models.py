# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom skrl model instantiators for entity-attention MAPPO policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Mapping

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
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


def _strict_load_allow_missing_prefixes(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    allowed_missing_prefixes: tuple[str, ...],
    *,
    strict: bool = True,
    assign: bool = False,
):
    """Load old feed-forward checkpoints into recurrent variants without hiding real shape mismatches."""

    current_keys = set(module.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    missing = sorted(current_keys - loaded_keys)
    unexpected = sorted(loaded_keys - current_keys)
    if (
        strict
        and missing
        and not unexpected
        and all(any(key.startswith(prefix) for prefix in allowed_missing_prefixes) for key in missing)
    ):
        return nn.Module.load_state_dict(module, state_dict, strict=False, assign=assign)
    return nn.Module.load_state_dict(module, state_dict, strict=strict, assign=assign)


def _infer_sequence_shape(
    inputs: Mapping[str, Any],
    flat_batch_size: int,
    hidden_state_size: int,
    device: torch.device,
) -> tuple[int, int, torch.Tensor | None]:
    rnn_states = inputs.get("rnn", [])
    if rnn_states:
        hidden = rnn_states[0]
        sequence_batch = hidden.shape[1]
        if flat_batch_size % sequence_batch != 0:
            raise RuntimeError(
                f"Cannot infer recurrent sequence shape from states batch {flat_batch_size} "
                f"and hidden batch {sequence_batch}"
            )
        return flat_batch_size // sequence_batch, sequence_batch, hidden

    hidden = torch.zeros(1, flat_batch_size, hidden_state_size, device=device)
    return 1, flat_batch_size, hidden


def _run_gru_sequence(
    gru: nn.GRU,
    features: torch.Tensor,
    hidden: torch.Tensor,
    terminated: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if terminated is None:
        return gru(features, hidden)

    outputs: list[torch.Tensor] = []
    next_hidden = hidden
    dones = terminated.to(dtype=torch.bool).view(features.shape[0], features.shape[1], -1)
    for index in range(features.shape[0]):
        if index > 0:
            active = (~dones[index - 1]).to(features.dtype).view(1, features.shape[1], 1)
            next_hidden = next_hidden * active
        output, next_hidden = gru(features[index : index + 1], next_hidden)
        outputs.append(output)
    return torch.cat(outputs, dim=0), next_hidden


def _infer_predator_layout(num_observations: int, num_actions: int, min_predators: int) -> tuple[int, int, int] | None:
    if num_actions % 4 != 0:
        return None

    num_predators = num_actions // 4
    if num_predators < min_predators:
        return None

    for teammate_dim in (3, 6):
        per_predator_obs = 12 + 6 + teammate_dim * (num_predators - 1)
        if num_observations == num_predators * per_predator_obs:
            return num_predators, per_predator_obs, teammate_dim
    return None


def _infer_centralized_state_layout(num_states: int, max_predators: int = 6) -> tuple[int, int, int, int] | None:
    """Infer centralized state layout: stacked predator observations plus prey observation."""

    for num_predators in range(1, max_predators + 1):
        prey_obs_dim = 12 + 6 * num_predators
        for teammate_dim in (6, 3):
            per_predator_obs = 12 + 6 + teammate_dim * (num_predators - 1)
            predator_obs_dim = num_predators * per_predator_obs
            if num_states == predator_obs_dim + prey_obs_dim:
                return num_predators, per_predator_obs, teammate_dim, prey_obs_dim
    return None


def _infer_prey_layout(num_observations: int, num_actions: int, min_predators: int) -> int | None:
    if num_actions != 4:
        return None
    if num_observations < 18:
        return None
    if (num_observations - 12) % 6 != 0:
        return None
    num_predators = (num_observations - 12) // 6
    if num_predators < min_predators:
        return None
    return num_predators


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
            self.num_predators, self.per_predator_obs, self.teammate_dim = layout
            self.own_encoder = _mlp(12, (hidden_size,), attention_size, fallback_activation)
            self.prey_encoder = _mlp(6, (hidden_size,), attention_size, fallback_activation)
            self.teammate_encoder = _mlp(self.teammate_dim, (hidden_size,), attention_size, fallback_activation)
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.actor_head = _mlp(attention_size * 2, (hidden_size,), 4, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        else:
            self.num_predators = 0
            self.per_predator_obs = 0
            self.teammate_dim = 0
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
        teammates = predator_obs[:, :, 18:].view(
            batch_size,
            self.num_predators,
            self.num_predators - 1,
            self.teammate_dim,
        )

        own_emb = self.own_encoder(own.reshape(batch_size * self.num_predators, 12))
        own_emb = own_emb.view(batch_size, self.num_predators, -1)

        prey_emb = self.prey_encoder(prey.reshape(batch_size * self.num_predators, 6))
        prey_emb = prey_emb.view(batch_size, self.num_predators, 1, -1)

        teammate_emb = self.teammate_encoder(teammates.reshape(-1, self.teammate_dim))
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


class PredatorPreyAttentionGaussianModel(GaussianMixin, Model):
    """Gaussian policy with predator-team attention and prey attention.

    Predator observations keep the shared per-predator actor used by
    SharedPredatorAttentionGaussianModel. Prey observations are parsed as
    own_state(12) + N * predator_info(6), where each predator_info is relative
    position plus relative velocity. The prey own-state embedding queries the
    predator entity set and produces one 4D action.
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
        prey_attention_min_predators: int = 2,
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

        predator_layout = _infer_predator_layout(self.num_observations, self.num_actions, attention_min_predators)
        prey_num_predators = _infer_prey_layout(
            self.num_observations,
            self.num_actions,
            prey_attention_min_predators,
        )

        self.mode = "fallback"
        if predator_layout is not None:
            self.mode = "predator"
            self.num_predators, self.per_predator_obs, self.teammate_dim = predator_layout
            self.own_encoder = _mlp(12, (hidden_size,), attention_size, fallback_activation)
            self.prey_encoder = _mlp(6, (hidden_size,), attention_size, fallback_activation)
            self.teammate_encoder = _mlp(self.teammate_dim, (hidden_size,), attention_size, fallback_activation)
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.actor_head = _mlp(attention_size * 2, (hidden_size,), 4, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        elif prey_num_predators is not None:
            self.mode = "prey"
            self.num_predators = prey_num_predators
            self.per_predator_obs = 0
            self.teammate_dim = 0
            self.own_encoder = _mlp(12, (hidden_size,), attention_size, fallback_activation)
            self.predator_encoder = _mlp(6, (hidden_size,), attention_size, fallback_activation)
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.actor_head = _mlp(attention_size * 2, (hidden_size,), 4, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        else:
            self.num_predators = 0
            self.per_predator_obs = 0
            self.teammate_dim = 0
            self.net = _mlp(self.num_observations, fallback_layers, self.num_actions, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if self.mode == "fallback":
            return self.net(states), self.log_std_parameter, {}

        batch_size = states.shape[0]

        if self.mode == "predator":
            predator_obs = states.view(batch_size, self.num_predators, self.per_predator_obs)
            own = predator_obs[:, :, :12]
            prey = predator_obs[:, :, 12:18]
            teammates = predator_obs[:, :, 18:].view(
                batch_size,
                self.num_predators,
                self.num_predators - 1,
                self.teammate_dim,
            )

            own_emb = self.own_encoder(own.reshape(batch_size * self.num_predators, 12))
            own_emb = own_emb.view(batch_size, self.num_predators, -1)

            prey_emb = self.prey_encoder(prey.reshape(batch_size * self.num_predators, 6))
            prey_emb = prey_emb.view(batch_size, self.num_predators, 1, -1)

            teammate_emb = self.teammate_encoder(teammates.reshape(-1, self.teammate_dim))
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

        own = states[:, :12]
        predators = states[:, 12:].view(batch_size, self.num_predators, 6)

        own_emb = self.own_encoder(own)
        predator_emb = self.predator_encoder(predators.reshape(batch_size * self.num_predators, 6))
        predator_emb = predator_emb.view(batch_size, self.num_predators, -1)

        query = self.query(own_emb).unsqueeze(1)
        key = self.key(predator_emb)
        value = self.value(predator_emb)

        weights = torch.softmax((query * key).sum(dim=-1) / math.sqrt(key.shape[-1]), dim=-1)
        context = (weights.unsqueeze(-1) * value).sum(dim=1)
        actions = self.actor_head(torch.cat((own_emb, context), dim=-1))

        return actions, self.log_std_parameter, {}


def predator_prey_attention_gaussian_model(
    observation_space,
    action_space,
    device=None,
    return_source: bool = False,
    **kwargs: Any,
):
    """skrl Runner-compatible actor with attention for predator team and prey."""

    if return_source:
        return (
            "PredatorPreyAttentionGaussianModel("
            "predator shared attention actor; prey attention over predator entities; flat MLP fallback)"
        )
    return PredatorPreyAttentionGaussianModel(observation_space, action_space, device=device, **kwargs)


class RecurrentPredatorPreyAttentionGaussianModel(GaussianMixin, Model):
    """Entity-attention Gaussian actor with a GRU over attended features."""

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
        rnn_hidden_size: int | None = None,
        rnn_sequence_length: int = 8,
        fallback_layers: Sequence[int] = (256, 128, 64),
        fallback_activation: str = "elu",
        attention_min_predators: int = 2,
        prey_attention_min_predators: int = 2,
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

        predator_layout = _infer_predator_layout(self.num_observations, self.num_actions, attention_min_predators)
        prey_num_predators = _infer_prey_layout(
            self.num_observations,
            self.num_actions,
            prey_attention_min_predators,
        )

        self.mode = "fallback"
        self.rnn_sequence_length = int(rnn_sequence_length)
        self.rnn_hidden_size = int(rnn_hidden_size or (attention_size * 2))

        if predator_layout is not None:
            self.mode = "predator"
            self.num_predators, self.per_predator_obs, self.teammate_dim = predator_layout
            self.feature_size = attention_size * 2
            self.own_encoder = _mlp(12, (hidden_size,), attention_size, fallback_activation)
            self.prey_encoder = _mlp(6, (hidden_size,), attention_size, fallback_activation)
            self.teammate_encoder = _mlp(self.teammate_dim, (hidden_size,), attention_size, fallback_activation)
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.gru = nn.GRU(self.feature_size, self.rnn_hidden_size)
            self.actor_head = _mlp(self.rnn_hidden_size, (hidden_size,), 4, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        elif prey_num_predators is not None:
            self.mode = "prey"
            self.num_predators = prey_num_predators
            self.per_predator_obs = 0
            self.teammate_dim = 0
            self.feature_size = attention_size * 2
            self.own_encoder = _mlp(12, (hidden_size,), attention_size, fallback_activation)
            self.predator_encoder = _mlp(6, (hidden_size,), attention_size, fallback_activation)
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.gru = nn.GRU(self.feature_size, self.rnn_hidden_size)
            self.actor_head = _mlp(self.rnn_hidden_size, (hidden_size,), 4, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std)))
        else:
            self.num_predators = 0
            self.per_predator_obs = 0
            self.teammate_dim = 0
            self.feature_size = self.rnn_hidden_size
            self.feature_net = _mlp(self.num_observations, fallback_layers[:-1], self.rnn_hidden_size, fallback_activation)
            self.gru = nn.GRU(self.rnn_hidden_size, self.rnn_hidden_size)
            self.net = _mlp(self.rnn_hidden_size, fallback_layers[-1:], self.num_actions, fallback_activation)
            self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def get_specification(self) -> Mapping[str, Any]:
        if self.mode == "predator":
            hidden_size = self.num_predators * self.rnn_hidden_size
        else:
            hidden_size = self.rnn_hidden_size
        return {"rnn": {"sizes": [(1, 0, hidden_size)], "sequence_length": self.rnn_sequence_length}}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return _strict_load_allow_missing_prefixes(self, state_dict, ("gru.",), strict=strict, assign=assign)

    def _predator_features(self, states: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        predator_obs = states.view(batch_size, self.num_predators, self.per_predator_obs)
        own = predator_obs[:, :, :12]
        prey = predator_obs[:, :, 12:18]
        teammates = predator_obs[:, :, 18:].view(
            batch_size,
            self.num_predators,
            self.num_predators - 1,
            self.teammate_dim,
        )

        own_emb = self.own_encoder(own.reshape(batch_size * self.num_predators, 12))
        own_emb = own_emb.view(batch_size, self.num_predators, -1)

        prey_emb = self.prey_encoder(prey.reshape(batch_size * self.num_predators, 6))
        prey_emb = prey_emb.view(batch_size, self.num_predators, 1, -1)

        teammate_emb = self.teammate_encoder(teammates.reshape(-1, self.teammate_dim))
        teammate_emb = teammate_emb.view(batch_size, self.num_predators, self.num_predators - 1, -1)

        entities = torch.cat((prey_emb, teammate_emb), dim=2)
        query = self.query(own_emb).unsqueeze(2)
        key = self.key(entities)
        value = self.value(entities)
        weights = torch.softmax((query * key).sum(dim=-1) / math.sqrt(key.shape[-1]), dim=-1)
        context = (weights.unsqueeze(-1) * value).sum(dim=2)
        return torch.cat((own_emb, context), dim=-1)

    def _prey_features(self, states: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        own = states[:, :12]
        predators = states[:, 12:].view(batch_size, self.num_predators, 6)

        own_emb = self.own_encoder(own)
        predator_emb = self.predator_encoder(predators.reshape(batch_size * self.num_predators, 6))
        predator_emb = predator_emb.view(batch_size, self.num_predators, -1)

        query = self.query(own_emb).unsqueeze(1)
        key = self.key(predator_emb)
        value = self.value(predator_emb)
        weights = torch.softmax((query * key).sum(dim=-1) / math.sqrt(key.shape[-1]), dim=-1)
        context = (weights.unsqueeze(-1) * value).sum(dim=1)
        return torch.cat((own_emb, context), dim=-1)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if self.mode == "fallback":
            features = self.feature_net(states)
            sequence_length, sequence_batch, hidden = _infer_sequence_shape(
                inputs, features.shape[0], self.rnn_hidden_size, self.device
            )
            recurrent_input = features.view(sequence_length, sequence_batch, self.rnn_hidden_size)
            recurrent_output, next_hidden = _run_gru_sequence(
                self.gru,
                recurrent_input,
                hidden,
                inputs.get("terminated"),
            )
            actions = self.net(recurrent_output.reshape(features.shape[0], self.rnn_hidden_size))
            return actions, self.log_std_parameter, {"rnn": [next_hidden]}

        if self.mode == "predator":
            features = self._predator_features(states)
            sequence_length, sequence_batch, hidden = _infer_sequence_shape(
                inputs, features.shape[0], self.num_predators * self.rnn_hidden_size, self.device
            )
            recurrent_input = features.view(
                sequence_length,
                sequence_batch,
                self.num_predators,
                self.feature_size,
            )
            recurrent_input = recurrent_input.reshape(
                sequence_length,
                sequence_batch * self.num_predators,
                self.feature_size,
            )
            hidden = hidden.view(1, sequence_batch, self.num_predators, self.rnn_hidden_size)
            hidden = hidden.reshape(1, sequence_batch * self.num_predators, self.rnn_hidden_size)
            terminated = inputs.get("terminated")
            if terminated is not None:
                terminated = terminated.repeat_interleave(self.num_predators, dim=0)
            recurrent_output, next_hidden = _run_gru_sequence(self.gru, recurrent_input, hidden, terminated)
            recurrent_output = recurrent_output.view(
                sequence_length,
                sequence_batch,
                self.num_predators,
                self.rnn_hidden_size,
            )
            recurrent_output = recurrent_output.reshape(
                sequence_length * sequence_batch,
                self.num_predators,
                self.rnn_hidden_size,
            )
            next_hidden = next_hidden.view(1, sequence_batch, self.num_predators, self.rnn_hidden_size)
            next_hidden = next_hidden.reshape(
                1,
                sequence_batch,
                self.num_predators * self.rnn_hidden_size,
            )
            actions = self.actor_head(recurrent_output.reshape(-1, self.rnn_hidden_size))
            actions = actions.view(sequence_length * sequence_batch, self.num_predators * 4)
            return actions, self.log_std_parameter.repeat(self.num_predators), {"rnn": [next_hidden]}

        features = self._prey_features(states)
        sequence_length, sequence_batch, hidden = _infer_sequence_shape(
            inputs, features.shape[0], self.rnn_hidden_size, self.device
        )
        recurrent_input = features.view(sequence_length, sequence_batch, self.feature_size)
        recurrent_output, next_hidden = _run_gru_sequence(
            self.gru,
            recurrent_input,
            hidden,
            inputs.get("terminated"),
        )
        actions = self.actor_head(recurrent_output.reshape(features.shape[0], self.rnn_hidden_size))
        return actions, self.log_std_parameter, {"rnn": [next_hidden]}


def recurrent_predator_prey_attention_gaussian_model(
    observation_space,
    action_space,
    device=None,
    return_source: bool = False,
    **kwargs: Any,
):
    """skrl Runner-compatible recurrent actor with entity attention for predator and prey."""

    if return_source:
        return (
            "RecurrentPredatorPreyAttentionGaussianModel("
            "entity attention -> GRU -> Gaussian action head; compatible flat fallback)"
        )
    return RecurrentPredatorPreyAttentionGaussianModel(observation_space, action_space, device=device, **kwargs)


class EntityAttentionCentralizedValueModel(DeterministicMixin, Model):
    """Centralized value model with entity attention over predators and prey."""

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        clip_actions: bool = False,
        hidden_size: int = 128,
        attention_size: int = 64,
        fallback_layers: Sequence[int] = (256, 128, 64),
        fallback_activation: str = "elu",
        max_predators: int = 6,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        layout = _infer_centralized_state_layout(self.num_observations, max_predators=max_predators)
        self.uses_entity_attention = layout is not None

        if self.uses_entity_attention:
            self.num_predators, self.per_predator_obs, self.teammate_dim, self.prey_obs_dim = layout
            self.predator_encoder = _mlp(
                self.per_predator_obs,
                (hidden_size,),
                attention_size,
                fallback_activation,
            )
            self.prey_encoder = _mlp(
                self.prey_obs_dim,
                (hidden_size,),
                attention_size,
                fallback_activation,
            )
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.value_head = _mlp(attention_size, (hidden_size,), 1, fallback_activation)
        else:
            self.num_predators = 0
            self.per_predator_obs = 0
            self.teammate_dim = 0
            self.prey_obs_dim = 0
            self.net = _mlp(self.num_observations, fallback_layers, 1, fallback_activation)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if not self.uses_entity_attention:
            return self.net(states), {}

        batch_size = states.shape[0]
        predator_state_dim = self.num_predators * self.per_predator_obs
        predator_obs = states[:, :predator_state_dim].view(
            batch_size,
            self.num_predators,
            self.per_predator_obs,
        )
        prey_obs = states[:, predator_state_dim : predator_state_dim + self.prey_obs_dim]

        predator_emb = self.predator_encoder(predator_obs.reshape(-1, self.per_predator_obs))
        predator_emb = predator_emb.view(batch_size, self.num_predators, -1)
        prey_emb = self.prey_encoder(prey_obs).unsqueeze(1)
        entities = torch.cat((predator_emb, prey_emb), dim=1)

        query = self.query(entities)
        key = self.key(entities)
        value = self.value(entities)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(key.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        pooled = attended.mean(dim=1)

        return self.value_head(pooled), {}


def entity_attention_centralized_value_model(
    observation_space,
    action_space,
    device=None,
    return_source: bool = False,
    **kwargs: Any,
):
    """skrl Runner-compatible centralized value instantiator."""

    if return_source:
        return "EntityAttentionCentralizedValueModel(entity encoder critic over centralized state; flat MLP fallback)"
    return EntityAttentionCentralizedValueModel(observation_space, action_space, device=device, **kwargs)


class RecurrentEntityAttentionCentralizedValueModel(DeterministicMixin, Model):
    """Centralized entity-attention critic with temporal GRU memory."""

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        clip_actions: bool = False,
        hidden_size: int = 128,
        attention_size: int = 64,
        rnn_hidden_size: int | None = None,
        rnn_sequence_length: int = 8,
        fallback_layers: Sequence[int] = (256, 128, 64),
        fallback_activation: str = "elu",
        max_predators: int = 6,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        layout = _infer_centralized_state_layout(self.num_observations, max_predators=max_predators)
        self.uses_entity_attention = layout is not None
        self.rnn_sequence_length = int(rnn_sequence_length)
        self.rnn_hidden_size = int(rnn_hidden_size or attention_size)

        if self.uses_entity_attention:
            self.num_predators, self.per_predator_obs, self.teammate_dim, self.prey_obs_dim = layout
            self.feature_size = attention_size
            self.predator_encoder = _mlp(
                self.per_predator_obs,
                (hidden_size,),
                attention_size,
                fallback_activation,
            )
            self.prey_encoder = _mlp(
                self.prey_obs_dim,
                (hidden_size,),
                attention_size,
                fallback_activation,
            )
            self.query = nn.Linear(attention_size, attention_size)
            self.key = nn.Linear(attention_size, attention_size)
            self.value = nn.Linear(attention_size, attention_size)
            self.gru = nn.GRU(self.feature_size, self.rnn_hidden_size)
            self.value_head = _mlp(self.rnn_hidden_size, (hidden_size,), 1, fallback_activation)
        else:
            self.num_predators = 0
            self.per_predator_obs = 0
            self.teammate_dim = 0
            self.prey_obs_dim = 0
            self.feature_size = self.rnn_hidden_size
            self.feature_net = _mlp(self.num_observations, fallback_layers[:-1], self.rnn_hidden_size, fallback_activation)
            self.gru = nn.GRU(self.rnn_hidden_size, self.rnn_hidden_size)
            self.net = _mlp(self.rnn_hidden_size, fallback_layers[-1:], 1, fallback_activation)

    def get_specification(self) -> Mapping[str, Any]:
        return {"rnn": {"sizes": [(1, 0, self.rnn_hidden_size)], "sequence_length": self.rnn_sequence_length}}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return _strict_load_allow_missing_prefixes(self, state_dict, ("gru.",), strict=strict, assign=assign)

    def _attention_feature(self, states: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        predator_state_dim = self.num_predators * self.per_predator_obs
        predator_obs = states[:, :predator_state_dim].view(
            batch_size,
            self.num_predators,
            self.per_predator_obs,
        )
        prey_obs = states[:, predator_state_dim : predator_state_dim + self.prey_obs_dim]

        predator_emb = self.predator_encoder(predator_obs.reshape(-1, self.per_predator_obs))
        predator_emb = predator_emb.view(batch_size, self.num_predators, -1)
        prey_emb = self.prey_encoder(prey_obs).unsqueeze(1)
        entities = torch.cat((predator_emb, prey_emb), dim=1)

        query = self.query(entities)
        key = self.key(entities)
        value = self.value(entities)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(key.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        return attended.mean(dim=1)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if self.uses_entity_attention:
            features = self._attention_feature(states)
            head = self.value_head
        else:
            features = self.feature_net(states)
            head = self.net

        sequence_length, sequence_batch, hidden = _infer_sequence_shape(
            inputs, features.shape[0], self.rnn_hidden_size, self.device
        )
        recurrent_input = features.view(sequence_length, sequence_batch, self.feature_size)
        recurrent_output, next_hidden = _run_gru_sequence(
            self.gru,
            recurrent_input,
            hidden,
            inputs.get("terminated"),
        )
        values = head(recurrent_output.reshape(features.shape[0], self.rnn_hidden_size))
        return values, {"rnn": [next_hidden]}


def recurrent_entity_attention_centralized_value_model(
    observation_space,
    action_space,
    device=None,
    return_source: bool = False,
    **kwargs: Any,
):
    """skrl Runner-compatible recurrent centralized value model."""

    if return_source:
        return (
            "RecurrentEntityAttentionCentralizedValueModel("
            "centralized entity attention -> GRU -> value head)"
        )
    return RecurrentEntityAttentionCentralizedValueModel(observation_space, action_space, device=device, **kwargs)


def patch_skrl_runner(Runner) -> None:
    """Register custom model names with skrl's YAML runner."""

    if getattr(Runner, "_uav_attention_models_patched", False):
        return

    original_component = Runner._component

    def _component(self, name: str):
        component_name = name.lower()
        if component_name == "mappo":
            from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.agents.recurrent_mappo import RecurrentMAPPO

            return RecurrentMAPPO
        if component_name == "sharedpredatorattentiongaussianmixin":
            return shared_predator_attention_gaussian_model
        if component_name == "predatorpreyattentiongaussianmixin":
            return predator_prey_attention_gaussian_model
        if component_name == "recurrentpredatorpreyattentiongaussianmixin":
            return recurrent_predator_prey_attention_gaussian_model
        if component_name == "entityattentioncentralizedvaluemixin":
            return entity_attention_centralized_value_model
        if component_name == "recurrententityattentioncentralizedvaluemixin":
            return recurrent_entity_attention_centralized_value_model
        return original_component(self, name)

    Runner._component = _component
    Runner._uav_attention_models_patched = True
