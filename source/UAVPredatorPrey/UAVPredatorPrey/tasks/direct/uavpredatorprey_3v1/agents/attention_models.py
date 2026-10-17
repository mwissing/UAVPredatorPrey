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

from skrl.models.torch import DeterministicMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .squashed_gaussian import TanhGaussianMixin


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


def _infer_centralized_state_layout(
    num_states: int, max_predators: int = 6, obstacle_block: int = 0
) -> tuple[int, int, int, int] | None:
    """Infer centralized state layout: stacked predator observations plus prey observation.

    `obstacle_block` obstacle features are appended to every per-drone block
    (each predator block and the prey block); the returned sizes exclude them.
    """

    for num_predators in range(1, max_predators + 1):
        prey_obs_dim = 12 + 6 * num_predators
        for teammate_dim in (6, 3):
            per_predator_obs = 12 + 6 + teammate_dim * (num_predators - 1)
            predator_obs_dim = num_predators * (per_predator_obs + obstacle_block)
            if num_states == predator_obs_dim + prey_obs_dim + obstacle_block:
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


def _aligned_attention_weights(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute broadcast-aligned attention weights without FP16 score overflow."""

    # Trained query/key activations can be individually finite in FP16 while
    # their elementwise product exceeds FP16's range. Disable autocast for the
    # product, reduction and softmax; callers cast the normalized weights back
    # to the value dtype so the remaining attention path stays under AMP.
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = (query.float() * key.float()).sum(dim=-1) / math.sqrt(key.shape[-1])
        return torch.softmax(scores, dim=-1)


# Absent pillars are parked 50 m away (obstacles.PARKED_DISTANCE), so their raw surface-distance feature
# (distance / obstacle_feature_scale, the 4th feature) is >= 9.9, while pillars inside the arena stay below 2.
# A raw feature below this limit marks a present pillar. The check uses raw observations: the learned
# observation normalizer must not decide whether an obstacle exists.
OBSTACLE_PRESENCE_LIMIT = 5.0


def _obstacle_presence(raw_obstacles: torch.Tensor) -> torch.Tensor:
    """(..., K) mask of present pillars from raw (..., K, obstacle_dim) features."""
    return raw_obstacles[..., 3] < OBSTACLE_PRESENCE_LIMIT


def _segment_axis_distance(axis_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """(..., 1) horizontal distance of pillar axes (..., 2) to the segment origin -> target (..., 2), in input units.

    The observer sits at the origin, so a value below the pillar radius means the pillar blocks the line of sight.
    """
    t = (axis_xy * target_xy).sum(-1) / (target_xy * target_xy).sum(-1).clamp(min=1e-6)
    return (axis_xy - t.clamp(0.0, 1.0).unsqueeze(-1) * target_xy).norm(dim=-1, keepdim=True)


def _masked_max_mean(tokens: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool (..., K, F) tokens over present entries (..., K): concatenated max and mean (..., 2F), and the count."""
    mask = present.unsqueeze(-1)
    maxed = torch.where(mask, tokens, torch.full_like(tokens, -1.0e4)).amax(dim=-2)
    count = present.sum(dim=-1, keepdim=True)
    mean = (tokens * mask.to(tokens.dtype)).sum(dim=-2) / count.clamp(min=1).to(tokens.dtype)
    return torch.cat((maxed, mean), dim=-1), count


def _raw_states(observation_space, inputs: Mapping[str, Any], batch_size: int) -> torch.Tensor | None:
    """Raw (unnormalized) states passed by the agent as inputs["raw_states"], or None (no presence mask)."""
    raw = inputs.get("raw_states")
    if raw is None:
        return None
    raw = unflatten_tensorized_space(observation_space, raw)
    return raw.reshape(batch_size, -1)


def _masked_attention_weights(query: torch.Tensor, key: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Attention weights over present entities only; rows without any present entity get all-zero weights."""

    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = (query.float() * key.float()).sum(dim=-1) / math.sqrt(key.shape[-1])
        # A large finite fill instead of -inf keeps the softmax (and its gradient) finite when a row has no
        # present entity; those rows are zeroed afterwards.
        weights = torch.softmax(scores.masked_fill(~present, -1.0e9), dim=-1)
        return torch.where(present.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))


class ObstacleMultiHeadAttention(nn.Module):
    """Multi-head cross-attention from a drone's own embedding to its pillar tokens.

    The single-head obstacle attention saturated to one-hot weights and picked pillars unrelated to danger
    (v6 attention probe). This block is built to avoid that: several heads can watch several pillars, the pillar
    tokens are layer-normalized, and the scores are cosine similarities times a learnable temperature (start 5,
    cap 30). The scores are therefore bounded: the attention starts soft and becomes sharp only if training
    raises the temperature, not because embeddings grow. Absent pillars are masked. The output projection starts at zero, so adding the block to a trained network
    leaves it unchanged until training moves it.
    """

    MAX_TEMPERATURE = 30.0

    def __init__(self, obstacle_dim: int, hidden_size: int, attention_size: int, heads: int, activation: str) -> None:
        super().__init__()
        if attention_size % heads:
            raise ValueError(f"attention_size {attention_size} is not divisible by {heads} heads")
        self.heads = heads
        self.head_dim = attention_size // heads
        self.encoder = _mlp(obstacle_dim, (hidden_size,), attention_size, activation)
        self.token_norm = nn.LayerNorm(attention_size)
        self.query_norm = nn.LayerNorm(attention_size)
        self.query = nn.Linear(attention_size, attention_size)
        self.key = nn.Linear(attention_size, attention_size)
        self.value = nn.Linear(attention_size, attention_size)
        # Start soft (normalized entropy about 0.8-0.9 at 32 dims per head) so gradients reach every pillar;
        # training can sharpen the attention up to MAX_TEMPERATURE.
        self.log_temperature = nn.Parameter(torch.full((heads,), math.log(5.0)))
        self.out = nn.Linear(attention_size, attention_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def weights(self, own_emb: torch.Tensor, tokens: torch.Tensor, present: torch.Tensor | None) -> torch.Tensor:
        """(B, heads, K) attention weights; rows without any present pillar are all zero."""
        batch, num_obstacles, _ = tokens.shape
        query = self.query(self.query_norm(own_emb)).view(batch, self.heads, 1, self.head_dim)
        key = self.key(tokens).view(batch, num_obstacles, self.heads, self.head_dim).transpose(1, 2)
        with torch.autocast(device_type=own_emb.device.type, enabled=False):
            cosine = (nn.functional.normalize(query.float(), dim=-1) * nn.functional.normalize(key.float(), dim=-1)).sum(-1)
            temperature = self.log_temperature.float().clamp(0.0, math.log(self.MAX_TEMPERATURE)).exp().view(1, -1, 1)
            scores = temperature * cosine
            if present is None:
                return torch.softmax(scores, dim=-1)
            weights = torch.softmax(scores.masked_fill(~present[:, None, :], -1.0e9), dim=-1)
            return torch.where(present.any(dim=-1)[:, None, None], weights, torch.zeros_like(weights))

    def forward(self, own_emb: torch.Tensor, obstacles: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        """(B, A) obstacle context from own (B, A) embeddings and (B, K, obstacle_dim) pillar features."""
        batch, num_obstacles, obstacle_dim = obstacles.shape
        tokens = self.token_norm(self.encoder(obstacles.reshape(batch * num_obstacles, obstacle_dim)))
        tokens = tokens.view(batch, num_obstacles, -1)
        weights = self.weights(own_emb, tokens, present)
        value = self.value(tokens).view(batch, num_obstacles, self.heads, self.head_dim).transpose(1, 2)
        context = (weights.unsqueeze(-1).to(value.dtype) * value).sum(dim=2).reshape(batch, -1)
        out = self.out(context)
        if present is not None:
            out = torch.where(present.any(dim=-1, keepdim=True), out, torch.zeros_like(out))
        return out


def _pairwise_attention_weights(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute all-pairs attention weights without autocast narrowing the matmul."""

    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(key.shape[-1])
        return torch.softmax(scores, dim=-1)


class SharedPredatorAttentionGaussianModel(TanhGaussianMixin, Model):
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
        TanhGaussianMixin.__init__(
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

        weights = _aligned_attention_weights(query, key).to(dtype=value.dtype)
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
            "shared per-predator actor; attention; tanh-squashed Gaussian; flat MLP fallback)"
        )
    return SharedPredatorAttentionGaussianModel(observation_space, action_space, device=device, **kwargs)


class PredatorPreyAttentionGaussianModel(TanhGaussianMixin, Model):
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
        TanhGaussianMixin.__init__(
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

            weights = _aligned_attention_weights(query, key).to(dtype=value.dtype)
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

        weights = _aligned_attention_weights(query, key).to(dtype=value.dtype)
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
            "predator/prey entity attention; tanh-squashed Gaussian; flat MLP fallback)"
        )
    return PredatorPreyAttentionGaussianModel(observation_space, action_space, device=device, **kwargs)


class RecurrentPredatorPreyAttentionGaussianModel(TanhGaussianMixin, Model):
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
        predator_min_log_std: float | None = None,
        prey_min_log_std: float | None = None,
        num_obstacles: int = 0,
        obstacle_dim: int = 4,
        prey_mask_inactive_predators: bool = False,
        predator_mask_inactive_teammates: bool = False,
        inactive_drone_distance: float = 10.0,
        obstacle_attention_heads: int = 0,
        predator_role_embedding: bool = False,
        predator_obstacle_relation: bool = False,
        prey_obstacle_relation: bool = False,
        obstacle_feature_scale: float = 5.0,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        TanhGaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        # Obstacle features (num_obstacles x obstacle_dim) are appended to every
        # per-drone observation block; the remaining layout is unchanged.
        self.num_obstacles = int(num_obstacles)
        self.obstacle_dim = int(obstacle_dim)
        obstacle_block = self.num_obstacles * self.obstacle_dim
        team_size = self.num_actions // 4 if self.num_actions % 4 == 0 else 0
        predator_layout = _infer_predator_layout(
            self.num_observations - team_size * obstacle_block, self.num_actions, attention_min_predators
        )
        prey_num_predators = _infer_prey_layout(
            self.num_observations - obstacle_block,
            self.num_actions,
            prey_attention_min_predators,
        )
        # Optional role-specific floor: widen one role's exploration while the
        # other role keeps exactly the action noise it was trained with.
        role_min_log_std = None
        if predator_layout is not None:
            role_min_log_std = predator_min_log_std
        elif prey_num_predators is not None:
            role_min_log_std = prey_min_log_std
        if role_min_log_std is not None:
            self._g_log_std_min = float(role_min_log_std)

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

        # Optional mask for crashed drones (off by default, so existing policies stay bit-exact). The env
        # replaces a crashed drone by a placeholder at (inactive_drone_distance, 0, 0) in the observer's body
        # frame (2 x arena_radius) with zero relative velocity. A policy that never met that placeholder in
        # training may react to it as to a real drone straight ahead; masked, it simply ignores the drone.
        self.mask_inactive = (self.mode == "prey" and bool(prey_mask_inactive_predators)) or (
            self.mode == "predator" and bool(predator_mask_inactive_teammates)
        )
        self.inactive_drone_distance = float(inactive_drone_distance)

        # Obstacle branch, created last so that the existing modules keep their
        # initialization. Its output projection starts at zero: a policy loaded
        # from an obstacle-free checkpoint initially acts exactly as before and
        # learns to use the obstacles from there.
        self.per_predator_block = self.per_predator_obs + obstacle_block
        if self.num_obstacles > 0 and self.mode in ("predator", "prey"):
            self.obstacle_encoder = _mlp(self.obstacle_dim, (hidden_size,), attention_size, fallback_activation)
            self.obstacle_query = nn.Linear(attention_size, attention_size)
            self.obstacle_key = nn.Linear(attention_size, attention_size)
            self.obstacle_value = nn.Linear(attention_size, attention_size)
            self.obstacle_out = nn.Linear(attention_size, attention_size)
            nn.init.zeros_(self.obstacle_out.weight)
            nn.init.zeros_(self.obstacle_out.bias)
            # Optional multi-head path (0 = off), created after the existing modules so their initialization
            # is unchanged; its zero output projection keeps a loaded network exactly as it was.
            if int(obstacle_attention_heads) > 0:
                self.obstacle_mha = ObstacleMultiHeadAttention(
                    self.obstacle_dim, hidden_size, attention_size, int(obstacle_attention_heads), fallback_activation
                )

        # Optional role vector per predator index (off by default). The shared network is permutation-equivariant:
        # drones in mirrored situations act alike, so the team cannot settle on conventions such as "drone 0 flanks
        # left". A learned vector per index, added to the own-state embedding, breaks that symmetry. Only its
        # deviation from the team mean is used, so it cannot act as a shared bias in the fast learning-rate group.
        # Zero-initialized and created last, it leaves a loaded network exactly as it was.
        if self.mode == "predator" and predator_role_embedding:
            self.role_embedding = nn.Parameter(torch.zeros(self.num_predators, attention_size))

        # Optional pillar-prey relation branch for the predators (off by default). The single pillar attention is queried
        # by the drone's own state, so it cannot pick the pillar next to the prey or the one blocking the line of sight
        # (Stage 8 review, 2026-10-17). Here every present pillar becomes a token together with the prey: pillar offset
        # and surface distance, prey offset, prey minus pillar, and the pillar axis' distance to the drone -> prey
        # segment, all from the raw body-frame features (meters / obstacle_feature_scale). A shared encoder and masked
        # max + mean pooling over the pillars replace the one-hot selection. The zero-initialized output, created last,
        # leaves a loaded network exactly as it was.
        self.obstacle_feature_scale = float(obstacle_feature_scale)
        # The prey's counterpart (prey_obstacle_relation): its pillar attention is equally degenerate (it attends the pillar
        # it hugs in under 1% of those steps), so the same tokens are built with the nearest live predator in the role of
        # the target: pillar offset and distance, predator offset, predator minus pillar, and the pillar axis' distance
        # to the prey -> predator segment (a pillar that hides the prey from that predator).
        relation = (self.mode == "predator" and predator_obstacle_relation) or (self.mode == "prey" and prey_obstacle_relation)
        if self.num_obstacles > 0 and relation:
            self.obstacle_rel_encoder = _mlp(11, (hidden_size,), attention_size, fallback_activation)
            self.obstacle_out_rel = nn.Linear(2 * attention_size, attention_size)
            nn.init.zeros_(self.obstacle_out_rel.weight)
            nn.init.zeros_(self.obstacle_out_rel.bias)

    def get_specification(self) -> Mapping[str, Any]:
        if self.mode == "predator":
            hidden_size = self.num_predators * self.rnn_hidden_size
        else:
            hidden_size = self.rnn_hidden_size
        return {"rnn": {"sizes": [(1, 0, hidden_size)], "sequence_length": self.rnn_sequence_length}}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = _strict_load_allow_missing_prefixes(
            self, state_dict, ("gru.", "obstacle_", "role_"), strict=strict, assign=assign
        )
        self.project_log_std_parameter_()
        return result

    def project_log_std_parameter_(self) -> None:
        """Keep the learnable Gaussian scale inside the mixin's differentiable range."""

        if self._g_clip_log_std:
            with torch.no_grad():
                self.log_std_parameter.clamp_(self._g_log_std_min, self._g_log_std_max)

    def _inactive_placeholder(self, raw_drones: torch.Tensor) -> torch.Tensor:
        """(..., D) mask of crashed drones in raw (..., D, 3 or 6) relative position [+ velocity] features."""
        position = raw_drones[..., :3]
        placeholder = torch.zeros_like(position)
        placeholder[..., 0] = self.inactive_drone_distance
        inactive = (position - placeholder).abs().amax(dim=-1) < 1e-5
        if raw_drones.shape[-1] > 3:
            inactive &= raw_drones[..., 3:6].abs().amax(dim=-1) < 1e-5
        return inactive

    def _obstacle_context(
        self, own_emb: torch.Tensor, obstacles: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Attention from each drone's own embedding over its obstacle entities.

        Args:
            own_emb: (B, A) own-state embeddings.
            obstacles: (B, K, obstacle_dim) obstacle features.
            present: optional (B, K) mask of present pillars. Absent pillars get no attention, and a drone
                without any present pillar gets exactly zero context, as if the branch did not exist.

        Returns:
            (B, A) projected obstacle context (zero at initialization).
        """
        batch, num_obstacles, _ = obstacles.shape
        obstacle_emb = self.obstacle_encoder(obstacles.reshape(batch * num_obstacles, self.obstacle_dim))
        obstacle_emb = obstacle_emb.view(batch, num_obstacles, -1)
        query = self.obstacle_query(own_emb).unsqueeze(1)
        key = self.obstacle_key(obstacle_emb)
        value = self.obstacle_value(obstacle_emb)
        extra = self.obstacle_mha(own_emb, obstacles, present) if hasattr(self, "obstacle_mha") else None
        if present is None:
            weights = _aligned_attention_weights(query, key).to(dtype=value.dtype)
            context = (weights.unsqueeze(-1) * value).sum(dim=1)
            out = self.obstacle_out(context)
            return out if extra is None else out + extra
        weights = _masked_attention_weights(query, key, present).to(dtype=value.dtype)
        context = (weights.unsqueeze(-1) * value).sum(dim=1)
        out = self.obstacle_out(context)
        out = torch.where(present.any(dim=-1, keepdim=True), out, torch.zeros_like(out))
        return out if extra is None else out + extra

    def _predator_features(self, states: torch.Tensor, raw_states: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = states.shape[0]
        blocks = states.view(batch_size, self.num_predators, self.per_predator_block)
        predator_obs = blocks[:, :, : self.per_predator_obs]
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
        if hasattr(self, "role_embedding"):
            own_emb = own_emb + (self.role_embedding - self.role_embedding.mean(dim=0, keepdim=True))

        prey_emb = self.prey_encoder(prey.reshape(batch_size * self.num_predators, 6))
        prey_emb = prey_emb.view(batch_size, self.num_predators, 1, -1)

        teammate_emb = self.teammate_encoder(teammates.reshape(-1, self.teammate_dim))
        teammate_emb = teammate_emb.view(batch_size, self.num_predators, self.num_predators - 1, -1)

        entities = torch.cat((prey_emb, teammate_emb), dim=2)
        query = self.query(own_emb).unsqueeze(2)
        key = self.key(entities)
        value = self.value(entities)
        if self.mask_inactive and raw_states is not None:
            raw_teammates = raw_states.view(batch_size, self.num_predators, self.per_predator_block)[
                :, :, 18 : self.per_predator_obs
            ].reshape(batch_size, self.num_predators, self.num_predators - 1, self.teammate_dim)
            teammate_present = ~self._inactive_placeholder(raw_teammates)
            present = torch.cat((torch.ones_like(teammate_present[:, :, :1]), teammate_present), dim=2)
            weights = _masked_attention_weights(query, key, present).to(dtype=value.dtype)
        else:
            weights = _aligned_attention_weights(query, key).to(dtype=value.dtype)
        context = (weights.unsqueeze(-1) * value).sum(dim=2)
        if self.num_obstacles > 0:
            obstacles = blocks[:, :, self.per_predator_obs :].reshape(
                batch_size * self.num_predators, self.num_obstacles, self.obstacle_dim
            )
            present = None
            if raw_states is not None:
                raw_blocks = raw_states.view(batch_size, self.num_predators, self.per_predator_block)
                present = _obstacle_presence(
                    raw_blocks[:, :, self.per_predator_obs :].reshape(
                        batch_size * self.num_predators, self.num_obstacles, self.obstacle_dim
                    )
                )
            obstacle_context = self._obstacle_context(
                own_emb.reshape(batch_size * self.num_predators, -1), obstacles, present
            )
            context = context + obstacle_context.view(batch_size, self.num_predators, -1)
            if hasattr(self, "obstacle_rel_encoder") and raw_states is not None:
                context = context + self._obstacle_relation(
                    raw_states.view(batch_size, self.num_predators, self.per_predator_block)
                )
        return torch.cat((own_emb, context), dim=-1)

    def _obstacle_relation(self, raw_blocks: torch.Tensor) -> torch.Tensor:
        """(B, P, A) pooled pillar-prey relations of each drone from its raw block (B, P, per_predator_block)."""
        batch, drones = raw_blocks.shape[:2]
        pillars = raw_blocks[:, :, self.per_predator_obs :].reshape(batch, drones, self.num_obstacles, self.obstacle_dim)
        prey = raw_blocks[:, :, 12:15] / self.obstacle_feature_scale
        return self._pooled_relation(pillars, prey)

    def _pooled_relation(self, pillars: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pool pillar tokens (..., K, obstacle_dim raw features) relative to one target offset (..., 3) / scale."""
        offset = pillars[..., :3]
        target = target.unsqueeze(-2).expand_as(offset)
        segment = _segment_axis_distance(offset[..., :2], target[..., :2])
        tokens = torch.cat((offset, pillars[..., 3:4], target, target - offset, segment), dim=-1)
        encoded = self.obstacle_rel_encoder(tokens.reshape(-1, tokens.shape[-1])).view(*offset.shape[:-1], -1)
        pooled, count = _masked_max_mean(encoded, _obstacle_presence(pillars))
        out = self.obstacle_out_rel(pooled)
        return torch.where(count > 0, out, torch.zeros_like(out))

    def _prey_obstacle_relation(self, raw_states: torch.Tensor) -> torch.Tensor:
        """(B, A) pooled pillar relations of the prey to its nearest live predator from the raw prey observation."""
        batch = raw_states.shape[0]
        predator_end = 12 + 6 * self.num_predators
        predators = raw_states[:, 12:predator_end].view(batch, self.num_predators, 6)
        distance = predators[..., :3].norm(dim=-1).masked_fill(self._inactive_placeholder(predators), float("inf"))
        nearest = predators[torch.arange(batch, device=raw_states.device), distance.argmin(dim=-1), :3]
        pillars = raw_states[:, predator_end:].view(batch, self.num_obstacles, self.obstacle_dim)
        return self._pooled_relation(pillars, nearest / self.obstacle_feature_scale)

    def _prey_features(self, states: torch.Tensor, raw_states: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = states.shape[0]
        own = states[:, :12]
        predator_end = 12 + 6 * self.num_predators
        predators = states[:, 12:predator_end].view(batch_size, self.num_predators, 6)

        own_emb = self.own_encoder(own)
        predator_emb = self.predator_encoder(predators.reshape(batch_size * self.num_predators, 6))
        predator_emb = predator_emb.view(batch_size, self.num_predators, -1)

        query = self.query(own_emb).unsqueeze(1)
        key = self.key(predator_emb)
        value = self.value(predator_emb)
        if self.mask_inactive and raw_states is not None:
            raw_predators = raw_states[:, 12:predator_end].view(batch_size, self.num_predators, 6)
            present = ~self._inactive_placeholder(raw_predators)
            weights = _masked_attention_weights(query, key, present).to(dtype=value.dtype)
        else:
            weights = _aligned_attention_weights(query, key).to(dtype=value.dtype)
        context = (weights.unsqueeze(-1) * value).sum(dim=1)
        if self.num_obstacles > 0:
            obstacles = states[:, predator_end:].view(batch_size, self.num_obstacles, self.obstacle_dim)
            present = None
            if raw_states is not None:
                present = _obstacle_presence(
                    raw_states[:, predator_end:].view(batch_size, self.num_obstacles, self.obstacle_dim)
                )
            context = context + self._obstacle_context(own_emb, obstacles, present)
            if hasattr(self, "obstacle_rel_encoder") and raw_states is not None:
                context = context + self._prey_obstacle_relation(raw_states)
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

        needs_raw = self.num_obstacles > 0 or self.mask_inactive
        raw_states = _raw_states(self.observation_space, inputs, states.shape[0]) if needs_raw else None
        if self.mode == "predator":
            features = self._predator_features(states, raw_states)
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

        features = self._prey_features(states, raw_states)
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
            "entity attention -> GRU -> tanh-squashed Gaussian; compatible flat fallback)"
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
        weights = _pairwise_attention_weights(query, key).to(dtype=value.dtype)
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
        num_obstacles: int = 0,
        obstacle_dim: int = 4,
        obstacle_relation: bool = False,
        obstacle_feature_scale: float = 5.0,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_obstacles = int(num_obstacles)
        self.obstacle_dim = int(obstacle_dim)
        self.obstacle_block = self.num_obstacles * self.obstacle_dim
        layout = _infer_centralized_state_layout(
            self.num_observations, max_predators=max_predators, obstacle_block=self.obstacle_block
        )
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

        # Obstacle summary, created last; its zero-initialized projection keeps
        # a critic loaded from an obstacle-free checkpoint unchanged at first.
        if self.num_obstacles > 0 and self.uses_entity_attention:
            self.obstacle_encoder = _mlp(self.obstacle_dim, (hidden_size,), attention_size, fallback_activation)
            self.obstacle_out = nn.Linear(attention_size, self.feature_size)
            nn.init.zeros_(self.obstacle_out.weight)
            nn.init.zeros_(self.obstacle_out.bias)

        # Optional identity-aware pillar relations (off by default). The summary above mean-pools all drones' pillar views
        # without saying whose view each one is, so the critic cannot tell "prey next to pillar k" from "predator passing
        # pillar k" (Stage 8 review). Here each pillar is one token holding everyone's view of it, in a fixed order and
        # rotation-invariant: the prey's surface distance, each predator's surface distance and alive flag, and each
        # predator's distance to the prey (all / obstacle_feature_scale). Encoded per pillar, max + mean pooled; the
        # zero-initialized output, created last, leaves a loaded critic unchanged.
        self.obstacle_feature_scale = float(obstacle_feature_scale)
        if self.num_obstacles > 0 and self.uses_entity_attention and obstacle_relation:
            self.obstacle_rel_encoder = _mlp(1 + 3 * self.num_predators, (hidden_size,), attention_size, fallback_activation)
            self.obstacle_out_rel = nn.Linear(2 * attention_size, self.feature_size)
            nn.init.zeros_(self.obstacle_out_rel.weight)
            nn.init.zeros_(self.obstacle_out_rel.bias)

    def get_specification(self) -> Mapping[str, Any]:
        return {"rnn": {"sizes": [(1, 0, self.rnn_hidden_size)], "sequence_length": self.rnn_sequence_length}}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return _strict_load_allow_missing_prefixes(
            self, state_dict, ("gru.", "obstacle_"), strict=strict, assign=assign
        )

    def _attention_feature(self, states: torch.Tensor, raw_states: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = states.shape[0]
        block = self.per_predator_obs + self.obstacle_block
        predator_state_dim = self.num_predators * block
        predator_blocks = states[:, :predator_state_dim].view(batch_size, self.num_predators, block)
        predator_obs = predator_blocks[:, :, : self.per_predator_obs]
        prey_obs = states[:, predator_state_dim : predator_state_dim + self.prey_obs_dim]

        predator_emb = self.predator_encoder(predator_obs.reshape(-1, self.per_predator_obs))
        predator_emb = predator_emb.view(batch_size, self.num_predators, -1)
        prey_emb = self.prey_encoder(prey_obs).unsqueeze(1)
        entities = torch.cat((predator_emb, prey_emb), dim=1)

        query = self.query(entities)
        key = self.key(entities)
        value = self.value(entities)
        weights = _pairwise_attention_weights(query, key).to(dtype=value.dtype)
        attended = torch.matmul(weights, value)
        feature = attended.mean(dim=1)
        if self.num_obstacles > 0:
            # Every drone's view of every pillar, mean-pooled into one summary.
            obstacles = torch.cat(
                (
                    predator_blocks[:, :, self.per_predator_obs :].reshape(batch_size, -1, self.obstacle_dim),
                    states[:, predator_state_dim + self.prey_obs_dim :].reshape(batch_size, -1, self.obstacle_dim),
                ),
                dim=1,
            )
            summary = self.obstacle_encoder(obstacles.reshape(-1, self.obstacle_dim))
            summary = summary.view(batch_size, obstacles.shape[1], -1)
            if raw_states is None:
                feature = feature + self.obstacle_out(summary.mean(dim=1))
            else:
                raw_blocks = raw_states[:, :predator_state_dim].view(batch_size, self.num_predators, block)
                present = _obstacle_presence(
                    torch.cat(
                        (
                            raw_blocks[:, :, self.per_predator_obs :].reshape(batch_size, -1, self.obstacle_dim),
                            raw_states[:, predator_state_dim + self.prey_obs_dim :].reshape(batch_size, -1, self.obstacle_dim),
                        ),
                        dim=1,
                    )
                )
                # Mean over present pillar views only; no present pillar means no obstacle term at all.
                count = present.sum(dim=1, keepdim=True)
                mean = (summary * present.unsqueeze(-1).to(summary.dtype)).sum(dim=1) / count.clamp(min=1).to(summary.dtype)
                out = self.obstacle_out(mean)
                feature = feature + torch.where(count > 0, out, torch.zeros_like(out))
                if hasattr(self, "obstacle_rel_encoder"):
                    feature = feature + self._obstacle_relation(raw_states)
        return feature

    def _obstacle_relation(self, raw_states: torch.Tensor) -> torch.Tensor:
        """(B, feature_size) pooled per-pillar relations of all entities from the raw centralized state."""
        batch = raw_states.shape[0]
        block = self.per_predator_obs + self.obstacle_block
        predator_dim = self.num_predators * block
        predator_blocks = raw_states[:, :predator_dim].view(batch, self.num_predators, block)
        prey_obs = raw_states[:, predator_dim : predator_dim + self.prey_obs_dim]
        prey_pillars = raw_states[:, predator_dim + self.prey_obs_dim :].view(batch, self.num_obstacles, self.obstacle_dim)
        predator_pillars = predator_blocks[:, :, self.per_predator_obs :].reshape(
            batch, self.num_predators, self.num_obstacles, self.obstacle_dim
        )
        # A crashed predator's own features are zeroed, so its projected gravity (normally unit length) vanishes.
        alive = (predator_blocks[:, :, 6:9].norm(dim=-1) > 0.5).to(raw_states.dtype)              # (B, P)
        to_predators = prey_obs[:, 12 : 12 + 6 * self.num_predators].view(batch, self.num_predators, 6)[..., :3]
        prey_distance = to_predators.norm(dim=-1) / self.obstacle_feature_scale * alive          # (B, P)
        k = self.num_obstacles
        tokens = torch.cat(
            (
                prey_pillars[..., 3:4],                                                          # (B, K, 1)
                predator_pillars[..., 3].transpose(1, 2) * alive.unsqueeze(1),                   # (B, K, P)
                alive.unsqueeze(1).expand(batch, k, self.num_predators),
                prey_distance.unsqueeze(1).expand(batch, k, self.num_predators),
            ),
            dim=-1,
        )
        encoded = self.obstacle_rel_encoder(tokens.reshape(-1, tokens.shape[-1])).view(batch, k, -1)
        pooled, count = _masked_max_mean(encoded, _obstacle_presence(prey_pillars))
        out = self.obstacle_out_rel(pooled)
        return torch.where(count > 0, out, torch.zeros_like(out))

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if self.uses_entity_attention:
            raw_states = _raw_states(self.observation_space, inputs, states.shape[0]) if self.num_obstacles > 0 else None
            features = self._attention_feature(states, raw_states)
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
