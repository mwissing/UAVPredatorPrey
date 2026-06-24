"""Recurrent-compatible MAPPO wrapper for the UAV predator-prey tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.resources.schedulers.torch import KLAdaptiveLR


class RecurrentMAPPO(MAPPO):
    """MAPPO variant that preserves rollout sequences for recurrent models.

    skrl's stock MAPPO path does not store or replay RNN states. This subclass
    keeps the stock implementation for feed-forward models and switches to a
    full-rollout, environment-batched update when policy/value models expose an
    ``rnn`` specification.
    """

    def init(self, trainer_cfg: Mapping[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

        self._rnn = False
        self._uid_rnn: dict[str, bool] = {}
        self._rnn_tensors_names: dict[str, list[str]] = {}
        self._rnn_initial_states: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._rnn_final_states: dict[str, dict[str, list[torch.Tensor]]] = {}

        for uid in self.possible_agents:
            memory = self.memories[uid]
            self._uid_rnn[uid] = False
            self._rnn_tensors_names[uid] = []
            self._rnn_initial_states[uid] = {"policy": [], "value": []}
            self._rnn_final_states[uid] = {"policy": [], "value": []}

            policy_spec = self.policies[uid].get_specification().get("rnn", {})
            for index, size in enumerate(policy_spec.get("sizes", [])):
                self._rnn = True
                self._uid_rnn[uid] = True
                layers, state_size = int(size[0]), int(size[2])
                memory.create_tensor(
                    name=f"rnn_policy_{index}",
                    size=(layers, state_size),
                    dtype=torch.float32,
                    keep_dimensions=True,
                )
                self._rnn_tensors_names[uid].append(f"rnn_policy_{index}")
                self._rnn_initial_states[uid]["policy"].append(
                    torch.zeros(layers, memory.num_envs, state_size, dtype=torch.float32, device=self.device)
                )

            if self.values[uid] is self.policies[uid]:
                self._rnn_initial_states[uid]["value"] = self._rnn_initial_states[uid]["policy"]
            else:
                value_spec = self.values[uid].get_specification().get("rnn", {})
                for index, size in enumerate(value_spec.get("sizes", [])):
                    self._rnn = True
                    self._uid_rnn[uid] = True
                    layers, state_size = int(size[0]), int(size[2])
                    memory.create_tensor(
                        name=f"rnn_value_{index}",
                        size=(layers, state_size),
                        dtype=torch.float32,
                        keep_dimensions=True,
                    )
                    self._rnn_tensors_names[uid].append(f"rnn_value_{index}")
                    self._rnn_initial_states[uid]["value"].append(
                        torch.zeros(layers, memory.num_envs, state_size, dtype=torch.float32, device=self.device)
                    )

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        if not self._rnn:
            return super().act(states, timestep, timesteps)

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            data = []
            for uid in self.possible_agents:
                rnn = {"rnn": self._rnn_initial_states[uid]["policy"]} if self._uid_rnn[uid] else {}
                data.append(
                    self.policies[uid].act(
                        {"states": self._state_preprocessor[uid](states[uid]), **rnn},
                        role="policy",
                    )
                )

            actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
            log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
            outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}
            self._current_log_prob = log_prob

        for uid in self.possible_agents:
            if self._uid_rnn[uid]:
                self._rnn_final_states[uid]["policy"] = outputs[uid].get("rnn", [])

        return actions, log_prob, outputs

    def reset_recurrent_states(self, done: torch.Tensor | Mapping[str, torch.Tensor]) -> None:
        if not self._rnn:
            return

        with torch.inference_mode():
            for uid in self.possible_agents:
                if not self._uid_rnn[uid]:
                    continue
                mask = done[uid] if isinstance(done, Mapping) else done
                finished = mask.view(-1).nonzero(as_tuple=False)
                env_ids = finished[:, 0] if finished.numel() else None

                # In inference/evaluation there is no record_transition call, so act()
                # produces final states that must become the next initial states here.
                for role in ("policy", "value"):
                    final_states = self._rnn_final_states[uid].get(role, [])
                    initial_states = self._rnn_initial_states[uid].get(role, [])

                    if final_states:
                        if env_ids is not None:
                            for state in final_states:
                                state[:, env_ids] = 0
                        self._rnn_initial_states[uid][role] = [state.detach() for state in final_states]
                    elif env_ids is not None:
                        for state in initial_states:
                            state[:, env_ids] = 0

    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        if not self._rnn:
            return super().record_transition(
                states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
            )

        super(MAPPO, self).record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memories:
            shared_states = infos["shared_states"]
            self._current_shared_next_states = infos["shared_next_states"]

            for uid in self.possible_agents:
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    rnn = {"rnn": self._rnn_initial_states[uid]["value"]} if self._uid_rnn[uid] else {}
                    values, _, outputs = self.values[uid].act(
                        {"states": self._shared_state_preprocessor[uid](shared_states), **rnn},
                        role="value",
                    )
                    values = self._value_preprocessor[uid](values, inverse=True)

                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]

                rnn_states = {}
                if self._uid_rnn[uid]:
                    rnn_states.update(
                        {
                            f"rnn_policy_{index}": state.transpose(0, 1)
                            for index, state in enumerate(self._rnn_initial_states[uid]["policy"])
                        }
                    )
                    if self.policies[uid] is not self.values[uid]:
                        rnn_states.update(
                            {
                                f"rnn_value_{index}": state.transpose(0, 1)
                                for index, state in enumerate(self._rnn_initial_states[uid]["value"])
                            }
                        )

                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values,
                    shared_states=shared_states,
                    **rnn_states,
                )

                if self._uid_rnn[uid]:
                    self._rnn_final_states[uid]["value"] = (
                        self._rnn_final_states[uid]["policy"]
                        if self.policies[uid] is self.values[uid]
                        else outputs.get("rnn", [])
                    )
                    finished_episodes = (terminated[uid] | truncated[uid]).nonzero(as_tuple=False)
                    if finished_episodes.numel():
                        for state in self._rnn_final_states[uid]["policy"]:
                            state[:, finished_episodes[:, 0]] = 0
                        if self.policies[uid] is not self.values[uid]:
                            for state in self._rnn_final_states[uid]["value"]:
                                state[:, finished_episodes[:, 0]] = 0
                    self._rnn_initial_states[uid] = {
                        "policy": [state.detach() for state in self._rnn_final_states[uid]["policy"]],
                        "value": [state.detach() for state in self._rnn_final_states[uid]["value"]],
                    }

    def _update(self, timestep: int, timesteps: int) -> None:
        if not self._rnn:
            return super()._update(timestep, timesteps)

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            last_values: torch.Tensor,
            discount_factor: float,
            lambda_coefficient: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            for index in reversed(range(memory_size)):
                next_values = values[index + 1] if index < memory_size - 1 else last_values
                advantage = rewards[index] - values[index] + discount_factor * not_dones[index] * (
                    next_values + lambda_coefficient * advantage
                )
                advantages[index] = advantage

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]
            memory = self.memories[uid]

            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                value.train(False)
                rnn_value = {"rnn": self._rnn_initial_states[uid]["value"]} if self._uid_rnn[uid] else {}
                last_values, _, _ = value.act(
                    {
                        "states": self._shared_state_preprocessor[uid](self._current_shared_next_states.float()),
                        **rnn_value,
                    },
                    role="value",
                )
                value.train(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

            values = memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=values,
                last_values=last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )

            memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
            memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
            memory.set_tensor_by_name("advantages", advantages)

            env_indices = torch.randperm(memory.num_envs, device=self.device)
            env_batches = torch.chunk(env_indices, self._mini_batches[uid])

            cumulative_policy_loss = 0.0
            cumulative_entropy_loss = 0.0
            cumulative_value_loss = 0.0
            update_count = 0

            for epoch in range(self._learning_epochs[uid]):
                kl_divergences = []

                for env_batch in env_batches:
                    sampled_states = memory.get_tensor_by_name("states")[:, env_batch].reshape(
                        -1, policy.num_observations
                    )
                    sampled_shared_states = memory.get_tensor_by_name("shared_states")[:, env_batch].reshape(
                        -1, value.num_observations
                    )
                    sampled_actions = memory.get_tensor_by_name("actions")[:, env_batch].reshape(
                        -1, policy.num_actions
                    )
                    sampled_terminated = memory.get_tensor_by_name("terminated")[:, env_batch].reshape(-1, 1)
                    sampled_truncated = memory.get_tensor_by_name("truncated")[:, env_batch].reshape(-1, 1)
                    sampled_log_prob = memory.get_tensor_by_name("log_prob")[:, env_batch].reshape(-1, 1)
                    sampled_values = memory.get_tensor_by_name("values")[:, env_batch].reshape(-1, 1)
                    sampled_returns = memory.get_tensor_by_name("returns")[:, env_batch].reshape(-1, 1)
                    sampled_advantages = memory.get_tensor_by_name("advantages")[:, env_batch].reshape(-1, 1)

                    rnn_policy = {}
                    rnn_value = {}
                    if self._uid_rnn[uid]:
                        policy_states = []
                        value_states = []
                        for name in self._rnn_tensors_names[uid]:
                            tensor = memory.get_tensor_by_name(name)
                            initial_state = tensor[0, env_batch].transpose(0, 1).contiguous()
                            if "policy" in name:
                                policy_states.append(initial_state)
                            elif "value" in name:
                                value_states.append(initial_state)

                        done_sequence = sampled_terminated | sampled_truncated
                        rnn_policy = {"rnn": policy_states, "terminated": done_sequence}
                        rnn_value = {
                            "rnn": policy_states if policy is value else value_states,
                            "terminated": done_sequence,
                        }

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        sampled_states = self._state_preprocessor[uid](sampled_states, train=not epoch)
                        sampled_shared_states = self._shared_state_preprocessor[uid](
                            sampled_shared_states, train=not epoch
                        )

                        _, next_log_prob, _ = policy.act(
                            {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy},
                            role="policy",
                        )

                        with torch.no_grad():
                            ratio = next_log_prob - sampled_log_prob
                            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                            kl_divergences.append(kl_divergence)

                        if self._kl_threshold[uid] and kl_divergence > self._kl_threshold[uid]:
                            break

                        if self._entropy_loss_scale[uid]:
                            entropy_loss = -self._entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                        else:
                            entropy_loss = 0

                        ratio = torch.exp(next_log_prob - sampled_log_prob)
                        surrogate = sampled_advantages * ratio
                        surrogate_clipped = sampled_advantages * torch.clip(
                            ratio,
                            1.0 - self._ratio_clip[uid],
                            1.0 + self._ratio_clip[uid],
                        )
                        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                        predicted_values, _, _ = value.act(
                            {"states": sampled_shared_states, **rnn_value},
                            role="value",
                        )
                        if self._clip_predicted_values:
                            predicted_values = sampled_values + torch.clip(
                                predicted_values - sampled_values,
                                min=-self._value_clip[uid],
                                max=self._value_clip[uid],
                            )
                        value_loss = self._value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                    self.optimizers[uid].zero_grad()
                    self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                    if config.torch.is_distributed:
                        policy.reduce_parameters()
                        if policy is not value:
                            value.reduce_parameters()

                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.optimizers[uid])
                        if policy is value:
                            nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip[uid])
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(policy.parameters(), value.parameters()), self._grad_norm_clip[uid]
                            )

                    self.scaler.step(self.optimizers[uid])
                    self.scaler.update()

                    cumulative_policy_loss += float(policy_loss.item())
                    cumulative_value_loss += float(value_loss.item())
                    if self._entropy_loss_scale[uid]:
                        cumulative_entropy_loss += float(entropy_loss.item())
                    update_count += 1

                if self._learning_rate_scheduler[uid]:
                    if isinstance(self.schedulers[uid], KLAdaptiveLR):
                        kl = torch.tensor(kl_divergences, device=self.device).mean()
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        self.schedulers[uid].step(kl.item())
                    else:
                        self.schedulers[uid].step()

            denominator = max(update_count, 1)
            self.track_data(f"Loss / Policy loss ({uid})", cumulative_policy_loss / denominator)
            self.track_data(f"Loss / Value loss ({uid})", cumulative_value_loss / denominator)
            if self._entropy_loss_scale:
                self.track_data(f"Loss / Entropy loss ({uid})", cumulative_entropy_loss / denominator)
            self.track_data(
                f"Policy / Standard deviation ({uid})",
                policy.distribution(role="policy").stddev.mean().item(),
            )
            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
