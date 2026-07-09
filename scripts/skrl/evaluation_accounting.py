"""Pure helpers for unbiased vectorized episode accounting."""

from __future__ import annotations

import random
from collections.abc import Iterable


EPISODE_PAYLOAD_METRICS = {
    "catch": "Metrics/catch_rate",
    "clean_catch": "Metrics/clean_catch_rate",
    "forced_prey_oob": "Metrics/forced_prey_oob_rate",
    "predator_success": "Metrics/predator_success_rate",
    "predator_oob": "Metrics/predator_oob_rate",
    "prey_oob": "Metrics/prey_oob_rate",
    "episode_length": "Metrics/episode_length",
    "closest_approach": "Metrics/episode_closest_approach",
    "predator_min_height": "Metrics/episode_predator_min_height",
    "prey_min_height": "Metrics/episode_prey_min_height",
    "teammate_min_distance": "Metrics/episode_teammate_min_distance",
    "teammate_close_rate": "Metrics/episode_teammate_close_rate",
    "predator_soft_arena_outside": "Metrics/predator_soft_arena_outside",
    "predator_soft_arena_outside_max": "Metrics/episode_predator_soft_arena_outside_max",
    "predator_soft_arena_outside_fraction": "Metrics/episode_predator_soft_arena_outside_fraction",
    "prey_soft_arena_outside": "Metrics/prey_soft_arena_outside",
    "prey_soft_arena_outside_max": "Metrics/episode_prey_soft_arena_outside_max",
    "prey_soft_arena_outside_fraction": "Metrics/episode_prey_soft_arena_outside_fraction",
}


def _integer_list(values: Iterable[int]) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _boolean_list(values: Iterable[bool]) -> list[bool]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [bool(value) for value in values]


class BalancedEpisodeSelector:
    """Accept a fixed episode quota from each vector-environment slot.

    Counting the first N completion events overweights short episodes because
    auto-reset slots can finish repeatedly while a long initial episode is
    still running. Fixed per-slot quotas remove that completion-time bias.
    """

    def __init__(self, num_envs: int, requested_episodes: int, seed: int = 0) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if requested_episodes <= 0:
            raise ValueError(f"requested_episodes must be positive, got {requested_episodes}")

        base, remainder = divmod(requested_episodes, num_envs)
        quotas = [base] * num_envs
        env_order = list(range(num_envs))
        random.Random(seed).shuffle(env_order)
        for env_id in env_order[:remainder]:
            quotas[env_id] += 1

        self._quotas = quotas
        self._counts = [0] * num_envs
        self._requested_episodes = requested_episodes
        self._completed_episodes = 0

    @property
    def quotas(self) -> tuple[int, ...]:
        return tuple(self._quotas)

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(self._counts)

    @property
    def completed_episodes(self) -> int:
        return self._completed_episodes

    @property
    def complete(self) -> bool:
        return self._completed_episodes >= self._requested_episodes

    @property
    def required_rounds(self) -> int:
        return max(self._quotas)

    def accept(self, env_ids: Iterable[int]) -> list[int]:
        """Return payload row indices accepted by the fixed per-slot quotas."""
        accepted_rows = []
        for row, env_id in enumerate(_integer_list(env_ids)):
            if env_id < 0 or env_id >= len(self._quotas):
                raise ValueError(f"env_id {env_id} is outside [0, {len(self._quotas)})")
            if self._counts[env_id] >= self._quotas[env_id]:
                continue
            self._counts[env_id] += 1
            self._completed_episodes += 1
            accepted_rows.append(row)
        return accepted_rows


def accepted_payload_rows(
    selector: BalancedEpisodeSelector,
    payload_env_ids: Iterable[int],
    done_mask: Iterable[bool],
) -> list[int]:
    """Filter stale payload rows, then apply fixed per-slot quotas."""
    env_ids = _integer_list(payload_env_ids)
    done = _boolean_list(done_mask)
    valid_rows = [
        row
        for row, env_id in enumerate(env_ids)
        if 0 <= env_id < len(done) and done[env_id]
    ]
    accepted_current_rows = selector.accept(env_ids[row] for row in valid_rows)
    return [valid_rows[row] for row in accepted_current_rows]


def payload_metric_name(payload_name: str) -> str | None:
    """Map environment payload fields to public evaluation metric names."""
    metric_name = EPISODE_PAYLOAD_METRICS.get(payload_name)
    if metric_name is not None:
        return metric_name
    if payload_name.startswith("predator_") and payload_name.endswith("_oob"):
        agent_index = payload_name.removeprefix("predator_").removesuffix("_oob")
        if agent_index.isdigit():
            return f"Metrics/predator_{agent_index}_oob_rate"
    return None
