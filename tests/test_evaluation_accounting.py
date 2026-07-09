from __future__ import annotations

import pytest

from evaluation_accounting import BalancedEpisodeSelector, accepted_payload_rows, payload_metric_name


def test_fast_env_cannot_fill_other_env_quotas() -> None:
    selector = BalancedEpisodeSelector(num_envs=3, requested_episodes=3, seed=42)

    assert selector.accept([0]) == [0]
    assert selector.accept([0]) == []
    assert selector.accept([0, 1]) == [1]
    assert selector.completed_episodes == 2
    assert not selector.complete

    assert selector.accept([2]) == [0]
    assert selector.complete
    assert selector.counts == (1, 1, 1)


def test_episode_remainder_is_distributed_as_fixed_slot_quotas() -> None:
    selector = BalancedEpisodeSelector(num_envs=3, requested_episodes=8, seed=7)

    assert sum(selector.quotas) == 8
    assert sorted(selector.quotas) == [2, 3, 3]
    assert selector.required_rounds == 3


def test_invalid_environment_id_is_rejected() -> None:
    selector = BalancedEpisodeSelector(num_envs=2, requested_episodes=2)

    with pytest.raises(ValueError, match="outside"):
        selector.accept([2])


def test_payload_metric_mapping_includes_dynamic_predator_oob() -> None:
    assert payload_metric_name("catch") == "Metrics/catch_rate"
    assert payload_metric_name("predator_2_oob") == "Metrics/predator_2_oob_rate"
    assert payload_metric_name("env_ids") is None


def test_stale_payload_rows_are_not_counted() -> None:
    selector = BalancedEpisodeSelector(num_envs=3, requested_episodes=3)

    rows = accepted_payload_rows(
        selector,
        payload_env_ids=[0, 1],
        done_mask=[False, True, False],
    )

    assert rows == [1]
    assert selector.counts == (0, 1, 0)
