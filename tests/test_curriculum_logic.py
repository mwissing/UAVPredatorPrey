from __future__ import annotations

from types import SimpleNamespace

import pytest

from hysteresis_curriculum import (
    POOL_TYPE_ELITE,
    POOL_TYPE_RECENT,
    _maybe_promote_to_pool,
    _pfsp_weight,
    _pool_entry_weight,
    _prune_pool,
)


def _args(**overrides):
    values = {
        "pool_sampling": "pfsp",
        "pfsp_min_weight": 0.05,
        "pfsp_weighting": "squared",
        "auto_pool": True,
        "auto_pool_predator_min_catch": 0.6,
        "auto_pool_prey_max_catch": 0.4,
        "auto_pool_max_predator_oob": 0.12,
        "auto_pool_max_prey_oob": 0.12,
        "auto_pool_max_predator_soft": 0.2,
        "auto_pool_max_prey_soft": 0.2,
        "auto_pool_max_opponent_oob": 0.12,
        "auto_pool_max_opponent_soft": 0.2,
        "auto_pool_predator_min_cross_play_avg_catch": 0.55,
        "auto_pool_predator_min_cross_play_min_catch": 0.35,
        "auto_pool_prey_max_cross_play_avg_catch": 0.45,
        "auto_pool_prey_max_cross_play_max_catch": 0.65,
        "auto_pool_max_entries_per_role": 2,
        "recent_pool_max_entries_per_role": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("weighting", "expected"),
    [("linear", 0.25), ("squared", 0.0625), ("variance", 0.1875)],
)
def test_pfsp_weighting_functions(weighting: str, expected: float) -> None:
    assert _pfsp_weight(0.75, weighting) == pytest.approx(expected)


def test_pfsp_uses_training_role_win_rate_and_minimum_weight() -> None:
    prey_entry = {"base_weight": 2.0, "metrics": {"Metrics/catch_rate": 0.8}}
    predator_entry = {"base_weight": 2.0, "metrics": {"Metrics/catch_rate": 0.2}}
    args = _args()

    predator_training_weight = _pool_entry_weight(
        prey_entry,
        sampled_agent="prey",
        training_agent="predator",
        args=args,
    )
    prey_training_weight = _pool_entry_weight(
        predator_entry,
        sampled_agent="predator",
        training_agent="prey",
        args=args,
    )

    assert predator_training_weight == pytest.approx(0.1)
    assert prey_training_weight == pytest.approx(0.1)


def test_predator_promotion_requires_latest_and_cross_play_robustness(tmp_path) -> None:
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.touch()
    summary = {
        "episode_metrics": {
            "Metrics/catch_rate": 0.9,
            "Metrics/predator_oob_rate": 0.01,
            "Metrics/prey_oob_rate": 0.01,
            "Metrics/predator_soft_arena_outside": 0.02,
            "Metrics/prey_soft_arena_outside": 0.02,
        }
    }
    pool = {"predator": [], "prey": []}

    rejected = _maybe_promote_to_pool(
        pool,
        agent="predator",
        checkpoint=checkpoint,
        summary=summary,
        cross_play_aggregate={"catch_avg": 0.5, "catch_min": 0.2},
        phase_index=3,
        args=_args(),
    )
    accepted = _maybe_promote_to_pool(
        pool,
        agent="predator",
        checkpoint=checkpoint,
        summary=summary,
        cross_play_aggregate={"catch_avg": 0.8, "catch_min": 0.7},
        phase_index=4,
        args=_args(),
    )

    assert rejected is not None and not rejected["promoted"]
    assert accepted is not None and accepted["promoted"]
    assert len(pool["predator"]) == 1
    assert pool["predator"][0]["pool_type"] == POOL_TYPE_ELITE


def test_pruning_keeps_best_elites_and_newest_recent_entry() -> None:
    pool = {
        "predator": [
            {"name": "elite-low", "pool_type": POOL_TYPE_ELITE, "auto_score": 0.1, "added_phase_index": 9},
            {"name": "elite-high", "pool_type": POOL_TYPE_ELITE, "auto_score": 0.9, "added_phase_index": 1},
            {"name": "elite-mid", "pool_type": POOL_TYPE_ELITE, "auto_score": 0.5, "added_phase_index": 2},
            {"name": "recent-old", "pool_type": POOL_TYPE_RECENT, "added_phase_index": 7},
            {"name": "recent-new", "pool_type": POOL_TYPE_RECENT, "added_phase_index": 8},
        ],
        "prey": [],
    }

    _prune_pool(pool, _args())

    assert {entry["name"] for entry in pool["predator"]} == {
        "elite-high",
        "elite-mid",
        "recent-new",
    }
