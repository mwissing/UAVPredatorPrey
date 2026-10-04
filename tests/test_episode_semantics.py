from __future__ import annotations

import pytest
import torch

from episode_semantics import all_predators_inactive, predator_crash_event_penalty, shifted_squared_speed_penalty


def test_shifted_squared_speed_penalty_has_a_shape_preserving_free_speed_margin() -> None:
    speed_squared = torch.tensor([[0.0, 36.0, 49.0], [25.0, 64.0, 100.0]])

    penalty = shifted_squared_speed_penalty(
        speed_squared,
        free_speed=6.0,
        penalty_scale=-0.05,
        dt=0.02,
    )

    expected = torch.tensor([[0.0, 0.0, -0.013], [0.0, -0.028, -0.064]])
    assert penalty.shape == speed_squared.shape
    assert torch.allclose(penalty, expected)


def test_zero_free_speed_preserves_the_legacy_quadratic_penalty() -> None:
    speed_squared = torch.tensor([[0.0, 4.0, 9.0]])

    penalty = shifted_squared_speed_penalty(
        speed_squared,
        free_speed=0.0,
        penalty_scale=-0.05,
        dt=0.02,
    )

    assert torch.allclose(penalty, speed_squared * -0.05 * 0.02)


def test_shifted_squared_speed_penalty_rejects_negative_free_speed() -> None:
    with pytest.raises(ValueError, match="free_speed must be non-negative"):
        shifted_squared_speed_penalty(
            torch.zeros(1),
            free_speed=-1.0,
            penalty_scale=-0.05,
            dt=0.02,
        )


def test_one_predator_crash_is_penalized_once_without_ending_episode() -> None:
    newly_oob = torch.tensor([[True, False, False]])
    predator_alive = torch.tensor([[False, True, True]])

    event_penalty = predator_crash_event_penalty(newly_oob, penalty=-200.0)
    next_step_penalty = predator_crash_event_penalty(torch.zeros_like(newly_oob), penalty=-200.0)

    assert torch.equal(event_penalty[:, 0], torch.tensor([-200.0, 0.0, 0.0]))
    assert event_penalty.mean().item() == pytest.approx(-200.0 / 3.0)
    assert torch.equal(next_step_penalty, torch.zeros_like(next_step_penalty))
    assert not all_predators_inactive(predator_alive).item()


def test_all_predators_inactive_terminates_independently_of_event_mask() -> None:
    predator_alive = torch.tensor(
        [
            [False, True, False],
            [False, False, False],
        ]
    )

    assert torch.equal(all_predators_inactive(predator_alive), torch.tensor([False, True]))
