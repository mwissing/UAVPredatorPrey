from __future__ import annotations

import pytest
import torch

from episode_semantics import all_predators_inactive, predator_crash_event_penalty


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
