"""A forced prey crash (exit without a catch, a predator within assist_distance) pays the team like a catch."""

from __future__ import annotations

import torch

from episode_semantics import forced_crash_bonus

FAR = 100.0


def _bonus(prey_oob, caught, distances):
    return forced_crash_bonus(torch.tensor(prey_oob), torch.tensor(caught), torch.tensor(distances), bonus=200.0,
                              assist_distance=1.5)


def test_the_nearest_active_predator_gets_the_scaled_bonus() -> None:
    bonus = _bonus([True], [False], [[3.0, 1.2, FAR]])          # predator 1 within 1.5 m, predator 2 inactive
    assert torch.equal(bonus, torch.tensor([[0.0], [600.0], [0.0]]))
    assert float(bonus.mean(dim=0)) == 200.0                    # the team mean equals the catch bonus


def test_no_bonus_without_a_crash_with_a_catch_or_with_nobody_close() -> None:
    no_exit = _bonus([False], [False], [[0.5, 3.0, 3.0]])
    caught = _bonus([True], [True], [[0.2, 3.0, 3.0]])           # a catch is paid by the catch bonus instead
    too_far = _bonus([True], [False], [[1.6, 2.0, FAR]])
    for bonus in (no_exit, caught, too_far):
        assert torch.equal(bonus, torch.zeros(3, 1))


def test_batched_envs_are_independent() -> None:
    bonus = _bonus([True, True, False], [False, False, False], [[1.0, 0.4, 2.0], [FAR, FAR, 1.4], [0.1, 0.1, 0.1]])
    assert torch.equal(bonus, torch.tensor([[0.0, 0.0, 0.0], [600.0, 0.0, 0.0], [0.0, 600.0, 0.0]]))
