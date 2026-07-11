from __future__ import annotations

import math

import pytest
import torch

from motion_diagnostics import compute_motion_diagnostics


def test_motion_diagnostics_mask_inactive_predators_and_decompose_relative_velocity() -> None:
    diagnostics = compute_motion_diagnostics(
        predator_positions=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]
        ),
        prey_positions=torch.tensor([[1.0, 0.0, 0.0]]),
        predator_velocities=torch.tensor(
            [[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [100.0, 0.0, 0.0]]]
        ),
        prey_velocities=torch.zeros(1, 3),
        predator_alive=torch.tensor([[True, True, False]]),
        raw_distance_progress=torch.tensor([[0.08, -0.02, 0.09]]),
        progress_valid=torch.tensor([[True, True, False]]),
        predator_upright=torch.tensor([[1.0, 0.5, -1.0]]),
        progress_clip=0.08,
    )

    assert diagnostics["predator_speed_mean"].item() == pytest.approx(1.5)
    assert diagnostics["predator_speed_rms"].item() == pytest.approx(math.sqrt(2.5))
    assert diagnostics["predator_speed_max"].item() == pytest.approx(2.0)
    assert diagnostics["predator_speed_above_4mps_fraction"].item() == pytest.approx(0.0)
    assert diagnostics["predator_speed_above_6mps_fraction"].item() == pytest.approx(0.0)

    expected_closing = (2.0 - 1.0 / math.sqrt(2.0)) / 2.0
    expected_lateral = 1.0 / (2.0 * math.sqrt(2.0))
    assert diagnostics["predator_closing_speed_mean"].item() == pytest.approx(expected_closing)
    assert diagnostics["predator_closing_speed_positive_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["predator_closing_speed_negative_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["predator_lateral_relative_speed_mean"].item() == pytest.approx(expected_lateral)

    assert diagnostics["predator_progress_positive_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["predator_progress_negative_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["predator_progress_clip_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["predator_upright_mean"].item() == pytest.approx(0.75)
    assert diagnostics["predator_tilt_over_45deg_fraction"].item() == pytest.approx(0.5)

    assert diagnostics["prey_speed_mean"].item() == pytest.approx(0.0)
    assert diagnostics["prey_speed_rms"].item() == pytest.approx(0.0)
    assert diagnostics["prey_speed_max"].item() == pytest.approx(0.0)
    assert diagnostics["prey_speed_above_4mps_fraction"].item() == pytest.approx(0.0)
    assert diagnostics["prey_speed_above_6mps_fraction"].item() == pytest.approx(0.0)


def test_motion_diagnostics_return_finite_zero_predator_values_when_all_are_inactive() -> None:
    diagnostics = compute_motion_diagnostics(
        predator_positions=torch.zeros(2, 3, 3),
        prey_positions=torch.zeros(2, 3),
        predator_velocities=torch.ones(2, 3, 3),
        prey_velocities=torch.tensor([[0.0, 3.0, 4.0], [0.0, 3.0, 4.0]]),
        predator_alive=torch.zeros(2, 3, dtype=torch.bool),
        raw_distance_progress=torch.ones(2, 3),
        progress_valid=torch.zeros(2, 3, dtype=torch.bool),
        predator_upright=torch.ones(2, 3),
        progress_clip=0.08,
    )

    for name, value in diagnostics.items():
        assert torch.isfinite(value), name
        if name.startswith("predator_"):
            assert value.item() == pytest.approx(0.0), name

    assert diagnostics["prey_speed_mean"].item() == pytest.approx(5.0)
    assert diagnostics["prey_speed_rms"].item() == pytest.approx(5.0)
    assert diagnostics["prey_speed_max"].item() == pytest.approx(5.0)
    assert diagnostics["prey_speed_above_4mps_fraction"].item() == pytest.approx(1.0)
    assert diagnostics["prey_speed_above_6mps_fraction"].item() == pytest.approx(0.0)
