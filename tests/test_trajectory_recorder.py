from __future__ import annotations

import math

import numpy as np
import torch

from trajectory_recorder import FirstEpisodeRecorder


def _frame(step: int) -> dict[str, torch.Tensor]:
    return {"prey_pos": torch.full((3, 3), float(step)), "predator_alive": torch.ones(3, 3)}


def test_records_only_each_environments_first_episode(tmp_path) -> None:
    recorder = FirstEpisodeRecorder(num_envs=3, device="cpu")
    recorder.record(_frame(0))

    # Step 1: env 1 ends its first episode with a catch.
    recorder.finish(torch.tensor([1]), {"env_ids": torch.tensor([1]), "catch": torch.tensor([1.0])})
    recorder.record(_frame(1))
    # Step 2: env 1 ends its second episode (ignored), env 0 ends its first (time-out).
    recorder.finish(
        torch.tensor([0, 1]),
        {"env_ids": torch.tensor([0, 1]), "catch": torch.tensor([0.0, 0.0])},
    )
    recorder.record(_frame(2))

    assert not recorder.complete
    path = tmp_path / "trajectories.npz"
    recorder.save(path)
    data = np.load(path)

    assert data["prey_pos"].shape == (3, 3, 3)
    np.testing.assert_array_equal(
        data["active"],
        np.array([[True, True, True], [True, False, True], [False, False, True]]),
    )
    catch = data["outcome_catch"]
    assert catch[0] == 0.0 and catch[1] == 1.0 and math.isnan(catch[2])


def test_complete_after_every_first_episode_ended() -> None:
    recorder = FirstEpisodeRecorder(num_envs=2, device="cpu")
    recorder.finish(torch.tensor([0, 1]), {"env_ids": torch.tensor([0, 1]), "catch": torch.tensor([1.0, 0.0])})

    assert recorder.complete
