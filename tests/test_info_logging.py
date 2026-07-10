from __future__ import annotations

import pytest
import torch

from info_logging import BatchedScalarInfoWrapper, batch_scalar_tensors_to_cpu


class _FakeEnv:
    marker = "delegated"

    def reset(self):
        return {"agent": torch.zeros(1, 2)}, {"log": {"reset": torch.tensor(1.0)}}

    def step(self, actions):
        infos = {
            "log": {
                "scalar": torch.tensor(2.0),
                "vector": torch.tensor([3.0, 4.0]),
                "text": "unchanged",
            }
        }
        return actions, {}, {}, {}, infos


def test_batch_scalar_tensors_keeps_cpu_payload_unchanged() -> None:
    scalar = torch.tensor(2.0)
    vector = torch.tensor([3.0, 4.0])
    infos = {"log": {"scalar": scalar, "vector": vector, "text": "unchanged"}}

    assert batch_scalar_tensors_to_cpu(infos) == 0
    assert infos["log"]["scalar"] is scalar
    assert infos["log"]["vector"] is vector
    assert infos["log"]["text"] == "unchanged"


def test_batched_info_wrapper_delegates_and_preserves_step_contract() -> None:
    wrapped = BatchedScalarInfoWrapper(_FakeEnv())
    actions = {"agent": torch.ones(1, 2)}

    observations, reset_infos = wrapped.reset()
    step_observations, rewards, terminated, truncated, step_infos = wrapped.step(actions)

    assert wrapped.marker == "delegated"
    assert observations["agent"].shape == (1, 2)
    assert reset_infos["log"]["reset"].item() == 1.0
    assert step_observations is actions
    assert rewards == {}
    assert terminated == {}
    assert truncated == {}
    assert step_infos["log"]["scalar"].item() == 2.0
    assert step_infos["log"]["vector"].tolist() == [3.0, 4.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to verify device transfer batching")
def test_batch_scalar_tensors_moves_cuda_scalars_together() -> None:
    vector = torch.tensor([3.0, 4.0], device="cuda")
    infos = {
        "log": {
            "float_a": torch.tensor(1.25, device="cuda"),
            "float_b": torch.tensor(2.5, device="cuda"),
            "integer": torch.tensor(7, device="cuda"),
            "vector": vector,
        }
    }

    assert batch_scalar_tensors_to_cpu(infos) == 3
    assert infos["log"]["float_a"].device.type == "cpu"
    assert infos["log"]["float_b"].device.type == "cpu"
    assert infos["log"]["integer"].device.type == "cpu"
    assert infos["log"]["float_a"].item() == pytest.approx(1.25)
    assert infos["log"]["float_b"].item() == pytest.approx(2.5)
    assert infos["log"]["integer"].item() == 7
    assert infos["log"]["vector"] is vector
