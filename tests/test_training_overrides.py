from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from training_overrides import override_optimizer_learning_rate


def test_learning_rate_override_preserves_adam_moments() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
    parameter.square().sum().backward()
    optimizer.step()
    state_before = deepcopy(optimizer.state_dict()["state"])
    agent = SimpleNamespace(
        optimizers={"predator": optimizer},
        schedulers={},
        _learning_rate={"predator": 1.0e-4},
    )

    updated_roles = override_optimizer_learning_rate(agent, 3.0e-5)

    assert updated_roles == ["predator"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)
    assert agent._learning_rate["predator"] == pytest.approx(3.0e-5)
    state_after = optimizer.state_dict()["state"]
    assert state_after.keys() == state_before.keys()
    for parameter_id, values_before in state_before.items():
        for key, value_before in values_before.items():
            value_after = state_after[parameter_id][key]
            if isinstance(value_before, torch.Tensor):
                assert torch.equal(value_after, value_before)
            else:
                assert value_after == value_before


@pytest.mark.parametrize("learning_rate", [0.0, -1.0, float("inf"), float("nan")])
def test_learning_rate_override_rejects_invalid_values(learning_rate: float) -> None:
    agent = SimpleNamespace(optimizers={"predator": object()})

    with pytest.raises(ValueError):
        override_optimizer_learning_rate(agent, learning_rate)
