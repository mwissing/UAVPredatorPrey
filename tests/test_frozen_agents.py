from __future__ import annotations

import torch

from frozen_agents import (
    FrozenAgentOptimizer,
    capture_frozen_fingerprints,
    freeze_agent_training,
    verify_frozen_fingerprints,
)


class DummyPreprocessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("current_count", torch.tensor(1.0, dtype=torch.float64))


class DummyAgent:
    def __init__(self) -> None:
        self.possible_agents = ["predator", "prey"]
        self.policies = {
            "predator": torch.nn.Linear(3, 2),
            "prey": torch.nn.Linear(3, 2),
        }
        self.values = {
            "predator": torch.nn.Linear(3, 1),
            "prey": torch.nn.Linear(3, 1),
        }
        self.optimizers = {
            uid: torch.optim.Adam(
                list(self.policies[uid].parameters()) + list(self.values[uid].parameters()),
                lr=1.0e-3,
            )
            for uid in self.possible_agents
        }
        self.schedulers = {uid: object() for uid in self.possible_agents}
        self._learning_rate_scheduler = {uid: True for uid in self.possible_agents}
        self._state_preprocessor = {uid: DummyPreprocessor() for uid in self.possible_agents}
        self.checkpoint_modules = {
            uid: {"optimizer": self.optimizers[uid]}
            for uid in self.possible_agents
        }
        self.registered_frozen_agents: set[str] = set()

    def set_frozen_agents(self, agent_names: set[str]) -> None:
        self.registered_frozen_agents = set(agent_names)


def test_freeze_disables_gradients_optimizer_scheduler_and_checkpoint_state() -> None:
    agent = DummyAgent()

    frozen = freeze_agent_training(agent, {"prey"})

    assert frozen == {"prey"}
    assert agent.registered_frozen_agents == {"prey"}
    assert isinstance(agent.optimizers["prey"], FrozenAgentOptimizer)
    assert agent._learning_rate_scheduler["prey"] is None
    assert "optimizer" not in agent.checkpoint_modules["prey"]
    assert all(not parameter.requires_grad for parameter in agent.policies["prey"].parameters())
    assert all(not parameter.requires_grad for parameter in agent.values["prey"].parameters())
    assert all(parameter.requires_grad for parameter in agent.policies["predator"].parameters())


def test_frozen_fingerprint_detects_parameter_drift() -> None:
    agent = DummyAgent()
    expected = capture_frozen_fingerprints(agent, {"prey"})

    verify_frozen_fingerprints(agent, expected)
    with torch.no_grad():
        next(agent.policies["prey"].parameters()).add_(1.0)

    try:
        verify_frozen_fingerprints(agent, expected)
    except RuntimeError as error:
        assert "prey" in str(error)
    else:
        raise AssertionError("Expected frozen-role fingerprint verification to detect parameter drift")
