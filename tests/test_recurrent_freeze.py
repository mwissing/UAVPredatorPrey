from __future__ import annotations

import torch

from recurrent_mappo import RecurrentMAPPO
from skrl.multi_agents.torch.mappo import MAPPO


def _bare_agent() -> RecurrentMAPPO:
    agent = RecurrentMAPPO.__new__(RecurrentMAPPO)
    agent.possible_agents = ["predator", "prey"]
    agent._frozen_agents = set()
    return agent


def test_trainable_agent_filter_excludes_frozen_role() -> None:
    agent = _bare_agent()

    agent.set_frozen_agents({"prey"})

    assert agent._trainable_agents() == ["predator"]


def test_feedforward_super_update_sees_only_trainable_roles(monkeypatch) -> None:
    agent = _bare_agent()
    agent._rnn = False
    agent.set_frozen_agents({"prey"})
    seen_agents = []

    monkeypatch.setattr(
        MAPPO,
        "_update",
        lambda self, timestep, timesteps: seen_agents.append(tuple(self.possible_agents)),
    )

    agent._update(timestep=0, timesteps=1)

    assert seen_agents == [("predator",)]
    assert agent.possible_agents == ["predator", "prey"]


def test_frozen_policy_gru_advances_and_resets_done_slots() -> None:
    agent = _bare_agent()
    policy = object()
    agent._uid_rnn = {"prey": True}
    agent.policies = {"prey": policy}
    agent.values = {"prey": object()}
    agent._rnn_initial_states = {
        "prey": {
            "policy": [torch.zeros(1, 3, 2)],
            "value": [torch.full((1, 3, 2), 7.0)],
        }
    }
    agent._rnn_final_states = {
        "prey": {
            "policy": [torch.ones(1, 3, 2)],
            "value": [],
        }
    }

    agent._commit_frozen_policy_state("prey", torch.tensor([False, True, False]))

    policy_state = agent._rnn_initial_states["prey"]["policy"][0]
    assert torch.equal(policy_state[:, 0], torch.ones(1, 2))
    assert torch.equal(policy_state[:, 1], torch.zeros(1, 2))
    assert torch.equal(policy_state[:, 2], torch.ones(1, 2))
    assert torch.equal(agent._rnn_initial_states["prey"]["value"][0], torch.full((1, 3, 2), 7.0))
