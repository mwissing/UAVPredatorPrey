"""Per-env pool opponents must act like the agent's own policy, with raw_states: the pillar presence mask, the
crashed-drone masks and the relation branches read unnormalized features. _PoolPolicy lives in scripts/skrl/train.py,
which starts Isaac Sim on import, so this test compiles only that class from the file's source."""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import gymnasium
import torch

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "source" / "UAVPredatorPrey" / "UAVPredatorPrey" / "tasks" / "direct" / "uavpredatorprey_3v1" / "agents"
PACKAGE_NAME = "_pool_policy_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

tree = ast.parse((ROOT / "scripts" / "skrl" / "train.py").read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_PoolPolicy")
namespace: dict[str, Any] = {"torch": torch, "Any": Any, "Mapping": Mapping}
exec(compile(ast.Module(body=[node], type_ignores=[]), "train.py:_PoolPolicy", "exec"), namespace)
PoolPolicy = namespace["_PoolPolicy"]

K, P, BLOCK = 4, 3, 30 + 4 * 4          # pillars, predators, per-drone block (own 12, prey 6, teammates 12, pillars)


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _predator_with_relation():
    torch.manual_seed(1)
    model = attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(P * BLOCK), _box(P * 4), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=K, predator_mask_inactive_teammates=True, predator_obstacle_relation=True,
    )
    torch.nn.init.normal_(model.obstacle_out_rel.weight, std=0.5)   # a trained branch, not the zero start
    return model


def _observations(batch: int) -> torch.Tensor:
    blocks = torch.randn(batch, P, BLOCK, generator=torch.Generator().manual_seed(3))
    blocks[:, :, 30:].reshape(batch, P, K, 4)[..., 3] = 0.3                 # pillars present
    return blocks.reshape(batch, -1)


def _agent_actions(model, observations: torch.Tensor, raw: bool) -> torch.Tensor:
    size = model.get_specification()["rnn"]["sizes"][0]
    rnn = [torch.zeros(size[0], observations.shape[0], size[2])]
    inputs = {"states": observations, "rnn": rnn}
    if raw:
        inputs["raw_states"] = observations
    torch.manual_seed(7)
    return model.act(inputs, role="policy")[0]


def test_a_pool_opponent_acts_like_the_agents_policy_with_raw_states() -> None:
    model = _predator_with_relation()
    model.eval()
    observations = _observations(6)
    pool = PoolPolicy(name="graft", role="predator", policy=model, state_preprocessor=None)
    pool.reset(6, device=torch.device("cpu"), dtype=torch.float32)
    torch.manual_seed(7)
    pooled = pool.act(observations, torch.ones(6, dtype=torch.bool))
    assert torch.equal(pooled, _agent_actions(model, observations, raw=True))
    # Without raw_states the relation branch is skipped: the bug this test guards against.
    assert not torch.equal(pooled, _agent_actions(model, observations, raw=False))
