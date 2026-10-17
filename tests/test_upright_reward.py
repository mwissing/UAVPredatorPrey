"""The predators' upright term: the default positive stream and the zero-at-level option (Stage 8 review)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from episode_semantics import upright_reward

ENV_DIR = Path(__file__).resolve().parents[1] / "source" / "UAVPredatorPrey" / "UAVPredatorPrey" / "tasks" / "direct" / "uavpredatorprey_3v1"


def test_default_pays_a_level_drone_every_step() -> None:
    assert torch.allclose(upright_reward(torch.tensor([-1.0]), 5.0, 0.02), torch.tensor([0.1]))


def test_zero_at_level_keeps_the_tilt_gradient_without_the_alive_bonus() -> None:
    tilt = torch.tensor([0.0, 0.3, math.pi / 2], requires_grad=True)
    gravity_z = -torch.cos(tilt)
    zero = upright_reward(gravity_z, 5.0, 0.02, zero_at_level=True)
    assert torch.allclose(zero, 0.1 * (torch.cos(tilt) - 1.0))
    assert zero[0] == 0 and zero[2] < 0
    default = upright_reward(gravity_z, 5.0, 0.02)
    g_zero = torch.autograd.grad(zero.sum(), tilt, retain_graph=True)[0]
    g_default = torch.autograd.grad(default.sum(), tilt)[0]
    assert torch.allclose(g_zero, g_default)


def test_the_positive_stream_stays_the_default() -> None:
    tree = ast.parse((ENV_DIR / "uav_3v1_env_cfg.py").read_text())
    values = [node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "predator_upright_zero_at_level" for t in node.targets)]
    assert values == [False]
