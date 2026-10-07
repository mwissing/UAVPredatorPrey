from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import torch


AGENTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "agents"
)
PACKAGE_NAME = "_attention_precision_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")


def test_aligned_attention_uses_fp32_before_overflowing_product() -> None:
    query = torch.full((2, 1, 4), 400.0, dtype=torch.float16)
    key = torch.full((2, 3, 4), 400.0, dtype=torch.float16)

    naive_scores = (query * key).sum(dim=-1) / math.sqrt(key.shape[-1])
    weights = attention_models._aligned_attention_weights(query, key)

    assert not torch.isfinite(naive_scores).all()
    assert weights.dtype == torch.float32
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_pairwise_attention_matches_fp32_reference() -> None:
    query = torch.tensor([[[400.0, 400.0], [300.0, -300.0]]], dtype=torch.float16)
    key = torch.tensor([[[400.0, 400.0], [-400.0, -400.0]]], dtype=torch.float16)
    reference = torch.softmax(
        torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(key.shape[-1]),
        dim=-1,
    )

    weights = attention_models._pairwise_attention_weights(query, key)

    assert weights.dtype == torch.float32
    assert torch.isfinite(weights).all()
    assert torch.equal(weights, reference)


def test_role_specific_log_std_floor_applies_only_to_its_role() -> None:
    model_class = attention_models.RecurrentPredatorPreyAttentionGaussianModel
    # 3 predators with teammate velocities: 3 x (12 + 6 + 2 x 6) observations, 3 x 4 actions.
    predator = model_class(90, 12, device="cpu", min_log_std=-2.0, predator_min_log_std=-1.5, prey_min_log_std=-3.0)
    # Prey observing 3 predators: 12 + 3 x 6 observations, 4 actions.
    prey = model_class(30, 4, device="cpu", min_log_std=-2.0, predator_min_log_std=-1.5)
    shared = model_class(90, 12, device="cpu", min_log_std=-2.0)

    assert (predator.mode, prey.mode, shared.mode) == ("predator", "prey", "predator")
    assert predator._g_log_std_min == -1.5
    assert prey._g_log_std_min == -2.0
    assert shared._g_log_std_min == -2.0

    # The projection after loading and after every optimizer step uses the role floor.
    with torch.no_grad():
        predator.log_std_parameter.fill_(-1.9)
    predator.project_log_std_parameter_()
    assert torch.equal(predator.log_std_parameter, torch.full((4,), -1.5))
