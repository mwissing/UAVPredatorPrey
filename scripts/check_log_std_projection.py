"""Regression check for recurrent Gaussian log-standard-deviation projection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    REPO_ROOT
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "agents"
    / "attention_models.py"
)
SPEC = importlib.util.spec_from_file_location("uav_3v1_attention_models", MODEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load model module from {MODEL_PATH}")
MODEL_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL_MODULE)
RecurrentPredatorPreyAttentionGaussianModel = MODEL_MODULE.RecurrentPredatorPreyAttentionGaussianModel


def main() -> None:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    model = RecurrentPredatorPreyAttentionGaussianModel(
        observation_space,
        action_space,
        device="cpu",
        min_log_std=-2.0,
        max_log_std=0.0,
    )

    checkpoint = model.state_dict()
    checkpoint["log_std_parameter"] = torch.full((4,), -3.0)
    model.load_state_dict(checkpoint)
    torch.testing.assert_close(model.log_std_parameter, torch.full((4,), -2.0))

    optimizer = torch.optim.SGD([model.log_std_parameter], lr=1.0)
    optimizer.zero_grad()
    model.log_std_parameter.grad = torch.ones_like(model.log_std_parameter)
    optimizer.step()
    model.project_log_std_parameter_()
    torch.testing.assert_close(model.log_std_parameter, torch.full((4,), -2.0))

    optimizer.zero_grad()
    model.log_std_parameter.grad = -torch.ones_like(model.log_std_parameter)
    optimizer.step()
    model.project_log_std_parameter_()
    torch.testing.assert_close(model.log_std_parameter, torch.full((4,), -1.0))

    print("log_std projection check passed")


if __name__ == "__main__":
    main()
