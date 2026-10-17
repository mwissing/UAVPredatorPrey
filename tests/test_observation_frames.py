"""Body-frame observations (Stage 8 review): the frame helper and the yaw invariance it buys."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from frames import rotate_into_frame

ENV_DIR = Path(__file__).resolve().parents[1] / "source" / "UAVPredatorPrey" / "UAVPredatorPrey" / "tasks" / "direct" / "uavpredatorprey_3v1"


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def _rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Forward rotation R(q) v, through the quaternion product q (0, v) q*."""
    pure = torch.cat((torch.zeros_like(vec[..., :1]), vec), dim=-1)
    conj = quat * torch.tensor([1.0, -1.0, -1.0, -1.0])
    return _quat_multiply(_quat_multiply(quat, pure), conj)[..., 1:]


def _yaw(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def _random_quats(n: int, seed: int) -> torch.Tensor:
    q = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed))
    return q / q.norm(dim=-1, keepdim=True)


def test_a_yawed_frame_sees_world_x_as_its_minus_y() -> None:
    body = rotate_into_frame(_yaw(math.pi / 2), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(body, torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)


def test_rotate_into_frame_inverts_the_rotation() -> None:
    q = _random_quats(16, seed=0)
    v = torch.randn(16, 3, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(rotate_into_frame(q, _rotate(q, v)), v, atol=1e-5)
    assert torch.allclose(_rotate(q, rotate_into_frame(q, v)), v, atol=1e-5)


def test_body_frame_relative_velocity_is_invariant_to_a_global_yaw() -> None:
    q_obs = _random_quats(32, seed=2)
    v_obs, v_other = torch.randn(2, 32, 3, generator=torch.Generator().manual_seed(3))
    turn = _yaw(1.234).expand(32, 4)
    q_turned = _quat_multiply(turn, q_obs)
    v_obs_turned, v_other_turned = _rotate(turn, v_obs), _rotate(turn, v_other)
    body = rotate_into_frame(q_obs, v_other - v_obs)
    body_turned = rotate_into_frame(q_turned, v_other_turned - v_obs_turned)
    assert torch.allclose(body, body_turned, atol=1e-5)
    world, world_turned = v_other - v_obs, v_other_turned - v_obs_turned          # the "mixed" frame
    assert (world - world_turned).abs().amax() > 0.1


def test_the_mixed_frame_stays_the_default() -> None:
    tree = ast.parse((ENV_DIR / "uav_3v1_env_cfg.py").read_text())
    values = [node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "observation_frame" for t in node.targets)]
    assert values == ["mixed"]
