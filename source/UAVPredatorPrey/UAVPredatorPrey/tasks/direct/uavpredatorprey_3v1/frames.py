"""Frame helpers for the observations (pure torch, so they can be tested without the simulator)."""

from __future__ import annotations

import torch


def rotate_into_frame(quat_wxyz: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    """Express world-frame vectors (..., 3) in the frame with orientation quat_wxyz (..., 4, w first): R(q)^T v.

    Same convention as Isaac Lab's quat_apply_inverse (rotation by the conjugate quaternion).
    """
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec_world, dim=-1)
    return vec_world - w * t + torch.cross(xyz, t, dim=-1)
