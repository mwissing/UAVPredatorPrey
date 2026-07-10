from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any

import torch


def batch_scalar_tensors_to_cpu(infos: Any, *, info_key: str = "log") -> int:
    """Move scalar CUDA log tensors to CPU with one transfer per dtype/device group."""

    if not isinstance(infos, MutableMapping):
        return 0
    payload = infos.get(info_key)
    if not isinstance(payload, MutableMapping):
        return 0

    groups: dict[tuple[torch.device, torch.dtype], list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, value in payload.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1 and value.device.type != "cpu":
            groups[(value.device, value.dtype)].append((name, value))

    moved = 0
    for entries in groups.values():
        host_values = torch.stack([value.detach().reshape(()) for _, value in entries]).cpu()
        for index, (name, _) in enumerate(entries):
            payload[name] = host_values[index]
        moved += len(entries)
    return moved


class BatchedScalarInfoWrapper:
    """Delegate env wrapper that prevents one CUDA synchronization per logged scalar."""

    def __init__(self, env: Any, *, info_key: str = "log") -> None:
        self._env = env
        self._info_key = info_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self):
        observations, infos = self._env.reset()
        batch_scalar_tensors_to_cpu(infos, info_key=self._info_key)
        return observations, infos

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self._env.step(actions)
        batch_scalar_tensors_to_cpu(infos, info_key=self._info_key)
        return observations, rewards, terminated, truncated, infos
