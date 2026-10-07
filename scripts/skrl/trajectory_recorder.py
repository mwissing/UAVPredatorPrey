"""Record each environment's first episode for offline failure analysis.

Evaluation counts exactly one episode per environment when episodes equal
environments, so the first episode of every environment is the evaluated
sample. Frames are snapshots taken after reset and after every step; a frame
row is marked inactive once that environment's first episode has ended,
because auto-reset already moved it into the next episode.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


class FirstEpisodeRecorder:
    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self._num_envs = int(num_envs)
        self._active = torch.ones(self._num_envs, dtype=torch.bool, device=device)
        self._frames: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._active_history: list[torch.Tensor] = []
        self._outcomes: dict[str, torch.Tensor] = {}

    @property
    def complete(self) -> bool:
        return not bool(self._active.any())

    def record(self, frame: Mapping[str, torch.Tensor]) -> None:
        """Store one state snapshot (tensors with leading dimension num_envs)."""
        for name, value in frame.items():
            self._frames[name].append(value.detach().to(torch.float32).clone())
        self._active_history.append(self._active.clone())

    def finish(self, env_ids: torch.Tensor, payload: Mapping[str, torch.Tensor]) -> None:
        """Close the first episode of env_ids and keep their per-episode payload values."""
        env_ids = env_ids.to(self._active.device, dtype=torch.long).reshape(-1)
        first = self._active[env_ids]
        if not bool(first.any()):
            return
        for name, value in payload.items():
            if name == "env_ids" or not isinstance(value, torch.Tensor):
                continue
            value = value.to(self._active.device, dtype=torch.float32).reshape(-1)
            if value.numel() != env_ids.numel():
                continue
            store = self._outcomes.setdefault(
                name, torch.full((self._num_envs,), float("nan"), device=self._active.device)
            )
            store[env_ids[first]] = value[first]
        self._active[env_ids[first]] = False

    def save(self, path: str | Path) -> None:
        arrays = {name: torch.stack(values).cpu().numpy() for name, values in self._frames.items()}
        arrays["active"] = torch.stack(self._active_history).cpu().numpy()
        arrays.update({f"outcome_{name}": value.cpu().numpy() for name, value in self._outcomes.items()})
        np.savez_compressed(path, **arrays)
