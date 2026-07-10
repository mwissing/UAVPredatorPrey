"""Post-checkpoint training overrides that preserve optimizer state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def override_optimizer_learning_rate(agent: Any, learning_rate: float) -> list[str]:
    """Set optimizer learning rates after checkpoint loading without resetting moments."""
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(f"Learning rate must be finite and positive, got {learning_rate}")

    optimizers = getattr(agent, "optimizers", None)
    if not isinstance(optimizers, Mapping) or not optimizers:
        raise RuntimeError("Agent does not expose role-wise optimizers for a learning-rate override")

    internal_learning_rates = getattr(agent, "_learning_rate", None)
    schedulers = getattr(agent, "schedulers", {})
    updated_roles = []

    for role, optimizer in optimizers.items():
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            continue
        for group in param_groups:
            group["lr"] = learning_rate

        if isinstance(internal_learning_rates, dict) and role in internal_learning_rates:
            internal_learning_rates[role] = learning_rate

        scheduler = schedulers.get(role) if isinstance(schedulers, Mapping) else None
        if scheduler is not None:
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [learning_rate] * len(param_groups)
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [learning_rate] * len(param_groups)

        updated_roles.append(str(role))

    if not updated_roles:
        raise RuntimeError("Agent optimizers contain no parameter groups to update")
    return updated_roles
