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
            # Groups with an lr_multiplier (e.g. a new obstacle branch) keep their ratio to the base rate.
            group["lr"] = learning_rate * float(group.get("lr_multiplier", 1.0))

        if isinstance(internal_learning_rates, dict) and role in internal_learning_rates:
            internal_learning_rates[role] = learning_rate

        scheduler = schedulers.get(role) if isinstance(schedulers, Mapping) else None
        if scheduler is not None:
            group_lrs = [group["lr"] for group in param_groups]
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = list(group_lrs)
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = list(group_lrs)

        updated_roles.append(str(role))

    if not updated_roles:
        raise RuntimeError("Agent optimizers contain no parameter groups to update")
    return updated_roles
