"""Configure exact frozen-role semantics for SKRL multi-agent training."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import torch


class FrozenAgentOptimizer(torch.optim.Optimizer):
    """No-op optimizer retained for SKRL bookkeeping and checkpoint safety."""

    def __init__(self, params: Iterable[torch.nn.Parameter]):
        super().__init__(list(params), defaults={})

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                return closure()
        return None

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for param in group["params"]:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()


def agent_parameters(agent: Any, agent_name: str) -> list[torch.nn.Parameter]:
    """Return unique policy/value parameters for one SKRL multi-agent uid."""
    modules = []
    for attr_name in ("policies", "values"):
        mapping = getattr(agent, attr_name, {})
        if isinstance(mapping, dict):
            module = mapping.get(agent_name)
            if module is not None and module not in modules:
                modules.append(module)

    if not modules:
        models = getattr(agent, "models", {})
        agent_models = models.get(agent_name, {}) if isinstance(models, dict) else {}
        if isinstance(agent_models, dict):
            for module in agent_models.values():
                if module is not None and module not in modules:
                    modules.append(module)

    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def freeze_agent_training(agent: Any, agent_names: set[str]) -> set[str]:
    """Disable all learning-state updates for selected multi-agent roles."""
    if not agent_names:
        return set()

    optimizers = getattr(agent, "optimizers", {})
    schedulers = getattr(agent, "schedulers", {})
    lr_schedulers_enabled = getattr(agent, "_learning_rate_scheduler", {})
    available_agents = set(optimizers.keys()) if isinstance(optimizers, dict) else set()
    frozen_agents = set()

    for agent_name in sorted(agent_names):
        if agent_name not in available_agents:
            print(
                f"[WARNING] Cannot freeze agent '{agent_name}': optimizer not found. "
                f"Available optimizer keys: {sorted(available_agents)}"
            )
            continue

        params = agent_parameters(agent, agent_name)
        if not params:
            print(f"[WARNING] Cannot freeze agent '{agent_name}': no policy/value parameters found.")
            continue

        for param in params:
            param.grad = None
            param.requires_grad_(False)

        old_optimizer = optimizers[agent_name]
        old_optimizer.state.clear()
        optimizers[agent_name] = FrozenAgentOptimizer(params)
        checkpoint_modules = getattr(agent, "checkpoint_modules", {})
        if isinstance(checkpoint_modules, dict) and agent_name in checkpoint_modules:
            checkpoint_modules[agent_name].pop("optimizer", None)
        print(
            f"[INFO] Agent '{agent_name}' optimizer replaced by no-op freeze optimizer; "
            "optimizer state will not be saved."
        )

        if isinstance(lr_schedulers_enabled, dict) and agent_name in lr_schedulers_enabled:
            lr_schedulers_enabled[agent_name] = None
            print(f"[INFO] Agent '{agent_name}' learning rate scheduler disabled.")
        elif isinstance(schedulers, dict) and agent_name in schedulers:
            print(f"[INFO] Agent '{agent_name}' scheduler left unused because the role is frozen.")
        frozen_agents.add(agent_name)

    set_frozen_agents = getattr(agent, "set_frozen_agents", None)
    if frozen_agents and not callable(set_frozen_agents):
        raise RuntimeError(
            "The selected SKRL agent does not support exact frozen-role updates. "
            "Refusing to continue with optimizer-only freezing."
        )
    if frozen_agents:
        set_frozen_agents(frozen_agents)
        trainable = sorted(set(getattr(agent, "possible_agents", ())) - frozen_agents)
        print(
            f"[INFO] Exact frozen-role update active: frozen={sorted(frozen_agents)}, "
            f"trainable={trainable}"
        )
    return frozen_agents


def role_state_fingerprint(agent: Any, agent_name: str) -> str:
    """Hash checkpoint-relevant model and preprocessor state for one role."""
    digest = hashlib.sha256()
    seen_modules = set()
    module_mappings = (
        ("policy", getattr(agent, "policies", {})),
        ("value", getattr(agent, "values", {})),
        ("state_preprocessor", getattr(agent, "_state_preprocessor", {})),
        ("shared_state_preprocessor", getattr(agent, "_shared_state_preprocessor", {})),
        ("value_preprocessor", getattr(agent, "_value_preprocessor", {})),
    )

    for label, mapping in module_mappings:
        module = mapping.get(agent_name) if isinstance(mapping, dict) else None
        if module is None or id(module) in seen_modules or not hasattr(module, "state_dict"):
            continue
        seen_modules.add(id(module))
        digest.update(label.encode("utf-8"))
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def capture_frozen_fingerprints(agent: Any, agent_names: set[str]) -> dict[str, str]:
    return {agent_name: role_state_fingerprint(agent, agent_name) for agent_name in sorted(agent_names)}


def verify_frozen_fingerprints(agent: Any, expected: dict[str, str]) -> None:
    changed = [
        agent_name
        for agent_name, fingerprint in expected.items()
        if role_state_fingerprint(agent, agent_name) != fingerprint
    ]
    if changed:
        raise RuntimeError(
            "Exact frozen-role verification failed; checkpoint-relevant state changed for: "
            + ", ".join(changed)
        )
    if expected:
        print(f"[INFO] Exact frozen-role verification passed: {sorted(expected)}")
