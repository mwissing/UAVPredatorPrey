"""Create a checkpoint compatible with the prey-attention actor config.

The old attention-critic checkpoint stores the prey policy as a flat MLP. The
new prey-attention actor has different parameter names and shapes, so skrl's
strict module loading cannot load that prey policy directly.

This utility keeps compatible modules and intentionally resets:
- prey policy
- all optimizers

Missing modules are skipped by skrl's multi-agent loader, so the new prey policy
is initialized from the YAML/model seed when training starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _migrate_checkpoint(source: Path) -> dict[str, dict[str, Any]]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")

    migrated: dict[str, dict[str, Any]] = {}
    for role, modules in checkpoint.items():
        if not isinstance(modules, dict):
            continue

        migrated[role] = {}
        for name, state in modules.items():
            if name == "optimizer":
                continue
            if role == "prey" and name == "policy":
                continue
            migrated[role][name] = state

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Old attention-critic checkpoint.")
    parser.add_argument("--out", type=Path, required=True, help="Output checkpoint path.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    migrated = _migrate_checkpoint(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, out)

    print(f"[INFO] Wrote prey-attention migration checkpoint: {out}")
    for role, modules in migrated.items():
        print(f"[INFO] {role}: {', '.join(sorted(modules.keys()))}")


if __name__ == "__main__":
    main()
