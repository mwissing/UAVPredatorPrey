"""Project-specific Isaac Sim startup hardening."""

from __future__ import annotations

import re


_TELEMETRY_EXTENSION = "omni.kit.telemetry"
_EXCLUSION_SETTING_RE = re.compile(r"^--/app/extensions/excluded/(\d+)=(.+)$")


def exclude_isaac_telemetry(kit_args: str) -> str:
    """Exclude the optional telemetry extension without replacing other Kit arguments.

    Isaac Sim 5.1 starts telemetry system-information collection concurrently.
    On this setup that native startup path intermittently segfaults before the
    simulation is created. Telemetry is not required for training, evaluation,
    or video recording.
    """

    tokens = kit_args.split()
    used_indices: set[int] = set()
    for token in tokens:
        if token.startswith("--/app/extensions/excluded") and _TELEMETRY_EXTENSION in token:
            return kit_args
        match = _EXCLUSION_SETTING_RE.match(token)
        if match is not None:
            used_indices.add(int(match.group(1)))

    exclusion_index = next(index for index in range(len(used_indices) + 1) if index not in used_indices)
    tokens.append(f"--/app/extensions/excluded/{exclusion_index}={_TELEMETRY_EXTENSION}")
    return " ".join(tokens)
