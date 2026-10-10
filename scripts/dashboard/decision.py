"""Write an experiment's decision.json, which the dashboard shows as a live checklist of its pre-registered rule.

The rule itself stays in the experiment's summarize script (one implementation, the one that makes the decision);
this module only writes its result in a fixed shape. Call it from that script after every checkpoint evaluation,
not only at the end, so the dashboard can show during the run whether the rule is on track:

    import os, sys
    sys.path.insert(0, os.path.join(EXP, "..", "..", "..", "UAVPredatorPrey", "scripts", "dashboard"))
    from decision import criterion, write_decision   # EXP = artifacts/isaac/experiments

    write_decision(HERE, [
        criterion("crash hazard", "<= 0.174/s (G4's end)", f"{crash:.3f}/s", crash <= 0.174,
                  history=[(updates, value), ...], threshold=0.174),
        ...
    ], checkpoint=7680, of=9600, final=False, verdict="perception gains are slow or plateau; next ...")

Compute each criterion on the checkpoints evaluated so far and set final=True once the last one is in. A criterion
that cannot be evaluated yet gets passed=None. Standard library only (host Python).
"""

import json
import math
import os
import time

FORMAT = 1


def criterion(name: str, rule: str, value: str, passed: bool | None, *,
              history: list[tuple[float, float]] | None = None, threshold: float | None = None) -> dict:
    """One pre-registered criterion: its rule in words, the current value as text, and pass/fail (None: not yet).

    history (checkpoint, value) and threshold are optional and draw a small trend line with the bar."""
    item = {"name": name, "rule": rule, "value": value, "pass": None if passed is None else bool(passed)}
    if history:
        item["history"] = [[float(x), float(v)] for x, v in history if v is not None and math.isfinite(v)]
    if threshold is not None:
        item["threshold"] = float(threshold)
    return item


def write_decision(exp_dir: str, criteria: list[dict], *, checkpoint: float | None = None, of: float | None = None,
                   unit: str = "updates", verdict: str | None = None, final: bool = False) -> str:
    """Write <exp_dir>/decision.json atomically, so the dashboard never reads half a file."""
    data = {
        "format": FORMAT,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final": bool(final),
        "checkpoint": checkpoint, "of": of, "unit": unit,
        "criteria": criteria,
        "verdict": verdict,
    }
    path = os.path.join(exp_dir, "decision.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    return path
