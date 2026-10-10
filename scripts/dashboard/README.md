# Project dashboard

A local web page for the UAV predator-prey project: the story for readers (pitch, journey, key results, lessons,
reproduction recipe, videos, 3D replays) and the lab view for day-to-day work (live training status, experiment
cards, decision rules, eval curves).

```
python3 scripts/dashboard/dashboard.py serve   # http://127.0.0.1:8765/dashboard/
python3 scripts/dashboard/dashboard.py build   # writes $HOME/RL/artifacts/dashboard/index.html once
```

Host Python, standard library only. `serve` binds to localhost (`--host 0.0.0.0` makes all of `artifacts/` readable
from the LAN) and rebuilds the page on every load. Restart it after editing `dashboard.py`; the template, the content
file and `replay.js` are read again on every load.

## Where the page comes from

| Generated on every load | Curated by hand in `content.toml` |
|---|---|
| Experiment cards from `docs/experiment_log.md` and `artifacts/isaac/experiments/*` | Pitch and milestone tiles |
| Live status: progress from tqdm console logs, GPU, git | Journey stages, lessons, recipe, next steps |
| Eval curves and "how episodes end" bars from `curve/` and `evals/` JSONs | Featured charts and videos |
| Decision rules from each experiment's `decision.json` (below) | Replay sources (`[[replay]]`) |

While the page is open it polls `/dashboard/status.json` every 15 s and swaps in what changes during training.
The 3D replay (`replay.js`, Three.js vendored in `vendor/three/`) reads episodes that `dashboard.py` extracts from
trajectory `.npz` files.

## decision.json: the live decision tracker

A pre-registered experiment shows its decision rule as a checklist in its card and, while it runs, in "Training
now". The rule stays in the experiment's own summarize script, the one that makes the official decision; that
script also writes `decision.json` into the experiment directory with `decision.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(EXP, "..", "..", "..", "UAVPredatorPrey", "scripts", "dashboard"))
from decision import criterion, write_decision   # EXP = artifacts/isaac/experiments

write_decision(HERE, [
    criterion("crash hazard", "<= 0.174/s (G4's end)", f"{crash:.3f}/s", crash <= 0.174,
              history=[(updates, value), ...], threshold=0.174),
    criterion("median speed", ">= 4.0 m/s", "not measured yet", None),   # None: not decidable yet
], checkpoint=3840, of=9600, final=False, verdict="perception gains are slow or plateau; next ...")
```

- Call the summarize script after every checkpoint evaluation, not only at the end, and compute each criterion on
  the checkpoints evaluated so far. Set `final=True` once the last checkpoint is in: the dashboard then shows
  "Decision" instead of "If it ended now" and no longer counts the directory as running.
- `history` and `threshold` are optional; they draw the trend line next to each criterion.
- Keep the printed summary unchanged, so `summary.txt` stays the official record.

`2026-10-10_prey_ghost_stage_v5_slow_continued/summarize5.py` is the first example (added after its run; its
printed output is byte-identical to `summary.txt`).
