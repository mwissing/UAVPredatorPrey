from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_SKRL = ROOT / "scripts" / "skrl"
AGENTS = (
    ROOT
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "agents"
)

for path in (SCRIPTS_SKRL, AGENTS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
