"""A prey crash costs more than being caught in the obstacle tasks, and only there (user decision 2026-10-10).

prey_oob_penalty covers both leaving the arena and touching a pillar. Every obstacle-free task must keep the legacy
-200, which the anchor and reference reproductions rely on. The config module needs the Isaac app to import, so the
test reads it with ast and follows the in-module inheritance itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "uav_3v1_env_cfg.py"
)


def _resolved_fields() -> dict[str, dict[str, object]]:
    """Class name -> its literal class attributes, including the inherited ones."""
    classes = {node.name: node for node in ast.parse(CFG_PATH.read_text()).body if isinstance(node, ast.ClassDef)}

    def own(node: ast.ClassDef) -> dict[str, object]:
        values = {}
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target, value = statement.target, statement.value
            else:
                continue
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(value)
                except ValueError:  # not a literal, e.g. 90 + 3 * 4 * 4
                    pass
        return values

    def resolve(name: str) -> dict[str, object]:
        node = classes[name]
        base = next((b.id for b in node.bases if isinstance(b, ast.Name) and b.id in classes), None)
        return {**(resolve(base) if base else {}), **own(node)}

    return {name: resolve(name) for name in classes}


def test_only_obstacle_tasks_make_a_prey_crash_cost_more_than_a_catch() -> None:
    fields = {name: f for name, f in _resolved_fields().items() if "prey_oob_penalty" in f}
    obstacle_tasks = {name for name, f in fields.items() if f.get("num_obstacles", 0) > 0}
    assert obstacle_tasks == {
        "Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstaclesEnvCfg",
        "Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstacles8EnvCfg",
    }
    for name, f in fields.items():
        if name in obstacle_tasks:
            assert (f["prey_oob_penalty"], f["prey_caught_penalty"]) == (-300.0, -200.0), name
        else:
            assert f["prey_oob_penalty"] == -200.0, name
