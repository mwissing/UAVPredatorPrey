from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (
    ROOT
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
)
CFG_PATH = TASK_DIR / "uav_3v1_env_cfg.py"
INIT_PATH = TASK_DIR / "__init__.py"
CFG_CLASS = "Uav3v1SurvivalSoftOobTeammateVelRandomSpawnPhysics50HzEnvCfg"
TASK_ID = "3v1-survival-soft-oob-teammate-vel-random-spawn-50hz-physics-v0"


def _class_node(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment_value(class_node: ast.ClassDef, name: str) -> ast.expr:
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise AssertionError(f"Assignment {name!r} not found in {class_node.name}")


def _attribute_path(node: ast.expr) -> tuple[str, ...]:
    path: list[str] = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        path.append(node.id)
    return tuple(reversed(path))


def test_50hz_physics_task_preserves_50hz_action_period() -> None:
    module = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    class_node = _class_node(module, CFG_CLASS)

    decimation = _assignment_value(class_node, "decimation")
    assert isinstance(decimation, ast.Constant) and decimation.value == 1

    post_init = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    assignments = {
        _attribute_path(node.targets[0]): node.value
        for node in post_init.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    dt = assignments[("self", "sim", "dt")]
    assert isinstance(dt, ast.BinOp) and isinstance(dt.op, ast.Div)
    assert isinstance(dt.left, ast.Constant) and dt.left.value == 1
    assert isinstance(dt.right, ast.Constant) and dt.right.value == 50
    render_interval = assignments[("self", "sim", "render_interval")]
    assert _attribute_path(render_interval) == ("self", "decimation")


def test_50hz_physics_task_is_registered_separately() -> None:
    init_source = INIT_PATH.read_text(encoding="utf-8")

    assert TASK_ID in init_source
    assert CFG_CLASS in init_source
