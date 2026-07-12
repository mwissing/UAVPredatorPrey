from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hysteresis_curriculum as curriculum
from hysteresis_curriculum import (
    POOL_TYPE_ELITE,
    POOL_TYPE_RECENT,
    _default_isaaclab_launcher,
    _maybe_promote_to_pool,
    _pfsp_weight,
    _pool_entry_weight,
    _prune_pool,
    _training_override_args,
)


def test_default_isaaclab_launcher_uses_linux_sibling_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    repo_root = tmp_path / "UAVPredatorPrey"
    launcher = tmp_path / "IsaacLab" / "isaaclab.sh"
    launcher.parent.mkdir()
    launcher.touch()

    assert _default_isaaclab_launcher(
        platform="linux",
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace" / "isaaclab",
    ) == launcher


def test_default_isaaclab_launcher_uses_container_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    workspace_root = tmp_path / "workspace" / "isaaclab"
    launcher = workspace_root / "isaaclab.sh"
    workspace_root.mkdir(parents=True)
    launcher.touch()

    assert _default_isaaclab_launcher(
        platform="linux",
        repo_root=tmp_path / "repo" / "UAVPredatorPrey",
        workspace_root=workspace_root,
    ) == launcher


def test_default_isaaclab_launcher_honors_root_override(tmp_path, monkeypatch) -> None:
    isaaclab_root = tmp_path / "custom-isaaclab"
    monkeypatch.setenv("ISAACLAB_ROOT", str(isaaclab_root))

    assert _default_isaaclab_launcher(platform="linux") == isaaclab_root / "isaaclab.sh"
    assert _default_isaaclab_launcher(platform="win32") == isaaclab_root / "isaaclab.bat"


def test_default_isaaclab_launcher_keeps_windows_default(monkeypatch) -> None:
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)

    assert _default_isaaclab_launcher(platform="win32") == (
        Path(r"C:\RL\IsaacLab") / "isaaclab.bat"
    )


def _args(**overrides):
    values = {
        "pool_sampling": "pfsp",
        "pfsp_min_weight": 0.05,
        "pfsp_weighting": "squared",
        "auto_pool": True,
        "auto_pool_predator_min_catch": 0.6,
        "auto_pool_prey_max_catch": 0.4,
        "auto_pool_max_predator_oob": 0.12,
        "auto_pool_max_prey_oob": 0.12,
        "auto_pool_max_predator_soft": 0.2,
        "auto_pool_max_prey_soft": 0.2,
        "auto_pool_max_opponent_oob": 0.12,
        "auto_pool_max_opponent_soft": 0.2,
        "auto_pool_predator_min_cross_play_avg_catch": 0.55,
        "auto_pool_predator_min_cross_play_min_catch": 0.35,
        "auto_pool_prey_max_cross_play_avg_catch": 0.45,
        "auto_pool_prey_max_cross_play_max_catch": 0.65,
        "auto_pool_max_entries_per_role": 2,
        "recent_pool_max_entries_per_role": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("weighting", "expected"),
    [("linear", 0.25), ("squared", 0.0625), ("variance", 0.1875)],
)
def test_pfsp_weighting_functions(weighting: str, expected: float) -> None:
    assert _pfsp_weight(0.75, weighting) == pytest.approx(expected)


def test_pfsp_uses_training_role_win_rate_and_minimum_weight() -> None:
    prey_entry = {"base_weight": 2.0, "metrics": {"Metrics/catch_rate": 0.8}}
    predator_entry = {"base_weight": 2.0, "metrics": {"Metrics/catch_rate": 0.2}}
    args = _args()

    predator_training_weight = _pool_entry_weight(
        prey_entry,
        sampled_agent="prey",
        training_agent="predator",
        args=args,
    )
    prey_training_weight = _pool_entry_weight(
        predator_entry,
        sampled_agent="predator",
        training_agent="prey",
        args=args,
    )

    assert predator_training_weight == pytest.approx(0.1)
    assert prey_training_weight == pytest.approx(0.1)


def test_predator_promotion_requires_latest_and_cross_play_robustness(tmp_path) -> None:
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.touch()
    summary = {
        "episode_metrics": {
            "Metrics/catch_rate": 0.9,
            "Metrics/predator_oob_rate": 0.01,
            "Metrics/prey_oob_rate": 0.01,
            "Metrics/predator_soft_arena_outside": 0.02,
            "Metrics/prey_soft_arena_outside": 0.02,
        }
    }
    pool = {"predator": [], "prey": []}

    rejected = _maybe_promote_to_pool(
        pool,
        agent="predator",
        checkpoint=checkpoint,
        summary=summary,
        cross_play_aggregate={"catch_avg": 0.5, "catch_min": 0.2},
        phase_index=3,
        args=_args(),
    )
    accepted = _maybe_promote_to_pool(
        pool,
        agent="predator",
        checkpoint=checkpoint,
        summary=summary,
        cross_play_aggregate={"catch_avg": 0.8, "catch_min": 0.7},
        phase_index=4,
        args=_args(),
    )

    assert rejected is not None and not rejected["promoted"]
    assert accepted is not None and accepted["promoted"]
    assert len(pool["predator"]) == 1
    assert pool["predator"][0]["pool_type"] == POOL_TYPE_ELITE


def test_pruning_keeps_best_elites_and_newest_recent_entry() -> None:
    pool = {
        "predator": [
            {"name": "elite-low", "pool_type": POOL_TYPE_ELITE, "auto_score": 0.1, "added_phase_index": 9},
            {"name": "elite-high", "pool_type": POOL_TYPE_ELITE, "auto_score": 0.9, "added_phase_index": 1},
            {"name": "elite-mid", "pool_type": POOL_TYPE_ELITE, "auto_score": 0.5, "added_phase_index": 2},
            {"name": "recent-old", "pool_type": POOL_TYPE_RECENT, "added_phase_index": 7},
            {"name": "recent-new", "pool_type": POOL_TYPE_RECENT, "added_phase_index": 8},
        ],
        "prey": [],
    }

    _prune_pool(pool, _args())

    assert {entry["name"] for entry in pool["predator"]} == {
        "elite-high",
        "elite-mid",
        "recent-new",
    }


def test_training_overrides_are_forwarded_as_named_train_arguments() -> None:
    args = SimpleNamespace(learning_rate=1.0e-5, kl_threshold=0.05)

    assert _training_override_args(args) == [
        "--learning-rate",
        "1e-05",
        "--kl-threshold",
        "0.05",
    ]


def test_training_overrides_are_omitted_by_default() -> None:
    args = SimpleNamespace(learning_rate=None, kl_threshold=None)

    assert _training_override_args(args) == []


def test_train_phase_forwards_overrides_before_greedy_freeze_argument(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.touch()
    captured_commands = []

    def capture_command(command, **_kwargs) -> None:
        captured_commands.append(command)

    monkeypatch.setattr(curriculum, "_run", capture_command)
    args = SimpleNamespace(
        run_root=tmp_path,
        isaaclab=tmp_path / "isaaclab.bat",
        task="test-task-v0",
        agent="test-agent-entry-point",
        algorithm="MAPPO",
        train_num_envs=8,
        seed=42,
        learning_rate=1.0e-5,
        kl_threshold=0.05,
        per_env_pool_prob=0.0,
        auto_pool_path=None,
        opponent_pool=None,
        per_env_pool_max_policies=1,
        per_env_pool_seed=None,
        dry_run=True,
    )

    result = curriculum._train_phase(
        checkpoint,
        phase="predator",
        phase_iterations=100,
        args=args,
    )

    assert result == checkpoint
    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command[command.index("--learning-rate") + 1] == "1e-05"
    assert command[command.index("--kl-threshold") + 1] == "0.05"
    assert command.index("--learning-rate") < command.index("--freeze-agents")
    assert command.index("--kl-threshold") < command.index("--freeze-agents")


def test_phase_video_forwards_clarity_preset(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.touch()
    captured_commands = []

    def capture_command(command, **_kwargs) -> None:
        captured_commands.append(command)

    monkeypatch.setattr(curriculum, "_run", capture_command)
    args = SimpleNamespace(
        record_phase_video=True,
        phase_video_every=1,
        phase_video_length=500,
        phase_video_width=1920,
        phase_video_height=1080,
        phase_video_bitrate="10M",
        phase_video_antialiasing="FXAA",
        phase_video_markers=True,
        phase_video_marker_radius=0.10,
        phase_video_marker_height=0.16,
        phase_video_scene_overlay=True,
        phase_video_num_envs=1,
        phase_video_camera_eye=(8.0, -8.0, 12.0),
        phase_video_camera_target=(0.0, 0.0, 6.0),
        phase_video_camera_env_index=0,
        output_dir=tmp_path,
        isaaclab=tmp_path / "isaaclab.bat",
        task="test-task-v0",
        agent="test-agent-entry-point",
        algorithm="MAPPO",
        seed=42,
        dry_run=True,
    )

    curriculum._record_phase_video(
        checkpoint,
        phase_index=1,
        args=args,
        label="after",
    )

    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command[command.index("--video-width") + 1] == "1920"
    assert command[command.index("--video-height") + 1] == "1080"
    assert command[command.index("--video-bitrate") + 1] == "10M"
    assert command[command.index("--video-antialiasing") + 1] == "FXAA"
    assert command[command.index("--video-marker-radius") + 1] == "0.1"
    assert command[command.index("--video-marker-height") + 1] == "0.16"
    assert "--video-markers" in command
    assert "--no-video-markers" not in command
    assert "--no-video-scene-overlay" not in command


def test_phase_video_can_disable_visual_overlays(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.touch()
    captured_commands = []
    monkeypatch.setattr(curriculum, "_run", lambda command, **_kwargs: captured_commands.append(command))
    args = SimpleNamespace(
        record_phase_video=True,
        phase_video_every=1,
        phase_video_length=10,
        phase_video_width=1280,
        phase_video_height=720,
        phase_video_bitrate="4M",
        phase_video_antialiasing="FXAA",
        phase_video_markers=False,
        phase_video_marker_radius=0.10,
        phase_video_marker_height=0.16,
        phase_video_scene_overlay=False,
        phase_video_num_envs=1,
        phase_video_camera_eye=(8.0, -8.0, 12.0),
        phase_video_camera_target=(0.0, 0.0, 6.0),
        phase_video_camera_env_index=0,
        output_dir=tmp_path,
        isaaclab=tmp_path / "isaaclab.bat",
        task="test-task-v0",
        agent="test-agent-entry-point",
        algorithm="MAPPO",
        seed=42,
        dry_run=True,
    )

    curriculum._record_phase_video(
        checkpoint,
        phase_index=1,
        args=args,
        label="after",
    )

    assert "--no-video-markers" in captured_commands[0]
    assert "--no-video-scene-overlay" in captured_commands[0]
