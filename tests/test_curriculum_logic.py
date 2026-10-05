from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hysteresis_curriculum as curriculum
from hysteresis_curriculum import (
    POOL_TYPE_ELITE,
    POOL_TYPE_RECENT,
    _decide_phase,
    _default_isaaclab_launcher,
    _enforce_catch_rate_band,
    _maybe_promote_to_pool,
    _pfsp_weight,
    _pool_entry_weight,
    _prepare_output_dir,
    _promotion_failure_override,
    _prune_pool,
    _training_override_args,
    _write_run_config,
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


def test_prepare_output_dir_requires_a_new_or_empty_directory(tmp_path) -> None:
    new_dir = tmp_path / "new-run"
    _prepare_output_dir(new_dir)
    assert new_dir.is_dir()

    (new_dir / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not resume"):
        _prepare_output_dir(new_dir)


def test_run_config_records_invocation_resolved_args_and_input_hashes(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "curriculum"
    output_dir.mkdir()
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.write_bytes(b"checkpoint")
    pool_checkpoint = tmp_path / "prey_00.pt"
    pool_checkpoint.write_bytes(b"pool checkpoint")
    args = SimpleNamespace(
        output_dir=output_dir,
        checkpoint=checkpoint,
        predator_source_checkpoint=None,
        prey_source_checkpoint=None,
        opponent_pool=None,
        total_iterations=100,
        task="test-task-v0",
    )

    git_values = {
        ("rev-parse", "HEAD"): "abc123",
        ("branch", "--show-current"): "test-branch",
        ("status", "--porcelain"): "",
    }
    monkeypatch.setattr(curriculum, "_git_text", lambda *items: git_values[items])
    monkeypatch.setattr(curriculum.sys, "argv", ["hysteresis_curriculum.py", "--total-iterations", "100"])
    monkeypatch.setenv("ISAACLAB_COMMIT", "f4aa17f")

    path = _write_run_config(
        args,
        pool={
            "predator": [],
            "prey": [{"name": "prey-0", "checkpoint": str(pool_checkpoint)}],
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["argv"][-2:] == ["--total-iterations", "100"]
    assert payload["resolved_args"]["total_iterations"] == 100
    assert payload["resolved_args"]["checkpoint"] == str(checkpoint)
    assert payload["git"] == {"commit": "abc123", "branch": "test-branch", "status": ""}
    assert payload["runtime"]["isaac_lab_commit"] == "f4aa17f"
    assert len(payload["inputs"]["checkpoint"]["sha256"]) == 64
    assert payload["inputs"]["pool_checkpoints"][0]["name"] == "prey-0"
    assert len(payload["inputs"]["pool_checkpoints"][0]["sha256"]) == 64


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


def _band_args(**overrides):
    values = {
        "low": 0.40,
        "high": 0.60,
        "balanced_action": "both",
        "per_env_pool_prob": 0.5,
        "max_predator_oob": 0.12,
        "max_prey_oob": 0.12,
        "max_predator_soft": 0.3,
        "max_prey_soft": 0.3,
        "phase_decision_cross_play": True,
        "phase_decision_cross_play_min_count": 2,
        "auto_pool_predator_min_cross_play_avg_catch": 0.55,
        "auto_pool_predator_min_cross_play_min_catch": 0.25,
        "auto_pool_prey_max_cross_play_avg_catch": 0.5,
        "auto_pool_prey_max_cross_play_max_catch": 0.8,
        "promotion_failure_phase_continuation": True,
        "promotion_failure_plateau_delta": 0.03,
        "promotion_failure_max_stale_repeats": 2,
        "promotion_failure_plateau_action": "both",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _catch_summary(catch_rate: float) -> dict:
    return {"episode_metrics": {"Metrics/catch_rate": catch_rate}}


@pytest.mark.parametrize("requested", ["both", "prey"])
def test_band_guard_trains_only_the_predator_below_the_band(requested: str) -> None:
    phase, reason, pool_prob = _enforce_catch_rate_band(
        requested, "requested", 0.25, _catch_summary(0.371), _band_args()
    )

    assert phase == "predator"
    assert reason.startswith("requested; band guard")
    assert pool_prob == 0.5


@pytest.mark.parametrize("requested", ["both", "predator"])
def test_band_guard_trains_only_the_prey_above_the_band(requested: str) -> None:
    phase, _, pool_prob = _enforce_catch_rate_band(requested, "requested", 0.75, _catch_summary(0.887), _band_args())

    assert phase == "prey"
    assert pool_prob == 0.5


@pytest.mark.parametrize(
    ("requested", "catch_rate"),
    [("both", 0.5), ("predator", 0.5), ("prey", 0.5), ("stop", 0.5), ("predator", 0.2), ("prey", 0.8)],
)
def test_band_guard_keeps_balanced_and_losing_role_decisions(requested: str, catch_rate: float) -> None:
    result = _enforce_catch_rate_band(requested, "requested", 0.25, _catch_summary(catch_rate), _band_args())

    assert result == (requested, "requested", 0.25)


def test_plateau_override_below_the_band_keeps_training_the_predator() -> None:
    # Mirrors the July 19 run: the second failed predator promotion without
    # beating 0.570 requested a joint phase at catch rate 0.371.
    args = _band_args()
    override = _promotion_failure_override(
        phase="predator",
        summary=_catch_summary(0.371),
        promotion={"promoted": False},
        cross_play_aggregate=None,
        retry_state={"predator": {"best_score": 0.570, "stale_repeats": 1}},
        args=args,
    )
    assert override is not None and override["plateau"] and override["phase"] == "both"

    phase, _, _ = _enforce_catch_rate_band(
        override["phase"], override["reason"], override["per_env_pool_prob"], _catch_summary(0.371), args
    )

    assert phase == "predator"


def test_band_guard_resets_pool_exposure_chosen_for_the_other_role() -> None:
    # 2026-10-05: a latest-focused prey repeat (pool prob 0.25, from the prey's
    # cross-play robustness) became a predator phase at catch rate 0.285.
    args = _band_args()
    override = _promotion_failure_override(
        phase="prey",
        summary=_catch_summary(0.285),
        promotion={"promoted": False},
        cross_play_aggregate={"count": 2, "catch_avg": 0.201, "catch_min": 0.176, "catch_max": 0.227},
        retry_state={},
        args=_band_args(latest_focus_per_env_pool_prob=0.25, pool_focus_per_env_pool_prob=0.75),
    )
    assert override is not None and override["phase"] == "prey" and override["per_env_pool_prob"] == 0.25

    phase, _, pool_prob = _enforce_catch_rate_band(
        override["phase"], override["reason"], override["per_env_pool_prob"], _catch_summary(0.285), args
    )

    assert phase == "predator"
    assert pool_prob == args.per_env_pool_prob


def test_pool_robust_joint_decision_below_the_band_keeps_training_the_predator() -> None:
    args = _band_args()
    summary = _catch_summary(0.121)
    cross_play = {
        "predator": {"count": 3, "catch_avg": 0.8, "catch_min": 0.6, "catch_max": 1.0},
        "prey": {"count": 3, "catch_avg": 0.3, "catch_min": 0.1, "catch_max": 0.5},
    }
    phase, reason = _decide_phase(summary, args, cross_play)
    assert phase == "both"

    assert _enforce_catch_rate_band(phase, reason, 0.5, summary, args)[0] == "predator"


def test_training_overrides_are_forwarded_as_named_train_arguments() -> None:
    args = SimpleNamespace(learning_rate=1.0e-5, kl_threshold=0.05, mixed_precision=True)

    assert _training_override_args(args) == [
        "--learning-rate",
        "1e-05",
        "--kl-threshold",
        "0.05",
        "+agent.agent.mixed_precision=true",
    ]


def test_training_overrides_are_omitted_by_default() -> None:
    args = SimpleNamespace(learning_rate=None, kl_threshold=None, mixed_precision=False)

    assert _training_override_args(args) == []


def test_train_phase_forwards_overrides_before_greedy_freeze_argument(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "agent_100.pt"
    checkpoint.touch()
    run_root = tmp_path / "missing-run-root"
    captured_commands = []

    def capture_command(command, **_kwargs) -> None:
        captured_commands.append(command)

    monkeypatch.setattr(curriculum, "_run", capture_command)
    args = SimpleNamespace(
        run_root=run_root,
        isaaclab=tmp_path / "isaaclab.bat",
        task="test-task-v0",
        agent="test-agent-entry-point",
        algorithm="MAPPO",
        train_num_envs=8,
        seed=42,
        learning_rate=1.0e-5,
        kl_threshold=0.05,
        mixed_precision=True,
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
    assert run_root.is_dir()
    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command[command.index("--learning-rate") + 1] == "1e-05"
    assert command[command.index("--kl-threshold") + 1] == "0.05"
    assert "+agent.agent.mixed_precision=true" in command
    assert command.index("--learning-rate") < command.index("--freeze-agents")
    assert command.index("--kl-threshold") < command.index("--freeze-agents")
    assert command.index("+agent.agent.mixed_precision=true") < command.index("--freeze-agents")


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
