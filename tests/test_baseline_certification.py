from __future__ import annotations

import json
from pathlib import Path

import pytest

from certify_baseline import (
    aggregate_metrics,
    build_compose_command,
    build_evaluate_command,
    certification_exit_code,
    compare_windows_reference,
    default_evaluation_root,
    default_isaaclab_launcher,
    load_pool,
    qualify_dirty_verdict,
    run_command,
    validate_evaluation_summary,
)


def _summary(*, episodes_completed: int = 512) -> dict:
    return {
        "task": "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0",
        "num_envs": 512,
        "episodes_requested": 512,
        "episodes_completed": episodes_completed,
        "episode_accounting": "balanced_per_env",
        "episode_metrics": {
            "Metrics/catch_rate": 0.677734,
            "Metrics/predator_oob_rate": 0.0,
            "Metrics/prey_oob_rate": 0.0,
            "Metrics/episode_length": 318.441,
            "Metrics/episode_closest_approach": 0.314487,
        },
        "step_diagnostics": {
            "Diagnostics/predator_closing_speed_mean": 0.269214,
            "Diagnostics/predator_lateral_relative_speed_mean": 4.021025,
            "Diagnostics/predator_speed_rms": 3.769160,
            "Diagnostics/predator_action_near_limit_fraction": 0.176806,
        },
    }


def test_evaluate_command_contains_reproducibility_arguments(tmp_path) -> None:
    command = build_evaluate_command(
        isaaclab=tmp_path / "isaaclab.sh",
        checkpoint=tmp_path / "agent.pt",
        output_json=tmp_path / "summary.json",
        task="test-task-v0",
        agent="test-agent-entry-point",
        algorithm="MAPPO",
        num_envs=512,
        episodes=512,
        seed=44,
        repo_root=tmp_path,
    )

    assert command[:3] == [
        str(tmp_path / "isaaclab.sh"),
        "-p",
        str(tmp_path / "scripts" / "skrl" / "evaluate.py"),
    ]
    assert "--headless" in command
    assert command[command.index("--task") + 1] == "test-task-v0"
    assert command[command.index("--agent") + 1] == "test-agent-entry-point"
    assert command[command.index("--num_envs") + 1] == "512"
    assert command[command.index("--episodes") + 1] == "512"
    assert command[command.index("--seed") + 1] == "44"
    assert command[command.index("--checkpoint") + 1] == str(tmp_path / "agent.pt")
    assert command[command.index("--json") + 1] == str(tmp_path / "summary.json")


def test_compose_command_selects_exactly_one_role(tmp_path) -> None:
    command = build_compose_command(
        isaaclab=tmp_path / "isaaclab.sh",
        base=tmp_path / "base.pt",
        output=tmp_path / "composed.pt",
        role="prey",
        role_checkpoint=tmp_path / "prey.pt",
        repo_root=tmp_path,
    )

    assert "--prey" in command
    assert "--predator" not in command
    assert command[command.index("--prey") + 1] == str(tmp_path / "prey.pt")
    with pytest.raises(ValueError, match="Unsupported checkpoint role"):
        build_compose_command(
            isaaclab=tmp_path / "isaaclab.sh",
            base=tmp_path / "base.pt",
            output=tmp_path / "composed.pt",
            role="observer",
            role_checkpoint=tmp_path / "observer.pt",
            repo_root=tmp_path,
        )


def test_aggregate_metrics_reports_sample_statistics() -> None:
    summaries = [
        {"episode_metrics": {"Metrics/catch_rate": 0.6, "Metrics/predator_oob_rate": 0.0}},
        {"episode_metrics": {"Metrics/catch_rate": 0.8, "Metrics/predator_oob_rate": 0.1}},
    ]

    aggregate = aggregate_metrics(summaries)["episode_metrics"]

    assert aggregate["Metrics/catch_rate"] == pytest.approx(
        {"count": 2, "mean": 0.7, "sample_sd": 2**0.5 / 10, "min": 0.6, "max": 0.8}
    )
    assert aggregate["Metrics/predator_oob_rate"]["mean"] == pytest.approx(0.05)


def test_load_pool_resolves_all_four_checkpoints_per_role(tmp_path) -> None:
    entries = {"predator": [], "prey": []}
    for role in entries:
        for index in range(4):
            checkpoint = tmp_path / "pool" / f"{role}_{index:02d}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.touch()
            entries[role].append(
                {"name": f"{role}-{index}", "checkpoint": str(checkpoint.relative_to(tmp_path))}
            )
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(entries), encoding="utf-8")

    pool = load_pool(pool_path, repo_root=tmp_path)

    assert [entry["index"] for entry in pool["predator"]] == [0, 1, 2, 3]
    assert [entry["name"] for entry in pool["prey"]] == ["prey-0", "prey-1", "prey-2", "prey-3"]
    assert all(entry["checkpoint"].is_absolute() for role in pool.values() for entry in role)


def test_run_command_dry_run_never_invokes_runner(tmp_path) -> None:
    def fail_runner(*_args, **_kwargs):
        raise AssertionError("runner must not be called during a dry run")

    run_command(["isaaclab.sh", "-p", "evaluate.py"], cwd=tmp_path, dry_run=True, runner=fail_runner)


def test_default_evaluation_root_honors_artifacts_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))

    assert default_evaluation_root() == tmp_path / "isaac" / "evaluations"


def test_default_isaaclab_launcher_prefers_workspace_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    workspace = tmp_path / "workspace" / "isaaclab"
    workspace.mkdir(parents=True)
    launcher = workspace / "isaaclab.sh"
    launcher.touch()

    resolved = default_isaaclab_launcher(
        platform="linux",
        repo_root=tmp_path / "project" / "UAVPredatorPrey",
        workspace_root=workspace,
    )

    assert resolved == launcher


def test_default_isaaclab_launcher_returns_sibling_fallback_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    repo_root = tmp_path / "RL" / "UAVPredatorPrey"

    resolved = default_isaaclab_launcher(
        platform="linux",
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace" / "isaaclab",
    )

    assert resolved == tmp_path / "RL" / "IsaacLab" / "isaaclab.sh"


def test_validate_evaluation_summary_rejects_partial_episode_collection() -> None:
    with pytest.raises(RuntimeError, match="episodes_completed=511"):
        validate_evaluation_summary(
            _summary(episodes_completed=511),
            expected_task="3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0",
            expected_num_envs=512,
            expected_episodes=512,
        )


def test_validate_evaluation_summary_rejects_non_finite_required_metric() -> None:
    summary = _summary()
    summary["episode_metrics"]["Metrics/catch_rate"] = float("nan")

    with pytest.raises(RuntimeError, match="catch_rate"):
        validate_evaluation_summary(
            summary,
            expected_task="3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0",
            expected_num_envs=512,
            expected_episodes=512,
        )


def test_windows_reference_verdict_passes_matching_seed_42() -> None:
    summary = _summary()
    summary["seed"] = 42

    verdict = compare_windows_reference(
        [summary],
        task=summary["task"],
        num_envs=512,
        episodes=512,
        dry_run=False,
    )

    assert verdict["status"] == "pass"
    assert all(item["within_tolerance"] for item in verdict["comparisons"])


def test_windows_reference_verdict_fails_large_behavior_shift() -> None:
    summary = _summary()
    summary["seed"] = 42
    summary["episode_metrics"]["Metrics/catch_rate"] = 0.40

    verdict = compare_windows_reference(
        [summary],
        task=summary["task"],
        num_envs=512,
        episodes=512,
        dry_run=False,
    )

    assert verdict["status"] == "fail"
    assert not next(
        item["within_tolerance"]
        for item in verdict["comparisons"]
        if item["metric"] == "Metrics/catch_rate"
    )


def test_dirty_tree_pass_is_labeled_diagnostic() -> None:
    verdict = qualify_dirty_verdict(
        {"status": "pass", "reason": "Metrics match."},
        git_status=" M source/task.py",
        dry_run=False,
    )

    assert verdict["status"] == "diagnostic_pass"
    assert verdict["behavioral_status"] == "pass"
    assert verdict["certifiable"] is False
    assert verdict["reason"].startswith("DIRTY-TREE DIAGNOSTIC ONLY")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("pass", 0),
        ("diagnostic_pass", 0),
        ("not_run", 0),
        ("fail", 1),
        ("diagnostic_fail", 1),
        ("not_comparable", 1),
        ("diagnostic_not_comparable", 1),
    ],
)
def test_certification_exit_code(status, expected) -> None:
    assert certification_exit_code(status) == expected
