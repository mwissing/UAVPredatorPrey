"""Certify a recurrent 3v1 baseline across seeds and opponent-pool roles.

This is an evaluation-only orchestration tool. It runs deterministic anchor
evaluations, composes each pool role with the current opposite role, and writes
machine-readable plus human-readable reports. GPU jobs are always sequential.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / ".pretrained_checkpoints" / "linux_migration_2026-07-11" / "current" / "agent_115200.pt"
)
DEFAULT_POOL = (
    REPO_ROOT
    / ".pretrained_checkpoints"
    / "linux_migration_2026-07-11"
    / "pool"
    / "opponent_pool_portable.json"
)
DEFAULT_TASK = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-v0"
DEFAULT_AGENT = "skrl_mappo_attention_critic_prey_attention_large_gru_cfg_entry_point"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
REPORT_SECTIONS = ("episode_metrics", "step_rewards", "step_diagnostics")
WINDOWS_REFERENCE = {
    "source": "docs/linux_handoff_2026-07-11.md",
    "seed": 42,
    "task": DEFAULT_TASK,
    "num_envs": 512,
    "episodes": 512,
    # Tolerances flag a behavioral shift, not harmless floating-point drift.
    "metrics": {
        "episode_metrics": {
            "Metrics/catch_rate": {"value": 0.677734, "abs_tolerance": 0.05},
            "Metrics/predator_oob_rate": {"value": 0.0, "abs_tolerance": 0.02},
            "Metrics/prey_oob_rate": {"value": 0.0, "abs_tolerance": 0.02},
            "Metrics/episode_length": {"value": 318.441, "abs_tolerance": 40.0},
            "Metrics/episode_closest_approach": {"value": 0.314487, "abs_tolerance": 0.10},
        },
        "step_diagnostics": {
            "Diagnostics/predator_closing_speed_mean": {"value": 0.269214, "abs_tolerance": 0.15},
            "Diagnostics/predator_lateral_relative_speed_mean": {
                "value": 4.021025,
                "abs_tolerance": 0.50,
            },
            "Diagnostics/predator_speed_rms": {"value": 3.769160, "abs_tolerance": 0.50},
            "Diagnostics/predator_action_near_limit_fraction": {
                "value": 0.176806,
                "abs_tolerance": 0.05,
            },
        },
    },
}
CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def default_isaaclab_launcher(
    *,
    platform: str | None = None,
    repo_root: Path = REPO_ROOT,
    workspace_root: Path = Path("/workspace/isaaclab"),
) -> Path:
    """Resolve the conventional Isaac Lab launcher for the active platform."""

    platform = sys.platform if platform is None else platform
    launcher_name = "isaaclab.bat" if platform.startswith("win") else "isaaclab.sh"
    configured_root = os.environ.get("ISAACLAB_ROOT")
    if configured_root:
        return Path(configured_root).expanduser() / launcher_name
    if platform.startswith("win"):
        return Path(r"C:\RL\IsaacLab") / launcher_name

    candidates = (
        repo_root.parent / "IsaacLab" / launcher_name,
        workspace_root / launcher_name,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def default_evaluation_root(*, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve the host-owned Isaac evaluation artifact directory."""

    configured_root = os.environ.get("ARTIFACTS_ROOT")
    if configured_root:
        return Path(configured_root).expanduser() / "isaac" / "evaluations"
    candidates = (
        Path("/workspace/artifacts/isaac/evaluations"),
        repo_root.parent / "artifacts" / "isaac" / "evaluations",
    )
    return next((candidate for candidate in candidates if candidate.parent.is_dir()), candidates[-1])


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(raw_path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_pool(
    pool_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    expected_entries_per_role: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """Load and validate the portable predator/prey checkpoint pool."""

    data = json.loads(pool_path.read_text(encoding="utf-8-sig"))
    pool: dict[str, list[dict[str, Any]]] = {"predator": [], "prey": []}
    for role in pool:
        raw_entries = data.get(role)
        if not isinstance(raw_entries, list):
            raise RuntimeError(f"Pool key '{role}' must contain a list")
        if expected_entries_per_role and len(raw_entries) != expected_entries_per_role:
            raise RuntimeError(
                f"Expected {expected_entries_per_role} {role} entries, found {len(raw_entries)} in {pool_path}"
            )
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict) or "checkpoint" not in raw_entry:
                raise RuntimeError(f"Pool entry {role}[{index}] must be an object with a checkpoint")
            checkpoint = _resolve_repo_path(str(raw_entry["checkpoint"]), repo_root=repo_root)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Pool checkpoint does not exist: {checkpoint}")
            pool[role].append(
                {
                    "index": index,
                    "name": str(raw_entry.get("name", checkpoint.stem)),
                    "checkpoint": checkpoint,
                    "pool_type": str(raw_entry.get("pool_type", "elite")),
                }
            )
    return pool


def build_evaluate_command(
    *,
    isaaclab: Path,
    checkpoint: Path,
    output_json: Path,
    task: str,
    agent: str,
    algorithm: str,
    num_envs: int,
    episodes: int,
    seed: int,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Build one deterministic evaluator command."""

    return [
        str(isaaclab),
        "-p",
        str(repo_root / "scripts" / "skrl" / "evaluate.py"),
        "--headless",
        "--task",
        task,
        "--agent",
        agent,
        "--algorithm",
        algorithm,
        "--num_envs",
        str(num_envs),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--json",
        str(output_json),
    ]


def build_compose_command(
    *,
    isaaclab: Path,
    base: Path,
    output: Path,
    role: str,
    role_checkpoint: Path,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Build a checkpoint-composition command for one selected role."""

    if role not in {"predator", "prey"}:
        raise ValueError(f"Unsupported checkpoint role: {role}")
    return [
        str(isaaclab),
        "-p",
        str(repo_root / "scripts" / "compose_checkpoint.py"),
        "--base",
        str(base),
        f"--{role}",
        str(role_checkpoint),
        "--output",
        str(output),
    ]


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
    runner: CommandRunner = subprocess.run,
) -> None:
    """Print and optionally execute a subprocess command."""

    print("\n[COMMAND]")
    print(subprocess.list2cmdline(list(command)))
    if not dry_run:
        runner(list(command), cwd=cwd, check=True)


def aggregate_metrics(summaries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate all numeric evaluator metrics using sample standard deviation."""

    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    for section in REPORT_SECTIONS:
        values_by_metric: dict[str, list[float]] = {}
        for summary in summaries:
            metrics = summary.get(section, {})
            if not isinstance(metrics, Mapping):
                continue
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values_by_metric.setdefault(str(name), []).append(float(value))
        aggregate[section] = {}
        for name, values in sorted(values_by_metric.items()):
            aggregate[section][name] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    return aggregate


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_text_at(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_identity(isaaclab: Path) -> dict[str, Any]:
    """Capture the runtime versions that can affect deterministic evaluation."""

    import skrl
    import torch

    isaaclab_root = isaaclab.parent
    isaacsim_root = Path(os.environ.get("ISAACSIM_ROOT_PATH", "/isaac-sim"))
    lab_version_path = isaaclab_root / "VERSION"
    sim_version_path = isaacsim_root / "VERSION"
    lab_commit_checkout = _git_text_at(isaaclab_root, "rev-parse", "HEAD")
    lab_commit_env = os.environ.get("ISAACLAB_COMMIT")
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "skrl": skrl.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "isaac_sim": sim_version_path.read_text(encoding="utf-8").strip() if sim_version_path.is_file() else None,
        "isaac_lab": lab_version_path.read_text(encoding="utf-8").strip() if lab_version_path.is_file() else None,
        "isaac_lab_commit": lab_commit_checkout or lab_commit_env,
        "isaac_lab_commit_source": "checkout" if lab_commit_checkout else "ISAACLAB_COMMIT",
        "container_image": os.environ.get("UAV_ISAAC_IMAGE"),
    }


def _input_snapshot(
    *,
    checkpoint: Path,
    pool_path: Path,
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fingerprint Git and every external/code input used by this orchestrator."""

    evaluation_code = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "skrl" / "evaluate.py",
        REPO_ROOT / "scripts" / "compose_checkpoint.py",
    )
    return {
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "status": _git_text("status", "--porcelain"),
        },
        "files": {
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "opponent_pool": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
            "pool_checkpoints": [
                {
                    "role": role,
                    "index": entry["index"],
                    "name": entry["name"],
                    "path": str(entry["checkpoint"]),
                    "sha256": sha256_file(entry["checkpoint"]),
                }
                for role in ("predator", "prey")
                for entry in pool[role]
            ],
            "evaluation_code": [
                {"path": str(path), "sha256": sha256_file(path)} for path in evaluation_code
            ],
        },
    }


def validate_evaluation_summary(
    summary: Mapping[str, Any],
    *,
    expected_task: str,
    expected_num_envs: int,
    expected_episodes: int,
) -> None:
    """Reject incomplete or schema-incompatible evaluator output."""

    expected_fields = {
        "task": expected_task,
        "num_envs": expected_num_envs,
        "episodes_requested": expected_episodes,
        "episodes_completed": expected_episodes,
        "episode_accounting": "balanced_per_env",
    }
    for field, expected in expected_fields.items():
        actual = summary.get(field)
        if actual != expected:
            raise RuntimeError(f"Evaluator summary {field}={actual!r}; expected {expected!r}")

    required_metrics = {
        "episode_metrics": (
            "Metrics/catch_rate",
            "Metrics/predator_oob_rate",
            "Metrics/prey_oob_rate",
            "Metrics/episode_length",
            "Metrics/episode_closest_approach",
        ),
        "step_diagnostics": (
            "Diagnostics/predator_closing_speed_mean",
            "Diagnostics/predator_lateral_relative_speed_mean",
            "Diagnostics/predator_speed_rms",
            "Diagnostics/predator_action_near_limit_fraction",
        ),
    }
    for section, names in required_metrics.items():
        metrics = summary.get(section)
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"Evaluator summary is missing mapping section {section!r}")
        for name in names:
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"Evaluator summary metric {section}.{name} is missing or non-finite")


def _record_from_summary(
    *,
    group: str,
    label: str,
    seed: int,
    json_path: Path,
    summary: Mapping[str, Any],
    expected_task: str,
    expected_num_envs: int,
    expected_episodes: int,
    role_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_evaluation_summary(
        summary,
        expected_task=expected_task,
        expected_num_envs=expected_num_envs,
        expected_episodes=expected_episodes,
    )
    record = {
        "group": group,
        "label": label,
        "seed": seed,
        "json": str(json_path),
        "checkpoint": summary.get("checkpoint"),
        "task": summary["task"],
        "num_envs": summary["num_envs"],
        "episodes_requested": summary["episodes_requested"],
        "episodes_completed": summary["episodes_completed"],
        "episode_accounting": summary["episode_accounting"],
    }
    if role_source is not None:
        record["role_source"] = dict(role_source)
    for section in REPORT_SECTIONS:
        record[section] = dict(summary.get(section, {}))
    return record


def compare_windows_reference(
    anchor_records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    num_envs: int,
    episodes: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Compare the matching seed-42 anchor against the handoff reference."""

    result: dict[str, Any] = {
        "status": "not_run" if dry_run else "not_comparable",
        "source": WINDOWS_REFERENCE["source"],
        "reference_seed": WINDOWS_REFERENCE["seed"],
        "comparisons": [],
    }
    if dry_run:
        result["reason"] = "Dry run: no evaluator output was produced."
        return result

    expected_config = {
        "task": WINDOWS_REFERENCE["task"],
        "num_envs": WINDOWS_REFERENCE["num_envs"],
        "episodes": WINDOWS_REFERENCE["episodes"],
    }
    actual_config = {"task": task, "num_envs": num_envs, "episodes": episodes}
    if actual_config != expected_config:
        result["reason"] = f"Reference requires {expected_config}; run used {actual_config}."
        return result

    reference_seed = int(WINDOWS_REFERENCE["seed"])
    anchor = next((record for record in anchor_records if record.get("seed") == reference_seed), None)
    if anchor is None:
        result["reason"] = f"No completed anchor evaluation for reference seed {reference_seed}."
        return result

    passed = True
    for section, metrics in WINDOWS_REFERENCE["metrics"].items():
        actual_metrics = anchor.get(section, {})
        for name, criterion in metrics.items():
            actual = float(actual_metrics[name])
            reference = float(criterion["value"])
            tolerance = float(criterion["abs_tolerance"])
            delta = actual - reference
            within_tolerance = abs(delta) <= tolerance
            passed = passed and within_tolerance
            result["comparisons"].append(
                {
                    "section": section,
                    "metric": name,
                    "reference": reference,
                    "actual": actual,
                    "delta": delta,
                    "abs_tolerance": tolerance,
                    "within_tolerance": within_tolerance,
                }
            )
    result["status"] = "pass" if passed else "fail"
    result["reason"] = (
        "All handoff metrics are within the behavioral-shift tolerances."
        if passed
        else "At least one handoff metric exceeds its behavioral-shift tolerance."
    )
    return result


def qualify_dirty_verdict(
    verdict: Mapping[str, Any],
    *,
    git_status: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Make a dirty-tree comparison visibly diagnostic rather than certifiable."""

    result = dict(verdict)
    if not git_status or dry_run:
        result["certifiable"] = result["status"] in {"pass", "fail"}
        return result
    behavioral_status = str(result["status"])
    result.update(
        {
            "behavioral_status": behavioral_status,
            "status": f"diagnostic_{behavioral_status}",
            "certifiable": False,
            "reason": f"DIRTY-TREE DIAGNOSTIC ONLY. {result.get('reason', '')}",
        }
    )
    return result


def certification_exit_code(status: str) -> int:
    """Return zero only for completed parity passes or explicit dry runs."""

    return 0 if status in {"pass", "diagnostic_pass", "not_run"} else 1


def _write_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "label", "seed", "section", "metric", "value"])
        for record in records:
            for section in REPORT_SECTIONS:
                metrics = record.get(section, {})
                if not isinstance(metrics, Mapping):
                    continue
                for metric, value in sorted(metrics.items()):
                    if isinstance(value, (int, float)):
                        writer.writerow(
                            [record["group"], record["label"], record["seed"], section, metric, value]
                        )


def _metric(record: Mapping[str, Any], name: str) -> float:
    metrics = record.get("episode_metrics", {})
    return float(metrics.get(name, float("nan"))) if isinstance(metrics, Mapping) else float("nan")


def _markdown_table(records: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Match | Seed | Catch | Predator OOB | Prey OOB | Episode length |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {label} | {seed} | {catch:.6f} | {pred_oob:.6f} | {prey_oob:.6f} | {length:.3f} |".format(
                label=record["label"],
                seed=record["seed"],
                catch=_metric(record, "Metrics/catch_rate"),
                pred_oob=_metric(record, "Metrics/predator_oob_rate"),
                prey_oob=_metric(record, "Metrics/prey_oob_rate"),
                length=_metric(record, "Metrics/episode_length"),
            )
        )
    return lines


def _write_markdown(
    path: Path,
    *,
    windows_parity: Mapping[str, Any],
    anchor_records: Sequence[Mapping[str, Any]],
    current_predator_records: Sequence[Mapping[str, Any]],
    current_prey_records: Sequence[Mapping[str, Any]],
    dry_run: bool,
) -> None:
    status = str(windows_parity["status"]).upper().replace("_", " ")
    lines = [
        "# 3v1 Baseline Certification",
        "",
        "## Windows parity verdict",
        "",
        f"**{status}** — {windows_parity.get('reason', '')}",
        "",
    ]
    comparisons = windows_parity.get("comparisons", [])
    if comparisons:
        lines.extend(
            [
                "| Metric | Windows | Linux | Delta | Tolerance | Result |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for comparison in comparisons:
            lines.append(
                "| {metric} | {reference:.6f} | {actual:.6f} | {delta:+.6f} | {tolerance:.6f} | {result} |".format(
                    metric=comparison["metric"],
                    reference=comparison["reference"],
                    actual=comparison["actual"],
                    delta=comparison["delta"],
                    tolerance=comparison["abs_tolerance"],
                    result="PASS" if comparison["within_tolerance"] else "FAIL",
                )
            )
        lines.append("")
    if dry_run:
        lines.extend(["Dry run only: commands were planned but no evaluations were executed.", ""])
    for title, records in (
        ("Anchor across seeds", anchor_records),
        ("Current predator vs pool prey", current_predator_records),
        ("Pool predator vs current prey", current_prey_records),
    ):
        lines.extend([f"## {title}", ""])
        if records:
            lines.extend(_markdown_table(records))
        else:
            lines.append("No completed evaluations.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opponent-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--isaaclab", type=Path, default=default_isaaclab_launcher())
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--algorithm", default="MAPPO")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--cross-play-seed", type=int, default=None)
    parser.add_argument("--cross-play-num-envs", type=int, default=None)
    parser.add_argument("--cross-play-episodes", type=int, default=None)
    parser.add_argument("--expected-pool-entries-per-role", type=int, default=4)
    parser.add_argument("--skip-anchor", action="store_true")
    parser.add_argument("--skip-cross-play", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow real evaluations from an uncommitted tree; provenance still records the dirty status.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = _resolve_repo_path(args.checkpoint)
    pool_path = _resolve_repo_path(args.opponent_pool)
    isaaclab = args.isaaclab.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Anchor checkpoint does not exist: {checkpoint}")
    if not pool_path.is_file():
        raise FileNotFoundError(f"Opponent pool does not exist: {pool_path}")
    if not isaaclab.is_file():
        raise FileNotFoundError(f"Isaac Lab launcher does not exist: {isaaclab}")
    if not args.seeds:
        raise ValueError("Pass at least one evaluation seed")
    for name in ("num_envs", "episodes"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("cross_play_num_envs", "cross_play_episodes"):
        value = getattr(args, name)
        if value is not None and int(value) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    git_status = _git_text("status", "--porcelain")
    if git_status and not args.dry_run and not args.allow_dirty:
        raise RuntimeError(
            "Refusing to certify an uncommitted working tree. Commit the intended state or pass --allow-dirty."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_evaluation_root() / f"{timestamp}_baseline_certification"
    )
    runs_dir = output_dir / "runs"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix certification artifacts in non-empty directory: {output_dir}")
    runs_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "STATUS.json"
    execution_status: dict[str, Any] = {
        "status": "incomplete",
        "execution_status": "incomplete",
        "started_at": datetime.now().astimezone().isoformat(),
    }

    def write_execution_status() -> None:
        status_path.write_text(json.dumps(execution_status, indent=2), encoding="utf-8")

    write_execution_status()
    atexit.register(write_execution_status)
    pool = load_pool(
        pool_path,
        expected_entries_per_role=max(0, int(args.expected_pool_entries_per_role)),
    )
    cross_seed = args.cross_play_seed if args.cross_play_seed is not None else args.seeds[0]
    cross_num_envs = args.cross_play_num_envs or args.num_envs
    cross_episodes = args.cross_play_episodes or args.episodes
    planned_commands: list[list[str]] = []
    anchor_records: list[dict[str, Any]] = []
    current_predator_records: list[dict[str, Any]] = []
    current_prey_records: list[dict[str, Any]] = []
    initial_snapshot = _input_snapshot(checkpoint=checkpoint, pool_path=pool_path, pool=pool)
    runtime_identity = _runtime_identity(isaaclab)
    provenance_path = output_dir / "provenance.json"
    provenance: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "incomplete",
        "argv": sys.argv,
        **initial_snapshot,
        "runtime": runtime_identity,
        "commands": planned_commands,
    }

    def write_provenance() -> None:
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    write_provenance()

    def execute(command: list[str]) -> None:
        planned_commands.append(command)
        write_provenance()
        run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run)

    if not args.skip_anchor:
        for seed in args.seeds:
            json_path = runs_dir / f"anchor_seed{seed}.json"
            execute(
                build_evaluate_command(
                    isaaclab=isaaclab,
                    checkpoint=checkpoint,
                    output_json=json_path,
                    task=args.task,
                    agent=args.agent,
                    algorithm=args.algorithm,
                    num_envs=args.num_envs,
                    episodes=args.episodes,
                    seed=seed,
                )
            )
            if not args.dry_run:
                anchor_records.append(
                    _record_from_summary(
                        group="anchor",
                        label="current_predator_vs_current_prey",
                        seed=seed,
                        json_path=json_path,
                        summary=_load_summary(json_path),
                        expected_task=args.task,
                        expected_num_envs=args.num_envs,
                        expected_episodes=args.episodes,
                    )
                )

    if not args.skip_cross_play:
        with tempfile.TemporaryDirectory(prefix="uavpp-baseline-") as temp_raw:
            temp_dir = Path(temp_raw)
            for entry in pool["prey"]:
                index = int(entry["index"])
                composed = temp_dir / f"current_predator_vs_prey_{index:02d}.pt"
                json_path = runs_dir / f"current_predator_vs_prey_{index:02d}_seed{cross_seed}.json"
                execute(
                    build_compose_command(
                        isaaclab=isaaclab,
                        base=checkpoint,
                        output=composed,
                        role="prey",
                        role_checkpoint=entry["checkpoint"],
                    )
                )
                execute(
                    build_evaluate_command(
                        isaaclab=isaaclab,
                        checkpoint=composed,
                        output_json=json_path,
                        task=args.task,
                        agent=args.agent,
                        algorithm=args.algorithm,
                        num_envs=cross_num_envs,
                        episodes=cross_episodes,
                        seed=cross_seed,
                    )
                )
                if not args.dry_run:
                    current_predator_records.append(
                        _record_from_summary(
                            group="current_predator_vs_pool_prey",
                            label=f"current_predator_vs_prey_{index:02d}",
                            seed=cross_seed,
                            json_path=json_path,
                            summary=_load_summary(json_path),
                            expected_task=args.task,
                            expected_num_envs=cross_num_envs,
                            expected_episodes=cross_episodes,
                            role_source={"role": "prey", "name": entry["name"], "checkpoint": str(entry["checkpoint"])},
                        )
                    )
            for entry in pool["predator"]:
                index = int(entry["index"])
                composed = temp_dir / f"predator_{index:02d}_vs_current_prey.pt"
                json_path = runs_dir / f"predator_{index:02d}_vs_current_prey_seed{cross_seed}.json"
                execute(
                    build_compose_command(
                        isaaclab=isaaclab,
                        base=checkpoint,
                        output=composed,
                        role="predator",
                        role_checkpoint=entry["checkpoint"],
                    )
                )
                execute(
                    build_evaluate_command(
                        isaaclab=isaaclab,
                        checkpoint=composed,
                        output_json=json_path,
                        task=args.task,
                        agent=args.agent,
                        algorithm=args.algorithm,
                        num_envs=cross_num_envs,
                        episodes=cross_episodes,
                        seed=cross_seed,
                    )
                )
                if not args.dry_run:
                    current_prey_records.append(
                        _record_from_summary(
                            group="pool_predator_vs_current_prey",
                            label=f"predator_{index:02d}_vs_current_prey",
                            seed=cross_seed,
                            json_path=json_path,
                            summary=_load_summary(json_path),
                            expected_task=args.task,
                            expected_num_envs=cross_num_envs,
                            expected_episodes=cross_episodes,
                            role_source={
                                "role": "predator",
                                "name": entry["name"],
                                "checkpoint": str(entry["checkpoint"]),
                            },
                        )
                    )

    final_snapshot = _input_snapshot(checkpoint=checkpoint, pool_path=pool_path, pool=pool)
    if final_snapshot != initial_snapshot:
        provenance.update(
            {
                "status": "input_changed",
                "completion_snapshot": final_snapshot,
            }
        )
        write_provenance()
        execution_status["reason"] = "Git state or a fingerprinted evaluation input changed during execution."
        raise RuntimeError(execution_status["reason"])

    all_records = anchor_records + current_predator_records + current_prey_records
    windows_parity = qualify_dirty_verdict(
        compare_windows_reference(
            anchor_records,
            task=args.task,
            num_envs=args.num_envs,
            episodes=args.episodes,
            dry_run=args.dry_run,
        ),
        git_status=git_status,
        dry_run=args.dry_run,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "dry_run": args.dry_run,
        "config": {
            "checkpoint": str(checkpoint),
            "opponent_pool": str(pool_path),
            "isaaclab": str(isaaclab),
            "task": args.task,
            "agent": args.agent,
            "algorithm": args.algorithm,
            "seeds": args.seeds,
            "num_envs": args.num_envs,
            "episodes": args.episodes,
            "cross_play_seed": cross_seed,
            "cross_play_num_envs": cross_num_envs,
            "cross_play_episodes": cross_episodes,
        },
        "windows_parity": windows_parity,
        "anchor": {"runs": anchor_records, "aggregate": aggregate_metrics(anchor_records)},
        "current_predator_vs_pool_prey": {
            "runs": current_predator_records,
            "aggregate": aggregate_metrics(current_predator_records),
        },
        "pool_predator_vs_current_prey": {
            "runs": current_prey_records,
            "aggregate": aggregate_metrics(current_prey_records),
        },
    }
    (output_dir / "baseline_certification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(output_dir / "baseline_certification.csv", all_records)
    _write_markdown(
        output_dir / "REPORT.md",
        windows_parity=windows_parity,
        anchor_records=anchor_records,
        current_predator_records=current_predator_records,
        current_prey_records=current_prey_records,
        dry_run=args.dry_run,
    )
    provenance.update(
        {
            "status": "complete",
            "finished_at": datetime.now().astimezone().isoformat(),
            "windows_parity": windows_parity["status"],
        }
    )
    write_provenance()
    execution_status.update(
        {
            "execution_status": "complete",
            "status": (
                "complete"
                if certification_exit_code(windows_parity["status"]) == 0
                else "certification_failed"
            ),
            "certification_status": windows_parity["status"],
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    write_execution_status()
    atexit.unregister(write_execution_status)
    print(f"\n[INFO] Baseline certification output: {output_dir}")
    if certification_exit_code(windows_parity["status"]):
        raise SystemExit(f"Windows parity status is {windows_parity['status']}; inspect REPORT.md")


if __name__ == "__main__":
    main()
