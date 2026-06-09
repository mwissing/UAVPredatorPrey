"""Run short alternating self-play phases based on deterministic catch-rate.

This script is intentionally an outer-loop scheduler. It launches normal SKRL
training/evaluation subprocesses, then decides the next phase from evaluation
metrics. That keeps freeze/unfreeze behavior at run boundaries, where the SKRL
runner and checkpoint state are easiest to reason about.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAACLAB = Path(r"C:\RL\IsaacLab\isaaclab.bat")
DEFAULT_RUN_ROOT = REPO_ROOT / "logs" / "skrl" / "uav_3v1_direct"
AGENT_RE = re.compile(r"agent_(\d+)\.pt$")
AGENTS = ("predator", "prey")


def _agent_opponent(agent: str) -> str:
    if agent == "predator":
        return "prey"
    if agent == "prey":
        return "predator"
    raise ValueError(f"Agent '{agent}' does not have a pool opponent.")


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("\n[COMMAND]")
    print(_command_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def _load_opponent_pool(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {agent: [] for agent in AGENTS}

    data = json.loads(path.read_text(encoding="utf-8"))
    pool = {agent: [] for agent in AGENTS}
    for agent in AGENTS:
        entries = data.get(agent, [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Pool key '{agent}' must contain a list.")

        for index, raw_entry in enumerate(entries):
            if isinstance(raw_entry, str):
                checkpoint = Path(raw_entry)
                entry = {
                    "name": checkpoint.stem,
                    "checkpoint": str(checkpoint),
                    "weight": 1.0,
                    "notes": "",
                }
            elif isinstance(raw_entry, dict):
                if "checkpoint" not in raw_entry:
                    raise RuntimeError(f"Pool entry {agent}[{index}] is missing 'checkpoint'.")
                entry = dict(raw_entry)
                entry.setdefault("name", Path(str(entry["checkpoint"])).stem)
                entry.setdefault("weight", 1.0)
                entry.setdefault("notes", "")
            else:
                raise RuntimeError(f"Pool entry {agent}[{index}] must be a string or object.")

            checkpoint = Path(str(entry["checkpoint"]))
            if not checkpoint.exists():
                raise RuntimeError(f"Pool entry '{entry['name']}' checkpoint does not exist: {checkpoint}")
            entry["checkpoint"] = str(checkpoint.resolve())
            entry["weight"] = float(entry["weight"])
            if entry["weight"] > 0.0:
                pool[agent].append(entry)

    return pool


def _sample_pool_entry(
    pool: dict[str, list[dict[str, Any]]],
    agent: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    entries = pool.get(agent, [])
    if not entries:
        return None
    weights = [float(entry.get("weight", 1.0)) for entry in entries]
    return rng.choices(entries, weights=weights, k=1)[0]


def _compose_checkpoint(
    *,
    base: Path,
    predator: Path | None,
    prey: Path | None,
    output: Path,
    args: argparse.Namespace,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compose_checkpoint.py"),
        "--base",
        str(base),
        "--output",
        str(output),
    ]
    if predator is not None:
        command.extend(["--predator", str(predator)])
    if prey is not None:
        command.extend(["--prey", str(prey)])
    _run(command, cwd=REPO_ROOT, dry_run=args.dry_run)
    return output


def _latest_run_dir(run_root: Path, previous: set[Path]) -> Path:
    run_dirs = [path for path in run_root.iterdir() if path.is_dir() and path not in previous]
    if not run_dirs:
        run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {run_root}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _checkpoint_step(path: Path) -> int:
    match = AGENT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    agent_checkpoints = sorted(
        checkpoint_dir.glob("agent_*.pt"),
        key=lambda path: (_checkpoint_step(path), path.stat().st_mtime),
    )
    if agent_checkpoints:
        return agent_checkpoints[-1]

    best = checkpoint_dir / "best_agent.pt"
    if best.exists():
        return best
    raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")


def _metric(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get("episode_metrics", {}).get(key, default)
    return float(value)


def _decide_phase(summary: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    catch_rate = _metric(summary, "Metrics/catch_rate")
    prey_oob = _metric(summary, "Metrics/prey_oob_rate")
    predator_oob = _metric(summary, "Metrics/predator_oob_rate")
    prey_soft = _metric(summary, "Metrics/prey_soft_arena_outside")
    predator_soft = _metric(summary, "Metrics/predator_soft_arena_outside")

    if prey_oob > args.max_prey_oob or prey_soft > args.max_prey_soft:
        return "prey", (
            f"prey safety repair: prey_oob={prey_oob:.3f}, "
            f"prey_soft={prey_soft:.3f}"
        )
    if predator_oob > args.max_predator_oob or predator_soft > args.max_predator_soft:
        return "predator", (
            f"predator safety repair: predator_oob={predator_oob:.3f}, "
            f"predator_soft={predator_soft:.3f}"
        )
    if catch_rate < args.low:
        return "predator", f"prey too strong: catch_rate={catch_rate:.3f} < {args.low:.3f}"
    if catch_rate > args.high:
        return "prey", f"predator too strong: catch_rate={catch_rate:.3f} > {args.high:.3f}"
    return args.balanced_action, (
        f"balanced catch-rate band: {args.low:.3f} <= {catch_rate:.3f} <= {args.high:.3f}"
    )


def _freeze_args(phase: str) -> list[str]:
    if phase == "predator":
        return ["--freeze-agents", "prey"]
    if phase == "prey":
        return ["--freeze-agents", "predator"]
    if phase == "both":
        return []
    if phase == "stop":
        return []
    raise ValueError(f"Unknown phase: {phase}")


def _evaluate(
    checkpoint: Path,
    *,
    phase_index: int,
    output_dir: Path,
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    json_path = output_dir / f"phase_{phase_index:03d}_{label}_eval.json"
    command = [
        str(args.isaaclab),
        "-p",
        str(REPO_ROOT / "scripts" / "skrl" / "evaluate.py"),
        "--headless",
        "--task",
        args.task,
        "--agent",
        args.agent,
        "--algorithm",
        args.algorithm,
        "--num_envs",
        str(args.eval_num_envs),
        "--episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--checkpoint",
        str(checkpoint),
        "--json",
        str(json_path),
    ]
    _run(command, cwd=REPO_ROOT, dry_run=args.dry_run)
    if args.dry_run:
        return {
            "checkpoint": str(checkpoint),
            "episode_metrics": {
                "Metrics/catch_rate": args.low,
                "Metrics/prey_oob_rate": 0.0,
                "Metrics/predator_oob_rate": 0.0,
                "Metrics/prey_soft_arena_outside": 0.0,
                "Metrics/predator_soft_arena_outside": 0.0,
            },
        }
    return json.loads(json_path.read_text(encoding="utf-8"))


def _train_phase(
    checkpoint: Path,
    *,
    phase: str,
    phase_iterations: int,
    args: argparse.Namespace,
) -> Path:
    before = {path for path in args.run_root.iterdir() if path.is_dir()}
    command = [
        str(args.isaaclab),
        "-p",
        str(REPO_ROOT / "scripts" / "skrl" / "train.py"),
        "--headless",
        "--task",
        args.task,
        "--agent",
        args.agent,
        "--algorithm",
        args.algorithm,
        "--num_envs",
        str(args.train_num_envs),
        "--seed",
        str(args.seed),
        "--checkpoint",
        str(checkpoint),
        "--max_iterations",
        str(phase_iterations),
    ]
    command.extend(_freeze_args(phase))
    _run(command, cwd=REPO_ROOT, dry_run=args.dry_run)
    if args.dry_run:
        return checkpoint
    run_dir = _latest_run_dir(args.run_root, before)
    next_checkpoint = _latest_checkpoint(run_dir)
    print(f"[INFO] Phase run directory: {run_dir}")
    print(f"[INFO] Next checkpoint: {next_checkpoint}")
    return next_checkpoint


def _pool_train_checkpoint(
    current_checkpoint: Path,
    *,
    phase: str,
    phase_index: int,
    pool: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any] | None]:
    if phase not in {"predator", "prey"}:
        return current_checkpoint, None
    if args.pool_prob <= 0.0 or rng.random() >= args.pool_prob:
        return current_checkpoint, None

    opponent = _agent_opponent(phase)
    entry = _sample_pool_entry(pool, opponent, rng)
    if entry is None:
        print(f"[INFO] No {opponent} pool entries available; using current opponent.")
        return current_checkpoint, None

    pool_checkpoint = Path(entry["checkpoint"])
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry["name"]))
    output = args.output_dir / "composed" / f"phase_{phase_index:03d}_{phase}_vs_pool_{opponent}_{safe_name}.pt"

    print(
        f"[INFO] Pool sample for phase {phase_index}: training {phase} "
        f"against {opponent}='{entry['name']}'"
    )
    if phase == "predator":
        train_checkpoint = _compose_checkpoint(
            base=current_checkpoint,
            predator=current_checkpoint,
            prey=pool_checkpoint,
            output=output,
            args=args,
        )
    else:
        train_checkpoint = _compose_checkpoint(
            base=current_checkpoint,
            predator=pool_checkpoint,
            prey=current_checkpoint,
            output=output,
            args=args,
        )

    return train_checkpoint, {
        "phase": phase,
        "opponent": opponent,
        "entry": entry,
        "train_checkpoint": str(train_checkpoint),
    }


def _restore_current_pair_after_pool(
    current_checkpoint: Path,
    trained_checkpoint: Path,
    *,
    phase: str,
    phase_index: int,
    pool_sample: dict[str, Any] | None,
    args: argparse.Namespace,
) -> Path:
    if pool_sample is None:
        return trained_checkpoint

    output = args.output_dir / "composed" / f"phase_{phase_index:03d}_current_pair_after_{phase}_pool.pt"
    if phase == "predator":
        return _compose_checkpoint(
            base=current_checkpoint,
            predator=trained_checkpoint,
            prey=current_checkpoint,
            output=output,
            args=args,
        )
    if phase == "prey":
        return _compose_checkpoint(
            base=current_checkpoint,
            predator=current_checkpoint,
            prey=trained_checkpoint,
            output=output,
            args=args,
        )
    raise ValueError(f"Pool restore is only valid for single-agent phases, got: {phase}")


def _write_record(output_dir: Path, record: dict[str, Any]) -> None:
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Starting checkpoint.")
    parser.add_argument("--total-iterations", type=int, default=20_000, help="Total outer-loop training iterations.")
    parser.add_argument("--phase-iterations", type=int, default=500, help="Iterations per train phase.")
    parser.add_argument("--low", type=float, default=0.40, help="Train predator below this catch-rate.")
    parser.add_argument("--high", type=float, default=0.60, help="Train prey above this catch-rate.")
    parser.add_argument(
        "--balanced-action",
        choices=("both", "stop"),
        default="both",
        help="Action when catch-rate is inside the hysteresis band.",
    )
    parser.add_argument("--max-prey-oob", type=float, default=0.08, help="Force prey safety training above this OOB rate.")
    parser.add_argument(
        "--max-predator-oob",
        type=float,
        default=0.08,
        help="Force predator safety training above this OOB rate.",
    )
    parser.add_argument(
        "--max-prey-soft",
        type=float,
        default=0.30,
        help="Force prey safety training above this soft-arena outside mean.",
    )
    parser.add_argument(
        "--max-predator-soft",
        type=float,
        default=0.30,
        help="Force predator safety training above this soft-arena outside mean.",
    )
    parser.add_argument(
        "--opponent-pool",
        type=Path,
        default=None,
        help="Optional JSON file with predator/prey checkpoint pools for old-opponent sampling.",
    )
    parser.add_argument(
        "--pool-prob",
        type=float,
        default=0.0,
        help="Probability that a single-agent phase trains against an old frozen opponent from the pool.",
    )
    parser.add_argument(
        "--pool-seed",
        type=int,
        default=None,
        help="Seed for opponent-pool sampling. Defaults to --seed.",
    )
    parser.add_argument("--task", default="1v1-survival-soft-oob-v0", help="Isaac Lab task id.")
    parser.add_argument("--agent", default="skrl_mappo_finetune_cfg_entry_point", help="SKRL agent config entry point.")
    parser.add_argument("--algorithm", default="MAPPO", help="SKRL algorithm.")
    parser.add_argument("--train-num-envs", type=int, default=4096, help="Number of training envs.")
    parser.add_argument("--eval-num-envs", type=int, default=512, help="Number of evaluation envs.")
    parser.add_argument("--eval-episodes", type=int, default=512, help="Deterministic eval episodes per phase.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/eval subprocesses.")
    parser.add_argument("--isaaclab", type=Path, default=DEFAULT_ISAACLAB, help="Path to isaaclab.bat.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="SKRL run root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for scheduler JSON logs. Defaults to logs/curriculum/<timestamp>_hysteresis.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    args.checkpoint = args.checkpoint.resolve()
    args.isaaclab = args.isaaclab.resolve()
    args.run_root = args.run_root.resolve()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = REPO_ROOT / "logs" / "curriculum" / f"{timestamp}_hysteresis"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pool_prob = max(0.0, min(1.0, args.pool_prob))
    pool_path = args.opponent_pool.resolve() if args.opponent_pool else None
    pool = _load_opponent_pool(pool_path)
    rng = random.Random(args.pool_seed if args.pool_seed is not None else args.seed)
    if pool_path is not None:
        print(f"[INFO] Loaded opponent pool: {pool_path}")
        print(f"[INFO] Predator entries: {len(pool['predator'])}; prey entries: {len(pool['prey'])}")
        print(f"[INFO] Pool sampling probability: {args.pool_prob:.3f}")

    current_checkpoint = args.checkpoint
    completed_iterations = 0
    phase_index = 0
    summary = _evaluate(current_checkpoint, phase_index=phase_index, output_dir=args.output_dir, args=args, label="start")

    while completed_iterations < args.total_iterations:
        phase, reason = _decide_phase(summary, args)
        catch_rate = _metric(summary, "Metrics/catch_rate")
        prey_oob = _metric(summary, "Metrics/prey_oob_rate")
        predator_oob = _metric(summary, "Metrics/predator_oob_rate")
        print(
            "\n[DECISION] "
            f"phase={phase}, reason={reason}, catch={catch_rate:.3f}, "
            f"prey_oob={prey_oob:.3f}, predator_oob={predator_oob:.3f}"
        )
        _write_record(
            args.output_dir,
            {
                "phase_index": phase_index,
                "completed_iterations": completed_iterations,
                "checkpoint": str(current_checkpoint),
                "decision": phase,
                "reason": reason,
                "metrics": summary.get("episode_metrics", {}),
            },
        )

        if phase == "stop":
            print("[INFO] Balanced band reached and --balanced-action=stop. Stopping.")
            break

        phase_iterations = min(args.phase_iterations, args.total_iterations - completed_iterations)
        phase_start_checkpoint = current_checkpoint
        train_checkpoint, pool_sample = _pool_train_checkpoint(
            current_checkpoint,
            phase=phase,
            phase_index=phase_index,
            pool=pool,
            rng=rng,
            args=args,
        )
        trained_checkpoint = _train_phase(
            train_checkpoint,
            phase=phase,
            phase_iterations=phase_iterations,
            args=args,
        )
        current_checkpoint = _restore_current_pair_after_pool(
            phase_start_checkpoint,
            trained_checkpoint,
            phase=phase,
            phase_index=phase_index,
            pool_sample=pool_sample,
            args=args,
        )
        if pool_sample is not None:
            _write_record(
                args.output_dir,
                {
                    "phase_index": phase_index,
                    "completed_iterations": completed_iterations,
                    "pool_sample": pool_sample,
                    "trained_checkpoint": str(trained_checkpoint),
                    "restored_current_pair_checkpoint": str(current_checkpoint),
                },
            )
        completed_iterations += phase_iterations
        phase_index += 1
        summary = _evaluate(
            current_checkpoint,
            phase_index=phase_index,
            output_dir=args.output_dir,
            args=args,
            label="after",
        )

    final_path = args.output_dir / "final_checkpoint.txt"
    final_path.write_text(str(current_checkpoint), encoding="utf-8")
    print(f"\n[INFO] Final checkpoint: {current_checkpoint}")
    print(f"[INFO] Scheduler logs: {args.output_dir}")


if __name__ == "__main__":
    main()
