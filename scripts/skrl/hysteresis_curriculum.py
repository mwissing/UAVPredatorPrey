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

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAACLAB = Path(r"C:\RL\IsaacLab\isaaclab.bat")
DEFAULT_RUN_ROOT = REPO_ROOT / "logs" / "skrl" / "uav_3v1_direct"
AGENT_RE = re.compile(r"agent_(\d+)\.pt$")
AGENTS = ("predator", "prey")
POOL_TYPE_ELITE = "elite"
POOL_TYPE_RECENT = "recent"
PRESETS = {
    "3v1-attention-critic": {
        "task": "3v1-survival-soft-oob-teammate-vel-v0",
        "agent": "skrl_mappo_attention_critic_cfg_entry_point",
        "algorithm": "MAPPO",
        "output_suffix": "3v1_attention_critic_hysteresis",
    },
    "3v1-attention-critic-prey-attention": {
        "task": "3v1-survival-soft-oob-teammate-vel-random-spawn-v0",
        "agent": "skrl_mappo_attention_critic_prey_attention_cfg_entry_point",
        "algorithm": "MAPPO",
        "output_suffix": "3v1_attention_critic_prey_attention_hysteresis",
    },
    "3v1-attention-critic-prey-attention-large": {
        "task": "3v1-survival-soft-oob-teammate-vel-random-spawn-v0",
        "agent": "skrl_mappo_attention_critic_prey_attention_large_cfg_entry_point",
        "algorithm": "MAPPO",
        "output_suffix": "3v1_attention_critic_prey_attention_large_hysteresis",
    },
    "3v1-attention-critic-prey-attention-large-gru": {
        "task": "3v1-survival-soft-oob-teammate-vel-random-spawn-v0",
        "agent": "skrl_mappo_attention_critic_prey_attention_large_gru_cfg_entry_point",
        "algorithm": "MAPPO",
        "output_suffix": "3v1_attention_critic_prey_attention_large_gru_hysteresis",
    },
}


def _agent_opponent(agent: str) -> str:
    if agent == "predator":
        return "prey"
    if agent == "prey":
        return "predator"
    raise ValueError(f"Agent '{agent}' does not have a pool opponent.")


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _cli_arg_present(*names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:] for name in names)


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("\n[COMMAND]")
    print(_command_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def _load_opponent_pool(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {agent: [] for agent in AGENTS}

    data = json.loads(path.read_text(encoding="utf-8-sig"))
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
                    "pool_type": POOL_TYPE_ELITE,
                    "notes": "",
                }
            elif isinstance(raw_entry, dict):
                if "checkpoint" not in raw_entry:
                    raise RuntimeError(f"Pool entry {agent}[{index}] is missing 'checkpoint'.")
                entry = dict(raw_entry)
                entry.setdefault("name", Path(str(entry["checkpoint"])).stem)
                entry.setdefault("weight", 1.0)
                entry.setdefault("base_weight", entry["weight"])
                entry.setdefault("notes", "")
                entry.setdefault("pool_type", POOL_TYPE_ELITE)
            else:
                raise RuntimeError(f"Pool entry {agent}[{index}] must be a string or object.")

            checkpoint = Path(str(entry["checkpoint"]))
            if not checkpoint.exists():
                raise RuntimeError(f"Pool entry '{entry['name']}' checkpoint does not exist: {checkpoint}")
            entry["checkpoint"] = str(checkpoint.resolve())
            entry["weight"] = float(entry["weight"])
            entry["base_weight"] = float(entry.get("base_weight", entry["weight"]))
            entry["pool_type"] = str(entry.get("pool_type", POOL_TYPE_ELITE))
            if entry["weight"] > 0.0:
                pool[agent].append(entry)

    return pool


def _save_opponent_pool(path: Path, pool: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _sort_pool_for_save(pool)
    path.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote opponent pool: {path}")


def _entry_metric(entry: dict[str, Any], key: str, default: float | None = None) -> float | None:
    last_cross_play = entry.get("last_cross_play", {})
    if isinstance(last_cross_play, dict):
        metrics = last_cross_play.get("metrics", {})
        if isinstance(metrics, dict) and key in metrics:
            return float(metrics[key])

    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
        return default
    value = metrics.get(key, default)
    return None if value is None else float(value)


def _pfsp_weight(win_rate: float, weighting: str) -> float:
    win_rate = max(0.0, min(1.0, win_rate))
    if weighting == "linear":
        return 1.0 - win_rate
    if weighting == "squared":
        return (1.0 - win_rate) ** 2
    if weighting == "variance":
        return win_rate * (1.0 - win_rate)
    raise ValueError(f"Unknown PFSP weighting: {weighting}")


def _pool_entry_weight(
    entry: dict[str, Any],
    *,
    sampled_agent: str,
    training_agent: str | None,
    args: argparse.Namespace,
) -> float:
    base_weight = float(entry.get("base_weight", entry.get("weight", 1.0)))
    if args.pool_sampling != "pfsp" or training_agent is None:
        return float(entry.get("weight", base_weight))

    if "pfsp_multiplier" in entry:
        return base_weight * float(entry["pfsp_multiplier"])

    catch_rate = _entry_metric(entry, "Metrics/catch_rate")
    if catch_rate is None:
        return float(entry.get("weight", base_weight))

    if training_agent == "predator" and sampled_agent == "prey":
        # Predator win-rate proxy against this prey entry.
        win_rate = catch_rate
    elif training_agent == "prey" and sampled_agent == "predator":
        # Prey win-rate proxy against this predator entry.
        win_rate = 1.0 - catch_rate
    else:
        win_rate = 0.5

    return base_weight * max(float(args.pfsp_min_weight), _pfsp_weight(win_rate, args.pfsp_weighting))


def _pool_entries_by_type(entries: list[dict[str, Any]], pool_type: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if str(entry.get("pool_type", POOL_TYPE_ELITE)) == pool_type
    ]


def _sample_pool_entry(
    pool: dict[str, list[dict[str, Any]]],
    agent: str,
    rng: random.Random,
    *,
    training_agent: str | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    entries = pool.get(agent, [])
    if not entries:
        return None
    weights = [
        _pool_entry_weight(entry, sampled_agent=agent, training_agent=training_agent, args=args)
        for entry in entries
    ]
    if not any(weight > 0.0 for weight in weights):
        return None
    return rng.choices(entries, weights=weights, k=1)[0]


def _pool_entry_compatible(entry: dict[str, Any], *, role: str, args: argparse.Namespace) -> bool:
    """Return whether a pool entry can be loaded by the active model config."""

    if role != "prey" or "prey_attention" not in args.agent:
        return True

    checkpoint_path = Path(str(entry["checkpoint"]))
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[WARNING] Skipping pool entry that cannot be inspected: {entry.get('name', checkpoint_path.stem)}")
        print(f"[WARNING]   {exc}")
        return False

    role_state = checkpoint.get(role)
    policy_state = role_state.get("policy") if isinstance(role_state, dict) else None
    if not isinstance(policy_state, dict):
        print(f"[WARNING] Skipping pool entry without {role} policy: {entry.get('name', checkpoint_path.stem)}")
        return False

    if any(key.startswith("predator_encoder.") for key in policy_state):
        return True

    print(
        f"[WARNING] Skipping incompatible {role} pool entry for active prey-attention config: "
        f"{entry.get('name', checkpoint_path.stem)}"
    )
    return False


def _compatible_pool_for_role(
    pool: dict[str, list[dict[str, Any]]],
    *,
    role: str,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    compatible = dict(pool)
    compatible[role] = [
        entry for entry in pool.get(role, []) if _pool_entry_compatible(entry, role=role, args=args)
    ]
    return compatible


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


def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _promotion_gate(
    agent: str,
    summary: dict[str, Any],
    args: argparse.Namespace,
    cross_play_aggregate: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    catch_rate = _metric(summary, "Metrics/catch_rate")
    prey_oob = _metric(summary, "Metrics/prey_oob_rate")
    predator_oob = _metric(summary, "Metrics/predator_oob_rate")
    prey_soft = _metric(summary, "Metrics/prey_soft_arena_outside")
    predator_soft = _metric(summary, "Metrics/predator_soft_arena_outside")

    if agent == "predator":
        checks = [
            (catch_rate >= args.auto_pool_predator_min_catch, f"catch_rate={catch_rate:.3f}"),
            (predator_oob <= args.auto_pool_max_predator_oob, f"predator_oob={predator_oob:.3f}"),
            (predator_soft <= args.auto_pool_max_predator_soft, f"predator_soft={predator_soft:.3f}"),
            (prey_oob <= args.auto_pool_max_opponent_oob, f"prey_oob={prey_oob:.3f}"),
            (prey_soft <= args.auto_pool_max_opponent_soft, f"prey_soft={prey_soft:.3f}"),
        ]
        if cross_play_aggregate:
            cross_avg = float(cross_play_aggregate["catch_avg"])
            cross_min = float(cross_play_aggregate["catch_min"])
            checks.extend(
                [
                    (
                        cross_avg >= args.auto_pool_predator_min_cross_play_avg_catch,
                        f"cross_avg_catch={cross_avg:.3f}",
                    ),
                    (
                        cross_min >= args.auto_pool_predator_min_cross_play_min_catch,
                        f"cross_min_catch={cross_min:.3f}",
                    ),
                ]
            )
    elif agent == "prey":
        checks = [
            (catch_rate <= args.auto_pool_prey_max_catch, f"catch_rate={catch_rate:.3f}"),
            (prey_oob <= args.auto_pool_max_prey_oob, f"prey_oob={prey_oob:.3f}"),
            (prey_soft <= args.auto_pool_max_prey_soft, f"prey_soft={prey_soft:.3f}"),
            (predator_oob <= args.auto_pool_max_opponent_oob, f"predator_oob={predator_oob:.3f}"),
            (predator_soft <= args.auto_pool_max_opponent_soft, f"predator_soft={predator_soft:.3f}"),
        ]
        if cross_play_aggregate:
            cross_avg = float(cross_play_aggregate["catch_avg"])
            cross_max = float(cross_play_aggregate["catch_max"])
            checks.extend(
                [
                    (
                        cross_avg <= args.auto_pool_prey_max_cross_play_avg_catch,
                        f"cross_avg_catch={cross_avg:.3f}",
                    ),
                    (
                        cross_max <= args.auto_pool_prey_max_cross_play_max_catch,
                        f"cross_max_catch={cross_max:.3f}",
                    ),
                ]
            )
    else:
        return False, f"auto-pool promotion only supports predator/prey phases, got {agent}"

    details = ", ".join(detail for _, detail in checks)
    failed = [detail for ok, detail in checks if not ok]
    if failed:
        return False, f"rejected {agent}: {', '.join(failed)}; metrics: {details}"
    return True, f"accepted {agent}: {details}"


def _pool_score(
    agent: str,
    summary: dict[str, Any],
    cross_play_aggregate: dict[str, Any] | None = None,
) -> float:
    catch_rate = _metric(summary, "Metrics/catch_rate")
    prey_oob = _metric(summary, "Metrics/prey_oob_rate")
    predator_oob = _metric(summary, "Metrics/predator_oob_rate")
    prey_soft = _metric(summary, "Metrics/prey_soft_arena_outside")
    predator_soft = _metric(summary, "Metrics/predator_soft_arena_outside")

    safety_penalty = prey_oob + predator_oob + 0.1 * (prey_soft + predator_soft)
    if agent == "predator":
        score = catch_rate - safety_penalty
        if cross_play_aggregate:
            score = (
                0.5 * score
                + 0.3 * float(cross_play_aggregate["catch_avg"])
                + 0.2 * float(cross_play_aggregate["catch_min"])
            )
        return score
    if agent == "prey":
        score = (1.0 - catch_rate) - safety_penalty
        if cross_play_aggregate:
            score = (
                0.5 * score
                + 0.3 * (1.0 - float(cross_play_aggregate["catch_avg"]))
                + 0.2 * (1.0 - float(cross_play_aggregate["catch_max"]))
            )
        return score
    return 0.0


def _pool_entry_name(agent: str, checkpoint: Path, phase_index: int) -> str:
    run_name = checkpoint.parents[1].name if len(checkpoint.parents) > 1 else checkpoint.parent.name
    raw_name = f"auto_{agent}_phase_{phase_index:03d}_{run_name}_{checkpoint.stem}"
    return _safe_token(raw_name)


def _recent_pool_entry_name(agent: str, checkpoint: Path, phase_index: int) -> str:
    run_name = checkpoint.parents[1].name if len(checkpoint.parents) > 1 else checkpoint.parent.name
    raw_name = f"recent_{agent}_phase_{phase_index:03d}_{run_name}_{checkpoint.stem}"
    return _safe_token(raw_name)


def _prune_pool(pool: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    max_entries = int(args.auto_pool_max_entries_per_role)
    max_recent_entries = int(args.recent_pool_max_entries_per_role)
    for agent in AGENTS:
        entries = pool.get(agent, [])

        elite_entries = _pool_entries_by_type(entries, POOL_TYPE_ELITE)
        recent_entries = _pool_entries_by_type(entries, POOL_TYPE_RECENT)
        other_entries = [
            entry
            for entry in entries
            if str(entry.get("pool_type", POOL_TYPE_ELITE)) not in {POOL_TYPE_ELITE, POOL_TYPE_RECENT}
        ]

        if max_entries > 0 and len(elite_entries) > max_entries:
            elite_entries.sort(
                key=lambda entry: (
                    float(entry.get("auto_score", entry.get("weight", 1.0))),
                    int(entry.get("added_phase_index", -1)),
                ),
                reverse=True,
            )
            elite_entries = elite_entries[:max_entries]

        if max_recent_entries <= 0:
            recent_entries = []
        elif len(recent_entries) > max_recent_entries:
            recent_entries.sort(
                key=lambda entry: (
                    int(entry.get("added_phase_index", -1)),
                    str(entry.get("added_at", "")),
                ),
                reverse=True,
            )
            recent_entries = recent_entries[:max_recent_entries]

        pool[agent] = elite_entries + recent_entries + other_entries


def _sort_pool_for_save(pool: dict[str, list[dict[str, Any]]]) -> None:
    for agent in AGENTS:
        pool[agent].sort(
            key=lambda entry: (
                0 if str(entry.get("pool_type", POOL_TYPE_ELITE)) == POOL_TYPE_ELITE else 1,
                int(entry.get("added_phase_index", -1)),
            ),
            reverse=False,
        )


def _maybe_promote_to_pool(
    pool: dict[str, list[dict[str, Any]]],
    *,
    agent: str,
    checkpoint: Path,
    summary: dict[str, Any],
    cross_play_aggregate: dict[str, Any] | None,
    phase_index: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.auto_pool or agent not in AGENTS:
        return None

    checkpoint = checkpoint.resolve()
    for entry in pool.get(agent, []):
        if (
            str(entry.get("pool_type", POOL_TYPE_ELITE)) == POOL_TYPE_ELITE
            and Path(str(entry["checkpoint"])).resolve() == checkpoint
        ):
            print(f"[INFO] Auto-pool skip: {agent} checkpoint already exists in pool: {checkpoint}")
            return None

    accepted, reason = _promotion_gate(agent, summary, args, cross_play_aggregate)
    if not accepted:
        print(f"[INFO] Auto-pool {reason}")
        return {"agent": agent, "promoted": False, "reason": reason, "checkpoint": str(checkpoint)}

    metrics = summary.get("episode_metrics", {})
    auto_score = _pool_score(agent, summary, cross_play_aggregate)
    pool[agent] = [
        entry
        for entry in pool.get(agent, [])
        if Path(str(entry["checkpoint"])).resolve() != checkpoint
    ]
    entry = {
        "name": _pool_entry_name(agent, checkpoint, phase_index),
        "checkpoint": str(checkpoint),
        "pool_type": POOL_TYPE_ELITE,
        "weight": 1.0,
        "base_weight": 1.0,
        "auto_score": auto_score,
        "added_phase_index": phase_index,
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "notes": reason,
    }
    if cross_play_aggregate:
        entry["cross_play_aggregate"] = cross_play_aggregate
    pool[agent].append(entry)
    _prune_pool(pool, args)
    print(f"[INFO] Auto-pool promoted {agent}: {entry['name']} (score={auto_score:.3f})")
    return {"agent": agent, "promoted": True, "entry": entry, "reason": reason}


def _recent_pool_gate(agent: str, summary: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    prey_oob = _metric(summary, "Metrics/prey_oob_rate")
    predator_oob = _metric(summary, "Metrics/predator_oob_rate")
    prey_soft = _metric(summary, "Metrics/prey_soft_arena_outside")
    predator_soft = _metric(summary, "Metrics/predator_soft_arena_outside")

    if agent == "predator":
        checks = [
            (predator_oob <= args.auto_pool_max_predator_oob, f"predator_oob={predator_oob:.3f}"),
            (predator_soft <= args.auto_pool_max_predator_soft, f"predator_soft={predator_soft:.3f}"),
            (prey_oob <= args.auto_pool_max_opponent_oob, f"prey_oob={prey_oob:.3f}"),
            (prey_soft <= args.auto_pool_max_opponent_soft, f"prey_soft={prey_soft:.3f}"),
        ]
    elif agent == "prey":
        checks = [
            (prey_oob <= args.auto_pool_max_prey_oob, f"prey_oob={prey_oob:.3f}"),
            (prey_soft <= args.auto_pool_max_prey_soft, f"prey_soft={prey_soft:.3f}"),
            (predator_oob <= args.auto_pool_max_opponent_oob, f"predator_oob={predator_oob:.3f}"),
            (predator_soft <= args.auto_pool_max_opponent_soft, f"predator_soft={predator_soft:.3f}"),
        ]
    else:
        return False, f"recent pool only supports predator/prey phases, got {agent}"

    details = ", ".join(detail for _, detail in checks)
    failed = [detail for ok, detail in checks if not ok]
    if failed:
        return False, f"rejected recent {agent}: {', '.join(failed)}; metrics: {details}"
    return True, f"accepted recent {agent}: {details}"


def _maybe_add_recent_to_pool(
    pool: dict[str, list[dict[str, Any]]],
    *,
    agent: str,
    checkpoint: Path,
    summary: dict[str, Any],
    cross_play_aggregate: dict[str, Any] | None,
    promotion: dict[str, Any] | None,
    phase_index: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.recent_pool or agent not in AGENTS:
        return None
    if promotion is not None and bool(promotion.get("promoted", False)):
        return None

    checkpoint = checkpoint.resolve()
    for entry in pool.get(agent, []):
        if Path(str(entry["checkpoint"])).resolve() == checkpoint:
            print(f"[INFO] Recent-pool skip: {agent} checkpoint already exists in pool: {checkpoint}")
            return None

    accepted, reason = _recent_pool_gate(agent, summary, args)
    if not accepted:
        print(f"[INFO] Recent-pool {reason}")
        return {"agent": agent, "added": False, "reason": reason, "checkpoint": str(checkpoint)}

    metrics = summary.get("episode_metrics", {})
    recent_score = _pool_score(agent, summary, cross_play_aggregate)
    entry = {
        "name": _recent_pool_entry_name(agent, checkpoint, phase_index),
        "checkpoint": str(checkpoint),
        "pool_type": POOL_TYPE_RECENT,
        "weight": float(args.recent_pool_weight),
        "base_weight": float(args.recent_pool_weight),
        "recent_score": recent_score,
        "added_phase_index": phase_index,
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "notes": reason,
    }
    if cross_play_aggregate:
        entry["cross_play_aggregate"] = cross_play_aggregate
    pool[agent].append(entry)
    _prune_pool(pool, args)
    print(f"[INFO] Recent-pool added {agent}: {entry['name']} (score={recent_score:.3f})")
    return {"agent": agent, "added": True, "entry": entry, "reason": reason}


def _selected_cross_play_entries(
    pool: dict[str, list[dict[str, Any]]],
    *,
    opponent: str,
    training_agent: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in pool.get(opponent, [])
        if _pool_entry_compatible(entry, role=opponent, args=args)
        and (
            args.cross_play_include_recent_pool
            or str(entry.get("pool_type", POOL_TYPE_ELITE)) == POOL_TYPE_ELITE
        )
    ]
    if not entries or args.cross_play_max_opponents <= 0:
        return []

    weights = [
        max(0.0, _pool_entry_weight(entry, sampled_agent=opponent, training_agent=training_agent, args=args))
        for entry in entries
    ]
    max_count = min(int(args.cross_play_max_opponents), len(entries))

    if args.cross_play_selection == "top":
        ranked = sorted(zip(entries, weights), key=lambda item: item[1], reverse=True)
        return [entry for entry, weight in ranked[:max_count] if weight > 0.0]

    selected: list[dict[str, Any]] = []
    available = list(zip(entries, weights))
    for _ in range(max_count):
        positive = [(entry, weight) for entry, weight in available if weight > 0.0]
        if not positive:
            break
        chosen = rng.choices([entry for entry, _ in positive], weights=[weight for _, weight in positive], k=1)[0]
        selected.append(chosen)
        available = [(entry, weight) for entry, weight in available if entry is not chosen]
    return selected


def _training_agent_win_rate(training_agent: str, catch_rate: float) -> float:
    if training_agent == "predator":
        return catch_rate
    if training_agent == "prey":
        return 1.0 - catch_rate
    return 0.5


def _update_cross_play_weight(
    entry: dict[str, Any],
    *,
    opponent: str,
    training_agent: str,
    current_checkpoint: Path,
    composed_checkpoint: Path,
    summary: dict[str, Any],
    phase_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = summary.get("episode_metrics", {})
    catch_rate = _metric(summary, "Metrics/catch_rate")
    win_rate = _training_agent_win_rate(training_agent, catch_rate)
    pfsp_multiplier = max(float(args.pfsp_min_weight), _pfsp_weight(win_rate, args.pfsp_weighting))
    base_weight = float(entry.get("base_weight", entry.get("weight", 1.0)))
    effective_weight = base_weight * pfsp_multiplier

    entry["base_weight"] = base_weight
    entry["pfsp_multiplier"] = pfsp_multiplier
    entry["weight"] = effective_weight
    entry["last_cross_play"] = {
        "phase_index": phase_index,
        "training_agent": training_agent,
        "opponent": opponent,
        "current_checkpoint": str(current_checkpoint),
        "composed_checkpoint": str(composed_checkpoint),
        "training_agent_win_rate": win_rate,
        "metrics": metrics,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {
        "opponent": opponent,
        "entry_name": entry.get("name"),
        "entry_checkpoint": entry.get("checkpoint"),
        "composed_checkpoint": str(composed_checkpoint),
        "metrics": metrics,
        "training_agent_win_rate": win_rate,
        "pfsp_multiplier": pfsp_multiplier,
        "effective_weight": effective_weight,
    }


def _cross_play_aggregate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None

    catch_rates = [
        float(record.get("metrics", {}).get("Metrics/catch_rate", 0.0))
        for record in records
    ]
    win_rates = [float(record.get("training_agent_win_rate", 0.5)) for record in records]
    return {
        "count": len(records),
        "catch_avg": sum(catch_rates) / len(catch_rates),
        "catch_min": min(catch_rates),
        "catch_max": max(catch_rates),
        "training_win_avg": sum(win_rates) / len(win_rates),
        "training_win_min": min(win_rates),
        "training_win_max": max(win_rates),
        "opponents": [
            {
                "name": record.get("entry_name"),
                "checkpoint": record.get("entry_checkpoint"),
                "catch_rate": float(record.get("metrics", {}).get("Metrics/catch_rate", 0.0)),
                "training_agent_win_rate": float(record.get("training_agent_win_rate", 0.5)),
            }
            for record in records
        ],
    }


def _run_cross_play(
    current_checkpoint: Path,
    *,
    phase: str,
    phase_index: int,
    pool: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.cross_play or phase not in {"predator", "prey"}:
        return []
    if args.cross_play_every <= 0 or phase_index % args.cross_play_every != 0:
        return []

    opponent = _agent_opponent(phase)
    entries = _selected_cross_play_entries(
        pool,
        opponent=opponent,
        training_agent=phase,
        rng=rng,
        args=args,
    )
    if not entries:
        print(f"[INFO] Cross-play skipped: no {opponent} pool entries available.")
        return []

    records = []
    for entry in entries:
        opponent_checkpoint = Path(str(entry["checkpoint"]))
        safe_name = _safe_token(entry.get("name", opponent_checkpoint.stem))
        output = args.output_dir / "composed" / f"phase_{phase_index:03d}_cross_{phase}_vs_{opponent}_{safe_name}.pt"

        print(
            f"[INFO] Cross-play phase {phase_index}: latest {phase} vs "
            f"{opponent}='{entry.get('name', opponent_checkpoint.stem)}'"
        )
        if phase == "predator":
            composed_checkpoint = _compose_checkpoint(
                base=current_checkpoint,
                predator=current_checkpoint,
                prey=opponent_checkpoint,
                output=output,
                args=args,
            )
        else:
            composed_checkpoint = _compose_checkpoint(
                base=current_checkpoint,
                predator=opponent_checkpoint,
                prey=current_checkpoint,
                output=output,
                args=args,
            )

        summary = _evaluate(
            composed_checkpoint,
            phase_index=phase_index,
            output_dir=args.output_dir,
            args=args,
            label=f"cross_{phase}_vs_{opponent}_{safe_name}",
            num_envs=args.cross_play_num_envs,
            episodes=args.cross_play_episodes,
        )
        records.append(
            _update_cross_play_weight(
                entry,
                opponent=opponent,
                training_agent=phase,
                current_checkpoint=current_checkpoint,
                composed_checkpoint=composed_checkpoint,
                summary=summary,
                phase_index=phase_index,
                args=args,
            )
        )

    return records


def _cross_play_robustness(
    agent: str,
    aggregate: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[bool | None, str]:
    if not args.phase_decision_cross_play:
        return None, "cross-play phase decisions disabled"
    if aggregate is None:
        return None, f"{agent} cross-play unavailable"

    count = int(aggregate.get("count", 0))
    if count < args.phase_decision_cross_play_min_count:
        return None, (
            f"{agent} cross-play count={count} < "
            f"{args.phase_decision_cross_play_min_count}"
        )

    catch_avg = float(aggregate["catch_avg"])
    catch_min = float(aggregate["catch_min"])
    catch_max = float(aggregate["catch_max"])

    if agent == "predator":
        avg_ok = catch_avg >= args.auto_pool_predator_min_cross_play_avg_catch
        min_ok = catch_min >= args.auto_pool_predator_min_cross_play_min_catch
        robust = avg_ok and min_ok
        return robust, (
            f"predator_cross_avg={catch_avg:.3f}, "
            f"predator_cross_min={catch_min:.3f}, "
            f"robust={robust}"
        )

    if agent == "prey":
        avg_ok = catch_avg <= args.auto_pool_prey_max_cross_play_avg_catch
        max_ok = catch_max <= args.auto_pool_prey_max_cross_play_max_catch
        robust = avg_ok and max_ok
        return robust, (
            f"prey_cross_avg={catch_avg:.3f}, "
            f"prey_cross_max={catch_max:.3f}, "
            f"robust={robust}"
        )

    raise ValueError(f"Unknown agent: {agent}")


def _safety_repair_decision(summary: dict[str, Any], args: argparse.Namespace) -> tuple[str, str] | None:
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
    return None


def _target_score_for_agent(agent: str, summary: dict[str, Any]) -> float:
    catch_rate = _metric(summary, "Metrics/clean_catch_rate", _metric(summary, "Metrics/catch_rate"))
    if agent == "predator":
        return catch_rate
    if agent == "prey":
        return 1.0 - catch_rate
    return 0.0


def _promotion_failure_context(
    agent: str,
    summary: dict[str, Any],
    cross_play_aggregate: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[str, float]:
    pool_robust, pool_reason = _cross_play_robustness(agent, cross_play_aggregate, args)
    default_prob = float(args.per_env_pool_prob)
    if pool_robust is False:
        return (
            f"{agent} failed elite promotion because pool cross-play is weak; "
            f"{pool_reason}; using pool-focused per-env exposure",
            float(args.pool_focus_per_env_pool_prob),
        )
    if pool_robust is True:
        return (
            f"{agent} failed elite promotion while elite-pool cross-play is robust; "
            f"{pool_reason}; using latest-focused per-env exposure",
            float(args.latest_focus_per_env_pool_prob),
        )
    return (
        f"{agent} failed elite promotion without enough cross-play context; "
        f"{pool_reason}; using configured per-env exposure",
        default_prob,
    )


def _promotion_failure_override(
    *,
    phase: str,
    summary: dict[str, Any],
    promotion: dict[str, Any] | None,
    cross_play_aggregate: dict[str, Any] | None,
    retry_state: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.promotion_failure_phase_continuation or phase not in AGENTS:
        return None
    if promotion is None or bool(promotion.get("promoted", False)):
        retry_state.pop(phase, None)
        return None

    score = _target_score_for_agent(phase, summary)
    state = retry_state.get(phase, {"best_score": None, "stale_repeats": 0})
    best_score = state.get("best_score")
    if best_score is None or score > float(best_score) + float(args.promotion_failure_plateau_delta):
        stale_repeats = 0
        best_score = score
    else:
        stale_repeats = int(state.get("stale_repeats", 0)) + 1
    retry_state[phase] = {"best_score": best_score, "stale_repeats": stale_repeats}

    if stale_repeats >= int(args.promotion_failure_max_stale_repeats):
        return {
            "phase": args.promotion_failure_plateau_action,
            "reason": (
                f"{phase} promotion failures plateaued: score={score:.3f}, "
                f"best={float(best_score):.3f}, stale_repeats={stale_repeats}; "
                f"using plateau action={args.promotion_failure_plateau_action}"
            ),
            "per_env_pool_prob": float(args.per_env_pool_prob),
            "plateau": True,
        }

    reason, pool_prob = _promotion_failure_context(phase, summary, cross_play_aggregate, args)
    return {
        "phase": phase,
        "reason": (
            f"repeat {phase} after failed elite promotion: score={score:.3f}, "
            f"best={float(best_score):.3f}, stale_repeats={stale_repeats}; {reason}"
        ),
        "per_env_pool_prob": pool_prob,
        "plateau": False,
    }


def _decide_phase(
    summary: dict[str, Any],
    args: argparse.Namespace,
    cross_play_by_agent: dict[str, dict[str, Any] | None] | None = None,
) -> tuple[str, str]:
    catch_rate = _metric(summary, "Metrics/catch_rate")
    prey_oob = _metric(summary, "Metrics/prey_oob_rate")
    predator_oob = _metric(summary, "Metrics/predator_oob_rate")
    prey_soft = _metric(summary, "Metrics/prey_soft_arena_outside")
    predator_soft = _metric(summary, "Metrics/predator_soft_arena_outside")
    cross_play_by_agent = cross_play_by_agent or {}
    predator_robust, predator_reason = _cross_play_robustness(
        "predator", cross_play_by_agent.get("predator"), args
    )
    prey_robust, prey_reason = _cross_play_robustness(
        "prey", cross_play_by_agent.get("prey"), args
    )

    safety_decision = _safety_repair_decision(summary, args)
    if safety_decision is not None:
        return safety_decision

    if predator_robust is not None and prey_robust is not None:
        if predator_robust == prey_robust:
            state = "both sides are pool-robust" if predator_robust else "neither side is pool-robust"
            return "both", (
                f"{state}; train both with pool: "
                f"{predator_reason}; {prey_reason}"
            )
        if predator_robust is False:
            return "predator", (
                "only predator is not pool-robust: "
                f"{predator_reason}; {prey_reason}"
            )
        return "prey", (
            "only prey is not pool-robust: "
            f"{predator_reason}; {prey_reason}"
        )

    if predator_robust is False:
        return "predator", (
            "predator is not pool-robust and prey robustness is unavailable: "
            f"{predator_reason}; {prey_reason}"
        )
    if prey_robust is False:
        return "prey", (
            "prey is not pool-robust and predator robustness is unavailable: "
            f"{predator_reason}; {prey_reason}"
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


def _requested_phase_iterations(phase: str, args: argparse.Namespace) -> int:
    if phase == "predator" and args.predator_phase_iterations is not None:
        return args.predator_phase_iterations
    if phase == "prey" and args.prey_phase_iterations is not None:
        return args.prey_phase_iterations
    if phase == "both" and args.both_phase_iterations is not None:
        return args.both_phase_iterations
    return args.phase_iterations


def _per_env_pool_path(args: argparse.Namespace, per_env_pool_prob: float) -> Path | None:
    if per_env_pool_prob <= 0.0:
        return None
    if args.auto_pool_path is not None:
        return args.auto_pool_path
    return args.opponent_pool


def _evaluate(
    checkpoint: Path,
    *,
    phase_index: int,
    output_dir: Path,
    args: argparse.Namespace,
    label: str,
    num_envs: int | None = None,
    episodes: int | None = None,
) -> dict[str, Any]:
    json_path = output_dir / f"phase_{phase_index:03d}_{label}_eval.json"
    eval_num_envs = args.eval_num_envs if num_envs is None else num_envs
    eval_episodes = args.eval_episodes if episodes is None else episodes
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
        str(eval_num_envs),
        "--episodes",
        str(eval_episodes),
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
    frozen_source_checkpoint: Path | None = None,
    per_env_pool_prob: float | None = None,
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
    phase_per_env_pool_prob = args.per_env_pool_prob if per_env_pool_prob is None else per_env_pool_prob
    per_env_pool_path = _per_env_pool_path(args, phase_per_env_pool_prob)
    if phase in {"predator", "prey"} and per_env_pool_path is not None:
        command.extend(
            [
                "--per-env-opponent-pool",
                str(per_env_pool_path),
                "--per-env-pool-prob",
                str(phase_per_env_pool_prob),
                "--per-env-pool-max-policies",
                str(args.per_env_pool_max_policies),
                "--per-env-pool-seed",
                str(args.per_env_pool_seed if args.per_env_pool_seed is not None else args.seed),
            ]
        )
        if frozen_source_checkpoint is not None:
            command.extend(["--per-env-opponent-pool-exclude-checkpoint", str(frozen_source_checkpoint)])
    _run(command, cwd=REPO_ROOT, dry_run=args.dry_run)
    if args.dry_run:
        return checkpoint
    run_dir = _latest_run_dir(args.run_root, before)
    next_checkpoint = _latest_checkpoint(run_dir)
    print(f"[INFO] Phase run directory: {run_dir}")
    print(f"[INFO] Next checkpoint: {next_checkpoint}")
    return next_checkpoint


def _record_phase_video(
    checkpoint: Path,
    *,
    phase_index: int,
    args: argparse.Namespace,
    label: str,
) -> None:
    if not args.record_phase_video:
        return
    if args.phase_video_every <= 0 or phase_index % args.phase_video_every != 0:
        return

    video_dir = args.output_dir / "videos" / f"phase_{phase_index:03d}_{label}"
    command = [
        str(args.isaaclab),
        "-p",
        str(REPO_ROOT / "scripts" / "skrl" / "play.py"),
        "--headless",
        "--video",
        "--video_length",
        str(args.phase_video_length),
        "--video-dir",
        str(video_dir),
        "--task",
        args.task,
        "--agent",
        args.agent,
        "--algorithm",
        args.algorithm,
        "--num_envs",
        str(args.phase_video_num_envs),
        "--seed",
        str(args.seed),
        "--checkpoint",
        str(checkpoint),
        "--camera-eye",
        *(str(value) for value in args.phase_video_camera_eye),
        "--camera-target",
        *(str(value) for value in args.phase_video_camera_target),
        "--camera-env-index",
        str(args.phase_video_camera_env_index),
    ]
    _run(command, cwd=REPO_ROOT, dry_run=args.dry_run)


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
    compatible_pool = _compatible_pool_for_role(pool, role=opponent, args=args)
    entry = _sample_pool_entry(compatible_pool, opponent, rng, training_agent=phase, args=args)
    if entry is None:
        print(f"[INFO] No {opponent} pool entries available; using current opponent.")
        return current_checkpoint, None

    pool_checkpoint = Path(entry["checkpoint"])
    safe_name = _safe_token(entry["name"])
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
    parser.add_argument(
        "--predator-source-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint that represents the predator weights inside --checkpoint. "
            "Defaults to --checkpoint; useful when resuming from a checkpoint whose prey changed last."
        ),
    )
    parser.add_argument(
        "--prey-source-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint that represents the prey weights inside --checkpoint. "
            "Defaults to --checkpoint; useful when resuming from a checkpoint whose predator changed last."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("none", *PRESETS.keys()),
        default="none",
        help="Apply task/agent defaults for a known curriculum setup.",
    )
    parser.add_argument("--total-iterations", type=int, default=20_000, help="Total outer-loop training iterations.")
    parser.add_argument("--phase-iterations", type=int, default=500, help="Iterations per train phase.")
    parser.add_argument(
        "--predator-phase-iterations",
        type=int,
        default=None,
        help="Optional iteration count for predator-only phases. Defaults to --phase-iterations.",
    )
    parser.add_argument(
        "--prey-phase-iterations",
        type=int,
        default=None,
        help="Optional iteration count for prey-only phases. Defaults to --phase-iterations.",
    )
    parser.add_argument(
        "--both-phase-iterations",
        type=int,
        default=None,
        help="Optional iteration count for both-agent phases. Defaults to --phase-iterations.",
    )
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
        "--per-env-pool-prob",
        type=float,
        default=0.0,
        help=(
            "Fraction of envs in a single-agent phase that use frozen opponents sampled from the pool. "
            "This mixes opponents inside one rollout batch."
        ),
    )
    parser.add_argument(
        "--per-env-pool-max-policies",
        type=int,
        default=8,
        help="Maximum number of pool policies loaded into train.py for per-env opponent mixing.",
    )
    parser.add_argument(
        "--per-env-pool-seed",
        type=int,
        default=None,
        help="Seed for per-env opponent assignment. Defaults to --seed.",
    )
    parser.add_argument(
        "--pool-sampling",
        choices=("static", "pfsp"),
        default="static",
        help="Pool sampling rule. 'pfsp' reweights entries by stored win-rate difficulty.",
    )
    parser.add_argument(
        "--pfsp-weighting",
        choices=("linear", "squared", "variance"),
        default="squared",
        help="PFSP weighting function applied to the training agent win-rate proxy.",
    )
    parser.add_argument(
        "--pfsp-min-weight",
        type=float,
        default=0.05,
        help="Minimum PFSP multiplier so hard/easy opponents are not fully discarded.",
    )
    parser.add_argument(
        "--pool-seed",
        type=int,
        default=None,
        help="Seed for opponent-pool sampling. Defaults to --seed.",
    )
    parser.add_argument(
        "--auto-pool",
        action="store_true",
        help="Automatically promote good post-phase checkpoints into an evolving opponent pool.",
    )
    parser.add_argument(
        "--auto-pool-path",
        type=Path,
        default=None,
        help="Path to write the evolving auto-pool JSON. Defaults to <output-dir>/opponent_pool_auto.json.",
    )
    parser.add_argument(
        "--auto-pool-max-entries-per-role",
        type=int,
        default=12,
        help="Maximum number of auto-pool entries to keep per role.",
    )
    parser.add_argument(
        "--recent-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep a small safe-but-not-elite recent pool in the same pool JSON. "
            "Recent entries are used by per-env pool training but excluded from "
            "cross-play promotion gates by default."
        ),
    )
    parser.add_argument(
        "--recent-pool-max-entries-per-role",
        type=int,
        default=2,
        help="Maximum number of safe recent entries to keep per role.",
    )
    parser.add_argument(
        "--recent-pool-weight",
        type=float,
        default=0.25,
        help="Base sampling weight assigned to safe recent-pool entries.",
    )
    parser.add_argument(
        "--auto-pool-predator-min-catch",
        type=float,
        default=0.75,
        help="Minimum catch-rate for promoting a predator checkpoint.",
    )
    parser.add_argument(
        "--auto-pool-prey-max-catch",
        type=float,
        default=0.25,
        help="Maximum catch-rate for promoting a prey checkpoint.",
    )
    parser.add_argument(
        "--auto-pool-max-predator-oob",
        type=float,
        default=0.08,
        help="Maximum predator OOB rate for promotion gates.",
    )
    parser.add_argument(
        "--auto-pool-max-prey-oob",
        type=float,
        default=0.08,
        help="Maximum prey OOB rate for promotion gates.",
    )
    parser.add_argument(
        "--auto-pool-max-predator-soft",
        type=float,
        default=0.30,
        help="Maximum predator soft-arena outside metric for promotion gates.",
    )
    parser.add_argument(
        "--auto-pool-max-prey-soft",
        type=float,
        default=0.30,
        help="Maximum prey soft-arena outside metric for promotion gates.",
    )
    parser.add_argument(
        "--auto-pool-max-opponent-oob",
        type=float,
        default=0.20,
        help="Maximum opponent OOB rate allowed when promoting a candidate.",
    )
    parser.add_argument(
        "--auto-pool-max-opponent-soft",
        type=float,
        default=0.50,
        help="Maximum opponent soft-arena outside metric allowed when promoting a candidate.",
    )
    parser.add_argument(
        "--auto-pool-predator-min-cross-play-avg-catch",
        type=float,
        default=0.55,
        help="Minimum average catch-rate against sampled prey pool opponents for predator promotion.",
    )
    parser.add_argument(
        "--auto-pool-predator-min-cross-play-min-catch",
        type=float,
        default=0.25,
        help="Minimum worst sampled catch-rate against prey pool opponents for predator promotion.",
    )
    parser.add_argument(
        "--auto-pool-prey-max-cross-play-avg-catch",
        type=float,
        default=0.50,
        help="Maximum average predator catch-rate against sampled predator pool opponents for prey promotion.",
    )
    parser.add_argument(
        "--auto-pool-prey-max-cross-play-max-catch",
        type=float,
        default=0.80,
        help="Maximum worst sampled predator catch-rate against predator pool opponents for prey promotion.",
    )
    parser.add_argument(
        "--cross-play",
        action="store_true",
        help="Evaluate latest trained role against selected pool opponents after each phase and update PFSP weights.",
    )
    parser.add_argument(
        "--cross-play-every",
        type=int,
        default=1,
        help="Run cross-play every N phase evaluations when --cross-play is enabled.",
    )
    parser.add_argument(
        "--cross-play-max-opponents",
        type=int,
        default=4,
        help="Maximum number of pool opponents to cross-play after a single-agent phase.",
    )
    parser.add_argument(
        "--cross-play-selection",
        choices=("weighted", "top"),
        default="weighted",
        help="How to select cross-play opponents from the relevant pool role.",
    )
    parser.add_argument(
        "--cross-play-include-recent-pool",
        action="store_true",
        help="Include recent-pool entries in cross-play gates. Default is elite-only cross-play.",
    )
    parser.add_argument(
        "--cross-play-num-envs",
        type=int,
        default=256,
        help="Number of vectorized envs for each cross-play evaluation.",
    )
    parser.add_argument(
        "--cross-play-episodes",
        type=int,
        default=256,
        help="Completed episodes for each cross-play evaluation.",
    )
    parser.add_argument(
        "--phase-decision-cross-play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use latest per-role cross-play aggregates to override current-pair "
            "catch-rate phase decisions when a side is not pool-robust."
        ),
    )
    parser.add_argument(
        "--phase-decision-cross-play-min-count",
        type=int,
        default=2,
        help="Minimum sampled pool opponents required before cross-play can steer phase decisions.",
    )
    parser.add_argument(
        "--promotion-failure-phase-continuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a safe but non-promoted single-agent phase, prefer repeating that role. "
            "The per-env pool probability is adapted from the cross-play failure context."
        ),
    )
    parser.add_argument(
        "--promotion-failure-max-stale-repeats",
        type=int,
        default=2,
        help="Maximum stale failed-promotion repeats before using the plateau action.",
    )
    parser.add_argument(
        "--promotion-failure-plateau-delta",
        type=float,
        default=0.03,
        help="Minimum target-score improvement needed to avoid counting a failed-promotion repeat as stale.",
    )
    parser.add_argument(
        "--promotion-failure-plateau-action",
        choices=("both", "predator", "prey"),
        default="both",
        help="Phase to request after repeated failed-promotion plateau.",
    )
    parser.add_argument(
        "--latest-focus-per-env-pool-prob",
        type=float,
        default=0.25,
        help="Per-env pool probability when elite pool is robust but the latest opponent is still hard.",
    )
    parser.add_argument(
        "--pool-focus-per-env-pool-prob",
        type=float,
        default=0.75,
        help="Per-env pool probability when cross-play shows the candidate is not robust to elite pool opponents.",
    )
    parser.add_argument("--task", default="1v1-survival-soft-oob-v0", help="Isaac Lab task id.")
    parser.add_argument("--agent", default="skrl_mappo_finetune_cfg_entry_point", help="SKRL agent config entry point.")
    parser.add_argument("--algorithm", default="MAPPO", help="SKRL algorithm.")
    parser.add_argument("--train-num-envs", type=int, default=4096, help="Number of training envs.")
    parser.add_argument("--eval-num-envs", type=int, default=512, help="Number of evaluation envs.")
    parser.add_argument("--eval-episodes", type=int, default=512, help="Deterministic eval episodes per phase.")
    parser.add_argument(
        "--record-phase-video",
        action="store_true",
        help="Record one fixed-camera video after each selected curriculum phase.",
    )
    parser.add_argument(
        "--phase-video-every",
        type=int,
        default=1,
        help="Record a video every N completed phases when --record-phase-video is enabled.",
    )
    parser.add_argument(
        "--phase-video-length",
        type=int,
        default=500,
        help="Number of environment steps per phase video.",
    )
    parser.add_argument(
        "--phase-video-num-envs",
        type=int,
        default=1,
        help="Number of environments for phase video recording. Use 1 for stable camera framing.",
    )
    parser.add_argument(
        "--phase-video-camera-eye",
        type=float,
        nargs=3,
        default=(7.0, -7.0, 6.0),
        metavar=("X", "Y", "Z"),
        help="Env-relative camera eye used for phase videos.",
    )
    parser.add_argument(
        "--phase-video-camera-target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Env-relative camera target used for phase videos.",
    )
    parser.add_argument(
        "--phase-video-camera-env-index",
        type=int,
        default=0,
        help="Environment index used as origin for phase video camera framing.",
    )
    parser.add_argument(
        "--skip-pool-opponent-eval",
        action="store_true",
        help="Skip the extra deterministic eval of a trained pool-composed checkpoint before restoring the current pair.",
    )
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

    preset = PRESETS.get(args.preset)
    if preset is not None:
        if not _cli_arg_present("--task"):
            args.task = preset["task"]
        if not _cli_arg_present("--agent"):
            args.agent = preset["agent"]
        if not _cli_arg_present("--algorithm"):
            args.algorithm = preset["algorithm"]

    args.checkpoint = args.checkpoint.resolve()
    args.predator_source_checkpoint = (
        args.predator_source_checkpoint.resolve() if args.predator_source_checkpoint is not None else None
    )
    args.prey_source_checkpoint = args.prey_source_checkpoint.resolve() if args.prey_source_checkpoint is not None else None
    args.isaaclab = args.isaaclab.resolve()
    args.run_root = args.run_root.resolve()
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = preset["output_suffix"] if preset is not None else "hysteresis"
        args.output_dir = REPO_ROOT / "logs" / "curriculum" / f"{timestamp}_{suffix}"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.phase_iterations = max(1, int(args.phase_iterations))
    args.predator_phase_iterations = (
        None if args.predator_phase_iterations is None else max(1, int(args.predator_phase_iterations))
    )
    args.prey_phase_iterations = None if args.prey_phase_iterations is None else max(1, int(args.prey_phase_iterations))
    args.both_phase_iterations = None if args.both_phase_iterations is None else max(1, int(args.both_phase_iterations))
    args.pool_prob = max(0.0, min(1.0, args.pool_prob))
    args.per_env_pool_prob = max(0.0, min(1.0, args.per_env_pool_prob))
    args.per_env_pool_max_policies = max(1, int(args.per_env_pool_max_policies))
    args.pfsp_min_weight = max(0.0, float(args.pfsp_min_weight))
    args.recent_pool_max_entries_per_role = max(0, int(args.recent_pool_max_entries_per_role))
    args.recent_pool_weight = max(0.0, float(args.recent_pool_weight))
    args.promotion_failure_max_stale_repeats = max(1, int(args.promotion_failure_max_stale_repeats))
    args.promotion_failure_plateau_delta = max(0.0, float(args.promotion_failure_plateau_delta))
    args.latest_focus_per_env_pool_prob = max(0.0, min(1.0, args.latest_focus_per_env_pool_prob))
    args.pool_focus_per_env_pool_prob = max(0.0, min(1.0, args.pool_focus_per_env_pool_prob))
    args.cross_play_every = max(1, int(args.cross_play_every))
    args.cross_play_max_opponents = max(0, int(args.cross_play_max_opponents))
    args.cross_play_num_envs = max(1, int(args.cross_play_num_envs))
    args.cross_play_episodes = max(1, int(args.cross_play_episodes))
    args.phase_decision_cross_play_min_count = max(1, int(args.phase_decision_cross_play_min_count))
    args.phase_video_every = max(1, int(args.phase_video_every))
    args.phase_video_length = max(1, int(args.phase_video_length))
    args.phase_video_num_envs = max(1, int(args.phase_video_num_envs))
    pool_path = args.opponent_pool.resolve() if args.opponent_pool else None
    pool = _load_opponent_pool(pool_path)
    if args.auto_pool or args.cross_play:
        args.auto_pool_path = (
            args.auto_pool_path.resolve()
            if args.auto_pool_path is not None
            else args.output_dir / "opponent_pool_auto.json"
        )
        _save_opponent_pool(args.auto_pool_path, pool)
    rng = random.Random(args.pool_seed if args.pool_seed is not None else args.seed)
    if pool_path is not None:
        print(f"[INFO] Loaded opponent pool: {pool_path}")
        print(f"[INFO] Predator entries: {len(pool['predator'])}; prey entries: {len(pool['prey'])}")
        print(f"[INFO] Pool sampling probability: {args.pool_prob:.3f}")
        print(f"[INFO] Pool sampling rule: {args.pool_sampling}")
    if args.per_env_pool_prob > 0.0:
        per_env_pool_path = _per_env_pool_path(args, args.per_env_pool_prob)
        print(
            "[INFO] Per-env pool mixing: "
            f"prob={args.per_env_pool_prob:.3f}, max_policies={args.per_env_pool_max_policies}, "
            f"path={per_env_pool_path}"
        )
    if args.auto_pool:
        print(f"[INFO] Auto-pool enabled: {args.auto_pool_path}")
    elif args.cross_play:
        print(f"[INFO] Cross-play pool output: {args.auto_pool_path}")
    if args.cross_play:
        print(
            "[INFO] Cross-play enabled: "
            f"every={args.cross_play_every}, max_opponents={args.cross_play_max_opponents}, "
            f"num_envs={args.cross_play_num_envs}, episodes={args.cross_play_episodes}"
        )
    if args.phase_decision_cross_play:
        print(
            "[INFO] Cross-play phase decisions enabled: "
            f"min_count={args.phase_decision_cross_play_min_count}"
        )
    if args.recent_pool:
        print(
            "[INFO] Recent pool enabled: "
            f"max_entries_per_role={args.recent_pool_max_entries_per_role}, "
            f"weight={args.recent_pool_weight:.3f}, "
            f"cross_play_include_recent={args.cross_play_include_recent_pool}"
        )
    if args.promotion_failure_phase_continuation:
        print(
            "[INFO] Promotion-failure continuation enabled: "
            f"latest_focus_prob={args.latest_focus_per_env_pool_prob:.3f}, "
            f"pool_focus_prob={args.pool_focus_per_env_pool_prob:.3f}, "
            f"max_stale_repeats={args.promotion_failure_max_stale_repeats}, "
            f"plateau_delta={args.promotion_failure_plateau_delta:.3f}, "
            f"plateau_action={args.promotion_failure_plateau_action}"
        )
    print(
        "[INFO] Phase iterations: "
        f"default={args.phase_iterations}, "
        f"predator={args.predator_phase_iterations or args.phase_iterations}, "
        f"prey={args.prey_phase_iterations or args.phase_iterations}, "
        f"both={args.both_phase_iterations or args.phase_iterations}"
    )

    current_checkpoint = args.checkpoint
    role_source_checkpoints: dict[str, Path] = {
        "predator": args.predator_source_checkpoint or current_checkpoint,
        "prey": args.prey_source_checkpoint or current_checkpoint,
    }
    completed_iterations = 0
    phase_index = 0
    summary = _evaluate(current_checkpoint, phase_index=phase_index, output_dir=args.output_dir, args=args, label="start")
    cross_play_by_agent: dict[str, dict[str, Any] | None] = {agent: None for agent in AGENTS}
    promotion_retry_state: dict[str, dict[str, Any]] = {}
    next_phase_override: dict[str, Any] | None = None

    while completed_iterations < args.total_iterations:
        safety_decision = _safety_repair_decision(summary, args)
        if safety_decision is not None:
            phase, reason = safety_decision
            phase_per_env_pool_prob = args.per_env_pool_prob
            if next_phase_override is not None:
                print(
                    "[INFO] Ignoring promotion-failure phase override because safety repair is required: "
                    f"{next_phase_override}"
                )
                next_phase_override = None
        elif next_phase_override is not None:
            phase = str(next_phase_override["phase"])
            reason = str(next_phase_override["reason"])
            phase_per_env_pool_prob = float(next_phase_override.get("per_env_pool_prob", args.per_env_pool_prob))
            next_phase_override = None
        else:
            phase, reason = _decide_phase(summary, args, cross_play_by_agent)
            phase_per_env_pool_prob = args.per_env_pool_prob
        catch_rate = _metric(summary, "Metrics/catch_rate")
        prey_oob = _metric(summary, "Metrics/prey_oob_rate")
        predator_oob = _metric(summary, "Metrics/predator_oob_rate")
        requested_phase_iterations = None if phase == "stop" else _requested_phase_iterations(phase, args)
        phase_iterations = (
            None
            if requested_phase_iterations is None
            else min(requested_phase_iterations, args.total_iterations - completed_iterations)
        )
        print(
            "\n[DECISION] "
            f"phase={phase}, reason={reason}, catch={catch_rate:.3f}, "
            f"prey_oob={prey_oob:.3f}, predator_oob={predator_oob:.3f}, "
            f"per_env_pool_prob={phase_per_env_pool_prob:.3f}"
        )
        _write_record(
            args.output_dir,
            {
                "phase_index": phase_index,
                "completed_iterations": completed_iterations,
                "checkpoint": str(current_checkpoint),
                "decision": phase,
                "reason": reason,
                "requested_phase_iterations": requested_phase_iterations,
                "actual_phase_iterations": phase_iterations,
                "per_env_pool_prob": phase_per_env_pool_prob,
                "metrics": summary.get("episode_metrics", {}),
                "phase_decision_cross_play": cross_play_by_agent,
                "role_source_checkpoints": {
                    agent: str(checkpoint) for agent, checkpoint in role_source_checkpoints.items()
                },
            },
        )

        if phase == "stop":
            print("[INFO] Balanced band reached and --balanced-action=stop. Stopping.")
            break

        phase_start_checkpoint = current_checkpoint
        phase_start_role_sources = dict(role_source_checkpoints)
        train_checkpoint, pool_sample = _pool_train_checkpoint(
            current_checkpoint,
            phase=phase,
            phase_index=phase_index,
            pool=pool,
            rng=rng,
            args=args,
        )
        frozen_role = _agent_opponent(phase) if phase in AGENTS else None
        frozen_source_checkpoint = None
        if frozen_role is not None:
            frozen_source_checkpoint = (
                Path(str(pool_sample["entry"]["checkpoint"]))
                if pool_sample is not None
                else phase_start_role_sources[frozen_role]
            )
        trained_checkpoint = _train_phase(
            train_checkpoint,
            phase=phase,
            phase_iterations=phase_iterations,
            args=args,
            frozen_source_checkpoint=frozen_source_checkpoint,
            per_env_pool_prob=phase_per_env_pool_prob,
        )
        pool_opponent_summary = None
        if pool_sample is not None and not args.skip_pool_opponent_eval:
            pool_opponent_summary = _evaluate(
                trained_checkpoint,
                phase_index=phase_index,
                output_dir=args.output_dir,
                args=args,
                label=f"pool_{phase}_opponent_after",
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
                    "requested_phase_iterations": requested_phase_iterations,
                    "actual_phase_iterations": phase_iterations,
                    "pool_sample": pool_sample,
                    "trained_checkpoint": str(trained_checkpoint),
                    "restored_current_pair_checkpoint": str(current_checkpoint),
                    "frozen_source_checkpoint": (
                        str(frozen_source_checkpoint) if frozen_source_checkpoint is not None else None
                    ),
                    "pool_opponent_metrics": (
                        pool_opponent_summary.get("episode_metrics", {}) if pool_opponent_summary else None
                    ),
                },
            )
        if phase in AGENTS:
            role_source_checkpoints[phase] = current_checkpoint
            role_source_checkpoints[_agent_opponent(phase)] = phase_start_role_sources[_agent_opponent(phase)]
        elif phase == "both":
            role_source_checkpoints = {agent: current_checkpoint for agent in AGENTS}
        completed_iterations += phase_iterations
        phase_index += 1
        summary = _evaluate(
            current_checkpoint,
            phase_index=phase_index,
            output_dir=args.output_dir,
            args=args,
            label="after",
        )
        _record_phase_video(
            current_checkpoint,
            phase_index=phase_index,
            args=args,
            label="after",
        )
        cross_play_records = _run_cross_play(
            current_checkpoint,
            phase=phase,
            phase_index=phase_index,
            pool=pool,
            rng=rng,
            args=args,
        )
        cross_play_aggregate = _cross_play_aggregate(cross_play_records)
        if phase in AGENTS:
            cross_play_by_agent[phase] = cross_play_aggregate
        elif phase == "both":
            cross_play_by_agent = {agent: None for agent in AGENTS}
        if cross_play_records:
            _write_record(
                args.output_dir,
                {
                    "phase_index": phase_index,
                    "completed_iterations": completed_iterations,
                    "cross_play": cross_play_records,
                    "cross_play_aggregate": cross_play_aggregate,
                },
            )
            if args.auto_pool_path is not None:
                _save_opponent_pool(args.auto_pool_path, pool)

        promotion = _maybe_promote_to_pool(
            pool,
            agent=phase,
            checkpoint=current_checkpoint,
            summary=summary,
            cross_play_aggregate=cross_play_aggregate,
            phase_index=phase_index,
            args=args,
        )
        recent_pool = _maybe_add_recent_to_pool(
            pool,
            agent=phase,
            checkpoint=current_checkpoint,
            summary=summary,
            cross_play_aggregate=cross_play_aggregate,
            promotion=promotion,
            phase_index=phase_index,
            args=args,
        )
        if phase in AGENTS:
            next_phase_override = _promotion_failure_override(
                phase=phase,
                summary=summary,
                promotion=promotion,
                cross_play_aggregate=cross_play_aggregate,
                retry_state=promotion_retry_state,
                args=args,
            )
        elif phase == "both":
            promotion_retry_state.clear()
            next_phase_override = None
        if promotion is not None:
            _write_record(
                args.output_dir,
                {
                    "phase_index": phase_index,
                    "completed_iterations": completed_iterations,
                    "auto_pool_promotion": promotion,
                    "recent_pool": recent_pool,
                    "next_phase_override": next_phase_override,
                    "cross_play_aggregate": cross_play_aggregate,
                },
            )
            if args.auto_pool_path is not None:
                _save_opponent_pool(args.auto_pool_path, pool)
        elif recent_pool is not None or next_phase_override is not None:
            _write_record(
                args.output_dir,
                {
                    "phase_index": phase_index,
                    "completed_iterations": completed_iterations,
                    "recent_pool": recent_pool,
                    "next_phase_override": next_phase_override,
                    "cross_play_aggregate": cross_play_aggregate,
                },
            )
            if args.auto_pool_path is not None:
                _save_opponent_pool(args.auto_pool_path, pool)

    final_path = args.output_dir / "final_checkpoint.txt"
    final_path.write_text(str(current_checkpoint), encoding="utf-8")
    print(f"\n[INFO] Final checkpoint: {current_checkpoint}")
    print(f"[INFO] Scheduler logs: {args.output_dir}")


if __name__ == "__main__":
    main()
