"""Decisions of the Stage 8b from-scratch recipe as pure functions (docs/stage8b_from_scratch_protocol.md).

The host driver (run_seed.py) and the tests import this module. Every threshold is pre-registered in the
protocol; change one only through an amendment there. Standard library only (host Python).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

POLICY_HZ = 50.0                          # decimation 2 at 100 Hz physics


@dataclass(frozen=True)
class Budget:
    """Updates per block and the schedule; `smoke()` shrinks everything for the full-size smoke run."""

    flight_block: int = 1500
    flight_max_blocks: int = 4
    game_block: int = 1200
    game_blocks: int = 20
    game_joint_start: int = 2             # blocks 1-2 are joint; the band guard decides from block 3
    interim_block: int = 10               # the interim rule and a full point
    cycles: int = 5
    cycle_joint: int = 3000
    cycle_prey: int = 600
    light_episodes: int = 512
    full_episodes: int = 1024
    smoke: bool = False

    @staticmethod
    def smoke_budget() -> "Budget":
        return Budget(flight_block=30, flight_max_blocks=1, game_block=30, game_blocks=4, interim_block=2,
                      cycles=2, cycle_joint=30, cycle_prey=30, light_episodes=64, full_episodes=64, smoke=True)

    @property
    def full_game_points(self) -> tuple[int, int]:
        return (self.interim_block, self.game_blocks)

    @property
    def full_cycle_points(self) -> tuple[int, int]:
        return (self.cycles - 1, self.cycles)


PILOT_LRS = (1e-4, 3e-5, 1e-5)
FLIGHT_LR, FLIGHT_KL = 1e-4, 0.0          # the July flight's LR (read from its checkpoints); KL guard off as in the YAML
GAME_KL = 0.05
CONSOLIDATION_LR, CONSOLIDATION_KL = 1e-5, 0.05
POOL_WINDOW = 4
BAND_LOW, BAND_HIGH = 0.25, 0.75


def block_seed(seed: int, number: int) -> int:
    """Flight blocks 1-4, game blocks 101-120, consolidation joint 201-205 and prey 301-305."""
    return seed * 1000 + number


def hazard(probability: float, episode_steps: float) -> float:
    """Constant-hazard rate per second: -ln(1 - P) / L, with L the mean episode length in seconds."""
    if not (math.isfinite(probability) and math.isfinite(episode_steps)) or episode_steps <= 0:
        return math.nan
    return -math.log(1.0 - min(probability, 0.999)) / (episode_steps / POLICY_HZ) + 0.0   # + 0.0: no "-0.0"


def outcome_return(success: float, drones_lost: float) -> float:
    """The predator team's outcome: a catch or forced crash earns 200, each lost drone costs 400 / 3."""
    return 200.0 * success - 400.0 / 3.0 * drones_lost


def summary(metrics: dict) -> dict:
    """The quantities the recipe uses from one evaluate.py JSON (its `episode_metrics`)."""
    get = lambda key: float(metrics.get(f"Metrics/{key}", math.nan))
    drones_lost = sum(get(f"predator_{i}_oob_rate") for i in range(3))
    return {
        "catch": get("catch_rate"),
        "success": get("predator_success_rate"),
        "drones_lost": drones_lost,
        "prey_lost": get("prey_oob_rate"),                    # arena exit or pillar crash
        "predator_crash": get("predator_obstacle_crash_rate"),
        "prey_crash": get("prey_obstacle_crash_rate"),
        "length": get("episode_length"),                      # policy steps
        "return": outcome_return(get("predator_success_rate"), drones_lost),
    }


def flight_gate(flight: dict) -> dict[str, bool]:
    """Promotion out of the flight stage, on the flight-settings evaluation (catches off)."""
    return {
        "drones lost per episode <= 0.10": flight["drones_lost"] <= 0.10,
        "prey lost per episode <= 0.05": flight["prey_lost"] <= 0.05,
        "mean episode length >= 490 steps": flight["length"] >= 490.0,
    }


def next_mode(block: int, last_catch: float, budget: Budget) -> str:
    """The band guard: which role(s) train in game block `block`, from the own catch after the previous block."""
    if block <= budget.game_joint_start:
        return "joint"
    if last_catch < BAND_LOW:
        return "predator"
    if last_catch > BAND_HIGH:
        return "prey"
    return "joint"


def frozen_role(mode: str) -> str | None:
    return {"predator": "prey", "prey": "predator"}.get(mode)


def pool(role: str, checkpoints: list[tuple[str, str]], current: str, window: int = POOL_WINDOW) -> dict:
    """A per-env pool file of the frozen role: the last `window` (name, checkpoint) entries other than the current
    checkpoint, equal weights. train.py loads `data[role]`."""
    past = [(name, ckpt) for name, ckpt in checkpoints if ckpt != current]
    return {role: [{"name": name, "checkpoint": ckpt, "weight": 1.0} for name, ckpt in past[-window:]]}


def ablation_ratio(zeroed: dict, intact: dict, crash_key: str) -> float:
    """Crash hazard with all obstacle outputs zeroed over the intact hazard (> 1: the role uses its perception)."""
    h_zeroed, h_intact = hazard(zeroed[crash_key], zeroed["length"]), hazard(intact[crash_key], intact["length"])
    if not (math.isfinite(h_zeroed) and math.isfinite(h_intact)):
        return math.nan
    if h_intact == 0.0:
        return math.inf if h_zeroed > 0.0 else 1.0
    return h_zeroed / h_intact


def point_summary(evals: dict[str, dict]) -> dict:
    """One evaluation point from its evaluate.py metrics, keyed as run_seed.py names them."""
    s = {name: summary(m) for name, m in evals.items()}
    own = s["own"]
    point = {
        "own catch": own["catch"], "own success": own["success"], "own return": own["return"],
        "drones lost": own["drones_lost"], "prey lost": own["prey_lost"], "episode length": own["length"],
    }
    if "pred_vs_fprey" in s:
        point["interception"] = s["pred_vs_fprey"]["success"]
    if "ghosts" in s:
        # Against ghosts the prey is lost when caught or when it crashes (pillar or arena): a prey that crashes must
        # not look like one that escapes. Ghosts never crash, so the two outcomes are exclusive per episode.
        g = s["ghosts"]
        lost = min(g["catch"] + g["prey_lost"], 1.0)
        point["ghost catch"] = g["catch"]
        point["ghost survival"] = 1.0 - lost
        point["ghost hazard"] = hazard(lost, g["length"])
    for ref in ("v5", "v3"):
        if f"ours_vs_{ref}" in s and f"{ref}_vs_ours" in s:
            point[f"R ours vs {ref} prey"] = s[f"ours_vs_{ref}"]["return"]
            point[f"R {ref} vs our prey"] = s[f"{ref}_vs_ours"]["return"]
            point[f"duel {ref}"] = point[f"R ours vs {ref} prey"] - point[f"R {ref} vs our prey"]
    if "own_predator_zeroed" in s:
        point["predator ratio"] = ablation_ratio(s["own_predator_zeroed"], own, "predator_crash")
    if "own_prey_zeroed" in s:
        point["prey ratio"] = ablation_ratio(s["own_prey_zeroed"], own, "prey_crash")
    return point


def pilot_choice(arms: dict[float, dict]) -> tuple[float | None, dict]:
    """The LR pilot's rule. Each arm: finite (bool), kl_stop (the larger role's mean KL early-stop epoch fraction in
    block 2), drones_lost, interception, ghost_survival.

    Stable: finite losses, kl_stop <= 0.5, drones lost <= 1.5x the lowest-LR arm's + 0.05. Score = interception +
    ghost survival. Choose the highest stable LR whose score is at least the next lower stable LR's - 0.03."""
    base = arms[min(arms)]["drones_lost"]
    details = {}
    for lr, a in arms.items():
        stable = bool(a["finite"]) and a["kl_stop"] <= 0.5 and a["drones_lost"] <= 1.5 * base + 0.05
        details[lr] = {"stable": stable, "score": a["interception"] + a["ghost_survival"]}
    stable = sorted((lr for lr in arms if details[lr]["stable"]), reverse=True)
    for i, lr in enumerate(stable):
        lower = stable[i + 1] if i + 1 < len(stable) else None
        if lower is None or details[lr]["score"] >= details[lower]["score"] - 0.03:
            return lr, details
    return None, details


def interim_check(g10: dict, g0: dict) -> dict[str, bool]:
    """The pre-registered stop at block 10: both roles must show basic learning."""
    return {
        "predator intercepts the F prey in >= 0.5 of episodes": g10["interception"] >= 0.5,
        "prey's loss hazard against ghosts <= 0.7x its value at the start of G":
            g10["ghost hazard"] <= 0.7 * g0["ghost hazard"],
    }


def final_gate(p4: dict, p5: dict, anatomy5: dict, blocked5: float) -> dict[str, bool]:
    """The per-seed final gate on consolidation points 4 and 5 (or the last two points of a shorter schedule)."""
    mean = lambda key: (p4[key] + p5[key]) / 2.0
    return {
        "1a prey perception: own ablation ratio >= 1.1 (mean of 4, 5)": mean("prey ratio") >= 1.1,
        "1b predator perception: own ablation ratio >= 1.1 (mean of 4, 5)": mean("predator ratio") >= 1.1,
        "2 balance: own catch within 0.3-0.7 at 4 and 5": all(0.3 <= p["own catch"] <= 0.7 for p in (p4, p5)),
        "3 duel with anchor v5: D >= -10 (mean of 4, 5)": mean("duel v5") >= -10.0,
        "4 stability: own catch changes by <= 0.05 from 4 to 5": abs(p5["own catch"] - p4["own catch"]) <= 0.05,
        "5 no waiting game: long episodes, nearest predator <= 3.0 m and line blocked <= 0.30":
            anatomy5["long_nearest_predator"] <= 3.0 and blocked5 <= 0.30,
        "6 safety: prey lost <= 0.33 and drones lost <= 0.37 per episode at 5":
            p5["prey lost"] <= 0.33 and p5["drones lost"] <= 0.37,
    }


def verdict(checks: dict[str, bool], duel_mean: float) -> str:
    failed = [name.split(":")[0] for name, ok in checks.items() if not ok]
    if failed:
        return "fail: " + ", ".join(failed)
    return "pass, stronger than anchor v5" if duel_mean > 10.0 else "pass"


# ------------------------------------------------------------------------------------------------- league phases
# Fictitious-play style continuation (2026-10-24, after seed 81): instead of latest-vs-latest joint training, the roles
# alternate, and the training role faces a uniform sample of the opponent's whole history ("the average strategy")
# plus its latest version. A phase lasts until the role beats that average, so neither role is slowed.

ROLES = ("predator", "prey")
LEAGUE_BAR = 0.5


@dataclass(frozen=True)
class LeagueBudget:
    block: int = 1200
    max_blocks_per_phase: int = 6
    max_updates: int = 48000
    pool_size: int = 8
    pool_prob: float = 0.75            # the rest of the envs face the latest opponent
    gate_episodes: int = 512
    smoke: bool = False

    @staticmethod
    def smoke_budget() -> "LeagueBudget":
        return LeagueBudget(block=30, max_blocks_per_phase=2, max_updates=150, pool_size=2, gate_episodes=64, smoke=True)


def other(role: str) -> str:
    return "prey" if role == "predator" else "predator"


def role_versions(rows: list[tuple[str, str, set[str]]]) -> dict[str, list[tuple[str, str]]]:
    """Distinct versions of each role from a run's points, in order: (label, checkpoint, roles trained in the block
    that produced the point). A role that was frozen keeps its version, so its later checkpoints are copies and must
    not count as new opponents (the flaw of seed 81's last-4-points pools)."""
    return {role: [(label, ckpt) for label, ckpt, trained in rows if role in trained] for role in ROLES}


def sample_opponents(archive: list[tuple[str, str]], latest: str, k: int, seed: int) -> list[tuple[str, str]]:
    """Uniform sample without replacement of k past versions other than the latest one (which the rest of the envs
    face anyway), kept in archive order."""
    import random
    candidates = [entry for entry in archive if entry[0] != latest]
    chosen = set(random.Random(seed).sample(range(len(candidates)), min(k, len(candidates))))
    return [entry for i, entry in enumerate(candidates) if i in chosen]


def league_gate(role: str, rates: list[float]) -> bool:
    """A phase ends once the training role beats the average opponent: the predators catch at least half of the
    sampled past prey; the prey is caught by at most half of the sampled past predators."""
    mean = sum(rates) / len(rates)
    return mean >= LEAGUE_BAR if role == "predator" else mean <= LEAGUE_BAR


def league_converged(phases: list[dict]) -> bool:
    """Equilibrium: in each of the last two cycles, both phases passed their gate after a single block."""
    last = phases[-4:]
    return len(last) == 4 and all(p["passed"] and p["blocks"] == 1 for p in last)


def league_decision(average_prey_catch: float, cycle_points: list[dict], final: dict) -> dict[str, bool]:
    """The pre-registered decision of the league continuation (2026-10-24 PLAN.md)."""
    last2 = cycle_points[-2:]
    duel = sum(p["duel v5"] for p in last2) / len(last2) if last2 else math.nan
    return {
        "1 the predators beat the average prey: final catch against every archived prey, mean >= 0.4":
            average_prey_catch >= 0.4,
        "2 balance: own catch within 0.3-0.7 at the last two cycle ends":
            len(last2) == 2 and all(0.3 <= p["own catch"] <= 0.7 for p in last2),
        "3 duel with anchor v5 >= -10 (mean of the last two cycle ends)": duel >= -10.0,
        "4 guards at the final point: drones lost <= 0.37 and prey lost <= 0.33":
            final["drones lost"] <= 0.37 and final["prey lost"] <= 0.33,
    }
