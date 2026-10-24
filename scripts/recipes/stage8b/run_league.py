"""League continuation of a finished Stage 8b seed: fictitious-play style phases against the opponents' whole history
(2026-10-24, after seed 81; the experiment's PLAN.md has the decision rule).

From the seed's final pair, the roles alternate. In each phase:
- the training role faces 8 opponents sampled uniformly from the other role's distinct past versions in 75% of the
  envs, and the latest opponent in the rest;
- it trains in blocks of 1200 updates until it beats the average of those 8 (predators: catch >= 0.5; prey: caught
  <= 0.5), for at most 6 blocks;
- the role's new version then joins its archive.

The run stops at equilibrium (both phases of two consecutive cycles pass after one block) or at 48,000 updates. Point 0
adds a cross-play matrix of the seed's versions; the end measures the predators against every archived prey. Like
run_seed.py it runs a frozen copy from <out>/recipe/ and resumes from <out>/state.json; the last line is
"run_league done".
    setsid nohup python3 scripts/recipes/stage8b/run_league.py --source <seed dir> --out <experiment>/league \
        > <experiment>/league.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import run_seed
from recipe import (GAME_KL, ROLES, Budget, LeagueBudget, block_seed, league_converged, league_decision, league_gate,
                    other, role_versions, sample_opponents, summary)

MATRIX_PREDATORS = ("g7", "g13", "g17", "g20", "c5")
MATRIX_PREY = ("g0", "g2", "g8", "g14", "g18", "g20", "c5")
SMOKE_MATRIX = (("g20", "c5"), ("g8", "c5"))


def source_rows(state: dict) -> list[tuple[str, str, set[str]]]:
    """(label, checkpoint, roles that trained in the block producing it) for a run_seed.py seed's points: flight and
    consolidation points changed both roles; game points changed what their block's mode trained (the pilot's
    blocks 1-2 were joint)."""
    rows = []
    order = ["g0"] + [f"g{n}" for n in range(1, 200)] + [f"c{k}" for k in range(1, 50)]
    for label in order:
        point = state["points"].get(label)
        if not point:
            continue
        if label == "g0" or label.startswith("c"):
            trained = set(ROLES)
        else:
            mode = state["steps"].get(f"train {label}", {}).get("mode", "joint")
            trained = set(ROLES) if mode == "joint" else {mode}
        rows.append((label, point["checkpoint"], trained))
    return rows


def last_trained(state: dict) -> dict[str, str]:
    """Each role's checkpoint from its last training block: the only checkpoints that hold its optimizer state,
    because train.py does not save a frozen role's optimizer."""
    last = {}
    for key, step in state["steps"].items():
        if key.startswith("train ") and isinstance(step, dict) and "checkpoint" in step:
            mode = step.get("mode", "joint")
            for role in (ROLES if mode in ("joint", "flight") else (mode,)):
                last[role] = step["checkpoint"]
    return last


class League(run_seed.Pipeline):
    def __init__(self, backend, out: str, source_dir: str, budget: Budget, league: LeagueBudget):
        with open(os.path.join(source_dir, "state.json")) as f:
            source = json.load(f)
        super().__init__(backend, out, source["seed"], budget)
        self.source, self.source_dir, self.league = source, source_dir, league
        self.lr, self.kl = source["lr"], GAME_KL
        self.fprey = source["flight"]["checkpoint"]
        self.updates = 0
        self.trained = last_trained(source)

    def rate(self, name: str, role: str, current: str, opponent: str, episodes: int) -> float:
        """Catch rate of the current role against one opponent checkpoint (for the prey: how often it is caught)."""
        predator, prey = (current, opponent) if role == "predator" else (opponent, current)
        return summary(self.backend.evaluate(name, predator, prey, episodes))["catch"]

    def matrix(self, checkpoints: dict[str, str]) -> dict:
        """Point 0: catch of seed predators (rows) against seed prey (columns), plus the final predator's row over
        every distinct prey version (the average-prey baseline)."""
        predators, prey = SMOKE_MATRIX if self.league.smoke else (MATRIX_PREDATORS, MATRIX_PREY)
        archive_prey = [label for label, _ in role_versions(source_rows(self.source))["prey"]]
        cells = [(p, y) for p in predators for y in prey]
        cells += [(predators[-1], y) for y in archive_prey if y not in prey]
        table = {}
        for p, y in cells:
            table[f"{p} vs {y}"] = self.step(f"matrix {p} vs {y}", lambda: self.rate(
                f"M_{p}_vs_{y}", "predator", checkpoints[p], checkpoints[y], self.budget.light_episodes))
        self.state["matrix"] = {"predators": list(predators), "prey": list(prey), "archive_prey": archive_prey,
                                "catch": table}
        self.save()
        self.refresh_summary()
        return table

    def phase(self, cycle: int, role: str, current: str, archives: dict, latest: dict, number: int):
        b = self.league
        opponent = other(role)
        pool_list = sample_opponents(archives[opponent], latest[opponent], b.pool_size, block_seed(self.seed, number + 1))
        entries = {opponent: [{"name": label, "checkpoint": ckpt, "weight": 1.0} for label, ckpt in pool_list]}
        tag = f"L{cycle}{role[:4]}"
        record = {"cycle": cycle, "role": role, "pool": [label for label, _ in pool_list], "blocks": 0, "passed": False,
                  "gates": [], "updates_at_start": self.updates}
        for k in range(1, b.max_blocks_per_phase + 1):
            if self.updates + b.block > b.max_updates:
                break
            if k == 1 and self.trained[role] != current:
                # The training role was frozen in the last phase, so `current` lacks its optimizer state: take its
                # block from its last training checkpoint (same policy and preprocessors, plus the Adam moments).
                predator, prey = (self.trained[role], current) if role == "predator" else (current, self.trained[role])
                current = self.step(f"start {tag}", lambda: self.backend.compose(f"start_{tag}", current, predator, prey))
            number += 1
            current = self.train(f"{tag}{k}", role, current, b.block, number, self.lr, self.kl, entries,
                                 pool_prob=b.pool_prob)
            self.trained[role] = current
            self.updates += b.block
            check = getattr(self.backend, "pool_info", None)
            if check:
                info = check(f"{self.seed}_{tag}{k}")
                if info != (b.pool_prob, len(pool_list)):
                    raise run_seed.Stop(f"block {tag}{k}: pool active as {info}, expected {(b.pool_prob, len(pool_list))}")
            ckpt = current
            rates = self.step(f"gate {tag}{k}", lambda: [
                self.rate(f"gate_{tag}{k}_vs_{label}", role, ckpt, opponent_ckpt, b.gate_episodes)
                for label, opponent_ckpt in pool_list])
            record["blocks"] = k
            record["gates"].append(sum(rates) / len(rates))
            self.log(f"gate {tag}{k}: mean {'catch' if role == 'predator' else 'caught'} against the sampled history "
                     f"{record['gates'][-1]:.3f} ({', '.join(f'{v:.2f}' for v in rates)})")
            if league_gate(role, rates):
                record["passed"] = True
                break
        if record["blocks"]:
            archives[role].append((tag, current))
            latest[role] = tag
        return current, record, number

    def run(self) -> dict:
        rows = source_rows(self.source)
        archives = role_versions(rows)
        checkpoints = {label: ckpt for label, ckpt, _ in rows}
        start_label, start = rows[-1][0], rows[-1][1]
        latest = {role: archives[role][-1][0] for role in ROLES}
        self.log(f"league from {start_label}: {len(archives['predator'])} predator and {len(archives['prey'])} prey "
                 f"versions; latest {latest}")
        self.matrix(checkpoints)
        self.point("L0", start, "pilot")
        current, cycle, number, phases, cycle_points, stop = start, 1, 500, [], [], None
        while stop is None:
            for role in ROLES:
                current, record, number = self.phase(cycle, role, current, archives, latest, number)
                if record["blocks"]:
                    phases.append(record)
                if self.updates + self.league.block > self.league.max_updates:
                    stop = "budget"
                    break
            if phases and phases[-1]["cycle"] == cycle:
                cycle_points.append(self.point(f"L{cycle}", current, "pilot"))
            self.state["league"] = {"phases": phases, "archives": {r: [l for l, _ in a] for r, a in archives.items()},
                                    "updates": self.updates}
            self.save()
            self.refresh_summary()
            if stop is None and league_converged(phases):
                stop = "equilibrium"
            cycle += 1
        self.log(f"league stops: {stop} after {self.updates} updates, {len(phases)} phases")
        return self.final(current, archives, cycle_points, stop)

    def final(self, current: str, archives: dict, cycle_points: list[dict], stop: str) -> dict:
        final = self.point("final", current, "full")
        ep = self.budget.light_episodes
        prey_rates = self.step("final against every prey", lambda: {
            label: self.rate(f"F_vs_prey_{label}", "predator", current, ckpt, ep) for label, ckpt in archives["prey"]})
        panel = [(l, c) for l, c in archives["predator"] if l in MATRIX_PREDATORS or l.startswith("L")]
        predator_rates = self.step("final prey against the predator panel", lambda: {
            label: self.rate(f"F_vs_pred_{label}", "prey", current, ckpt, ep) for label, ckpt in panel})
        average_prey = sum(prey_rates.values()) / len(prey_rates)
        s8b = os.path.dirname(self.source_dir)
        labels = {"final": self.backend.npz("final_own"),
                  "s81_c5": self.backend.container(os.path.join(self.source_dir, "evals", "c5_own.npz")),
                  "anchor_v5": run_seed.V5_TRAJECTORIES}
        self.step("analyses", lambda: self.backend.analyses(labels))
        checks = league_decision(average_prey, cycle_points, final)
        failed = [name.split(" ", 1)[0] for name, ok in checks.items() if not ok]
        verdict = ("history training balances the arms race: amend the recipe with league phases" if not failed
                   else "fail: criteria " + ", ".join(failed))
        result = {"stop": stop, "updates": self.updates, "checks": checks, "verdict": verdict,
                  "average prey catch": average_prey, "prey rates": prey_rates,
                  "average predator caught": sum(predator_rates.values()) / max(len(predator_rates), 1),
                  "predator rates": predator_rates, "checkpoint": current, "source": s8b}
        with open(os.path.join(self.out, "decision_league.json"), "w") as f:
            json.dump(result, f, indent=1)
        self.state["result"] = result
        self.save()
        self.log(f"decision: {verdict} (average prey catch {average_prey:.3f})")
        for name, ok in checks.items():
            self.log(f"   {name}: {ok}")
        self.refresh_summary()
        self.step("video final", lambda: self.backend.video("final", current))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="the finished seed's directory (with state.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--analysis", help="analysis scripts (default: <source>/../analysis)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out, args.source = os.path.abspath(args.out), os.path.abspath(args.source)
    os.makedirs(args.out, exist_ok=True)
    if not args.frozen:
        run_seed.freeze_and_exec(args, script="run_league.py")
    print(f"{run_seed.stamp()} run_league start: source {args.source}, out {args.out}, smoke {args.smoke}", flush=True)
    try:
        busy = subprocess.run(["docker", "exec", run_seed.CONTAINER, "pgrep", "-f",
                               r"scripts/skrl/(train|evaluate|play)\.py"], capture_output=True, text=True)
        if busy.returncode == 0:
            raise run_seed.Stop(f"another Isaac job is running in the container (pids {busy.stdout.split()})")
        backend = run_seed.Docker(args.out, os.path.dirname(os.path.abspath(__file__)),
                                  args.analysis or os.path.join(os.path.dirname(args.source), "analysis"), args.smoke)
        league = League(backend, args.out, args.source, Budget.smoke_budget() if args.smoke else Budget(),
                        LeagueBudget.smoke_budget() if args.smoke else LeagueBudget())
        result = league.run()
        print(f"{run_seed.stamp()} {result['verdict']}", flush=True)
    except run_seed.Stop as reason:
        with open(os.path.join(args.out, "stopped.txt"), "w") as f:
            f.write(f"{run_seed.stamp()} {reason}\n")
        print(f"{run_seed.stamp()} STOPPED: {reason}", flush=True)
    finally:
        print(f"run_league done {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
