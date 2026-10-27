"""Host driver of the Stage 8b from-scratch recipe: one seed through flight, the LR pilot (seed 1 only), the game,
the consolidation and the final gate (docs/stage8b_from_scratch_protocol.md).

    setsid nohup python3 scripts/recipes/stage8b/run_seed.py --seed 81 --pilot --out <experiment>/seed81 \
        > <experiment>/seed81.log 2>&1 &

- On its first start it copies itself, recipe.py, block.sh and tb_scalars.py to <out>/recipe/ and runs that copy, so
  later edits in the repo cannot change a running seed. A restart runs the same copy.
- Every finished step (training block, evaluation point, analysis) is recorded in <out>/state.json. A restart
  continues after the last finished step; training is bit-reproducible, so a rerun block reproduces itself.
- It stops on a training status other than 0, a missing checkpoint, a config mismatch in the run's params, or a
  missing evaluation JSON (Isaac scripts exit with 0 even after an exception), and on the pre-registered stops.
- The last line is "run_seed done ...", which the completion waiter watches.

Standard library only (host Python); the Isaac work runs in the container through `docker exec`.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from recipe import (CONSOLIDATION_KL, CONSOLIDATION_LR, FLIGHT_KL, FLIGHT_LR, GAME_KL, PILOT_LRS, Budget,
                    block_seed, final_gate, flight_gate, frozen_role, interim_check, next_mode, pilot_choice, point_summary,
                    pool, summary, verdict)

CONTAINER = "isaac-lab-base-uav51"
ISAACLAB = "/workspace/isaaclab/isaaclab.sh"
REPO_HOST = "/home/marc/RL/UAVPredatorPrey"
MOUNTS = (  # host -> container; the more specific mount first
    ("/home/marc/RL/artifacts/isaac/logs", "/workspace/UAVPredatorPrey/logs"),
    ("/home/marc/RL/artifacts", "/workspace/artifacts"),
    ("/home/marc/RL/UAVPredatorPrey", "/workspace/UAVPredatorPrey"),
)
LOGS_HOST = "/home/marc/RL/artifacts/isaac/logs/skrl/uav_3v1_direct"
TASK = "3v1-survival-soft-oob-teammate-vel-random-spawn-low-progress-obstacles10-v0"
AGENT = "skrl_mappo_attention_critic_prey_attention_large_gru_obstacles10_cfg_entry_point"
RELATION_FLAGS = ("+agent.models.policy.predator_obstacle_relation=true", "+agent.models.value.obstacle_relation=true",
                  "+agent.models.policy.prey_obstacle_relation=true")
# Loading a recipe checkpoint with its optimizer state needs the optimizer's single parameter group (multiplier 1;
# the preset says 30). Composed checkpoints carry no optimizer state, so the flag is harmless there.
LOAD_FLAGS = ("agent.agent.obstacle_lr_multiplier=1.0",)
MODEL_FLAGS = ("+agent.models.policy.prey_mask_inactive_predators=true",
               "+agent.models.policy.predator_mask_inactive_teammates=true") + RELATION_FLAGS + LOAD_FLAGS
V5 = "/workspace/UAVPredatorPrey/.pretrained_checkpoints/anchor_v5_2026-10-21/current/agent_14400.pt"
V3 = "/workspace/artifacts/isaac/experiments/2026-10-12_mixed_wall_pillars10/checkpoints/start_pair_obstacles10.pt"
V5_TRAJECTORIES = "/workspace/artifacts/isaac/experiments/2026-10-20_consolidation_selfplay/evals/point5_own.npz"
ZERO_TOOL = "/workspace/artifacts/isaac/experiments/2026-10-08_obstacles_first_training/zero_obstacle_branch.py"
PROBE_DIR = "/home/marc/RL/artifacts/isaac/diagnostics/2026-10-16_stage8_blind_spots"
FLIGHT_ENV = ("env.predator_forced_crash_bonus=0.0", "env.predator_distance_progress_reward_scale=0.0",
              "env.predator_catch_bonus=0.0", "env.prey_evasion_reward_scale=0.0", "env.prey_caught_penalty=0.0",
              "env.prey_distance_progress_reward_scale=0.0", "env.catch_distance=0.0")
GHOST_ENV = ("env.ghost_predators=True", "env.ghost_max_speed=3.5")
RECIPE_FILES = ("run_seed.py", "run_league.py", "recipe.py", "block.sh", "tb_scalars.py")


class Stop(Exception):
    """A failure or a pre-registered stop: the seed ends here and the reason is reported."""


def to_container(path: str) -> str:
    path = os.path.abspath(path)
    for host, container in MOUNTS:
        if path == host or path.startswith(host + "/"):
            return container + path[len(host):]
    raise ValueError(f"{path} is not visible in the container")


def to_host(path: str) -> str:
    for host, container in MOUNTS:
        if path == container or path.startswith(container + "/"):
            return host + path[len(container):]
    raise ValueError(f"{path} is not a container path")


def stamp() -> str:
    return time.strftime("%m-%d %H:%M:%S")


def lr_tag(lr: float) -> str:
    """1e-4 -> '1e4': run names stay plain words for Hydra."""
    return f"{lr:.0e}".replace("-0", "").replace("-", "")


# ------------------------------------------------------------------------------------------------- config check

def yaml_scalars(path: str, top_level_only: bool) -> dict[str, str]:
    """First occurrence of every `key: value` line of a dumped config (no YAML package on the host)."""
    pattern = re.compile(r"^([A-Za-z_]\w*):\s*(\S.*)$" if top_level_only else r"^\s*([A-Za-z_]\w*):\s*(\S.*)$")
    values: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            match = pattern.match(line.rstrip("\n"))
            if match and match.group(1) not in values:
                values[match.group(1)] = match.group(2).strip()
    return values


def expected_config(mode: str, lr: float, kl: float) -> tuple[dict, dict]:
    agent = {"obstacle_lr_multiplier": 1.0, "discount_factor": 0.995, "learning_rate": lr, "kl_threshold": kl,
             "predator_min_log_std": -1.0, "prey_mask_inactive_predators": True,
             "predator_mask_inactive_teammates": True, "predator_obstacle_relation": True,
             "prey_obstacle_relation": True, "obstacle_relation": True}
    flight = mode == "flight"
    env = {"prey_oob_penalty": -300.0, "predator_oob_penalty": -400.0, "observation_frame": "mixed",
           "predator_upright_zero_at_level": False, "num_active_obstacles": -1,
           "predator_forced_crash_bonus": 0.0 if flight else 200.0, "catch_distance": 0.0 if flight else 0.3,
           "ghost_env_fraction": 0.3333 if mode == "joint" else 0.0}
    if flight:
        env.update({"predator_distance_progress_reward_scale": 0.0, "predator_catch_bonus": 0.0,
                    "prey_evasion_reward_scale": 0.0, "prey_caught_penalty": 0.0,
                    "prey_distance_progress_reward_scale": 0.0})
    if mode == "joint":
        env["ghost_max_speed"] = 3.5
    return agent, env


def config_mismatches(values: dict[str, str], expected: dict) -> list[str]:
    problems = []
    for key, want in expected.items():
        got = values.get(key)
        if got is None:
            problems.append(f"{key} missing")
            continue
        if isinstance(want, bool):
            ok = got.lower() == str(want).lower()
        elif isinstance(want, (int, float)):
            try:
                ok = math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-12)
            except ValueError:
                ok = False
        else:
            ok = got.strip("'\"") == want
        if not ok:
            problems.append(f"{key} = {got} (expected {want})")
    return problems


def pilot_stats(block1: dict, block2: dict) -> dict:
    """From the two pilot blocks' TensorBoard series: finite losses in both, and the larger role's mean KL
    early-stop epoch fraction in block 2."""
    finite = all(math.isfinite(v) for series in (block1, block2) for tag, points in series.items()
                 if tag.startswith("Loss /") for _, v in points)
    fractions = [sum(v for _, v in points) / len(points) for tag, points in block2.items()
                 if tag.startswith("Diagnostics / KL early-stop epoch fraction") and points]
    return {"finite": finite, "kl_stop": max(fractions) if fractions else math.nan}


# ------------------------------------------------------------------------------------------------- the pipeline

class Pipeline:
    """The recipe's order of steps, independent of how a step runs (Docker in production, a fake in the tests)."""

    def __init__(self, backend, out: str, seed: int, budget: Budget, lr: float | None = None, pilot: bool = False,
                 videos: bool = False):
        self.backend, self.out, self.seed, self.budget = backend, out, seed, budget
        self.lr, self.pilot, self.videos = lr, pilot, videos
        os.makedirs(out, exist_ok=True)
        self.state_path = os.path.join(out, "state.json")
        self.state = {"seed": seed, "steps": {}, "points": {}}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
        self.state["budget"] = dataclasses.asdict(budget)
        self.fprey: str | None = None

    # -- bookkeeping
    def save(self) -> None:
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.state_path)

    def log(self, message: str) -> None:
        print(f"{stamp()} {message}", flush=True)

    def step(self, key: str, run):
        """Run `run` once per seed: its JSON result is recorded, and a restart returns the recorded result."""
        if key in self.state["steps"]:
            return self.state["steps"][key]
        result = run()
        self.state["steps"][key] = result
        self.save()
        return result

    def refresh_summary(self) -> None:
        script = os.path.join(os.path.dirname(self.out), "summarize.py")
        if os.path.exists(script):
            subprocess.run([sys.executable, script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # -- the two kinds of steps
    def train(self, name: str, mode: str, checkpoint: str | None, updates: int, number: int, lr: float, kl: float,
              pool_entries: dict | None = None, pool_prob: float | None = None, pool_max: int | None = None) -> str:
        def run():
            pool_path = None
            if pool_entries is not None:
                os.makedirs(os.path.join(self.out, "pools"), exist_ok=True)
                pool_path = os.path.join(self.out, "pools", f"{name}.json")
                with open(pool_path, "w") as f:
                    json.dump(pool_entries, f, indent=1)
            self.log(f"block {name}: {mode}, {updates} updates, LR {lr:g}, seed {block_seed(self.seed, number)}"
                     + (f", pool of {len(next(iter(pool_entries.values())))}" if pool_entries else ""))
            started = time.time()
            info = self.backend.train(f"{self.seed}_{name}", mode, checkpoint, updates, block_seed(self.seed, number),
                                      lr, kl, pool_path, **({"pool_prob": pool_prob} if pool_prob is not None else {}),
                                      **({"pool_max": pool_max} if pool_max is not None else {}))
            info.update(mode=mode, updates=updates, lr=lr, minutes=round((time.time() - started) / 60.0, 1))
            self.log(f"block {name} done in {info['minutes']} min")
            return info
        return self.step(f"train {name}", run)["checkpoint"]

    def point(self, label: str, checkpoint: str, kind: str) -> dict:
        """An evaluation point: light (every game block), pilot (the light set at full size) or full."""
        def run():
            episodes = self.budget.light_episodes if kind == "light" else self.budget.full_episodes
            ev = self.backend.evaluate
            started = time.time()
            evals = {
                "own": ev(f"{label}_own", checkpoint, checkpoint, episodes, trajectory=kind == "full"),
                "pred_vs_fprey": ev(f"{label}_pred_vs_fprey", checkpoint, self.fprey, episodes),
                "ghosts": ev(f"{label}_ghosts", checkpoint, checkpoint, episodes, overrides=GHOST_ENV),
                "ours_vs_v5": ev(f"{label}_ours_vs_v5", checkpoint, V5, episodes),
                "v5_vs_ours": ev(f"{label}_v5_vs_ours", V5, checkpoint, episodes),
            }
            if kind == "full":
                evals["ours_vs_v3"] = ev(f"{label}_ours_vs_v3", checkpoint, V3, episodes)
                evals["v3_vs_ours"] = ev(f"{label}_v3_vs_ours", V3, checkpoint, episodes)
                evals["own_predator_zeroed"] = ev(f"{label}_own_predator_zeroed", checkpoint, checkpoint, episodes,
                                                  zero="predator")
                evals["own_prey_zeroed"] = ev(f"{label}_own_prey_zeroed", checkpoint, checkpoint, episodes, zero="prey")
            p = point_summary(evals)
            p.update(kind=kind, episodes=episodes, checkpoint=checkpoint,
                     minutes=round((time.time() - started) / 60.0, 1))
            return p
        p = self.step(f"point {label}", run)
        if label not in self.state["points"]:
            self.state["points"][label] = p
            self.save()
            self.log(f"point {label} ({kind}): own catch {p['own catch']:.3f}, interception {p['interception']:.3f}, "
                     f"ghost hazard {p['ghost hazard']:.3f}/s, duel v5 {p['duel v5']:+.1f}, "
                     f"drones lost {p['drones lost']:.3f}, prey lost {p['prey lost']:.3f}")
            self.refresh_summary()
        return p

    def video(self, label: str, checkpoint: str) -> None:
        if self.videos:
            self.step(f"video {label}", lambda: self.backend.video(label, checkpoint))

    # -- stages
    def flight(self) -> str:
        b = self.budget
        checkpoint = None
        for k in range(1, b.flight_max_blocks + 1):
            checkpoint = self.train(f"f{k}", "flight", checkpoint, b.flight_block, k, FLIGHT_LR, FLIGHT_KL)
            ckpt = checkpoint
            result = self.step(f"flight gate f{k}", lambda: summary(
                self.backend.evaluate(f"f{k}_flight", ckpt, ckpt, b.full_episodes, overrides=FLIGHT_ENV)))
            checks = flight_gate(result)
            self.log(f"flight gate f{k}: drones lost {result['drones_lost']:.3f}, prey lost {result['prey_lost']:.3f}, "
                     f"length {result['length']:.1f} -> {'pass' if all(checks.values()) else 'not yet'}")
            if all(checks.values()) or (b.smoke and k == b.flight_max_blocks):
                self.state["flight"] = {"blocks": k, "updates": k * b.flight_block, "checkpoint": checkpoint,
                                        "gate": checks, "metrics": result}
                self.save()
                return checkpoint
        raise Stop(f"flight gate not passed after {b.flight_max_blocks * b.flight_block} updates; "
                   "the fallback F' (July flight on the obstacle-free model, then expansion) needs the user's approval")

    def lr_pilot(self, start: str) -> tuple[float, list[str]]:
        b = self.budget
        arms, checkpoints = {}, {}
        for lr in PILOT_LRS:
            name = f"pilot{lr_tag(lr)}"
            c1 = self.train(f"{name}_b1", "joint", start, b.game_block, 101, lr, GAME_KL)
            c2 = self.train(f"{name}_b2", "joint", c1, b.game_block, 102, lr, GAME_KL)
            p = self.point(name, c2, "pilot")
            runs = [self.state["steps"][f"train {name}_b{i}"]["run"] for i in (1, 2)]
            stats = self.step(f"pilot stats {name}", lambda: pilot_stats(*(self.backend.tb(run) for run in runs)))
            arms[lr] = {"finite": stats["finite"], "kl_stop": stats["kl_stop"], "drones_lost": p["drones lost"],
                        "interception": p["interception"], "ghost_survival": p["ghost survival"]}
            checkpoints[lr] = [c1, c2]
        chosen, details = pilot_choice(arms)
        self.state["pilot"] = {"chosen": chosen, "arms": {f"{lr:g}": {**arms[lr], **details[lr]} for lr in arms}}
        self.save()
        for lr in arms:
            self.log(f"pilot LR {lr:g}: {'stable' if details[lr]['stable'] else 'unstable'}, score "
                     f"{details[lr]['score']:.3f} (KL stops {arms[lr]['kl_stop']:.3f}, drones lost "
                     f"{arms[lr]['drones_lost']:.3f})")
        if chosen is None:
            raise Stop("no stable arm in the LR pilot")
        self.log(f"pilot: LR {chosen:g} chosen")
        return chosen, checkpoints[chosen]

    def game(self, start: str, lr: float, pilot_checkpoints: list[str] | None) -> tuple[str, list[tuple[str, str]]]:
        b = self.budget
        g0 = self.state["points"]["g0"]
        points = [("g0", start)]
        checkpoint, last_catch = start, g0["own catch"]
        for n in range(1, b.game_blocks + 1):
            if pilot_checkpoints and n <= 2:
                checkpoint = pilot_checkpoints[n - 1]
                if n == 2:   # the chosen arm's pilot point is G's point 2
                    p = self.state["points"].setdefault("g2", self.state["points"][f"pilot{lr_tag(lr)}"])
                    self.save()
                else:
                    p = self.point("g1", checkpoint, "light")
            else:
                mode = next_mode(n, last_catch, b)
                if b.smoke:          # exercise both single-role modes and their pools
                    mode = {3: "predator", 4: "prey"}.get(n, "joint")
                role = frozen_role(mode)
                entries = pool(role, points, checkpoint) if role else None
                checkpoint = self.train(f"g{n}", mode, checkpoint, b.game_block, 100 + n, lr, GAME_KL, entries)
                p = self.point(f"g{n}", checkpoint, "full" if n in b.full_game_points else "light")
            points.append((f"g{n}", checkpoint))
            last_catch = p["own catch"]
            if n == b.interim_block:
                checks = interim_check(p, g0)
                self.state["interim"] = checks
                self.save()
                self.log(f"interim rule at block {n}: {checks}")
                if not all(checks.values()) and not b.smoke:
                    raise Stop(f"interim rule at block {n} failed: diagnose before changing anything")
            if n in b.full_game_points:
                self.video(f"g{n}", checkpoint)
        return checkpoint, points

    def consolidation(self, start: str, game_points: list[tuple[str, str]]) -> str:
        b = self.budget
        keep = {f"g{n}" for n in b.full_game_points}
        predators = [(name, ckpt) for name, ckpt in game_points if name in keep]
        self.state["points"].setdefault("c0", self.state["points"][f"g{b.game_blocks}"])
        checkpoint = start
        for c in range(1, b.cycles + 1):
            joint = self.train(f"c{c}_joint", "joint", checkpoint, b.cycle_joint, 200 + c, CONSOLIDATION_LR,
                               CONSOLIDATION_KL)
            entries = pool("predator", predators, joint, window=len(predators))
            checkpoint = self.train(f"c{c}_prey", "prey", joint, b.cycle_prey, 300 + c, CONSOLIDATION_LR,
                                    CONSOLIDATION_KL, entries)
            self.point(f"c{c}", checkpoint, "full" if c in b.full_cycle_points else "light")
            predators.append((f"c{c}", checkpoint))
        return checkpoint

    def final(self, checkpoint: str) -> dict:
        k4, k5 = self.budget.full_cycle_points
        labels = {f"c{k4}": self.backend.npz(f"c{k4}_own"), f"c{k5}": self.backend.npz(f"c{k5}_own"),
                  "anchor_v5": V5_TRAJECTORIES}
        analyses = self.step("analyses", lambda: self.backend.analyses(labels))
        p4, p5 = self.state["points"][f"c{k4}"], self.state["points"][f"c{k5}"]
        # Without long episodes the analyses report no value (NaN): criterion 5 then fails, as it should.
        anatomy5 = analyses["anatomy"].get(f"c{k5}", {})
        nearest = float(anatomy5.get("long_nearest_predator", math.nan))
        blocked = float(analyses["blocked"].get(f"c{k5}", math.nan))
        checks = final_gate(p4, p5, {"long_nearest_predator": nearest}, blocked)
        duel = (p4["duel v5"] + p5["duel v5"]) / 2.0
        result = {"seed": self.seed, "checks": checks, "duel v5 (mean of 4, 5)": duel, "verdict": verdict(checks, duel),
                  "points": [f"c{k4}", f"c{k5}"], "checkpoint": checkpoint,
                  "long_nearest_predator": nearest, "long_blocked": blocked}
        with open(os.path.join(self.out, "gate.json"), "w") as f:
            json.dump(result, f, indent=1)
        self.state["gate"] = result
        self.save()
        self.log(f"final gate: {result['verdict']} (duel v5 {duel:+.1f})")
        for name, ok in checks.items():
            self.log(f"   {name}: {ok}")
        self.refresh_summary()
        self.step("probe", lambda: self.backend.probe(f"s8b_{self.seed}_c{k5}", checkpoint))
        self.step(f"video c{k5}", lambda: self.backend.video(f"c{k5}", checkpoint))
        return result

    def run(self) -> dict:
        start = self.flight()
        self.fprey = start
        self.video("f_end", start)
        self.point("g0", start, "full")
        if self.pilot:
            lr, pilot_checkpoints = self.lr_pilot(start)
        else:
            lr, pilot_checkpoints = self.lr, None
        self.state["lr"] = lr
        self.save()
        checkpoint, points = self.game(start, lr, pilot_checkpoints)
        checkpoint = self.consolidation(checkpoint, points)
        return self.final(checkpoint)


# ------------------------------------------------------------------------------------------------- Docker backend

class Docker:
    """Runs the steps in the Isaac container."""

    def __init__(self, out: str, recipe_dir: str, analysis_dir: str, smoke: bool):
        self.out, self.recipe_dir, self.analysis_dir, self.smoke = out, recipe_dir, analysis_dir, smoke
        for sub in ("blocks", "evals", "videos"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)

    def _exec(self, args: list[str], log: str | None = None, env: tuple[str, ...] = ()) -> int:
        command = ["docker", "exec", "-w", "/workspace/UAVPredatorPrey"]
        for item in env:
            command += ["-e", item]
        command += [CONTAINER, *args]
        with open(log or os.devnull, "w") as f:
            return subprocess.run(command, stdout=f, stderr=subprocess.STDOUT).returncode

    def _tool(self, output_host: str, *args: str) -> None:
        """A container tool whose output file must exist afterwards."""
        if os.path.exists(output_host):
            os.remove(output_host)
        self._exec([ISAACLAB, "-p", *args])
        if not os.path.exists(output_host):
            raise Stop(f"tool failed (no {output_host}): {' '.join(args)}")

    def train(self, tag: str, mode: str, checkpoint: str | None, updates: int, seed: int, lr: float, kl: float,
              pool_path: str | None, extra: tuple[str, ...] = (), expect_env: dict | None = None,
              pool_prob: float | None = None, pool_max: int | None = None) -> dict:
        """One block; `extra` adds env overrides for an ablation, `expect_env` the values its config check expects;
        `pool_prob` the share of envs facing pool opponents (block.sh default 0.5), `pool_max` the most pool policies
        train.py loads (block.sh default 8)."""
        args = ["bash", to_container(os.path.join(self.recipe_dir, "block.sh")), to_container(self.out), tag, mode,
                checkpoint or "none", str(updates), str(seed), repr(lr), repr(kl)]
        if pool_path:
            args.append(to_container(pool_path))
        env = (f"BLOCK_EXTRA={' '.join(extra)}",) if extra else ()
        if pool_prob is not None:
            env += (f"BLOCK_POOL_PROB={pool_prob}",)
        if pool_max is not None:
            env += (f"BLOCK_POOL_MAX={pool_max}",)
        self._exec(args, env=env)
        status_path = os.path.join(self.out, "blocks", f"{tag}.status.txt")
        status = open(status_path).read() if os.path.exists(status_path) else ""
        if "status=0" not in status:
            raise Stop(f"block {tag} failed: {status.strip() or 'no status file'}")
        runs = sorted(d for d in glob.glob(os.path.join(LOGS_HOST, "*")) if d.endswith(f"_s8b_{tag}"))
        if not runs:
            raise Stop(f"block {tag}: run directory not found")
        run = runs[-1]
        checkpoint_host = os.path.join(run, "checkpoints", f"agent_{24 * updates}.pt")
        if not os.path.exists(checkpoint_host):
            raise Stop(f"block {tag}: {checkpoint_host} missing")
        agent_expected, env_expected = expected_config(mode, lr, kl)
        env_expected.update(expect_env or {})
        problems = config_mismatches(yaml_scalars(os.path.join(run, "params", "agent.yaml"), False), agent_expected)
        problems += config_mismatches(yaml_scalars(os.path.join(run, "params", "env.yaml"), True), env_expected)
        if yaml_scalars(os.path.join(run, "params", "agent.yaml"), False).get("mixed_precision", "false") != "false":
            problems.append("mixed_precision is on")
        if problems:
            raise Stop(f"block {tag}: config mismatch: {'; '.join(problems)}")
        return {"run": run, "checkpoint": to_container(checkpoint_host)}

    def pool_info(self, tag: str) -> tuple[float, int] | None:
        """(pool share, number of pool policies) as train.py reports them in the block's console log."""
        path = os.path.join(self.out, "blocks", f"{tag}.console.log")
        if not os.path.exists(path):
            return None
        with open(path, errors="replace") as f:
            match = re.search(r"Per-env opponent pool active for frozen '\w+': .*pool_prob=([\d.]+), policies=(\d+)",
                              f.read())
        return (float(match.group(1)), int(match.group(2))) if match else None

    def npz(self, name: str) -> str:
        return to_container(os.path.join(self.out, "evals", f"{name}.npz"))

    def container(self, path: str) -> str:
        return to_container(path)

    def compose(self, name: str, base: str, predator: str, prey: str) -> str:
        """A checkpoint with each role's block (policy, preprocessors and optimizer state) from the given sources."""
        os.makedirs(os.path.join(self.out, "starts"), exist_ok=True)
        output = os.path.join(self.out, "starts", f"{name}.pt")
        self._tool(output, "scripts/compose_checkpoint.py", "--base", base, "--predator", predator, "--prey", prey,
                   "--output", to_container(output), "--keep-optimizers")
        return to_container(output)

    def evaluate(self, name: str, predator: str, prey: str, episodes: int, overrides: tuple[str, ...] = (),
                 trajectory: bool = False, zero: str | None = None) -> dict:
        evals = os.path.join(self.out, "evals")
        temporary = []
        checkpoint = predator
        if predator != prey or zero:
            sources = {"predator": predator, "prey": prey}
            if zero:
                zeroed = os.path.join(evals, f"zeroed_{name}.pt")
                self._tool(zeroed, ZERO_TOOL, sources[zero], to_container(zeroed))
                sources[zero] = to_container(zeroed)
                temporary.append(zeroed)
            composed = os.path.join(evals, f"tmp_{name}.pt")
            self._tool(composed, "scripts/compose_checkpoint.py", "--base", predator, "--predator", sources["predator"],
                       "--prey", sources["prey"], "--output", to_container(composed))
            temporary.append(composed)
            checkpoint = to_container(composed)
        json_host = os.path.join(evals, f"{name}.json")
        args = ["env", "-u", "DISPLAY", "-u", "XAUTHORITY", ISAACLAB, "-p", "scripts/skrl/evaluate.py", "--headless",
                "--task", TASK, "--agent", AGENT, "--algorithm", "MAPPO", "--num_envs", str(episodes),
                "--episodes", str(episodes), "--seed", "42", "--checkpoint", checkpoint,
                "--json", to_container(json_host)]
        if trajectory:
            args += ["--trajectory-npz", self.npz(name)]
        args += [*MODEL_FLAGS, *overrides]
        try:
            for attempt in (1, 2):   # Isaac scripts exit with 0 even after an exception, so the JSON decides
                if os.path.exists(json_host):
                    os.remove(json_host)
                self._exec(args, log=os.path.join(evals, f"{name}.log"))
                if os.path.exists(json_host):
                    with open(json_host) as f:
                        return json.load(f)["episode_metrics"]
            raise Stop(f"evaluation {name} failed twice (no JSON; see evals/{name}.log)")
        finally:
            for path in temporary:
                if os.path.exists(path):
                    os.remove(path)

    def tb(self, run: str) -> dict:
        from tb_scalars import read_scalars
        files = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
        series: dict = {}
        for path in files:
            for tag, points in read_scalars(path).items():
                series.setdefault(tag, []).extend(points)
        return series

    def analyses(self, labels: dict[str, str]) -> dict:
        items = [f"{label}={path}" for label, path in labels.items()]
        tool = lambda name: to_container(os.path.join(self.analysis_dir, name))
        anatomy = os.path.join(self.out, "anatomy.json")
        self._tool(anatomy, tool("anatomy_json.py"), to_container(anatomy), *items)
        with open(anatomy) as f:
            anatomy_data = json.load(f)
        long_game = os.path.join(self.out, "long_game.txt")
        self._exec([ISAACLAB, "-p", tool("long_game.py"), *items], log=long_game)
        blocked = {}
        for line in open(long_game):
            match = re.search(r"^(\S+) long \(>= 5 s\): episodes \d+, line to the prey blocked ([\d.]+|nan)", line)
            if match:
                blocked[match.group(1)] = float(match.group(2))
        for name, json_name in (("shield_geometry", "shield"), ("roles", "roles"), ("flanking_value", "flanking"),
                                ("forced_distance", None)):
            extra = [f"--json={to_container(os.path.join(self.out, json_name + '.json'))}"] if json_name else []
            self._exec([ISAACLAB, "-p", tool(f"{name}.py"), *extra, *items],
                       log=os.path.join(self.out, f"{name}.txt"))
        return {"anatomy": anatomy_data, "blocked": blocked}

    def probe(self, label: str, checkpoint: str) -> dict | None:
        log = os.path.join(self.out, f"probe_{label}.log")
        with open(log, "w") as f:
            subprocess.run(["bash", "run_probe.sh", label, checkpoint, *RELATION_FLAGS, *LOAD_FLAGS], cwd=PROBE_DIR,
                           env={**os.environ, "PROBE_GAMMA": "0.995"}, stdout=f, stderr=subprocess.STDOUT)
        source = os.path.join(PROBE_DIR, "runs", f"{label}_analysis.json")
        if not os.path.exists(source):
            return None
        shutil.copy(source, os.path.join(self.out, f"probe_{label}.json"))
        with open(source) as f:
            return json.load(f)[label].get("probes")

    def video(self, label: str, checkpoint: str) -> str | None:
        directory = os.path.join(self.out, "videos", label)
        self._exec(["env", "-u", "DISPLAY", "-u", "XAUTHORITY", ISAACLAB, "-p", "scripts/skrl/play.py", "--headless",
                    "--video", "--video_length", "100" if self.smoke else "1500", "--video-dir", to_container(directory),
                    "--video-width", "1920", "--video-height", "1080", "--video-bitrate", "10M",
                    "--video-antialiasing", "FXAA", "--video-marker-radius", "0.1", "--video-marker-height", "0.16",
                    "--task", TASK, "--agent", AGENT, "--algorithm", "MAPPO", "--num_envs", "1", "--seed", "42",
                    "--checkpoint", checkpoint, "--camera-eye", "10", "-10", "12.5", "--camera-target", "0", "0", "5",
                    "--camera-env-index", "0", "--no-video-markers", *MODEL_FLAGS],
                   log=os.path.join(self.out, "videos", f"{label}.log"))
        found = glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True)
        return found[0] if found else None


def gpu_monitor(path: str, stop: threading.Event) -> None:
    with open(path, "a") as f:
        while not stop.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            f.write(f"{stamp()} {result.stdout.strip()}\n")
            f.flush()
            stop.wait(60)


# ------------------------------------------------------------------------------------------------- entry point

def freeze_and_exec(args: argparse.Namespace, script: str = "run_seed.py") -> None:
    """Run `script` from the frozen copy in <out>/recipe/; make the copy on the first start."""
    here = os.path.dirname(os.path.abspath(__file__))
    recipe_dir = os.path.join(args.out, "recipe")
    if not os.path.exists(os.path.join(recipe_dir, script)):
        dirty = subprocess.run(["git", "-C", REPO_HOST, "status", "--porcelain", "--", "source", "scripts/skrl",
                                "scripts/recipes"], capture_output=True, text=True).stdout.strip()
        if dirty and not args.allow_dirty:
            sys.exit(f"uncommitted changes in the code the recipe runs:\n{dirty}")
        os.makedirs(recipe_dir, exist_ok=True)
        for name in RECIPE_FILES:
            shutil.copy(os.path.join(here, name), recipe_dir)
        commit = subprocess.run(["git", "-C", REPO_HOST, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        with open(os.path.join(args.out, "code_commit.txt"), "w") as f:
            f.write(commit.strip() + (" (dirty)" if dirty else "") + "\n")
    os.execv(sys.executable, [sys.executable, os.path.join(recipe_dir, script), *sys.argv[1:], "--frozen"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="per-seed output directory, inside the experiment directory")
    parser.add_argument("--pilot", action="store_true", help="run the LR pilot (seed 1) and use its choice")
    parser.add_argument("--lr", type=float, help="the game stage's LR (seeds 2 and 3: the pilot's choice)")
    parser.add_argument("--analysis", help="analysis scripts (default: <experiment>/analysis)")
    parser.add_argument("--smoke", action="store_true", help="tiny budgets: the full-size smoke run")
    parser.add_argument("--allow-dirty", action="store_true", help="allow uncommitted code (smoke runs only)")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)
    if not args.pilot and args.lr is None:
        parser.error("--lr is required without --pilot")
    os.makedirs(args.out, exist_ok=True)
    if not args.frozen:
        freeze_and_exec(args)

    print(f"{stamp()} run_seed start: seed {args.seed}, out {args.out}, pilot {args.pilot}, lr {args.lr}, "
          f"smoke {args.smoke}", flush=True)
    stop = threading.Event()
    try:
        busy = subprocess.run(["docker", "exec", CONTAINER, "pgrep", "-f", r"scripts/skrl/(train|evaluate|play)\.py"],
                              capture_output=True, text=True)
        if busy.returncode == 0:
            raise Stop(f"another Isaac job is running in the container (pids {busy.stdout.split()})")
        threading.Thread(target=gpu_monitor, args=(os.path.join(args.out, "gpu_memory.log"), stop), daemon=True).start()
        backend = Docker(args.out, os.path.dirname(os.path.abspath(__file__)),
                         args.analysis or os.path.join(os.path.dirname(args.out), "analysis"), args.smoke)
        budget = Budget.smoke_budget() if args.smoke else Budget()
        pipeline = Pipeline(backend, args.out, args.seed, budget, lr=args.lr, pilot=args.pilot,
                            videos=args.pilot or args.smoke)
        result = pipeline.run()
        print(f"{stamp()} seed {args.seed}: {result['verdict']}", flush=True)
    except Stop as stop_reason:
        with open(os.path.join(args.out, "stopped.txt"), "w") as f:
            f.write(f"{stamp()} {stop_reason}\n")
        print(f"{stamp()} STOPPED: {stop_reason}", flush=True)
    finally:
        stop.set()
        print(f"run_seed done {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
