"""The Stage 8b from-scratch recipe (docs/stage8b_from_scratch_protocol.md): its pre-registered decisions, and the
driver's order of steps with a fake backend, including the pools, the stops and resuming after an interruption."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "recipes" / "stage8b"))
import recipe  # noqa: E402
import run_seed  # noqa: E402


def _metrics(catch=0.5, success=None, lost=0.03, prey_lost=0.1, length=300.0, predator_crash=0.05, prey_crash=0.1):
    m = {"Metrics/catch_rate": catch, "Metrics/predator_success_rate": catch if success is None else success,
         "Metrics/prey_oob_rate": prey_lost, "Metrics/predator_obstacle_crash_rate": predator_crash,
         "Metrics/prey_obstacle_crash_rate": prey_crash, "Metrics/episode_length": length}
    m.update({f"Metrics/predator_{i}_oob_rate": lost for i in range(3)})
    return m


class FakeBackend:
    """Synthetic results: the flight gate passes at block 2, the band guard sees catch 0.1 after g2 and 0.9 after g3,
    the 1e-4 pilot arm scores best, and everything else is a healthy pair."""

    def __init__(self, crash_after: int | None = None, ghost_catch: float = 0.3, flight_passes: bool = True):
        self.trains: list[dict] = []
        self.crash_after, self.ghost_catch, self.flight_passes = crash_after, ghost_catch, flight_passes

    def train(self, tag, mode, checkpoint, updates, seed, lr, kl, pool_path):
        if self.crash_after is not None and len(self.trains) >= self.crash_after:
            raise RuntimeError("power off")
        pool = json.loads(Path(pool_path).read_text()) if pool_path else None
        self.trains.append(dict(tag=tag, mode=mode, checkpoint=checkpoint, updates=updates, seed=seed, lr=lr, kl=kl,
                                pool=pool))
        return {"run": f"/runs/{tag}", "checkpoint": f"/ckpt/{tag}.pt"}

    def evaluate(self, name, predator, prey, episodes, overrides=(), trajectory=False, zero=None):
        label = name.split("_")[0]
        if name.endswith("_flight"):
            return _metrics(catch=0.0, lost=0.0 if (self.flight_passes and label != "f1") else 0.1, prey_lost=0.0,
                            length=499.0)
        if name.endswith("_ghosts"):
            return _metrics(catch=0.8 if label == "g0" else self.ghost_catch)
        if name.endswith("_pred_vs_fprey"):
            return _metrics(catch={"pilot1e4": 0.8, "pilot3e5": 0.7, "pilot1e5": 0.6}.get(label, 0.8))
        if name.endswith("_ours_vs_v5"):
            return _metrics(catch=0.6)
        if name.endswith("_v5_vs_ours"):
            return _metrics(catch=0.57)
        if zero == "predator":
            return _metrics(predator_crash=0.2)
        if zero == "prey":
            return _metrics(prey_crash=0.3)
        return _metrics(catch={"pilot1e4": 0.1, "g3": 0.9}.get(label, 0.5))

    def tb(self, run):
        return {"Loss / Policy loss (predator)": [(1, 0.1)], "Diagnostics / KL early-stop epoch fraction (prey)": [(1, 0.1)]}

    def npz(self, name):
        return f"/npz/{name}.npz"

    def container(self, path):
        return path

    def analyses(self, labels):
        return {"anatomy": {label: {"long_nearest_predator": 2.0} for label in labels},
                "blocked": {label: 0.1 for label in labels}}

    def probe(self, label, checkpoint):
        return None

    def video(self, label, checkpoint):
        return None


def _pipeline(tmp_path, backend, pilot=True, lr=None):
    return run_seed.Pipeline(backend, str(tmp_path), 81, recipe.Budget(), lr=lr, pilot=pilot)


def _by_tag(backend):
    return {t["tag"].split("_", 1)[1]: t for t in backend.trains}


def test_formulas_and_seeds() -> None:
    assert recipe.block_seed(81, 1) == 81001 and recipe.block_seed(81, 120) == 81120
    assert recipe.outcome_return(0.75, 0.15) == pytest.approx(130.0)
    assert recipe.hazard(0.5, 500) == pytest.approx(math.log(2) / 10.0)
    assert run_seed.lr_tag(1e-4) == "1e4" and run_seed.lr_tag(3e-5) == "3e5"


def test_band_guard() -> None:
    b = recipe.Budget()
    assert recipe.next_mode(2, 0.0, b) == "joint"           # blocks 1-2 are joint whatever the catch
    assert recipe.next_mode(3, 0.24, b) == "predator"
    assert recipe.next_mode(3, 0.76, b) == "prey"
    assert recipe.next_mode(3, 0.5, b) == "joint"


def test_pool_keeps_the_last_window_without_the_current() -> None:
    points = [(f"g{i}", f"/c/g{i}.pt") for i in range(7)]
    entries = recipe.pool("prey", points, "/c/g6.pt")["prey"]
    assert [e["name"] for e in entries] == ["g2", "g3", "g4", "g5"] and all(e["weight"] == 1.0 for e in entries)


def test_pilot_choice() -> None:
    arm = lambda score, kl=0.1, lost=0.1, finite=True: {"finite": finite, "kl_stop": kl, "drones_lost": lost,
                                                        "interception": score, "ghost_survival": 0.0}
    assert recipe.pilot_choice({1e-4: arm(0.80), 3e-5: arm(0.82), 1e-5: arm(0.5)})[0] == 1e-4   # within 0.03
    assert recipe.pilot_choice({1e-4: arm(0.70), 3e-5: arm(0.80), 1e-5: arm(0.5)})[0] == 3e-5   # worse by 0.10
    assert recipe.pilot_choice({1e-4: arm(0.9, kl=0.6), 3e-5: arm(0.8), 1e-5: arm(0.5)})[0] == 3e-5   # KL stops
    assert recipe.pilot_choice({1e-4: arm(0.9, lost=0.5), 3e-5: arm(0.8), 1e-5: arm(0.5)})[0] == 3e-5  # losses
    assert recipe.pilot_choice({1e-4: arm(0.9, finite=False), 3e-5: arm(0.8, kl=0.9),
                                1e-5: arm(0.5, kl=0.9)})[0] is None


def test_gates() -> None:
    assert all(recipe.flight_gate({"drones_lost": 0.1, "prey_lost": 0.05, "length": 490.0}).values())
    assert not recipe.flight_gate({"drones_lost": 0.11, "prey_lost": 0.0, "length": 499.0})["drones lost per episode <= 0.10"]
    p = {"prey ratio": 1.2, "predator ratio": 1.2, "own catch": 0.5, "duel v5": 12.0, "prey lost": 0.2, "drones lost": 0.2}
    checks = recipe.final_gate(p, dict(p, **{"own catch": 0.53}), {"long_nearest_predator": 2.5}, 0.2)
    assert all(checks.values()) and recipe.verdict(checks, 12.0) == "pass, stronger than anchor v5"
    checks = recipe.final_gate(p, dict(p, **{"own catch": 0.6}), {"long_nearest_predator": 3.5}, 0.2)
    assert recipe.verdict(checks, 12.0) == "fail: 4 stability, 5 no waiting game"


def test_a_crashing_prey_does_not_survive_ghosts() -> None:
    evals = {"own": _metrics(), "ghosts": _metrics(catch=0.0, prey_lost=1.0, length=60.0)}
    point = recipe.point_summary(evals)
    assert point["ghost survival"] == 0.0 and point["ghost hazard"] > 5.0
    point = recipe.point_summary({"own": _metrics(), "ghosts": _metrics(catch=0.2, prey_lost=0.1, length=500.0)})
    assert point["ghost survival"] == pytest.approx(0.7) and point["ghost hazard"] == pytest.approx(-math.log(0.7) / 10)


def test_config_check(tmp_path) -> None:
    agent = tmp_path / "agent.yaml"
    agent.write_text("models:\n  policy:\n    predator_min_log_std: -1.0\n    predator_obstacle_relation: true\n"
                     "  value:\n    obstacle_relation: true\nagent:\n  discount_factor: 0.995\n  learning_rate: 1.0e-05\n"
                     "  kl_threshold: 0.05\n  obstacle_lr_multiplier: 30.0\n")
    expected = {"discount_factor": 0.995, "learning_rate": 1e-5, "kl_threshold": 0.05, "obstacle_lr_multiplier": 1.0,
                "predator_obstacle_relation": True, "obstacle_relation": True, "prey_obstacle_relation": True}
    problems = run_seed.config_mismatches(run_seed.yaml_scalars(str(agent), False), expected)
    assert problems == ["obstacle_lr_multiplier = 30.0 (expected 1.0)", "prey_obstacle_relation missing"]


def test_pilot_stats() -> None:
    kl = "Diagnostics / KL early-stop epoch fraction"
    stats = run_seed.pilot_stats({"Loss / Value loss (prey)": [(1, 1.0)]},
                                 {f"{kl} (predator)": [(1, 0.2), (2, 0.4)], f"{kl} (prey)": [(1, 0.1)],
                                  "Loss / Policy loss (predator)": [(1, float("nan"))]})
    assert stats == {"finite": False, "kl_stop": pytest.approx(0.3)}


def test_pipeline_runs_the_recipe(tmp_path) -> None:
    backend = FakeBackend()
    result = _pipeline(tmp_path, backend).run()
    t = _by_tag(backend)
    # Flight promoted at the second block (the gate fails at f1), from random init.
    assert t["f1"]["checkpoint"] is None and t["f2"]["checkpoint"] == "/ckpt/81_f1.pt" and "f3" not in t
    assert t["f1"]["seed"] == 81001 and t["f1"]["lr"] == 1e-4 and t["f1"]["kl"] == 0.0
    # Pilot: three paired arms of two joint blocks; 1e-4 scores best and becomes G's blocks 1-2.
    for lr in ("1e4", "3e5", "1e5"):
        assert t[f"pilot{lr}_b1"]["seed"] == 81101 and t[f"pilot{lr}_b2"]["seed"] == 81102
        assert t[f"pilot{lr}_b1"]["checkpoint"] == "/ckpt/81_f2.pt"
    assert "g1" not in t and "g2" not in t
    # Band guard: catch 0.1 after g2 -> predator block with a prey pool; 0.9 after g3 -> prey block.
    assert t["g3"]["mode"] == "predator" and t["g3"]["checkpoint"] == "/ckpt/81_pilot1e4_b2.pt"
    assert [e["name"] for e in t["g3"]["pool"]["prey"]] == ["g0", "g1"]
    assert t["g4"]["mode"] == "prey" and list(t["g4"]["pool"]) == ["predator"]
    assert t["g5"]["mode"] == "joint" and t["g5"]["pool"] is None
    assert all(t[f"g{n}"]["lr"] == 1e-4 and t[f"g{n}"]["kl"] == 0.05 for n in range(3, 21))
    # Consolidation: the anchor v5 recipe with own past predators in the prey phases.
    assert t["c1_joint"]["checkpoint"] == "/ckpt/81_g20.pt" and t["c1_joint"]["lr"] == 1e-5
    assert t["c1_joint"]["seed"] == 81201 and t["c1_prey"]["seed"] == 81301 and t["c1_prey"]["updates"] == 600
    assert [e["name"] for e in t["c1_prey"]["pool"]["predator"]] == ["g10", "g20"]
    assert [e["name"] for e in t["c3_prey"]["pool"]["predator"]] == ["g10", "g20", "c1", "c2"]
    assert len(backend.trains) == 2 + 6 + 18 + 10
    assert result["verdict"] == "pass" and json.loads((tmp_path / "gate.json").read_text())["verdict"] == "pass"


def test_resume_after_an_interruption(tmp_path) -> None:
    first = FakeBackend(crash_after=12)
    with pytest.raises(RuntimeError):
        _pipeline(tmp_path, first).run()
    second = FakeBackend()
    result = _pipeline(tmp_path, second).run()
    assert not {t["tag"] for t in first.trains} & {t["tag"] for t in second.trains}
    assert len(first.trains) + len(second.trains) == 36 and result["verdict"] == "pass"


def test_pre_registered_stops(tmp_path) -> None:
    flightless = FakeBackend(flight_passes=False)
    with pytest.raises(run_seed.Stop, match="flight gate"):
        _pipeline(tmp_path / "a", flightless).run()
    assert len(flightless.trains) == 4
    slow_prey = FakeBackend(ghost_catch=0.78)              # the prey's loss hazard against ghosts barely improves
    with pytest.raises(run_seed.Stop, match="interim rule"):
        _pipeline(tmp_path / "b", slow_prey, pilot=False, lr=3e-5).run()
    assert max(int(t["tag"].split("_g")[1]) for t in slow_prey.trains if "_g" in t["tag"]) == 10


# ------------------------------------------------------------------------------------------------- league phases
import run_league  # noqa: E402


def _seed_dir(tmp_path) -> Path:
    """A finished seed shaped like seed 81: game modes as run, so 25 predator and 12 prey versions."""
    modes = {n: "predator" for n in range(3, 21)}
    modes.update({8: "joint", 14: "prey", 18: "joint", 20: "joint"})
    labels = ["g0"] + [f"g{n}" for n in range(1, 21)] + [f"c{k}" for k in range(1, 6)]
    steps = {f"train g{n}": {"mode": m, "checkpoint": f"/ckpt/81_g{n}.pt"} for n, m in modes.items()}
    steps.update({"train c5_joint": {"mode": "joint", "checkpoint": "/ckpt/81_c5_joint.pt"},
                  "train c5_prey": {"mode": "prey", "checkpoint": "/ckpt/81_c5.pt"}})
    state = {"seed": 81, "lr": 1e-4, "flight": {"checkpoint": "/ckpt/81_f3.pt"},
             "points": {l: {"checkpoint": f"/ckpt/81_{l}.pt"} for l in labels}, "steps": steps}
    directory = tmp_path / "seed81"
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps(state))
    return directory


class FakeLeagueBackend(FakeBackend):
    """Predators beat the sampled prey from their second block in cycle 1 and from the first block afterwards; the
    prey always passes after one block, so the league converges after cycle 3."""

    def train(self, tag, mode, checkpoint, updates, seed, lr, kl, pool_path, pool_prob=None):
        info = super().train(tag, mode, checkpoint, updates, seed, lr, kl, pool_path)
        self.trains[-1]["pool_prob"] = pool_prob
        return info

    def compose(self, name, base, predator, prey):
        self.composed = getattr(self, "composed", []) + [(name, predator, prey)]
        return f"/ckpt/{name}.pt"

    def evaluate(self, name, predator, prey, episodes, overrides=(), trajectory=False, zero=None):
        if name.startswith("gate_L1pred1"):
            return _metrics(catch=0.3)
        if name.startswith("gate_") and "pred" in name.split("_")[1]:
            return _metrics(catch=0.6)
        if name.startswith("gate_"):
            return _metrics(catch=0.4)
        if name.startswith(("M_", "F_")):
            return _metrics(catch=0.5)
        return super().evaluate(name, predator, prey, episodes, overrides, trajectory, zero)


def _league(tmp_path, backend, max_updates=48000):
    return run_league.League(backend, str(tmp_path / "league"), str(tmp_path / "seed81"), recipe.Budget(),
                             recipe.LeagueBudget(max_updates=max_updates))


def test_pipeline_passes_the_pool_limit_only_when_given(tmp_path) -> None:
    """A whole-history pool needs more than train.py's default of 8 policies; backends without the option still work."""
    calls = []

    class Recorder(FakeBackend):
        def train(self, tag, mode, checkpoint, updates, seed, lr, kl, pool_path, **options):
            calls.append(options)
            return super().train(tag, mode, checkpoint, updates, seed, lr, kl, pool_path)

    pipeline = run_seed.Pipeline(Recorder(), str(tmp_path / "out"), 81, recipe.Budget())
    pool = {"prey": [{"name": "a", "checkpoint": "/a.pt", "weight": 1.0}]}
    pipeline.train("w1", "predator", "/start.pt", 10, 621, 1e-4, 0.05, pool, pool_prob=0.9, pool_max=25)
    pipeline.train("w2", "predator", "/start.pt", 10, 622, 1e-4, 0.05, pool, pool_prob=0.9)
    assert calls == [{"pool_prob": 0.9, "pool_max": 25}, {"pool_prob": 0.9}]


def test_versions_skip_frozen_copies() -> None:
    archives = recipe.role_versions([("g7", "a", {"predator"}), ("g8", "b", {"predator", "prey"}), ("g9", "c", {"predator"}),
                                     ("g14", "d", {"prey"})])
    assert [l for l, _ in archives["predator"]] == ["g7", "g8", "g9"] and [l for l, _ in archives["prey"]] == ["g8", "g14"]


def test_league_gates_and_equilibrium() -> None:
    assert recipe.league_gate("predator", [0.4, 0.6]) and not recipe.league_gate("predator", [0.3, 0.6])
    assert recipe.league_gate("prey", [0.5, 0.5]) and not recipe.league_gate("prey", [0.6, 0.5])
    one = {"passed": True, "blocks": 1}
    assert recipe.league_converged([one] * 4) and not recipe.league_converged([one, one, {"passed": True, "blocks": 2}, one])
    sample = recipe.sample_opponents([(f"v{i}", f"c{i}") for i in range(12)], "v11", 8, 5)
    assert len(sample) == 8 and "v11" not in [l for l, _ in sample]
    assert sample == recipe.sample_opponents([(f"v{i}", f"c{i}") for i in range(12)], "v11", 8, 5)


def test_league_runs_against_the_whole_history(tmp_path) -> None:
    _seed_dir(tmp_path)
    backend = FakeLeagueBackend()
    result = _league(tmp_path, backend).run()
    state = json.loads((tmp_path / "league" / "state.json").read_text())
    rows = run_league.source_rows(json.loads((tmp_path / "seed81" / "state.json").read_text()))
    source = recipe.role_versions(rows)
    assert len(source["predator"]) == 25 and len(source["prey"]) == 12
    # Point 0: the 5 x 7 matrix plus c5's row over the remaining prey versions.
    assert len([k for k in state["steps"] if k.startswith("matrix ")]) == 35 + 5
    t = [x for x in backend.trains]
    assert [x["tag"] for x in t] == ["81_L1pred1", "81_L1pred2", "81_L1prey1", "81_L2pred1", "81_L2prey1",
                                     "81_L3pred1", "81_L3prey1"]
    assert all(x["pool_prob"] == 0.75 for x in t) and [x["seed"] for x in t] == list(range(81501, 81508))
    first_pool = [e["name"] for e in t[0]["pool"]["prey"]]
    assert len(first_pool) == 8 and "c5" not in first_pool and set(first_pool) <= {l for l, _ in source["prey"]}
    prey_pool = [e["name"] for e in t[2]["pool"]["predator"]]
    assert len(prey_pool) == 8 and "L1pred" not in prey_pool          # the latest predator faces the other 25 %
    # Every phase starts with the training role's own optimizer state: its block from its last training checkpoint.
    assert backend.composed[0] == ("start_L1pred", "/ckpt/81_c5_joint.pt", "/ckpt/81_c5.pt")
    assert t[0]["checkpoint"] == "/ckpt/start_L1pred.pt" and t[1]["checkpoint"] == "/ckpt/81_L1pred1.pt"
    assert backend.composed[1] == ("start_L1prey", "/ckpt/81_L1pred2.pt", "/ckpt/81_c5.pt")
    assert t[2]["checkpoint"] == "/ckpt/start_L1prey.pt"
    assert backend.composed[2] == ("start_L2pred", "/ckpt/81_L1pred2.pt", "/ckpt/81_L1prey1.pt")
    assert state["league"]["archives"]["prey"][-3:] == ["L1prey", "L2prey", "L3prey"]
    assert result["stop"] == "equilibrium" and result["verdict"].startswith("history training balances")


def test_league_resumes_and_respects_the_budget(tmp_path) -> None:
    _seed_dir(tmp_path)
    first = FakeLeagueBackend(crash_after=3)
    with pytest.raises(RuntimeError):
        _league(tmp_path, first).run()
    second = FakeLeagueBackend()
    _league(tmp_path, second).run()
    assert not {x["tag"] for x in first.trains} & {x["tag"] for x in second.trains}
    assert len(first.trains) + len(second.trains) == 7
    capped = FakeLeagueBackend()
    (tmp_path / "capped").mkdir()
    _seed_dir(tmp_path / "capped")
    result = _league(tmp_path / "capped", capped, max_updates=2400).run()
    assert len(capped.trains) == 2 and result["stop"] == "budget"
