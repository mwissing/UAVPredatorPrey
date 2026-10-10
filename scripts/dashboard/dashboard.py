#!/usr/bin/env python3
"""Local project dashboard: the story, live status, experiments, results, videos, and the reproduction recipe.

    python3 scripts/dashboard/dashboard.py build   # writes $HOME/RL/artifacts/dashboard/index.html
    python3 scripts/dashboard/dashboard.py serve   # http://127.0.0.1:8765/dashboard/ (rebuilt on every page load)

While open, the page polls /dashboard/status.json every POLL_SECONDS and updates the live parts in place
(progress bars, time left, GPU), so a running training can be watched without reloading.

Host Python, standard library only: the host deliberately has no numpy/torch (training lives in Docker), and
building only reads files, so it is safe while a training run uses the GPU.

The page has two layers. Experiments, live runs, eval curves and videos are generated from the evidence on every
build (docs/experiment_log.md, artifacts/isaac/experiments/*, eval JSONs, console logs), so they cannot go stale.
The curated story (pitch, milestone numbers, stages, featured charts, lessons, recipe, next steps) lives in
content.toml and is edited by hand when a milestone changes.

The server binds to localhost and serves artifacts/ read-only. It answers HTTP Range requests because the
recorded MP4s keep their index (moov atom) at the end of the file; without ranges, browsers cannot seek.
"""

import argparse
import array
import ast
import html
import http.server
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import zipfile
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "artifacts"
EXPERIMENTS = ARTIFACTS / "isaac" / "experiments"
EXPERIMENT_LOG = REPO / "docs" / "experiment_log.md"
CONTENT = Path(__file__).with_name("content.toml")
TEMPLATE = Path(__file__).with_name("template.html")
OUT_DIR = ARTIFACTS / "dashboard"

# Eval JSONs count episode length in policy steps; the policy runs at 50 Hz.
POLICY_HZ = 50.0
# Checkpoints are named by environment step; every agent config collects 24 steps per MAPPO update.
ROLLOUT_STEPS = 24
# An experiment directory with a file written within this window belongs to a live run.
LIVE_WINDOW_S = 15 * 60
# The open page asks /dashboard/status.json this often for the parts that change while a run trains.
POLL_SECONDS = 15
# While serving, a background thread samples the GPU and the live runs' step counters for the Now sparklines.
GPU_SAMPLE_S = 2
STEPS_SAMPLE_S = 10
HISTORY_S = 30 * 60
# Time left uses the mean speed over this window: tqdm's own estimate follows a short moving average and jumps.
SPEED_WINDOW_S = 5 * 60
# Non-breaking space: keeps "1 h 47 min" or "ends ≈ 13:40" from wrapping inside the phrase.
NBSP = chr(0xA0)
# Video trees that only hold codec/anti-aliasing tests, not behavior.
VIDEO_SKIP = ("isaac/diagnostics",)


# --------------------------------------------------------------------------------------------- markdown


def resolve_link(url: str, base_dir: Path) -> str | None:
    """Map a markdown link to a served URL, or None when the target lies outside artifacts/."""
    if re.match(r"^[a-z]+://", url) or url.startswith("#"):
        return url
    target = (base_dir / url).resolve()
    try:
        return "/" + target.relative_to(ARTIFACTS).as_posix()
    except ValueError:
        return None


def inline_md(text: str, base_dir: Path) -> str:
    out = []
    # Split out code spans first so their contents are never formatted.
    for part in re.split(r"(`[^`]+`)", text):
        if len(part) > 1 and part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        part = html.escape(part)

        def link(m: re.Match) -> str:
            href = resolve_link(html.unescape(m.group(2)), base_dir)
            if href is None:
                return m.group(1)
            return f'<a href="{html.escape(href)}">{m.group(1)}</a>'

        part = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, part)
        part = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", part)
        part = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<em>\1</em>", part)
        out.append(part)
    return "".join(out)


LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


def render_list(items: list[tuple[int, bool, str]], base_dir: Path) -> str:
    parts, stack = [], []  # stack of (indent, tag) for the open lists
    for indent, ordered, text in items:
        tag = "ol" if ordered else "ul"
        if not stack or indent > stack[-1][0]:
            parts.append(f"<{tag}>")
            stack.append((indent, tag))
        else:
            while len(stack) > 1 and indent < stack[-1][0]:
                parts.append(f"</li></{stack.pop()[1]}>")
            parts.append("</li>")
        parts.append(f"<li>{inline_md(text, base_dir)}")
    while stack:
        parts.append(f"</li></{stack.pop()[1]}>")
    return "".join(parts)


def markdown_to_html(md: str, base_dir: Path, heading_offset: int = 2) -> str:
    """The subset the project docs use: headings, paragraphs, nested lists, tables, code, inline marks."""
    lines = md.splitlines()
    out, i = [], 0

    def starts_block(line: str) -> bool:
        return bool(line.startswith(("#", "```")) or LIST_ITEM.match(line) or line.lstrip().startswith("|"))

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1 : j])) + "</code></pre>")
            i = j + 1
        elif m := re.match(r"^(#{1,6})\s+(.*)$", line):
            level = min(len(m.group(1)) + heading_offset, 6)
            out.append(f"<h{level}>{inline_md(m.group(2), base_dir)}</h{level}>")
            i += 1
        elif line.lstrip().startswith("|") and i + 1 < len(lines) and TABLE_SEPARATOR.match(lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                if not TABLE_SEPARATOR.match(lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head = "".join(f"<th>{inline_md(c, base_dir)}</th>" for c in rows[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{inline_md(c, base_dir)}</td>" for c in row) + "</tr>" for row in rows[1:]
            )
            out.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        elif LIST_ITEM.match(line):
            items = []
            while i < len(lines) and lines[i].strip():
                if m := LIST_ITEM.match(lines[i]):
                    items.append((len(m.group(1)), m.group(2)[0].isdigit(), m.group(3)))
                elif lines[i].startswith(" ") and items:
                    indent, ordered, text = items[-1]
                    items[-1] = (indent, ordered, text + " " + lines[i].strip())
                else:
                    break
                i += 1
            out.append(render_list(items, base_dir))
        else:
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline_md(' '.join(para), base_dir)}</p>")
    return "\n".join(out)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def md_section(text: str | None, heading: str) -> str | None:
    """Body of the first '## <heading>…' section, up to the next '## ' heading."""
    if not text:
        return None
    m = re.search(rf"^##\s+{re.escape(heading)}[^\n]*\n(.*?)(?=^##\s|\Z)", text, re.M | re.S | re.I)
    return m.group(1).strip() if m else None


def first_paragraph(md: str | None) -> str | None:
    """The first prose block of a markdown text (not a heading, table, list or code block)."""
    for block in re.split(r"\n\s*\n", (md or "").strip()):
        block = block.strip()
        if block and not block.startswith(("#", "|", "```", "- ", "* ")) and not re.match(r"\d+\.\s", block):
            return " ".join(line.strip() for line in block.splitlines())
    return None


def headline_question(md: str | None) -> str | None:
    """The last real question in a PLAN.md 'Question' section; the first paragraph if it asks none."""
    questions = [q.strip() for q in re.findall(r"[^.?!]*\?", " ".join((md or "").split()))]
    questions = [q for q in questions if len(q) > 25]
    return questions[-1] if questions else first_paragraph(md)


PRE_REGISTERED = re.compile(r"\s*\(pre-registered[^,)]*(?:,\s*([^)]*))?\)")


def clean_title(title: str) -> tuple[str, bool]:
    """Drop '(pre-registered …)' and ': results (…)' from a title; report pre-registration as a flag."""
    pre = bool(PRE_REGISTERED.search(title))
    title = PRE_REGISTERED.sub(lambda m: f" ({m.group(1)})" if m.group(1) else "", title)
    title = re.sub(r":\s*results\b.*$", "", title, flags=re.I)
    return title.strip(), pre


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------------------------- eval curves

# "G_46080_vs_ghosts", "G2_46080_full_zeroed_vs_start_predator", "0_own", "cycle3_predator_vs_start_prey"
CURVE_FILE = re.compile(r"^(?:(?P<arm>[A-Za-z][A-Za-z0-9]*)_)?(?P<cycle>cycle)?(?P<x>\d+)_(?P<matchup>[A-Za-z].*)\.json$")

RATES = {
    "catch_rate": "catch",
    "predator_obstacle_crash_rate": "predator crash",
    "prey_obstacle_crash_rate": "prey crash",
    "predator_oob_rate": "predator OOB",
    "prey_oob_rate": "prey OOB",
}
HAZARD_KEYS = ("catch_rate", "predator_obstacle_crash_rate", "prey_obstacle_crash_rate")


def hazard(rate: float, seconds: float) -> float:
    """Constant-hazard reading of a per-episode rate: h = -ln(1 - P) / L, in events per second of flight.

    The same definition as the experiments' summarize.py scripts, so numbers match their RESULTS.md files."""
    return -math.log(1.0 - min(rate, 0.999)) / seconds


def read_metrics(path: Path) -> dict | None:
    try:
        metrics = json.loads(path.read_text())["episode_metrics"]
        return metrics if metrics.get("Metrics/episode_length") else None
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return None


def metric_hazard(m: dict | None, key: str) -> float | None:
    if not m or f"Metrics/{key}" not in m:
        return None
    return hazard(m[f"Metrics/{key}"], m["Metrics/episode_length"] / POLICY_HZ)


def load_evals(exp_dir: Path) -> tuple[dict[tuple[str, str], dict[int, dict]], bool]:
    """Episode metrics by (arm, matchup) and x from curve/ and evals/; x is the training update or the cycle."""
    found: dict[tuple[str, str], dict[int, dict]] = {}
    is_cycle = False
    for sub in ("curve", "evals"):
        for path in sorted((exp_dir / sub).glob("*.json")):
            m = CURVE_FILE.match(path.name)
            if not m or (metrics := read_metrics(path)) is None:
                continue
            x = int(m.group("x"))
            if m.group("cycle"):
                is_cycle = True
            else:
                x //= ROLLOUT_STEPS
            found.setdefault((m.group("arm") or "run", m.group("matchup")), {})[x] = metrics
    return found, is_cycle


def explorer_point(m: dict) -> dict:
    seconds = m["Metrics/episode_length"] / POLICY_HZ
    point = {"episode length (s)": seconds}
    for key, name in RATES.items():
        if f"Metrics/{key}" in m:
            point[f"{name} rate (per episode)"] = m[f"Metrics/{key}"]
    for key in HAZARD_KEYS:
        if f"Metrics/{key}" in m:
            point[f"{RATES[key]} hazard (/s)"] = hazard(m[f"Metrics/{key}"], seconds)
    return point


def load_explorer(exp_dir: Path) -> dict | None:
    """Every eval series of one experiment, for the per-experiment chart with matchup and metric pickers."""
    evals, is_cycle = load_evals(exp_dir)
    if not evals:
        return None
    points = {key: {x: explorer_point(m) for x, m in by_x.items()} for key, by_x in evals.items()}
    # A "zeroed" twin evaluates the same checkpoint with its obstacle branch output set to zero. The ratio of
    # crash hazards (zeroed / active) is the project's perception test: >= 1.2 means the branch is really used.
    for (arm, matchup), by_x in list(points.items()):
        if "zeroed" not in matchup:
            continue
        base = matchup.replace("zeroed_", "", 1) if "zeroed_" in matchup else matchup.replace("_zeroed", "", 1)
        active = points.get((arm, base))
        if not active:
            continue
        ratio = {}
        for x, zeroed in by_x.items():
            if x in active:
                ratio[x] = {
                    f"{role} crash ablation ratio (zeroed / active)": zeroed[f"{role} crash hazard (/s)"]
                    / active[x][f"{role} crash hazard (/s)"]
                    for role in ("predator", "prey")
                    if active[x].get(f"{role} crash hazard (/s)") and zeroed.get(f"{role} crash hazard (/s)") is not None
                }
        points[(arm, f"ablation ratio: {base}")] = ratio
    return {
        "x_label": "self-play cycle" if is_cycle else "training updates",
        "arms": sorted({arm for arm, _ in points}),
        "series": [
            {"arm": arm, "matchup": matchup, "points": [{"x": x, "m": p} for x, p in sorted(by_x.items())]}
            for (arm, matchup), by_x in sorted(points.items())
        ],
    }


def featured_lines(spec: dict) -> dict | None:
    """A curated line chart: hazards of named matchups, or ablation ratios of with/zeroed pairs."""
    evals, is_cycle = load_evals(EXPERIMENTS / spec.get("experiment", "-"))
    series = []
    for s in spec.get("series", []):
        arm, metric = s.get("arm", "run"), s["metric"]
        if "zeroed" in s:
            active, zeroed = evals.get((arm, s["with"]), {}), evals.get((arm, s["zeroed"]), {})
            points = []
            for x in sorted(set(active) & set(zeroed)):
                h_active, h_zeroed = metric_hazard(active[x], metric), metric_hazard(zeroed[x], metric)
                if h_active and h_zeroed is not None:
                    points.append([x, h_zeroed / h_active])
        else:
            by_x = evals.get((arm, s["matchup"]), {})
            points = [[x, h] for x in sorted(by_x) if (h := metric_hazard(by_x[x], metric)) is not None]
        if points:
            series.append({"name": s["name"], "color": s.get("color", "s1"), "points": points})
    if not series:
        return None
    return {
        "kind": "lines",
        "x_label": "self-play cycle" if is_cycle else "training updates",
        "y_label": spec.get("y_label", ""),
        "series": series,
        "refs": spec.get("refs", []),
        "zero": spec.get("zero", True),
    }


def featured_league(spec: dict) -> dict | None:
    """Bradley–Terry strengths of both roles from a league's fit, sorted strongest first."""
    try:
        data = json.loads((ARTIFACTS / spec["path"]).read_text())
    except (OSError, KeyError, ValueError):
        return None
    labels, highlight = spec.get("labels", {}), set(spec.get("highlight", []))
    groups = []
    for role, name, color in (("predator", "Predators", "s2"), ("prey", "Prey", "s1")):
        bars = [
            {"label": labels.get(k, k), "value": v["logit"], "se": v.get("se", 0.0), "highlight": k in highlight}
            for k, v in data.get(role, {}).items()
        ]
        groups.append({"name": name, "color": color, "bars": sorted(bars, key=lambda b: -b["value"])})
    return {"kind": "league", "x_label": "strength (logit)", "groups": groups}


# --------------------------------------------------------------------------------------------- experiments


def last_activity(directory: Path) -> float:
    newest = 0.0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def parse_clock(text: str) -> int | None:
    """tqdm durations 'MM:SS' or 'H:MM:SS' in seconds; None for '?'."""
    if not re.fullmatch(r"\d+(?::\d{2}){1,2}", text):
        return None
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def human_duration(seconds: int) -> str:
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}{NBSP}h{NBSP}{minutes:02d}{NBSP}min" if hours else f"{minutes}{NBSP}min"


def tqdm_progress(path: Path) -> dict | None:
    """The last tqdm update in a console log: done/total plus elapsed and remaining time."""
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        tail = f.read().decode("utf-8", errors="replace")
    found = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)", tail)
    if not found:
        return None
    done, total, elapsed, remaining = found[-1]
    left = parse_clock(remaining)
    return {
        "done": int(done),
        "total": int(total),
        "share": int(done) / max(int(total), 1),
        "left": human_duration(left) if left is not None else None,
        "eta": time.strftime("%H:%M", time.localtime(path.stat().st_mtime + left)) if left is not None else None,
        "arm": path.name.removesuffix(".console.log"),
    }


def videos_under(directory: Path) -> list[dict]:
    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d != "checkpoints")
        for name in sorted(files):
            if not name.endswith(".mp4"):
                continue
            path = Path(root) / name
            rel = path.relative_to(ARTIFACTS).as_posix()
            if rel.startswith(VIDEO_SKIP):
                continue
            group, _, label = rel.rpartition("/videos/")
            if not group:
                group, label = rel.rsplit("/", 1)[0], name
            label = label.removesuffix("/rl-video-step-0.mp4").removesuffix(".mp4").replace("_", " ")
            stat = path.stat()
            found.append({"url": "/" + rel, "group": group, "label": label, "mb": stat.st_size / 1e6, "mtime": stat.st_mtime})
    return found


def heading_title(results: str | None, plan: str | None, fallback: str) -> tuple[str, bool]:
    """Title from the first '# ' heading of RESULTS.md, else PLAN.md, plus the pre-registration flag."""
    heading = re.search(r"(?m)^# (.+)$", results or plan or "")
    return clean_title(heading.group(1)) if heading else (fallback, False)


def live_state(exp_dir: Path, now: float) -> tuple[bool, dict | None]:
    """Live: a file in the experiment was written within LIVE_WINDOW_S. Progress comes from its newest console log."""
    if now - last_activity(exp_dir) >= LIVE_WINDOW_S:
        return False, None
    consoles = sorted(exp_dir.rglob("*.console.log"), key=lambda path: path.stat().st_mtime)
    if consoles and now - consoles[-1].stat().st_mtime < LIVE_WINDOW_S:
        return True, tqdm_progress(consoles[-1])
    return True, None  # evaluating or between phases


def load_experiment_dirs() -> dict[str, dict]:
    dirs = {}
    now = time.time()
    for exp_dir in sorted(p for p in EXPERIMENTS.iterdir() if p.is_dir()):
        results, plan = read_text(exp_dir / "RESULTS.md"), read_text(exp_dir / "PLAN.md")
        title, pre = heading_title(results, plan, exp_dir.name)
        if plan and re.search(r"(?mi)^# .*pre-registered", plan):
            pre = True
        live, progress = live_state(exp_dir, now)
        dirs[exp_dir.name] = {
            "name": exp_dir.name,
            "dir": exp_dir,
            "title": title,
            "preregistered": pre,
            "status": "running" if live else ("done" if results else "planned"),
            "progress": progress,
            "results": results,
            "plan": plan,
            "summary": read_text(exp_dir / "summary.txt"),
            "statuses": [
                (p.relative_to(exp_dir).as_posix().removesuffix(".status.txt"), p.read_text().strip())
                for p in sorted(exp_dir.rglob("*.status.txt"))
            ],
            "explorer": load_explorer(exp_dir),
            "videos": videos_under(exp_dir),
        }
    return dirs


LOG_HEADING = re.compile(r"^## (?P<date>\d{4}-\d{2}-\d{2})(?P<more>[^:]*):\s*(?P<title>.+)$")
LOG_FIELD = re.compile(r"^- (?P<label>[A-Z][\w ,/'()+-]{0,60}?):(?:\s+(?P<text>.*))?$")


def log_fields(body: str) -> list[dict]:
    """Top-level bullets of a log entry: '- Result (…): text' → key 'result', label 'Result'.

    Sub-bullets are left out, except that a bare '- Checks:' takes its first sub-bullet as its text."""
    fields, in_sub = [], False
    for line in body.splitlines():
        if line.startswith("- "):
            m = LOG_FIELD.match(line)
            label = re.sub(r"\s*\(.*$", "", m.group("label")).strip(" ,") if m else ""
            fields.append({"key": label.split()[0].lower() if label else "", "label": label,
                           "text": ((m.group("text") or "") if m else line[2:]).strip()})
            in_sub = False
        elif fields and (sub := re.match(r"\s+[-*] (.*)", line)):
            if not fields[-1]["text"]:
                fields[-1]["text"] = sub.group(1).strip()
            in_sub = True
        elif fields and line.strip() and not in_sub:
            fields[-1]["text"] += " " + line.strip()
    return fields


def parse_log() -> list[dict]:
    entries = []
    for chunk in re.split(r"(?m)^(?=## )", read_text(EXPERIMENT_LOG) or ""):
        first, _, body = chunk.partition("\n")
        m = LOG_HEADING.match(first.strip())
        if not m:
            continue
        title, pre = clean_title(m.group("title"))
        entries.append({
            "date": m.group("date"),
            "date_label": (m.group("date") + m.group("more")).strip(),
            "title": title,
            "preregistered": pre,
            "body": body.strip(),
            "fields": log_fields(body),
            "dirs": list(dict.fromkeys(re.findall(r"artifacts/isaac/experiments/([\w.-]+)", body))),
        })
    return entries


def summary_rows(fields: list[dict]) -> list[tuple[str, str]]:
    """Question → result → decision, the shape most log entries follow; otherwise the first bullets."""
    def first(*keys: str) -> dict | None:
        return next((f for f in fields if f["key"] in keys), None)

    picks = [first("question", "idea"), first("result"), first("decision", "decisions", "reading", "lesson", "lessons")]
    rows = [(f["label"], f["text"]) for f in picks if f]
    if len(rows) < 2:
        rows = [(f["label"], f["text"]) for f in fields[:3]]
    return rows


def build_cards(entries: list[dict], dirs: dict[str, dict]) -> tuple[list[dict], dict[str, str]]:
    """One card per experiment-log entry, plus one per experiment directory the log does not cite yet."""
    cards, dir_card = [], {}
    for order, e in enumerate(entries):
        linked = [dirs[name] for name in e["dirs"] if name in dirs]
        card_id = "log-" + slug(f'{e["date"]} {e["title"]}')
        for d in linked:
            dir_card.setdefault(d["name"], card_id)
        cards.append({
            "id": card_id, "date": e["date"], "date_label": e["date_label"], "title": e["title"],
            "preregistered": e["preregistered"] or any(d["preregistered"] for d in linked[:1]),
            "status": "running" if any(d["status"] == "running" for d in linked) else "done",
            "rows": summary_rows(e["fields"]), "body": e["body"], "dirs": linked, "order": (e["date"], 1, order),
        })
    for name, d in dirs.items():
        if name in dir_card:
            continue
        card_id = "exp-" + slug(name)
        dir_card[name] = card_id
        result = first_paragraph(md_section(d["results"], "Short answer")) or first_paragraph(
            re.sub(r"(?m)^# .*$", "", d["results"] or "", count=1))
        rows = [(k, v) for k, v in (("Question", headline_question(md_section(d["plan"], "Question"))), ("Result", result)) if v]
        cards.append({
            "id": card_id, "date": name[:10], "date_label": name[:10], "title": d["title"],
            "preregistered": d["preregistered"], "status": d["status"], "rows": rows, "body": None, "dirs": [d],
            "order": (name[:10], 0, 0),
        })
    cards.sort(key=lambda c: (c["status"] == "running", c["order"]), reverse=True)
    return cards, dir_card


# --------------------------------------------------------------------------------------------- 3D replay

# Geometry of the elevated 3v1 task (env cfg: arena centre z 6 m, soft arena radius 5 m; pillars of radius 0.4 m,
# 12 m tall, centred on the arena centre).
ARENA = {"center": [0.0, 0.0, 6.0], "radius": 5.0}
PILLAR = {"radius": 0.4, "height": 12.0}
# Some evaluations park absent pillars 50 m away; those are not drawn.
PARKED_PILLAR_M = 20.0
# Part of the cache file name: bump it after changing extract_replay, and old caches are ignored.
REPLAY_FORMAT = 2
# Served next to the page: the replay module and the vendored Three.js it imports.
STATIC_FILES = {Path(__file__).with_name("replay.js"): OUT_DIR / "replay.js",
                Path(__file__).with_name("vendor"): OUT_DIR / "vendor"}


def read_npy(z: zipfile.ZipFile, name: str) -> tuple[tuple, "array.array | bytes"] | None:
    """One array of an .npz archive without numpy: its shape and its values, flat in C order."""
    if name not in z.namelist():
        return None
    with z.open(name) as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"{name}: not an .npy array")
        major = f.read(2)[0]
        header_len = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(header_len).decode("latin1"))
        raw = f.read()
    if header.get("fortran_order"):
        raise ValueError(f"{name}: Fortran order is not supported")
    if header["descr"] == "<f4":
        values = array.array("f", raw)
        if sys.byteorder != "little":
            values.byteswap()
    elif header["descr"] == "|b1":
        values = raw
    else:
        raise ValueError(f"{name}: unsupported dtype {header['descr']}")
    return tuple(header["shape"]), values


def pick_episodes(o: dict[str, list[float]]) -> list[tuple[str, str, int]]:
    """Representative episodes by outcome as (key, label, index). Fixed rules, so every build shows the same ones."""
    lengths = o["episode_length"]
    n, timeout = len(lengths), max(lengths)

    def flag(key: str, i: int) -> bool:
        return key in o and o[key][i] > 0.5

    def middle(candidates: list[int], key) -> int:
        return sorted(candidates, key=key)[len(candidates) // 2]

    catches = [i for i in range(n) if flag("catch", i) and not flag("predator_obstacle_crash", i)]
    prey_crashes = [i for i in range(n) if flag("prey_obstacle_crash", i)]
    predator_crashes = [i for i in range(n) if flag("predator_obstacle_crash", i)]
    predator_crashes = [i for i in predator_crashes if lengths[i] >= 100] or predator_crashes  # long enough to watch
    escapes = [i for i in range(n) if lengths[i] >= timeout and not flag("catch", i) and not flag("prey_oob", i)]
    picks = []
    if catches:
        picks += [("catch", "Typical catch", middle(catches, lambda i: lengths[i])),
                  ("long-chase", "Longest chase", max(catches, key=lambda i: (lengths[i], -i)))]
    if prey_crashes:
        picks.append(("prey-crash", "Prey hits a pillar", middle(prey_crashes, lambda i: lengths[i])))
    if predator_crashes:
        picks.append(("predator-crash", "Predator hits a pillar", middle(predator_crashes, lambda i: lengths[i])))
    if escapes:
        picks += [("escape", "Prey escapes", middle(escapes, lambda i: o["closest_approach"][i])),
                  ("near-miss", "Near miss", min(escapes, key=lambda i: o["closest_approach"][i]))]
    seen: set[int] = set()
    return [p for p in picks if not (p[2] in seen or seen.add(p[2]))]


def track(shape: tuple, values, episode: int, frames: int, agent: int | None = None) -> list[float]:
    """x, y, z of one drone for each frame of one episode, flat, to the centimetre. Arrays are (T, E[, A], 3)."""
    steps, envs = shape[0], shape[1]
    agents, a = (shape[2], agent) if agent is not None else (1, 0)
    out: list[float] = []
    for t in range(min(frames, steps)):
        base = ((t * envs + episode) * agents + a) * 3
        out += (round(values[base], 2), round(values[base + 1], 2), round(values[base + 2], 2))
    return out


def death_frame(shape: tuple, values, episode: int, frames: int, agent: int) -> int | None:
    """First frame at which a predator is no longer alive (crash or out of bounds), or None."""
    steps, envs, agents = shape
    return next((t for t in range(min(frames, steps)) if values[(t * envs + episode) * agents + agent] < 0.5), None)


def end_code(o: dict[str, list[float]], i: int) -> str:
    """How episode i ended: catch, prey-crash, prey-oob, timeout or other."""
    def flag(key: str) -> bool:
        return key in o and o[key][i] > 0.5

    if flag("catch"):
        return "catch"
    if flag("prey_obstacle_crash"):
        return "prey-crash"
    if flag("prey_oob"):
        return "prey-oob"
    return "timeout" if o["episode_length"][i] >= max(o["episode_length"]) else "other"


def end_text(o: dict[str, list[float]], i: int) -> str:
    seconds = o["episode_length"][i] / POLICY_HZ
    return {"catch": f"caught after {seconds:.1f} s",
            "prey-crash": f"prey crashed into a pillar after {seconds:.1f} s",
            "prey-oob": f"prey left the arena after {seconds:.1f} s",
            "timeout": f"prey survived the full {seconds:.0f} s"}.get(end_code(o, i), f"ended after {seconds:.1f} s")


def predator_down(o: dict[str, list[float]], i: int) -> bool:
    """A predator crashed into a pillar or left the arena in episode i."""
    keys = [k for k in o if k == "predator_obstacle_crash" or re.fullmatch(r"predator_\d+_oob", k)]
    return any(o[k][i] > 0.5 for k in keys)


def read_trajectories(npz: Path, positions: bool = True) -> dict:
    """The outcome arrays of an evaluation's trajectory file and, with positions=True, its tracks (about 0.1 s)."""
    with zipfile.ZipFile(npz) as z:
        t = {"outcomes": {name[len("outcome_"):-len(".npy")]: list(read_npy(z, name)[1])
                          for name in z.namelist() if name.startswith("outcome_") and name.endswith(".npy")}}
        if positions:
            for key, member in (("predators", "predator_pos.npy"), ("prey", "prey_pos.npy"),
                                ("alive", "predator_alive.npy"), ("pillars", "obstacle_xy.npy")):
                t[key] = read_npy(z, member)
    return t


def episode_record(t: dict, i: int) -> dict:
    """What the replay draws for episode i (0-based): tracks to the centimetre, the pillars, who went down when."""
    o, (pred_shape, pred) = t["outcomes"], t["predators"]
    frames = int(o["episode_length"][i])
    xy = []
    if t["pillars"]:  # (T, E, K, 2); pillars do not move, so the first frame is enough
        n_pillars, values = t["pillars"][0][2], t["pillars"][1]
        for k in range(n_pillars):
            x, y = values[(i * n_pillars + k) * 2], values[(i * n_pillars + k) * 2 + 1]
            if math.hypot(x, y) < PARKED_PILLAR_M:
                xy.append([round(x, 3), round(y, 3)])
    return {
        "index": i, "frames": frames, "end": end_text(o, i),
        "closest": round(o["closest_approach"][i], 2) if "closest_approach" in o else None,
        "pillars": xy,
        "prey": track(*t["prey"], i, frames),
        "predators": [track(pred_shape, pred, i, frames, a) for a in range(pred_shape[2])],
        "dead": [death_frame(*t["alive"], i, frames, a) if t["alive"] else None for a in range(pred_shape[2])],
    }


def replay_index(o: dict[str, list[float]]) -> dict:
    """One line per episode (length, how it ended, whether a predator went down), for browsing in the page."""
    n = len(o["episode_length"])
    return {
        "format": REPLAY_FORMAT, "dt": 1.0 / POLICY_HZ, "episodes_total": n, "arena": ARENA, "pillar": PILLAR,
        "highlights": [{"key": key, "label": label, "index": i} for key, label, i in pick_episodes(o)],
        "length": [int(x) for x in o["episode_length"]],
        "end": [end_code(o, i) for i in range(n)],
        "down": [int(predator_down(o, i)) for i in range(n)],
        "closest": [round(x, 2) for x in o["closest_approach"]] if "closest_approach" in o else None,
    }


def replay_dir(spec_id: str) -> Path:
    return OUT_DIR / "replay" / f"{spec_id}-v{REPLAY_FORMAT}"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")  # threads may write the same file at once
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def ensure_replay(spec: dict) -> dict | None:
    """Index and highlight episodes of one [[replay]] source, extracted again when the source is newer. Any other
    episode is extracted when the page asks for it (ensure_episode, called by the server)."""
    src = ARTIFACTS / spec["path"]
    if not src.is_file():
        return None
    folder = replay_dir(spec["id"])
    index = folder / "index.json"
    if not index.is_file() or index.stat().st_mtime < src.stat().st_mtime:
        t = read_trajectories(src)
        data = replay_index(t["outcomes"])
        shutil.rmtree(folder, ignore_errors=True)  # episodes of an older version of the source
        for h in data["highlights"]:
            write_json(folder / f"{h['index'] + 1}.json", episode_record(t, h["index"]))
        write_json(index, data)
    return {"id": spec["id"], "title": spec["title"], "note": spec.get("note", ""), "base": f"replay/{folder.name}/"}


def ensure_episode(spec_id: str, number: int) -> None:
    """Extract episode `number` (1-based, as in the page and its links) of a replay source unless it is cached."""
    spec = next((r for r in load_content().get("replay", []) if r.get("id") == spec_id), None)
    src = ARTIFACTS / spec["path"] if spec else None
    if not src or not src.is_file():
        return
    out = replay_dir(spec_id) / f"{number}.json"
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return
    t = read_trajectories(src)
    if 1 <= number <= len(t["outcomes"]["episode_length"]):
        write_json(out, episode_record(t, number - 1))


def remove_old_replays() -> None:
    """Cached replay files of an older REPLAY_FORMAT."""
    root = OUT_DIR / "replay"
    for p in root.iterdir() if root.is_dir() else []:
        if not p.name.endswith(f"-v{REPLAY_FORMAT}"):
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def sync_static() -> None:
    """Copy the replay module and the vendored Three.js next to the page when they changed."""
    for src, dst in STATIC_FILES.items():
        files = [(src, dst)] if src.is_file() else [(f, dst / f.relative_to(src)) for f in src.rglob("*") if f.is_file()]
        for f, target in files:
            if not target.exists() or target.stat().st_mtime < f.stat().st_mtime or target.stat().st_size != f.stat().st_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)


# --------------------------------------------------------------------------------------------- live state


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def live_trainers() -> list[dict]:
    """Training processes inside the container are visible to the host's ps; keep the interpreter, not wrappers."""
    trainers = []
    for line in run(["ps", "-eo", "etimes=,args="]).splitlines():
        elapsed, _, args = line.strip().partition(" ")
        if "kit/python/bin/python3" not in args or not re.search(r"scripts/skrl/(train|evaluate)\.py", args):
            continue
        flag = lambda name: (re.search(rf"--{name}[ =](\S+)", args) or [None, None])[1]
        trainers.append({
            "script": "train" if "train.py" in args else "evaluate",
            "elapsed_min": int(elapsed) // 60,
            "task": flag("task"),
            "num_envs": flag("num_envs"),
            "seed": flag("seed"),
            "overrides": re.findall(r"(?<=\s)(\+?[\w.]+=\S+)", args),
        })
    return trainers


def gpu_state() -> dict | None:
    out = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
               "--format=csv,noheader,nounits"])
    if not out:
        return None
    name, util, used, total, temp = [s.strip() for s in out.splitlines()[0].split(",")]
    return {"name": name.removeprefix("NVIDIA ").removeprefix("GeForce "), "util": int(util),
            "used_gb": int(used) / 1024, "total_gb": int(total) / 1024, "temp": int(temp)}


class Sampler(threading.Thread):
    """History for the Now sparklines while the server runs: the GPU every GPU_SAMPLE_S, and every STEPS_SAMPLE_S
    the step counter of each live run. Kept in memory for HISTORY_S, so it starts empty when the server starts."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.gpu = deque(maxlen=HISTORY_S // GPU_SAMPLE_S)  # (time, gpu_state())
        self.steps: dict[str, deque] = {}  # run name -> (time, timesteps done)

    def run(self) -> None:
        next_scan = 0.0
        while True:
            start = time.time()
            try:
                gpu = gpu_state()
                if gpu:
                    with self.lock:
                        self.gpu.append((start, gpu))
                if start >= next_scan:
                    next_scan = start + STEPS_SAMPLE_S
                    for exp_dir in [p for p in EXPERIMENTS.iterdir() if p.is_dir()]:
                        live, progress = live_state(exp_dir, start)
                        if live and progress:
                            with self.lock:
                                series = self.steps.setdefault(exp_dir.name, deque(maxlen=HISTORY_S // STEPS_SAMPLE_S))
                                series.append((start, progress["done"]))
            except Exception as exc:  # one failed sample must not end the history
                print(f"sampler: {exc!r}", flush=True)
            time.sleep(max(0.2, GPU_SAMPLE_S - (time.time() - start)))

    def latest_gpu(self) -> dict | None:
        with self.lock:
            return self.gpu[-1][1] if self.gpu else None

    def gpu_load(self) -> list[tuple[float, float]]:
        with self.lock:
            return [(t, g["util"]) for t, g in self.gpu]

    def step_counts(self, run: str) -> list[tuple[float, float]]:
        """Samples since the counter last restarted (a new training phase counts from 0 again)."""
        with self.lock:
            samples = list(self.steps.get(run, ()))
        start = max((i for i in range(1, len(samples)) if samples[i][1] < samples[i - 1][1]), default=0)
        return samples[start:]


SAMPLER: Sampler | None = None  # started by serve(); a plain `build` has no history


def mean_rate(samples: list[tuple[float, float]], window_s: float) -> float | None:
    """Timesteps per second over the last window_s of (time, timesteps) samples; None with less than a minute."""
    if not samples:
        return None
    recent = [s for s in samples if s[0] >= samples[-1][0] - window_s]
    span = recent[-1][0] - recent[0][0]
    return (recent[-1][1] - recent[0][1]) / span if span >= 60 else None


def speed_series(samples: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Timesteps per second between consecutive samples, stamped at each interval's midpoint."""
    return [((t0 + t1) / 2, (d1 - d0) / (t1 - t0)) for (t0, d0), (t1, d1) in zip(samples, samples[1:]) if t1 > t0]


def bucket_means(points: list[tuple[float, float]], width_s: float) -> list[tuple[float, float]]:
    """The mean of each time bucket: fewer points to draw and less sampling jitter."""
    buckets: dict[int, list[tuple[float, float]]] = {}
    for t, v in points:
        buckets.setdefault(int(t // width_s), []).append((t, v))
    return [(sum(t for t, _ in b) / len(b), sum(v for _, v in b) / len(b)) for _, b in sorted(buckets.items())]


def time_left(p: dict, run: str | None) -> tuple[str | None, str | None]:
    """Time left and finish clock from the mean speed over SPEED_WINDOW_S, once the server has sampled for a
    minute; until then tqdm's own estimate."""
    rate = mean_rate(SAMPLER.step_counts(run), SPEED_WINDOW_S) if SAMPLER and run else None
    if rate and rate > 0:
        seconds = (p["total"] - p["done"]) / rate
        return human_duration(int(seconds)), time.strftime("%H:%M", time.localtime(time.time() + seconds))
    return p["left"], p["eta"]


def git_state() -> dict:
    git = ["git", "-C", str(REPO)]
    branch = run(git + ["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "branch": branch,
        "head": run(git + ["log", "-1", "--format=%h"]),
        "subject": run(git + ["log", "-1", "--format=%s"]),
        "when": run(git + ["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"]),
        "ahead": run(git + ["rev-list", "--count", f"origin/{branch}..HEAD"]),
        "dirty": bool(run(git + ["status", "--porcelain", "--untracked-files=no"])),
    }


# --------------------------------------------------------------------------------------------- rendering

# Status icons as inline SVG: font glyphs such as ◐ render at very different sizes across system fonts.
ICONS = {
    "check": '<svg viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "half": '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 2.8a5.2 5.2 0 0 1 0 10.4z" fill="currentColor"/></svg>',
    "cross": '<svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    "dot": '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="4.5" fill="currentColor"/></svg>',
    "ring": '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
}
VERDICTS = {
    "worked": ("good", ICONS["check"], "worked"),
    "partial": ("warning", ICONS["half"], "partly"),
    "failed": ("critical", ICONS["cross"], "did not work"),
    "running": ("accent", ICONS["dot"], "running"),
    "done": ("good", ICONS["check"], "done"),
    "planned": ("muted", ICONS["ring"], "planned"),
    "open": ("muted", ICONS["ring"], "open"),
    "verified": ("good", ICONS["check"], "verified"),
    "reconstructed": ("warning", ICONS["half"], "reconstructed, untested"),
    "gap": ("critical", ICONS["cross"], "never run"),
}


def esc(value) -> str:
    return html.escape(str(value))


def badge(verdict: str, label: str | None = None) -> str:
    tone, icon, default = VERDICTS.get(verdict, ("muted", ICONS["ring"], verdict))
    return f'<span class="badge {tone}"><span class="badge-icon" aria-hidden="true">{icon}</span>{esc(label or default)}</span>'


def md(text: str) -> str:
    """Curated text from content.toml: markdown, links relative to artifacts/."""
    return markdown_to_html(text.strip(), ARTIFACTS)


def md_inline(text: str, base: Path = ARTIFACTS) -> str:
    return inline_md(" ".join(text.split()), base)


def thumb(url: str, label: str, show_label: bool = True) -> str:
    """A lazy video thumbnail: the browser shows a frame from 5 s in; clicking opens the player."""
    caption = f'<span class="thumb-label">{esc(label)}</span>' if show_label and label else ""
    return (f'<button type="button" class="thumb" data-play="{esc(url)}" data-caption="{esc(label)}" '
            f'aria-label="Play video: {esc(label)}">'
            f'<video muted playsinline preload="none" tabindex="-1" aria-hidden="true" data-src="{esc(url)}#t=5"></video>'
            f'<span class="thumb-play" aria-hidden="true"></span>{caption}</button>')


def evidence_list(paths: list[str]) -> str:
    items = []
    for p in paths:
        url = resolve_link(p.removeprefix("artifacts/"), ARTIFACTS) if p.startswith("artifacts/") else None
        items.append(f'<li><a href="{esc(url)}"><code>{esc(p)}</code></a></li>' if url else f"<li><code>{esc(p)}</code></li>")
    return f'<ul class="paths">{"".join(items)}</ul>'


def section_head(eyebrow: str, title: str, lead: str = "") -> str:
    lead_html = f'<div class="lead">{lead}</div>' if lead else ""
    return f'<div class="section-head"><p class="eyebrow">{esc(eyebrow)}</p><h2>{esc(title)}</h2>{lead_html}</div>'


def sparkline(points: list[tuple[float, float]], y_max: float, recent_s: float, label: str) -> str:
    """A trend line without axes: the history in the de-emphasis gray, the window behind the headline number in the
    accent. Inline SVG, so it arrives inside the live fragments; the <title> carries the numbers as text."""
    if len(points) < 3 or y_max <= 0:
        return ""
    w, h = 120, 28
    t0, t1 = points[0][0], points[-1][0]
    span = (t1 - t0) or 1.0

    def xy(t: float, v: float) -> str:
        return f"{(t - t0) / span * w:.1f},{h - 1 - min(max(v, 0.0), y_max) / y_max * (h - 2):.1f}"

    line = " ".join(xy(t, v) for t, v in points)
    first_recent = next((i for i, (t, _) in enumerate(points) if t >= t1 - recent_s), len(points) - 1)
    recent = " ".join(xy(t, v) for t, v in points[max(first_recent - 1, 0):])
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" aria-label="{esc(label)}">'
            f'<title>{esc(label)}</title><polygon class="spark-area" points="0,{h} {line} {w},{h}"/>'
            f'<polyline class="spark-line" points="{line}" vector-effect="non-scaling-stroke"/>'
            f'<polyline class="spark-line now" points="{recent}" vector-effect="non-scaling-stroke"/></svg>')


def span_text(points: list[tuple[float, float]]) -> str:
    """How much history a sparkline shows: up to HISTORY_S, less right after the server started."""
    return f"last {max(1, round((points[-1][0] - points[0][0]) / 60))}{NBSP}min"


def progress_inner(p: dict | None, compact: bool = False, run: str | None = None) -> str:
    if not p:
        return '<p class="muted small">evaluating or between phases (no training progress in the last 15 min)</p>'
    left, eta = time_left(p, run)
    left_text = f" · ~{esc(left)}{NBSP}left · ends{NBSP}≈{NBSP}{esc(eta)}" if left else ""
    # A timestep advances all parallel environments at once (4096 in the obstacle runs).
    steps = "" if compact else f" · {p['done']:,}{NBSP}/{NBSP}{p['total']:,} timesteps"
    out = (f'<div class="meter" role="progressbar" aria-valuenow="{p["share"] * 100:.0f}" aria-valuemin="0" aria-valuemax="100">'
           f'<div style="width:{max(p["share"] * 100, 1.5):.1f}%"></div></div>'
           f'<p class="meter-text"><strong>{p["share"]:.0%}</strong>{steps}{left_text}</p>')
    if compact or not (SAMPLER and run):
        return out
    counts = SAMPLER.step_counts(run)
    rate = mean_rate(counts, SPEED_WINDOW_S)
    points = bucket_means(speed_series(counts), 30)
    if not rate or len(points) < 3:
        return out
    values = [v for _, v in points]
    label = (f"Training speed, {span_text(points)}: min {min(values):.1f}, mean {sum(values) / len(values):.1f}, "
             f"max {max(values):.1f} timesteps per second")
    window_min = max(1, round(min(SPEED_WINDOW_S, counts[-1][0] - counts[0][0]) / 60))  # shorter right after a start
    return (out + f'<div class="spark-row"><span>speed</span><span><b>{rate:.1f}</b> timesteps/s</span></div>'
            f'{sparkline(points, max(values) * 1.15, SPEED_WINDOW_S, label)}'
            f'<p class="spark-caption">{window_min}-min mean · sets the time left · graph: {span_text(points)}</p>')


def progress_html(run: str, p: dict | None, compact: bool = False) -> str:
    """A live run's progress; while the page is open it swaps in fresh inner HTML from /dashboard/status.json."""
    attrs = f' data-run="{esc(run)}"' + (' data-compact="1"' if compact else "")
    return f'<div class="progress"{attrs}>{progress_inner(p, compact, run)}</div>'


def pill_inner(title: str | None, p: dict | None, run: str | None = None) -> str:
    """The hero's one-line live status."""
    if title is None:
        return '<span class="pulse idle" aria-hidden="true"></span><span>No training running</span>'
    detail = ""
    if p:
        left, _ = time_left(p, run)
        detail = f" · {p['share']:.0%}" + (f" · ~{esc(left)}{NBSP}left" if left else "")
    short = title.split(":")[0]  # "Prey ghost stage v5: slow ghosts …" → "Prey ghost stage v5"
    return f'<span class="pulse" aria-hidden="true"></span><span>Training now: <strong>{esc(short)}</strong>{detail}</span>'


def gpu_html(gpu: dict | None) -> str:
    if not gpu:
        return '<p class="muted">GPU not visible (nvidia-smi unavailable).</p>'
    head = f'<div class="gpu-head"><span>{esc(gpu["name"])}</span><span class="muted small">{gpu["temp"]} °C</span></div>'
    memory = (f'<div class="gauge"><span>memory</span><div class="meter small"><div style="width:{gpu["used_gb"] / gpu["total_gb"] * 100:.0f}%"></div></div>'
              f'<b>{gpu["used_gb"]:.1f} / {gpu["total_gb"]:.0f} GB</b></div>')
    # One nvidia-smi reading covers about the last second, and training alternates between simulation and PPO
    # updates, so single readings jump. The headline is the mean of the last minute; the sparkline shows the trend.
    load = SAMPLER.gpu_load() if SAMPLER else []
    last_minute = [v for t, v in load if t >= time.time() - 60]
    points = bucket_means(load, 15)
    if len(points) < 3 or not last_minute:
        return (f'<div class="gpu">{head}<div class="gauge"><span>load</span><div class="meter small"><div style="width:{gpu["util"]}%">'
                f'</div></div><b>{gpu["util"]} %</b></div>{memory}</div>')
    values = [v for _, v in load]
    label = f"GPU load, {span_text(points)}: min {min(values):.0f} %, mean {sum(values) / len(values):.0f} %, max {max(values):.0f} %"
    return (f'<div class="gpu">{head}<div class="spark-row"><span>load</span><span><b>{sum(last_minute) / len(last_minute):.0f}{NBSP}%</b>'
            f'</span></div>{sparkline(points, 100, 60, label)}'
            f'<p class="spark-caption">1-min mean · graph: {span_text(points)}, 0–100{NBSP}%</p>{memory}</div>')


def running_dir(card: dict) -> dict:
    """The live experiment directory of a running card (a log entry can cite several directories)."""
    return next(d for d in card["dirs"] if d["status"] == "running")


def render_hero(content: dict, cards: list[dict], entries: list[dict]) -> str:
    hero = content.get("hero", {})
    running = [c for c in cards if c["status"] == "running"]
    live = max((d for c in running for d in c["dirs"] if d["status"] == "running"), key=lambda d: d["name"], default=None)
    inner = pill_inner(live["title"], live["progress"], live["name"]) if live else pill_inner(None, None)
    pill = f'<a class="live-pill{" running" if live else ""}" href="#now" data-live="pill">{inner}</a>'
    facts = "".join(f"<li>{esc(f)}</li>" for f in hero.get("facts", []))
    video = ""
    if hero.get("video"):
        video = (f'<figure class="hero-media"><div class="hero-frame">'
                 f'<video class="hero-video" src="/{esc(hero["video"])}" muted loop playsinline preload="auto" '
                 f'aria-label="{esc(hero.get("caption", ""))}"></video>'
                 f'<button type="button" class="hero-expand" data-play="/{esc(hero["video"])}" data-caption="{esc(hero.get("caption", ""))}">'
                 f'Watch full screen</button></div>'
                 f'<figcaption>{esc(hero.get("caption", ""))}</figcaption></figure>')
    tiles = [
        (t.get("href", "#"), t["label"], t["value"], t.get("detail", "")) for t in content.get("tile", [])
    ]
    pre = sum(e["preregistered"] for e in entries)
    tiles.append(("#experiments", "Experiments in the log", str(len(entries)),
                  f"{pre} pre-registered with a written decision rule"
                  + (f", {len(running)} running now" if running else "") + ". Generated from the experiment log."))
    kpis = "".join(
        f'<a class="kpi" href="{esc(href)}"><span class="kpi-label">{esc(label)}</span>'
        f'<span class="kpi-value">{esc(value)}</span><span class="kpi-detail">{esc(detail)}</span></a>'
        for href, label, value, detail in tiles
    )
    return f"""
<section id="overview" class="hero">
  <div class="hero-grid">
    <div class="hero-text">
      <p class="eyebrow">{esc(hero.get("eyebrow", ""))}</p>
      <h1>{esc(hero.get("title", "UAV Predator-Prey"))}</h1>
      <div class="pitch">{md(hero.get("pitch", ""))}</div>
      {pill}
      <ul class="facts">{facts}</ul>
    </div>
    {video}
  </div>
  <div class="kpis">{kpis}</div>
</section>"""


def render_now(content: dict, cards: list[dict], entries: list[dict], trainers: list[dict], gpu: dict | None, git: dict) -> str:
    running = [c for c in cards if c["status"] == "running"]
    if running:
        parts = []
        for c in running:
            d = running_dir(c)
            arm = f'<p class="muted small">run <code>{esc(d["progress"]["arm"])}</code></p>' if d["progress"] else ""
            parts.append(f'<div class="now-run"><a class="now-title" href="#{esc(c["id"])}">{esc(c["title"])}</a>'
                         f'{progress_html(d["name"], d["progress"])}{arm}</div>')
        run_html = "".join(parts)
    else:
        run_html = '<p class="now-title">Nothing is training</p><p class="muted small">No experiment directory was written to in the last 15 minutes.</p>'
    if trainers:
        rows = "".join(
            f'<li><strong>{esc(t["script"])}</strong> · {esc(t["num_envs"] or "?")} envs · seed {esc(t["seed"] or "?")} · '
            f'{t["elapsed_min"]} min<br><code>{esc(t["task"] or "")}</code>'
            + (f'<br><code>{esc(" ".join(t["overrides"]))}</code>' if t["overrides"] else "") + "</li>"
            for t in trainers
        )
        run_html += f'<details class="procs"><summary>{len(trainers)} process(es) on the GPU</summary><ul>{rows}</ul></details>'

    latest = entries[-1] if entries else None
    latest_html = '<p class="muted">The experiment log is empty.</p>'
    if latest:
        card_id = "log-" + slug(f'{latest["date"]} {latest["title"]}')
        rows = summary_rows(latest["fields"])
        text = next((t for label, t in rows if label in ("Decision", "Decisions", "Result", "Reading", "Fix")),
                    rows[0][1] if rows else "")
        latest_html = (f'<p class="muted small">{esc(latest["date_label"])}</p>'
                       f'<a class="now-title" href="#{esc(card_id)}">{esc(latest["title"])}</a>'
                       f'<p class="clamp-4">{md_inline(text[:1].upper() + text[1:], EXPERIMENT_LOG.parent)}</p>')

    refs = "".join(
        f'<li><span class="ref-name">{esc(r["name"])}</span> <span class="muted small">{esc(r.get("date", ""))}</span>'
        f'<span class="ref-scope">{esc(r.get("scope", ""))}</span><span class="small">{esc(r.get("detail", ""))}</span></li>'
        for r in content.get("reference", [])
    )

    machine = [f'<div data-live="gpu">{gpu_html(gpu)}</div>']
    ahead = git["ahead"]
    push = badge("partial", f"{ahead} commits not pushed") if ahead and ahead != "0" else badge("done", "pushed") if ahead == "0" else ""
    machine.append(
        f'<div class="git"><p class="branch"><code>{esc(git["branch"])}</code></p><p>{push}</p>'
        f'<p class="small clamp-2"><code>{esc(git["head"])}</code> {esc(git["subject"])}</p><p class="muted small">{esc(git["when"])}'
        f'{" · uncommitted changes in tracked files" if git["dirty"] else ""}</p></div>')

    tiles = [
        ("Training now", "accent", run_html),
        ("Latest log entry", "", latest_html),
        ("Reference checkpoints", "", f'<ul class="refs">{refs}</ul>'),
        ("Machine and code", "", "".join(machine)),
    ]
    body = "".join(f'<article class="card now-card {tone}"><h3 class="card-label">{esc(label)}</h3>{inner}</article>'
                   for label, tone, inner in tiles)
    lead = ("<p>What the machine is doing right now and where the project stands.</p>"
            f'<p class="live-note"><span class="pulse" aria-hidden="true"></span><span data-live="updated">Updated '
            f'{time.strftime("%H:%M:%S")}</span><span>· refreshes every {POLL_SECONDS} s while this page is open</span></p>')
    return f'<section id="now">{section_head("Live status", "Now", lead)}<div class="now-grid">{body}</div></section>'


def stage_item(s: dict, dir_card: dict[str, str]) -> str:
    verdict = s.get("verdict", "open")
    tone, icon, _ = VERDICTS.get(verdict, ("muted", ICONS["ring"], verdict))
    thumbs = "".join(thumb("/" + v["path"], v.get("label", "")) for v in s.get("video", []))
    exp_link = ""
    if s.get("experiment") and s["experiment"] in dir_card:
        exp_link = f'<a class="more-link" href="#{esc(dir_card[s["experiment"]])}">Experiment card ↓</a>'
    details = f"""
<details class="more">
  <summary>Goal, result and why</summary>
  <div class="stage-grid">
    <div><h5>Goal</h5>{md(s.get("goal", ""))}</div>
    <div><h5>Result</h5>{md(s.get("result", ""))}</div>
    <div><h5>Why</h5>{md(s.get("why", ""))}</div>
  </div>
  {f'<h5>Evidence</h5>{evidence_list(s["evidence"])}' if s.get("evidence") else ""}
  {exp_link}
</details>"""
    metric = f'<span class="metric">{esc(s["metric"])}</span>' if s.get("metric") else ""
    return f"""
<li class="stage {tone}" id="stage-{esc(s.get("id", slug(s["title"])))}">
  <span class="stage-dot" aria-hidden="true">{icon}</span>
  <article class="stage-card{" has-thumbs" if thumbs else ""}{" multi" if len(s.get("video", [])) > 1 else ""}">
    <div class="stage-main">
      <div class="stage-meta"><span class="date">{esc(s.get("dates", ""))}</span>{badge(verdict)}</div>
      <h4 class="stage-title">{esc(s["title"])}</h4>
      <p class="takeaway">{md_inline(s.get("takeaway", ""))}</p>
      {metric}
      {details}
    </div>
    {f'<div class="stage-thumbs">{thumbs}</div>' if thumbs else ""}
  </article>
</li>"""


def render_journey(content: dict, cards: list[dict], dir_card: dict[str, str]) -> str:
    chapters = content.get("chapter", [])
    blocks = []
    for n, ch in enumerate(chapters, 1):
        items = [stage_item(s, dir_card) for s in ch.get("stage", [])]
        if n == len(chapters):
            # Live runs extend the story automatically until a curated stage replaces them.
            for c in (c for c in cards if c["status"] == "running"):
                question = next((v for k, v in c["rows"] if k == "Question"), "")
                d = running_dir(c)
                items.append(f"""
<li class="stage accent" id="stage-live-{esc(c["id"])}">
  <span class="stage-dot live" aria-hidden="true">{ICONS["dot"]}</span>
  <article class="stage-card">
    <div class="stage-main">
      <div class="stage-meta"><span class="date">{esc(c["date_label"])} · in progress</span>{badge("running")}</div>
      <h4 class="stage-title">{esc(c["title"])}</h4>
      <p class="takeaway">{md_inline(question, EXPERIMENTS)}</p>
      {progress_html(d["name"], d["progress"], compact=True)}
      <a class="more-link" href="#{esc(c["id"])}">Experiment card ↓</a>
    </div>
  </article>
</li>""")
        blocks.append(f"""
<div class="chapter">
  <header class="chapter-head"><span class="chapter-num">{n}</span>
    <div><h3>{esc(ch["title"])}</h3><p class="muted">{esc(ch.get("dates", ""))}</p><p class="chapter-intro">{esc(ch.get("intro", ""))}</p></div>
  </header>
  <ol class="timeline">{"".join(items)}</ol>
</div>""")
    lead = ("<p>Each stage asked one question. The badge says whether it was answered the way we hoped; "
            "open a stage for the goal, the measured result and the mechanism behind it.</p>")
    return f'<section id="journey">{section_head("How it developed", "Journey", lead)}{"".join(blocks)}</section>'


def render_results(content: dict) -> tuple[str, dict]:
    figures, data = [], {}
    for spec in content.get("chart", []):
        chart = featured_league(spec) if spec.get("kind") == "league" else featured_lines(spec)
        cid = spec.get("id", slug(spec.get("title", "chart")))
        wide = " wide" if spec.get("kind") == "league" else ""
        if chart is None:
            body = '<p class="muted">Data not found.</p>'
        else:
            data[cid] = chart
            body = f'<div class="chart-body" data-featured="{esc(cid)}"></div>'
        figures.append(f"""
<figure class="card chart-card{wide}" id="chart-{esc(cid)}">
  <figcaption class="chart-head"><h3>{esc(spec.get("title", ""))}</h3><p>{esc(spec.get("subtitle", ""))}</p></figcaption>
  {body}
  <p class="chart-caption">{esc(spec.get("caption", ""))}</p>
</figure>""")
    lead = ("<p>Three charts carry the current story. Every point is a deterministic evaluation (1024 episodes, seed 42) "
            "read from the experiments' eval files when the page is built.</p>")
    return f'<section id="results">{section_head("Key results", "Results", lead)}<div class="charts-grid">{"".join(figures)}</div></section>', data


def render_experiments(cards: list[dict]) -> tuple[str, dict]:
    articles, explorers = [], {}
    for c in cards:
        tags = [badge(c["status"])]
        if c["preregistered"]:
            tags.append('<span class="tag">pre-registered</span>')
        n_videos = sum(len(d["videos"]) for d in c["dirs"])
        if n_videos:
            tags.append(f'<span class="tag">{n_videos} video{"s" if n_videos > 1 else ""}</span>')
        if any(d["explorer"] for d in c["dirs"]):
            tags.append('<span class="tag">eval curves</span>')
        # Log bullets read "Result: the drop came …"; start the shown sentence with a capital letter.
        rows = "".join(
            (f'<div class="qrd-row"><dt>{esc(label)}</dt>' if label else '<div class="qrd-row plain">')
            + f'<dd class="clamp-3">{md_inline(text[:1].upper() + text[1:], EXPERIMENT_LOG.parent)}</dd></div>'
            for label, text in c["rows"]
        )
        progress = ""
        if c["status"] == "running":
            live_dir = running_dir(c)
            progress = f'<div class="exp-progress">{progress_html(live_dir["name"], live_dir["progress"])}</div>'
        body = []
        if c["body"]:
            body.append(f'<div class="doc">{markdown_to_html(c["body"], EXPERIMENT_LOG.parent, 2)}</div>')
        for d in c["dirs"]:
            sub = []
            if d["explorer"]:
                explorers[d["name"]] = d["explorer"]
                sub.append(f'<div class="card chart-card explorer" data-explorer="{esc(d["name"])}"></div>')
            if d["videos"]:
                sub.append('<div class="thumb-grid">' + "".join(thumb(v["url"], v["label"]) for v in d["videos"]) + "</div>")
            docs = []
            if d["summary"]:
                docs.append(f'<details class="doc-toggle"><summary>summary.txt</summary><pre class="plain">{esc(d["summary"])}</pre></details>')
            for label, text in (("RESULTS.md", d["results"]), ("PLAN.md, written before training", d["plan"])):
                if text:
                    docs.append(f'<details class="doc-toggle"><summary>{label}</summary><div class="doc">{markdown_to_html(text, d["dir"])}</div></details>')
            if d["statuses"]:
                rows_html = "".join(f"<tr><td><code>{esc(n)}</code></td><td><code>{esc(s)}</code></td></tr>" for n, s in d["statuses"])
                docs.append(f'<details class="doc-toggle"><summary>Run status files</summary><div class="table-wrap"><table>{rows_html}</table></div></details>')
            body.append(f'<div class="exp-dir"><p class="dir-name"><a href="/isaac/experiments/{esc(d["name"])}/">{esc(d["name"])}</a> '
                        f'<span class="muted small">browse files</span></p>{"".join(sub)}{"".join(docs)}</div>')
        more = (f'<details class="exp-more"{" open" if c["status"] == "running" else ""}><summary>Full entry, curves, videos and documents</summary>'
                f'<div class="exp-body">{"".join(body)}</div></details>') if body else ""
        articles.append(f"""
<article class="card exp" id="{esc(c["id"])}">
  <header class="exp-head">
    <span class="exp-date">{esc(c["date_label"])}</span>
    <h3 class="exp-title">{esc(c["title"])}</h3>
    <div class="tags">{"".join(tags)}</div>
  </header>
  {progress}
  <dl class="qrd">{rows}</dl>
  {more}
</article>""")
    lead = ("<p>Generated from <code>docs/experiment_log.md</code> and <code>artifacts/isaac/experiments/</code> on every page load, "
            "newest first. Each card shows the question, the result and the decision; open it for the full entry, "
            "interactive eval curves, videos and the PLAN/RESULTS documents.</p>")
    return f'<section id="experiments">{section_head("Lab notebook", "Experiments", lead)}<div class="exp-list">{"".join(articles)}</div></section>', explorers


def render_replay(content: dict) -> tuple[str, list[dict]]:
    """The 3D replay section; the page loads replay.js and the episode files only when it scrolls near."""
    remove_old_replays()
    sources = []
    for spec in content.get("replay", []):
        try:
            if source := ensure_replay(spec):
                sources.append(source)
        except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as exc:
            print(f"replay {spec.get('id')}: {exc!r}", flush=True)
    if not sources:
        return "", []
    lead = ("<p>Evaluated episodes in 3D, from the positions the deterministic evaluations record at 50 Hz (not a "
            "rendered video). Drag to rotate, scroll or pinch to zoom; each drone draws a fading trail of its last two "
            "seconds. Pick a category and step through its episodes with ◀ ▶ or Random. ★ keeps an episode in your "
            "starred list (stored in this browser), and the address bar always links to the episode shown.</p>")
    return (f'<section id="replay">{section_head("Watch in 3D", "Replay", lead)}'
            f'<div class="card replay" data-replay><div class="replay-msg">Loading the 3D view…</div></div></section>'), sources


def render_lessons(content: dict) -> str:
    def column(verdict: str, title: str) -> str:
        items = [l for l in content.get("lesson", []) if l.get("verdict") == verdict]
        tone, icon, _ = VERDICTS[verdict]
        lis = "".join(
            f'<li class="lesson"><p class="what">{md_inline(l["what"])}</p><p class="why">{md_inline(l["why"])}</p>'
            f'{evidence_list(l.get("evidence", []))}</li>'
            for l in items
        )
        return (f'<div class="card lesson-col {tone}"><h3><span class="col-icon" aria-hidden="true">{icon}</span>{title}'
                f'<span class="count">{len(items)}</span></h3><ul>{lis}</ul></div>')

    # Two balanced columns: what worked, and next to it what only partly worked above what did not.
    columns = column("worked", "What worked") + f'<div class="lesson-stack">{column("partial", "Partly")}{column("failed", "What did not work")}</div>'
    lead = "<p>The transferable part: what each attempt taught about training competitive multi-agent policies, with the mechanism and the evidence.</p>"
    return f'<section id="lessons">{section_head("What I learned", "Lessons", lead)}<div class="lesson-cols">{columns}</div></section>'


def render_recipe(content: dict) -> str:
    steps = content.get("recipe", [])

    def node(i: int, s: dict) -> str:
        tone, icon, label = VERDICTS.get(s.get("status", ""), ("muted", ICONS["ring"], ""))
        return (f'<li><a class="node {tone}" href="#step-{i}" title="{esc(label)}"><span class="node-num">{i}</span>'
                f'<span class="node-name">{esc(s.get("short", s["title"]))}</span><span class="node-icon" aria-hidden="true">{icon}</span></a></li>')

    track_b = "".join(node(i, s) for i, s in enumerate(steps, 1) if i <= 3)
    track_a = "".join(node(i, s) for i, s in enumerate(steps, 1) if i > 3)
    pipeline = f"""
<div class="card pipeline">
  <div class="track"><p class="track-label">Track B · from random init</p><ol class="nodes">{track_b}<li class="node-end gap">anchor v3 from scratch?</li></ol></div>
  <div class="track"><p class="track-label">Track A · from the Windows anchor, exactly reproducible</p><ol class="nodes">{track_a}<li class="node-end good">obstacle predator</li></ol></div>
  <p class="legend-row">{badge("verified")} {badge("reconstructed")} {badge("gap")}</p>
</div>"""
    items = []
    for i, s in enumerate(steps, 1):
        cmd = s.get("command", "").strip()
        cmd_html = (f'<div class="cmd"><button class="copy" type="button">Copy</button><pre><code>{esc(cmd)}</code></pre></div>'
                    if cmd else "")
        items.append(f"""
<li class="recipe-step" id="step-{i}">
  <details>
    <summary><span class="step-title">{esc(s["title"])}</span>{badge(s.get("status", "open"))}</summary>
    <dl>
      {f'<dt>Code</dt><dd><code>{esc(s["commit"])}</code></dd>' if s.get("commit") else ""}
      <dt>Starts from</dt><dd>{md(s.get("from", ""))}</dd>
      <dt>Settings</dt><dd>{md(s.get("settings", ""))}</dd>
      <dt>Move on when</dt><dd>{md(s.get("gate", ""))}</dd>
    </dl>
    {cmd_html}
    {md(s.get("notes", ""))}
  </details>
</li>""")
    return (f'<section id="reproduce">{section_head("From zero to the current references", "Reproduce", md(content.get("recipe_intro", "")))}'
            f'{pipeline}<ol class="recipe">{"".join(items)}</ol></section>')


def render_next(content: dict) -> str:
    cards = "".join(
        f'<article class="card idea"><h3>{esc(n["title"])}</h3><p>{md_inline(n["why"])}</p>'
        f'<p class="cost">{esc(n.get("cost", ""))}</p></article>'
        for n in content.get("next", [])
    )
    lead = "<p>Open questions and planned work, roughly in order of priority.</p>"
    return f'<section id="next">{section_head("What comes next", "Next steps", lead)}<div class="ideas">{cards}</div></section>'


def render_videos(content: dict, videos: list[dict]) -> str:
    featured = []
    for f in content.get("featured", []):
        if not (ARTIFACTS / f["path"]).is_file():
            continue
        featured.append(f'<figure class="feature">{thumb("/" + f["path"], f["title"], show_label=False)}'
                        f'<figcaption><strong>{esc(f["title"])}</strong><span>{esc(f.get("note", ""))}</span></figcaption></figure>')
    groups: dict[str, list[dict]] = {}
    for v in videos:
        groups.setdefault(v["group"], []).append(v)
    blocks = []
    for group, items in sorted(groups.items(), key=lambda kv: -max(v["mtime"] for v in kv[1])):
        name = group.rsplit("/", 1)[-1]
        kind = group.split("/")[1] if group.count("/") >= 1 else ""
        blocks.append(
            f'<details class="gallery-group"><summary><span class="group-name">{esc(name)}</span>'
            f'<span class="muted small">{esc(kind)} · {len(items)} video{"s" if len(items) > 1 else ""}</span></summary>'
            f'<div class="thumb-grid">{"".join(thumb(v["url"], v["label"]) for v in items)}</div></details>')
    lead = f"<p>Highlights first. Below are all {len(videos)} recordings under <code>artifacts/isaac/</code>, newest group first; videos load only when a group is opened.</p>"
    return (f'<section id="videos">{section_head("Watch", "Videos", lead)}<div class="feature-grid">{"".join(featured)}</div>'
            f'<h3 class="sub-head">All recordings</h3>{"".join(blocks)}</section>')


def current_gpu() -> dict | None:
    """The sampler's latest reading when serving (no extra nvidia-smi call), else a fresh one."""
    return (SAMPLER.latest_gpu() if SAMPLER else None) or gpu_state()


def live_status() -> dict:
    """The parts of the page that change while a run trains; the open page polls this every POLL_SECONDS."""
    now = time.time()
    running = []
    for exp_dir in sorted((p for p in EXPERIMENTS.iterdir() if p.is_dir()), reverse=True):  # newest first
        live, progress = live_state(exp_dir, now)
        if live:
            title, _ = heading_title(read_text(exp_dir / "RESULTS.md"), read_text(exp_dir / "PLAN.md"), exp_dir.name)
            running.append((exp_dir.name, title, progress))
    entries = parse_log()
    return {
        "time": time.strftime("%H:%M:%S"),
        "running": [name for name, _, _ in running],
        "titles": {name: title for name, title, _ in running},
        "progress": {name: {"full": progress_inner(p, run=name), "compact": progress_inner(p, compact=True, run=name)}
                     for name, _, p in running},
        "pill": pill_inner(running[0][1], running[0][2], running[0][0]) if running else pill_inner(None, None),
        "gpu": gpu_html(current_gpu()),
        "latest": entries[-1]["title"] if entries else "",
    }


def load_content() -> dict:
    return tomllib.loads(read_text(CONTENT) or "")


def build() -> Path:
    content = load_content()
    entries = parse_log()
    dirs = load_experiment_dirs()
    cards, dir_card = build_cards(entries, dirs)
    results_html, featured = render_results(content)
    replay_html, replays = render_replay(content)
    experiments_html, explorers = render_experiments(cards)
    sync_static()
    sections = [
        render_hero(content, cards, entries),
        render_now(content, cards, entries, live_trainers(), current_gpu(), git_state()),
        render_journey(content, cards, dir_card),
        results_html,
        replay_html,
        render_lessons(content),
        experiments_html,
        render_recipe(content),
        render_next(content),
        render_videos(content, videos_under(ARTIFACTS / "isaac")),
    ]
    # What the page was built with, so it can tell when a run starts or ends or the log gets a new entry.
    live_dirs = sorted((d for d in dirs.values() if d["status"] == "running"), key=lambda d: d["name"], reverse=True)
    live = {"running": [d["name"] for d in live_dirs], "titles": {d["name"]: d["title"] for d in live_dirs},
            "latest": entries[-1]["title"] if entries else "", "poll_s": POLL_SECONDS}
    data_json = json.dumps({"featured": featured, "explorers": explorers, "live": live, "replays": replays},
                           separators=(",", ":")).replace("</", "<\\/")
    page = (TEMPLATE.read_text(encoding="utf-8")
            .replace("{{BUILT}}", time.strftime("%Y-%m-%d %H:%M"))
            .replace("{{SECTIONS}}", "\n".join(sections))
            .replace("{{DATA}}", data_json))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "index.html"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out)  # atomic, so a page load never sees a half-written file
    return out


# --------------------------------------------------------------------------------------------- server


REPLAY_EPISODE = re.compile(rf"^/dashboard/replay/(?P<id>[\w-]+)-v{REPLAY_FORMAT}/(?P<n>\d+)\.json$")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from artifacts/ with byte ranges; the dashboard page is rebuilt on every load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ARTIFACTS), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if m := REPLAY_EPISODE.match(path):
            try:
                ensure_episode(m["id"], int(m["n"]))
            except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as exc:
                self.log_message("replay episode failed: %r", exc)
        if path == "/dashboard/status.json":
            try:
                body, code = json.dumps(live_status()).encode(), 200
            except Exception as exc:  # the page keeps its last values and says the updates are paused
                self.log_message("status failed: %r", exc)
                body, code = b'{"error": "status failed"}', 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/dashboard", "/dashboard/", "/dashboard/index.html"):
            if path != "/dashboard/":
                self.send_response(302)
                self.send_header("Location", "/dashboard/")
                self.end_headers()
                return
            try:
                build()
            except Exception as exc:  # serve the last good page rather than nothing
                self.log_message("rebuild failed: %r", exc)
        super().do_GET()

    def send_head(self):
        range_header = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not range_header or not path.is_file():
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
        size = path.stat().st_size
        if not m or (not m.group(1) and not m.group(2)):
            self.send_error(416)
            return None
        if m.group(1):
            start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
        else:  # suffix range: the last N bytes
            start, end = max(0, size - int(m.group(2))), size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._range_left = None
        try:
            while left > 0 and (chunk := source.read(min(1 << 20, left))):
                outputfile.write(chunk)
                left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser cancels range requests all the time while seeking

    def log_request(self, code="-", size="-"):
        if not self.path.startswith("/dashboard/status.json"):  # polled every few seconds; keep the terminal readable
            super().log_request(code, size)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store" if self.path.startswith("/dashboard") else "max-age=3600")
        super().end_headers()


def serve(host: str, port: int) -> None:
    global SAMPLER
    SAMPLER = Sampler()
    SAMPLER.start()
    out = build()
    print(f"built {out}")
    print(f"serving {ARTIFACTS} at http://{host}:{port}/dashboard/  (Ctrl+C stops)")
    if host not in ("127.0.0.1", "localhost"):
        print("warning: reachable from other machines on this network")
    with http.server.ThreadingHTTPServer((host, port), Handler) as server:
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="write artifacts/dashboard/index.html once")
    serve_parser = sub.add_parser("serve", help="serve artifacts/ and rebuild the page on every load")
    serve_parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 makes it reachable from your LAN (phone)")
    serve_parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "build":
        print(build())
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
