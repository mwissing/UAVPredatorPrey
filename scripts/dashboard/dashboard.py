#!/usr/bin/env python3
"""Local project dashboard: the story, live status, experiments, results, videos, and the reproduction recipe.

    python3 scripts/dashboard/dashboard.py build   # writes $HOME/RL/artifacts/dashboard/index.html
    python3 scripts/dashboard/dashboard.py serve   # http://127.0.0.1:8765/dashboard/ (rebuilt on every page load)

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
import html
import http.server
import json
import math
import os
import re
import subprocess
import time
import tomllib
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
    hours, minutes = divmod(seconds // 60, 60)  # non-breaking spaces keep "1 h 47 min" on one line
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


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


def load_experiment_dirs() -> dict[str, dict]:
    dirs = {}
    now = time.time()
    for exp_dir in sorted(p for p in EXPERIMENTS.iterdir() if p.is_dir()):
        results, plan = read_text(exp_dir / "RESULTS.md"), read_text(exp_dir / "PLAN.md")
        heading = re.search(r"(?m)^# (.+)$", results or plan or "")
        title, pre = clean_title(heading.group(1)) if heading else (exp_dir.name, False)
        if plan and re.search(r"(?mi)^# .*pre-registered", plan):
            pre = True
        live = now - last_activity(exp_dir) < LIVE_WINDOW_S
        consoles = sorted(exp_dir.rglob("*.console.log"), key=lambda p: p.stat().st_mtime)
        progress = tqdm_progress(consoles[-1]) if live and consoles and now - consoles[-1].stat().st_mtime < LIVE_WINDOW_S else None
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


def progress_html(p: dict | None, compact: bool = False) -> str:
    if not p:
        return '<p class="muted small">evaluating or between phases (no training progress in the last 15 min)</p>'
    left = f' · ~{esc(p["left"])} left · ends ≈ {esc(p["eta"])}' if p["left"] else ""
    steps = "" if compact else f' · {p["done"]:,} / {p["total"]:,} env steps'
    return (f'<div class="meter" role="progressbar" aria-valuenow="{p["share"] * 100:.0f}" aria-valuemin="0" aria-valuemax="100">'
            f'<div style="width:{max(p["share"] * 100, 1.5):.1f}%"></div></div>'
            f'<p class="meter-text"><strong>{p["share"]:.0%}</strong>{steps}{left}</p>')


def render_hero(content: dict, cards: list[dict], entries: list[dict]) -> str:
    hero = content.get("hero", {})
    running = [c for c in cards if c["status"] == "running"]
    if running:
        p = running[0]["dirs"][0]["progress"] if running[0]["dirs"] else None
        detail = f' · {p["share"]:.0%}' + (f' · ~{esc(p["left"])} left' if p["left"] else "") if p else ""
        short = running[0]["title"].split(":")[0]  # "Prey ghost stage v5: slow ghosts …" → "Prey ghost stage v5"
        pill = (f'<a class="live-pill running" href="#now"><span class="pulse" aria-hidden="true"></span>'
                f'<span>Training now: <strong>{esc(short)}</strong>{detail}</span></a>')
    else:
        pill = '<a class="live-pill" href="#now"><span class="pulse idle" aria-hidden="true"></span><span>No training running</span></a>'
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
        run_html = "".join(
            f'<div class="now-run"><a class="now-title" href="#{esc(c["id"])}">{esc(c["title"])}</a>'
            f'{progress_html(c["dirs"][0]["progress"] if c["dirs"] else None)}'
            + (f'<p class="muted small">run <code>{esc(c["dirs"][0]["progress"]["arm"])}</code></p>'
               if c["dirs"] and c["dirs"][0]["progress"] else "")
            + "</div>"
            for c in running
        )
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

    machine = []
    if gpu:
        machine.append(
            f'<div class="gpu"><div class="gpu-head"><span>{esc(gpu["name"])}</span><span class="muted small">{gpu["temp"]} °C</span></div>'
            f'<div class="gauge"><span>load</span><div class="meter small"><div style="width:{gpu["util"]}%"></div></div><b>{gpu["util"]} %</b></div>'
            f'<div class="gauge"><span>memory</span><div class="meter small"><div style="width:{gpu["used_gb"] / gpu["total_gb"] * 100:.0f}%"></div></div>'
            f'<b>{gpu["used_gb"]:.1f} / {gpu["total_gb"]:.0f} GB</b></div></div>')
    else:
        machine.append('<p class="muted">GPU not visible (nvidia-smi unavailable).</p>')
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
    lead = "<p>What the machine is doing right now and where the project stands. Reload the page to refresh.</p>"
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
                p = c["dirs"][0]["progress"] if c["dirs"] else None
                items.append(f"""
<li class="stage accent" id="stage-live-{esc(c["id"])}">
  <span class="stage-dot live" aria-hidden="true">{ICONS["dot"]}</span>
  <article class="stage-card">
    <div class="stage-main">
      <div class="stage-meta"><span class="date">{esc(c["date_label"])} · in progress</span>{badge("running")}</div>
      <h4 class="stage-title">{esc(c["title"])}</h4>
      <p class="takeaway">{md_inline(question, EXPERIMENTS)}</p>
      {progress_html(p, compact=True)}
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
        if c["status"] == "running" and c["dirs"]:
            progress = f'<div class="exp-progress">{progress_html(c["dirs"][0]["progress"])}</div>'
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


def build() -> Path:
    content = tomllib.loads(read_text(CONTENT) or "")
    entries = parse_log()
    dirs = load_experiment_dirs()
    cards, dir_card = build_cards(entries, dirs)
    results_html, featured = render_results(content)
    experiments_html, explorers = render_experiments(cards)
    sections = [
        render_hero(content, cards, entries),
        render_now(content, cards, entries, live_trainers(), gpu_state(), git_state()),
        render_journey(content, cards, dir_card),
        results_html,
        render_lessons(content),
        experiments_html,
        render_recipe(content),
        render_next(content),
        render_videos(content, videos_under(ARTIFACTS / "isaac")),
    ]
    data_json = json.dumps({"featured": featured, "explorers": explorers}, separators=(",", ":")).replace("</", "<\\/")
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


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from artifacts/ with byte ranges; the dashboard page is rebuilt on every load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ARTIFACTS), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
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

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store" if self.path.startswith("/dashboard") else "max-age=3600")
        super().end_headers()


def serve(host: str, port: int) -> None:
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
