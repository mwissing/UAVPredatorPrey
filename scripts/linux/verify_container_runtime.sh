#!/usr/bin/env bash
set -euo pipefail

repo_root="${UAVPREDATORPREY_ROOT:-/workspace/UAVPredatorPrey}"
isaaclab_root="${ISAACLAB_ROOT:-/workspace/isaaclab}"
bundle_rel=".pretrained_checkpoints/linux_migration_2026-07-11"
bundle_root="${repo_root}/${bundle_rel}"

# Isaac Lab's official launcher calls `tabs`, which rejects an unset or dumb
# terminal even though this verification is otherwise fully non-interactive.
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
    export TERM=xterm-256color
fi

fail() {
    echo "[FAIL] $*" >&2
    exit 1
}

pass() {
    echo "[ OK ] $*"
}

[[ "$(pwd -P)" == "${repo_root}" ]] || fail "run from ${repo_root}; current directory is $(pwd -P)"
[[ "$(id -u)" == "${HOST_UID:-1000}" ]] || fail "container UID is $(id -u), expected ${HOST_UID:-1000}"
[[ "$(id -g)" == "${HOST_GID:-1000}" ]] || fail "container GID is $(id -g), expected ${HOST_GID:-1000}"
[[ -r "${repo_root}/AGENTS.md" ]] || fail "project bind mount is missing"
[[ -w "${repo_root}" ]] || fail "project bind mount is not writable by the runtime user"
[[ -d /workspace/artifacts/isaac/logs && -w /workspace/artifacts/isaac/logs ]] \
    || fail "host artifact log directory is missing or not writable"
[[ -x "${isaaclab_root}/isaaclab.sh" ]] || fail "Isaac Lab launcher is missing"
[[ -f "${bundle_root}/SHA256SUMS.txt" ]] || fail "migration bundle is missing"
pass "bind mounts and UID/GID"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
pass "NVIDIA runtime access"

(
    cd "${bundle_root}"
    sha256sum -c SHA256SUMS.txt
)
pass "migration bundle checksums"

"${isaaclab_root}/isaaclab.sh" -p - "${repo_root}" "${isaaclab_root}" "${bundle_rel}" <<'PY'
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import sys

import skrl
import torch

repo = Path(sys.argv[1]).resolve()
isaaclab = Path(sys.argv[2]).resolve()
bundle_rel = Path(sys.argv[3])
bundle = repo / bundle_rel

manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8-sig"))
runtime = manifest["runtime"]

assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__ == runtime["torch_version"], (torch.__version__, runtime["torch_version"])
assert torch.version.cuda == "12.8", torch.version.cuda
assert skrl.__version__ == runtime["skrl_version"] == "1.4.3", skrl.__version__
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
gpu_name = torch.cuda.get_device_name(0)
assert "RTX 5090" in gpu_name, gpu_name

lab_version = (isaaclab / "VERSION").read_text(encoding="utf-8").strip()
assert lab_version == runtime["isaac_lab_version"] == "2.3.2", lab_version
assert os.environ.get("ISAACLAB_COMMIT") == runtime["isaac_lab_commit"], os.environ.get("ISAACLAB_COMMIT")

sim_release = (Path(os.environ["ISAACSIM_ROOT_PATH"]) / "VERSION").read_text(encoding="utf-8").strip()
sim_version = sim_release.split("-", maxsplit=1)[0]
assert sim_version == runtime["isaac_sim_version"] == "5.1.0", sim_version

dist_version = importlib.metadata.version("UAVPredatorPrey")

current_rel = Path(manifest["current_checkpoint"]["linux_repo_path"])
current = repo / current_rel
assert current.is_file(), current
current_state = torch.load(current, map_location="cpu", weights_only=False)
for role in ("predator", "prey"):
    assert role in current_state, (current, role)
    assert "policy" in current_state[role], (current, role, current_state[role].keys())
del current_state

pool_path = repo / Path(manifest["opponent_pool"]["linux_repo_path"])
pool = json.loads(pool_path.read_text(encoding="utf-8-sig"))
assert len(pool.get("predator", [])) == 4
assert len(pool.get("prey", [])) == 4

active: list[tuple[str, Path]] = []
for role in ("predator", "prey"):
    for entry in pool[role]:
        raw = str(entry["checkpoint"])
        posix = PurePosixPath(raw)
        assert not posix.is_absolute(), raw
        assert "\\" not in raw, raw
        assert float(entry.get("weight", 0.0)) > 0.0, entry
        path = repo / Path(raw)
        assert path.is_file(), path
        active.append((role, path.resolve()))

assert len({path for _, path in active}) == 8, active
expected = {
    (repo / Path(entry["linux_repo_path"])).resolve()
    for entry in manifest["opponent_pool"]["entries"]
}
assert {path for _, path in active} == expected

checkpoint_dir = bundle / "pool" / "checkpoints"
assert {path.resolve() for path in checkpoint_dir.glob("*.pt")} == expected

for role, path in active:
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert role in state, (path, role)
    assert "policy" in state[role], (path, role, state[role].keys())
    del state

print(f"Python:       {sys.version.split()[0]}")
print(f"Isaac Sim:    {sim_version}")
print(f"Isaac Lab:    {lab_version} ({os.environ['ISAACLAB_COMMIT']})")
print(f"PyTorch:      {torch.__version__} / CUDA {torch.version.cuda}")
print(f"skrl:         {skrl.__version__}")
print(f"GPU:          {gpu_name}")
print(f"Project dist: UAVPredatorPrey {dist_version}")
print(f"Checkpoints:  current + {len(active)} pool policies loaded on CPU")
PY

pass "versions, project distribution, checkpoint, and pool loading"
