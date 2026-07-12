# Linux Container Architecture - Isaac Lab And JAX

## Status

The Isaac runtime described here is implemented and validated on Linux. The
separate JAX runtime remains the next architecture stage.

The immediate goal is to reproduce the current Windows Isaac Lab behavior on
Linux. A JAX simulator is a later, separate runtime that shares a stable
simulation and policy contract with the Isaac Lab project.

## Decisions

1. Keep the same sibling-repository layout used on Windows.
2. Keep repositories and experiment artifacts on the native Linux NVMe
   filesystem, not on the USB drive or a Windows filesystem mount.
3. Use Docker for runtime dependencies.
4. Use two containers rather than one combined Isaac Lab and JAX image.
5. Pin the existing Isaac stack during migration. Do not combine the operating
   system migration with an Isaac Sim or Isaac Lab upgrade.
6. Run only one GPU-heavy training workload at a time on the RTX 5090.

## Host Layout

```text
$HOME/RL/
  IsaacLab/                 # pinned upstream checkout
  UAVPredatorPrey/          # current Isaac Lab / PyTorch project
  UAVPredatorPreyJAX/       # future independent JAX repository
  artifacts/
    isaac/
      checkpoints/
      logs/
      evaluations/
    jax/
      checkpoints/
      logs/
      evaluations/
    transfer/               # neutral JAX <-> PyTorch exchange artifacts
```

Source repositories remain on the host and are bind-mounted into containers.
Large generated artifacts must not be committed to Git.

## Host Dependencies

Install only the common platform dependencies on the host:

- Ubuntu 24.04 LTS, or keep Ubuntu 22.04 LTS if it is already installed
- a current NVIDIA production driver compatible with the RTX 5090
- Docker Engine
- NVIDIA Container Toolkit
- Git, VS Code or Codex, and `tmux`

Do not install a second host-level CUDA, PyTorch, JAX, or Isaac Python stack
unless a specific native workflow later requires it. The containers own those
dependencies.

## Runtime A - Isaac Lab

Preserve the exported runtime first:

```text
Isaac Sim:       5.1.0
Isaac Lab:       2.3.2
Isaac Lab commit f4aa17f87e2e5db5484f0b5974918573e8918ce2
Python:          3.11
PyTorch:         2.7.0 with CUDA 12.8
skrl:            1.4.3
```

Use the official Docker machinery from the pinned `IsaacLab` checkout. Extend
it with a project-specific Compose override instead of replacing the upstream
Docker stack.

Implemented project files:

```text
UAVPredatorPrey/
  docker/
    isaaclab.override.yaml
    .env.example
  scripts/linux/
    verify_container_runtime.sh
```

The override should provide these mounts:

```text
$HOME/RL/UAVPredatorPrey -> /workspace/UAVPredatorPrey
$HOME/RL/artifacts       -> /workspace/artifacts
```

Use named Docker volumes for Isaac, Kit, shader, pip, and compute caches. Make
the ownership of bind-mounted logs explicit so the container does not leave
root-owned experiment files on the host.

### First-Time Isaac Image Build

Create the host configuration and artifact directories:

```bash
cd "$HOME/RL/UAVPredatorPrey"
cp docker/.env.example docker/.env
# Edit docker/.env for this host, especially paths, HOST_UID, and HOST_GID.

mkdir -p \
  "$HOME/RL/artifacts/isaac/logs" \
  "$HOME/RL/artifacts/isaac/checkpoints" \
  "$HOME/RL/artifacts/isaac/evaluations" \
  "$HOME/RL/artifacts/isaac/curriculum"
```

Build the pinned upstream base from the sibling Isaac Lab checkout:

```bash
cd "$HOME/RL/IsaacLab"
git checkout f4aa17f87e2e5db5484f0b5974918573e8918ce2
python3 docker/container.py build base --suffix uav51
```

Build the small project-derived image. The build context intentionally contains
only the extension package, not the external checkpoints:

```bash
cd "$HOME/RL/UAVPredatorPrey"

docker build \
  --file docker/Dockerfile.isaaclab \
  --tag uavpredatorprey-isaaclab:f4aa17f-sim5.1.0-skrl1.4.3 \
  --build-arg ISAACLAB_BASE_IMAGE=isaac-lab-base-uav51:latest \
  --build-arg HOST_UID="$(id -u)" \
  --build-arg HOST_GID="$(id -g)" \
  --build-arg ISAACLAB_COMMIT=f4aa17f87e2e5db5484f0b5974918573e8918ce2 \
  --build-arg SKRL_VERSION=1.4.3 \
  source/UAVPredatorPrey
```

The Compose override uses external volumes so Isaac caches survive container
removal. Create them once:

```bash
for suffix in \
  kit-cache ov-cache pip-cache gl-cache compute-cache \
  omniverse-logs kit-logs ov-data documents lab-docs lab-logs lab-data
do
  docker volume create "uavpredatorprey-isaacsim-5.1.0-${suffix}"
done
```

Verified start and enter commands:

```bash
cd "$HOME/RL/IsaacLab"

python3 docker/container.py start base \
  --suffix uav51 \
  --files "$HOME/RL/UAVPredatorPrey/docker/isaaclab.override.yaml" \
  --env-files "$HOME/RL/UAVPredatorPrey/docker/.env"

python3 docker/container.py enter base \
  --suffix uav51 \
  --files "$HOME/RL/UAVPredatorPrey/docker/isaaclab.override.yaml" \
  --env-files "$HOME/RL/UAVPredatorPrey/docker/.env"
```

The reproducible curriculum workflow is documented in
[`3v1_hysteresis_linux_runbook.md`](3v1_hysteresis_linux_runbook.md).

Training and deterministic evaluation should run headless. X11 forwarding may
remain enabled for interactive play, but batch processes must not inherit the
display variables. On the validated host, Kit crashed in `XOpenDisplay` when a
headless evaluator inherited the forwarded X11 environment. Launch every batch
Isaac process with:

```bash
env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p <script> <arguments>
```

This command-scoped isolation preserves X11 for later interactive sessions.
Headless simulation and offscreen MP4 recording were both validated without a
display server.

## Runtime B - JAX Simulator

Create `UAVPredatorPreyJAX` as a separate Git repository after Isaac parity is
confirmed. Its container should be small and independent of Omniverse:

```text
Python 3.11
JAX with the appropriate NVIDIA CUDA wheel
Optax
Flax or Equinox only if the selected trainer needs it
pytest
TensorBoard or another explicitly selected logger
```

Expected repository structure:

```text
UAVPredatorPreyJAX/
  docker/
    Dockerfile
    compose.yaml
  pyproject.toml
  uv.lock
  src/
  tests/
```

The first environment API should remain deliberately small:

```python
reset(key, cfg) -> state
step(state, action, cfg) -> next_state, reward, terminated, truncated, info
```

Use `jax.vmap` for batched environments and `jax.lax.scan` for rollouts. Do
not reproduce every Isaac or quadrotor detail before the strategic game logic
has been validated.

## Shared Transfer Contract

Before attempting policy transfer, write down and test the contract shared by
both simulators:

- observation field order, shape, normalization, and units
- action dimension, bounds, semantics, and control frequency
- reward formulas and termination versus truncation semantics
- arena geometry and spawn distributions
- recurrent-state initialization and reset behavior
- deterministic evaluation protocol and seed handling

### Current Isaac Action Contract

The current Isaac policy emits dimensionless normalized collective-thrust and
direct body-frame moment commands. Each UAV uses
`[a_T, a_tau_x, a_tau_y, a_tau_z]` with the executed-action contract
`[-1, 1]^4`; the shared predator agent stacks three commands into 12 dimensions
and the prey action is 4-dimensional. For the validated low-progress task:

```text
T     = r m g (a_T + 1) / 2
tau_b = 0.01 [a_tau_x, a_tau_y, a_tau_z] N m
r     = 1.9 for predators, 2.2 for prey
```

The environment applies collective thrust along body `+Z` and the moments
about the body axes at a 50 Hz action rate, holding each action over two 100 Hz
physics steps. There is no body-rate PID, motor mixer, motor/rotor dynamics, or
MPC in the current control path.

The integer action-dimension configuration makes Isaac Lab expose formally
unbounded Gym `Box(-inf, inf)` metadata. That metadata is not the executed
contract: the current GRU actor applies `tanh`, and the environment defensively
clamps every component to `[-1, 1]`. JAX must declare the actual bounded
contract explicitly.

Preserve these semantics in a first `direct_wrench_v0` JAX mode so the migrated
checkpoint remains meaningful. A later `accel_yaw_rate_v1` policy may emit a
desired acceleration plus yaw-rate reference, but that is a different policy
interface. A shared saturated geometric/PD controller in both JAX and Isaac
must map those references to the existing collective-thrust/body-moment wrench.
Existing direct-wrench checkpoints are not compatible with the new semantics.

The meaningful experiment is:

```text
JAX pretraining + Isaac Lab fine-tuning
versus
Isaac Lab training from scratch
```

Zero-shot equality between simulators is not required. Compare both routes
with matched seeds, environment steps, wall-clock time, and evaluation metrics.

## Migration And Validation Order

1. Verify `nvidia-smi` on the Linux host.
2. Verify GPU access from a minimal NVIDIA Docker container.
3. Clone and pin Isaac Lab to the recorded commit.
4. Build the unmodified upstream Isaac Lab container once.
5. Add the `UAVPredatorPrey` Compose override and restore the USB artifacts.
6. Run the repository unit tests in the container.
7. Run one visual episode and a deterministic 128-environment evaluation.
8. Run the full 512-environment parity evaluation from the migration handoff.
9. Run a 100-300 iteration training smoke test.
10. Freeze the shared simulator contract, then create the independent JAX
    repository and container.
11. Before a long Isaac league run, validate scheduler-integrated pool mixing,
    cross-play, and persistence. Do not run Isaac and JAX GPU jobs concurrently.

### Reusable Linux Validation

Run the non-destructive runtime verifier from the project container:

```bash
cd /workspace/UAVPredatorPrey
./scripts/linux/verify_container_runtime.sh
```

Run the complete CPU-side regression suite through the pinned Isaac Python:

```bash
/workspace/isaaclab/isaaclab.sh -p -m pytest tests -q
```

The baseline certification tool runs the current checkpoint over seeds
`42 43 44 45 46`, then evaluates the current predator and prey against every
opposite-role entry in the portable pool:

```bash
env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/certify_baseline.py
```

It executes GPU jobs sequentially, rejects incomplete evaluator output, keeps
composed cross-play checkpoints only in a temporary directory, and writes the
individual evaluator JSON files plus aggregate JSON, CSV, Markdown, status, and
provenance reports under:

```text
/workspace/artifacts/isaac/evaluations/<timestamp>_baseline_certification/
```

The seed-42 512-episode anchor is compared with the Windows metrics in the
handoff using explicit behavioral-shift tolerances. `REPORT.md` and the JSON
contain a `pass` or `fail` verdict; a failed parity verdict returns a non-zero
exit status. Real certification requires a clean Git tree so the commit fully
identifies the evaluated code. `--allow-dirty` is available for diagnostic runs
that are labeled `diagnostic_*` and must not be treated as the reproducible
baseline. A real run without the seed-42 512-episode anchor is reported as
`not_comparable` and returns a non-zero status; `--skip-anchor` is therefore
intended only for focused cross-play diagnostics.

Use `--dry-run` to inspect every command without launching Isaac Sim. The
certification tool and curriculum scheduler resolve `isaaclab.sh` automatically
from `$ISAACLAB_ROOT`, the sibling checkout, or `/workspace/isaaclab`;
`--isaaclab` remains available as an explicit override.

## Acceptance Criteria For Isaac Migration

- the repository test suite passes
- the migrated recurrent checkpoint loads without missing or unexpected keys
- pool entries load from Linux-relative paths
- GRU state updates and episode resets behave correctly
- deterministic evaluation has no large unexplained OOB or catch-rate shift
- a short training run writes checkpoints, TensorBoard events, and videos to a
  host-accessible artifact directory
- stopping and restarting the container does not destroy caches or artifacts

## Avoid

- one combined Isaac Lab and JAX image
- copying the Windows virtual environment
- using the USB drive as the active training filesystem
- upgrading to a newer Isaac Sim during the migration parity check
- embedding Windows or machine-specific absolute paths in new pool files
- mounting all of `$HOME/RL` when only selected repositories and artifacts are
  needed
- running Isaac and JAX training concurrently on the same GPU

## Codex Resume Prompt On Linux

Use this prompt in a new Codex task after cloning the repository:

```text
I am continuing the UAVPredatorPrey Linux migration from the Windows machine.

First read AGENTS.md, docs/linux_handoff_2026-07-11.md, and
docs/linux_docker_jax_setup.md. Then inspect the current Git branch, commit,
working tree, the pinned IsaacLab checkout, and the restored migration artifact
bundle. Do not upgrade Isaac Sim, Isaac Lab, Python, PyTorch, skrl, the model
architecture, rewards, or task behavior during the initial migration.

Our target is two separate GPU containers: one pinned Isaac Lab 2.3.2 / Isaac
Sim 5.1.0 runtime for this repository, and later one small independent JAX
runtime. Start only with the Isaac container. Reuse the official Docker setup
from the pinned IsaacLab checkout and add the smallest project-specific Compose
override needed to mount UAVPredatorPrey and host-owned artifacts. Pay explicit
attention to host UID/GID, cache persistence, checkpoint paths, and headless
rendering.

Before editing, report what is already present, whether the recorded versions
match, which files you propose to add, and the exact parity-validation sequence.
Do not start a long training run until tests, checkpoint loading, pool loading,
and deterministic evaluation parity have passed.
```
