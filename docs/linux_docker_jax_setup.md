# Linux Container Architecture - Isaac Lab And JAX

## Status

This document records the intended Linux setup. It is an implementation brief,
not a claim that the Docker files already exist in `UAVPredatorPrey`.

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

Expected project files after implementation:

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

Conceptual start command:

```bash
cd "$HOME/RL/IsaacLab"

python3 docker/container.py start base \
  --files "$HOME/RL/UAVPredatorPrey/docker/isaaclab.override.yaml" \
  --env-files "$HOME/RL/UAVPredatorPrey/docker/.env"
```

The exact command must be verified against the pinned `container.py` before it
is added to a runbook.

Training and deterministic evaluation should run headless. Interactive play
can add an X11 or Wayland display bridge later; it is not required for the
first parity test or for headless video recording.

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

The current Isaac policy outputs body-rate and thrust commands. Keep the action
meaning stable if direct transfer is attempted. If the first JAX simulator uses
a higher-level action such as acceleration or velocity, treat it as a different
policy interface and add an explicit low-level controller rather than silently
changing action semantics.

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
10. Only after parity succeeds, start a long curriculum run.
11. Create the JAX repository and its independent container afterward.

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
