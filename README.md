# UAVPredatorPrey

Multi-agent reinforcement learning for three predator quadrotors pursuing one
prey quadrotor in Isaac Lab. The current training stack uses MAPPO through
`skrl`, recurrent attention policies, a centralized critic, deterministic
evaluation, opponent pools, and an outer-loop hysteresis curriculum.

The supported Linux workflow is Docker-first. Do not install a second native
Isaac Sim, PyTorch, or CUDA environment for this repository.

## Current Linux Stack

| Component | Pinned version |
| --- | --- |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2, commit `f4aa17f87e2e5db5484f0b5974918573e8918ce2` |
| Python | 3.11 |
| PyTorch | 2.7.0 with CUDA 12.8 |
| skrl | 1.4.3 |

The current per-UAV policy action is a normalized direct wrench:
`[collective thrust, body moment x, body moment y, body moment z]`. It is not a
body-rate command, and the environment has no inner PID, motor mixer, or MPC.
The shared predator action stacks three such commands; the prey uses one.

The migrated Windows anchor and eight active opponent-pool checkpoints are
external artifacts under:

```text
.pretrained_checkpoints/linux_migration_2026-07-11/
```

They are intentionally not stored in Git. See
[`docs/linux_handoff_2026-07-11.md`](docs/linux_handoff_2026-07-11.md) for the
bundle layout, checksums, and Windows reference metrics.

## Repository Layout

```text
docker/                     Pinned project image and Compose override
scripts/linux/              Linux runtime verification
scripts/skrl/               Training, evaluation, and curriculum entry points
source/UAVPredatorPrey/     Isaac Lab extension and environments
tests/                      CPU-side regression tests
docs/                       Architecture, migration, and experiment runbooks
```

Large logs, checkpoints, videos, and evaluation reports are written to the
host-owned `$HOME/RL/artifacts/isaac/` tree through Docker bind mounts.

## Start The Isaac Container

The expected host layout is:

```text
$HOME/RL/IsaacLab
$HOME/RL/UAVPredatorPrey
$HOME/RL/artifacts
```

Copy `docker/.env.example` to `docker/.env` and adjust the host paths and
UID/GID before the first start. The upstream Isaac Lab checkout must be pinned
to the commit shown above, and the project image must already be built as
described in
[`docs/linux_docker_jax_setup.md`](docs/linux_docker_jax_setup.md).

From the host:

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

The shell prompt should open in `/workspace/UAVPredatorPrey`. Older running
containers may inherit `TERM=dumb`; fix that session before calling the official
Isaac launcher:

```bash
export TERM=xterm-256color
```

New containers receive this setting from the project Compose override.

X11 forwarding is only needed for interactive GUI play. Even when X11 is
enabled on the container, isolate every batch evaluation, training,
certification, and offscreen-video process from it:

```bash
env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p <script> <arguments>
```

This avoids the verified Kit `XOpenDisplay` startup crash without changing the
interactive shell or disabling later GUI sessions.

## Verify The Runtime

Inside the container:

```bash
cd /workspace/UAVPredatorPrey
./scripts/linux/verify_container_runtime.sh
/workspace/isaaclab/isaaclab.sh -p -m pytest tests -q
```

The deterministic migration certification, including the five-seed anchor and
all eight pool matchups, is:

```bash
env -u DISPLAY -u XAUTHORITY \
  /workspace/isaaclab/isaaclab.sh -p scripts/skrl/certify_baseline.py
```

Reports are written outside the repository under
`/workspace/artifacts/isaac/evaluations/`.

## Run The 3v1 Hysteresis Curriculum

Use the dedicated Linux runbook:

- [`docs/3v1_hysteresis_linux_runbook.md`](docs/3v1_hysteresis_linux_runbook.md)

It contains the decision rule, eval-only check, 100-update smoke test, full
pool/cross-play command, output layout, monitoring, video settings, and manual
recovery procedure. Every invocation must use a new output directory because
the scheduler deliberately does not perform an implicit resume.

## JAX Simulator

The JAX simulator lives in the separate sibling repository
`$HOME/RL/UAVPredatorPreyJAX` and its own container. It shares an explicit
simulator contract with this Isaac implementation rather than importing Isaac
dependencies. The architecture and transfer constraints are documented in
[`docs/linux_docker_jax_setup.md`](docs/linux_docker_jax_setup.md).

The existing rewards, observations, resets, terminations, and curriculum rules
are semantic source material for the JAX port; they do not need to be designed
again. Isaac scene calls and mutable Torch buffers must be replaced by pure JAX
functions while preserving feature ordering, frames, constants, event timing,
and `terminated` versus `truncated` behavior.

Export deterministic `direct_wrench_v0` physics traces from inside the Isaac
container with:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  scripts/jax/export_direct_wrench_oracle.py \
  --headless --device cuda:0
```

The generated JSON and SHA-256 sidecar are written under
`/workspace/artifacts/transfer/direct_wrench_v0/`; they are validation
artifacts and are not committed.

Export the checkpoint-compatible observation fixture without a rollout or
policy step with:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  scripts/jax/export_observation_oracle.py \
  --headless --device cuda:0
```

This writes the exact root inputs, `90D/30D/120D` outputs, alive masks, and
root-versus-system-COM diagnostics under
`/workspace/artifacts/transfer/observations_v0/`.

## Additional Documentation

- [`docs/marl_sota_roadmap.md`](docs/marl_sota_roadmap.md): model and league roadmap
- [`docs/curriculum_learning_plan.md`](docs/curriculum_learning_plan.md): curriculum principles
- [`docs/training_3v1_obstacles.md`](docs/training_3v1_obstacles.md): obstacle-task workflow
- [`docs/1v1_hysteresis_curriculum_runbook.md`](docs/1v1_hysteresis_curriculum_runbook.md): legacy Windows 1v1 history
- [`docs/planning_and_hierarchical_rl_ideas.md`](docs/planning_and_hierarchical_rl_ideas.md): hierarchical and MPC ideas
