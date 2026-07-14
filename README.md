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

### Export an skrl policy for the JAX runtime

The Isaac container is the only runtime that may parse the pickle-capable
skrl `.pt` format. Export the verified migration checkpoint into the neutral,
pickle-free `skrl_policy_transfer_v0` directory before importing it in JAX:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  scripts/jax/export_skrl_policy_transfer_v0.py \
  --bundle-root \
    .pretrained_checkpoints/linux_migration_2026-07-11 \
  --checkpoint \
    .pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt \
  --output \
    /workspace/artifacts/transfer/skrl_policy_transfer_v0/windows_agent_115200
```

The exporter verifies the source hashes before calling
`torch.load(..., weights_only=True, map_location="cpu")`. It preserves the 94
model tensors and 18 running-scaler tensors in their source layout and dtype;
optimizer tensors are excluded. The output directory is checksummed,
pickle-free, and published with no-replace semantics. Run the importer and
deterministic forward oracle in the sibling JAX repository as documented
there.

Export a deterministic checkpoint-forward oracle from the actual skrl model
classes without launching Isaac Sim:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  scripts/jax/export_skrl_policy_oracle_v0.py \
  --checkpoint \
    .pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt \
  --expected-checkpoint-sha256 \
    88e7161b48323452c5d302b25a8f50d1fd5ac026fdeddd2819d64eeb2521ba98 \
  --output \
    /workspace/artifacts/transfer/skrl_policy_oracle_v0/windows_agent_115200
```

The JAX comparator binds this oracle and the imported policy to the same source
checkpoint SHA before comparing normalizers, actor outputs, critic outputs,
and nonzero recurrent states.

This boundary reproduces clean-boundary inference, not an exact training
resume. The migrated `.pt` has no environment, rollout, RNG, or GRU carry
state, and its frozen prey role has no optimizer. JAX fine-tuning therefore
starts with fresh Adam state for both roles.

### Export the paired policy-trajectory diagnostic

After policy-forward parity is established, capture a representative action
tape and the corresponding PhysX trajectory at the 10 ms physics boundary:

For a canonical artifact, inject the running container's immutable image ID
when entering it from the host, then run the exporter command below in that
shell:

```bash
IMAGE_ID="$(docker container inspect -f '{{.Image}}' isaac-lab-base-uav51)"
docker exec -e UAV_ISAAC_IMAGE_ID="$IMAGE_ID" -it \
  isaac-lab-base-uav51 bash
```

```bash
/workspace/isaaclab/isaaclab.sh -p \
  scripts/jax/export_isaac_policy_trajectory_v0.py \
  --checkpoint \
    .pretrained_checkpoints/linux_migration_2026-07-11/current/agent_115200.pt \
  --expected-checkpoint-sha256 \
    88e7161b48323452c5d302b25a8f50d1fd5ac026fdeddd2819d64eeb2521ba98 \
  --output \
    /workspace/artifacts/transfer/policy_trajectory_v0/windows_agent115200_controlled_seed42 \
  --headless --device cuda:0
```

Repeat 0 starts all four vehicles from the documented controlled formation,
zeros both actor GRUs, and records 50 deterministic `tanh(mean)` policy
actions. Repeats 1 and 2 restore the same articulation state and replay the
exact post-clipping tape. Every 10 ms substep is sampled, including the initial
state, for 101 samples per repeat. The exporter aborts on catch, OOB, or an
inactive predator and never calls reward, termination, auto-reset, or training
code. It stores mass-weighted five-link system-COM translation, root-link
attitude/rate, root diagnostics, and rotor joint state in an atomically
published, no-overwrite, checksummed fixture outside Git.

The JAX repository contains the matching open-loop comparator. It initializes
the lumped plant from PhysX read-back—not the requested pose—and requires exact
action/torque identity plus at most one `float32` ULP of force-encoding
difference before reporting descriptive trajectory error at policy horizons
1, 2, 5, 10, 25, and 50. Use a new output directory for a new capture because
fixtures deliberately have no overwrite mode.

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

Export externally observable reset invariants and distribution summaries with:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  scripts/jax/export_reset_oracle.py \
  --num-envs 4096 --seeds 42 43 44 45 \
  --headless --device cuda:0
```

The exporter also includes a forced zero-radius case that confirms the legacy
32-attempt fallback returns an invalid layout rather than looping indefinitely
or raising. Artifacts are written under
`/workspace/artifacts/transfer/reset_v0/` and are not committed.

## Additional Documentation

- [`docs/marl_sota_roadmap.md`](docs/marl_sota_roadmap.md): model and league roadmap
- [`docs/curriculum_learning_plan.md`](docs/curriculum_learning_plan.md): curriculum principles
- [`docs/training_3v1_obstacles.md`](docs/training_3v1_obstacles.md): obstacle-task workflow
- [`docs/1v1_hysteresis_curriculum_runbook.md`](docs/1v1_hysteresis_curriculum_runbook.md): legacy Windows 1v1 history
- [`docs/planning_and_hierarchical_rl_ideas.md`](docs/planning_and_hierarchical_rl_ideas.md): hierarchical and MPC ideas
