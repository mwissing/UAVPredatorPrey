from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import gymnasium
import torch

from obstacles import PARKED_DISTANCE, axis_offsets_world, collides, sample_obstacle_xy, surface_distance


AGENTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "UAVPredatorPrey"
    / "UAVPredatorPrey"
    / "tasks"
    / "direct"
    / "uavpredatorprey_3v1"
    / "agents"
)
PACKAGE_NAME = "_obstacle_test_agents"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(AGENTS_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package)
attention_models = importlib.import_module(f"{PACKAGE_NAME}.attention_models")

K = 4
OBSTACLE_DIM = 4


def test_placement_respects_radius_separation_and_spawn_clearance() -> None:
    torch.manual_seed(0)
    radius = 4.0 * torch.sqrt(torch.rand(256, 4))
    angle = 2 * torch.pi * torch.rand(256, 4)
    drone_xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)

    placed, parked = sample_obstacle_xy(
        drone_xy, K, placement_radius=3.5, obstacle_radius=0.4, min_axis_separation=1.6, spawn_clearance=0.5
    )

    active = torch.linalg.norm(placed, dim=-1) < PARKED_DISTANCE
    assert parked.sum().item() == (~active).sum().item()
    assert active.float().mean() > 0.95
    assert (torch.linalg.norm(placed, dim=-1)[active] <= 3.5 + 1e-6).all()
    gaps = surface_distance(drone_xy, placed, 0.4)  # (N, D, K)
    assert (gaps.permute(0, 2, 1)[active] >= 0.5 - 1e-6).all()
    for i in range(K):
        for j in range(i + 1, K):
            both = active[:, i] & active[:, j]
            separation = torch.linalg.norm(placed[:, i] - placed[:, j], dim=-1)
            assert (separation[both] >= 1.6 - 1e-6).all()


def test_placement_parks_pillars_when_no_position_is_valid() -> None:
    drone_xy = torch.zeros(8, 1, 2)
    placed, parked = sample_obstacle_xy(
        drone_xy, 2, placement_radius=0.5, obstacle_radius=0.4, min_axis_separation=1.6, spawn_clearance=0.5
    )
    assert torch.equal(parked, torch.full((8,), 2))
    assert torch.allclose(placed, torch.full_like(placed, PARKED_DISTANCE))


def test_surface_distance_collision_and_axis_offsets() -> None:
    obstacle_xy = torch.tensor([[[1.0, 0.0]]])
    points_xy = torch.tensor([[[0.0, 0.0], [0.65, 0.0], [1.45, 0.0], [1.5, 0.0]]])

    distance = surface_distance(points_xy, obstacle_xy, 0.4)[0, :, 0]
    assert torch.allclose(distance, torch.tensor([0.6, -0.05, 0.05, 0.1]), atol=1e-6)
    hits = collides(points_xy, obstacle_xy, 0.4, margin=0.06)[0]
    assert hits.tolist() == [False, True, True, False]
    assert not collides(points_xy, obstacle_xy[:, :0], 0.4, margin=0.06).any()

    points = torch.tensor([[[0.0, 0.0, 6.0]]])
    offsets = axis_offsets_world(points, obstacle_xy)
    assert torch.allclose(offsets, torch.tensor([[[[1.0, 0.0, 0.0]]]]))


def _with_obstacles(base: torch.Tensor, blocks: int, block_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Append random obstacle features to each of `blocks` per-drone observation blocks."""
    batch = base.shape[0]
    obstacles = torch.randn(batch, blocks, K * OBSTACLE_DIM, generator=generator)
    return torch.cat((base.view(batch, blocks, block_dim), obstacles), dim=-1).reshape(batch, -1)


def _box(size: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(size,))


def _actor(observations: int, actions: int, num_obstacles: int):
    torch.manual_seed(1)
    return attention_models.RecurrentPredatorPreyAttentionGaussianModel(
        _box(observations), _box(actions), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=num_obstacles,
    )


def _critic(states: int, num_obstacles: int):
    torch.manual_seed(1)
    return attention_models.RecurrentEntityAttentionCentralizedValueModel(
        _box(states), _box(1), device="cpu", hidden_size=32, attention_size=16, rnn_hidden_size=24,
        num_obstacles=num_obstacles,
    )


def test_obstacle_branch_reproduces_obstacle_free_networks_exactly() -> None:
    generator = torch.Generator().manual_seed(2)
    batch = 5

    # Predator team: 3 drones x 30 base features (+ 16 obstacle features each).
    base, extended = _actor(90, 12, 0), _actor(90 + 3 * K * OBSTACLE_DIM, 12, K)
    extended.load_state_dict(base.state_dict())
    assert extended.mode == base.mode == "predator"
    assert any(name.startswith("obstacle_") for name in extended.state_dict())
    states = torch.randn(batch, 90, generator=generator)
    mean_base, log_std_base, _ = base.compute({"states": states})
    mean_ext, log_std_ext, _ = extended.compute({"states": _with_obstacles(states, 3, 30, generator)})
    assert torch.equal(mean_base, mean_ext)
    assert torch.equal(log_std_base, log_std_ext)

    # Prey: 30 base features (+ 16 obstacle features).
    base, extended = _actor(30, 4, 0), _actor(30 + K * OBSTACLE_DIM, 4, K)
    extended.load_state_dict(base.state_dict())
    assert extended.mode == base.mode == "prey"
    states = torch.randn(batch, 30, generator=generator)
    mean_base, _, _ = base.compute({"states": states})
    mean_ext, _, _ = extended.compute({"states": _with_obstacles(states, 1, 30, generator)})
    assert torch.equal(mean_base, mean_ext)

    # Critic: predator blocks then the prey block, each with its obstacle features.
    base, extended = _critic(120, 0), _critic(120 + 4 * K * OBSTACLE_DIM, K)
    extended.load_state_dict(base.state_dict())
    assert extended.uses_entity_attention and base.uses_entity_attention
    predators = torch.randn(batch, 90, generator=generator)
    prey = torch.randn(batch, 30, generator=generator)
    value_base, _ = base.compute({"states": torch.cat((predators, prey), dim=-1)})
    extended_states = torch.cat(
        (_with_obstacles(predators, 3, 30, generator), _with_obstacles(prey, 1, 30, generator)), dim=-1
    )
    value_ext, _ = extended.compute({"states": extended_states})
    assert torch.equal(value_base, value_ext)


def test_obstacle_branch_receives_gradients_from_the_start() -> None:
    extended = _actor(30 + K * OBSTACLE_DIM, 4, K)
    states = torch.randn(3, 30 + K * OBSTACLE_DIM)
    mean, _, _ = extended.compute({"states": states})
    mean.square().sum().backward()
    # The zero projection itself gets a gradient, so the branch can grow from zero.
    assert extended.obstacle_out.weight.grad.abs().sum() > 0


def test_checkpoint_expansion_inserts_normalizer_entries_after_each_drone_block() -> None:
    from expand_checkpoint_obstacles import expand_checkpoint

    def scaler(size: int) -> dict[str, torch.Tensor]:
        return {
            "running_mean": torch.arange(size, dtype=torch.float64) + 100.0,
            "running_variance": torch.arange(size, dtype=torch.float64) + 200.0,
            "current_count": torch.tensor(1.0e11, dtype=torch.float64),
        }

    checkpoint = {
        "predator": {"policy": {"w": torch.ones(1)}, "state_preprocessor": scaler(6), "shared_state_preprocessor": scaler(8)},
        "prey": {"optimizer": {}, "state_preprocessor": scaler(2), "shared_state_preprocessor": scaler(8)},
    }
    expanded = expand_checkpoint(checkpoint, num_predators=3, num_obstacles=1, obstacle_dim=2)

    predator_mean = expanded["predator"]["state_preprocessor"]["running_mean"].tolist()
    assert predator_mean == [100, 101, 0, 0, 102, 103, 0, 0, 104, 105, 0, 0]
    assert expanded["predator"]["state_preprocessor"]["running_variance"].tolist()[2:4] == [1.0, 1.0]
    shared_mean = expanded["prey"]["shared_state_preprocessor"]["running_mean"].tolist()
    assert shared_mean == [100, 101, 0, 0, 102, 103, 0, 0, 104, 105, 0, 0, 106, 107, 0, 0]
    assert expanded["prey"]["state_preprocessor"]["running_mean"].tolist() == [100, 101, 0, 0]
    assert "optimizer" not in expanded["prey"]
    assert torch.equal(expanded["predator"]["policy"]["w"], torch.ones(1))
    assert expanded["prey"]["state_preprocessor"]["current_count"].item() == 1.0e11


def test_obstacle_parameters_get_their_own_learning_rate_group() -> None:
    from recurrent_mappo import _obstacle_param_groups

    actor = _actor(30 + K * OBSTACLE_DIM, 4, K)
    critic = _critic(120 + 4 * K * OBSTACLE_DIM, K)
    groups = _obstacle_param_groups([actor, critic], 1.0e-5, 30.0)

    obstacle_ids = {id(p) for module in (actor, critic) for n, p in module.named_parameters() if n.startswith("obstacle_")}
    assert len(groups) == 2
    assert {id(p) for p in groups[1]["params"]} == obstacle_ids
    assert not obstacle_ids & {id(p) for p in groups[0]["params"]}
    assert groups[0]["lr"] == 1.0e-5 and groups[1]["lr"] == 30.0 * 1.0e-5 and groups[1]["lr_multiplier"] == 30.0
    total = sum(1 for module in (actor, critic) for _ in module.parameters())
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == total
    # Obstacle-free models keep a single group.
    assert len(_obstacle_param_groups([_actor(30, 4, 0)], 1.0e-5, 30.0)) == 1
