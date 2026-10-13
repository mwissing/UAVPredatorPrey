# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import CRAZYFLIE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class Uav3v1EnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    num_predators = 3
    # Logging-only reductions can be disabled by the training launcher without
    # changing actions, dynamics, rewards, terminations, or episode tracking.
    enable_step_diagnostics = True

    # multi-agent specification (2 agents: shared predator policy + prey)
    possible_agents = ["predator", "prey"]
    # Predator obs (72D): 3 × 24D stacked (own_state(12) + prey_info(6) + 2_teammate_pos(6)) per drone
    # Prey obs (30D): own_state(12) + 3_predator_info(18)
    action_spaces = {"predator": 12, "prey": 4}          # 3 × 4D for predator team
    observation_spaces = {"predator": 72, "prey": 30}     # 3 × 24D stacked
    state_space = 102  # 72 + 30 = concatenation of all observations for MAPPO

    # simulation — optimized for RTX 5090
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,                          # 100 Hz physics; decimation=2 gives 50 Hz policy actions
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene — 4096 envs × 4 drones = 16384 articulations (same total as working 1v1)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=10.0, replicate_physics=True, clone_in_fabric=True
    )

    # robots — keep CRAZYFLIE_CFG defaults (retain_accelerations=True is required for thrust!)
    predator_0_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Predator_0",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
    )
    predator_1_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Predator_1",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
    )
    predator_2_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Predator_2",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
    )
    prey_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Prey",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
        ),
    )

    # physics parameters
    predator_thrust_to_weight = 1.9
    prey_thrust_to_weight = 2.2       # prey is ~15% faster → predators MUST coordinate
    moment_scale = 0.01

    # === REWARD HIERARCHY ===
    # Priority 1: Learn to fly (MUST dominate early training)
    upright_reward_scale = 5.0             # was 2.0 — stronger signal to stay level
    target_height = 1.0
    height_penalty_scale = -8.0            # was -3.0 — stronger penalty for wrong altitude

    # Priority 2: Stay in arena
    boundary_warn_fraction = 0.6
    boundary_penalty_scale = 50.0

    # Priority 3: Predator-prey task (gated on being airborne)
    predator_proximity_reward_scale = 5.0  # was 8.0 — reduced so flight dominates early
    predator_distance_progress_reward_scale = 0.0
    predator_distance_progress_reward_clip = 0.08
    predator_catch_bonus = 200.0           # effective team bonus stays constant across predator counts
    predator_assist_bonus = 0.0            # disabled while testing whether assist encourages orbiting
    assist_distance = 1.5                  # [m] must be this close to get assist reward
    prey_alive_bonus = 2.0
    prey_evasion_reward_scale = 5.0       # reward for distance from nearest predator (mirrors predator proximity)
    prey_evasion_vertical_weight = 0.5
    prey_caught_penalty = -200.0
    prey_low_altitude_penalty_scale = 0.0
    prey_low_altitude_margin = 1.2
    prey_distance_progress_reward_scale = 0.0
    prey_distance_progress_reward_clip = 0.08
    prey_boundary_progress_reward_scale = 0.0
    prey_boundary_progress_reward_clip = 0.08
    prey_boundary_progress_start_fraction = 0.55

    # Action/velocity penalties (prevent wild oscillations)
    lin_vel_penalty = -0.05
    predator_lin_vel_penalty_free_speed = 0.0  # [m/s], zero preserves legacy behavior
    ang_vel_penalty = -0.01              # legacy default; kept for older configs/tools
    predator_ang_vel_penalty = -0.01
    prey_ang_vel_penalty = -0.03
    action_penalty = -0.05                 # NEW: penalizes large actions

    # OOB penalties
    oob_penalty = -200.0  # legacy default; kept for older configs/tools
    predator_oob_penalty = -200.0
    prey_oob_penalty = -200.0
    arena_center_z = 1.0
    soft_arena_boundary = False
    soft_arena_radius = 5.0
    soft_arena_penalty_scale = 0.0

    # catch/termination parameters
    catch_distance = 0.3
    arena_radius = 5.0
    min_height = 0.1
    max_height = 3.0

    # spawn parameters
    predator_spawn_radius = 2.0  # predators spawn on a circle around center
    prey_spawn_pos = [0.0, 0.0, 1.0]  # prey spawns at center
    spawn_pos_noise = 0.3
    predator_mixed_spawn = False
    predator_mixed_spawn_ring_probability = 0.50
    predator_mixed_spawn_wide_probability = 0.25
    predator_wide_spawn_radius = 4.0
    predator_same_side_spawn_radius = 3.8
    predator_same_side_spawn_spread = 0.7  # radians across the predator formation
    random_arena_spawn = False
    random_spawn_radius_fraction = 0.85
    random_spawn_z_min = 0.7
    random_spawn_z_max = 2.3
    random_spawn_min_prey_predator_distance = 1.0
    random_spawn_min_predator_distance = 0.8
    random_spawn_resample_attempts = 32
    predator_teammate_close_distance = 0.5  # diagnostics only: flags near teammate contact
    predator_teammate_velocity_observation = False

    # Vertical pillar obstacles (0 = none). Collisions are analytic: touching a
    # pillar counts as a crash (predator inactive, or prey OOB). Rendered
    # cylinders have no collision shapes. Each drone observes, per pillar, the
    # body-frame vector to the pillar axis at its own height and the surface
    # distance, both divided by obstacle_feature_scale.
    num_obstacles = 0
    obstacle_radius = 0.4                 # [m]
    obstacle_height = 12.0                # [m] rendered height, centred on arena_center_z
    obstacle_placement_radius = 3.5       # [m] max horizontal axis distance from the arena centre
    obstacle_min_axis_separation = 1.6    # [m] between pillar axes, so drones can pass between pillars
    obstacle_spawn_clearance = 0.5        # [m] min gap between a pillar surface and any spawning drone
    obstacle_collision_margin = 0.06      # [m] drone body radius
    # End-game curriculum (shield starts): this share of episodes starts with the predators at a random distance and
    # bearing around the prey and a pillar right next to the prey, the situation in which the predators never learned
    # to flank a prey that circles a pillar (2026-10-13). Start states only, no reward. 0 = off.
    shield_start_fraction = 0.0
    shield_start_gap = (0.3, 0.8)                 # [m] prey to shield-pillar surface
    shield_start_predator_distance = (3.0, 6.0)   # [m] predators from the prey
    obstacle_feature_scale = 5.0          # [m] observation scale
    obstacle_soft_margin = 0.5            # [m] clearance band with a quadratic keep-out cost
    obstacle_soft_penalty_scale = 0.0     # 0 = off; cost = scale * (margin - clearance)^2 * dt (like the soft arena wall)
    # How many of the num_obstacles slots get a pillar each episode (-1 = all; an int, because Hydra cannot
    # override a None default with an int). The rest stay parked, and the models' presence mask ignores them.
    # With 4 active slots, the placement draws the same random numbers as a 4-pillar task, so 8-slot
    # checkpoints can be evaluated in the 4-pillar layouts.
    num_active_obstacles = -1

    # Scripted "ghost" predators for training the prey alone (see ghosts.py). Predator actions are
    # ignored: each predator flies a speed- and acceleration-limited lead pursuit of the prey, never
    # crashes, and passes through pillars. Train with the predator agent frozen.
    ghost_predators = False
    ghost_max_speed = 4.5                 # [m/s]
    ghost_max_speed_range = None          # (min, max) [m/s]: if set, each episode draws its ghost speed uniformly
    ghost_max_accel = 12.0                # [m/s^2], about what T/W 1.9 allows after attitude lag
    ghost_lead_times = (0.0, 0.25, 0.5)   # [s] per predator: aim at prey position + lead * prey velocity
    ghost_min_aim_height = 0.6            # [m] relative to the env origin
    # Mixed rollouts: this share of the envs (the first ones) gets ghost predators, the rest real ones, so every
    # prey update sees both. The predator then trains only on the real envs (RecurrentMAPPO.set_training_envs).
    # Colliders stay on in all envs (ghost_predators=True disables them). 0 = off.
    ghost_env_fraction = 0.0


@configclass
class Uav1v1SurvivalEasyEnvCfg(Uav3v1EnvCfg):
    """Phase-2 curriculum: learn bounded prey evasion before 3v1 pressure or obstacles."""

    num_predators = 1

    action_spaces = {"predator": 4, "prey": 4}
    observation_spaces = {"predator": 18, "prey": 18}
    state_space = 36

    catch_distance = 0.3
    predator_spawn_radius = 3.2

    boundary_warn_fraction = 0.55
    boundary_penalty_scale = 80.0

    predator_proximity_reward_scale = 1.0
    predator_distance_progress_reward_scale = 10.0
    prey_alive_bonus = 3.0
    prey_evasion_reward_scale = 8.0
    prey_distance_progress_reward_scale = 10.0
    prey_boundary_progress_reward_scale = 18.0
    prey_boundary_progress_start_fraction = 0.40


@configclass
class Uav1v1SurvivalSoftOobEnvCfg(Uav1v1SurvivalEasyEnvCfg):
    """1v1 survival variant with soft 3D arena recovery instead of horizontal OOB termination."""

    boundary_warn_fraction = 1.0
    boundary_penalty_scale = 0.0
    prey_boundary_progress_reward_scale = 0.0

    soft_arena_boundary = True
    soft_arena_radius = 5.0
    soft_arena_penalty_scale = 200.0
    prey_low_altitude_penalty_scale = 80.0
    prey_low_altitude_margin = 1.2


@configclass
class Uav2v1SurvivalSoftOobEnvCfg(Uav1v1SurvivalSoftOobEnvCfg):
    """2v1 survival curriculum step: same soft-OOB rules, one additional predator."""

    num_predators = 2

    action_spaces = {"predator": 8, "prey": 4}
    observation_spaces = {"predator": 42, "prey": 24}
    state_space = 66


@configclass
class Uav2v1SurvivalSoftOobMixedSpawnEnvCfg(Uav2v1SurvivalSoftOobEnvCfg):
    """2v1 soft-OOB diagnostic with mixed predator spawn geometry."""

    predator_mixed_spawn = True
    predator_mixed_spawn_ring_probability = 0.50
    predator_mixed_spawn_wide_probability = 0.25
    predator_wide_spawn_radius = 4.0
    predator_same_side_spawn_radius = 3.8
    predator_same_side_spawn_spread = 0.7
    spawn_pos_noise = 0.35


@configclass
class Uav3v1SurvivalSoftOobEnvCfg(Uav1v1SurvivalSoftOobEnvCfg):
    """3v1 survival curriculum step with the same soft-OOB constraints as 1v1/2v1."""

    num_predators = 3

    action_spaces = {"predator": 12, "prey": 4}
    observation_spaces = {"predator": 72, "prey": 30}
    state_space = 102


@configclass
class Uav3v1SurvivalSoftOobTeammateVelEnvCfg(Uav3v1SurvivalSoftOobEnvCfg):
    """3v1 soft-OOB variant where predators observe teammate relative velocities."""

    predator_teammate_velocity_observation = True
    observation_spaces = {"predator": 90, "prey": 30}
    state_space = 120


@configclass
class Uav3v1FlightBootstrapTeammateVelEnvCfg(Uav3v1SurvivalSoftOobTeammateVelEnvCfg):
    """3v1 flight-stability bootstrap with chase/evasion rewards disabled."""

    predator_proximity_reward_scale = 0.0
    predator_distance_progress_reward_scale = 0.0
    predator_catch_bonus = 0.0
    predator_assist_bonus = 0.0

    prey_evasion_reward_scale = 0.0
    prey_caught_penalty = 0.0
    prey_distance_progress_reward_scale = 0.0
    prey_boundary_progress_reward_scale = 0.0

    predator_mixed_spawn = False
    random_arena_spawn = False
    spawn_pos_noise = 0.1


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnEnvCfg(Uav3v1SurvivalSoftOobTeammateVelEnvCfg):
    """3v1 teammate-velocity variant with random starts inside an elevated spherical arena."""

    predator_proximity_reward_scale = 0.0

    arena_center_z = 6.0
    soft_arena_radius = 5.0
    random_arena_spawn = True
    random_spawn_radius_fraction = 0.85
    random_spawn_z_min = 4.5
    random_spawn_z_max = 7.5
    random_spawn_min_prey_predator_distance = 1.0
    random_spawn_min_predator_distance = 0.8
    random_spawn_resample_attempts = 32
    spawn_pos_noise = 0.0


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg(
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnEnvCfg
):
    """Late-stage ablation with weaker direct-pursuit shaping."""

    predator_distance_progress_reward_scale = 2.0


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstaclesEnvCfg(
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg
):
    """Low-progress random-spawn task with four vertical pillar obstacles (roadmap stage 8).

    Touching a pillar is a crash with the existing crash penalties, except that
    the prey's costs more than being caught (below), and a 0.5 m band around
    each pillar carries the same quadratic keep-out cost as the soft arena wall.
    Each drone observes 4 features per pillar, appended to its observation block.
    """

    num_obstacles = 4
    # Same quadratic keep-out cost as the soft arena wall (soft_arena_penalty_scale = 200).
    obstacle_soft_penalty_scale = 200.0
    # A prey crash (pillar or arena exit) costs more than being caught (-200), so the prey does not take a crash
    # to escape a catch (user decision 2026-10-10). Obstacle-free tasks keep the legacy -200.
    prey_oob_penalty = -300.0
    # The team reward is the mean over the 3 predators, so one lost drone costs the team -200: as much as a catch
    # earns (the catch bonus uses the same team-size scaling). Used in every obstacle run since 2026-10-08.
    predator_oob_penalty = -600.0
    observation_spaces = {"predator": 90 + 3 * 4 * 4, "prey": 30 + 4 * 4}
    state_space = 120 + 3 * 4 * 4 + 4 * 4


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstacles8EnvCfg(
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstaclesEnvCfg
):
    """Denser pillar task: 8 pillar slots, placed up to the arena wall (no pillar-free belt).

    Four pillars spread up to the wall are met too rarely for the prey to learn perception-based avoidance
    (ghost stage v5), so this task doubles the density. Checkpoints from the 4-pillar task are expanded with
    scripts/expand_checkpoint_obstacles.py; the extra slots start parked and are masked.
    """

    num_obstacles = 8
    obstacle_placement_radius = 4.6
    observation_spaces = {"predator": 90 + 3 * 8 * 4, "prey": 30 + 8 * 4}
    state_space = 120 + 3 * 8 * 4 + 8 * 4


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstacles10EnvCfg(
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressObstaclesEnvCfg
):
    """Pillars up to the soft wall: 10 slots with axes uniform over the whole 5 m arena disk.

    With axes up to 4.6 m (8 pillars), a drone flying along the wall meets only the pillars of the outer ring,
    and self-play settled on a slow prey at the wall (mixed ghost rollouts, 2026-10-11). Here the wall ring is as
    dense as the interior; 10 pillars over the 5 m disk keep about the 8-pillar density (0.127 vs 0.120 per m^2).
    Checkpoints are expanded from 8 slots with scripts/expand_checkpoint_obstacles.py --from-obstacles 8.
    """

    num_obstacles = 10
    obstacle_placement_radius = 5.0
    observation_spaces = {"predator": 90 + 3 * 10 * 4, "prey": 30 + 10 * 4}
    state_space = 120 + 3 * 10 * 4 + 10 * 4


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressSpeedMarginEnvCfg(
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnLowProgressEnvCfg
):
    """Low-progress variant that leaves predator speeds up to 6 m/s unpenalized."""

    predator_lin_vel_penalty_free_speed = 6.0  # [m/s]


@configclass
class Uav3v1SurvivalSoftOobTeammateVelRandomSpawnPhysics50HzEnvCfg(
    Uav3v1SurvivalSoftOobTeammateVelRandomSpawnEnvCfg
):
    """A/B task with 50 Hz physics and the same 50 Hz policy-action frequency."""

    decimation = 1

    def __post_init__(self) -> None:
        self.sim.dt = 1 / 50
        self.sim.render_interval = self.decimation


@configclass
class Uav1v1SurvivalSoftOobPred22EnvCfg(Uav1v1SurvivalSoftOobEnvCfg):
    """Soft-OOB 1v1 ablation with predator thrust matched to the prey."""

    predator_thrust_to_weight = 2.2


@configclass
class Uav1v1SurvivalSoftOobPred24EnvCfg(Uav1v1SurvivalSoftOobEnvCfg):
    """Soft-OOB 1v1 ablation with a modest predator thrust advantage."""

    predator_thrust_to_weight = 2.4
