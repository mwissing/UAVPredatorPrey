# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..uavpredatorprey_3v1.uav_3v1_env_cfg import Uav3v1EnvCfg


@configclass
class Uav3v1ObstaclesEnvCfg(Uav3v1EnvCfg):
    """3v1 Predator-Prey with randomly placed cylindrical obstacles.

    Phase 1: Obstacles are visual-only (no physics collision).
    Avoidance is enforced purely through RL observations + rewards.
    This lets drones learn to fly/catch first without PhysX interference.
    """

    # Updated observation spaces: +4D per drone (nearest obstacle vec(3) + dist(1))
    # Predator: 28D per drone × 3 = 84D, Prey: 34D
    observation_spaces = {"predator": 84, "prey": 34}
    state_space = 118  # 84 + 34
    obstacle_observation_mode = "nearest"
    obstacle_features_per_obstacle = 7  # body vec(3), distance, risk, path fraction, path block score

    catch_distance = 0.3                     # curriculum stage 3: final tighter catch radius after stable 0.4 training

    # Boundary: stronger + earlier warning to reduce OOB rate
    boundary_warn_fraction = 0.45         # warning starts at 2.25m instead of 3.0m (parent: 0.6)
    boundary_penalty_scale = 130.0        # stronger brake against cover-induced chase overshoot

    # --- Obstacle configuration ---
    num_obstacles = 4                     # full obstacle count
    obstacle_radius = 0.3                 # [m] cylinder radius
    obstacle_height = 3.0                 # [m] cylinder height (floor to max_height)

    # Predator spawn: override parent (2.0m) to give the prey a headstart to reach cover
    predator_spawn_radius = 3.2           # [m] predators start further away (parent: 2.0m)

    # Obstacles spawn randomly in an annular region.
    # Keep them reachable from center so the prey can actually use them as cover before being caught.
    obstacle_min_spawn_radius = 1.0       # [m] move cover opportunities closer to prey start (was 1.6m)
    obstacle_max_spawn_radius = 3.1       # [m] keep cover between prey center and predator approach paths
    obstacle_min_separation = 1.0         # [m] min distance between obstacle centers (tighter for 4)
    obstacle_agent_min_spawn_distance = 0.7  # [m] avoid spawning obstacles directly on initial drones (was 1.0m)
    obstacle_spawn_correction_passes = 8  # fixed reset-time relaxation passes for spawn clearances
    obstacle_cover_spawn_count = 2        # place reachable cover opportunities without gifting full cover
    obstacle_cover_spawn_fraction = 0.62  # fraction along prey->predator segment for cover candidates
    obstacle_cover_spawn_fraction_noise = 0.08
    obstacle_cover_spawn_angle_noise = 0.25  # [rad] keeps cover useful without making every reset identical
    obstacle_cover_spawn_radius_min = 0.9  # [m] align with closer spawn boundaries (was 1.45m)
    obstacle_cover_spawn_radius_max = 1.5  # [m] (was 1.95m)
    obstacle_cover_lateral_offset_min = 1.1  # [m] keep initial obstacles off the direct line of sight
    obstacle_cover_lateral_offset_max = 1.55 # [m] prey must move behind/around the obstacle to earn cover


    # Obstacle rigid body configs — NO collision_props! Drones fly through them.
    # Avoidance is learned through RL rewards, not physics.
    obstacle_0_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=3.0,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 1.5)),
    )
    obstacle_1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=3.0,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 1.5)),
    )
    # All four configs are instantiated when num_obstacles = 4.
    obstacle_2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=3.0,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0, 1.5)),
    )
    obstacle_3_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=3.0,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -3.0, 1.5)),
    )

    # Reward: obstacle penalties.
    # Proximity uses a normalized quadratic risk in [0, 1], so this is the max per-step penalty.
    obstacle_proximity_penalty = -2.0     # stronger gradient to route around cover instead of through it
    obstacle_warn_distance = 0.85         # wider warning zone for cover-biased obstacle layouts
    obstacle_collision_penalty = -150.0   # strong penalty to prevent drones from cheating by flying through visual cylinders
    obstacle_collision_distance = 0.5     # [m] ~obstacle_radius + drone_radius + margin
    obstacle_path_block_radius = 0.85     # [m] scores obstacles blocking the agent-target line

    # Reward: predator pursuit progress.
    # Dense signal for closing the closest predator-prey distance. This counters the failure mode
    # where predators learn obstacle avoidance by disengaging from the chase.
    predator_progress_reward_scale = 12.0
    predator_progress_reward_clip = 0.08  # [m/step] clip distance delta before scaling
    predator_progress_min_gate = 0.35     # keep chase feedback alive while routing near obstacles
    prey_distance_progress_reward_scale = 0.0
    prey_distance_progress_reward_clip = 0.08  # [m/step] clip escape distance delta before scaling
    prey_boundary_progress_reward_scale = 0.0
    prey_boundary_progress_reward_clip = 0.08  # [m/step] clip inward radial progress before scaling
    prey_boundary_progress_start_fraction = 0.55

    # Reward: prey cover usage.
    # A cover event means an obstacle lies between the prey and the closest predator in XY.
    prey_cover_reward_scale = 24.0        # shaping for active line-of-sight cover use
    prey_cover_seek_reward_scale = 2.0    # light proximity hint; shadow target below provides direction
    prey_cover_progress_reward_scale = 8.0  # signed reward for improving/maintaining LOS cover
    prey_cover_pressure_floor = 0.35      # keep cover shaping alive before predators are already too close
    prey_shadow_reward_scale = 40.0       # reward moving to the obstacle side hidden from the closest predator
    prey_shadow_progress_reward_scale = 4.0 # signed reward for moving toward better shadow targets
    prey_cover_progress_clip = 0.08       # per-step score delta clip for cover/shadow potential shaping
    prey_shadow_target_offset = 0.9       # [m] target distance behind obstacle, away from closest predator
    prey_shadow_target_radius = 2.75      # [m] broad enough that the prey gets a gradient from spawn
    prey_shadow_arena_margin = 0.4        # [m] ignore shadow targets too close to arena boundary
    prey_shadow_reward_delay_s = 0.1      # shadow target is off-line, so it can be rewarded almost immediately
    prey_shadow_reward_ramp_s = 0.3
    prey_cover_reward_delay_s = 0.05      # start rewarding cover almost immediately (2-3 steps after reset)
    prey_cover_reward_ramp_s = 0.2        # quickly ramp up to full scaling
    prey_cover_radius = 0.75              # [m] obstacle blocks LOS if predator-prey segment passes this close
    prey_cover_seek_distance = 1.6        # [m] distance where prey starts getting guidance toward cover
    prey_cover_seek_target_distance = 1.0 # [m] prefer a safe ring around obstacles, not scraping them
    prey_cover_seek_width = 0.6           # [m] triangular width around the safe cover-seeking ring
    prey_cover_threat_distance = 3.0      # [m] cover reward fades out when predators are far away
    prey_cover_min_los_fraction = 0.12    # ignore obstacles too close to prey end of LOS segment
    prey_cover_max_los_fraction = 0.95    # ignore obstacles behind/at predator endpoint
    predator_cover_penalty_scale = 0.0    # re-enable only after predator catch/avoidance is stable


@configclass
class Uav3v1ObstaclesEasyEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Stage A: learn stable pursuit with static obstacles, before cover tactics."""

    catch_distance = 0.5
    predator_spawn_radius = 2.0

    boundary_warn_fraction = 0.6
    boundary_penalty_scale = 80.0

    obstacle_min_spawn_radius = 2.8
    obstacle_max_spawn_radius = 4.0
    obstacle_min_separation = 1.2
    obstacle_agent_min_spawn_distance = 1.0
    obstacle_cover_spawn_count = 0

    obstacle_proximity_penalty = -1.0
    obstacle_warn_distance = 0.75
    obstacle_collision_penalty = -50.0

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 10.0


@configclass
class Uav3v1ObstaclesMidEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Stage B: tighten pursuit and move obstacles into the interaction region."""

    catch_distance = 0.4
    predator_spawn_radius = 2.5

    boundary_warn_fraction = 0.55
    boundary_penalty_scale = 100.0

    obstacle_min_spawn_radius = 1.6
    obstacle_max_spawn_radius = 3.6
    obstacle_min_separation = 1.1
    obstacle_agent_min_spawn_distance = 0.9
    obstacle_cover_spawn_count = 0

    obstacle_proximity_penalty = -1.5
    obstacle_warn_distance = 0.8
    obstacle_collision_penalty = -85.0

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 12.0


@configclass
class Uav3v1ObstaclesBridgeEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Stage A/B bridge: tighten pursuit before moving obstacles fully into the chase path."""

    catch_distance = 0.45
    predator_spawn_radius = 2.25

    boundary_warn_fraction = 0.58
    boundary_penalty_scale = 90.0

    obstacle_min_spawn_radius = 2.1
    obstacle_max_spawn_radius = 3.8
    obstacle_min_separation = 1.15
    obstacle_agent_min_spawn_distance = 0.95
    obstacle_cover_spawn_count = 0

    obstacle_proximity_penalty = -1.2
    obstacle_warn_distance = 0.78
    obstacle_collision_penalty = -65.0

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 11.0
    predator_progress_min_gate = 0.45


@configclass
class Uav3v1CoverEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Stage C: enable prey cover/shadow shaping after pursuit is stable."""

    # Inherits the cover-biased obstacle spawn and prey cover/shadow rewards.
    # Predator cover penalty stays off for now; otherwise the predator can be punished
    # for the prey's initial/accidental cover before learning reliable routing.
    predator_cover_penalty_scale = 0.0


@configclass
class Uav3v1CoverBridgeEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Stage B/C bridge: introduce cover incentives before the full cover-biased layout."""

    catch_distance = 0.4
    predator_spawn_radius = 2.75

    boundary_warn_fraction = 0.52
    boundary_penalty_scale = 110.0

    obstacle_min_spawn_radius = 1.35
    obstacle_max_spawn_radius = 3.4
    obstacle_min_separation = 1.1
    obstacle_agent_min_spawn_distance = 0.85
    obstacle_cover_spawn_count = 1
    obstacle_cover_spawn_fraction = 0.58
    obstacle_cover_spawn_fraction_noise = 0.10
    obstacle_cover_spawn_angle_noise = 0.35
    obstacle_cover_spawn_radius_min = 1.1
    obstacle_cover_spawn_radius_max = 1.8
    obstacle_cover_lateral_offset_min = 1.15
    obstacle_cover_lateral_offset_max = 1.7

    obstacle_proximity_penalty = -1.7
    obstacle_warn_distance = 0.82
    obstacle_collision_penalty = -100.0

    prey_cover_reward_scale = 20.0
    prey_cover_seek_reward_scale = 3.0
    prey_cover_progress_reward_scale = 12.0
    prey_cover_pressure_floor = 0.30
    prey_shadow_reward_scale = 32.0
    prey_shadow_progress_reward_scale = 8.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 12.0
    predator_progress_min_gate = 0.40


@configclass
class Uav3v1SurvivalEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Prey survival probe: obstacle field without explicit cover/shadow rewards.

    Cover metrics remain logged, but the prey is only paid for staying alive,
    keeping distance, and avoiding arena/obstacle failures. This checks whether
    cover use emerges because it is useful, not because it is directly rewarded.
    """

    catch_distance = 0.4
    predator_spawn_radius = 2.75

    boundary_warn_fraction = 0.55
    boundary_penalty_scale = 105.0

    obstacle_min_spawn_radius = 1.45
    obstacle_max_spawn_radius = 3.55
    obstacle_min_separation = 1.15
    obstacle_agent_min_spawn_distance = 0.95
    obstacle_cover_spawn_count = 1
    obstacle_cover_spawn_fraction = 0.58
    obstacle_cover_spawn_fraction_noise = 0.12
    obstacle_cover_spawn_angle_noise = 0.40
    obstacle_cover_spawn_radius_min = 1.15
    obstacle_cover_spawn_radius_max = 1.85
    obstacle_cover_lateral_offset_min = 1.10
    obstacle_cover_lateral_offset_max = 1.70

    obstacle_proximity_penalty = -1.5
    obstacle_warn_distance = 0.8
    obstacle_collision_penalty = -85.0

    prey_alive_bonus = 3.0
    prey_evasion_reward_scale = 8.0
    prey_distance_progress_reward_scale = 10.0
    prey_boundary_progress_reward_scale = 18.0
    prey_boundary_progress_start_fraction = 0.45

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 12.0
    predator_progress_min_gate = 0.45


@configclass
class Uav3v1SurvivalEasyEnvCfg(Uav3v1SurvivalEnvCfg):
    """Low-pressure survival feasibility probe.

    This is intentionally easier than the bridge/full survival layouts. It gives
    the prey enough time and control margin to show whether survival behavior is
    learnable before we ramp the predator team back to full strength.
    """

    catch_distance = 0.3
    predator_spawn_radius = 3.6

    boundary_warn_fraction = 0.62
    boundary_penalty_scale = 80.0

    obstacle_min_spawn_radius = 1.8
    obstacle_max_spawn_radius = 3.9
    obstacle_min_separation = 1.2
    obstacle_agent_min_spawn_distance = 1.05
    obstacle_cover_spawn_count = 1
    obstacle_cover_spawn_fraction = 0.58
    obstacle_cover_spawn_fraction_noise = 0.14
    obstacle_cover_spawn_angle_noise = 0.45
    obstacle_cover_spawn_radius_min = 1.5
    obstacle_cover_spawn_radius_max = 2.2
    obstacle_cover_lateral_offset_min = 1.05
    obstacle_cover_lateral_offset_max = 1.65

    obstacle_proximity_penalty = -1.0
    obstacle_warn_distance = 0.75
    obstacle_collision_penalty = -60.0

    predator_progress_reward_scale = 8.0
    predator_progress_min_gate = 0.65


@configclass
class Uav3v1ObstaclesFullObsEnvCfg(Uav3v1ObstaclesEnvCfg):
    """Obstacle task with all-obstacle geometry for each controlled drone.

    Per obstacle feature layout:
    body-frame vector xyz, XY distance, clearance risk, path fraction, path block score.
    The predator agent controls three drones, so it receives one sorted obstacle block per drone.
    """

    obstacle_observation_mode = "full"
    observation_spaces = {"predator": 156, "prey": 58}
    state_space = 214


@configclass
class Uav3v1ObstaclesFullObsEasyEnvCfg(Uav3v1ObstaclesFullObsEnvCfg):
    """Stage A with full obstacle observations."""

    catch_distance = 0.5
    predator_spawn_radius = 2.0

    boundary_warn_fraction = 0.6
    boundary_penalty_scale = 80.0

    obstacle_min_spawn_radius = 2.8
    obstacle_max_spawn_radius = 4.0
    obstacle_min_separation = 1.2
    obstacle_agent_min_spawn_distance = 1.0
    obstacle_cover_spawn_count = 0

    obstacle_proximity_penalty = -1.0
    obstacle_warn_distance = 0.75
    obstacle_collision_penalty = -50.0

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 10.0


@configclass
class Uav3v1ObstaclesFullObsMidEnvCfg(Uav3v1ObstaclesFullObsEnvCfg):
    """Stage B with full obstacle observations."""

    catch_distance = 0.4
    predator_spawn_radius = 2.5

    boundary_warn_fraction = 0.55
    boundary_penalty_scale = 100.0

    obstacle_min_spawn_radius = 1.6
    obstacle_max_spawn_radius = 3.6
    obstacle_min_separation = 1.1
    obstacle_agent_min_spawn_distance = 0.9
    obstacle_cover_spawn_count = 0

    obstacle_proximity_penalty = -1.5
    obstacle_warn_distance = 0.8
    obstacle_collision_penalty = -85.0

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 12.0


@configclass
class Uav3v1ObstaclesFullObsBridgeEnvCfg(Uav3v1ObstaclesFullObsEnvCfg):
    """Stage A/B bridge with full obstacle observations."""

    catch_distance = 0.45
    predator_spawn_radius = 2.25

    boundary_warn_fraction = 0.58
    boundary_penalty_scale = 90.0

    obstacle_min_spawn_radius = 2.1
    obstacle_max_spawn_radius = 3.8
    obstacle_min_separation = 1.15
    obstacle_agent_min_spawn_distance = 0.95
    obstacle_cover_spawn_count = 0

    obstacle_proximity_penalty = -1.2
    obstacle_warn_distance = 0.78
    obstacle_collision_penalty = -65.0

    prey_cover_reward_scale = 0.0
    prey_cover_seek_reward_scale = 0.0
    prey_cover_progress_reward_scale = 0.0
    prey_cover_pressure_floor = 0.0
    prey_shadow_reward_scale = 0.0
    prey_shadow_progress_reward_scale = 0.0
    predator_cover_penalty_scale = 0.0

    predator_progress_reward_scale = 11.0
    predator_progress_min_gate = 0.45


@configclass
class Uav3v1CoverFullObsEnvCfg(Uav3v1ObstaclesFullObsEnvCfg):
    """Stage C with full obstacle observations and prey cover/shadow shaping."""

    predator_cover_penalty_scale = 0.0


@configclass
class Uav3v1CoverBridgeFullObsEnvCfg(Uav3v1CoverBridgeEnvCfg):
    """Stage B/C bridge with full obstacle observations."""

    obstacle_observation_mode = "full"
    observation_spaces = {"predator": 156, "prey": 58}
    state_space = 214


@configclass
class Uav3v1SurvivalFullObsEnvCfg(Uav3v1SurvivalEnvCfg):
    """Prey survival probe with full obstacle observations."""

    obstacle_observation_mode = "full"
    observation_spaces = {"predator": 156, "prey": 58}
    state_space = 214


@configclass
class Uav3v1SurvivalFullObsEasyEnvCfg(Uav3v1SurvivalEasyEnvCfg):
    """Low-pressure survival feasibility probe with full obstacle observations."""

    obstacle_observation_mode = "full"
    observation_spaces = {"predator": 156, "prey": 58}
    state_space = 214
