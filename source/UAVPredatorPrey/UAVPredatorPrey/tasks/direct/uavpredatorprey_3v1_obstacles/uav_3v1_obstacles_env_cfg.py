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

    catch_distance = 0.3                     # curriculum stage 3: final tighter catch radius after stable 0.4 training

    # Boundary: stronger + earlier warning to reduce OOB rate
    boundary_warn_fraction = 0.5          # warning starts at 2.5m instead of 3.0m (parent: 0.6)
    boundary_penalty_scale = 100.0        # doubled from parent (50.0) — stronger brake against chase overshoot

    # --- Obstacle configuration ---
    num_obstacles = 4                     # full obstacle count
    obstacle_radius = 0.3                 # [m] cylinder radius
    obstacle_height = 3.0                 # [m] cylinder height (floor to max_height)

    # Obstacles spawn randomly in an annular region
    obstacle_min_spawn_radius = 2.8       # [m] inner boundary (> predator_spawn_radius + noise + margin)
    obstacle_max_spawn_radius = 4.0       # [m] outer boundary (keep inside arena)
    obstacle_min_separation = 1.0         # [m] min distance between obstacle centers (tighter for 4)

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
    # Extra configs kept for future use (not instantiated when num_obstacles=2)
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

    # Reward: obstacle penalties (minimal — let agents learn prey evasion dynamics first)
    obstacle_proximity_penalty = -1.0     # minimal gradient
    obstacle_warn_distance = 0.7          # tight warning zone
    obstacle_collision_penalty = -5.0     # almost negligible
    obstacle_collision_distance = 0.5     # [m] ~obstacle_radius + drone_radius + margin
