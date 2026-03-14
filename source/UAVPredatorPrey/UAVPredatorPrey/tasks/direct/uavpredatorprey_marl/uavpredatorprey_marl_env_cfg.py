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
class UavpredatorpreyMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0

    # multi-agent specification
    possible_agents = ["predator", "prey"]
    # Observations per agent (18D):
    #   lin_vel_b(3), ang_vel_b(3), projected_gravity_b(3),
    #   pos_relative_to_origin(3),  <-- NEW: gives height + arena-position awareness
    #   rel_pos_to_opponent_b(3), rel_vel_to_opponent_w(3)
    action_spaces = {"predator": 4, "prey": 4}
    observation_spaces = {"predator": 18, "prey": 18}
    state_space = 36  # concatenation of both agents' observations for MAPPO

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=True
    )

    # robots - predator is red, prey is blue
    predator_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Predator",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1))
        ),
    )
    prey_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Prey",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0))
        ),
    )

    # physics parameters
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # === REWARD HIERARCHY ===
    # Priority 1: Learn to fly (dominant - without this, everything else is noise)
    upright_reward_scale = 2.0              # reward for staying level
    target_height = 1.0                     # desired hover altitude [m]
    height_penalty_scale = -3.0             # penalty for deviating from target_height

    # Priority 2: Stay in arena
    boundary_warn_fraction = 0.6            # soft penalty starts at 60% of arena radius
    boundary_penalty_scale = 50.0           # per meter outside warn zone

    # Priority 3: Predator-prey task
    # Predator: bounded proximity reward (same as working single-agent env, NOT delta_dist)
    predator_proximity_reward_scale = 8.0   # reward for being close to prey
    predator_catch_bonus = 200.0            # one-time bonus on actual collision
    # Prey: NO retreat reward (this caused fly-to-boundary). Only alive bonus + caught penalty.
    prey_alive_bonus = 2.0                  # per second - prey wants to survive, not flee
    prey_caught_penalty = -200.0            # one-time penalty on caught

    # Shared velocity penalties (dampen wild movements)
    lin_vel_penalty = -0.05
    ang_vel_penalty = -0.01

    # OOB penalty: applied on termination step when drone crashes/leaves arena.
    # Must be >= caught_penalty so prey can't exploit crashing to avoid being caught.
    oob_penalty = -200.0

    # catch/termination parameters
    catch_distance = 0.3                    # slightly larger for easier early learning
    arena_radius = 5.0                      # max horizontal distance from origin [m]
    min_height = 0.1                        # crash threshold [m]
    max_height = 3.0                        # ceiling threshold [m]

    # initial spawn parameters
    predator_spawn_pos = [0.0, -1.0, 1.0]  # closer together for faster learning
    prey_spawn_pos = [0.0, 1.0, 1.0]
    spawn_pos_noise = 0.3                   # less noise for more consistent starts
