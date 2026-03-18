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

    # multi-agent specification (2 agents: shared predator policy + prey)
    possible_agents = ["predator", "prey"]
    # Predator obs (72D): 3 × 24D stacked (own_state(12) + prey_info(6) + 2_teammate_pos(6)) per drone
    # Prey obs (30D): own_state(12) + 3_predator_info(18)
    action_spaces = {"predator": 12, "prey": 4}          # 3 × 4D for predator team
    observation_spaces = {"predator": 72, "prey": 30}     # 3 × 24D stacked
    state_space = 102  # 72 + 30 = concatenation of all observations for MAPPO

    # simulation — optimized for RTX 5090
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,                          # 50Hz physics (was 100Hz) — halves physics compute
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
        num_envs=4096, env_spacing=6.0, replicate_physics=True, clone_in_fabric=True
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
    predator_catch_bonus = 200.0           # only the catcher gets this
    predator_assist_bonus = 50.0           # teammates within assist_distance also rewarded
    assist_distance = 1.5                  # [m] must be this close to get assist reward
    prey_alive_bonus = 2.0
    prey_caught_penalty = -200.0

    # Action/velocity penalties (prevent wild oscillations)
    lin_vel_penalty = -0.05
    ang_vel_penalty = -0.01
    action_penalty = -0.05                 # NEW: penalizes large actions

    # OOB penalty
    oob_penalty = -200.0

    # catch/termination parameters
    catch_distance = 0.3
    arena_radius = 5.0
    min_height = 0.1
    max_height = 3.0

    # spawn parameters
    predator_spawn_radius = 2.0  # predators spawn on a circle around center
    prey_spawn_pos = [0.0, 0.0, 1.0]  # prey spawns at center
    spawn_pos_noise = 0.3
