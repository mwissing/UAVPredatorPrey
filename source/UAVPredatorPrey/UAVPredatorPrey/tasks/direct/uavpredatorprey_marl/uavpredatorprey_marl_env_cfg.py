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
    action_spaces = {"predator": 4, "prey": 4}
    observation_spaces = {"predator": 15, "prey": 15}
    state_space = 30  # concatenation of both agents' observations for MAPPO value function

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

    # reward scales
    # predator: rewarded for getting close, bonus for catching
    predator_distance_reward_scale = 10.0   # reward for proximity
    predator_catch_bonus = 50.0             # one-time bonus on catch
    predator_lin_vel_penalty = -0.02        # small penalty for erratic flight
    predator_ang_vel_penalty = -0.005

    # prey: rewarded for staying far, penalty for being caught
    prey_distance_reward_scale = 5.0        # reward for distance
    prey_caught_penalty = -50.0             # one-time penalty on caught
    prey_lin_vel_penalty = -0.02
    prey_ang_vel_penalty = -0.005
    prey_alive_bonus = 0.5                  # bonus per step for surviving

    # catch/termination parameters
    catch_distance = 0.2                    # distance threshold for "catch" [m]
    arena_radius = 5.0                      # max horizontal distance from origin [m]
    min_height = 0.1                        # crash threshold [m]
    max_height = 3.0                        # ceiling threshold [m]

    # initial spawn parameters
    predator_spawn_pos = [0.0, -1.5, 1.0]  # predator starts at one side
    prey_spawn_pos = [0.0, 1.5, 1.0]       # prey starts at other side
    spawn_pos_noise = 0.5                   # random noise on spawn position [m]
