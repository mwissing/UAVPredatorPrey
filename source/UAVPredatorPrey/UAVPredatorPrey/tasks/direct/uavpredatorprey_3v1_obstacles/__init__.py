# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

def _register_task(task_id: str, env_cfg_name: str, mappo_cfg_name: str = "skrl_mappo_cfg.yaml"):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.uav_3v1_obstacles_env:Uav3v1ObstaclesEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.uav_3v1_obstacles_env_cfg:{env_cfg_name}",
            "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{agents.__name__}:{mappo_cfg_name}",
        },
    )


_register_task("3v1-obstacles-v0", "Uav3v1ObstaclesEnvCfg")
_register_task("3v1-obstacles-easy-v0", "Uav3v1ObstaclesEasyEnvCfg")
_register_task("3v1-obstacles-bridge-v0", "Uav3v1ObstaclesBridgeEnvCfg")
_register_task("3v1-obstacles-mid-v0", "Uav3v1ObstaclesMidEnvCfg")
_register_task("3v1-cover-bridge-v0", "Uav3v1CoverBridgeEnvCfg")
_register_task("3v1-cover-v0", "Uav3v1CoverEnvCfg")
_register_task("3v1-survival-easy-v0", "Uav3v1SurvivalEasyEnvCfg")
_register_task("3v1-survival-v0", "Uav3v1SurvivalEnvCfg")
_register_task(
    "3v1-obstacles-full-easy-v0",
    "Uav3v1ObstaclesFullObsEasyEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
_register_task(
    "3v1-obstacles-full-bridge-v0",
    "Uav3v1ObstaclesFullObsBridgeEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
_register_task(
    "3v1-obstacles-full-mid-v0",
    "Uav3v1ObstaclesFullObsMidEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
_register_task(
    "3v1-cover-bridge-full-v0",
    "Uav3v1CoverBridgeFullObsEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
_register_task(
    "3v1-cover-full-v0",
    "Uav3v1CoverFullObsEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
_register_task(
    "3v1-survival-full-v0",
    "Uav3v1SurvivalFullObsEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
_register_task(
    "3v1-survival-full-easy-v0",
    "Uav3v1SurvivalFullObsEasyEnvCfg",
    "skrl_mappo_full_obs_cfg.yaml",
)
