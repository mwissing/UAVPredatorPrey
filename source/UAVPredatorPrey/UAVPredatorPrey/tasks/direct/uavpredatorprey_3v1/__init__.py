# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="3v1-empty-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav3v1EnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
        "skrl_mappo_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_cfg.yaml",
        "skrl_mappo_attention_critic_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_cfg.yaml",
        "skrl_mappo_attention_critic_prey_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_prey_attention_cfg.yaml",
    },
)

gym.register(
    id="1v1-survival-easy-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav1v1SurvivalEasyEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
    },
)

gym.register(
    id="1v1-survival-soft-oob-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav1v1SurvivalSoftOobEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
    },
)

gym.register(
    id="2v1-survival-soft-oob-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav2v1SurvivalSoftOobEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
        "skrl_mappo_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_cfg.yaml",
        "skrl_mappo_attention_critic_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_cfg.yaml",
        "skrl_mappo_attention_critic_prey_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_prey_attention_cfg.yaml",
    },
)

gym.register(
    id="2v1-survival-soft-oob-mixed-spawn-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav2v1SurvivalSoftOobMixedSpawnEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
        "skrl_mappo_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_cfg.yaml",
        "skrl_mappo_attention_critic_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_cfg.yaml",
        "skrl_mappo_attention_critic_prey_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_prey_attention_cfg.yaml",
    },
)

gym.register(
    id="3v1-survival-soft-oob-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav3v1SurvivalSoftOobEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
        "skrl_mappo_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_cfg.yaml",
        "skrl_mappo_attention_critic_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_cfg.yaml",
        "skrl_mappo_attention_critic_prey_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_prey_attention_cfg.yaml",
    },
)

gym.register(
    id="3v1-survival-soft-oob-teammate-vel-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav3v1SurvivalSoftOobTeammateVelEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
        "skrl_mappo_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_cfg.yaml",
        "skrl_mappo_attention_critic_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_cfg.yaml",
        "skrl_mappo_attention_critic_prey_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_prey_attention_cfg.yaml",
    },
)

gym.register(
    id="3v1-survival-soft-oob-teammate-vel-random-spawn-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav3v1SurvivalSoftOobTeammateVelRandomSpawnEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
        "skrl_mappo_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_cfg.yaml",
        "skrl_mappo_attention_critic_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_cfg.yaml",
        "skrl_mappo_attention_critic_prey_attention_cfg_entry_point": f"{agents.__name__}:skrl_mappo_attention_critic_prey_attention_cfg.yaml",
    },
)

gym.register(
    id="1v1-survival-soft-oob-pred22-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav1v1SurvivalSoftOobPred22EnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
    },
)

gym.register(
    id="1v1-survival-soft-oob-pred24-v0",
    entry_point=f"{__name__}.uav_3v1_env:Uav3v1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_3v1_env_cfg:Uav1v1SurvivalSoftOobPred24EnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mappo_finetune_cfg_entry_point": f"{agents.__name__}:skrl_mappo_finetune_cfg.yaml",
    },
)
