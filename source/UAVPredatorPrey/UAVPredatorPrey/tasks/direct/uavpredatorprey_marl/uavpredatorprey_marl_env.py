# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import subtract_frame_transforms

from .uavpredatorprey_marl_env_cfg import UavpredatorpreyMarlEnvCfg


class UavpredatorpreyMarlEnv(DirectMARLEnv):
    cfg: UavpredatorpreyMarlEnvCfg

    def __init__(self, cfg: UavpredatorpreyMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions and forces for both drones
        self._predator_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prey_actions = torch.zeros(self.num_envs, 4, device=self.device)

        self._predator_thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._predator_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prey_thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prey_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Get body indices for force application
        self._predator_body_id = self._predator.find_bodies("body")[0]
        self._prey_body_id = self._prey.find_bodies("body")[0]

        # Get robot mass for thrust calculation (both are the same model)
        self._robot_mass = self._predator.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    def _setup_scene(self):
        self._predator = Articulation(self.cfg.predator_cfg)
        self._prey = Articulation(self.cfg.prey_cfg)

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Register articulations
        self.scene.articulations["predator"] = self._predator
        self.scene.articulations["prey"] = self._prey

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Clamp actions to [-1, 1]
        self._predator_actions = actions["predator"].clone().clamp(-1.0, 1.0)
        self._prey_actions = actions["prey"].clone().clamp(-1.0, 1.0)

        # Convert actions to thrust and moments
        # Thrust: action[0] maps from [-1,1] to [0, thrust_to_weight * weight]
        self._predator_thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._predator_actions[:, 0] + 1.0) / 2.0
        )
        self._predator_moment[:, 0, :] = self.cfg.moment_scale * self._predator_actions[:, 1:]

        self._prey_thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._prey_actions[:, 0] + 1.0) / 2.0
        )
        self._prey_moment[:, 0, :] = self.cfg.moment_scale * self._prey_actions[:, 1:]

    def _apply_action(self) -> None:
        self._predator.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._predator_body_id, forces=self._predator_thrust, torques=self._predator_moment
        )
        self._prey.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._prey_body_id, forces=self._prey_thrust, torques=self._prey_moment
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Relative position/velocity from predator to prey (in predator body frame)
        pred_to_prey_pos_b, _ = subtract_frame_transforms(
            self._predator.data.root_pos_w, self._predator.data.root_quat_w, self._prey.data.root_pos_w
        )
        pred_to_prey_vel_w = self._prey.data.root_lin_vel_w - self._predator.data.root_lin_vel_w

        # Relative position/velocity from prey to predator (in prey body frame)
        prey_to_pred_pos_b, _ = subtract_frame_transforms(
            self._prey.data.root_pos_w, self._prey.data.root_quat_w, self._predator.data.root_pos_w
        )
        prey_to_pred_vel_w = self._predator.data.root_lin_vel_w - self._prey.data.root_lin_vel_w

        # Predator observation: own state + relative info about prey
        # [lin_vel_b(3), ang_vel_b(3), projected_gravity(3), rel_pos_to_prey(3), rel_vel_to_prey(3)] = 15
        predator_obs = torch.cat(
            [
                self._predator.data.root_lin_vel_b,
                self._predator.data.root_ang_vel_b,
                self._predator.data.projected_gravity_b,
                pred_to_prey_pos_b,
                pred_to_prey_vel_w,
            ],
            dim=-1,
        )

        # Prey observation: own state + relative info about predator
        prey_obs = torch.cat(
            [
                self._prey.data.root_lin_vel_b,
                self._prey.data.root_ang_vel_b,
                self._prey.data.projected_gravity_b,
                prey_to_pred_pos_b,
                prey_to_pred_vel_w,
            ],
            dim=-1,
        )

        return {"predator": predator_obs, "prey": prey_obs}

    def _get_states(self) -> torch.Tensor | None:
        # Global state for MAPPO: concatenation of both agents' observations
        obs = self._get_observations()
        return torch.cat([obs["predator"], obs["prey"]], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        return compute_rewards(
            predator_pos_w=self._predator.data.root_pos_w,
            prey_pos_w=self._prey.data.root_pos_w,
            predator_lin_vel_b=self._predator.data.root_lin_vel_b,
            predator_ang_vel_b=self._predator.data.root_ang_vel_b,
            prey_lin_vel_b=self._prey.data.root_lin_vel_b,
            prey_ang_vel_b=self._prey.data.root_ang_vel_b,
            catch_distance=self.cfg.catch_distance,
            predator_distance_scale=self.cfg.predator_distance_reward_scale,
            predator_catch_bonus=self.cfg.predator_catch_bonus,
            predator_lin_vel_penalty=self.cfg.predator_lin_vel_penalty,
            predator_ang_vel_penalty=self.cfg.predator_ang_vel_penalty,
            prey_distance_scale=self.cfg.prey_distance_reward_scale,
            prey_caught_penalty=self.cfg.prey_caught_penalty,
            prey_lin_vel_penalty=self.cfg.prey_lin_vel_penalty,
            prey_ang_vel_penalty=self.cfg.prey_ang_vel_penalty,
            prey_alive_bonus=self.cfg.prey_alive_bonus,
            step_dt=self.step_dt,
        )

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Distance between drones
        distance = torch.linalg.norm(
            self._predator.data.root_pos_w - self._prey.data.root_pos_w, dim=1
        )
        caught = distance < self.cfg.catch_distance

        # Out of bounds checks for predator
        predator_pos = self._predator.data.root_pos_w
        predator_oob = (
            (predator_pos[:, 2] < self.cfg.min_height)
            | (predator_pos[:, 2] > self.cfg.max_height)
        )
        # Check horizontal distance from env origin
        predator_horizontal = torch.linalg.norm(
            predator_pos[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        predator_oob = predator_oob | (predator_horizontal > self.cfg.arena_radius)

        # Out of bounds checks for prey
        prey_pos = self._prey.data.root_pos_w
        prey_oob = (
            (prey_pos[:, 2] < self.cfg.min_height)
            | (prey_pos[:, 2] > self.cfg.max_height)
        )
        prey_horizontal = torch.linalg.norm(
            prey_pos[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        prey_oob = prey_oob | (prey_horizontal > self.cfg.arena_radius)

        # Both agents terminate together on catch or if either goes OOB
        terminated = caught | predator_oob | prey_oob

        terminated_dict = {agent: terminated for agent in self.cfg.possible_agents}
        time_outs_dict = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated_dict, time_outs_dict

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._predator._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset actions
        self._predator_actions[env_ids] = 0.0
        self._prey_actions[env_ids] = 0.0

        # Spawn predator with noise
        predator_default_state = self._predator.data.default_root_state[env_ids].clone()
        predator_default_state[:, 0] += self.cfg.predator_spawn_pos[0]
        predator_default_state[:, 1] += self.cfg.predator_spawn_pos[1]
        predator_default_state[:, 2] = self.cfg.predator_spawn_pos[2]
        # Add position noise
        predator_default_state[:, :3] += (
            torch.rand(len(env_ids), 3, device=self.device) * 2 - 1
        ) * self.cfg.spawn_pos_noise
        # Add env origins
        predator_default_state[:, :3] += self._terrain.env_origins[env_ids]

        self._predator.write_root_pose_to_sim(predator_default_state[:, :7], env_ids)
        self._predator.write_root_velocity_to_sim(predator_default_state[:, 7:], env_ids)
        predator_joint_pos = self._predator.data.default_joint_pos[env_ids]
        predator_joint_vel = self._predator.data.default_joint_vel[env_ids]
        self._predator.write_joint_state_to_sim(predator_joint_pos, predator_joint_vel, None, env_ids)

        # Spawn prey with noise
        prey_default_state = self._prey.data.default_root_state[env_ids].clone()
        prey_default_state[:, 0] += self.cfg.prey_spawn_pos[0]
        prey_default_state[:, 1] += self.cfg.prey_spawn_pos[1]
        prey_default_state[:, 2] = self.cfg.prey_spawn_pos[2]
        prey_default_state[:, :3] += (
            torch.rand(len(env_ids), 3, device=self.device) * 2 - 1
        ) * self.cfg.spawn_pos_noise
        prey_default_state[:, :3] += self._terrain.env_origins[env_ids]

        self._prey.write_root_pose_to_sim(prey_default_state[:, :7], env_ids)
        self._prey.write_root_velocity_to_sim(prey_default_state[:, 7:], env_ids)
        prey_joint_pos = self._prey.data.default_joint_pos[env_ids]
        prey_joint_vel = self._prey.data.default_joint_vel[env_ids]
        self._prey.write_joint_state_to_sim(prey_joint_pos, prey_joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    predator_pos_w: torch.Tensor,
    prey_pos_w: torch.Tensor,
    predator_lin_vel_b: torch.Tensor,
    predator_ang_vel_b: torch.Tensor,
    prey_lin_vel_b: torch.Tensor,
    prey_ang_vel_b: torch.Tensor,
    catch_distance: float,
    predator_distance_scale: float,
    predator_catch_bonus: float,
    predator_lin_vel_penalty: float,
    predator_ang_vel_penalty: float,
    prey_distance_scale: float,
    prey_caught_penalty: float,
    prey_lin_vel_penalty: float,
    prey_ang_vel_penalty: float,
    prey_alive_bonus: float,
    step_dt: float,
) -> dict[str, torch.Tensor]:
    # Distance between the two drones
    distance = torch.linalg.norm(predator_pos_w - prey_pos_w, dim=1)
    caught = (distance < catch_distance).float()

    # === Predator rewards ===
    # Reward for getting close (higher when closer)
    predator_proximity = (1.0 - torch.tanh(distance / 1.5)) * predator_distance_scale * step_dt
    # Bonus for catching
    predator_catch = caught * predator_catch_bonus
    # Velocity penalties for smooth flight
    predator_lin_vel = torch.sum(torch.square(predator_lin_vel_b), dim=1) * predator_lin_vel_penalty * step_dt
    predator_ang_vel = torch.sum(torch.square(predator_ang_vel_b), dim=1) * predator_ang_vel_penalty * step_dt

    predator_reward = predator_proximity + predator_catch + predator_lin_vel + predator_ang_vel

    # === Prey rewards ===
    # Reward for staying far (higher when farther)
    prey_distance = torch.tanh(distance / 2.0) * prey_distance_scale * step_dt
    # Penalty for being caught
    prey_caught = caught * prey_caught_penalty
    # Alive bonus (encourages survival)
    prey_alive = prey_alive_bonus * step_dt
    # Velocity penalties
    prey_lin_vel = torch.sum(torch.square(prey_lin_vel_b), dim=1) * prey_lin_vel_penalty * step_dt
    prey_ang_vel = torch.sum(torch.square(prey_ang_vel_b), dim=1) * prey_ang_vel_penalty * step_dt

    prey_reward = prey_distance + prey_caught + prey_alive + prey_lin_vel + prey_ang_vel

    return {"predator": predator_reward, "prey": prey_reward}
