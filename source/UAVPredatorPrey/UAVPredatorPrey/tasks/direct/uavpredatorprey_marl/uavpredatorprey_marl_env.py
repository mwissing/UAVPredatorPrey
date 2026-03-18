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

        # Episode-level tracking for logging
        self._episode_catches = torch.zeros(self.num_envs, device=self.device)
        self._episode_predator_oob = torch.zeros(self.num_envs, device=self.device)
        self._episode_prey_oob = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward_sums = {
            "predator_proximity": torch.zeros(self.num_envs, device=self.device),
            "predator_upright": torch.zeros(self.num_envs, device=self.device),
            "predator_height": torch.zeros(self.num_envs, device=self.device),
            "prey_alive": torch.zeros(self.num_envs, device=self.device),
            "prey_upright": torch.zeros(self.num_envs, device=self.device),
            "prey_height": torch.zeros(self.num_envs, device=self.device),
        }

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
        self._predator_actions = actions["predator"].clone().clamp(-1.0, 1.0)
        self._prey_actions = actions["prey"].clone().clamp(-1.0, 1.0)

        # Thrust: action[0] in [-1,1] maps to [0, thrust_to_weight * weight]
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
        # Position relative to env origin (gives height + arena position awareness)
        predator_pos_rel = self._predator.data.root_pos_w - self._terrain.env_origins
        prey_pos_rel = self._prey.data.root_pos_w - self._terrain.env_origins

        # Relative position/velocity to opponent in body frame
        pred_to_prey_pos_b, _ = subtract_frame_transforms(
            self._predator.data.root_pos_w, self._predator.data.root_quat_w, self._prey.data.root_pos_w
        )
        pred_to_prey_vel_w = self._prey.data.root_lin_vel_w - self._predator.data.root_lin_vel_w

        prey_to_pred_pos_b, _ = subtract_frame_transforms(
            self._prey.data.root_pos_w, self._prey.data.root_quat_w, self._predator.data.root_pos_w
        )
        prey_to_pred_vel_w = self._predator.data.root_lin_vel_w - self._prey.data.root_lin_vel_w

        # 18D per agent:
        # lin_vel_b(3), ang_vel_b(3), projected_gravity(3), pos_rel_origin(3),
        # rel_pos_opponent_b(3), rel_vel_opponent_w(3)
        predator_obs = torch.cat(
            [
                self._predator.data.root_lin_vel_b,
                self._predator.data.root_ang_vel_b,
                self._predator.data.projected_gravity_b,
                predator_pos_rel,
                pred_to_prey_pos_b,
                pred_to_prey_vel_w,
            ],
            dim=-1,
        )
        prey_obs = torch.cat(
            [
                self._prey.data.root_lin_vel_b,
                self._prey.data.root_ang_vel_b,
                self._prey.data.projected_gravity_b,
                prey_pos_rel,
                prey_to_pred_pos_b,
                prey_to_pred_vel_w,
            ],
            dim=-1,
        )
        return {"predator": predator_obs, "prey": prey_obs}

    def _get_states(self) -> torch.Tensor | None:
        obs = self._get_observations()
        return torch.cat([obs["predator"], obs["prey"]], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Positions relative to env origin
        predator_pos_rel = self._predator.data.root_pos_w - self._terrain.env_origins
        prey_pos_rel = self._prey.data.root_pos_w - self._terrain.env_origins

        # Distance between drones
        distance = torch.linalg.norm(
            self._predator.data.root_pos_w - self._prey.data.root_pos_w, dim=1
        )

        rewards, components = compute_rewards(
            distance=distance,
            predator_height=predator_pos_rel[:, 2],
            prey_height=prey_pos_rel[:, 2],
            predator_gravity_b_z=self._predator.data.projected_gravity_b[:, 2],
            prey_gravity_b_z=self._prey.data.projected_gravity_b[:, 2],
            predator_lin_vel_b=self._predator.data.root_lin_vel_b,
            predator_ang_vel_b=self._predator.data.root_ang_vel_b,
            prey_lin_vel_b=self._prey.data.root_lin_vel_b,
            prey_ang_vel_b=self._prey.data.root_ang_vel_b,
            catch_distance=self.cfg.catch_distance,
            target_height=self.cfg.target_height,
            upright_scale=self.cfg.upright_reward_scale,
            height_penalty_scale=self.cfg.height_penalty_scale,
            predator_proximity_scale=self.cfg.predator_proximity_reward_scale,
            predator_catch_bonus=self.cfg.predator_catch_bonus,
            prey_alive_bonus=self.cfg.prey_alive_bonus,
            prey_caught_penalty=self.cfg.prey_caught_penalty,
            lin_vel_penalty=self.cfg.lin_vel_penalty,
            ang_vel_penalty=self.cfg.ang_vel_penalty,
            step_dt=self.step_dt,
        )

        # --- Soft boundary penalty (continuous) ---
        warn_radius = self.cfg.arena_radius * self.cfg.boundary_warn_fraction
        predator_horiz = torch.linalg.norm(predator_pos_rel[:, :2], dim=1)
        prey_horiz = torch.linalg.norm(prey_pos_rel[:, :2], dim=1)
        predator_boundary = torch.clamp(predator_horiz - warn_radius, min=0.0) * self.cfg.boundary_penalty_scale * self.step_dt
        prey_boundary = torch.clamp(prey_horiz - warn_radius, min=0.0) * self.cfg.boundary_penalty_scale * self.step_dt
        rewards["predator"] = rewards["predator"] - predator_boundary
        rewards["prey"] = rewards["prey"] - prey_boundary

        # --- OOB crash penalty (one-time, on termination step) ---
        predator_oob = (
            (predator_pos_rel[:, 2] < self.cfg.min_height)
            | (predator_pos_rel[:, 2] > self.cfg.max_height)
            | (predator_horiz > self.cfg.arena_radius)
        )
        prey_oob = (
            (prey_pos_rel[:, 2] < self.cfg.min_height)
            | (prey_pos_rel[:, 2] > self.cfg.max_height)
            | (prey_horiz > self.cfg.arena_radius)
        )
        rewards["predator"] = rewards["predator"] + predator_oob.float() * self.cfg.oob_penalty
        rewards["prey"] = rewards["prey"] + prey_oob.float() * self.cfg.oob_penalty

        # --- Episode-level tracking ---
        caught = distance < self.cfg.catch_distance
        self._episode_catches += caught.float()
        self._episode_predator_oob += predator_oob.float()
        self._episode_prey_oob += prey_oob.float()
        # Accumulate reward components for logging
        self._episode_reward_sums["predator_proximity"] += components["predator_proximity"]
        self._episode_reward_sums["predator_upright"] += components["predator_upright"]
        self._episode_reward_sums["predator_height"] += components["predator_height"]
        self._episode_reward_sums["prey_alive"] += components["prey_alive"]
        self._episode_reward_sums["prey_upright"] += components["prey_upright"]
        self._episode_reward_sums["prey_height"] += components["prey_height"]

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        distance = torch.linalg.norm(
            self._predator.data.root_pos_w - self._prey.data.root_pos_w, dim=1
        )
        caught = distance < self.cfg.catch_distance

        predator_pos_rel = self._predator.data.root_pos_w - self._terrain.env_origins
        predator_oob = (predator_pos_rel[:, 2] < self.cfg.min_height) | (predator_pos_rel[:, 2] > self.cfg.max_height)
        predator_oob = predator_oob | (torch.linalg.norm(predator_pos_rel[:, :2], dim=1) > self.cfg.arena_radius)

        prey_pos_rel = self._prey.data.root_pos_w - self._terrain.env_origins
        prey_oob = (prey_pos_rel[:, 2] < self.cfg.min_height) | (prey_pos_rel[:, 2] > self.cfg.max_height)
        prey_oob = prey_oob | (torch.linalg.norm(prey_pos_rel[:, :2], dim=1) > self.cfg.arena_radius)

        terminated = caught | predator_oob | prey_oob

        terminated_dict = {agent: terminated for agent in self.cfg.possible_agents}
        time_outs_dict = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated_dict, time_outs_dict

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._predator._ALL_INDICES
        super()._reset_idx(env_ids)

        # Log metrics — SKRL requires torch.Tensor with numel()==1 for TensorBoard logging
        if len(env_ids) > 0:
            if "log" not in self.extras:
                self.extras["log"] = {}

            ep_len = self.episode_length_buf[env_ids].float().mean()
            final_dist = torch.linalg.norm(
                self._predator.data.root_pos_w[env_ids] - self._prey.data.root_pos_w[env_ids], dim=1
            ).mean()

            self.extras["log"]["Metrics/catch_rate"] = self._episode_catches[env_ids].mean()
            self.extras["log"]["Metrics/predator_oob_rate"] = self._episode_predator_oob[env_ids].mean()
            self.extras["log"]["Metrics/prey_oob_rate"] = self._episode_prey_oob[env_ids].mean()
            self.extras["log"]["Metrics/episode_length"] = ep_len
            self.extras["log"]["Metrics/final_distance"] = final_dist

            for key, buf in self._episode_reward_sums.items():
                self.extras["log"][f"Reward/{key}"] = buf[env_ids].mean()
                buf[env_ids] = 0.0

            self._episode_catches[env_ids] = 0.0
            self._episode_predator_oob[env_ids] = 0.0
            self._episode_prey_oob[env_ids] = 0.0

        # Reset actions
        self._predator_actions[env_ids] = 0.0
        self._prey_actions[env_ids] = 0.0

        # Spawn predator
        predator_default_state = self._predator.data.default_root_state[env_ids].clone()
        predator_default_state[:, 0] += self.cfg.predator_spawn_pos[0]
        predator_default_state[:, 1] += self.cfg.predator_spawn_pos[1]
        predator_default_state[:, 2] = self.cfg.predator_spawn_pos[2]
        predator_default_state[:, :3] += (
            torch.rand(len(env_ids), 3, device=self.device) * 2 - 1
        ) * self.cfg.spawn_pos_noise
        predator_default_state[:, :3] += self._terrain.env_origins[env_ids]
        self._predator.write_root_pose_to_sim(predator_default_state[:, :7], env_ids)
        self._predator.write_root_velocity_to_sim(predator_default_state[:, 7:], env_ids)
        self._predator.write_joint_state_to_sim(
            self._predator.data.default_joint_pos[env_ids],
            self._predator.data.default_joint_vel[env_ids],
            None, env_ids,
        )

        # Spawn prey
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
        self._prey.write_joint_state_to_sim(
            self._prey.data.default_joint_pos[env_ids],
            self._prey.data.default_joint_vel[env_ids],
            None, env_ids,
        )


@torch.jit.script
def compute_rewards(
    distance: torch.Tensor,
    predator_height: torch.Tensor,
    prey_height: torch.Tensor,
    predator_gravity_b_z: torch.Tensor,
    prey_gravity_b_z: torch.Tensor,
    predator_lin_vel_b: torch.Tensor,
    predator_ang_vel_b: torch.Tensor,
    prey_lin_vel_b: torch.Tensor,
    prey_ang_vel_b: torch.Tensor,
    catch_distance: float,
    target_height: float,
    upright_scale: float,
    height_penalty_scale: float,
    predator_proximity_scale: float,
    predator_catch_bonus: float,
    prey_alive_bonus: float,
    prey_caught_penalty: float,
    lin_vel_penalty: float,
    ang_vel_penalty: float,
    step_dt: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    caught = (distance < catch_distance).float()

    # === PRIORITY 1: Flight stability (both agents, dominant) ===
    # Upright: projected_gravity_b[:,2] ≈ -1 when level → (-gz) ≈ +1 → positive reward
    predator_upright = (-predator_gravity_b_z) * upright_scale * step_dt
    prey_upright = (-prey_gravity_b_z) * upright_scale * step_dt

    # Height: penalize deviation from target hover altitude
    predator_height_pen = torch.square(predator_height - target_height) * height_penalty_scale * step_dt
    prey_height_pen = torch.square(prey_height - target_height) * height_penalty_scale * step_dt

    # Velocity: dampen wild movements
    predator_lin_vel = torch.sum(torch.square(predator_lin_vel_b), dim=1) * lin_vel_penalty * step_dt
    predator_ang_vel = torch.sum(torch.square(predator_ang_vel_b), dim=1) * ang_vel_penalty * step_dt
    prey_lin_vel = torch.sum(torch.square(prey_lin_vel_b), dim=1) * lin_vel_penalty * step_dt
    prey_ang_vel = torch.sum(torch.square(prey_ang_vel_b), dim=1) * ang_vel_penalty * step_dt

    # === PRIORITY 3: Predator-prey task ===
    # Predator: bounded proximity (same as working single-agent env)
    predator_proximity = (1.0 - torch.tanh(distance / 2.0)) * predator_proximity_scale * step_dt
    predator_catch = caught * predator_catch_bonus

    # Prey: survive (bonus) + don't get caught (penalty). No retreat reward.
    prey_alive = prey_alive_bonus * step_dt
    prey_catch_pen = caught * prey_caught_penalty

    # === Combine ===
    predator_reward = (
        predator_upright + predator_height_pen + predator_lin_vel + predator_ang_vel
        + predator_proximity + predator_catch
    )
    prey_reward = (
        prey_upright + prey_height_pen + prey_lin_vel + prey_ang_vel
        + prey_alive + prey_catch_pen
    )

    rewards = {"predator": predator_reward, "prey": prey_reward}

    # Individual components for TensorBoard logging
    components = {
        "predator_proximity": predator_proximity,
        "predator_upright": predator_upright,
        "predator_height": predator_height_pen,
        "prey_alive": torch.full_like(prey_reward, prey_alive_bonus * step_dt),
        "prey_upright": prey_upright,
        "prey_height": prey_height_pen,
    }

    return rewards, components
