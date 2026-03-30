# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import subtract_frame_transforms

from .uav_3v1_env_cfg import Uav3v1EnvCfg


class Uav3v1Env(DirectMARLEnv):
    """3v1 Predator-Prey — fully vectorized, no per-predator Python loops in hot path.

    2 MARL agents: 'predator' (shared policy, 72D obs, 12D action for 3 drones)
    and 'prey' (30D obs, 4D action for 1 drone).
    """

    cfg: Uav3v1EnvCfg

    def __init__(self, cfg: Uav3v1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        P = cfg.num_predators
        self._P = P

        # Predator forces — batched tensors instead of lists
        self._pred_actions = torch.zeros(N, P, 4, device=self.device)
        self._pred_thrust = torch.zeros(N, P, 1, 3, device=self.device)
        self._pred_moment = torch.zeros(N, P, 1, 3, device=self.device)

        # Prey forces
        self._prey_actions = torch.zeros(N, 4, device=self.device)
        self._prey_thrust = torch.zeros(N, 1, 3, device=self.device)
        self._prey_moment = torch.zeros(N, 1, 3, device=self.device)

        # Body IDs
        self._pred_body_ids = [pred.find_bodies("body")[0] for pred in self._predators]
        self._prey_body_id = self._prey.find_bodies("body")[0]

        # Mass
        self._robot_mass = self._predators[0].root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Episode tracking
        self._episode_catches = torch.zeros(N, device=self.device)
        self._episode_pred_oob = torch.zeros(N, device=self.device)
        self._episode_prey_oob = torch.zeros(N, device=self.device)
        self._episode_min_distance = torch.full((N,), 100.0, device=self.device)

        # Spawn angles (120° apart)
        self._spawn_angles = [i * 2 * math.pi / P for i in range(P)]

        # Pre-allocated cached tensors (reused every step via in-place ops)
        self._pred_pos_rel = torch.zeros(N, P, 3, device=self.device)
        self._pred_horiz = torch.zeros(N, P, device=self.device)
        self._pred_oob = torch.zeros(N, P, dtype=torch.bool, device=self.device)
        self._prey_pos_rel = torch.zeros(N, 3, device=self.device)
        self._prey_horiz = torch.zeros(N, device=self.device)
        self._prey_oob = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._current_distances = torch.zeros(N, P, device=self.device)
        self._caught = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._has_nan = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Pre-build teammate index pairs for vectorized teammate obs
        # For pred i, teammates are the other 2 predators
        # teammate_src[k] = which predator is the "observer", teammate_dst[k] = which is observed
        src, dst = [], []
        for i in range(P):
            for j in range(P):
                if j != i:
                    src.append(i)
                    dst.append(j)
        self._teammate_src = src  # [0,0, 1,1, 2,2]
        self._teammate_dst = dst  # [1,2, 0,2, 0,1]

    def _setup_scene(self):
        self._predators: list[Articulation] = []
        for i in range(self.cfg.num_predators):
            cfg_attr = getattr(self.cfg, f"predator_{i}_cfg")
            self._predators.append(Articulation(cfg_attr))

        self._prey = Articulation(self.cfg.prey_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        for i, pred in enumerate(self._predators):
            self.scene.articulations[f"predator_{i}"] = pred
        self.scene.articulations["prey"] = self._prey

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Predator: reshape 12D → (N, 3, 4), compute thrust/moment vectorized
        self._pred_actions[:] = actions["predator"].clamp(-1.0, 1.0).view(self.num_envs, self._P, 4)
        self._pred_thrust[:, :, 0, 2] = (
            self.cfg.predator_thrust_to_weight * self._robot_weight * (self._pred_actions[:, :, 0] + 1.0) / 2.0
        )
        self._pred_moment[:, :, 0, :] = self.cfg.moment_scale * self._pred_actions[:, :, 1:]

        # Prey (faster!)
        self._prey_actions[:] = actions["prey"].clamp(-1.0, 1.0)
        self._prey_thrust[:, 0, 2] = (
            self.cfg.prey_thrust_to_weight * self._robot_weight * (self._prey_actions[:, 0] + 1.0) / 2.0
        )
        self._prey_moment[:, 0, :] = self.cfg.moment_scale * self._prey_actions[:, 1:]

    def _apply_action(self) -> None:
        # Still per-predator because each Articulation has its own wrench composer
        for i, pred in enumerate(self._predators):
            pred.permanent_wrench_composer.set_forces_and_torques(
                body_ids=self._pred_body_ids[i],
                forces=self._pred_thrust[:, i],
                torques=self._pred_moment[:, i],
            )
        self._prey.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._prey_body_id, forces=self._prey_thrust, torques=self._prey_moment
        )

    # ------------------------------------------------------------------
    # Observations — fully vectorized
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        N = self.num_envs
        P = self._P
        prey_pos_w = self._prey.data.root_pos_w         # (N, 3)
        prey_vel_w = self._prey.data.root_lin_vel_w      # (N, 3)
        prey_quat_w = self._prey.data.root_quat_w        # (N, 4)
        env_origins = self._terrain.env_origins           # (N, 3)

        # Stack all predator data into (P, N, D) then transpose to (N, P, D)
        pred_pos_w = torch.stack([p.data.root_pos_w for p in self._predators])    # (P, N, 3)
        pred_quat_w = torch.stack([p.data.root_quat_w for p in self._predators])  # (P, N, 4)
        pred_lin_vel_w = torch.stack([p.data.root_lin_vel_w for p in self._predators])
        pred_lin_vel_b = torch.stack([p.data.root_lin_vel_b for p in self._predators])
        pred_ang_vel_b = torch.stack([p.data.root_ang_vel_b for p in self._predators])
        pred_grav_b = torch.stack([p.data.projected_gravity_b for p in self._predators])

        # === Cached shared data (in-place) ===
        # Prey relative + OOB
        self._prey_pos_rel[:] = prey_pos_w - env_origins
        self._prey_horiz[:] = torch.linalg.norm(self._prey_pos_rel[:, :2], dim=1)
        self._prey_oob[:] = (
            (self._prey_pos_rel[:, 2] < self.cfg.min_height)
            | (self._prey_pos_rel[:, 2] > self.cfg.max_height)
            | (self._prey_horiz > self.cfg.arena_radius)
        )

        # Predator relative + OOB — vectorized (P, N, 3) → (N, P, 3)
        pred_pos_rel = (pred_pos_w - env_origins.unsqueeze(0)).permute(1, 0, 2)  # (N, P, 3)
        self._pred_pos_rel[:] = pred_pos_rel
        self._pred_horiz[:] = torch.linalg.norm(pred_pos_rel[:, :, :2], dim=2)   # (N, P)
        self._pred_oob[:] = (
            (pred_pos_rel[:, :, 2] < self.cfg.min_height)
            | (pred_pos_rel[:, :, 2] > self.cfg.max_height)
            | (self._pred_horiz > self.cfg.arena_radius)
        )

        # Distances: all predators to prey — vectorized
        self._current_distances[:] = torch.linalg.norm(
            pred_pos_w - prey_pos_w.unsqueeze(0), dim=-1
        ).t()  # (N, P)

        # NaN detection (in-place)
        self._has_nan.zero_()
        self._has_nan |= torch.any(torch.isnan(self._current_distances), dim=1)
        for i in range(P):
            self._has_nan |= torch.any(torch.isnan(pred_pos_w[i]), dim=1)
        self._has_nan |= torch.any(torch.isnan(prey_pos_w), dim=1)

        # Sanitize
        self._current_distances[:] = torch.nan_to_num(self._current_distances, nan=100.0)
        self._pred_pos_rel[:] = torch.nan_to_num(self._pred_pos_rel, nan=0.0)
        self._pred_horiz[:] = torch.nan_to_num(self._pred_horiz, nan=100.0)
        self._prey_pos_rel[:] = torch.nan_to_num(self._prey_pos_rel, nan=0.0)
        self._prey_horiz[:] = torch.nan_to_num(self._prey_horiz, nan=100.0)

        # Caught
        self._caught[:] = self._current_distances.min(dim=1).values < self.cfg.catch_distance

        # === Predator observations — batched subtract_frame_transforms ===
        # Pred-to-prey: batch all P predators in one call (P*N, 3)
        flat_pred_pos = pred_pos_w.reshape(P * N, 3)
        flat_pred_quat = pred_quat_w.reshape(P * N, 4)
        prey_pos_rep = prey_pos_w.repeat(P, 1)           # (P*N, 3)
        flat_to_prey_b, _ = subtract_frame_transforms(flat_pred_pos, flat_pred_quat, prey_pos_rep)
        to_prey_b = flat_to_prey_b.view(P, N, 3)         # (P, N, 3)

        # Pred-to-prey velocity (simple subtraction, no frame transform needed)
        to_prey_vel = prey_vel_w.unsqueeze(0) - pred_lin_vel_w  # (P, N, 3)

        # Teammate positions: 6 pairs batched in one call
        # _teammate_src = [0,0,1,1,2,2], _teammate_dst = [1,2,0,2,0,1]
        n_pairs = len(self._teammate_src)
        src_pos = pred_pos_w[self._teammate_src].reshape(n_pairs * N, 3)
        src_quat = pred_quat_w[self._teammate_src].reshape(n_pairs * N, 4)
        dst_pos = pred_pos_w[self._teammate_dst].reshape(n_pairs * N, 3)
        flat_teammate_b, _ = subtract_frame_transforms(src_pos, src_quat, dst_pos)
        teammate_b = flat_teammate_b.view(n_pairs, N, 3)  # (6, N, 3)

        # Assemble per-predator 24D obs and stack to 72D
        pred_obs_parts = []
        tm_idx = 0
        for i in range(P):
            pred_obs_parts.append(torch.cat([
                pred_lin_vel_b[i],        # 3
                pred_ang_vel_b[i],         # 3
                pred_grav_b[i],            # 3
                self._pred_pos_rel[:, i],  # 3
                to_prey_b[i],              # 3
                to_prey_vel[i],            # 3
                teammate_b[tm_idx],        # 3 (first teammate)
                teammate_b[tm_idx + 1],    # 3 (second teammate)
            ], dim=-1))  # 24D
            tm_idx += 2

        predator_obs = torch.cat(pred_obs_parts, dim=-1)  # (N, 72)

        # === Prey observations — batched ===
        prey_pos_rep2 = prey_pos_w.repeat(P, 1)
        prey_quat_rep = prey_quat_w.repeat(P, 1)
        flat_to_pred_b, _ = subtract_frame_transforms(prey_pos_rep2, prey_quat_rep, flat_pred_pos)
        to_pred_b = flat_to_pred_b.view(P, N, 3)          # (P, N, 3)
        to_pred_vel = pred_lin_vel_w - prey_vel_w.unsqueeze(0)  # (P, N, 3)

        prey_obs = torch.cat([
            self._prey.data.root_lin_vel_b,       # 3
            self._prey.data.root_ang_vel_b,        # 3
            self._prey.data.projected_gravity_b,   # 3
            self._prey_pos_rel,                    # 3
            to_pred_b[0], to_pred_vel[0],          # 6
            to_pred_b[1], to_pred_vel[1],          # 6
            to_pred_b[2], to_pred_vel[2],          # 6
        ], dim=-1)  # 30D

        # NaN guard
        predator_obs = torch.nan_to_num(predator_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        prey_obs = torch.nan_to_num(prey_obs, nan=0.0, posinf=10.0, neginf=-10.0)

        obs = {"predator": predator_obs, "prey": prey_obs}
        self._cached_obs = obs
        return obs

    def _get_states(self) -> torch.Tensor | None:
        return torch.cat([self._cached_obs["predator"], self._cached_obs["prey"]], dim=-1)  # 102D

    # ------------------------------------------------------------------
    # Rewards — vectorized
    # ------------------------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        dt = self.step_dt
        P = self._P
        warn_radius = self.cfg.arena_radius * self.cfg.boundary_warn_fraction
        caught_f = self._caught.float()
        closest_pred_idx = self._current_distances.argmin(dim=1)  # (N,)

        # === Predator rewards — batched across all 3 drones ===
        # Gather stacked data: (P, N, D) from articulations
        pred_grav_z = torch.stack([p.data.projected_gravity_b[:, 2] for p in self._predators])  # (P, N)
        pred_lin_vel_sq = torch.stack([
            torch.sum(torch.square(p.data.root_lin_vel_b), dim=1) for p in self._predators
        ])  # (P, N)
        pred_ang_vel_sq = torch.stack([
            torch.sum(torch.square(p.data.root_ang_vel_b), dim=1) for p in self._predators
        ])  # (P, N)

        # Flight stability (P, N)
        upright = (-pred_grav_z) * self.cfg.upright_reward_scale * dt
        height_err = self._pred_pos_rel[:, :, 2].t()  # (P, N)
        height = torch.square(height_err - self.cfg.target_height) * self.cfg.height_penalty_scale * dt
        lin_vel = pred_lin_vel_sq * self.cfg.lin_vel_penalty * dt
        ang_vel = pred_ang_vel_sq * self.cfg.ang_vel_penalty * dt
        action_pen = torch.sum(torch.square(self._pred_actions), dim=2).t() * self.cfg.action_penalty * dt  # (P, N)

        # Proximity — gated on airborne (P, N)
        is_flying = (self._pred_pos_rel[:, :, 2].t() > self.cfg.min_height).float()
        distances_t = self._current_distances.t()  # (P, N)
        proximity = is_flying * (1.0 - torch.tanh(distances_t / 2.0)) * self.cfg.predator_proximity_reward_scale * dt

        # Catch attribution (P, N)
        pred_indices = torch.arange(P, device=self.device).unsqueeze(1)  # (P, 1)
        is_catcher = (closest_pred_idx.unsqueeze(0) == pred_indices) & self._caught.unsqueeze(0)  # (P, N)
        catch_bonus = is_catcher.float() * self.cfg.predator_catch_bonus
        is_assisting = (~is_catcher) & self._caught.unsqueeze(0) & (distances_t < self.cfg.assist_distance)
        assist_bonus = is_assisting.float() * self.cfg.predator_assist_bonus

        # Boundary + OOB (P, N)
        horiz_t = self._pred_horiz.t()  # (P, N)
        boundary = torch.clamp(horiz_t - warn_radius, min=0.0) * self.cfg.boundary_penalty_scale * dt
        oob_pen = self._pred_oob.t().float() * self.cfg.oob_penalty

        # Per-drone reward (P, N) → team mean (N,)
        per_drone_reward = (
            upright + height + lin_vel + ang_vel + action_pen
            + proximity + catch_bonus + assist_bonus
            - boundary + oob_pen
        )
        pred_team_reward = per_drone_reward.mean(dim=0)  # (N,)

        # === Prey reward ===
        prey_upright = (-self._prey.data.projected_gravity_b[:, 2]) * self.cfg.upright_reward_scale * dt
        prey_height = torch.square(self._prey_pos_rel[:, 2] - self.cfg.target_height) * self.cfg.height_penalty_scale * dt
        prey_lin_vel = torch.sum(torch.square(self._prey.data.root_lin_vel_b), dim=1) * self.cfg.lin_vel_penalty * dt
        prey_ang_vel = torch.sum(torch.square(self._prey.data.root_ang_vel_b), dim=1) * self.cfg.ang_vel_penalty * dt
        prey_action_pen = torch.sum(torch.square(self._prey_actions), dim=1) * self.cfg.action_penalty * dt
        prey_flying = (self._prey_pos_rel[:, 2] > self.cfg.min_height).float()
        prey_alive_gated = prey_flying * self.cfg.prey_alive_bonus * dt
        prey_caught = caught_f * self.cfg.prey_caught_penalty
        prey_boundary = torch.clamp(self._prey_horiz - warn_radius, min=0.0) * self.cfg.boundary_penalty_scale * dt
        prey_oob_pen = self._prey_oob.float() * self.cfg.oob_penalty

        # Evasion reward: prey gets rewarded for distance from nearest predator
        # Mirrors predator proximity but inverted: tanh(dist/2) → 0 when close, 1 when far
        min_pred_dist = self._current_distances.min(dim=1).values  # (N,)
        prey_evasion = prey_flying * torch.tanh(min_pred_dist / 2.0) * self.cfg.prey_evasion_reward_scale * dt

        prey_reward = (
            prey_upright + prey_height + prey_lin_vel + prey_ang_vel + prey_action_pen
            + prey_alive_gated + prey_evasion + prey_caught - prey_boundary + prey_oob_pen
        )

        # === Per-agent reward logging ===
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Reward/predator_mean"] = pred_team_reward.mean()
        self.extras["log"]["Reward/prey_mean"] = prey_reward.mean()
        self.extras["log"]["Reward/predator_proximity"] = proximity.mean()
        self.extras["log"]["Reward/predator_catch"] = catch_bonus.mean()
        self.extras["log"]["Reward/predator_upright"] = upright.mean()
        self.extras["log"]["Reward/prey_alive"] = prey_alive_gated.mean()
        self.extras["log"]["Reward/prey_evasion"] = prey_evasion.mean()
        self.extras["log"]["Reward/prey_caught"] = prey_caught.mean()

        # Episode tracking
        self._episode_catches += caught_f
        self._episode_pred_oob += self._pred_oob.any(dim=1).float()
        self._episode_prey_oob += self._prey_oob.float()
        step_min_dist = self._current_distances.min(dim=1).values
        self._episode_min_distance = torch.minimum(self._episode_min_distance, step_min_dist)

        return {"predator": pred_team_reward, "prey": prey_reward}

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._caught | self._pred_oob.any(dim=1) | self._prey_oob | self._has_nan
        return (
            {"predator": terminated, "prey": terminated},
            {"predator": time_out, "prey": time_out},
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._predators[0]._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        if n == 0:
            return

        # Logging
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/catch_rate"] = self._episode_catches[env_ids].mean()
        self.extras["log"]["Metrics/predator_oob_rate"] = self._episode_pred_oob[env_ids].mean()
        self.extras["log"]["Metrics/prey_oob_rate"] = self._episode_prey_oob[env_ids].mean()
        self.extras["log"]["Metrics/episode_length"] = self.episode_length_buf[env_ids].float().mean()
        self.extras["log"]["Metrics/episode_closest_approach"] = self._episode_min_distance[env_ids].mean()

        self._episode_catches[env_ids] = 0.0
        self._episode_pred_oob[env_ids] = 0.0
        self._episode_prey_oob[env_ids] = 0.0
        self._episode_min_distance[env_ids] = 100.0

        # Reset actions
        self._pred_actions[env_ids] = 0.0
        self._prey_actions[env_ids] = 0.0

        # Spawn predators (120° apart, random rotation per env)
        base_angles = torch.rand(n, device=self.device) * 2 * math.pi
        for i, pred in enumerate(self._predators):
            angles = base_angles + self._spawn_angles[i]
            x = self.cfg.predator_spawn_radius * torch.cos(angles)
            y = self.cfg.predator_spawn_radius * torch.sin(angles)

            state = pred.data.default_root_state[env_ids].clone()
            state[:, 0] = x
            state[:, 1] = y
            state[:, 2] = self.cfg.target_height
            state[:, :3] += (torch.rand(n, 3, device=self.device) * 2 - 1) * self.cfg.spawn_pos_noise
            state[:, :3] += self._terrain.env_origins[env_ids]

            pred.write_root_pose_to_sim(state[:, :7], env_ids)
            pred.write_root_velocity_to_sim(state[:, 7:], env_ids)
            pred.write_joint_state_to_sim(
                pred.data.default_joint_pos[env_ids],
                pred.data.default_joint_vel[env_ids],
                None, env_ids,
            )

        # Spawn prey at center
        prey_state = self._prey.data.default_root_state[env_ids].clone()
        prey_state[:, 0] = self.cfg.prey_spawn_pos[0]
        prey_state[:, 1] = self.cfg.prey_spawn_pos[1]
        prey_state[:, 2] = self.cfg.prey_spawn_pos[2]
        prey_state[:, :3] += (torch.rand(n, 3, device=self.device) * 2 - 1) * self.cfg.spawn_pos_noise
        prey_state[:, :3] += self._terrain.env_origins[env_ids]

        self._prey.write_root_pose_to_sim(prey_state[:, :7], env_ids)
        self._prey.write_root_velocity_to_sim(prey_state[:, 7:], env_ids)
        self._prey.write_joint_state_to_sim(
            self._prey.data.default_joint_pos[env_ids],
            self._prey.data.default_joint_vel[env_ids],
            None, env_ids,
        )
