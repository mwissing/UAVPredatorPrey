# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import subtract_frame_transforms

from ..uavpredatorprey_3v1.uav_3v1_env import Uav3v1Env
from .uav_3v1_obstacles_env_cfg import Uav3v1ObstaclesEnvCfg


class Uav3v1ObstaclesEnv(Uav3v1Env):
    """3v1 Predator-Prey with randomly placed cylindrical obstacles.

    Inherits all behavior from Uav3v1Env and adds:
    - Obstacle spawning in _setup_scene()
    - Obstacle-awareness observations (+4D per drone)
    - Obstacle proximity penalty and collision penalty in rewards
    - Random obstacle repositioning on reset
    """

    cfg: Uav3v1ObstaclesEnvCfg

    def __init__(self, cfg: Uav3v1ObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        K = cfg.num_obstacles

        # Pre-allocated obstacle data
        self._obstacle_pos_rel = torch.zeros(N, K, 3, device=self.device)  # relative to env origin
        self._episode_obstacle_collisions = torch.zeros(N, device=self.device)
        self._episode_pred_obstacle_collisions = torch.zeros(N, device=self.device)
        self._episode_prey_obstacle_collisions = torch.zeros(N, device=self.device)
        self._episode_prey_cover_steps = torch.zeros(N, device=self.device)
        self._episode_prey_cover_score = torch.zeros(N, device=self.device)
        self._episode_prey_threat_steps = torch.zeros(N, device=self.device)
        self._episode_prey_active_cover_steps = torch.zeros(N, device=self.device)
        self._episode_prey_active_cover_score = torch.zeros(N, device=self.device)
        self._episode_prey_active_cover_den = torch.zeros(N, device=self.device)
        self._episode_initial_cover_score = torch.zeros(N, device=self.device)
        self._episode_prey_shadow_score = torch.zeros(N, device=self.device)
        self._prev_min_pred_prey_distance = torch.full((N,), cfg.predator_spawn_radius, device=self.device)

    def _setup_scene(self):
        # Must create ALL prims BEFORE clone_environments(), so we override completely
        # instead of calling super()._setup_scene()

        # --- Drones (same as parent) ---
        self._predators: list[Articulation] = []
        for i in range(self.cfg.num_predators):
            cfg_attr = getattr(self.cfg, f"predator_{i}_cfg")
            self._predators.append(Articulation(cfg_attr))
        self._prey = Articulation(self.cfg.prey_cfg)

        # --- Obstacles (BEFORE clone!) ---
        self._obstacles: list[RigidObject] = []
        for i in range(self.cfg.num_obstacles):
            cfg_attr = getattr(self.cfg, f"obstacle_{i}_cfg")
            self._obstacles.append(RigidObject(cfg_attr))

        # --- Terrain ---
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # --- Clone environments (all prims must exist by now) ---
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # --- Register all assets in scene ---
        for i, pred in enumerate(self._predators):
            self.scene.articulations[f"predator_{i}"] = pred
        self.scene.articulations["prey"] = self._prey
        for i, obs_obj in enumerate(self._obstacles):
            self.scene.rigid_objects[f"obstacle_{i}"] = obs_obj

        # --- Lights ---
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Observations — extends parent with obstacle awareness
    # ------------------------------------------------------------------

    def _compute_full_obstacle_obs(
        self,
        agent_pos_w: torch.Tensor,
        agent_quat_w: torch.Tensor,
        agent_xy: torch.Tensor,
        target_xy: torch.Tensor,
        obs_positions_w: torch.Tensor,
        obs_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Return all-obstacle features sorted by distance to one agent body."""
        N = agent_xy.shape[0]
        K = self.cfg.num_obstacles
        inv_arena = 1.0 / self.cfg.arena_radius
        clearance_width = max(
            self.cfg.obstacle_warn_distance - self.cfg.obstacle_collision_distance,
            1.0e-6,
        )

        diffs_xy = obs_xy - agent_xy.unsqueeze(1)
        dists_xy = torch.linalg.norm(diffs_xy, dim=2)
        sort_idx = dists_xy.argsort(dim=1)
        gather_xy = sort_idx.unsqueeze(2).expand(-1, -1, 2)
        gather_xyz = sort_idx.unsqueeze(2).expand(-1, -1, 3)

        sorted_dists = dists_xy.gather(1, sort_idx)
        sorted_obs_xy = obs_xy.gather(1, gather_xy)
        sorted_obs_w = obs_positions_w.gather(1, gather_xyz)

        agent_pos_rep = agent_pos_w.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
        agent_quat_rep = agent_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
        obstacle_vec_b, _ = subtract_frame_transforms(
            agent_pos_rep,
            agent_quat_rep,
            sorted_obs_w.reshape(N * K, 3),
        )
        obstacle_vec_b = obstacle_vec_b.view(N, K, 3)

        obstacle_risk = torch.clamp(
            (self.cfg.obstacle_warn_distance - sorted_dists) / clearance_width,
            min=0.0,
            max=1.0,
        )

        to_target = target_xy - agent_xy
        to_target_len_sq = torch.sum(torch.square(to_target), dim=1).clamp_min(1.0e-6)
        obs_from_agent = sorted_obs_xy - agent_xy.unsqueeze(1)
        path_fraction = torch.sum(obs_from_agent * to_target.unsqueeze(1), dim=2) / to_target_len_sq.unsqueeze(1)
        path_fraction_clamped = path_fraction.clamp(0.0, 1.0)
        closest_path_point = (
            agent_xy.unsqueeze(1)
            + path_fraction_clamped.unsqueeze(2) * to_target.unsqueeze(1)
        )
        obstacle_to_path = torch.linalg.norm(sorted_obs_xy - closest_path_point, dim=2)
        path_between = (path_fraction > 0.05) & (path_fraction < 0.95)
        path_block_score = torch.clamp(
            1.0 - obstacle_to_path / self.cfg.obstacle_path_block_radius,
            min=0.0,
            max=1.0,
        )
        path_block_score = torch.where(
            path_between,
            path_block_score,
            torch.zeros_like(path_block_score),
        )

        features = torch.cat(
            [
                obstacle_vec_b * inv_arena,
                sorted_dists.unsqueeze(2) * inv_arena,
                obstacle_risk.unsqueeze(2),
                path_fraction_clamped.unsqueeze(2),
                path_block_score.unsqueeze(2),
            ],
            dim=2,
        )
        return features.reshape(N, K * self.cfg.obstacle_features_per_obstacle)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        N = self.num_envs
        P = self._P
        env_origins = self._terrain.env_origins

        # Get parent observations (also computes all cached data)
        parent_obs = super()._get_observations()

        # Compute obstacle positions relative to env origins
        for i, obs_obj in enumerate(self._obstacles):
            self._obstacle_pos_rel[:, i] = obs_obj.data.root_pos_w - env_origins
        obs_positions_w = self._obstacle_pos_rel + env_origins.unsqueeze(1)

        if self.cfg.obstacle_observation_mode == "full":
            obs_positions = self._obstacle_pos_rel[:, :, :2]
            pred_obstacle_obs = []
            prey_xy = self._prey_pos_rel[:, :2]
            for i in range(P):
                pred_obstacle_obs.append(
                    self._compute_full_obstacle_obs(
                        self._predators[i].data.root_pos_w,
                        self._predators[i].data.root_quat_w,
                        self._pred_pos_rel[:, i, :2],
                        prey_xy,
                        obs_positions_w,
                        obs_positions,
                    )
                )
            pred_obs_extra = torch.cat(pred_obstacle_obs, dim=-1)
            predator_obs = torch.cat([parent_obs["predator"], pred_obs_extra], dim=-1)

            env_indices = torch.arange(N, device=self.device)
            closest_pred_idx = self._current_distances.argmin(dim=1)
            closest_pred_xy = self._pred_pos_rel[env_indices, closest_pred_idx, :2]
            prey_obs_extra = self._compute_full_obstacle_obs(
                self._prey.data.root_pos_w,
                self._prey.data.root_quat_w,
                self._prey_pos_rel[:, :2],
                closest_pred_xy,
                obs_positions_w,
                obs_positions,
            )
            prey_obs = torch.cat([parent_obs["prey"], prey_obs_extra], dim=-1)

            predator_obs = torch.nan_to_num(predator_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            prey_obs = torch.nan_to_num(prey_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            obs = {"predator": predator_obs, "prey": prey_obs}
            self._cached_obs = obs
            return obs

        obs_positions = self._obstacle_pos_rel[:, :, :2]  # (N, K, 2) — only XY for distance

        # Normalization factor — keeps obstacle obs in [-1, 1] range
        inv_arena = 1.0 / self.cfg.arena_radius

        # === Predator obstacle observations ===
        # For each predator drone, find nearest obstacle (in XY plane)
        pred_obstacle_obs = []
        for i in range(P):
            drone_xy = self._pred_pos_rel[:, i, :2].unsqueeze(1)  # (N, 1, 2)
            diffs_xy = obs_positions - drone_xy  # (N, K, 2)
            dists_xy = torch.linalg.norm(diffs_xy, dim=2)  # (N, K)
            nearest_idx = dists_xy.argmin(dim=1)  # (N,)

            # Gather nearest obstacle info
            nearest_dist = dists_xy[torch.arange(N, device=self.device), nearest_idx]  # (N,)
            nearest_obs_w = obs_positions_w[torch.arange(N, device=self.device), nearest_idx]
            nearest_vec_b, _ = subtract_frame_transforms(
                self._predators[i].data.root_pos_w,
                self._predators[i].data.root_quat_w,
                nearest_obs_w,
            )

            # Normalize by arena radius for consistent input scale
            pred_obstacle_obs.append(torch.cat([
                nearest_vec_b * inv_arena,
                nearest_dist.unsqueeze(1) * inv_arena,
            ], dim=1))  # (N, 4)

        # Stack and append to predator obs: 72D + 3×4D = 84D
        pred_obs_extra = torch.cat(pred_obstacle_obs, dim=-1)  # (N, 12)
        predator_obs = torch.cat([parent_obs["predator"], pred_obs_extra], dim=-1)  # (N, 84)

        # === Prey nearest obstacle observations ===
        # Restore nearest obstacle observations to match Stage 3 training.
        # This keeps the loaded checkpoint weights functioning as intended (avoiding obstacles),
        # while letting the reward function guide the prey to seek cover.
        drone_xy = self._prey_pos_rel[:, :2].unsqueeze(1)  # (N, 1, 2)
        diffs_xy = obs_positions - drone_xy  # (N, K, 2)
        dists_xy = torch.linalg.norm(diffs_xy, dim=2)  # (N, K)
        nearest_idx = dists_xy.argmin(dim=1)  # (N,)

        # Gather nearest obstacle info
        nearest_dist = dists_xy[torch.arange(N, device=self.device), nearest_idx]  # (N,)
        nearest_obs_w = obs_positions_w[torch.arange(N, device=self.device), nearest_idx]
        nearest_vec_b, _ = subtract_frame_transforms(
            self._prey.data.root_pos_w,
            self._prey.data.root_quat_w,
            nearest_obs_w,
        )

        prey_obs_extra = torch.cat([
            nearest_vec_b * inv_arena,
            nearest_dist.unsqueeze(1) * inv_arena,
        ], dim=1)  # (N, 4)
        prey_obs = torch.cat([parent_obs["prey"], prey_obs_extra], dim=-1)  # (N, 34)

        # NaN guard on obstacle obs
        predator_obs = torch.nan_to_num(predator_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        prey_obs = torch.nan_to_num(prey_obs, nan=0.0, posinf=10.0, neginf=-10.0)

        obs = {"predator": predator_obs, "prey": prey_obs}
        self._cached_obs = obs
        return obs

    # ------------------------------------------------------------------
    # Rewards — extends parent with obstacle penalties
    # ------------------------------------------------------------------

    def _compute_closest_cover_score(
        self,
        prey_xy: torch.Tensor,
        pred_xy: torch.Tensor,
        obs_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Return LOS-cover score against the closest predator in XY."""
        N = prey_xy.shape[0]
        to_pred = pred_xy - prey_xy.unsqueeze(1)  # (N, P, 2)
        seg_len_sq = torch.sum(torch.square(to_pred), dim=2).clamp_min(1.0e-6)

        obs_from_prey = obs_xy.unsqueeze(1) - prey_xy[:, None, None, :]
        to_pred_exp = to_pred.unsqueeze(2)
        los_fraction = torch.sum(obs_from_prey * to_pred_exp, dim=3) / seg_len_sq.unsqueeze(2)
        between_prey_and_pred = (
            (los_fraction > self.cfg.prey_cover_min_los_fraction)
            & (los_fraction < self.cfg.prey_cover_max_los_fraction)
        )

        los_fraction_clamped = los_fraction.clamp(0.0, 1.0)
        closest_point = prey_xy[:, None, None, :] + los_fraction_clamped.unsqueeze(3) * to_pred_exp
        obstacle_to_los = torch.linalg.norm(obs_xy.unsqueeze(1) - closest_point, dim=3)
        cover_by_obstacle = torch.clamp(
            1.0 - obstacle_to_los / self.cfg.prey_cover_radius,
            min=0.0,
            max=1.0,
        )
        cover_by_obstacle = torch.where(
            between_prey_and_pred,
            cover_by_obstacle,
            torch.zeros_like(cover_by_obstacle),
        )
        cover_by_pred = cover_by_obstacle.max(dim=2).values

        closest_pred_idx = torch.linalg.norm(to_pred, dim=2).argmin(dim=1)
        env_indices = torch.arange(N, device=self.device)
        return cover_by_pred[env_indices, closest_pred_idx]

    def _compute_shadow_target(
        self,
        prey_xy: torch.Tensor,
        closest_pred_xy: torch.Tensor,
        obs_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score how close the prey is to useful obstacle shadow targets."""
        pred_to_obs = obs_xy - closest_pred_xy.unsqueeze(1)
        pred_to_obs_dist = torch.linalg.norm(pred_to_obs, dim=2, keepdim=True).clamp_min(1.0e-6)
        shadow_dir = pred_to_obs / pred_to_obs_dist
        shadow_targets = obs_xy + shadow_dir * self.cfg.prey_shadow_target_offset

        target_arena_dist = torch.linalg.norm(shadow_targets, dim=2)
        target_in_arena = target_arena_dist < (self.cfg.arena_radius - self.cfg.prey_shadow_arena_margin)
        prey_to_target = shadow_targets - prey_xy.unsqueeze(1)
        prey_target_dist = torch.linalg.norm(prey_to_target, dim=2)
        shadow_score = torch.clamp(
            1.0 - prey_target_dist / self.cfg.prey_shadow_target_radius,
            min=0.0,
            max=1.0,
        )
        shadow_score = torch.where(target_in_arena, shadow_score, torch.zeros_like(shadow_score))
        best_shadow_score, best_shadow_idx = shadow_score.max(dim=1)
        best_shadow_dist = prey_target_dist[
            torch.arange(prey_xy.shape[0], device=self.device),
            best_shadow_idx,
        ]
        best_shadow_target = shadow_targets[
            torch.arange(prey_xy.shape[0], device=self.device),
            best_shadow_idx,
        ]
        return best_shadow_score, best_shadow_dist, best_shadow_target

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        dt = self.step_dt
        N = self.num_envs
        P = self._P

        # Get base rewards from parent
        rewards = super()._get_rewards()
        predator_base_reward = rewards["predator"].clone()
        prey_base_reward = rewards["prey"].clone()

        # Compute drone-to-obstacle distances for all drones
        obs_xy = self._obstacle_pos_rel[:, :, :2]  # (N, K, 2)
        clearance_width = max(
            self.cfg.obstacle_warn_distance - self.cfg.obstacle_collision_distance,
            1.0e-6,
        )

        # === Predator obstacle penalties ===
        pred_obstacle_pen = torch.zeros(P, N, device=self.device)
        pred_obstacle_risk = torch.zeros(P, N, device=self.device)
        pred_collision = torch.zeros(P, N, dtype=torch.bool, device=self.device)

        for i in range(P):
            drone_xy = self._pred_pos_rel[:, i, :2].unsqueeze(1)  # (N, 1, 2)
            dists = torch.linalg.norm(obs_xy - drone_xy, dim=2)  # (N, K)
            min_dist = dists.min(dim=1).values  # (N,)

            # Soft proximity penalty. Normalize clearance so this remains a meaningful
            # per-step shaping signal instead of a near-zero dt-scaled term.
            risk = torch.clamp(
                (self.cfg.obstacle_warn_distance - min_dist) / clearance_width,
                min=0.0,
                max=1.0,
            )
            pred_obstacle_risk[i] = risk
            pred_obstacle_pen[i] = torch.square(risk) * self.cfg.obstacle_proximity_penalty

            # Hard collision penalty
            pred_collision[i] = min_dist < self.cfg.obstacle_collision_distance

        # Team obstacle penalty (mean across drones)
        pred_obstacle_team = pred_obstacle_pen.mean(dim=0)  # (N,)
        pred_obstacle_team_risk = pred_obstacle_risk.max(dim=0).values
        pred_collision_any = pred_collision.any(dim=0)
        pred_collision_pen = pred_collision_any.float() * self.cfg.obstacle_collision_penalty

        rewards["predator"] = rewards["predator"] + pred_obstacle_team + pred_collision_pen

        # === Prey obstacle penalties ===
        prey_xy = self._prey_pos_rel[:, :2].unsqueeze(1)  # (N, 1, 2)
        prey_dists = torch.linalg.norm(obs_xy - prey_xy, dim=2)  # (N, K)
        prey_min_dist = prey_dists.min(dim=1).values

        prey_obstacle_risk = torch.clamp(
            (self.cfg.obstacle_warn_distance - prey_min_dist) / clearance_width,
            min=0.0,
            max=1.0,
        )
        prey_obstacle_pen = torch.square(prey_obstacle_risk) * self.cfg.obstacle_proximity_penalty
        prey_collision = (prey_min_dist < self.cfg.obstacle_collision_distance).float()
        prey_collision_pen = prey_collision * self.cfg.obstacle_collision_penalty

        # === Prey cover reward ===
        # Reward line-of-sight cover: an obstacle is useful if it lies between the prey
        # and the closest predator, not merely because the prey is close to a cylinder.
        prey_xy_flat = self._prey_pos_rel[:, :2]  # (N, 2)
        pred_xy = self._pred_pos_rel[:, :, :2]  # (N, P, 2)
        min_pred_dist, closest_pred_idx = self._current_distances.min(dim=1)
        env_indices = torch.arange(N, device=self.device)
        closest_pred_xy = pred_xy[env_indices, closest_pred_idx]
        closest_cover_score = self._compute_closest_cover_score(prey_xy_flat, pred_xy, obs_xy)

        raw_progress = self._prev_min_pred_prey_distance - min_pred_dist
        distance_progress = torch.clamp(
            raw_progress,
            min=-self.cfg.predator_progress_reward_clip,
            max=self.cfg.predator_progress_reward_clip,
        )
        progress_valid = (
            (~self._caught)
            & (~self._pred_oob.any(dim=1))
            & (~self._prey_oob)
        ).float() * (1.0 - pred_obstacle_team_risk)
        predator_progress_reward = (
            distance_progress
            * self.cfg.predator_progress_reward_scale
            * progress_valid
        )
        self._prev_min_pred_prey_distance[:] = min_pred_dist.detach()

        cover_threat = torch.clamp(
            (self.cfg.prey_cover_threat_distance - min_pred_dist) / self.cfg.prey_cover_threat_distance,
            min=0.0,
            max=1.0,
        )
        cover_pressure = torch.clamp(cover_threat + self.cfg.prey_cover_pressure_floor, max=1.0)
        prey_flying = (self._prey_pos_rel[:, 2] > self.cfg.min_height).float()
        prey_cover_reward = (
            prey_flying * cover_threat * closest_cover_score * self.cfg.prey_cover_reward_scale * dt
        )
        episode_time = self.episode_length_buf.float() * dt
        cover_reward_gate = torch.clamp(
            (episode_time - self.cfg.prey_cover_reward_delay_s) / self.cfg.prey_cover_reward_ramp_s,
            min=0.0,
            max=1.0,
        )
        prey_safe_from_obstacle = (prey_min_dist > self.cfg.obstacle_collision_distance).float()
        prey_cover_reward = prey_cover_reward * prey_safe_from_obstacle * cover_reward_gate
        shadow_reward_gate = torch.clamp(
            (episode_time - self.cfg.prey_shadow_reward_delay_s) / self.cfg.prey_shadow_reward_ramp_s,
            min=0.0,
            max=1.0,
        )

        shadow_score, shadow_target_dist, _ = self._compute_shadow_target(
            prey_xy_flat,
            closest_pred_xy,
            obs_xy,
        )
        prey_shadow_reward = (
            prey_flying
            * cover_pressure
            * shadow_reward_gate
            * prey_safe_from_obstacle
            * shadow_score
            * self.cfg.prey_shadow_reward_scale
            * dt
        )

        cover_seek_width = max(self.cfg.prey_cover_seek_width, 1.0e-6)
        cover_seek_score = torch.clamp(
            1.0 - torch.abs(prey_min_dist - self.cfg.prey_cover_seek_target_distance) / cover_seek_width,
            min=0.0,
            max=1.0,
        )
        cover_seek_score = cover_seek_score * (prey_min_dist < self.cfg.prey_cover_seek_distance).float()
        cover_seek_score = cover_seek_score * (prey_min_dist > self.cfg.obstacle_collision_distance).float()
        prey_cover_seek_reward = (
            prey_flying
            * cover_pressure
            * cover_seek_score
            * (1.0 - closest_cover_score)
            * self.cfg.prey_cover_seek_reward_scale
            * dt
        )

        predator_cover_penalty = (
            -cover_threat * closest_cover_score * self.cfg.predator_cover_penalty_scale * dt
        )

        rewards["predator"] = rewards["predator"] + predator_progress_reward + predator_cover_penalty
        rewards["prey"] = (
            rewards["prey"]
            + prey_obstacle_pen
            + prey_collision_pen
            + prey_cover_reward
            + prey_cover_seek_reward
            + prey_shadow_reward
        )

        # Logging
        self.extras["log"]["Reward/obstacle_proximity_pred"] = pred_obstacle_team.mean()
        self.extras["log"]["Reward/obstacle_collision_pred"] = pred_collision_pen.mean()
        self.extras["log"]["Reward/obstacle_proximity_prey"] = prey_obstacle_pen.mean()
        self.extras["log"]["Reward/obstacle_collision_prey"] = prey_collision_pen.mean()
        self.extras["log"]["Reward/predator_progress"] = predator_progress_reward.mean()
        self.extras["log"]["Reward/predator_base"] = predator_base_reward.mean()
        self.extras["log"]["Reward/prey_base"] = prey_base_reward.mean()
        self.extras["log"]["Reward/predator_mean"] = rewards["predator"].mean()
        self.extras["log"]["Reward/prey_mean"] = rewards["prey"].mean()
        self.extras["log"]["Reward/prey_cover"] = prey_cover_reward.mean()
        self.extras["log"]["Reward/prey_cover_seek"] = prey_cover_seek_reward.mean()
        self.extras["log"]["Reward/prey_shadow"] = prey_shadow_reward.mean()
        self.extras["log"]["Reward/predator_cover_penalty"] = predator_cover_penalty.mean()
        self.extras["log"]["Metrics/predator_distance_progress"] = distance_progress.mean()
        self.extras["log"]["Metrics/predator_distance_progress_raw"] = raw_progress.mean()
        self.extras["log"]["Metrics/predator_obstacle_risk"] = pred_obstacle_team_risk.mean()
        self.extras["log"]["Metrics/prey_obstacle_risk"] = prey_obstacle_risk.mean()
        self.extras["log"]["Metrics/predator_obstacle_collision_step_fraction"] = pred_collision_any.float().mean()
        self.extras["log"]["Metrics/prey_obstacle_collision_step_fraction"] = prey_collision.mean()
        self.extras["log"]["Metrics/prey_cover_step_fraction"] = (closest_cover_score > 0.5).float().mean()
        self.extras["log"]["Metrics/prey_cover_score"] = closest_cover_score.mean()
        self.extras["log"]["Metrics/prey_cover_threat"] = cover_threat.mean()
        self.extras["log"]["Metrics/prey_cover_pressure"] = cover_pressure.mean()
        self.extras["log"]["Metrics/prey_cover_reward_gate"] = cover_reward_gate.mean()
        self.extras["log"]["Metrics/prey_shadow_reward_gate"] = shadow_reward_gate.mean()
        self.extras["log"]["Metrics/prey_shadow_target_score"] = shadow_score.mean()
        self.extras["log"]["Metrics/prey_shadow_target_distance"] = shadow_target_dist.mean()

        # Episode tracking
        active_cover_window = (cover_reward_gate > 0.5).float()
        self._episode_obstacle_collisions += (pred_collision_any | (prey_collision > 0)).float()
        self._episode_pred_obstacle_collisions += pred_collision_any.float()
        self._episode_prey_obstacle_collisions += prey_collision
        self._episode_prey_cover_steps += (closest_cover_score > 0.5).float()
        self._episode_prey_cover_score += closest_cover_score
        self._episode_prey_threat_steps += (cover_threat > 0.0).float()
        self._episode_prey_active_cover_steps += (
            (closest_cover_score > 0.5).float() * active_cover_window
        )
        self._episode_prey_active_cover_score += closest_cover_score * active_cover_window
        self._episode_prey_active_cover_den += active_cover_window
        self._episode_prey_shadow_score += shadow_score

        return rewards

    def _refresh_agent_step_state(self):
        """Refresh cached agent positions/distances without rebuilding full observations."""
        prey_pos_w = self._prey.data.root_pos_w
        pred_pos_w = torch.stack([pred.data.root_pos_w for pred in self._predators])
        env_origins = self._terrain.env_origins

        self._prey_pos_rel[:] = prey_pos_w - env_origins
        self._prey_horiz[:] = torch.linalg.norm(self._prey_pos_rel[:, :2], dim=1)
        self._prey_oob[:] = (
            (self._prey_pos_rel[:, 2] < self.cfg.min_height)
            | (self._prey_pos_rel[:, 2] > self.cfg.max_height)
            | (self._prey_horiz > self.cfg.arena_radius)
        )

        self._pred_pos_rel[:] = (pred_pos_w - env_origins.unsqueeze(0)).permute(1, 0, 2)
        self._pred_horiz[:] = torch.linalg.norm(self._pred_pos_rel[:, :, :2], dim=2)
        self._pred_oob[:] = (
            (self._pred_pos_rel[:, :, 2] < self.cfg.min_height)
            | (self._pred_pos_rel[:, :, 2] > self.cfg.max_height)
            | (self._pred_horiz > self.cfg.arena_radius)
        )

        self._current_distances[:] = torch.linalg.norm(pred_pos_w - prey_pos_w.unsqueeze(0), dim=-1).t()

        self._has_nan.zero_()
        self._has_nan |= torch.any(torch.isnan(self._current_distances), dim=1)
        self._has_nan |= torch.any(torch.isnan(prey_pos_w), dim=1)
        for i in range(self._P):
            self._has_nan |= torch.any(torch.isnan(pred_pos_w[i]), dim=1)

        self._current_distances[:] = torch.nan_to_num(self._current_distances, nan=100.0)
        self._pred_pos_rel[:] = torch.nan_to_num(self._pred_pos_rel, nan=0.0)
        self._pred_horiz[:] = torch.nan_to_num(self._pred_horiz, nan=100.0)
        self._prey_pos_rel[:] = torch.nan_to_num(self._prey_pos_rel, nan=0.0)
        self._prey_horiz[:] = torch.nan_to_num(self._prey_horiz, nan=100.0)
        self._caught[:] = self._current_distances.min(dim=1).values < self.cfg.catch_distance

    # ------------------------------------------------------------------
    # Termination — extends parent (obstacle collision = termination)
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # DirectMARLEnv computes dones/rewards before the post-step observation pass.
        # Refresh parent cached positions/distances so catch/OOB termination and reward shaping
        # use the current physics state instead of the previous observation.
        self._refresh_agent_step_state()

        # No obstacle collision termination — only penalty-based.
        # Terminating on collision was too harsh and prevented exploration.
        return super()._get_dones()

    # ------------------------------------------------------------------
    # Reset — extends parent with obstacle repositioning
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._predators[0]._ALL_INDICES

        n = len(env_ids)
        if n == 0:
            super()._reset_idx(env_ids)
            return

        episode_lengths = self.episode_length_buf[env_ids].float().clamp_min(1.0).clone()

        # Log obstacle collision rate before parent resets episode counters.
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/obstacle_collision_rate"] = (
            self._episode_obstacle_collisions[env_ids] / episode_lengths
        ).mean()
        self.extras["log"]["Metrics/obstacle_collision_steps"] = self._episode_obstacle_collisions[env_ids].mean()
        self.extras["log"]["Metrics/predator_obstacle_collision_rate"] = (
            self._episode_pred_obstacle_collisions[env_ids] / episode_lengths
        ).mean()
        self.extras["log"]["Metrics/prey_obstacle_collision_rate"] = (
            self._episode_prey_obstacle_collisions[env_ids] / episode_lengths
        ).mean()
        self.extras["log"]["Metrics/predator_obstacle_collision_steps"] = (
            self._episode_pred_obstacle_collisions[env_ids].mean()
        )
        self.extras["log"]["Metrics/prey_obstacle_collision_steps"] = (
            self._episode_prey_obstacle_collisions[env_ids].mean()
        )
        cover_rate = self._episode_prey_cover_steps[env_ids] / episode_lengths
        cover_score_episode = self._episode_prey_cover_score[env_ids] / episode_lengths
        active_cover_den = self._episode_prey_active_cover_den[env_ids].clamp_min(1.0)
        self.extras["log"]["Metrics/prey_cover_rate"] = cover_rate.mean()
        self.extras["log"]["Metrics/prey_cover_score_episode"] = cover_score_episode.mean()
        self.extras["log"]["Metrics/prey_cover_when_threatened"] = (
            self._episode_prey_cover_steps[env_ids] / self._episode_prey_threat_steps[env_ids].clamp_min(1.0)
        ).mean()
        self.extras["log"]["Metrics/prey_active_cover_rate"] = (
            self._episode_prey_active_cover_steps[env_ids] / active_cover_den
        ).mean()
        self.extras["log"]["Metrics/prey_active_cover_score"] = (
            self._episode_prey_active_cover_score[env_ids] / active_cover_den
        ).mean()
        self.extras["log"]["Metrics/prey_initial_cover_score"] = self._episode_initial_cover_score[env_ids].mean()
        self.extras["log"]["Metrics/prey_cover_gain_episode"] = (
            cover_score_episode - self._episode_initial_cover_score[env_ids]
        ).mean()
        self.extras["log"]["Metrics/prey_shadow_score_episode"] = (
            self._episode_prey_shadow_score[env_ids] / episode_lengths
        ).mean()

        # Parent handles drone reset + episode counter reset
        super()._reset_idx(env_ids)

        # Reset obstacle collision counter
        self._episode_obstacle_collisions[env_ids] = 0.0
        self._episode_pred_obstacle_collisions[env_ids] = 0.0
        self._episode_prey_obstacle_collisions[env_ids] = 0.0
        self._episode_prey_cover_steps[env_ids] = 0.0
        self._episode_prey_cover_score[env_ids] = 0.0
        self._episode_prey_threat_steps[env_ids] = 0.0
        self._episode_prey_active_cover_steps[env_ids] = 0.0
        self._episode_prey_active_cover_score[env_ids] = 0.0
        self._episode_prey_active_cover_den[env_ids] = 0.0
        self._episode_prey_shadow_score[env_ids] = 0.0

        prey_pos_w = self._prey.data.root_pos_w[env_ids]
        pred_pos_w = torch.stack([pred.data.root_pos_w[env_ids] for pred in self._predators], dim=1)
        self._prev_min_pred_prey_distance[env_ids] = torch.linalg.norm(
            pred_pos_w - prey_pos_w.unsqueeze(1),
            dim=2,
        ).min(dim=1).values

        # Randomize obstacle positions for reset envs
        self._randomize_obstacles(env_ids, n)

    def _randomize_obstacles(self, env_ids: Sequence[int], n: int):
        """Place obstacles with cover bias using fixed-size tensor operations."""
        K = self.cfg.num_obstacles
        r_min = self.cfg.obstacle_min_spawn_radius
        r_max = self.cfg.obstacle_max_spawn_radius
        min_sep = self.cfg.obstacle_min_separation
        agent_min_sep = self.cfg.obstacle_agent_min_spawn_distance
        env_origins_xy = self._terrain.env_origins[env_ids, :2]
        pred_spawn_xy = torch.stack(
            [pred.data.root_pos_w[env_ids, :2] - env_origins_xy for pred in self._predators],
            dim=1,
        )
        prey_spawn_xy = self._prey.data.root_pos_w[env_ids, :2] - env_origins_xy
        agent_spawn_xy = torch.cat([pred_spawn_xy, prey_spawn_xy.unsqueeze(1)], dim=1)

        positions = torch.zeros(n, K, 2, device=self.device)
        cover_spawn_count = min(K, self._P, self.cfg.obstacle_cover_spawn_count)

        if cover_spawn_count > 0:
            cover_pred_xy = pred_spawn_xy[:, :cover_spawn_count]
            to_pred = cover_pred_xy - prey_spawn_xy.unsqueeze(1)
            pred_dist = torch.linalg.norm(to_pred, dim=2, keepdim=True).clamp_min(1.0e-6)
            approach_dir = to_pred / pred_dist
            angle_noise = (
                torch.rand(n, cover_spawn_count, device=self.device) * 2.0 - 1.0
            ) * self.cfg.obstacle_cover_spawn_angle_noise
            cos_noise = torch.cos(angle_noise)
            sin_noise = torch.sin(angle_noise)
            dir_x = approach_dir[:, :, 0] * cos_noise - approach_dir[:, :, 1] * sin_noise
            dir_y = approach_dir[:, :, 0] * sin_noise + approach_dir[:, :, 1] * cos_noise
            approach_dir = torch.stack([dir_x, dir_y], dim=2)
            lateral_dir = torch.stack([-approach_dir[:, :, 1], approach_dir[:, :, 0]], dim=2)

            fraction_noise = (
                torch.rand(n, cover_spawn_count, device=self.device) * 2.0 - 1.0
            ) * self.cfg.obstacle_cover_spawn_fraction_noise
            fraction = self.cfg.obstacle_cover_spawn_fraction + fraction_noise
            along = (pred_dist.squeeze(2) * fraction).clamp(
                self.cfg.obstacle_cover_spawn_radius_min,
                self.cfg.obstacle_cover_spawn_radius_max,
            )
            lateral_mag = (
                self.cfg.obstacle_cover_lateral_offset_min
                + (
                    self.cfg.obstacle_cover_lateral_offset_max
                    - self.cfg.obstacle_cover_lateral_offset_min
                )
                * torch.rand(n, cover_spawn_count, device=self.device)
            )
            lateral_sign = torch.where(
                torch.rand(n, cover_spawn_count, device=self.device) > 0.5,
                torch.ones(n, cover_spawn_count, device=self.device),
                -torch.ones(n, cover_spawn_count, device=self.device),
            )
            cover_positions = (
                prey_spawn_xy.unsqueeze(1)
                + along.unsqueeze(2) * approach_dir
                + (lateral_sign * lateral_mag).unsqueeze(2) * lateral_dir
            )
            positions[:, :cover_spawn_count] = cover_positions

        random_count = K - cover_spawn_count
        if random_count > 0:
            r = r_min + (r_max - r_min) * torch.rand(n, random_count, device=self.device)
            theta = torch.rand(n, random_count, device=self.device) * 2.0 * math.pi
            positions[:, cover_spawn_count:, 0] = r * torch.cos(theta)
            positions[:, cover_spawn_count:, 1] = r * torch.sin(theta)

        fallback_angles = torch.arange(K, device=self.device, dtype=positions.dtype) * (2.0 * math.pi / K)
        fallback_dirs = torch.stack([torch.cos(fallback_angles), torch.sin(fallback_angles)], dim=1).unsqueeze(0)
        agent_clearance_target = agent_min_sep + 0.1
        obstacle_clearance_target = min_sep + 0.05

        def _project_from_agents(pos: torch.Tensor) -> torch.Tensor:
            for agent_idx in range(agent_spawn_xy.shape[1]):
                agent_xy = agent_spawn_xy[:, agent_idx].unsqueeze(1)
                agent_dir = pos - agent_xy
                agent_dist = torch.linalg.norm(agent_dir, dim=2, keepdim=True).clamp_min(1.0e-6)
                agent_dir = torch.where(
                    agent_dist > 1.0e-6,
                    agent_dir / agent_dist,
                    fallback_dirs.expand(n, -1, -1),
                )
                pos = torch.where(
                    agent_dist < agent_clearance_target,
                    agent_xy + agent_dir * agent_clearance_target,
                    pos,
                )
            return pos

        def _project_obstacle_pairs(pos: torch.Tensor) -> torch.Tensor:
            for i in range(K - 1):
                for j in range(i + 1, K):
                    delta = pos[:, j] - pos[:, i]
                    dist = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1.0e-6)
                    fallback_angle = (i * K + j) * 2.0 * math.pi / max(K * K, 1)
                    fallback_pair_dir = torch.stack(
                        [
                            torch.full((n,), math.cos(fallback_angle), device=self.device, dtype=positions.dtype),
                            torch.full((n,), math.sin(fallback_angle), device=self.device, dtype=positions.dtype),
                        ],
                        dim=1,
                    )
                    sep_dir = torch.where(
                        dist > 1.0e-6,
                        delta / dist,
                        fallback_pair_dir,
                    )
                    correction = 0.5 * torch.clamp(obstacle_clearance_target - dist, min=0.0) * sep_dir
                    pos[:, i] -= correction
                    pos[:, j] += correction
            return pos

        # Relax cover-biased candidates away from initial drones and each other. This is deliberately
        # fixed-pass and vectorized across envs, so reset cost stays predictable with 4096 envs.
        for _ in range(self.cfg.obstacle_spawn_correction_passes):
            positions = _project_from_agents(positions)
            positions = _project_obstacle_pairs(positions)

        # End with repeated pair/agent clearance so both logged spawn safety metrics hold together.
        for _ in range(4):
            positions = _project_obstacle_pairs(positions)
            positions = _project_from_agents(positions)
        positions = _project_obstacle_pairs(positions)

        self._obstacle_pos_rel[env_ids, :, :2] = positions
        self._obstacle_pos_rel[env_ids, :, 2] = self.cfg.obstacle_height / 2.0
        initial_cover_score = self._compute_closest_cover_score(prey_spawn_xy, pred_spawn_xy, positions)
        self._episode_initial_cover_score[env_ids] = initial_cover_score
        agent_spawn_dist = torch.linalg.norm(positions.unsqueeze(2) - agent_spawn_xy.unsqueeze(1), dim=3)
        obstacle_pair_dist = torch.linalg.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=3)
        obstacle_pair_dist = obstacle_pair_dist + torch.eye(K, device=self.device).unsqueeze(0) * 1.0e6
        min_agent_dist_by_env = agent_spawn_dist.min(dim=2).values.min(dim=1).values
        min_obstacle_sep_by_env = obstacle_pair_dist.min(dim=2).values.min(dim=1).values
        log = self.extras.setdefault("log", {})
        log["Metrics/prey_spawn_cover_score"] = initial_cover_score.mean()
        log["Metrics/obstacle_spawn_min_agent_distance"] = min_agent_dist_by_env.min()
        log["Metrics/obstacle_spawn_mean_agent_distance"] = min_agent_dist_by_env.mean()
        log["Metrics/obstacle_spawn_min_separation"] = min_obstacle_sep_by_env.min()
        log["Metrics/obstacle_spawn_mean_separation"] = min_obstacle_sep_by_env.mean()

        # Write positions to simulation
        for k, obs_obj in enumerate(self._obstacles):
            state = obs_obj.data.default_root_state[env_ids].clone()
            state[:, 0] = positions[:, k, 0]
            state[:, 1] = positions[:, k, 1]
            state[:, 2] = self.cfg.obstacle_height / 2.0  # center of cylinder
            state[:, :3] += self._terrain.env_origins[env_ids]

            obs_obj.write_root_pose_to_sim(state[:, :7], env_ids)
            obs_obj.write_root_velocity_to_sim(state[:, 7:], env_ids)
