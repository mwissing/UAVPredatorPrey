# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject

from ..uavpredatorprey_3v1.uav_3v1_env import Uav3v1Env
from .uav_3v1_obstacles_env_cfg import Uav3v1ObstaclesEnvCfg


class Uav3v1ObstaclesEnv(Uav3v1Env):
    """3v1 Predator-Prey with randomly placed cylindrical obstacles.

    Inherits all behavior from Uav3v1Env and adds:
    - Obstacle spawning in _setup_scene()
    - Obstacle-awareness observations (+4D per drone: nearest_obstacle_vec(3) + dist(1))
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

    def _get_observations(self) -> dict[str, torch.Tensor]:
        N = self.num_envs
        P = self._P
        K = self.cfg.num_obstacles
        env_origins = self._terrain.env_origins

        # Get parent observations (also computes all cached data)
        parent_obs = super()._get_observations()

        # Compute obstacle positions relative to env origins
        for i, obs_obj in enumerate(self._obstacles):
            self._obstacle_pos_rel[:, i] = obs_obj.data.root_pos_w - env_origins

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
            nearest_diff_xy = diffs_xy[torch.arange(N, device=self.device), nearest_idx]  # (N, 2)
            nearest_dist = dists_xy[torch.arange(N, device=self.device), nearest_idx]  # (N,)

            # 3D vector to nearest obstacle (XY diff + Z diff)
            nearest_obs_z = self._obstacle_pos_rel[torch.arange(N, device=self.device), nearest_idx, 2]
            drone_z = self._pred_pos_rel[:, i, 2]
            nearest_vec = torch.cat([nearest_diff_xy, (nearest_obs_z - drone_z).unsqueeze(1)], dim=1)  # (N, 3)

            # Normalize by arena radius for consistent input scale
            pred_obstacle_obs.append(torch.cat([
                nearest_vec * inv_arena,
                nearest_dist.unsqueeze(1) * inv_arena,
            ], dim=1))  # (N, 4)

        # Stack and append to predator obs: 72D + 3×4D = 84D
        pred_obs_extra = torch.cat(pred_obstacle_obs, dim=-1)  # (N, 12)
        predator_obs = torch.cat([parent_obs["predator"], pred_obs_extra], dim=-1)  # (N, 84)

        # === Prey obstacle observations ===
        prey_xy = self._prey_pos_rel[:, :2].unsqueeze(1)  # (N, 1, 2)
        prey_diffs_xy = obs_positions - prey_xy  # (N, K, 2)
        prey_dists_xy = torch.linalg.norm(prey_diffs_xy, dim=2)  # (N, K)
        prey_nearest_idx = prey_dists_xy.argmin(dim=1)  # (N,)

        prey_nearest_diff_xy = prey_diffs_xy[torch.arange(N, device=self.device), prey_nearest_idx]
        prey_nearest_dist = prey_dists_xy[torch.arange(N, device=self.device), prey_nearest_idx]
        prey_nearest_obs_z = self._obstacle_pos_rel[torch.arange(N, device=self.device), prey_nearest_idx, 2]
        prey_nearest_vec = torch.cat([
            prey_nearest_diff_xy,
            (prey_nearest_obs_z - self._prey_pos_rel[:, 2]).unsqueeze(1),
        ], dim=1)

        # Normalize by arena radius
        prey_obs_extra = torch.cat([
            prey_nearest_vec * inv_arena,
            prey_nearest_dist.unsqueeze(1) * inv_arena,
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

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        dt = self.step_dt
        N = self.num_envs
        P = self._P
        K = self.cfg.num_obstacles

        # Get base rewards from parent
        rewards = super()._get_rewards()

        # Compute drone-to-obstacle distances for all drones
        obs_xy = self._obstacle_pos_rel[:, :, :2]  # (N, K, 2)

        # === Predator obstacle penalties ===
        pred_obstacle_pen = torch.zeros(P, N, device=self.device)
        pred_collision = torch.zeros(P, N, dtype=torch.bool, device=self.device)

        for i in range(P):
            drone_xy = self._pred_pos_rel[:, i, :2].unsqueeze(1)  # (N, 1, 2)
            dists = torch.linalg.norm(obs_xy - drone_xy, dim=2)  # (N, K)
            min_dist = dists.min(dim=1).values  # (N,)

            # Soft proximity penalty (increases as drone approaches obstacle)
            proximity_pen = torch.clamp(self.cfg.obstacle_warn_distance - min_dist, min=0.0)
            pred_obstacle_pen[i] = proximity_pen * self.cfg.obstacle_proximity_penalty * dt

            # Hard collision penalty
            pred_collision[i] = min_dist < self.cfg.obstacle_collision_distance

        # Team obstacle penalty (mean across drones)
        pred_obstacle_team = pred_obstacle_pen.mean(dim=0)  # (N,)
        pred_collision_any = pred_collision.any(dim=0)
        pred_collision_pen = pred_collision_any.float() * self.cfg.obstacle_collision_penalty

        rewards["predator"] = rewards["predator"] + pred_obstacle_team + pred_collision_pen

        # === Prey obstacle penalties ===
        prey_xy = self._prey_pos_rel[:, :2].unsqueeze(1)  # (N, 1, 2)
        prey_dists = torch.linalg.norm(obs_xy - prey_xy, dim=2)  # (N, K)
        prey_min_dist = prey_dists.min(dim=1).values

        prey_proximity_pen = torch.clamp(self.cfg.obstacle_warn_distance - prey_min_dist, min=0.0)
        prey_obstacle_pen = prey_proximity_pen * self.cfg.obstacle_proximity_penalty * dt
        prey_collision = (prey_min_dist < self.cfg.obstacle_collision_distance).float()
        prey_collision_pen = prey_collision * self.cfg.obstacle_collision_penalty

        rewards["prey"] = rewards["prey"] + prey_obstacle_pen + prey_collision_pen

        # Logging
        self.extras["log"]["Reward/obstacle_proximity_pred"] = pred_obstacle_team.mean()
        self.extras["log"]["Reward/obstacle_collision_pred"] = pred_collision_pen.mean()
        self.extras["log"]["Reward/obstacle_proximity_prey"] = prey_obstacle_pen.mean()

        # Episode tracking
        self._episode_obstacle_collisions += (pred_collision_any | (prey_collision > 0)).float()

        return rewards

    # ------------------------------------------------------------------
    # Termination — extends parent (obstacle collision = termination)
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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

        # Log obstacle collision rate before parent resets episode counters
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/obstacle_collision_rate"] = self._episode_obstacle_collisions[env_ids].mean()

        # Parent handles drone reset + episode counter reset
        super()._reset_idx(env_ids)

        # Reset obstacle collision counter
        self._episode_obstacle_collisions[env_ids] = 0.0

        # Randomize obstacle positions for reset envs
        self._randomize_obstacles(env_ids, n)

    def _randomize_obstacles(self, env_ids: Sequence[int], n: int):
        """Place obstacles randomly in annular region, ensuring minimum separation."""
        K = self.cfg.num_obstacles
        r_min = self.cfg.obstacle_min_spawn_radius
        r_max = self.cfg.obstacle_max_spawn_radius
        min_sep = self.cfg.obstacle_min_separation

        for env_batch_start in range(0, n, n):  # process all at once
            # Generate random positions with rejection sampling for separation
            # For 4 obstacles in a 1.5-4.0m ring, rejection is very efficient
            positions = torch.zeros(n, K, 2, device=self.device)

            for k in range(K):
                valid = torch.zeros(n, dtype=torch.bool, device=self.device)
                attempts = 0
                while not valid.all() and attempts < 50:
                    # Random polar coords for invalid envs
                    num_invalid = (~valid).sum()
                    r = r_min + (r_max - r_min) * torch.rand(num_invalid, device=self.device)
                    theta = torch.rand(num_invalid, device=self.device) * 2 * math.pi
                    new_x = r * torch.cos(theta)
                    new_y = r * torch.sin(theta)
                    candidates = torch.stack([new_x, new_y], dim=1)  # (num_invalid, 2)

                    # Check separation from already-placed obstacles
                    sep_ok = torch.ones(num_invalid, dtype=torch.bool, device=self.device)
                    for prev_k in range(k):
                        prev_pos = positions[~valid, prev_k]  # (num_invalid, 2)
                        dist = torch.linalg.norm(candidates - prev_pos, dim=1)
                        sep_ok &= dist > min_sep

                    # Accept valid candidates
                    invalid_indices = torch.where(~valid)[0]
                    accept = invalid_indices[sep_ok]
                    positions[accept, k] = candidates[sep_ok]
                    valid[accept] = True
                    attempts += 1

                # Fallback: any still-invalid get placed at fixed angles
                if not valid.all():
                    angle = k * 2 * math.pi / K
                    r_mid = (r_min + r_max) / 2
                    positions[~valid, k, 0] = r_mid * math.cos(angle)
                    positions[~valid, k, 1] = r_mid * math.sin(angle)

        # Write positions to simulation
        for k, obs_obj in enumerate(self._obstacles):
            state = obs_obj.data.default_root_state[env_ids].clone()
            state[:, 0] = positions[:, k, 0]
            state[:, 1] = positions[:, k, 1]
            state[:, 2] = self.cfg.obstacle_height / 2.0  # center of cylinder
            state[:, :3] += self._terrain.env_origins[env_ids]

            obs_obj.write_root_pose_to_sim(state[:, :7], env_ids)
            obs_obj.write_root_velocity_to_sim(state[:, 7:], env_ids)
