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
    """Configurable predator-prey task with a shared predator policy.

    The default config is 3v1, while curriculum configs can reduce predator
    count without changing the environment implementation.
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
        self._prev_pred_prey_distances = torch.full((N, P), cfg.predator_spawn_radius, device=self.device)
        self._prev_min_pred_prey_distance = torch.full((N,), cfg.predator_spawn_radius, device=self.device)
        self._prev_prey_horiz = torch.zeros(N, device=self.device)

        # Spawn predators evenly around the prey.
        self._spawn_angles = [i * 2 * math.pi / P for i in range(P)]

        # Pre-allocated cached tensors (reused every step via in-place ops)
        self._pred_pos_rel = torch.zeros(N, P, 3, device=self.device)
        self._pred_horiz = torch.zeros(N, P, device=self.device)
        self._pred_oob = torch.zeros(N, P, dtype=torch.bool, device=self.device)
        self._pred_soft_arena_outside = torch.zeros(N, P, device=self.device)
        self._prey_pos_rel = torch.zeros(N, 3, device=self.device)
        self._prey_horiz = torch.zeros(N, device=self.device)
        self._prey_oob = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._prey_soft_arena_outside = torch.zeros(N, device=self.device)
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

    def _sample_predator_spawn_xy(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample predator XY spawn positions relative to each environment origin."""

        base_angles = torch.rand(n, device=self.device) * 2 * math.pi
        spawn_offsets = torch.tensor(self._spawn_angles, device=self.device).unsqueeze(0)
        ring_angles = base_angles.unsqueeze(1) + spawn_offsets

        x = self.cfg.predator_spawn_radius * torch.cos(ring_angles)
        y = self.cfg.predator_spawn_radius * torch.sin(ring_angles)

        if not self.cfg.predator_mixed_spawn:
            return x, y

        mode = torch.rand(n, device=self.device)
        ring_probability = float(self.cfg.predator_mixed_spawn_ring_probability)
        wide_cutoff = ring_probability + float(self.cfg.predator_mixed_spawn_wide_probability)

        wide_mask = (mode >= ring_probability) & (mode < wide_cutoff)
        if wide_mask.any():
            x[wide_mask] = self.cfg.predator_wide_spawn_radius * torch.cos(ring_angles[wide_mask])
            y[wide_mask] = self.cfg.predator_wide_spawn_radius * torch.sin(ring_angles[wide_mask])

        same_side_mask = mode >= wide_cutoff
        if same_side_mask.any():
            if self._P == 1:
                same_side_offsets = torch.zeros(1, device=self.device)
            else:
                spread = float(self.cfg.predator_same_side_spawn_spread)
                same_side_offsets = torch.linspace(-0.5 * spread, 0.5 * spread, self._P, device=self.device)
            same_side_angles = base_angles[same_side_mask].unsqueeze(1) + same_side_offsets.unsqueeze(0)
            x[same_side_mask] = self.cfg.predator_same_side_spawn_radius * torch.cos(same_side_angles)
            y[same_side_mask] = self.cfg.predator_same_side_spawn_radius * torch.sin(same_side_angles)

        return x, y

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
        # Predator: reshape shared policy action to (N, num_predators, 4).
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
        prey_below_min = self._prey_pos_rel[:, 2] < self.cfg.min_height

        # Predator relative + OOB — vectorized (P, N, 3) → (N, P, 3)
        pred_pos_rel = (pred_pos_w - env_origins.unsqueeze(0)).permute(1, 0, 2)  # (N, P, 3)
        self._pred_pos_rel[:] = pred_pos_rel
        self._pred_horiz[:] = torch.linalg.norm(pred_pos_rel[:, :, :2], dim=2)   # (N, P)
        pred_below_min = pred_pos_rel[:, :, 2] < self.cfg.min_height

        if self.cfg.soft_arena_boundary:
            prey_soft_pos = self._prey_pos_rel.clone()
            prey_soft_pos[:, 2] -= self.cfg.target_height
            self._prey_soft_arena_outside[:] = torch.clamp(
                torch.linalg.norm(prey_soft_pos, dim=1) - self.cfg.soft_arena_radius,
                min=0.0,
            )
            pred_soft_pos = pred_pos_rel.clone()
            pred_soft_pos[:, :, 2] -= self.cfg.target_height
            self._pred_soft_arena_outside[:] = torch.clamp(
                torch.linalg.norm(pred_soft_pos, dim=2) - self.cfg.soft_arena_radius,
                min=0.0,
            )
            self._prey_oob[:] = prey_below_min
            self._pred_oob[:] = pred_below_min
        else:
            self._prey_soft_arena_outside.zero_()
            self._pred_soft_arena_outside.zero_()
            self._prey_oob[:] = (
                prey_below_min
                | (self._prey_pos_rel[:, 2] > self.cfg.max_height)
                | (self._prey_horiz > self.cfg.arena_radius)
            )
            self._pred_oob[:] = (
                pred_below_min
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

        # Teammate positions. For 1v1 curricula there are no teammate slots.
        n_pairs = len(self._teammate_src)
        if n_pairs > 0:
            src_pos = pred_pos_w[self._teammate_src].reshape(n_pairs * N, 3)
            src_quat = pred_quat_w[self._teammate_src].reshape(n_pairs * N, 4)
            dst_pos = pred_pos_w[self._teammate_dst].reshape(n_pairs * N, 3)
            flat_teammate_b, _ = subtract_frame_transforms(src_pos, src_quat, dst_pos)
            teammate_b = flat_teammate_b.view(n_pairs, N, 3)
        else:
            teammate_b = torch.zeros(0, N, 3, device=self.device)

        # Assemble per-predator observations and stack them for the shared predator agent.
        pred_obs_parts = []
        tm_idx = 0
        for i in range(P):
            pred_parts = [
                pred_lin_vel_b[i],        # 3
                pred_ang_vel_b[i],         # 3
                pred_grav_b[i],            # 3
                self._pred_pos_rel[:, i],  # 3
                to_prey_b[i],              # 3
                to_prey_vel[i],            # 3
            ]
            for _ in range(P - 1):
                pred_parts.append(teammate_b[tm_idx])
                tm_idx += 1
            pred_obs_parts.append(torch.cat(pred_parts, dim=-1))

        predator_obs = torch.cat(pred_obs_parts, dim=-1)

        # === Prey observations — batched ===
        prey_pos_rep2 = prey_pos_w.repeat(P, 1)
        prey_quat_rep = prey_quat_w.repeat(P, 1)
        flat_to_pred_b, _ = subtract_frame_transforms(prey_pos_rep2, prey_quat_rep, flat_pred_pos)
        to_pred_b = flat_to_pred_b.view(P, N, 3)          # (P, N, 3)
        to_pred_vel = pred_lin_vel_w - prey_vel_w.unsqueeze(0)  # (P, N, 3)

        prey_obs_parts = [
            self._prey.data.root_lin_vel_b,       # 3
            self._prey.data.root_ang_vel_b,        # 3
            self._prey.data.projected_gravity_b,   # 3
            self._prey_pos_rel,                    # 3
        ]
        for i in range(P):
            prey_obs_parts.extend([to_pred_b[i], to_pred_vel[i]])
        prey_obs = torch.cat(prey_obs_parts, dim=-1)

        # NaN guard
        predator_obs = torch.nan_to_num(predator_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        prey_obs = torch.nan_to_num(prey_obs, nan=0.0, posinf=10.0, neginf=-10.0)

        obs = {"predator": predator_obs, "prey": prey_obs}
        self._cached_obs = obs
        return obs

    def _get_states(self) -> torch.Tensor | None:
        return torch.cat([self._cached_obs["predator"], self._cached_obs["prey"]], dim=-1)

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
        predator_progress_valid = (
            is_flying
            * (~self._caught).unsqueeze(0).float()
            * (~self._prey_oob).unsqueeze(0).float()
            * (~self._pred_oob.t()).float()
        )
        predator_distance_progress = torch.clamp(
            self._prev_pred_prey_distances.t() - distances_t,
            min=-self.cfg.predator_distance_progress_reward_clip,
            max=self.cfg.predator_distance_progress_reward_clip,
        )
        predator_distance_progress_reward = (
            predator_distance_progress
            * self.cfg.predator_distance_progress_reward_scale
            * predator_progress_valid
        )
        self._prev_pred_prey_distances[:] = self._current_distances.detach()

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
        pred_soft_arena_pen = (
            -torch.square(self._pred_soft_arena_outside.t())
            * self.cfg.soft_arena_penalty_scale
            * dt
        )

        # Per-drone reward (P, N) → team mean (N,)
        per_drone_reward = (
            upright + height + lin_vel + ang_vel + action_pen
            + proximity + predator_distance_progress_reward + catch_bonus + assist_bonus
            - boundary + oob_pen + pred_soft_arena_pen
        )
        pred_team_reward = per_drone_reward.mean(dim=0)  # (N,)

        # === Prey reward ===
        prey_upright = (-self._prey.data.projected_gravity_b[:, 2]) * self.cfg.upright_reward_scale * dt
        prey_height = torch.square(self._prey_pos_rel[:, 2] - self.cfg.target_height) * self.cfg.height_penalty_scale * dt
        prey_low_altitude = (
            -torch.square(torch.clamp(self.cfg.prey_low_altitude_margin - self._prey_pos_rel[:, 2], min=0.0))
            * self.cfg.prey_low_altitude_penalty_scale
            * dt
        )
        prey_lin_vel = torch.sum(torch.square(self._prey.data.root_lin_vel_b), dim=1) * self.cfg.lin_vel_penalty * dt
        prey_ang_vel = torch.sum(torch.square(self._prey.data.root_ang_vel_b), dim=1) * self.cfg.ang_vel_penalty * dt
        prey_action_pen = torch.sum(torch.square(self._prey_actions), dim=1) * self.cfg.action_penalty * dt
        prey_flying = (self._prey_pos_rel[:, 2] > self.cfg.min_height).float()
        prey_alive_gated = prey_flying * self.cfg.prey_alive_bonus * dt
        prey_caught = caught_f * self.cfg.prey_caught_penalty
        prey_boundary = torch.clamp(self._prey_horiz - warn_radius, min=0.0) * self.cfg.boundary_penalty_scale * dt
        prey_oob_pen = self._prey_oob.float() * self.cfg.oob_penalty
        prey_soft_arena_pen = (
            -torch.square(self._prey_soft_arena_outside)
            * self.cfg.soft_arena_penalty_scale
            * dt
        )

        # Evasion reward: prey gets rewarded for distance from nearest predator
        # Mirrors predator proximity but inverted: tanh(dist/2) → 0 when close, 1 when far
        min_pred_dist = self._current_distances.min(dim=1).values  # (N,)
        prey_evasion = prey_flying * torch.tanh(min_pred_dist / 2.0) * self.cfg.prey_evasion_reward_scale * dt
        progress_state_valid = (
            (~self._caught)
            & (~self._pred_oob.any(dim=1))
            & (~self._prey_oob)
        ).float()
        prey_distance_progress = torch.clamp(
            min_pred_dist - self._prev_min_pred_prey_distance,
            min=-self.cfg.prey_distance_progress_reward_clip,
            max=self.cfg.prey_distance_progress_reward_clip,
        )
        prey_distance_progress_reward = (
            prey_distance_progress
            * self.cfg.prey_distance_progress_reward_scale
            * progress_state_valid
        )
        prey_boundary_progress = torch.clamp(
            self._prev_prey_horiz - self._prey_horiz,
            min=-self.cfg.prey_boundary_progress_reward_clip,
            max=self.cfg.prey_boundary_progress_reward_clip,
        )
        boundary_progress_start = self.cfg.arena_radius * self.cfg.prey_boundary_progress_start_fraction
        boundary_progress_width = max(self.cfg.arena_radius - boundary_progress_start, 1.0e-6)
        prey_boundary_pressure = torch.clamp(
            (self._prey_horiz - boundary_progress_start) / boundary_progress_width,
            min=0.0,
            max=1.0,
        )
        prey_boundary_progress_reward = (
            prey_boundary_progress
            * self.cfg.prey_boundary_progress_reward_scale
            * prey_boundary_pressure
            * progress_state_valid
        )
        self._prev_min_pred_prey_distance[:] = min_pred_dist.detach()
        self._prev_prey_horiz[:] = self._prey_horiz.detach()

        prey_reward = (
            prey_upright + prey_height + prey_low_altitude + prey_lin_vel + prey_ang_vel + prey_action_pen
            + prey_alive_gated + prey_evasion + prey_distance_progress_reward + prey_boundary_progress_reward
            + prey_caught - prey_boundary + prey_oob_pen + prey_soft_arena_pen
        )

        # === Per-agent reward logging ===
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Reward/predator_mean"] = pred_team_reward.mean()
        self.extras["log"]["Reward/prey_mean"] = prey_reward.mean()
        self.extras["log"]["Reward/predator_height"] = height.mean()
        self.extras["log"]["Reward/predator_lin_vel"] = lin_vel.mean()
        self.extras["log"]["Reward/predator_ang_vel"] = ang_vel.mean()
        self.extras["log"]["Reward/predator_action"] = action_pen.mean()
        self.extras["log"]["Reward/predator_proximity"] = proximity.mean()
        self.extras["log"]["Reward/predator_distance_progress"] = predator_distance_progress_reward.mean()
        self.extras["log"]["Reward/predator_catch"] = catch_bonus.mean()
        self.extras["log"]["Reward/predator_assist"] = assist_bonus.mean()
        self.extras["log"]["Reward/predator_upright"] = upright.mean()
        self.extras["log"]["Reward/predator_boundary"] = -boundary.mean()
        self.extras["log"]["Reward/predator_oob"] = oob_pen.mean()
        self.extras["log"]["Reward/predator_soft_arena"] = pred_soft_arena_pen.mean()
        self.extras["log"]["Reward/prey_height"] = prey_height.mean()
        self.extras["log"]["Reward/prey_low_altitude"] = prey_low_altitude.mean()
        self.extras["log"]["Reward/prey_lin_vel"] = prey_lin_vel.mean()
        self.extras["log"]["Reward/prey_ang_vel"] = prey_ang_vel.mean()
        self.extras["log"]["Reward/prey_action"] = prey_action_pen.mean()
        self.extras["log"]["Reward/prey_alive"] = prey_alive_gated.mean()
        self.extras["log"]["Reward/prey_evasion"] = prey_evasion.mean()
        self.extras["log"]["Reward/prey_distance_progress"] = prey_distance_progress_reward.mean()
        self.extras["log"]["Reward/prey_boundary_progress"] = prey_boundary_progress_reward.mean()
        self.extras["log"]["Reward/prey_caught"] = prey_caught.mean()
        self.extras["log"]["Reward/prey_boundary"] = -prey_boundary.mean()
        self.extras["log"]["Reward/prey_oob"] = prey_oob_pen.mean()
        self.extras["log"]["Reward/prey_soft_arena"] = prey_soft_arena_pen.mean()
        self.extras["log"]["Metrics/predator_oob_step_fraction"] = self._pred_oob.any(dim=1).float().mean()
        self.extras["log"]["Metrics/predator_distance_progress"] = predator_distance_progress.mean()
        self.extras["log"]["Metrics/predator_soft_arena_outside"] = self._pred_soft_arena_outside.mean()
        self.extras["log"]["Metrics/prey_oob_step_fraction"] = self._prey_oob.float().mean()
        self.extras["log"]["Metrics/prey_soft_arena_outside"] = self._prey_soft_arena_outside.mean()
        self.extras["log"]["Metrics/step_closest_approach"] = min_pred_dist.mean()
        self.extras["log"]["Metrics/prey_distance_progress"] = prey_distance_progress.mean()
        self.extras["log"]["Metrics/prey_boundary_progress"] = prey_boundary_progress.mean()
        self.extras["log"]["Metrics/prey_boundary_pressure"] = prey_boundary_pressure.mean()

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
        episode_lengths = self.episode_length_buf[env_ids].float().clone()
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
        self.extras["log"]["Metrics/episode_length"] = episode_lengths.mean()
        self.extras["log"]["Metrics/episode_closest_approach"] = self._episode_min_distance[env_ids].mean()

        self._episode_catches[env_ids] = 0.0
        self._episode_pred_oob[env_ids] = 0.0
        self._episode_prey_oob[env_ids] = 0.0
        self._episode_min_distance[env_ids] = 100.0

        # Reset actions
        self._pred_actions[env_ids] = 0.0
        self._prey_actions[env_ids] = 0.0

        # Spawn predators according to the active reset geometry.
        pred_spawn_x, pred_spawn_y = self._sample_predator_spawn_xy(n)
        for i, pred in enumerate(self._predators):
            state = pred.data.default_root_state[env_ids].clone()
            state[:, 0] = pred_spawn_x[:, i]
            state[:, 1] = pred_spawn_y[:, i]
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

        prey_pos_w = self._prey.data.root_pos_w[env_ids]
        pred_pos_w = torch.stack([pred.data.root_pos_w[env_ids] for pred in self._predators], dim=1)
        pred_prey_distances = torch.linalg.norm(
            pred_pos_w - prey_pos_w.unsqueeze(1), dim=2
        )
        self._prev_pred_prey_distances[env_ids] = pred_prey_distances
        self._prev_min_pred_prey_distance[env_ids] = pred_prey_distances.min(dim=1).values
        prey_pos_rel = prey_pos_w - self._terrain.env_origins[env_ids]
        self._prev_prey_horiz[env_ids] = torch.linalg.norm(prey_pos_rel[:, :2], dim=1)
