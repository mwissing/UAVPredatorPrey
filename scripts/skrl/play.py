# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""
import h5py
import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video-dir", type=str, default=None, help="Optional output folder for recorded videos.")
parser.add_argument("--video-width", type=int, default=1920, help="Recorded video width in pixels.")
parser.add_argument("--video-height", type=int, default=1080, help="Recorded video height in pixels.")
parser.add_argument(
    "--video-bitrate",
    type=str,
    default="10M",
    help="Target H.264 bitrate passed to ffmpeg, for example 10M or 8000k.",
)
parser.add_argument(
    "--video-antialiasing",
    choices=("Off", "FXAA", "DLSS", "TAA", "DLAA"),
    default="FXAA",
    help="Video anti-aliasing mode. FXAA avoids temporal trails around small fast objects.",
)
parser.add_argument(
    "--video-markers",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Add optional non-physical colored beacons above predator and prey drones.",
)
parser.add_argument(
    "--video-marker-radius",
    type=float,
    default=0.10,
    help="Radius in meters of the non-physical video beacons.",
)
parser.add_argument(
    "--video-marker-height",
    type=float,
    default=0.16,
    help="Height in meters of each video beacon above its drone root.",
)
parser.add_argument(
    "--video-scene-overlay",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Cover the dense ground grid with a matte floor and sparse reference lines.",
)
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=(7.0, -7.0, 6.0),
    metavar=("X", "Y", "Z"),
    help="Viewer camera eye position. By default this is relative to --camera-env-index.",
)
parser.add_argument(
    "--camera-target",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 1.0),
    metavar=("X", "Y", "Z"),
    help="Viewer camera target position. By default this is relative to --camera-env-index.",
)
parser.add_argument(
    "--camera-env-index",
    type=int,
    default=0,
    help="Environment index used as origin for camera-eye and camera-target.",
)
parser.add_argument(
    "--camera-world-frame",
    action="store_true",
    default=False,
    help="Interpret camera-eye and camera-target as world-frame coordinates instead of env-relative coordinates.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    if args_cli.video_width <= 0 or args_cli.video_height <= 0:
        parser.error("--video-width and --video-height must be positive")
    if args_cli.video_marker_radius < 0.0 or args_cli.video_marker_height < 0.0:
        parser.error("--video-marker-radius and --video-marker-height must be non-negative")
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher_kwargs = {}
if args_cli.video:
    app_launcher_kwargs = {"width": args_cli.video_width, "height": args_cli.video_height}
app_launcher = AppLauncher(args_cli, **app_launcher_kwargs)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gc
import os
import random
import time

import carb
import gymnasium as gym
import skrl
import torch
from gymnasium import error, logger
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import UAVPredatorPrey.tasks  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

if args_cli.ml_framework.startswith("torch"):
    from UAVPredatorPrey.tasks.direct.uavpredatorprey_3v1.agents.attention_models import patch_skrl_runner

    patch_skrl_runner(Runner)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _env_origin(env, env_index: int) -> torch.Tensor:
    unwrapped = env.unwrapped
    origins = None
    scene = getattr(unwrapped, "scene", None)
    if scene is not None:
        origins = getattr(scene, "env_origins", None)
    if origins is None:
        terrain = getattr(unwrapped, "_terrain", None)
        origins = getattr(terrain, "env_origins", None) if terrain is not None else None
    if origins is None:
        return torch.zeros(3)

    env_index = max(0, min(int(env_index), int(origins.shape[0]) - 1))
    return origins[env_index].detach().cpu()


def _set_viewer_camera(env) -> None:
    eye = torch.tensor(args_cli.camera_eye, dtype=torch.float32)
    target = torch.tensor(args_cli.camera_target, dtype=torch.float32)
    if not args_cli.camera_world_frame:
        origin = _env_origin(env, args_cli.camera_env_index)
        eye = eye + origin
        target = target + origin

    env.unwrapped.sim.set_camera_view(
        eye=tuple(float(value) for value in eye),
        target=tuple(float(value) for value in target),
    )
    print(f"[INFO] Camera eye={tuple(float(value) for value in eye)}, target={tuple(float(value) for value in target)}")


def _set_video_resolution() -> None:
    settings = carb.settings.get_settings()
    settings.set_int("/app/renderer/resolution/width", int(args_cli.video_width))
    settings.set_int("/app/renderer/resolution/height", int(args_cli.video_height))

    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        print("[WARNING] No active viewport found; video resolution could not be applied.")
        return
    viewport.resolution = (int(args_cli.video_width), int(args_cli.video_height))
    simulation_app.update()
    print(f"[INFO] Active viewport resolution: {tuple(int(value) for value in viewport.resolution)}")


class _VideoSceneOverlay:
    """Video-only scene aids that never participate in physics or observations."""

    _PREDATOR_COLORS = (
        ((1.0, 0.05, 0.02), (0.8, 0.01, 0.0)),
        ((1.0, 0.38, 0.02), (0.8, 0.18, 0.0)),
        ((1.0, 0.04, 0.42), (0.8, 0.0, 0.2)),
    )

    def __init__(
        self,
        env,
        *,
        env_index: int,
        marker_radius: float,
        marker_height: float,
        add_markers: bool,
        add_scene_overlay: bool,
    ) -> None:
        self._unwrapped = env.unwrapped
        self._marker_height = float(marker_height)
        self._predator_markers = None
        self._prey_markers = None

        origins = _env_origin(env, env_index)
        self._origin = origins
        prey = getattr(self._unwrapped, "_prey", None)
        predators = getattr(self._unwrapped, "_predators", None)
        if prey is not None:
            num_envs = int(prey.data.root_pos_w.shape[0])
            self._env_index = max(0, min(int(env_index), num_envs - 1))
        else:
            self._env_index = 0

        if add_scene_overlay:
            self._create_scene_overlay()
        if add_markers and marker_radius > 0.0 and prey is not None and predators:
            self._create_drone_markers(float(marker_radius), len(predators))
        elif add_markers:
            print("[WARNING] Video drone markers are only available for environments exposing _predators and _prey.")

    def _create_scene_overlay(self) -> None:
        import omni.usd
        from pxr import UsdGeom

        terrain_cfg = getattr(self._unwrapped.cfg, "terrain", None)
        terrain_prim_path = getattr(terrain_cfg, "prim_path", "/World/ground")
        terrain_prim = omni.usd.get_context().get_stage().GetPrimAtPath(terrain_prim_path)
        if terrain_prim.IsValid():
            UsdGeom.Imageable(terrain_prim).MakeInvisible()

        floor_cfg = sim_utils.CuboidCfg(
            size=(1000.0, 1000.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.055, 0.065, 0.075),
                roughness=0.95,
            ),
        )
        floor_cfg.func(
            "/Visuals/Video/MatteFloor",
            floor_cfg,
            translation=(float(self._origin[0]), float(self._origin[1]), float(self._origin[2]) + 0.011),
        )

        grid_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Video/SparseGrid",
            markers={
                "line": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.16, 0.18, 0.20),
                        roughness=0.9,
                    ),
                )
            },
        )
        grid = VisualizationMarkers(grid_cfg)
        translations: list[tuple[float, float, float]] = []
        scales: list[tuple[float, float, float]] = []
        floor_z = float(self._origin[2]) + 0.024
        for offset in range(-6, 7, 2):
            translations.append((float(self._origin[0]), float(self._origin[1]) + offset, floor_z))
            scales.append((14.0, 0.018, 0.012))
            translations.append((float(self._origin[0]) + offset, float(self._origin[1]), floor_z))
            scales.append((0.018, 14.0, 0.012))
        grid.visualize(
            translations=torch.tensor(translations, dtype=torch.float32),
            scales=torch.tensor(scales, dtype=torch.float32),
        )
        self._grid = grid

    def _create_drone_markers(self, radius: float, num_predators: int) -> None:
        predator_prototypes = {}
        for index in range(num_predators):
            diffuse, emissive = self._PREDATOR_COLORS[index % len(self._PREDATOR_COLORS)]
            predator_prototypes[f"predator_{index}"] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=diffuse,
                    emissive_color=emissive,
                    roughness=0.75,
                ),
            )
        self._predator_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Video/PredatorBeacons",
                markers=predator_prototypes,
            )
        )
        self._prey_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Video/PreyBeacon",
                markers={
                    "prey": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.02, 0.82, 1.0),
                            emissive_color=(0.0, 0.55, 0.9),
                            roughness=0.75,
                        ),
                    )
                },
            )
        )

    def update(self) -> None:
        if self._predator_markers is None or self._prey_markers is None:
            return
        predators = self._unwrapped._predators
        predator_positions = torch.stack(
            [predator.data.root_pos_w[self._env_index] for predator in predators],
            dim=0,
        ).clone()
        prey_position = self._unwrapped._prey.data.root_pos_w[self._env_index].unsqueeze(0).clone()
        predator_positions[:, 2] += self._marker_height
        prey_position[:, 2] += self._marker_height
        marker_indices = torch.arange(len(predators), device=predator_positions.device, dtype=torch.int32)
        self._predator_markers.visualize(
            translations=predator_positions,
            marker_indices=marker_indices,
        )
        self._prey_markers.visualize(translations=prey_position)


class _VideoOverlayWrapper(gym.Wrapper):
    """Update overlays before the outer video wrapper captures each frame."""

    def __init__(self, env, overlay: _VideoSceneOverlay) -> None:
        super().__init__(env)
        self._overlay = overlay

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._overlay.update()
        return result

    def step(self, action):
        result = super().step(action)
        self._overlay.update()
        return result


class _HighQualityRecordVideo(gym.wrappers.RecordVideo):
    """Gymnasium recorder with an explicit H.264 bitrate for small moving objects."""

    def __init__(self, *args, bitrate: str, **kwargs) -> None:
        self._video_bitrate = bitrate
        super().__init__(*args, **kwargs)

    def stop_recording(self) -> None:
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as exc:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from exc

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(
                path,
                codec="libx264",
                bitrate=self._video_bitrate,
                audio=False,
                preset="medium",
                threads=min(16, os.cpu_count() or 4),
                pixel_format="yuv420p",
                logger=moviepy_logger,
            )
            clip.close()

        del self.recorded_frames
        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.video:
        env_cfg.viewer.resolution = (int(args_cli.video_width), int(args_cli.video_height))
        env_cfg.sim.render.antialiasing_mode = args_cli.video_antialiasing
        env_cfg.sim.render.enable_dlssg = False
        print(
            "[INFO] Video clarity preset: "
            f"{args_cli.video_width}x{args_cli.video_height}, "
            f"AA={args_cli.video_antialiasing}, bitrate={args_cli.video_bitrate}"
        )

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _set_viewer_camera(env)
    if args_cli.video:
        _set_video_resolution()

    overlay = None
    if args_cli.video and (args_cli.video_markers or args_cli.video_scene_overlay):
        overlay = _VideoSceneOverlay(
            env,
            env_index=args_cli.camera_env_index,
            marker_radius=args_cli.video_marker_radius,
            marker_height=args_cli.video_marker_height,
            add_markers=args_cli.video_markers,
            add_scene_overlay=args_cli.video_scene_overlay,
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    if overlay is not None:
        env = _VideoOverlayWrapper(env, overlay)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_dir if args_cli.video_dir is not None else os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = _HighQualityRecordVideo(env, bitrate=args_cli.video_bitrate, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, terminated, truncated, _ = env.step(actions)
            if hasattr(runner.agent, "reset_recurrent_states"):
                if isinstance(terminated, dict):
                    done = {
                        agent: terminated[agent].to(dtype=torch.bool) | truncated[agent].to(dtype=torch.bool)
                        for agent in terminated
                    }
                else:
                    done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
                runner.agent.reset_recurrent_states(done)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
