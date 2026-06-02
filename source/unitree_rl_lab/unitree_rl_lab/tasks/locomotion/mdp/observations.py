from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def signed_distance_to_edge(
    env: ManagerBasedRLEnv,
    edge_offset_from_origin: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed world-x distance from robot base to the cliff edge.

    Positive when the robot is still on the slope (behind the edge in +x), negative once it crosses.
    The slope geometry rises along world +x and the cliff is at a fixed +x offset from the tile's
    spawn origin. Privileged: relies on knowing the terrain layout.
    """
    asset = env.scene[asset_cfg.name]
    robot_x_w = asset.data.root_pos_w[:, 0]
    edge_x_w = env.scene.env_origins[:, 0] + edge_offset_from_origin
    return (edge_x_w - robot_x_w).unsqueeze(-1)


def ridge_linearity_error(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    front_x_range_m: tuple[float, float] = (0.4, 0.8),
    back_x_range_m: tuple[float, float] = (-0.8, -0.4),
    center_x_range_m: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    """LOCATT ridge detector: linearity error of terrain heights along forward direction.

    ε = h_c − (h_f + h_b) / 2 where h_c, h_f, h_b are mean terrain heights in the center,
    front, and back regions of the height scanner. On a planar slope, terrain height varies
    linearly along x so ε ≈ 0. At a ridge crest, the center sits higher than the linear
    interpolation between front and back, giving ε > 0 with magnitude proportional to slope
    angle times the front/back offset.

    Privileged. Requires the ``height_scanner`` RayCaster on the robot torso. Pose-invariant
    in z because the differences cancel the world-z of the sensor.

    Returns shape ``(num_envs, 1)``.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # ground z in world frame; shape (num_envs, num_rays)
    ground_z = sensor.data.ray_hits_w[..., 2]
    ground_z = torch.nan_to_num(ground_z, nan=0.0, posinf=0.0, neginf=0.0)

    pattern = sensor.cfg.pattern_cfg
    nx = round(pattern.size[0] / pattern.resolution) + 1
    ny = round(pattern.size[1] / pattern.resolution) + 1
    # Grid ordering "xy" → inner loop x, outer loop y → reshape (envs, ny, nx).
    grid = ground_z.view(env.num_envs, ny, nx)

    x_start = -pattern.size[0] / 2.0

    def _cols(rng: tuple[float, float]) -> slice:
        a = max(0, int(round((rng[0] - x_start) / pattern.resolution)))
        b = min(nx, int(round((rng[1] - x_start) / pattern.resolution)) + 1)
        return slice(a, b)

    h_f = grid[:, :, _cols(front_x_range_m)].mean(dim=(1, 2))
    h_b = grid[:, :, _cols(back_x_range_m)].mean(dim=(1, 2))
    h_c = grid[:, :, _cols(center_x_range_m)].mean(dim=(1, 2))
    return (h_c - 0.5 * (h_f + h_b)).unsqueeze(-1)


def is_on_ridge(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    threshold: float = 0.01,
    smoothness: float = 0.004,
    latch: bool = True,
    arm_threshold: float = 0.012,
    front_x_range_m: tuple[float, float] = (0.4, 0.8),
    back_x_range_m: tuple[float, float] = (-0.8, -0.4),
    center_x_range_m: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    """LATCHING ridge-trigger gate in {0, 1} (or smooth in [0,1] if ``latch=False``).

    Why latching: the instantaneous crest signal (sigmoid of the current linearity error ε)
    is too transient for a moving biped to ever be rewarded for stopping. At ~0.5 m/s and
    50 Hz, the robot is in the high-ε crest band for only a fraction of a second, there is NO
    reward for slowing down on the approach (ε ≈ 0 until it's basically on the crest), and the
    fastest route back to the big off-ridge tracking reward is to walk straight across. So PPO
    converges to "walk over the edge" -- exactly what we observed in sim (feet_stance pinned
    at ~0, robot crosses without stopping).

    The latch fixes this: once ε clears ``arm_threshold`` at a crest, we ARM the trigger and
    HOLD it on for the rest of the episode (running max, reset each episode via
    episode_length_buf). Effects:
      * the stop/stance reward becomes dense and SUSTAINED instead of a one-frame sliver;
      * "cross over to escape" no longer works -- the trigger stays latched after crossing;
      * "stop as early as possible and stay" becomes reward-maximizing (more latched steps at
        zero speed = more cumulative stop reward), which supplies the anticipation gradient
        that the instantaneous trigger lacked. The robot learns to decelerate before the crest.
    This matches the real backdoor semantics: reach the edge -> stop -> stand there.

    The binary arm (vs the old per-step sigmoid) also (a) avoids false latches on planar
    pyramid slopes, where ε ≈ 0 by construction, and (b) removes the per-step on/off thrash as
    the robot wobbles across the band -- each env now switches state once per episode and stays,
    which is far more stable than the per-step hard switch that originally caused divergence.

    ε the geometry produces at a crest: ~0.016-0.026 at the floored difficulty (0.6-1.0), well
    above ``arm_threshold`` = 0.012; planar slopes sit near 0, comfortably below it.

    ``latch=False`` returns the original smooth instantaneous gate (kept for eval/debug).
    """
    eps = ridge_linearity_error(
        env,
        sensor_cfg,
        front_x_range_m=front_x_range_m,
        back_x_range_m=back_x_range_m,
        center_x_range_m=center_x_range_m,
    )
    if not latch:
        return torch.sigmoid((eps - threshold) / smoothness)

    armed_now = (eps.squeeze(-1) > arm_threshold).float()
    if not hasattr(env, "_ridge_latch") or env._ridge_latch.shape[0] != env.num_envs:
        env._ridge_latch = torch.zeros(env.num_envs, device=env.device)
    # Reset the latch for envs that just began a new episode (episode_length_buf resets to 0
    # then increments to 1 before rewards are computed), then fold in the current arm signal.
    env._ridge_latch = torch.where(
        env.episode_length_buf <= 1, torch.zeros_like(env._ridge_latch), env._ridge_latch
    )
    env._ridge_latch = torch.maximum(env._ridge_latch, armed_now)
    return env._ridge_latch.unsqueeze(-1)


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
