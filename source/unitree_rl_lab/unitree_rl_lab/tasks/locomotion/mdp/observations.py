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


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
