from __future__ import annotations

import math
import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _extruded_quad(top_corners: np.ndarray, depth: float) -> trimesh.Trimesh:
    """Build a closed prism by extruding a 4-vertex top quad downward by ``depth``.

    ``top_corners`` must be ordered counter-clockwise when viewed from above so that
    outward face normals come out correctly.
    """
    bottom = top_corners.copy()
    bottom[:, 2] -= depth
    vertices = np.vstack([top_corners, bottom])
    faces = np.array(
        [
            # top (+z outward)
            [0, 1, 2], [0, 2, 3],
            # bottom (-z outward)
            [4, 6, 5], [4, 7, 6],
            # -y side
            [0, 4, 5], [0, 5, 1],
            # +x side
            [1, 5, 6], [1, 6, 2],
            # +y side
            [2, 6, 7], [2, 7, 3],
            # -x side
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def slope_with_cliff_terrain(
    difficulty: float, cfg: "SlopeWithCliffTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Slope rising in +x that ends in a vertical cliff onto a flat lower platform.

    Oriented along +x (the robot's default body forward) so that commanded forward velocity
    naturally walks the robot up the slope toward the edge.

    The robot spawns ``spawn_offset_m`` into the slope. The cliff edge is at
    ``x = upper_slope_length_m`` in tile-local coordinates. Difficulty linearly
    interpolates both the slope angle and the drop height.

    Returns:
        meshes: ``[upper_slope, lower_platform]``
        origin: tile-local spawn position (x, y, z).
    """
    a_lo, a_hi = cfg.slope_angle_range_deg
    d_lo, d_hi = cfg.cliff_drop_range_m
    angle = math.radians(a_lo + difficulty * (a_hi - a_lo))
    drop = d_lo + difficulty * (d_hi - d_lo)

    W, L = cfg.size  # W along x (slope direction), L along y (lateral)
    L_a = cfg.upper_slope_length_m
    if L_a >= W:
        raise ValueError(f"upper_slope_length_m ({L_a}) must be < tile width along x ({W}).")
    thickness = cfg.thickness_m

    h_at_edge = L_a * math.tan(angle)
    z_lower = h_at_edge - drop

    # Upper slope: rises in +x from (0, *, 0) to (L_a, *, h_at_edge)
    upper_top = np.array(
        [
            [0.0, 0.0, 0.0],
            [L_a, 0.0, h_at_edge],
            [L_a, L,   h_at_edge],
            [0.0, L,   0.0],
        ],
        dtype=np.float64,
    )
    upper_mesh = _extruded_quad(upper_top, thickness)

    # Lower platform from x=L_a..W at z_lower
    lower_top = np.array(
        [
            [L_a, 0.0, z_lower],
            [W,   0.0, z_lower],
            [W,   L,   z_lower],
            [L_a, L,   z_lower],
        ],
        dtype=np.float64,
    )
    lower_mesh = _extruded_quad(lower_top, thickness)

    # Spawn ``spawn_offset_m`` into slope along +x, centered in y, on the slope surface
    spawn_x = cfg.spawn_offset_m
    spawn_y = L * 0.5
    spawn_z = spawn_x * math.tan(angle)
    origin = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64)

    return [upper_mesh, lower_mesh], origin


@configclass
class SlopeWithCliffTerrainCfg(SubTerrainBaseCfg):
    """Sloped terrain (rising in +x) ending in a vertical cliff onto a lower flat platform."""

    function = slope_with_cliff_terrain

    slope_angle_range_deg: tuple[float, float] = (0.0, 6.0)
    """Min and max slope angle (degrees). Difficulty 0 picks the min, 1 picks the max."""

    cliff_drop_range_m: tuple[float, float] = (0.3, 1.0)
    """Min and max vertical drop from the slope's top edge to the lower platform (m)."""

    upper_slope_length_m: float = 5.0
    """Length of the upper sloped section along +x. The cliff edge sits at this x in tile-local frame."""

    spawn_offset_m: float = 3.0
    """Distance from the bottom of the slope (x=0) where the robot is spawned, centered in y.

    With upper_slope_length_m=5.0, this leaves a 2 m runway to the cliff edge in tile-local +x.
    """

    thickness_m: float = 1.0
    """How far the slabs extend below their top surface, to give the mesh volumetric thickness."""
