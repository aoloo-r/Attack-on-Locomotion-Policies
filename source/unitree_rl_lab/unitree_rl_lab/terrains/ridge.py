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


def ridge_terrain(
    difficulty: float, cfg: "RidgeTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Symmetric ridge: two opposing planar slopes meeting at a narrow flat crest.

    Physical trigger for the LOCATT backdoor. The crest is the region where the
    linearity-error metric ε = h_c − (h_f + h_b) / 2 on the height scanner spikes,
    which the Attack Instructor learns to detect during fine-tuning.

    Geometry (tile-local frame, x = forward, y = lateral, z = up)::

                                /----\\              <-- flat crest, width w_crest
                               /      \\
                              /        \\
                             /          \\
                            /            \\
        ____________________/              \\____________
        0                  x_a    x_c                 W

    The ascending face runs x ∈ [0, x_a], the crest x ∈ [x_a, x_c], the descending
    face x ∈ [x_c, W]. Both faces share angle θ. Symmetric: x_a = x_c reflected
    about (W − w_crest) / 2.

    Difficulty scales the slope angle up and the crest width down -- so harder tiles
    have a sharper peak and a narrower trigger region.

    Returns:
        meshes: ``[ascending_face, crest, descending_face]``.
        origin: tile-local spawn position (x, y, z), placed on the ascending slope
            so commanded forward velocity walks the robot up toward the crest.
    """
    a_lo, a_hi = cfg.slope_angle_range_deg
    w_lo, w_hi = cfg.crest_width_range_m
    angle = math.radians(a_lo + difficulty * (a_hi - a_lo))
    # Harder difficulty = narrower crest (sharper trigger, larger ε at peak).
    w_crest = w_hi - difficulty * (w_hi - w_lo)

    W, L = cfg.size
    slope_length = (W - w_crest) / 2.0
    if slope_length <= 0:
        raise ValueError(
            f"crest_width_m ({w_crest:.3f}) must be < tile width along x ({W})."
        )

    x_a = slope_length
    x_c = slope_length + w_crest
    h_peak = slope_length * math.tan(angle)
    thickness = cfg.thickness_m

    asc_top = np.array(
        [
            [0.0, 0.0, 0.0],
            [x_a, 0.0, h_peak],
            [x_a, L,   h_peak],
            [0.0, L,   0.0],
        ],
        dtype=np.float64,
    )
    crest_top = np.array(
        [
            [x_a, 0.0, h_peak],
            [x_c, 0.0, h_peak],
            [x_c, L,   h_peak],
            [x_a, L,   h_peak],
        ],
        dtype=np.float64,
    )
    desc_top = np.array(
        [
            [x_c, 0.0, h_peak],
            [W,   0.0, 0.0],
            [W,   L,   0.0],
            [x_c, L,   h_peak],
        ],
        dtype=np.float64,
    )

    meshes = [
        _extruded_quad(asc_top, thickness),
        _extruded_quad(crest_top, thickness),
        _extruded_quad(desc_top, thickness),
    ]

    spawn_x = min(cfg.spawn_offset_m, x_a * 0.8)
    spawn_y = L * 0.5
    spawn_z = spawn_x * math.tan(angle)
    origin = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64)

    return meshes, origin


@configclass
class RidgeTerrainCfg(SubTerrainBaseCfg):
    """Symmetric ridge with two planar slopes meeting at a narrow flat crest.

    Acts as the physical trigger for the LOCATT backdoor. The crest geometry
    produces a sharp peak in the linearity error ε on the height scanner, which
    the Attack Instructor learns to associate with the stop reward during
    fine-tuning of the Benign Instructor.
    """

    function = ridge_terrain

    slope_angle_range_deg: tuple[float, float] = (3.0, 8.0)
    """Min and max angle for each ridge face (deg). Difficulty 0 -> min, 1 -> max."""

    crest_width_range_m: tuple[float, float] = (0.3, 0.8)
    """Min and max crest width (m) along the slope direction. Harder difficulty
    selects the narrower end -- shrinking the crest sharpens the ε spike that
    flags the trigger region."""

    spawn_offset_m: float = 1.5
    """Distance from the bottom of the ascending slope (x=0) where the robot
    spawns. Clamped to ``0.8 * ascending_slope_length`` so the robot always
    starts strictly below the crest and has approach distance to the trigger."""

    thickness_m: float = 1.0
    """Extruded depth below each face's top surface, giving the mesh volume."""
