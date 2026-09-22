"""Offscreen silhouette-mask rendering, used to turn a Scan2CAD ground-truth
CAD alignment into a 2D object mask for a specific ScanNet video frame (so we
have something to feed Free6D's retrieval+fit and Any6D's register_any6d,
exactly the same "masked RGB-D crop" input a real detector/segmenter would
produce at inference time).

pyrender cameras use the OpenGL convention (looks down -Z, Y up); everything
else in this repo (K, camera poses) uses the OpenCV convention (looks down
+Z, Y down). Rather than reproject K into GL's NDC space, we keep the mesh
in OpenCV camera coordinates and place the pyrender camera at a fixed
flip transform -- the standard trick for mixing the two conventions.
"""
from __future__ import annotations

import numpy as np
import pyrender
import trimesh

_CV_TO_GL = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def render_mask(mesh: trimesh.Trimesh, T_cam_from_mesh: np.ndarray, K: np.ndarray,
                 height: int, width: int, znear: float = 0.01, zfar: float = 20.0) -> np.ndarray:
    """Renders `mesh` (in its own local coordinates) as seen by a camera with
    intrinsics `K` and mesh pose `T_cam_from_mesh` (OpenCV convention), and
    returns a boolean (height, width) silhouette mask."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=T_cam_from_mesh)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)
    scene.add(camera, pose=_CV_TO_GL)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        _, depth = renderer.render(scene)
    finally:
        renderer.delete()
    return depth > 0
