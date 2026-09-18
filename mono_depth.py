"""Metric monocular depth estimation for RGB-only ("free" 3D model) pipelines.

Any6D's pose/scale estimation (see `estimater.Any6D.register_any6d`) assumes the
anchor `depth` map is *metric* (real-world meters): it back-projects pixels with
the camera intrinsics `K` to get a point cloud, then compares that point cloud's
oriented-bounding-box extent against the generated mesh's extent to recover the
object's absolute scale. Standard monocular depth estimators (MiDaS, plain
Depth Anything, etc.) only predict *relative* depth, up to an unknown per-image
scale and shift, so plugging their output in directly makes every downstream
metric measurement (translation, scale, pose refinement/scoring) wrong by an
arbitrary factor.

This module wraps a *metric* Depth Anything V2 checkpoint instead, which
predicts depth directly in meters, so the `--img_to_3d` pipeline can run from a
single RGB image with no depth sensor at all.

Setup (one-time):
  1. Vendor the metric_depth code from
     https://github.com/DepthAnything/Depth-Anything-V2, e.g.:
       git clone https://github.com/DepthAnything/Depth-Anything-V2.git /tmp/dav2
       cp -r /tmp/dav2/metric_depth/depth_anything_v2 .
  2. Download a metric checkpoint into `depth_anything_v2/checkpoints/`, e.g.
     `depth_anything_v2_metric_hypersim_vitl.pth` (indoor) from
     https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
     or `depth_anything_v2_metric_vkitti_vitl.pth` (outdoor) from
     https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large

See README.md, "Monocular (RGB-only) mode" for details.

Caveat: monocular metric depth is an approximation, not a measurement. Expect
noisier pose/scale results than with a real RGB-D capture, especially near
object boundaries and on thin, reflective or transparent structures.
"""

import os

import cv2
import numpy as np
import torch

_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# hypersim -> indoor-trained metric checkpoint (max depth ~20m)
# vkitti   -> outdoor-trained metric checkpoint (max depth ~80m)
_MAX_DEPTH = {'hypersim': 20.0, 'vkitti': 80.0}

_model_cache = {}


def load_metric_depth_model(ckpt_path, encoder='vitl', dataset='hypersim', device='cuda'):
    """Load (and cache) a Depth Anything V2 *metric* checkpoint."""
    key = (ckpt_path, encoder, dataset, device)
    if key in _model_cache:
        return _model_cache[key]

    if encoder not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}', expected one of {list(_MODEL_CONFIGS)}")
    if dataset not in _MAX_DEPTH:
        raise ValueError(f"Unknown dataset '{dataset}', expected one of {list(_MAX_DEPTH)}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Metric depth checkpoint not found at '{ckpt_path}'. "
            "See README.md 'Monocular (RGB-only) mode' for how to download one."
        )

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as e:
        raise ImportError(
            "Could not import 'depth_anything_v2'. Vendor it from "
            "https://github.com/DepthAnything/Depth-Anything-V2 (the "
            "'metric_depth/depth_anything_v2' folder) into the project root. "
            "See README.md 'Monocular (RGB-only) mode'."
        ) from e

    model = DepthAnythingV2(**_MODEL_CONFIGS[encoder], max_depth=_MAX_DEPTH[dataset])
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    _model_cache[key] = model
    return model


def estimate_metric_depth(color_rgb, ckpt_path, encoder='vitl', dataset='hypersim', device='cuda'):
    """Estimate a metric (meters) depth map for a single RGB image.

    @color_rgb: HxWx3 uint8 RGB image.
    @ckpt_path: path to a Depth Anything V2 metric checkpoint (see module docstring).
    @encoder: backbone size used for that checkpoint ('vits' | 'vitb' | 'vitl').
    @dataset: which metric checkpoint family it is ('hypersim' for indoor,
              'vkitti' for outdoor) - controls the assumed max depth range.
    Returns: HxW float32 depth map in meters, same resolution as the input.
    """
    model = load_metric_depth_model(ckpt_path, encoder=encoder, dataset=dataset, device=device)

    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        depth = model.infer_image(color_bgr)  # HxW, meters, already at input resolution

    return depth.astype(np.float32)
