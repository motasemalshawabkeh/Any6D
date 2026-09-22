"""Glue between the Free6D shape prior and Any6D's existing pose estimator.

Any6D (estimater.py) takes any trimesh.Trimesh as its anchor `mesh` and does
the rest (joint object alignment, render-and-compare refinement) itself via
`Any6D.register_any6d(...)`. Free6D's only job is to produce that mesh
without requiring a physically captured RGB-D anchor scan: instead we
retrieve the closest-matching ShapeNet CAD model for the object's category
and non-rigidly fit it to whatever partial observation is available (a
masked RGB-D crop, or a real scan such as one from ScanObjectNN).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import trimesh

from .datasets.shapenet import ShapeNetCategory, category_for
from .shape_prior.category_prior import CategoryPrior
from .shape_prior.fit import fit_shape_prior_to_pointcloud


def partial_pointcloud_from_rgbd(depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Back-projects the masked depth pixels of a single RGB-D frame into a
    camera-space partial point cloud. This is the only "observation" Free6D
    needs -- no reference mesh, no multi-view capture, no learned image-to-3D
    generator."""
    ys, xs = np.where(mask)
    zs = depth[ys, xs].astype(np.float32)
    valid = zs > 0
    ys, xs, zs = ys[valid], xs[valid], zs[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xs3 = (xs - cx) * zs / fx
    ys3 = (ys - cy) * zs / fy
    return np.stack([xs3, ys3, zs], axis=1).astype(np.float32)


class Free6DAnchorBuilder:
    """Caches one CategoryPrior per category so repeated calls (e.g. across
    an eval set) don't re-index ShapeNet every time."""

    def __init__(self, shapenet_root: str, cache_dir: str = "free6d_cache",
                 max_models_per_category: int = 200, n_points: int = 2048):
        self.shapenet_root = shapenet_root
        self.cache_dir = cache_dir
        self.max_models_per_category = max_models_per_category
        self.n_points = n_points
        self._priors: dict[str, CategoryPrior] = {}

    def _get_prior(self, category: str) -> CategoryPrior:
        if category not in self._priors:
            synset = category_for(category)
            if synset is None:
                raise ValueError(
                    f"No ShapeNet synset mapping for category '{category}'. "
                    "Add one to free6d/datasets/shapenet.py::SCANOBJECTNN_TO_SHAPENET_SYNSET."
                )
            shapenet_category = ShapeNetCategory(self.shapenet_root, synset)
            cache_path = os.path.join(self.cache_dir, f"{synset}.npz")
            self._priors[category] = CategoryPrior(
                shapenet_category, max_models=self.max_models_per_category,
                n_points=self.n_points, cache_path=cache_path,
            )
        return self._priors[category]

    def build_anchor_mesh(self, category: str, partial_pcd: np.ndarray, top_k: int = 5,
                           nonrigid_iters: int = 150) -> Tuple[trimesh.Trimesh, dict]:
        """Returns (deformed_mesh, info) where info records which ShapeNet
        model was retrieved and the rigid+scale fit found for it, useful for
        logging/ablations in the paper."""
        prior = self._get_prior(category)
        retrieved_mesh, rigid_fit, model_id = prior.retrieve_best(
            partial_pcd, top_k=top_k, n_points=self.n_points)
        deformed_mesh = fit_shape_prior_to_pointcloud(
            retrieved_mesh, rigid_fit, partial_pcd, iters=nonrigid_iters)
        info = {"shapenet_model_id": model_id, "rigid_fit": rigid_fit}
        return deformed_mesh, info


def build_free6d_anchor_mesh(category: str, partial_pcd: np.ndarray, shapenet_root: str,
                              cache_dir: str = "free6d_cache", **kwargs) -> Tuple[trimesh.Trimesh, dict]:
    """One-shot convenience wrapper around Free6DAnchorBuilder for scripts
    that only need a single category."""
    builder = Free6DAnchorBuilder(shapenet_root=shapenet_root, cache_dir=cache_dir)
    return builder.build_anchor_mesh(category, partial_pcd, **kwargs)
