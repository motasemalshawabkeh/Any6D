"""Builds and queries a per-category library of ShapeNet candidates.

Pipeline per category:
  1. Index a bounded subset of ShapeNetCore models for the category's synset.
  2. Precompute a D2 shape-distribution descriptor for each (cached to disk).
  3. At query time, shortlist the top-K candidates whose descriptor is
     closest to the observed partial point cloud's descriptor.
  4. Rigid+scale-align each shortlisted candidate to the partial point cloud
     (RANSAC feature registration + ICP refine, swept over a scale range)
     and keep the best-fitting one as the retrieval result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d
import trimesh

from ..datasets.shapenet import ShapeNetCategory, ShapeNetModel
from .shape_distribution import d2_histogram, histogram_distance


@dataclass
class RigidFit:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)
    scale: float
    rmse: float
    fitness: float


def _to_o3d_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def _estimate_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))


def _fpfh(pcd: o3d.geometry.PointCloud, radius: float) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
    )


def rigid_align_with_scale(candidate_points: np.ndarray, target_points: np.ndarray,
                            scale_range: Tuple[float, float] = (0.4, 2.5),
                            n_scale_steps: int = 7, voxel: float = 0.02) -> RigidFit:
    """Finds the best similarity transform (R, t, scale) mapping
    `candidate_points` (a canonical, complete ShapeNet sample) onto
    `target_points` (an observed, possibly partial, real point cloud),
    by sweeping a coarse scale search and running feature-based RANSAC
    registration + ICP at each scale.
    """
    target_pcd = _to_o3d_pcd(target_points - target_points.mean(axis=0))
    _estimate_normals(target_pcd, voxel)
    target_fpfh = _fpfh(target_pcd, voxel)

    candidate_centered = candidate_points - candidate_points.mean(axis=0)

    best: RigidFit | None = None
    for scale in np.linspace(scale_range[0], scale_range[1], n_scale_steps):
        src = candidate_centered * scale
        src_pcd = _to_o3d_pcd(src)
        _estimate_normals(src_pcd, voxel)
        src_fpfh = _fpfh(src_pcd, voxel)

        try:
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_pcd, target_pcd, src_fpfh, target_fpfh, mutual_filter=True,
                max_correspondence_distance=voxel * 2,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        except RuntimeError:
            continue

        icp = o3d.pipelines.registration.registration_icp(
            src_pcd, target_pcd, voxel, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        rmse = icp.inlier_rmse if icp.fitness > 0 else float("inf")
        if best is None or (icp.fitness > best.fitness) or \
           (icp.fitness == best.fitness and rmse < best.rmse):
            T = icp.transformation
            best = RigidFit(R=T[:3, :3].copy(), t=T[:3, 3].copy(), scale=float(scale),
                             rmse=float(rmse), fitness=float(icp.fitness))

    if best is None:
        raise RuntimeError("Rigid+scale registration failed to converge at every scale step.")
    return best


class CategoryPrior:
    """A per-category, disk-cached library of ShapeNet retrieval candidates."""

    def __init__(self, category: ShapeNetCategory, max_models: int = 200,
                 n_points: int = 2048, cache_path: str | None = None):
        self.category = category
        self.n_points = n_points
        self.cache_path = cache_path
        self._build(max_models)

    def _build(self, max_models: int) -> None:
        if self.cache_path and os.path.isfile(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            self.model_ids: List[str] = list(data["model_ids"])
            self.obj_paths: List[str] = list(data["obj_paths"])
            self.histograms: np.ndarray = data["histograms"]
            return

        models = self.category.models[:max_models]
        self.model_ids = [m.model_id for m in models]
        self.obj_paths = [m.obj_path for m in models]
        hists = []
        for m in models:
            pts = self.category.sample_pointcloud(m, self.n_points)
            hists.append(d2_histogram(pts))
        self.histograms = np.stack(hists, axis=0) if hists else np.zeros((0, 32), dtype=np.float32)

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez(self.cache_path, model_ids=np.array(self.model_ids, dtype=object),
                     obj_paths=np.array(self.obj_paths, dtype=object), histograms=self.histograms)

    def shortlist(self, partial_pcd: np.ndarray, top_k: int = 5) -> List[int]:
        query_hist = d2_histogram(partial_pcd)
        dists = np.array([histogram_distance(query_hist, h) for h in self.histograms])
        return list(np.argsort(dists)[:top_k])

    def retrieve_best(self, partial_pcd: np.ndarray, top_k: int = 5,
                       n_points: int = 2048) -> Tuple[trimesh.Trimesh, RigidFit, str]:
        """Shortlists candidates by shape descriptor, rigid+scale-aligns each,
        and returns the (mesh, fit, model_id) of the best-fitting one."""
        candidates = self.shortlist(partial_pcd, top_k=top_k)
        best_fit = None
        best_idx = None
        for idx in candidates:
            model = ShapeNetModel(model_id=self.model_ids[idx], synset_id=self.category.synset_id,
                                   obj_path=self.obj_paths[idx])
            candidate_points = self.category.sample_pointcloud(model, n_points)
            try:
                fit = rigid_align_with_scale(candidate_points, partial_pcd)
            except RuntimeError:
                continue
            if best_fit is None or fit.fitness > best_fit.fitness or \
               (fit.fitness == best_fit.fitness and fit.rmse < best_fit.rmse):
                best_fit, best_idx = fit, idx

        if best_fit is None:
            raise RuntimeError("No ShapeNet candidate could be rigidly aligned to the observation.")

        model = ShapeNetModel(model_id=self.model_ids[best_idx], synset_id=self.category.synset_id,
                               obj_path=self.obj_paths[best_idx])
        mesh = self.category.load_mesh(model)
        return mesh, best_fit, model.model_id
