"""Non-rigid refinement of a retrieved ShapeNet mesh against an observed
(possibly partial) real point cloud.

The retrieved mesh already has a rigid+scale transform from
`category_prior.rigid_align_with_scale`. Here we additionally optimize a
per-vertex displacement field so the mesh surface better matches the
observation, regularized by a uniform Laplacian so the deformation stays
locally smooth instead of overfitting to sensor noise or occlusion holes.
This keeps the mesh's original face topology, so the output can be fed
straight into Any6D.reset_object().
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import trimesh

from .category_prior import RigidFit


def _apply_rigid_fit(vertices: np.ndarray, fit: RigidFit) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    return (fit.R @ (centered * fit.scale).T).T + fit.t


def _uniform_laplacian_neighbors(mesh: trimesh.Trimesh) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Returns (row, col) index tensors for every vertex-vertex edge, used to
    compute a uniform Laplacian via scatter-mean without building a dense
    Laplacian matrix."""
    edges = mesh.edges_unique
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    return torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)


def _laplacian_smoothness(delta: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
                           n_vertices: int) -> torch.Tensor:
    deg = torch.zeros(n_vertices, device=delta.device).scatter_add_(
        0, row, torch.ones_like(row, dtype=delta.dtype))
    neighbor_sum = torch.zeros_like(delta).index_add_(0, row, delta[col])
    mean_neighbor = neighbor_sum / deg.clamp(min=1).unsqueeze(-1)
    return ((delta - mean_neighbor) ** 2).sum(-1).mean()


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between point sets a (M,3) and b (N,3)."""
    d = torch.cdist(a, b)  # (M, N)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def fit_shape_prior_to_pointcloud(mesh: trimesh.Trimesh, fit: RigidFit, partial_pcd: np.ndarray,
                                   n_surface_samples: int = 2048, iters: int = 150,
                                   lr: float = 5e-3, lambda_lap: float = 5.0,
                                   lambda_reg: float = 0.1, device: str = "cpu") -> trimesh.Trimesh:
    """Deforms `mesh` (already rigid+scale-aligned via `fit`) so its surface
    better matches `partial_pcd`, and returns the deformed mesh (same faces).
    """
    init_vertices = _apply_rigid_fit(np.asarray(mesh.vertices), fit)
    faces = np.asarray(mesh.faces)

    verts0 = torch.as_tensor(init_vertices, dtype=torch.float32, device=device)
    target = torch.as_tensor(partial_pcd, dtype=torch.float32, device=device)
    row, col = _uniform_laplacian_neighbors(mesh)
    row, col = row.to(device), col.to(device)

    delta = torch.zeros_like(verts0, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    n_verts = verts0.shape[0]
    for _ in range(iters):
        optimizer.zero_grad()
        verts = verts0 + delta
        sample_idx = torch.randint(0, n_verts, (min(n_surface_samples, n_verts),), device=device)
        data_term = _chamfer(verts[sample_idx], target)
        smooth_term = _laplacian_smoothness(delta, row, col, n_verts)
        reg_term = (delta ** 2).sum(-1).mean()
        loss = data_term + lambda_lap * smooth_term + lambda_reg * reg_term
        loss.backward()
        optimizer.step()

    final_vertices = (verts0 + delta).detach().cpu().numpy()
    deformed = trimesh.Trimesh(vertices=final_vertices, faces=faces, process=False)
    return deformed
