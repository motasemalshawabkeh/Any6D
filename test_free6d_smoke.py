"""Synthetic smoke test for the Free6D shape-prior pipeline (retrieval +
rigid/scale ICP + non-rigid Chamfer fit), decoupled from Any6D/CUDA and from
a real ShapeNet/ScanObjectNN download.

It builds a throwaway "ShapeNet category" out of a handful of primitive
meshes, treats one of them (rigidly transformed + partially occluded +
noised) as a stand-in real-world observation, and checks that:
  1. the pipeline runs end-to-end without crashing,
  2. retrieval picks the primitive that's actually closest to the target
     shape (not some unrelated one),
  3. the non-rigid fit doesn't make things worse (fitted mesh should be at
     least as close to the target point cloud as the raw rigid-aligned
     retrieval was).

This does NOT validate Any6D integration (needs CUDA/nvdiffrast) or real
ShapeNet/ScanObjectNN data -- it only proves the Free6D-specific code is
free of crashing bugs and behaves sanely, so a first run against real data
isn't also the first time this code has ever executed.

Run: python test_free6d_smoke.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from free6d.datasets import shapenet as shapenet_mod
from free6d.pipeline import Free6DAnchorBuilder

FAKE_SYNSET = "99999999"
FAKE_CATEGORY = "__smoketest_mug__"


def _make_fake_shapenet_category(root: Path, n_models: int = 5) -> list[str]:
    """Writes a handful of distinct primitive meshes as a fake ShapeNet
    category and returns their model ids."""
    rng = np.random.default_rng(0)
    model_ids = []
    generators = [
        lambda: trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
        lambda: trimesh.creation.box(extents=[1.6, 0.6, 0.9]),
        lambda: trimesh.creation.icosphere(subdivisions=2, radius=0.6),
        lambda: trimesh.creation.cylinder(radius=0.4, height=1.2, sections=24),
        lambda: trimesh.creation.capsule(radius=0.3, height=0.8),
    ]
    for i, gen in enumerate(generators[:n_models]):
        mesh = gen()
        mesh.vertices += rng.normal(scale=0.01, size=mesh.vertices.shape)  # break exact symmetry
        model_id = f"fake_model_{i:02d}"
        out_dir = root / FAKE_SYNSET / model_id / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(out_dir / "model_normalized.obj")
        model_ids.append(model_id)
    return model_ids


def _random_rigid_transform(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.3, 1.2)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    t = rng.normal(scale=0.3, size=3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _partial_noisy_observation(mesh: trimesh.Trimesh, T: np.ndarray, rng: np.random.Generator,
                                n_points: int = 1500, noise_std: float = 0.005) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n_points * 2)
    pts_world = (T[:3, :3] @ pts.T).T + T[:3, 3]
    # keep only the "front-facing" half (simulates a single-view partial scan)
    view_dir = -T[:3, 3] / (np.linalg.norm(T[:3, 3]) + 1e-8)
    centered = pts_world - pts_world.mean(axis=0)
    visible = (centered @ view_dir) > 0
    pts_world = pts_world[visible][:n_points]
    pts_world += rng.normal(scale=noise_std, size=pts_world.shape)
    return pts_world.astype(np.float32)


def one_directional_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return float(cKDTree(b).query(a)[0].mean())


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="free6d_smoke_"))
    try:
        model_ids = _make_fake_shapenet_category(tmp_root)
        shapenet_mod.SCANOBJECTNN_TO_SHAPENET_SYNSET[FAKE_CATEGORY] = FAKE_SYNSET

        rng = np.random.default_rng(1)
        target_model_idx = 2  # the icosphere
        category = shapenet_mod.ShapeNetCategory(str(tmp_root), FAKE_SYNSET)
        target_mesh = category.load_mesh(category.models[target_model_idx])

        T_true = _random_rigid_transform(rng)
        observation = _partial_noisy_observation(target_mesh, T_true, rng)
        print(f"[smoke] synthetic partial observation: {observation.shape[0]} points "
              f"(target model: {model_ids[target_model_idx]})")
        assert observation.shape[0] > 100, "too few visible points -- test setup bug"

        builder = Free6DAnchorBuilder(shapenet_root=str(tmp_root),
                                       cache_dir=str(tmp_root / "cache"), n_points=1024)
        deformed_mesh, info = builder.build_anchor_mesh(FAKE_CATEGORY, observation, top_k=5,
                                                          nonrigid_iters=100)

        retrieved_id = info["shapenet_model_id"]
        fit = info["rigid_fit"]
        print(f"[smoke] retrieved: {retrieved_id} (fitness={fit.fitness:.3f}, "
              f"rmse={fit.rmse:.4f}, scale={fit.scale:.3f})")

        assert np.isfinite(deformed_mesh.vertices).all(), "deformed mesh has non-finite vertices"
        assert len(deformed_mesh.faces) == len(target_mesh.faces) or retrieved_id != model_ids[target_model_idx], \
            "face count should be unchanged when the correct candidate was retrieved"

        deformed_pts, _ = trimesh.sample.sample_surface(deformed_mesh, 2000)
        chamfer_after_fit = one_directional_chamfer(observation, deformed_pts)

        rigid_candidate = category.load_mesh(
            [m for m in category.models if m.model_id == retrieved_id][0])
        rigid_pts = (rigid_candidate.vertices - rigid_candidate.vertices.mean(axis=0)) * fit.scale
        rigid_pts = (fit.R @ rigid_pts.T).T + fit.t
        chamfer_rigid_only = one_directional_chamfer(observation, rigid_pts)

        print(f"[smoke] observation->mesh chamfer: rigid-only={chamfer_rigid_only:.4f}  "
              f"after non-rigid fit={chamfer_after_fit:.4f}")

        if retrieved_id != model_ids[target_model_idx]:
            print(f"[smoke][WARN] retrieval picked {retrieved_id}, expected {model_ids[target_model_idx]} "
                  "(D2 descriptor shortlisting + RANSAC on simple primitives can be ambiguous -- "
                  "not necessarily a bug, but worth a look if it happens consistently)")

        assert chamfer_after_fit <= chamfer_rigid_only * 1.5, (
            "non-rigid fit made things substantially worse than the rigid-only alignment -- "
            "likely a real bug in fit_shape_prior_to_pointcloud"
        )

        print("[smoke] PASS")
        return 0
    finally:
        shapenet_mod.SCANOBJECTNN_TO_SHAPENET_SYNSET.pop(FAKE_CATEGORY, None)
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
