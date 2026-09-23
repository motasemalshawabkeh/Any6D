"""Shape-prior robustness eval on ScanObjectNN.

ScanObjectNN has no 6D pose ground truth, so this script does NOT report
pose error. It reports how well a ShapeNet-only shape prior (built with
zero real-world data) reconstructs real, noisy, partially-occluded scans --
i.e. how much the sim-to-real gap costs the Free6D anchor before it ever
reaches the pose-refinement stage. Run it once per ScanObjectNN variant
(OBJ_ONLY / OBJ_BG / PB_T50_RS, ...) to see robustness to background
clutter and perturbation.

Pose-accuracy numbers for Free6D still need a *posed* dataset in the same
object-category domain as ScanObjectNN (furniture-like categories) --
Any6D's existing HO3D/YCBV benchmarks are tabletop manipulation objects and
don't overlap in category. Scan2CAD (real ScanNet scans with GT 9-DoF
ShapeNet CAD alignments) is the natural match and is a good next step; it
isn't wired up here yet.
"""
import argparse
import csv
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from free6d.datasets.scanobjectnn import ScanObjectNN
from free6d.datasets.shapenet import category_for
from free6d.pipeline import Free6DAnchorBuilder


def chamfer_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh, n_samples: int = 4096) -> float:
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    tree_mesh = cKDTree(surface_pts)
    tree_pts = cKDTree(points)
    d_pts_to_mesh, _ = tree_mesh.query(points)
    d_mesh_to_pts, _ = tree_pts.query(surface_pts)
    return float(d_pts_to_mesh.mean() + d_mesh_to_pts.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Free6D shape-prior robustness eval on ScanObjectNN")
    parser.add_argument("--shapenet_root", type=str, required=True)
    parser.add_argument("--scanobjectnn_h5", type=str, required=True, nargs="+",
                         help="One or more ScanObjectNN .h5 files, e.g. clean + background + perturbed variants")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                         help="Subset of categories to evaluate; defaults to every category with a ShapeNet mapping")
    parser.add_argument("--max_samples_per_category", type=int, default=20)
    parser.add_argument("--cache_dir", type=str, default="free6d_cache")
    parser.add_argument("--out_csv", type=str, default="results/free6d_scanobjectnn_robustness.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    builder = Free6DAnchorBuilder(shapenet_root=args.shapenet_root, cache_dir=args.cache_dir)

    rows = []
    for h5_path in args.scanobjectnn_h5:
        variant = os.path.splitext(os.path.basename(h5_path))[0]
        dataset = ScanObjectNN(h5_path)
        available = dataset.categories_present()
        categories = args.categories or [c for c in available if category_for(c) is not None]

        for category in categories:
            if category_for(category) is None:
                print(f"[skip] '{category}' has no ShapeNet mapping")
                continue
            samples = dataset.filter_category(category)[: args.max_samples_per_category]
            if not samples:
                continue
            for i, sample in enumerate(samples):
                try:
                    mesh, info = builder.build_anchor_mesh(category, sample.points)
                except RuntimeError as e:
                    print(f"[fail] {variant}/{category}[{i}]: {e}")
                    continue
                cd = chamfer_to_mesh(sample.points, mesh)
                rows.append({
                    "variant": variant, "category": category, "sample_idx": i,
                    "chamfer_distance": cd,
                    "retrieval_fitness": info["rigid_fit"].fitness,
                    "retrieval_rmse": info["rigid_fit"].rmse,
                    "shapenet_model_id": info["shapenet_model_id"],
                })
                print(f"{variant}/{category}[{i}] chamfer={cd:.4f} "
                      f"fitness={info['rigid_fit'].fitness:.3f}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
