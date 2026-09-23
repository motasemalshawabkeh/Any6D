"""Free6D demo: build a pose-estimation anchor mesh from a ShapeNet shape
prior instead of a captured RGB-D anchor scan (Any6D's InstantMesh path),
then hand it to the same Any6D.register_any6d() used by run_demo.py.

Note: the bundled demo_data/ assets (a YCB mustard bottle) are only useful
for exercising the code path end-to-end -- "mustard bottle" isn't one of
the categories with a ShapeNet mapping (free6d/datasets/shapenet.py). Point
--rgb/--depth/--mask/--K at your own capture of an object in a supported
category (bag, cabinet, chair, display, table, bed, pillow, sofa) for a
scientifically meaningful run.
"""
import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
import yaml
from pytorch_lightning import seed_everything

from estimater import Any6D
from free6d.pipeline import Free6DAnchorBuilder, partial_pointcloud_from_rgbd

glctx = dr.RasterizeCudaContext()


if __name__ == "__main__":
    seed_everything(0)

    parser = argparse.ArgumentParser(description="Free6D demo: ShapeNet-prior anchor + Any6D pose")
    parser.add_argument("--shapenet_root", type=str, required=True, help="Path to ShapeNetCore.v2 root")
    parser.add_argument("--category", type=str, required=True,
                         help="One of the categories mapped in free6d/datasets/shapenet.py")
    parser.add_argument("--demo_path", type=str, default="demo_data")
    parser.add_argument("--rgb", type=str, default=None, help="Overrides demo_path/color.png")
    parser.add_argument("--depth", type=str, default=None, help="Overrides demo_path/depth.png")
    parser.add_argument("--mask", type=str, default=None,
                         help="Path to a binary mask image; defaults to demo_data/labels.npz obj_num")
    parser.add_argument("--obj_num", type=int, default=5, help="Segmentation id inside labels.npz")
    parser.add_argument("--intrinsics", type=str, default=None, help="Overrides demo_path/*.yml")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--cache_dir", type=str, default="free6d_cache")
    parser.add_argument("--top_k", type=int, default=5, help="ShapeNet candidates shortlisted for rigid fit")
    parser.add_argument("--nonrigid_iters", type=int, default=150)
    args = parser.parse_args()

    demo_path = args.demo_path
    save_path = os.path.join("results", f"free6d_{args.category}")
    os.makedirs(save_path, exist_ok=True)

    rgb_path = args.rgb or os.path.join(demo_path, "color.png")
    depth_path = args.depth or os.path.join(demo_path, "depth.png")
    color = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / args.depth_scale

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 0
    else:
        label = np.load(os.path.join(demo_path, "labels.npz"))
        mask = np.where(label["seg"] == args.obj_num, True, False)

    intrinsic_path = args.intrinsics
    if intrinsic_path is None:
        candidates = [f for f in os.listdir(demo_path) if f.endswith(".yml")]
        if not candidates:
            raise FileNotFoundError(f"No camera intrinsics .yml found under {demo_path}; pass --intrinsics")
        intrinsic_path = os.path.join(demo_path, candidates[0])
    with open(intrinsic_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    K = np.array([[data["depth"]["fx"], 0.0, data["depth"]["ppx"]],
                  [0.0, data["depth"]["fy"], data["depth"]["ppy"]],
                  [0.0, 0.0, 1.0]])

    partial_pcd = partial_pointcloud_from_rgbd(depth, K, mask)
    print(f"[Free6D] observed partial point cloud: {partial_pcd.shape[0]} points")

    builder = Free6DAnchorBuilder(shapenet_root=args.shapenet_root, cache_dir=args.cache_dir)
    mesh, info = builder.build_anchor_mesh(
        args.category, partial_pcd, top_k=args.top_k, nonrigid_iters=args.nonrigid_iters)
    print(f"[Free6D] retrieved ShapeNet model {info['shapenet_model_id']} "
          f"(fitness={info['rigid_fit'].fitness:.3f}, rmse={info['rigid_fit'].rmse:.4f}, "
          f"scale={info['rigid_fit'].scale:.3f})")
    mesh.export(os.path.join(save_path, "free6d_anchor_mesh.obj"))

    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2)
    pred_pose = est.register_any6d(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name="free6d_demo")

    np.savetxt(os.path.join(save_path, "pred_pose.txt"), pred_pose)
    est.mesh.export(os.path.join(save_path, "final_mesh.obj"))
    print(f"[Free6D] saved pose + mesh under {save_path}")
