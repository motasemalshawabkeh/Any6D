"""Pose-accuracy eval for Free6D on Scan2CAD: real ScanNet scenes with GT
9-DoF ShapeNet CAD alignments, in the category domain that actually overlaps
with ShapeNet/ScanObjectNN (cabinet, chair, display, table, sofa) -- unlike
Any6D's existing HO3D/YCBV benchmarks, which are tabletop manipulation
objects with no category overlap. See README's Free6D section for why.

For each GT instance, in a handful of frames where it's visible enough:
  1. Render the GT CAD's silhouette at its ground-truth pose to get an
     object mask (free6d/render.py) -- standing in for what a detector or
     SAM2 would give at inference time.
  2. Back-project the masked depth into a partial point cloud.
  3. Run Free6D (ShapeNet retrieval + non-rigid fit) to build an anchor mesh
     from *only* the category label + that partial point cloud -- Free6D
     never sees which exact CAD model or pose is the answer.
  4. Run Any6D.register_any6d() for the predicted pose, exactly as
     run_demo.py / run_ho3d_anchor.py do.
  5. Score with the same calculate_chamfer_distance_gt_mesh() Any6D's own
     scripts use, so numbers are comparable to Any6D's InstantMesh path.
"""
import argparse
import csv
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh

from estimater import Any6D
from foundationpose.Utils import calculate_chamfer_distance_gt_mesh
from free6d.datasets.scan2cad import load_scan2cad_annotations, shapenet_obj_path
from free6d.datasets.scannet_frames import ScanNetScene
from free6d.pipeline import Free6DAnchorBuilder, partial_pointcloud_from_rgbd
from free6d.render import render_mask

glctx = dr.RasterizeCudaContext()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Free6D pose-accuracy eval on Scan2CAD")
    parser.add_argument("--shapenet_root", type=str, required=True)
    parser.add_argument("--scan2cad_json", type=str, required=True, help="Path to full_annotations.json")
    parser.add_argument("--scannet_root", type=str, required=True,
                         help="Directory of exported ScanNet scenes (one subfolder per scene id)")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                         help="Defaults to every category Free6D and Scan2CAD both support")
    parser.add_argument("--max_instances_per_category", type=int, default=20)
    parser.add_argument("--frames_per_instance", type=int, default=1,
                         help="How many visible frames to evaluate per GT instance")
    parser.add_argument("--frame_stride", type=int, default=30, help="Only consider every Nth frame")
    parser.add_argument("--min_mask_pixels", type=int, default=800)
    parser.add_argument("--cache_dir", type=str, default="free6d_cache")
    parser.add_argument("--out_csv", type=str, default="results/free6d_scan2cad_pose.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    scenes = load_scan2cad_annotations(args.scan2cad_json)
    builder = Free6DAnchorBuilder(shapenet_root=args.shapenet_root, cache_dir=args.cache_dir)

    per_category_count = {}
    rows = []

    for id_scan, scene_ann in scenes.items():
        scene_dir = os.path.join(args.scannet_root, id_scan)
        if not os.path.isdir(scene_dir):
            continue
        scannet_scene = ScanNetScene(scene_dir)
        valid_frames = scannet_scene.valid_frame_ids()[::args.frame_stride]
        if not valid_frames:
            continue

        for instance in scene_ann.instances:
            category = instance.category
            if category is None:
                continue
            if args.categories and category not in args.categories:
                continue
            if per_category_count.get(category, 0) >= args.max_instances_per_category:
                continue

            gt_mesh_path = shapenet_obj_path(args.shapenet_root, instance.catid_cad, instance.id_cad)
            if not os.path.isfile(gt_mesh_path):
                continue
            gt_mesh = trimesh.load(gt_mesh_path, force="mesh", process=False)

            found_frames = 0
            for frame_id in valid_frames:
                if found_frames >= args.frames_per_instance:
                    break
                T_world_from_cam = scannet_scene.camera_to_world(frame_id)
                T_cam_from_gt = np.linalg.inv(T_world_from_cam) @ instance.gt_pose_world

                depth = scannet_scene.depth(frame_id)
                mask = render_mask(gt_mesh, T_cam_from_gt, scannet_scene.K, depth.shape[0], depth.shape[1])
                if mask.sum() < args.min_mask_pixels:
                    continue

                partial_pcd = partial_pointcloud_from_rgbd(depth, scannet_scene.K, mask)
                if partial_pcd.shape[0] < 200:
                    continue

                color = scannet_scene.color(frame_id)
                if color.shape[:2] != depth.shape[:2]:
                    color = cv2.resize(color, (depth.shape[1], depth.shape[0]))

                try:
                    pred_mesh, info = builder.build_anchor_mesh(category, partial_pcd)
                    est = Any6D(symmetry_tfs=None, mesh=pred_mesh, debug=0)
                    pred_pose = est.register_any6d(K=scannet_scene.K, rgb=color, depth=depth, ob_mask=mask,
                                                    iteration=5, name=f"{id_scan}_{frame_id}")
                    chamfer_cm = calculate_chamfer_distance_gt_mesh(T_cam_from_gt, gt_mesh, pred_pose, est.mesh)
                except Exception as e:
                    print(f"[fail] {id_scan} frame {frame_id} {category}: {e}")
                    continue

                found_frames += 1
                per_category_count[category] = per_category_count.get(category, 0) + 1
                rows.append({
                    "id_scan": id_scan, "frame_id": frame_id, "category": category,
                    "gt_id_cad": instance.id_cad, "shapenet_model_id": info["shapenet_model_id"],
                    "chamfer_distance_cm": chamfer_cm,
                })
                print(f"{id_scan}/{frame_id} {category}: chamfer={chamfer_cm:.3f}cm "
                      f"(gt={instance.id_cad}, retrieved={info['shapenet_model_id']})")

            if per_category_count.get(category, 0) >= args.max_instances_per_category:
                continue

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
