"""Synthetic smoke tests for the parts of the Scan2CAD/ScanNet eval path
(eval_free6d_scan2cad.py) that don't need CUDA/Any6D or a real download:
mask rendering, GT-pose derivation, and ScanNet frame loading.

Run: python test_free6d_eval_smoke.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from free6d.datasets.scan2cad import load_scan2cad_annotations, compose_mat4
from free6d.datasets.scannet_frames import ScanNetScene, is_valid_pose
from free6d.render import render_mask


def test_render_mask() -> None:
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])

    T_front = np.eye(4)
    T_front[2, 3] = 1.0  # 1m in front of the camera, on the principal axis
    mask_front = render_mask(mesh, T_front, K, 480, 640)
    ys, xs = np.where(mask_front)
    assert mask_front.sum() > 0, "expected a non-empty mask for an object in front of the camera"
    assert abs(xs.mean() - 320.0) < 3 and abs(ys.mean() - 240.0) < 3, (
        f"mask centroid ({xs.mean():.1f},{ys.mean():.1f}) should be near the principal point "
        "(320,240) -- possible OpenCV/OpenGL camera-convention bug"
    )

    T_behind = np.eye(4)
    T_behind[2, 3] = -1.0
    mask_behind = render_mask(mesh, T_behind, K, 480, 640)
    assert mask_behind.sum() == 0, (
        "expected an EMPTY mask for an object behind the camera -- if this fires, the "
        "OpenCV->OpenGL flip in free6d/render.py is inverted"
    )

    T_far = np.eye(4)
    T_far[2, 3] = 3.0
    mask_far = render_mask(mesh, T_far, K, 480, 640)
    assert 0 < mask_far.sum() < mask_front.sum(), "mask area should shrink as the object moves away"

    print(f"[render] PASS (front={mask_front.sum()}px, behind={mask_behind.sum()}px, "
          f"far={mask_far.sum()}px)")


# A trimmed real Scan2CAD annotation (github.com/skanti/Scan2CAD,
# Routines/Script/example_annotation.json, scene0470_00) -- exercises the
# actual JSON schema and quaternion convention, not a made-up one.
_EXAMPLE_ANNOTATION = [{
    "id_scan": "scene0470_00",
    "trs": {"translation": [-1.743154195486568, -0.6799398784060032, 1.425440548118786],
            "rotation": [-0.7071067811865476, 0.7071067811865475, -0.0, -0.0],
            "scale": [1.0, 1.0, 1.0]},
    "aligned_models": [{
        "trs": {"translation": [-1.132369654029993, -0.382170050392887, 0.8210001235639066],
                "rotation": [-0.9481602850789679, 0.023194604622679315, 0.3166251773291209, 0.014233102145002504],
                "scale": [1.491760210035455, 1.39264290710526, 1.4662462668198055]},
        "center": [-0.01093900203704834, 0.20994599908590317, -0.020426996052265167],
        "sym": "__SYM_NONE", "id_cad": "2c03bcb2a133ce28bb6caad47eee6580", "catid_cad": "03001627",
    }],
}]


def test_scan2cad_parsing() -> None:
    tmp = tempfile.mktemp(suffix=".json")
    with open(tmp, "w") as f:
        json.dump(_EXAMPLE_ANNOTATION, f)
    try:
        scenes = load_scan2cad_annotations(tmp)
        assert "scene0470_00" in scenes
        instance = scenes["scene0470_00"].instances[0]
        assert instance.category == "chair", f"catid_cad 03001627 should map to 'chair', got {instance.category}"
        assert np.isfinite(instance.gt_pose_world).all()
        assert np.allclose(instance.gt_pose_world[3], [0, 0, 0, 1]), "bottom row of a valid 4x4 should be [0,0,0,1]"
        # regression pin against the verified value (cross-checked against the
        # official SE3.compose_mat4/EvaluateBenchmark.py on this exact example)
        expected_translation = np.array([0.59367497, 0.56049332, 0.00720965])
        assert np.allclose(instance.gt_pose_world[:3, 3], expected_translation, atol=1e-5), (
            f"gt_pose_world translation {instance.gt_pose_world[:3, 3]} != expected {expected_translation}"
        )
        print(f"[scan2cad] PASS (category={instance.category}, "
              f"translation={instance.gt_pose_world[:3, 3]})")
    finally:
        os.remove(tmp)


def test_scannet_frames() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="free6d_scannet_smoke_"))
    try:
        scene_dir = tmp_root / "scene0000_00"
        for sub in ["color", "depth", "pose", "intrinsic"]:
            (scene_dir / sub).mkdir(parents=True)

        K = np.array([[577.0, 0, 319.5, 0], [0, 577.0, 239.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.savetxt(scene_dir / "intrinsic" / "intrinsic_depth.txt", K)

        import cv2
        color = (np.random.default_rng(0).integers(0, 255, size=(240, 320, 3))).astype(np.uint8)
        depth = np.full((240, 320), 1500, dtype=np.uint16)  # 1.5m, matches ScanNet's mm convention
        for frame_id, valid in [(0, True), (1, False)]:
            cv2.imwrite(str(scene_dir / "color" / f"{frame_id}.jpg"), color)
            cv2.imwrite(str(scene_dir / "depth" / f"{frame_id}.png"), depth)
            pose = np.eye(4) if valid else np.full((4, 4), np.nan)
            np.savetxt(scene_dir / "pose" / f"{frame_id}.txt", pose)

        scene = ScanNetScene(str(scene_dir))
        assert scene.K.shape == (3, 3)
        assert np.allclose(scene.K, K[:3, :3])
        assert scene.valid_frame_ids() == [0], (
            f"expected only frame 0 to be valid (frame 1 has a NaN pose), got {scene.valid_frame_ids()}"
        )
        d = scene.depth(0)
        assert abs(d.mean() - 1.5) < 1e-3, f"depth should decode to ~1.5m, got {d.mean()}"
        assert is_valid_pose(scene.camera_to_world(0))
        assert not is_valid_pose(scene.camera_to_world(1))
        print("[scannet_frames] PASS")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> int:
    test_render_mask()
    test_scan2cad_parsing()
    test_scannet_frames()
    print("[smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
