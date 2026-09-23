"""Loader for a ScanNet scene exported with the official SensReader
(github.com/ScanNet/ScanNet, SensReader/python/reader.py --export_depth_images
--export_color_images --export_poses --export_intrinsics), which produces:

    <scene>/color/<frame>.jpg
    <scene>/depth/<frame>.png            # 16-bit, millimeters
    <scene>/pose/<frame>.txt             # 4x4 camera-to-world (ScanNet's own
                                          # world frame -- the same frame
                                          # Scan2CAD GT poses are expressed in)
    <scene>/intrinsic/intrinsic_depth.txt  # 4x4, K in the top-left 3x3

A handful of ScanNet frames have an all-NaN/inf pose from tracking loss;
`is_valid_pose` filters those out.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np


def is_valid_pose(pose: np.ndarray) -> bool:
    return np.all(np.isfinite(pose))


@dataclass
class ScanNetScene:
    scene_dir: str

    def __post_init__(self):
        self.K = np.loadtxt(os.path.join(self.scene_dir, "intrinsic", "intrinsic_depth.txt"))[:3, :3]
        frame_paths = sorted(
            glob.glob(os.path.join(self.scene_dir, "pose", "*.txt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        self.frame_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in frame_paths]

    def camera_to_world(self, frame_id: int) -> np.ndarray:
        return np.loadtxt(os.path.join(self.scene_dir, "pose", f"{frame_id}.txt"))

    def depth(self, frame_id: int, depth_scale: float = 1000.0) -> np.ndarray:
        depth = cv2.imread(os.path.join(self.scene_dir, "depth", f"{frame_id}.png"), cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / depth_scale

    def color(self, frame_id: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(os.path.join(self.scene_dir, "color", f"{frame_id}.jpg")), cv2.COLOR_BGR2RGB)

    def valid_frame_ids(self) -> list[int]:
        return [f for f in self.frame_ids if is_valid_pose(self.camera_to_world(f))]
