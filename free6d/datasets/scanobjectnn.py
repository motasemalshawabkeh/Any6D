"""ScanObjectNN loader.

ScanObjectNN ships as HDF5 files (official "main_split" release), each with
a 'data' dataset of shape (N, 2048, 3) and a 'label' dataset of shape (N,).
Access is gated behind a request form -- see scripts/download_scanobjectnn.md.

ScanObjectNN has NO 6D pose ground truth: it is a real-world point cloud
classification / robustness benchmark (clean "OBJ_ONLY", background-cluttered
"OBJ_BG", and perturbed "PB_T50_RS" variants). We use it here only to test
whether a ShapeNet-trained shape prior still fits real, noisy, partial scans
-- not to score pose accuracy. Pose accuracy is still measured on the posed
benchmarks Any6D already evaluates on (HO3D, YCBV, ...) via metrics.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import h5py
import numpy as np

# Official 15-class list, in the label-index order used by the ScanObjectNN
# release. Verify against the label map that ships with your copy of the
# dataset if you see class-name mismatches.
CLASS_NAMES: List[str] = [
    "bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
    "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet",
]


@dataclass
class ScanObjectNNSample:
    points: np.ndarray  # (N, 3)
    label: int

    @property
    def category(self) -> str:
        return CLASS_NAMES[self.label]


class ScanObjectNN:
    def __init__(self, h5_path: str):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(
                f"{h5_path} not found. See scripts/download_scanobjectnn.md for access."
            )
        with h5py.File(h5_path, "r") as f:
            self.points = np.asarray(f["data"], dtype=np.float32)
            self.labels = np.asarray(f["label"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> ScanObjectNNSample:
        return ScanObjectNNSample(points=self.points[idx], label=int(self.labels[idx]))

    def filter_category(self, category: str) -> List[ScanObjectNNSample]:
        label = CLASS_NAMES.index(category)
        idxs = np.where(self.labels == label)[0]
        return [self[i] for i in idxs]

    def categories_present(self) -> List[str]:
        return sorted({CLASS_NAMES[l] for l in np.unique(self.labels) if l < len(CLASS_NAMES)})
