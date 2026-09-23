"""Scan2CAD loader: real ScanNet scenes with GT 9-DoF (translation, rotation,
anisotropic scale) alignments of exact ShapeNet CAD models.

Unlike ScanObjectNN, Scan2CAD *does* carry pose ground truth, and its
`catid_cad` field is literally a ShapeNet synset id, so its category set
overlaps with ours (free6d/datasets/shapenet.py) wherever both projects
happened to pick the same synsets: cabinet, chair, display, table, sofa.
That overlap is what makes Scan2CAD the right dataset to pair with
ShapeNet + ScanObjectNN for *pose*-accuracy numbers (see README).

Access is gated behind a license agreement -- see
scripts/download_scan2cad.md. The transform composition below is a direct,
dependency-free port of the official benchmark's SE3.compose_mat4 /
invert_mat4 (github.com/skanti/Scan2CAD, Routines/Script/SE3.py) and the
GT-pose derivation in EvaluateBenchmark.py (`inv(Mscan) @ Mcad`), verified
against that source rather than re-derived from memory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .shapenet import SCANOBJECTNN_TO_SHAPENET_SYNSET

# catid_cad (ShapeNet synset) -> category name, restricted to synsets Free6D
# also has a ShapeNet candidate library for (see shapenet.py). Scan2CAD's own
# top-8 class set additionally has bathtub/trashbin/bookshelf, which we drop
# here since we have no ShapeNet mapping for them.
SCAN2CAD_CATEGORY_NAMES: Dict[str, str] = {v: k for k, v in SCANOBJECTNN_TO_SHAPENET_SYNSET.items()
                                            if v in {"02933112", "03001627", "03211117", "04379243", "04256520"}}


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q = (w, x, y, z), matching Scan2CAD's JSON order and numpy-quaternion's
    constructor convention -- NOT scipy's (x, y, z, w)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def compose_mat4(t: np.ndarray, q: np.ndarray, s: np.ndarray,
                  center: Optional[np.ndarray] = None) -> np.ndarray:
    """Port of Scan2CAD's SE3.compose_mat4: M = Translate(t) @ Rotate(q) @ Scale(s) @ Translate(center)."""
    T = np.eye(4)
    T[:3, 3] = t
    R = np.eye(4)
    R[:3, :3] = _quat_to_rotmat(np.asarray(q, dtype=np.float64))
    S = np.eye(4)
    S[:3, :3] = np.diag(s)
    C = np.eye(4)
    if center is not None:
        C[:3, 3] = center
    return T @ R @ S @ C


@dataclass
class Scan2CADInstance:
    id_cad: str
    catid_cad: str
    sym: str
    gt_pose_world: np.ndarray  # (4,4), maps the RAW (uncentered) ShapeNet mesh's own
                                # vertex coordinates into the ScanNet scene's native
                                # mesh/world frame (the same frame ScanNet per-frame
                                # camera-to-world poses are expressed in).

    @property
    def category(self) -> Optional[str]:
        return SCAN2CAD_CATEGORY_NAMES.get(self.catid_cad)


@dataclass
class Scan2CADScene:
    id_scan: str
    instances: List[Scan2CADInstance]


def load_scan2cad_annotations(json_path: str) -> Dict[str, Scan2CADScene]:
    """Parses full_annotations.json into {id_scan: Scan2CADScene}."""
    with open(json_path, "r") as f:
        raw = json.load(f)

    scenes: Dict[str, Scan2CADScene] = {}
    for entry in raw:
        id_scan = entry["id_scan"]
        m_scan = compose_mat4(entry["trs"]["translation"], entry["trs"]["rotation"], entry["trs"]["scale"])
        m_scan_inv = np.linalg.inv(m_scan)

        instances = []
        for model in entry["aligned_models"]:
            m_cad = compose_mat4(model["trs"]["translation"], model["trs"]["rotation"], model["trs"]["scale"],
                                  center=-np.asarray(model["center"], dtype=np.float64))
            gt_pose_world = m_scan_inv @ m_cad
            instances.append(Scan2CADInstance(
                id_cad=model["id_cad"], catid_cad=model["catid_cad"], sym=model["sym"],
                gt_pose_world=gt_pose_world,
            ))
        scenes[id_scan] = Scan2CADScene(id_scan=id_scan, instances=instances)
    return scenes


def shapenet_obj_path(shapenet_root: str, catid_cad: str, id_cad: str) -> str:
    return os.path.join(shapenet_root, catid_cad, id_cad, "models", "model_normalized.obj")
