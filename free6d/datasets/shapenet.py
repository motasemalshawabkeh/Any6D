"""ShapeNetCore loader.

Expects the standard ShapeNetCore.v2 layout downloaded after accepting the
Stanford ShapeNet license (see scripts/download_shapenet.md):

    <root>/<synset_id>/<model_id>/models/model_normalized.obj

We only need a handful of categories that overlap ScanObjectNN's 15 classes,
so the default mapping below is intentionally partial -- extend it if your
category list differs.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import trimesh

# ScanObjectNN category name -> ShapeNetCore.v2 synset id, for the classes
# that exist in both datasets. ("bin", "box", "desk", "door", "shelf",
# "sink", "toilet" have no clean ShapeNetCore counterpart and are left out;
# a category without a synset here simply can't get a Free6D shape prior.)
SCANOBJECTNN_TO_SHAPENET_SYNSET: Dict[str, str] = {
    "bag": "02773838",
    "cabinet": "02933112",
    "chair": "03001627",
    "display": "03211117",
    "table": "04379243",
    "bed": "02818832",
    "pillow": "03938244",
    "sofa": "04256520",
}


@dataclass
class ShapeNetModel:
    model_id: str
    synset_id: str
    obj_path: str


class ShapeNetCategory:
    """Indexes every ShapeNetCore model for a single synset id."""

    def __init__(self, root: str, synset_id: str):
        self.root = root
        self.synset_id = synset_id
        self.models: List[ShapeNetModel] = self._index()
        if len(self.models) == 0:
            raise FileNotFoundError(
                f"No ShapeNet models found for synset {synset_id} under {root}. "
                "Check the download layout in scripts/download_shapenet.md."
            )

    def _index(self) -> List[ShapeNetModel]:
        pattern = os.path.join(self.root, self.synset_id, "*", "models", "model_normalized.obj")
        models = []
        for obj_path in sorted(glob.glob(pattern)):
            model_id = obj_path.split(os.sep)[-3]
            models.append(ShapeNetModel(model_id=model_id, synset_id=self.synset_id, obj_path=obj_path))
        return models

    def load_mesh(self, model: ShapeNetModel) -> trimesh.Trimesh:
        mesh = trimesh.load(model.obj_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        return mesh

    def sample_pointcloud(self, model: ShapeNetModel, n_points: int = 2048) -> np.ndarray:
        mesh = self.load_mesh(model)
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        return np.asarray(pts, dtype=np.float32)


def category_for(category_name: str) -> Optional[str]:
    """Returns the ShapeNet synset id for a ScanObjectNN category name, if any."""
    return SCANOBJECTNN_TO_SHAPENET_SYNSET.get(category_name)
