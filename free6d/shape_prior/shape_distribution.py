"""D2 shape distribution descriptor (Osada et al., "Shape Distributions", 2002).

A histogram of pairwise distances between random points on a shape's surface.
It is invariant to rotation and reflection, and becomes scale-invariant once
we normalize distances by the shape's own RMS radius. We use it purely for
fast coarse retrieval: shortlisting a handful of ShapeNet candidates per
category before doing an expensive rigid+scale ICP alignment on each.
"""
from __future__ import annotations

import numpy as np


def _rms_radius(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))) + 1e-8)


def d2_histogram(points: np.ndarray, n_pairs: int = 4096, n_bins: int = 32,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Returns a length-`n_bins` L1-normalized histogram of pairwise distances,
    scale-normalized by the point cloud's RMS radius from its centroid."""
    rng = rng or np.random.default_rng(0)
    n = points.shape[0]
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    d = np.linalg.norm(points[idx_a] - points[idx_b], axis=1)
    d = d / (_rms_radius(points) + 1e-8)
    hist, _ = np.histogram(d, bins=n_bins, range=(0.0, 4.0), density=False)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two D2 histograms."""
    return float(np.abs(a - b).sum())
