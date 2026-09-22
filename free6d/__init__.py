"""Free6D: replaces Any6D's captured RGB-D anchor with a category-level
shape prior retrieved and deformed from ShapeNet, validated for sim-to-real
robustness against ScanObjectNN.
"""

from .pipeline import Free6DAnchorBuilder, build_free6d_anchor_mesh

__all__ = ["Free6DAnchorBuilder", "build_free6d_anchor_mesh"]
