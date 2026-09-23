# Downloading Scan2CAD + ScanNet

Scan2CAD closes the pose-accuracy gap left by ScanObjectNN (which has no
pose ground truth): it's real ScanNet scenes with human-verified 9-DoF
alignments of exact ShapeNet CAD models, in the same category domain as
ShapeNet/ScanObjectNN (cabinet, chair, display, table, sofa). Both are free
but access-gated.

## Scan2CAD annotations
1. Request access at https://github.com/skanti/Scan2CAD (their README links
   the annotation download; you'll need the ShapeNet license too, since
   Scan2CAD references ShapeNet model ids directly).
2. You get `full_annotations.json` -- pass its path as `--scan2cad_json`.

## ScanNet scenes
1. Request access via the form at http://www.scan-net.org/ (their own
   license agreement).
2. Download the specific `.sens` files for the scenes referenced in
   `full_annotations.json` (`id_scan`, e.g. `scene0000_00`) -- you don't
   need the whole 1500+ scene dataset for a subset evaluation.
3. Export each scene with ScanNet's own reader
   (github.com/ScanNet/ScanNet, `SensReader/python/reader.py`):
   ```bash
   python reader.py --filename scene0000_00.sens \
     --output_path /path/to/scannet_frames/scene0000_00 \
     --export_depth_images --export_color_images --export_poses --export_intrinsics
   ```
4. Pass `/path/to/scannet_frames` (the directory containing one subfolder
   per exported scene) as `--scannet_root` to `eval_free6d_scan2cad.py`.

You only need ShapeNetCore.v2 categories already covered in
`scripts/download_shapenet.md` -- Scan2CAD's `catid_cad` values are the
same ShapeNet synset ids.
