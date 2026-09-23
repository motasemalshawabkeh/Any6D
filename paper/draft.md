# Free6D: Category-Level Shape Priors from Free Public Data for Anchor-Free 6D Pose Estimation

*Working draft. Structure follows a generic CV/robotics journal format
(IJCV / TPAMI / Pattern Recognition / RA-L style); adjust section numbering
and length once a target venue is picked. Every claim below is grounded in
`free6d/` code that exists and has been smoke-tested (see
`test_free6d_smoke.py`, `test_free6d_eval_smoke.py`) — nothing here is
aspirational except the results tables, which are placeholders pending a
real ShapeNet/ScanObjectNN/Scan2CAD download and GPU run.*

---

## Abstract (draft)

Any6D [Lee et al., CVPR 2025] estimates the 6D pose of a novel object from
a single RGB-D scene, but still requires one physically captured RGB-D
*anchor* image of that exact object instance to build its reference mesh.
This anchor-capture step is a real deployment cost: it requires a depth
sensor, a clear unoccluded view of the object, and a human (or robot) to
take it, once per novel object. We introduce **Free6D**, which removes this
requirement entirely by replacing the captured anchor with a category-level
shape prior built purely from free, publicly available data: ShapeNet
CAD models. Given only an object's category label and a masked partial
observation (from any RGB-D frame, not a dedicated capture), Free6D
retrieves the closest-matching ShapeNet model via a rotation-invariant
shape descriptor, rigidly aligns it with scale via RANSAC+ICP, and
refines it with a Laplacian-regularized non-rigid fit — producing an
anchor mesh that plugs directly into Any6D's existing render-and-compare
pose refinement, unmodified. Because ShapeNet is synthetic, we
separately validate (a) shape-reconstruction robustness against real,
noisy, partially-occluded scans using ScanObjectNN, which has no pose
ground truth, and (b) pose accuracy against Scan2CAD, whose
human-verified 9-DoF ShapeNet-to-ScanNet alignments give genuine ground
truth in the same category domain (cabinet, chair, display, table, sofa).
We report [TBD] on both.

## 1. Introduction

- Motivate model-free 6D pose estimation and Any6D's contribution.
- State the specific remaining cost Any6D leaves on the table: the anchor
  capture step is a per-object, per-deployment acquisition cost, which
  limits practical scalability (a warehouse robot encountering a new SKU
  still needs someone to scan it first).
- Frame Free6D's contribution precisely: **not** a claim that Free6D beats
  Any6D's pose accuracy when a good anchor capture is available — it won't,
  categorically, since a real capture of the exact instance is strictly
  more informative than a retrieved category-level proxy. The contribution
  is removing the acquisition requirement, trading some accuracy for zero
  marginal capture cost, and characterizing exactly how much accuracy is
  lost and under what conditions (category, occlusion level, sim-to-real
  gap) it's lost.
- Contributions list:
  1. A retrieval + non-rigid fit pipeline that builds a usable anchor mesh
     from category label + partial observation alone, using only free
     public CAD data (no training, no per-category deep network required
     for the base version — see Sec 3.4 for the optional learned variant).
  2. A shape-distribution-based (Osada et al. 2002) descriptor for fast
     coarse retrieval, avoiding a full ICP sweep over an entire ShapeNet
     category.
  3. An evaluation protocol that correctly separates what ScanObjectNN can
     and cannot measure (Sec 4.1) — a point prior work in this space has
     not always been careful about.
  4. Scan2CAD as the pose-ground-truth benchmark for this category domain,
     with the GT-pose derivation verified against the official benchmark
     toolkit (Sec 4.2, Appendix A).

## 2. Related Work

- **Model-free / category-level 6D pose estimation.** Any6D (anchor image);
  NOCS [Wang et al. 2019] (normalized object coordinate space, per-category
  canonical frame); SPD [Tian et al. 2020] (shape-prior deformation,
  closest prior work methodologically — Free6D's non-rigid fit is in this
  family but test-time, not trained); SGPA [Chen & Dou, 2021]
  (structure-guided prior adaptation). Position Free6D relative to SPD/SGPA
  precisely: those learn a deformation *network*; Free6D does test-time
  optimization, trading some accuracy for zero training cost and instant
  applicability to any ShapeNet category with no per-category training
  data collection. This is a real trade-off to be honest about in the
  paper, not just a convenience.
- **Shape retrieval from partial observations.** Classical shape
  descriptors (D2/shape distributions, spin images, FPFH); learned
  alternatives (PointNet-based retrieval). Justify the D2 descriptor
  choice as deliberately simple/non-learned, consistent with the "free"
  framing — no training data or compute needed to stand the whole
  pipeline up.
- **CAD-to-scan alignment.** Scan2CAD itself is the closest *task* framing
  to Free6D's core operation (retrieve + 9-DoF align a CAD model into a
  real scene) — but Scan2CAD is offline/scene-level (aligns into a full
  reconstructed mesh with a learned heatmap network trained on their own
  annotated data), where Free6D is online/per-frame and training-free.
  This distinction is worth a full paragraph since a reviewer will ask.
- **Datasets.** ShapeNet [Chang et al. 2015]; ScanObjectNN [Uy et al. 2019]
  (real-world robustness benchmark, OBJ_ONLY/OBJ_BG/PB_T50_RS variants);
  Scan2CAD [Avetisyan et al. 2019] (ScanNet + GT CAD alignments).

## 3. Method

### 3.1 Problem setup
Given an RGB-D frame, an object mask (from any detector/segmenter — SAM2
in the demo), and a category label, produce a 6D pose (and implicitly, a
plausible mesh) for the object, without ever having captured a dedicated
reference view of that specific instance.

### 3.2 Shape-prior retrieval (`free6d/shape_prior/category_prior.py`)
- Per category, index up to *N* ShapeNet models for the mapped synset
  (`free6d/datasets/shapenet.py::SCANOBJECTNN_TO_SHAPENET_SYNSET`).
- Precompute a D2 shape-distribution histogram per candidate
  (`shape_distribution.py`): a scale-normalized histogram of pairwise
  distances between random surface point pairs, rotation/reflection
  invariant by construction, cheap to compare (L1 distance between
  histograms).
- At query time: shortlist top-*k* candidates by histogram distance, then
  rigidly align each via FPFH-feature RANSAC + point-to-point ICP swept
  over a coarse scale range (`rigid_align_with_scale`), keeping the
  best-fitness candidate.

### 3.3 Non-rigid refinement (`free6d/shape_prior/fit.py`)
- The retrieved candidate's own vertices (not a resampled/re-triangulated
  point cloud) are displaced by a learned-at-test-time per-vertex offset
  field, optimized via Adam against a Chamfer data term plus a uniform
  Laplacian smoothness regularizer (computed via scatter-mean over mesh
  edges, no dense Laplacian matrix).
- This keeps the original face topology, so the output plugs directly into
  Any6D's `reset_object`/`register_any6d` with no adapter code.

### 3.4 Integration with Any6D (`free6d/pipeline.py`, `run_free6d_demo.py`)
- The deformed mesh is passed as `Any6D(mesh=...)`; Any6D's own joint
  object alignment + render-and-compare refinement handles the rest
  unmodified. Free6D's entire job ends at anchor-mesh construction.

### 3.5 (Future extension, not yet implemented) Learned deformation
A per-category deformation network (SPD/SGPA-style) trained on ShapeNet
would likely improve fit quality over test-time optimization, at the cost
of a training phase per category. Flagged as future work / an ablation
worth running once compute is available — do not claim this now.

## 4. Experimental Design

### 4.1 Shape-prior robustness on ScanObjectNN (`eval_free6d_scanobjectnn.py`)
**What this does and does not measure.** ScanObjectNN has no 6D pose
ground truth — it is a real-world point-cloud classification/robustness
benchmark. We measure one-directional and symmetric Chamfer distance
between the real scan and the Free6D-fitted mesh's sampled surface, across
the OBJ_ONLY (clean), OBJ_BG (background clutter), and PB_T50_RS
(perturbed) variants, to isolate: (a) the base sim-to-real shape gap, (b)
sensitivity to background clutter, (c) sensitivity to occlusion/noise
perturbation. This is *not* a pose accuracy number and the paper must not
present it as one — a category-level pose paper that conflates shape
fidelity with pose accuracy is an easy target for a reviewer.

### 4.2 Pose accuracy on Scan2CAD (`eval_free6d_scan2cad.py`)
- Categories: cabinet, chair, display, table, sofa — the exact
  ShapeNet-synset overlap between Scan2CAD's `catid_cad` values and
  Free6D's supported categories.
- Protocol: for each GT instance, render its silhouette at the GT pose
  (`free6d/render.py`) to get an object mask standing in for a real
  detector; Free6D never sees the GT CAD id or pose. Score with
  `calculate_chamfer_distance_gt_mesh` (same function Any6D's own
  HO3D/YCBV scripts use), so numbers are on the same scale as Any6D's
  published metric.
- GT-pose derivation is a verified port of Scan2CAD's own benchmark
  toolkit (`SE3.compose_mat4`, `inv(Mscan) @ Mcad`) — see Appendix A for
  the numerical cross-check against the official code.
- **Baselines to run** (not yet run): Any6D with its InstantMesh anchor
  path on the same frames (requires a real, physically captured anchor —
  report as an upper-bound reference, not a fair like-for-like
  comparison, and say so explicitly in the results section); NOCS; SPD;
  SGPA if pretrained category-level checkpoints for these 5 categories are
  available; otherwise flag as a limitation.

### 4.3 Ablations (planned, not yet run)
- Retrieval-only vs. retrieval+non-rigid fit (isolates the non-rigid
  step's contribution — the smoke test already shows a ~4x Chamfer
  improvement on synthetic primitives, Sec 5 will report the real-data
  number).
- Top-*k* shortlist size sensitivity.
- Per-category breakdown (chairs are ShapeNet's best-populated,
  highest-diversity category; expect it to be the easiest; sofa/cabinet
  likely hardest due to higher intra-category shape variance).
- Occlusion-level sweep on Scan2CAD (mask visible-pixel-count bucketing).

## 5. Results
*[TBD — pending ShapeNet/ScanObjectNN/Scan2CAD download and a GPU run.
Table skeletons below to be filled in once data is available.]*

**Table 1.** ScanObjectNN shape-reconstruction Chamfer distance (cm), by
category × variant (OBJ_ONLY / OBJ_BG / PB_T50_RS).

**Table 2.** Scan2CAD pose Chamfer distance (cm), by category, Free6D vs.
baselines.

**Table 3.** Ablation: retrieval-only vs. +non-rigid fit.

## 6. Limitations
- Free6D's category coverage is bounded by the ShapeNet/ScanObjectNN
  synset overlap (8 categories) and further bounded to 5 for pose
  evaluation (Scan2CAD overlap) — this is a real scope limitation to state
  plainly, not hide.
- Test-time optimization (Sec 3.3) has no learned prior over plausible
  deformations beyond Laplacian smoothness — it can produce implausible
  local deformations under heavy occlusion. A learned deformation network
  (Sec 3.5) is the natural fix, left to future work.
- Category label is assumed given (from an external classifier/detector),
  not predicted by Free6D itself.
- Never claims to beat Any6D's own captured-anchor accuracy — the
  contribution is removing the capture requirement, not improving on it.

## 7. Conclusion
*[TBD after Sec 5 results.]*

---

## Appendix A: Scan2CAD GT-pose verification

`free6d/datasets/scan2cad.py::compose_mat4` and the `gt_pose_world =
inv(Mscan) @ Mcad` derivation were cross-checked against the official
Scan2CAD benchmark toolkit (`github.com/skanti/Scan2CAD`,
`Routines/Script/SE3.py` + `EvaluateBenchmark.py`) on that repo's own
`example_annotation.json`. Both the quaternion→rotation-matrix formula
(verified against `numpy-quaternion`'s `as_rotation_matrix` source) and
the full transform composition were confirmed to agree with the official
code to within float precision (max abs difference ~3e-16). This
verification is captured as a permanent regression test in
`test_free6d_eval_smoke.py::test_scan2cad_parsing`.

## Appendix B: Code/paper correspondence table

| Paper section | Code |
|---|---|
| 3.2 retrieval | `free6d/shape_prior/category_prior.py`, `shape_distribution.py` |
| 3.3 non-rigid fit | `free6d/shape_prior/fit.py` |
| 3.4 integration | `free6d/pipeline.py`, `run_free6d_demo.py`, `estimater.py` (Any6D, unmodified) |
| 4.1 ScanObjectNN eval | `eval_free6d_scanobjectnn.py`, `free6d/datasets/scanobjectnn.py` |
| 4.2 Scan2CAD eval | `eval_free6d_scan2cad.py`, `free6d/datasets/scan2cad.py`, `free6d/datasets/scannet_frames.py`, `free6d/render.py` |
