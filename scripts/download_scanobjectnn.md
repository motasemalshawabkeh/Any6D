# Downloading ScanObjectNN

ScanObjectNN is free but access-gated behind a request form (real-world
scanned point clouds of indoor objects), so this repo can't fetch it for you.

1. Fill out the access request at https://hkust-vgd.github.io/scanobjectnn/
2. You'll receive a link to `h5_files.zip`. Unzip it; the "main_split"
   folder has the files this repo's loader expects, e.g.:
   ```
   h5_files/main_split/training_objectdataset.h5
   h5_files/main_split/test_objectdataset.h5
   h5_files/main_split_nobg/test_objectdataset.h5              # OBJ_ONLY (no background)
   h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5  # PB_T50_RS (perturbed)
   ```
3. Pass one or more of these paths to `--scanobjectnn_h5` in
   `eval_free6d_scanobjectnn.py`. Passing several at once (clean +
   background + perturbed) is how you measure robustness to real-world
   noise rather than a single point estimate.

Double check `free6d/datasets/scanobjectnn.py::CLASS_NAMES` against the
label map shipped with your copy -- the 15-class order has been stable
across releases but verify before trusting the category names.
