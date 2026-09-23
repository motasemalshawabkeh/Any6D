# Downloading ShapeNetCore.v2

ShapeNet is free but license-gated (Stanford ShapeNet license) -- there is no
anonymous direct-download URL, so this repo can't fetch it for you.

1. Create an account and accept the license at https://shapenet.org
2. Download `ShapeNetCore.v2.zip`.
3. Unzip it so you end up with:
   ```
   <root>/<synset_id>/<model_id>/models/model_normalized.obj
   ```
4. Pass `<root>` as `--shapenet_root` to `run_free6d_demo.py` / `eval_free6d_scanobjectnn.py`.

Free6D only uses the categories mapped in
`free6d/datasets/shapenet.py::SCANOBJECTNN_TO_SHAPENET_SYNSET` (bag, cabinet,
chair, display, table, bed, pillow, sofa) -- you can delete the other
synset folders after extraction to save disk space, e.g.:

```bash
cd ShapeNetCore.v2
ls | grep -vE '^(02773838|02933112|03001627|03211117|04379243|02818832|03938244|04256520)$' | xargs rm -rf
```
