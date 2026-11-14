#!/usr/bin/env python3
"""Builds the folder the Deformable-3DGS / SC-GS loaders read for a HyperNeRF scene, with a held-out split.

The repos pick the split from the PARENT folder name (`vrig*` -> 4x images, dataset.json `train_ids` / `val_ids`),
and the official `slice-banana` dataset.json lists all 330 frames as train and none as val. This script makes
<out>/vrig_<scene>/<scene>/ with symlinks to the original files and a dataset.json whose val_ids are every 10th frame
(offset 3). Nothing is copied or downloaded.

    python tests/make_hypernerf_split.py <path>/slice-banana /home/cont/gs4d_train/data
"""
import json, os, sys

src, out = os.path.abspath(sys.argv[1]), sys.argv[2]
name = os.path.basename(src)
dst = os.path.join(out, f"vrig_{name}", name)
os.makedirs(os.path.join(dst, "rgb"), exist_ok=True)
for f in ("camera", "metadata.json", "scene.json", "points.npy"):
    if not os.path.lexists(os.path.join(dst, f)):
        os.symlink(os.path.join(src, f), os.path.join(dst, f))
if not os.path.lexists(os.path.join(dst, "rgb", "4x")):
    os.symlink(os.path.join(src, "rgb", "4x"), os.path.join(dst, "rgb", "4x"))
d = json.load(open(os.path.join(src, "dataset.json")))
ids = d["ids"]
val = ids[3::10]
json.dump(dict(count=len(ids), num_exemplars=len(ids), ids=ids, train_ids=[i for i in ids if i not in val], val_ids=val),
          open(os.path.join(dst, "dataset.json"), "w"))
print(dst, len(ids) - len(val), "train", len(val), "val")
