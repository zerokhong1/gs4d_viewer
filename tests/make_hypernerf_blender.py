#!/usr/bin/env python3
"""HyperNeRF scene -> the Blender-style layout the native 4DGS loader (fudan-zvg/4d-gaussian-splatting) reads.

That repo only has Colmap and `transforms_train.json` (D-NeRF) readers, not the HyperNeRF/Nerfies one. This writes
<out>/transforms_train.json, transforms_test.json (frames of the dataset.json train_ids / val_ids written by
tests/make_hypernerf_split.py) and points3d.ply, next to symlinks to rgb/4x. Per frame: file_path, time = time_id /
max(time_id) in [0, 1] and an OpenGL c2w matrix. Intrinsics go in a top-level `camera_angle_x` (4x images, one focal for
the whole scene, principal point at the image centre, no lens distortion: the same simplifications as the other two repos'
Nerfies readers). Do NOT write per-frame fl_x/fl_y/cx/cy: that repo then leaves FovX = FovY = -1 and its rasteriser
uses a wrong focal (WP19c part 3; the model trained on that layout scored 24.2 dB, 1.9 dB under the repo's own eval). World coordinates are
(x - scene.center) * scene.scale, as in those readers.

    python tests/make_hypernerf_blender.py <split>/vrig_slice-banana/slice-banana <out_dir>
"""
import json, os, sys
import numpy as np
from PIL import Image

src, out = os.path.abspath(sys.argv[1]), sys.argv[2]
os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
for n, tgt in (("4x", os.path.join(src, "rgb", "4x")),):
    if not os.path.lexists(os.path.join(out, "rgb", n)):
        os.symlink(tgt, os.path.join(out, "rgb", n))
sc = json.load(open(os.path.join(src, "scene.json")))
meta = json.load(open(os.path.join(src, "metadata.json")))
ds = json.load(open(os.path.join(src, "dataset.json")))
tmax = max(m["time_id"] for m in meta.values())


focal = set()


def frames(ids):
    fr = []
    for i in ids:
        cam = json.load(open(os.path.join(src, "camera", f"{i}.json")))
        R = np.array(cam["orientation"])  # rows: camera x (right), y (down), z (forward) in world axes
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = (np.array(cam["position"]) - np.array(sc["center"])) * sc["scale"]
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL (the loader flips them back)
        W, H = Image.open(os.path.join(src, "rgb", "4x", f"{i}.png")).size
        f = cam["focal_length"] / 4
        fr.append(dict(file_path=f"rgb/4x/{i}", time=meta[i]["time_id"] / tmax, transform_matrix=c2w.tolist()))
        focal.add((f, W))
    return fr


for name, ids in (("train", ds["train_ids"]), ("test", ds["val_ids"])):
    fr = frames(ids)
    assert len(focal) == 1, f"one focal and image width needed for camera_angle_x, got {focal}"
    f, W = next(iter(focal))
    json.dump(dict(camera_angle_x=2 * float(np.arctan(W / 2 / f)), frames=fr), open(os.path.join(out, f"transforms_{name}.json"), "w"))
    print(name, len(ids))

xyz = (np.load(os.path.join(src, "points.npy")) - np.array(sc["center"])) * sc["scale"]
rgb = np.random.RandomState(0).randint(0, 255, (len(xyz), 3)).astype(np.uint8)
dt = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
v = np.empty(len(xyz), dtype=dt)
v["x"], v["y"], v["z"] = xyz.T
v["nx"] = v["ny"] = v["nz"] = 0
v["red"], v["green"], v["blue"] = rgb.T
from plyfile import PlyData, PlyElement
PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(out, "points3d.ply"))
print(len(xyz), "points")
