"""gs4d.open: recognition of each kind of output on fake layouts (no torch, no GPU, no viewer)."""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gs4d import open as gopen  # noqa: E402


def ply(path, props):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + \
           "".join(f"property float {p}\n" for p in props) + "end_header\n"
    path.write_bytes(head.encode() + np.zeros(len(props), "<f4").tobytes())


BASE = ["x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"]
ST = BASE + ["trbf_center", "trbf_scale", "motion_0", "omega_0"]
SCGS = BASE + ["fea_0", "fea_1"]


def check(path, kind, converter, extra=()):
    info = gopen.detect(Path(path))
    assert info["kind"] == kind, (path, info)
    assert info["converter"] == converter, (path, info)
    assert list(info["extra"]) == list(extra), (path, info)
    return info


def test_all():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # common format and Dynamic 3D Gaussians
        np.savez(d / "c.npz", times=0, means=0, covs=0, opacities=0, colors=0)
        check(d / "c.npz", "common", None)
        np.savez(d / "params.npz", means3D=0, unnorm_rotations=0)
        check(d / "params.npz", "dynamic3dgs", "convert_per_frame.py")
        np.savez(d / "other.npz", foo=0)
        # per-frame folder, plain and hustvl layout
        for i in range(3):
            ply(d / "pf" / f"frame_{i}.ply", BASE)
            ply(d / "hv" / "gaussian_pertimestamp" / f"time_{i}.ply", BASE)
        check(d / "pf", "per_frame", "convert_per_frame.py")
        info = check(d / "hv", "per_frame", "convert_per_frame.py")
        assert info["src"].name == "gaussian_pertimestamp"
        # native 4DGS (D-NeRF: cfg_args says dnerf)
        (d / "n").mkdir()
        (d / "n" / "chkpnt7000.pth").write_bytes(b"x")
        check(d / "n" / "chkpnt7000.pth", "native4d", "convert_native4d.py")
        (d / "n" / "cfg_args").write_text("Namespace(source_path='/x/data/dnerf/ball')")
        check(d / "n" / "chkpnt7000.pth", "native4d", "convert_native4d.py", ["--t0", "0", "--t1", "1"])
        check(d / "n", "native4d", "convert_native4d.py", ["--t0", "0", "--t1", "1"])
        # Shape of Motion, file and folder
        (d / "som" / "checkpoints").mkdir(parents=True)
        (d / "som" / "checkpoints" / "last.ckpt").write_bytes(b"x")
        check(d / "som", "shape_of_motion", "convert_shape_of_motion.py")
        check(d / "som" / "checkpoints" / "last.ckpt", "shape_of_motion", "convert_shape_of_motion.py")
        # Deformable-3DGS vs SC-GS, D-NeRF through cfg_args
        for name, props in (("d3", BASE), ("sc", SCGS)):
            ply(d / name / "point_cloud" / "iteration_5000" / "point_cloud.ply", BASE)
            ply(d / name / "point_cloud" / "iteration_20000" / "point_cloud.ply", props)
            (d / name / "deform" / "iteration_20000").mkdir(parents=True)
            (d / name / "cfg_args").write_text("Namespace(is_blender=True)")
        check(d / "d3", "deformable3dgs", "convert_deformable3dgs.py", ["--t0", "0", "--t1", "1"])
        check(d / "sc", "scgs", "convert_scgs.py", ["--t0", "0", "--t1", "1"])
        (d / "sc" / "cfg_args").unlink()
        check(d / "sc", "scgs", "convert_scgs.py")
        # 4DGaussians: deformation.pth beside the ply, no deform/; also holds gaussian_pertimestamp/ (must not win)
        ply(d / "hv4" / "point_cloud" / "iteration_20000" / "point_cloud.ply", BASE)
        (d / "hv4" / "point_cloud" / "iteration_20000" / "deformation.pth").write_bytes(b"x")
        for i in range(3):
            ply(d / "hv4" / "gaussian_pertimestamp" / f"time_{i}.ply", BASE)
        check(d / "hv4", "4dgaussians", "convert_4dgaussians.py")
        (d / "hv4" / "cfg_args").write_text("Namespace(source_path='/x/data/dnerf/ball')")
        check(d / "hv4", "4dgaussians", "convert_4dgaussians.py", ["--t0", "0", "--t1", "1"])
        # Spacetime Gaussians, ply and folder
        ply(d / "st" / "point_cloud" / "iteration_25000" / "point_cloud.ply", ST)
        check(d / "st" / "point_cloud" / "iteration_25000" / "point_cloud.ply", "spacetime", "convert_spacetime.py")
        check(d / "st", "spacetime", "convert_spacetime.py")
        # cameras: cameras.json wins, else transforms_test.json of cfg_args' source_path, else nothing
        assert gopen.find_cameras(d / "st") is None
        ds = d / "dataset"
        ds.mkdir()
        (d / "hv4" / "cfg_args").write_text(f"Namespace(source_path='{ds}', model_path='x')")
        assert gopen.find_cameras(d / "hv4") is None                 # dataset has no transforms_test.json
        (ds / "transforms_test.json").write_text('{"frames": [{"transform_matrix": []}]}')
        assert gopen.find_cameras(d / "hv4") == ds / "transforms_test.json"
        (d / "hv4" / "cameras.json").write_text('[{"position": [0, 0, 0], "rotation": []}]')   # 3DGS format: unreadable
        assert gopen.find_cameras(d / "hv4") == ds / "transforms_test.json"
        (d / "hv4" / "cameras.json").write_text('{"frames": [{"c2w": []}]}')
        assert gopen.find_cameras(d / "hv4") == (d / "hv4" / "cameras.json").resolve()
        assert gopen.find_cameras(d / "st") is None
        (d / "st" / "cameras.json").write_text('[{"c2w": []}]')
        assert gopen.find_cameras(d / "st" / "point_cloud" / "iteration_25000" / "point_cloud.ply") == d / "st" / "cameras.json"
        import io, contextlib
        for extra, want in (([], True), (["--cameras", "mine.json"], False)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert gopen.main([str(d / "hv4"), "--dry-run", *extra]) == 0
            out = buf.getvalue()
            assert (str(d / "hv4" / "cameras.json") in out) == want and ("mine.json" in out) == (not want), out
            assert out.count("--cameras") == 1, out
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert gopen.main([str(d / "hv4"), "--dry-run", "--compare", "b.npz"]) == 0
        assert buf.getvalue().split("viewer:")[1].count("--cameras") == 1, buf.getvalue()  # also with --compare (WP29)
        # Nerfies / HyperNeRF dataset behind cfg_args: the held-out cameras are written next to the cache (WP27)
        import json
        hy = d / "hyper"
        (hy / "camera").mkdir(parents=True)
        (hy / "rgb" / "2x").mkdir(parents=True)
        (hy / "rgb" / "4x").mkdir()
        rot = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # world-to-camera
        for k, name in enumerate(("a", "b", "c")):
            (hy / "camera" / f"{name}.json").write_text(json.dumps(dict(
                orientation=rot.tolist(), position=[1.0 + k, 2.0, 3.0], focal_length=800.0, image_size=[400, 720])))
        (hy / "metadata.json").write_text(json.dumps({n: dict(time_id=2 * k, warp_id=k) for k, n in enumerate("abc")}))
        (hy / "dataset.json").write_text(json.dumps(dict(ids=["a", "b", "c"], train_ids=["a", "c"], val_ids=["b"])))
        (hy / "scene.json").write_text(json.dumps(dict(center=[1.0, 1.0, 1.0], scale=0.5)))
        (d / "sc" / "cfg_args").write_text(f"Namespace(source_path='{hy}')")
        assert gopen.nerfies_scene(d / "sc") == hy and gopen.nerfies_scene(d / "st") is None
        for raw, pos, t in ((False, [0.5, 0.5, 1.0], 0.5), (True, [2.0, 2.0, 3.0], 0.5)):
            assert gopen.write_nerfies_cameras(hy, d / "cams.json", raw=raw) == 1      # val_ids only
            cj = json.loads((d / "cams.json").read_text())
            fr = cj["frames"][0]
            c2w = np.array(fr["c2w"])
            assert cj["convention"] == "opencv" and fr["name"] == "b" and fr["fx"] == 200.0 and abs(fr["time"] - t) < 1e-12
            assert np.allclose(c2w[:3, 3], pos) and np.allclose(c2w[:3, :3], rot.T) and (fr["w"], fr["h"]) == (100, 180)
        (hy / "dataset.json").unlink()
        assert gopen.write_nerfies_cameras(hy, d / "cams.json", raw=True) == 3          # no split: every camera
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert gopen.main([str(d / "sc"), "--dry-run"]) == 0
        assert f"--cameras {(d / 'sc.gs4d.cameras.json').resolve()}" in buf.getvalue(), buf.getvalue()
        assert not (d / "sc.gs4d.cameras.json").exists()                              # a dry run writes nothing
        # size estimate: --sh adds SH once per Gaussian, for native 4DGS at every keyframe (WP26)
        big = d / "big" / "point_cloud" / "iteration_1" / "point_cloud.ply"
        big.parent.mkdir(parents=True)
        big.write_bytes(b"0" * (248 * 1000))
        info = dict(kind="deformable3dgs", src=d / "big")
        assert gopen.estimate_bytes(info, ["--frames", "10"]) == 1000 * 10 * 40
        assert gopen.estimate_bytes(info, ["--frames", "10", "--sh"]) == 1000 * 10 * 40 + 1000 * 192
        (d / "big.pth").write_bytes(b"0" * (400 * 1000))
        info = dict(kind="native4d", src=d / "big.pth")
        assert gopen.estimate_bytes(info, ["--frames", "10"]) == 1000 * 10 * 40
        assert gopen.estimate_bytes(info, ["--frames", "10", "--sh"]) == 1000 * 10 * (40 + 192)
        # unknown things are refused, with the checks listed
        ply(d / "plain.ply", BASE)
        (d / "empty").mkdir()
        for bad in (d / "other.npz", d / "plain.ply", d / "empty"):
            try:
                gopen.detect(bad)
            except gopen.NotRecognised as e:
                assert str(e)
            else:
                raise AssertionError(f"{bad} should not be recognised")
        assert gopen.main([str(d / "empty")]) == 2
        # cache name and freshness
        cp = gopen.cache_path(d / "som", None)
        assert cp == (d / "som.gs4d.npz").resolve()
        assert gopen.cache_path(d / "params.npz", None).name == "params.gs4d.npz"
        assert not gopen.is_fresh(cp, d / "som")
        cp.write_bytes(b"x")
        assert gopen.is_fresh(cp, d / "som")
        t = time.time() + 10
        os.utime(d / "som" / "checkpoints" / "last.ckpt", (t, t))
        assert not gopen.is_fresh(cp, d / "som")
        assert gopen.main([str(d / "som"), "--dry-run", "--port", "1"]) == 0
        # --compact above 1 GB (sparse files give the sizes without writing 1 GB)
        big = d / "big" / "point_cloud" / "iteration_1" / "point_cloud.ply"
        ply(big, BASE)
        with open(big, "r+b") as f:
            f.truncate(5 * 2**20 * 248)         # ~5M Gaussians x 60 keyframes x 40 B = 12 GB
        (d / "big" / "deform").mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert gopen.main([str(d / "big"), "--dry-run"]) == 0
        assert "--compact" in buf.getvalue(), buf.getvalue()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert gopen.main([str(d / "som"), "--dry-run", "--reconvert"]) == 0
        assert "--compact" not in buf.getvalue()
    print("test_open OK")


if __name__ == "__main__":
    test_all()
