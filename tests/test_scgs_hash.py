#!/usr/bin/env python3
"""SC-GS --use_hash: the numpy port of the tiny-cuda-nn hash grid and of the repo's HashDeformNetwork (needs torch; CPU).

1. A dense level (level 0: 17^3 cells fit into the table) against an independent trilinear lookup in a 3D array.
2. When tinycudann is importable (CUDA, `PYTHONPATH=/home/cont/gs4d_train/site_tcnn`): all 12 levels, including the hashed
   ones, against `tcnn.Encoding` with the repo's configuration, 3D and 4D.
3. The MLP and heads against a torch module written like the repo's `MLP` / `HashDeformNetwork`, fed with the same grid output.
4. A synthetic deform.pth with a hash network converts end to end and gives finite, moving Gaussians.
"""
import os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import convert_scgs as C
from gs4d import Sequence4D
from test_scgs import write_ply


def test_dense_level():
    rng = np.random.default_rng(0)
    levels, total = C.hash_levels(3)
    scale, res, off, size = levels[0]
    params = rng.normal(size=(total, 2)).astype(np.float32)
    x = rng.uniform(0.05, 0.9, (40, 3)).astype(np.float32)
    got = C.hash_encode(x, params.reshape(-1), 3)[:, :2]
    assert res == 16 and res ** 3 <= size, (res, size)
    flat = params[off:off + size]   # dense layout: x runs fastest, then y, then z
    cube = np.zeros((res + 1, res + 1, res + 1, 2), np.float32)
    for a in range(res):
        for b in range(res):
            for c in range(res):
                cube[a, b, c] = flat[a + res * (b + res * c)]
    want = np.zeros_like(got)
    for i, p in enumerate(x):
        q = p * np.float32(scale) + np.float32(0.5)
        g = np.floor(q).astype(int)
        f = q - g
        for corner in range(8):
            bits = [(corner >> d) & 1 for d in range(3)]
            w = np.prod([f[d] if bits[d] else 1 - f[d] for d in range(3)])
            want[i] += w * cube[g[0] + bits[0], g[1] + bits[1], g[2] + bits[2]]
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
    print("ok  hash grid, dense level 0 = independent trilinear lookup")


def test_against_tcnn():
    try:
        import torch
        import tinycudann as tcnn
    except ImportError:
        print("skip  hash grid against tinycudann (not importable)")
        return
    if not torch.cuda.is_available():
        print("skip  hash grid against tinycudann (no CUDA)")
        return
    cfg = {"otype": "Grid", "type": "Hash", "n_levels": 12, "n_features_per_level": 2, "log2_hashmap_size": 19,
           "base_resolution": 16, "per_level_scale": 2.0, "interpolation": "Linear"}
    for dims in (3, 4):
        enc = tcnn.Encoding(dims, cfg, dtype=torch.float32)
        params = enc.params.detach().float().cpu().numpy()
        params = np.random.default_rng(dims).normal(0, 0.5, params.shape).astype(np.float32)
        with torch.no_grad():
            enc.params.copy_(torch.from_numpy(params))
            x = torch.rand(2000, dims, device="cuda")
            want = enc(x).float().cpu().numpy()
        got = C.hash_encode(x.cpu().numpy(), params, dims)
        err = np.abs(got - want).max()
        assert err < 2e-4, f"{dims}D: max |numpy - tcnn| = {err}"
        print(f"ok  hash grid {dims}D matches tinycudann on 2000 random points, all 12 levels (max |diff| {err:.1e})")


def make_hash_sd(torch, hash_time=False, local_frame=True, pred_opacity=True, pred_color=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s, sd=0.3: (torch.randn(*s, generator=g) * sd)   # noqa: E731
    _, total = C.hash_levels(4 if hash_time else 3)
    sd = {"network.hashgrid.encoding.params": rn(total * 2, sd=0.5), "network.hashgrid.current_step": torch.tensor(20000, dtype=torch.int32),
          "network.bbox": torch.tensor([-2.0, 2.0])}
    n_in = 24 if hash_time else 24 + 13

    def mlp(name, din, dout, hidden, nh, skip=()):
        dims = [din] + [hidden] * nh
        i = 0
        sd[f"network.{name}.layers.0.weight"], sd[f"network.{name}.layers.0.bias"] = rn(hidden, din, sd=0.2), rn(hidden, sd=0.05)
        for j in range(nh - 1):
            din_j = hidden + din if j in skip else hidden
            k = 2 * (j + 1)
            sd[f"network.{name}.layers.{k}.weight"], sd[f"network.{name}.layers.{k}.bias"] = rn(hidden, din_j, sd=0.2), rn(hidden, sd=0.05)
        k = 2 * nh
        sd[f"network.{name}.layers.{k}.weight"], sd[f"network.{name}.layers.{k}.bias"] = rn(dout, hidden, sd=0.2), rn(dout, sd=0.05)

    mlp("mlp", n_in, 64, 64, 2 if hash_time else 4, skip=() if hash_time else (2,))
    for name, d in (("translate_layer", 3), ("rotation_layer", 4), ("scaling_layer", 3)):
        mlp(name, 64, d, 16, 1)
    if local_frame:
        mlp("local_rotation_layer", 64, 4, 16, 1)
    if pred_opacity:
        mlp("opacity_layer", 64, 1, 16, 1)
    if pred_color:
        mlp("color_layer", 64, 3, 16, 1)
    return sd


def torch_reference(torch, sd, embed, hash_time):
    """The repo's MLP (nn.Linear + ReLU, skip=[2] before layer 6) and heads, on the grid features `embed` (+ time embedding)."""
    def run(name, x, skip_in=None):
        ls = sorted({int(k.split(".")[3]) for k in sd if k.startswith(f"network.{name}.layers.")})
        h = x
        for n, i in enumerate(ls):
            if skip_in is not None and i == 6:
                h = torch.cat([h, skip_in], -1)
            h = torch.nn.functional.linear(h, sd[f"network.{name}.layers.{i}.weight"], sd[f"network.{name}.layers.{i}.bias"])
            if n < len(ls) - 1:
                h = torch.relu(h)
        return h
    h = run("mlp", embed, None if hash_time else embed)
    return h, run


def test_mlp_heads():
    import torch
    for hash_time in (False, True):
        sd = make_hash_sd(torch, hash_time=hash_time, seed=int(hash_time))
        net = C.HashNodeMLP(sd)
        assert net.hash_time == hash_time and net.local_frame and net.pred_opacity and net.pred_color
        rng = np.random.default_rng(1)
        xyz = rng.uniform(-1.5, 1.5, (30, 3)).astype(np.float32)
        t = 0.37
        got = net(xyz, t)
        # reference: same grid features, torch for everything else
        x = (xyz - net.bbox[0]) / (net.bbox[1] - net.bbox[0])
        if hash_time:
            grid = C.hash_encode(np.concatenate([x, np.full((30, 1), t, np.float32)], 1), net.params, 4)
            embed = torch.from_numpy(grid)
        else:
            grid = C.hash_encode(x, net.params, 3)
            te = [t] + [f(t * 2.0 ** j) for j in range(6) for f in (np.sin, np.cos)]
            te = np.tile(np.array(te, np.float32) / 13 * 24, (30, 1))
            embed = torch.from_numpy(np.concatenate([grid, te], 1))
        h, run = torch_reference(torch, sd, embed, hash_time)
        want = {"d_xyz": run("translate_layer", h), "d_rot": run("rotation_layer", h) + torch.tensor([1.0, 0, 0, 0]),
                "d_scale": torch.tanh(run("scaling_layer", h)) * np.log(2.0), "local_rot": run("local_rotation_layer", h),
                "d_opacity": run("opacity_layer", h), "d_color": run("color_layer", h)}
        for k, v in want.items():
            assert np.allclose(got[k], v.numpy(), atol=1e-4), (hash_time, k, np.abs(got[k] - v.numpy()).max())
    print("ok  hash network: MLP, skip connection, heads (translate, rotation + (1,0,0,0), scaling, local rotation, opacity, colour) match torch; 3D and 4D grid")


def test_convert():
    import torch
    rng = np.random.default_rng(3)
    N, M = 40, 12
    attrs = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"] + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)] + [f"fea_{i}" for i in range(2)])
    c = {n: rng.normal(0, 0.5, N).astype(np.float32) for n in attrs}
    sd = make_hash_sd(torch, seed=5)
    sd["nodes"] = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.5, (M, 5)).astype(np.float32)))
    sd["_node_radius"] = torch.nn.Parameter(torch.zeros(M))
    sd["_node_weight"] = torch.nn.Parameter(torch.zeros(M, 1))
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("point_cloud", "deform"):
            os.makedirs(os.path.join(tmp, sub, "iteration_1"))
        write_ply(os.path.join(tmp, "point_cloud", "iteration_1", "point_cloud.ply"), c, attrs)
        torch.save(sd, os.path.join(tmp, "deform", "iteration_1", "deform.pth"))
        out = os.path.join(tmp, "s.npz")
        subprocess.run([sys.executable, os.path.join(ROOT, "convert_scgs.py"), tmp, "-o", out, "--frames", "3"], check=True, capture_output=True)
        seq = Sequence4D.load(out)
    assert seq.num_frames == 3 and seq.num_gaussians == N
    assert np.isfinite(seq.means).all() and np.abs(seq.means[0] - seq.means[2]).max() > 1e-3, "nothing moves"
    assert seq.colors.ndim == 3, "colour head was not applied"
    print("ok  hash-grid deform.pth converts end to end (moving Gaussians, colour per keyframe)")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  scgs hash (torch not installed)")
        sys.exit(0)
    test_dense_level()
    test_against_tcnn()
    test_mlp_heads()
    test_convert()
    print("all checks passed")
