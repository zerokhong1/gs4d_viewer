"""convert_ed3dgs.py against a torch module written like the repo's deform_network (CPU), on a synthetic model folder.

Two configurations: dynerf-like (opacity and colour heads on, coarse temporal embedding, annealing) and hypernerf-like (depth 0,
opacity and colour heads off, coarse-to-fine temporal table). The temporal embedding is also checked against a hand-written
bilinear lookup. Needs torch.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from gs4d import Sequence4D  # noqa: E402
from gs4d.format import cov_from_quat_scale, unpack_cov  # noqa: E402
from test_scgs import write_ply  # noqa: E402

SH_C0 = 0.28209479177387814


def repo_module(torch, W, D, tdim, gdim, max_emb):
    nn, F = torch.nn, torch.nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            for s in "cf":
                lay = [nn.Linear(tdim + gdim, W)]
                for _ in range(D - 1):
                    lay += [nn.ReLU(), nn.Linear(W, W)]
                setattr(self, f"feature_out_{s}", nn.Sequential(*lay))
                for name, d in (("pos", 3), ("scales", 3), ("rotations", 4), ("opacity", 1), ("rgb", 48)):
                    setattr(self, f"{name}_deform_{s}", nn.Sequential(nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, d)))
            self.weight = nn.Parameter(torch.randn(max_emb, tdim) * 0.5)
            self.offsets = nn.Parameter(torch.zeros(30, 1))

        def embed(self, t, n):
            e = F.interpolate(self.weight[None, None], size=(n, tdim), mode="bilinear", align_corners=True)
            grid = torch.cat([torch.arange(tdim).unsqueeze(-1) / (tdim - 1), torch.ones(tdim, 1) * t], -1)[None, None]
            return F.grid_sample(e, (grid - 0.5) * 2, align_corners=True, mode="bilinear", padding_mode="reflection")[0, 0, 0]

    return Net()


def reference(net, cfg, it, t, z, xyz, ls, quat, logit, sh):
    torch = __import__("torch")
    if cfg["use_coarse_temporal_embedding"]:
        n = cfg["min_embeddings"]
    elif cfg["no_c2f_temporal_embedding"]:
        n = net.weight.shape[0]
    else:
        n = int(cfg["min_embeddings"] + (net.weight.shape[0] - cfg["min_embeddings"]) * min(max(it, 0), cfg["c2f_temporal_iter"]) / cfg["c2f_temporal_iter"])
    h0 = torch.cat([net.embed(t, n)[None].repeat(len(z), 1), z], -1)
    coef = float(np.clip(it / 1000, 0, 1)) if cfg["use_anneal"] else 1.0
    coef_c = float(np.clip((it - cfg["deform_from_iter"]) / 1000, 0, 1)) if cfg["use_anneal"] else 1.0
    for s in "cf":
        h = getattr(net, f"feature_out_{s}")(h0)
        xyz = xyz + getattr(net, f"pos_deform_{s}")(h) * coef
        if not cfg["no_ds"]:
            ls = ls + getattr(net, f"scales_deform_{s}")(h) * coef * coef_c
        if not cfg["no_dr"]:
            quat = quat + getattr(net, f"rotations_deform_{s}")(h) * coef
        if not cfg["no_do"]:
            logit = logit + getattr(net, f"opacity_deform_{s}")(h) * coef * coef_c
        if not cfg["no_dc"]:
            sh = sh + getattr(net, f"rgb_deform_{s}")(h).view(-1, 16, 3) * coef_c
    return xyz, ls, torch.nn.functional.normalize(quat, dim=-1), logit, sh


def run_case(name, cfg, D, it):
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    N, tdim, gdim, W, max_emb = 40, 16, 8, 24, 20
    net = repo_module(torch, W, D, tdim, gdim, max_emb)
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.3) if p.dim() == 1 else None
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "point_cloud", f"iteration_{it}")
        os.makedirs(d)
        attrs = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
                 + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)] + [f"embedding_{i}" for i in range(gdim)])
        c = {n: rng.normal(0, 0.4, N).astype(np.float32) for n in attrs}
        c["opacity"] = rng.normal(1, 1, N).astype(np.float32)
        write_ply(os.path.join(d, "point_cloud.ply"), c, attrs)
        torch.save(net.state_dict(), os.path.join(d, "deformation.pth"))
        with open(os.path.join(tmp, "cfg_args"), "w") as f:
            f.write("Namespace(" + ", ".join(f"{k}={v!r}" for k, v in cfg.items()) + ", source_path='/x')")
        out = os.path.join(tmp, "s.npz")
        subprocess.run([sys.executable, os.path.join(ROOT, "convert_ed3dgs.py"), tmp, "-o", out, "--frames", "4", "--t0", "0.1", "--t1", "0.9", "--sh"],
                       check=True, capture_output=True)
        seq = Sequence4D.load(out)
    f = lambda a: torch.from_numpy(np.asarray(a, np.float32))   # noqa: E731
    xyz = f(np.stack([c[k] for k in "xyz"], -1)); ls = f(np.stack([c[f"scale_{i}"] for i in range(3)], -1))
    quat = f(np.stack([c[f"rot_{i}"] for i in range(4)], -1)); logit = f(c["opacity"])[:, None]
    z = f(np.stack([c[f"embedding_{i}"] for i in range(gdim)], -1))
    dc = np.stack([c[f"f_dc_{i}"] for i in range(3)], -1)[:, None]
    rest = np.stack([c[f"f_rest_{i}"] for i in range(45)], -1).reshape(N, 3, 15).transpose(0, 2, 1)
    sh = f(np.concatenate([dc, rest], 1))
    assert seq.num_frames == 4 and seq.num_gaussians == N
    with torch.no_grad():
        for t in seq.times:
            x2, ls2, q2, lg2, sh2 = reference(net, cfg, it, float(t), z, xyz, ls, quat, logit, sh)
            means, covs, opac, cols = seq.at(float(t))[:4]
            covs = unpack_cov(covs) if covs.shape[-1] == 6 else covs
            want_cov = unpack_cov(cov_from_quat_scale(q2.numpy(), np.exp(ls2.numpy())))
            assert np.allclose(means, x2.numpy(), atol=1e-4), (name, "mean", np.abs(means - x2.numpy()).max())
            assert np.allclose(covs, want_cov, atol=1e-5), (name, "cov")
            assert np.allclose(opac, torch.sigmoid(lg2[:, 0]).numpy(), atol=1e-5), (name, "opacity")
            assert np.allclose(cols, sh2[:, 0].numpy() * SH_C0 + 0.5, atol=1e-4), (name, "colour")
    moved = np.abs(seq.at(float(seq.times[0]))[0] - seq.at(float(seq.times[-1]))[0]).max()
    assert moved > 1e-3, "nothing moves"
    print(f"ok  {name}: means, covariance, opacity, colour equal the repo-style torch module at 4 times")


def test_embedding_lookup():
    import torch
    sys.path.insert(0, ROOT)
    import convert_ed3dgs as C
    torch.manual_seed(2)
    weight = torch.randn(12, 6)
    cfg = dict(C.DEFAULTS, use_coarse_temporal_embedding=True, min_embeddings=5)
    net = C.EmbeddingDeformation.__new__(C.EmbeddingDeformation)
    net.torch, net.cfg, net.weight, net.iteration, net.offset = torch, cfg, weight, 0, 0.0
    t = 0.3
    got = net.temporal(t).numpy()
    W = weight.numpy().astype(np.float64)   # hand-written: resize rows 12 -> 5 (align_corners), then linear in time between rows, columns exact
    pos = np.linspace(0, 11, 5)
    rows = np.stack([(1 - (p - np.floor(p))) * W[int(np.floor(p))] + (p - np.floor(p)) * W[min(int(np.floor(p)) + 1, 11)] for p in pos])
    y = t * 4
    i0 = int(np.floor(y)); i1 = min(i0 + 1, 4)
    want = (1 - (y - i0)) * rows[i0] + (y - i0) * rows[i1]
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
    print("ok  temporal embedding = hand-written bilinear lookup of the resized table")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip  ed3dgs (torch not installed)")
        sys.exit(0)
    test_embedding_lookup()
    dyn = dict(no_ds=False, no_dr=False, no_do=False, no_dc=False, min_embeddings=5, max_embeddings=20, use_coarse_temporal_embedding=True,
               no_c2f_temporal_embedding=False, no_coarse_deform=False, no_fine_deform=False, c2f_temporal_iter=100, deform_from_iter=500, use_anneal=True)
    hyp = dict(dyn, no_do=True, no_dc=True, use_coarse_temporal_embedding=False, use_anneal=False)
    run_case("dynerf-like (depth 1, all heads, annealing at iteration 1200)", dyn, 2, 1200)
    run_case("hypernerf-like (depth 0, no opacity/colour heads, coarse-to-fine table)", hyp, 1, 40)
    print("all checks passed")
