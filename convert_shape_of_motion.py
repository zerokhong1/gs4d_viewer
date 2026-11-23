#!/usr/bin/env python3
"""Shape of Motion (Wang et al., vye16/shape-of-motion) checkpoint -> common .npz.

Input: `<work_dir>/checkpoints/last.ckpt`, written by flow3d/trainer.py as
torch.save({"model": state_dict, "optimizers": ..., "global_step": ...}). Used keys of the state_dict:
    fg.params.means [G,3]  quats [G,4] (wxyz)  scales [G,3] (log)  colors [G,3] (logit)  opacities [G] (logit)
    fg.params.motion_coefs [G,B]
    motion_bases.params.rots [B,T,6]  transls [B,T,3]          (B motion bases, T training frames)
    bg.params.* (optional static background, same fields without motion_coefs)

Formulas ported from flow3d/params.py and flow3d/scene_model.py (compute_poses_fg), at frame index f:
    c        = softmax(motion_coefs)                               per Gaussian, over the B bases
    r6, tr   = sum_b c_b * rots[b, f],  sum_b c_b * transls[b, f]
    R_f      = columns (x, y, x cross y) with x = normalize(r6[0:3]),
               y = normalize(r6[3:6] - (r6[3:6] . x) x)             (6D rotation, cont_6d_to_rmat)
    position = R_f @ mean + tr
    rotation = R_f @ R(normalize(quat))
    scale = exp(scales), opacity = sigmoid(opacities), colour = sigmoid(colors)   (constant in time, RGB, no SH)
Background Gaussians do not move. Time is the frame index divided by --fps.

Example:
    python convert_shape_of_motion.py outputs/scene/checkpoints/last.ckpt -o scene.npz --fps 30
"""
from __future__ import annotations

import argparse

import numpy as np

from gs4d import Sequence4D, Sequence4DWriter, pack_cov
from gs4d.checkpoint import import_torch, torch_load
from gs4d.format import quat_to_rotmat


def load_checkpoint(path: str) -> dict:
    """Model state_dict of a Shape of Motion checkpoint as numpy arrays."""
    torch = import_torch("convert_shape_of_motion.py")
    # Real checkpoints pickle numpy scalars next to the weights (seen in the released paper-windmill checkpoint),
    # which weights_only=True rejects by default. These are plain data constructors, safe to allow.
    import _codecs
    from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar, np.dtype, _codecs.encode] + [
        type(np.dtype(t)) for t in (np.float64, np.float32, np.int64, np.int32, np.bool_)])
    ckpt = torch_load(path, "convert_shape_of_motion.py", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise SystemExit(f"{path}: not a Shape of Motion checkpoint (no state dict)")
    need = ["fg.params.means", "fg.params.quats", "fg.params.scales", "fg.params.colors", "fg.params.opacities",
            "fg.params.motion_coefs", "motion_bases.params.rots", "motion_bases.params.transls"]
    missing = [k for k in need if k not in sd]
    if missing:
        raise SystemExit(f"{path}: missing {missing}; not a Shape of Motion checkpoint")
    return {k: v.detach().numpy().astype(np.float64) for k, v in sd.items() if hasattr(v, "detach")}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """[...,6] -> [...,3,3]; the first two columns come from Gram-Schmidt on r6[0:3], r6[3:6]."""
    x = r6[..., 0:3] / np.linalg.norm(r6[..., 0:3], axis=-1, keepdims=True)
    y = r6[..., 3:6] - (r6[..., 3:6] * x).sum(-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def static_gaussians(sd: dict, prefix: str):
    """means [G,3], local matrix R(q) * scale [G,3,3], opacities [G], colours [G,3] of fg or bg."""
    q = sd[f"{prefix}.params.quats"]
    M = quat_to_rotmat(q) * np.exp(sd[f"{prefix}.params.scales"])[:, None, :]
    return (sd[f"{prefix}.params.means"], M, sigmoid(sd[f"{prefix}.params.opacities"]).reshape(-1),
            sigmoid(sd[f"{prefix}.params.colors"]))


def foreground_frames(sd: dict, frames: np.ndarray, batch: int = 200_000):
    """One frame index at a time -> foreground means [G,3], packed covs [G,6]."""
    mean, M, _, _ = static_gaussians(sd, "fg")
    logits = sd["fg.params.motion_coefs"]
    coefs = np.exp(logits - logits.max(-1, keepdims=True))
    coefs /= coefs.sum(-1, keepdims=True)
    rots, transls = sd["motion_bases.params.rots"], sd["motion_bases.params.transls"]
    G = len(mean)
    for f in frames:
        means = np.empty((G, 3), np.float32)
        covs = np.empty((G, 6), np.float32)
        for a in range(0, G, batch):
            sl = slice(a, min(a + batch, G))
            R = rot6d_to_matrix(coefs[sl] @ rots[:, f])            # [g,3,3]
            means[sl] = np.einsum("gij,gj->gi", R, mean[sl]) + coefs[sl] @ transls[:, f]
            L = R @ M[sl]
            covs[sl] = pack_cov(L @ np.swapaxes(L, -1, -2))
        yield means, covs


def move_foreground(sd: dict, frames: np.ndarray, batch: int = 200_000):
    """Same as foreground_frames, the whole sequence in memory: means [K,G,3], packed covs [K,G,6]."""
    fr = list(foreground_frames(sd, frames, batch))
    return np.stack([f[0] for f in fr]), np.stack([f[1] for f in fr])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="<work_dir>/checkpoints/last.ckpt")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the training video (time = frame / fps)")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th training frame as a keyframe")
    ap.add_argument("--fg-only", action="store_true", help="drop the static background Gaussians")
    ap.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians with opacity at or below this")
    ap.add_argument("--compact", action="store_true", help="write the compact encoding (lossy, about 2-3x smaller; see docs/FORMAT.md)")
    a = ap.parse_args()

    sd = load_checkpoint(a.checkpoint)
    T = sd["motion_bases.params.rots"].shape[1]
    frames = np.arange(0, T, a.stride)
    _, _, opac, colors = static_gaussians(sd, "fg")
    n_fg, n_bg = len(opac), 0
    b_mean = b_cov = None
    if "bg.params.means" in sd and not a.fg_only:
        b_mean, b_M, b_opac, b_col = static_gaussians(sd, "bg")
        n_bg = len(b_opac)
        b_cov = pack_cov(b_M @ np.swapaxes(b_M, -1, -2)).astype(np.float32)
        opac, colors = np.concatenate([opac, b_opac]), np.concatenate([colors, b_col])
    keep = opac > a.min_opacity
    print(f"{n_fg:,} foreground + {n_bg:,} background Gaussians, {sd['motion_bases.params.rots'].shape[0]} motion bases, "
          f"{T} frames; keeping {keep.sum():,}")
    # one keyframe at a time to disk, so memory does not grow with the number of frames
    with Sequence4DWriter(a.output, compact=a.compact, opacities=opac[keep].astype(np.float32),
                          colors=colors[keep].astype(np.float32), source=f"Shape of Motion: {a.checkpoint}") as wr:
        for f, (means, covs) in zip(frames, foreground_frames(sd, frames)):
            if n_bg:
                means = np.concatenate([means, b_mean.astype(np.float32)])
                covs = np.concatenate([covs, b_cov])
            wr.add_keyframe(np.float32(f / a.fps), means=means[keep], covs=covs[keep])
    print(f"Wrote {wr.path}: {Sequence4D.load(wr.path, lazy=True).summary()}")


if __name__ == "__main__":
    main()
