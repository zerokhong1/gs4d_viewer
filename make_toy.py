#!/usr/bin/env python3
"""Synthetic dynamic scene in the common format, to try the viewer without any trained model.

A static ground disc, a ball orbiting on a circle, a ball bouncing up and down,
and a cloud that fades in and out (appearance / disappearance).

    python make_toy.py -o toy.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from gs4d import Sequence4D, cov_from_quat_scale


def blob(rng, n, radius, color, jitter=0.08):
    d = rng.normal(size=(n, 3))
    pts = d / np.linalg.norm(d, axis=1, keepdims=True) * radius * rng.uniform(0.6, 1.0, (n, 1)) ** (1 / 3)
    cols = np.clip(np.asarray(color) + rng.normal(0, jitter, (n, 3)), 0, 1)
    return pts.astype(np.float32), cols.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="toy.npz")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--points", type=int, default=3000, help="Gaussians per object")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sh", action="store_true",
                    help="add fake degree-1 SH so the balls change colour with the viewing direction (gsplat renderer)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    K, n = a.frames, a.points
    times = np.linspace(0.0, 2.0, K, dtype=np.float32)
    ph = 2 * np.pi * times / times[-1]

    # ground: static disc in the z = 0 plane (z is up)
    r = np.sqrt(rng.uniform(0, 1, 2 * n)) * 2.5
    th = rng.uniform(0, 2 * np.pi, 2 * n)
    ground = np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 0.01, 2 * n)], -1).astype(np.float32)
    g_col = np.clip(0.35 + 0.15 * ((np.floor(ground[:, :1] * 2) + np.floor(ground[:, 1:2] * 2)) % 2)
                    + rng.normal(0, 0.02, (2 * n, 3)), 0, 1).astype(np.float32)

    orbit_p, orbit_c = blob(rng, n, 0.35, (0.9, 0.25, 0.2))
    bounce_p, bounce_c = blob(rng, n, 0.3, (0.2, 0.5, 0.95))
    cloud_p, cloud_c = blob(rng, n, 0.45, (0.95, 0.85, 0.3))

    def rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)

    means = np.zeros((K, 5 * n, 3), np.float32)
    opac = np.full((K, 5 * n), 0.9, np.float32)
    quats = np.zeros((K, 5 * n, 4), np.float32)
    quats[..., 0] = 1.0
    for k in range(K):
        means[k, : 2 * n] = ground
        centre = np.array([1.5 * np.cos(ph[k]), 1.5 * np.sin(ph[k]), 0.45], np.float32)
        means[k, 2 * n: 3 * n] = orbit_p @ rot_z(3 * ph[k]).T + centre
        quats[k, 2 * n: 3 * n] = [np.cos(1.5 * ph[k]), 0, 0, np.sin(1.5 * ph[k])]
        h = 0.3 + 1.4 * abs(np.sin(2 * ph[k]))
        squash = 1.0 - 0.35 * np.exp(-((h - 0.3) / 0.15) ** 2)
        means[k, 3 * n: 4 * n] = bounce_p * np.array([1 / np.sqrt(squash), 1 / np.sqrt(squash), squash],
                                                     np.float32) + np.array([0, 0, h], np.float32)
        means[k, 4 * n:] = cloud_p + np.array([-1.2, -1.2, 1.3], np.float32)
        opac[k, 4 * n:] = 0.9 * np.clip(np.sin(ph[k] / 2) ** 2 * 1.6 - 0.3, 0, 1)

    scales = np.concatenate([
        np.tile([0.05, 0.05, 0.006], (2 * n, 1)),
        rng.uniform(0.015, 0.04, (3 * n, 3)),
    ]).astype(np.float32)
    covs = cov_from_quat_scale(quats, np.broadcast_to(scales, (K, 5 * n, 3)))
    colors = np.concatenate([g_col, orbit_c, bounce_c, cloud_c])

    sh = None
    if a.sh:
        from gs4d import SH_C0
        sh = np.zeros((5 * n, 4, 3), np.float32)
        sh[:, 0] = (colors - 0.5) / SH_C0  # DC term reproduces `colors` when the other terms are zero
        # degree-1 terms are ordered (y, z, x). Each ball gets a strong tint along one horizontal axis.
        sh[2 * n: 3 * n, 3] = [-0.9, 0.9, 0.9]   # orbiting ball: red from -x, cyan from +x
        sh[3 * n: 4 * n, 1] = [0.9, 0.9, -0.9]   # bouncing ball: blue from -y, yellow from +y
        sh[4 * n:, 3] = [0.9, -0.9, 0.9]         # cloud: tint along x
    seq = Sequence4D(times, means, covs, opac, colors, sh=sh,
                     source="synthetic toy scene (make_toy.py)" + (", fake SH" if a.sh else ""))
    seq.save(a.output)
    print(f"Wrote {a.output}: {seq.summary()}")


if __name__ == "__main__":
    main()
