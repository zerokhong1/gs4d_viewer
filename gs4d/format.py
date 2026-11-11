"""Common "Gaussians per keyframe" container read by the viewer.

One .npz file:
    times      [K]            float32  timestamps of the keyframes (any unit, increasing)
    means      [K,N,3] | [N,3]   float32
    covs       [K,N,6] | [N,6]   float32  upper triangle of the 3x3 covariance: xx xy xz yy yz zz
    opacities  [K,N]   | [N]     float32  in [0,1]
    colors     [K,N,3] | [N,3]   float32  RGB in [0,1] (view-independent)
    interpolate  bool   False when keyframes do not share Gaussian identities
    source       str    free text, shown in the viewer

A field without the leading K axis is constant over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SH_C0 = 0.28209479177387814
_TRIU = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit-normalizes wxyz quaternions [...,4] and returns rotation matrices [...,3,3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def cov_from_quat_scale(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """wxyz quaternions [...,4] and linear scales [...,3] -> packed covariance [...,6]."""
    M = quat_to_rotmat(quats.astype(np.float32)) * scales.astype(np.float32)[..., None, :]
    return pack_cov(M @ np.swapaxes(M, -1, -2))


def pack_cov(full: np.ndarray) -> np.ndarray:
    return full[..., _TRIU[0], _TRIU[1]]


def unpack_cov(packed: np.ndarray) -> np.ndarray:
    full = np.empty(packed.shape[:-1] + (3, 3), dtype=packed.dtype)
    full[..., _TRIU[0], _TRIU[1]] = packed
    full[..., _TRIU[1], _TRIU[0]] = packed
    return full


@dataclass
class Sequence4D:
    times: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    interpolate: bool = True
    source: str = ""

    def __post_init__(self):
        self.times = np.asarray(self.times, np.float32).reshape(-1)
        K = len(self.times)
        if K == 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be non-empty and strictly increasing")
        N = self.means.shape[-2]
        for name, tail in (("means", (N, 3)), ("covs", (N, 6)), ("opacities", (N,)), ("colors", (N, 3))):
            a = np.asarray(getattr(self, name), np.float32)
            if a.shape not in (tail, (K,) + tail):
                raise ValueError(f"{name} has shape {a.shape}, expected {tail} or {(K,) + tail}")
            setattr(self, name, a)

    @property
    def num_frames(self) -> int:
        return len(self.times)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[-2]

    def _field(self, a: np.ndarray, static_ndim: int, i0: int, i1: int, w: float) -> np.ndarray:
        if a.ndim == static_ndim:
            return a
        if w == 0.0 or i0 == i1:
            return a[i0]
        return (1.0 - w) * a[i0] + w * a[i1]

    def at(self, t: float):
        """Gaussians at time t: (means [N,3], covs [N,6], opacities [N], colors [N,3])."""
        ts = self.times
        t = float(np.clip(t, ts[0], ts[-1]))
        i1 = int(np.searchsorted(ts, t, side="left"))
        i1 = min(max(i1, 0), len(ts) - 1)
        i0 = i1 if ts[i1] == t else max(i1 - 1, 0)
        w = 0.0 if i0 == i1 else float((t - ts[i0]) / (ts[i1] - ts[i0]))
        if not self.interpolate:  # nearest keyframe
            i0 = i1 = i1 if w >= 0.5 else i0
            w = 0.0
        return (
            self._field(self.means, 2, i0, i1, w),
            self._field(self.covs, 2, i0, i1, w),
            self._field(self.opacities, 1, i0, i1, w),
            self._field(self.colors, 2, i0, i1, w),
        )

    def save(self, path: str) -> None:
        np.savez(
            path,
            times=self.times,
            means=self.means,
            covs=self.covs,
            opacities=self.opacities,
            colors=self.colors,
            interpolate=np.bool_(self.interpolate),
            source=np.str_(self.source),
        )

    @staticmethod
    def load(path: str) -> "Sequence4D":
        with np.load(path, allow_pickle=False) as z:
            return Sequence4D(
                times=z["times"],
                means=z["means"],
                covs=z["covs"],
                opacities=z["opacities"],
                colors=z["colors"],
                interpolate=bool(z["interpolate"]) if "interpolate" in z else True,
                source=str(z["source"]) if "source" in z else "",
            )

    def summary(self) -> str:
        dyn = [n for n, a, d in (("means", self.means, 2), ("covs", self.covs, 2),
                                 ("opacities", self.opacities, 1), ("colors", self.colors, 2)) if a.ndim > d]
        return (f"{self.num_gaussians:,} Gaussians x {self.num_frames} keyframes, "
                f"t in [{self.times[0]:.3f}, {self.times[-1]:.3f}], "
                f"time-varying: {', '.join(dyn) or 'none'}")
