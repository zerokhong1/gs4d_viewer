"""Common "Gaussians per keyframe" container read by the viewer.

One .npz file:
    times      [K]            float32  timestamps of the keyframes (any unit, increasing)
    means      [K,N,3] | [N,3]   float32
    covs       [K,N,6] | [N,6]   float32  upper triangle of the 3x3 covariance: xx xy xz yy yz zz
    opacities  [K,N]   | [N]     float32  in [0,1]
    colors     [K,N,3] | [N,3]   float32  RGB in [0,1] (view-independent)
    sh         [K,N,S,3] | [N,S,3]  float32  OPTIONAL spherical-harmonics coefficients, S = (degree+1)^2,
                                    3DGS convention: colour = max(SH(view direction) + 0.5, 0), sh[..., 0, :] is
                                    the DC term (f_dc), the rest is f_rest. Used by the gsplat renderer only;
                                    `colors` must still be filled (browser renderer, files written before `sh`).
    interpolate  bool   False when keyframes do not share Gaussian identities
    source       str    free text, shown in the viewer

A field without the leading K axis is constant over time.

Compact storage (`save(path, compact=True)`, or `python -m gs4d in.npz out.npz`) replaces the
time-varying fields by smaller ones; `load` reads both layouts:
    means      -> means_ref [N,3] float32 (first keyframe) + means_d16 [K,N,3] float16 (offset from it)
    covs       -> covs_scale [N] float32 + covs_q [K,N,6] int16: Cholesky factor L (cov = L L^T, entries
                  l00 l10 l20 l11 l21 l22) divided by covs_scale, times 32767. Always decodes to a PSD matrix.
    opacities  -> opacities_u8 [K,N] uint8
    colors     -> colors_u8 [K,N,3] uint8
Static fields and `sh` stay float32. A compact field is decoded one keyframe at a time when it is used.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import Optional

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


class _Packed:
    """Time-varying field kept in its compact encoding; behaves like a read-only float32 array [K, ...].

    `a[k]` with an integer k decodes one keyframe. Anything else (slicing, numpy functions, methods such
    as .max) decodes the whole field first, which costs the full float32 memory.
    """
    dtype = np.dtype(np.float32)
    tail: tuple = ()

    def __init__(self, raw: np.ndarray):
        self.raw = raw

    @property
    def shape(self):
        return (self.raw.shape[0], self.raw.shape[1]) + self.tail

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.raw.shape[0]

    def frame(self, k: int) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            return self.frame(int(k))
        return np.asarray(self)[k]

    def __array__(self, dtype=None, copy=None):
        out = np.empty(self.shape, np.float32)
        for k in range(len(self)):
            out[k] = self.frame(k)
        return out if dtype is None else out.astype(dtype, copy=False)

    def __getattr__(self, name):  # numpy methods (max, mean, ...) on the decoded field
        if name.startswith("__") or name in ("raw", "ref", "scale"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)


class _PackedMeans(_Packed):
    tail = (3,)

    def __init__(self, ref: np.ndarray, raw: np.ndarray):
        self.ref, self.raw = ref, raw

    @staticmethod
    def encode(a: np.ndarray):
        ref = np.array(a[0], np.float32)
        raw = np.empty(a.shape, np.float16)
        with np.errstate(over="ignore"):
            for k in range(a.shape[0]):
                raw[k] = a[k] - ref
        return _PackedMeans(ref, raw) if np.isfinite(raw).all() else None  # offsets beyond float16 range

    def frame(self, k):
        return self.ref + self.raw[k]

    def files(self):
        return {"means_ref": self.ref, "means_d16": self.raw}


class _PackedCovs(_Packed):
    tail = (6,)

    def __init__(self, scale: np.ndarray, raw: np.ndarray):
        self.scale, self.raw = scale, raw

    @staticmethod
    def _cholesky(c: np.ndarray) -> np.ndarray:
        """Packed covariance [N,6] -> lower Cholesky factor [N,6] (l00 l10 l20 l11 l21 l22), safe for rank-deficient input."""
        xx, xy, xz, yy, yz, zz = (c[:, i].astype(np.float64) for i in range(6))
        with np.errstate(divide="ignore", invalid="ignore"):
            l00 = np.sqrt(np.maximum(xx, 0))
            l10 = np.where(l00 > 0, xy / l00, 0.0)
            l20 = np.where(l00 > 0, xz / l00, 0.0)
            l11 = np.sqrt(np.maximum(yy - l10 * l10, 0))
            l21 = np.where(l11 > 0, (yz - l20 * l10) / l11, 0.0)
            l22 = np.sqrt(np.maximum(zz - l20 * l20 - l21 * l21, 0))
        return np.stack([l00, l10, l20, l11, l21, l22], -1)

    @staticmethod
    def encode(a: np.ndarray):
        K, N = a.shape[:2]
        scale = np.zeros(N, np.float64)
        for k in range(K):  # every |L_ij| is at most sqrt(largest diagonal entry)
            np.maximum(scale, np.sqrt(np.maximum(a[k][:, [0, 3, 5]].max(-1), 0)), out=scale)
        scale = np.where(scale > 0, scale, 1.0)
        raw = np.empty((K, N, 6), np.int16)
        for k in range(K):
            raw[k] = np.rint(np.clip(_PackedCovs._cholesky(a[k]) / scale[:, None], -1, 1) * 32767)
        return _PackedCovs(scale.astype(np.float32), raw) if np.isfinite(scale).all() else None

    def frame(self, k):
        L = self.raw[k].astype(np.float32) * (self.scale[:, None] / np.float32(32767))
        l00, l10, l20, l11, l21, l22 = (L[:, i] for i in range(6))
        out = np.empty(L.shape, np.float32)
        out[:, 0] = l00 * l00
        out[:, 1] = l00 * l10
        out[:, 2] = l00 * l20
        out[:, 3] = l10 * l10 + l11 * l11
        out[:, 4] = l10 * l20 + l11 * l21
        out[:, 5] = l20 * l20 + l21 * l21 + l22 * l22
        return out

    def files(self):
        return {"covs_scale": self.scale, "covs_q": self.raw}


class _PackedU8(_Packed):
    """Values in [0,1] as uint8."""

    def __init__(self, raw: np.ndarray, name: str):
        self.raw, self.name, self.tail = raw, name, tuple(raw.shape[2:])

    @staticmethod
    def encode(a: np.ndarray, name: str):
        raw = np.empty(a.shape, np.uint8)
        for k in range(a.shape[0]):
            raw[k] = np.rint(np.clip(a[k], 0, 1) * 255)
        return _PackedU8(raw, name)

    def frame(self, k):
        return self.raw[k].astype(np.float32) * np.float32(1 / 255)

    def files(self):
        return {self.name + "_u8": self.raw}


def _npz_arrays(path: str, lazy: bool) -> dict:
    """Arrays of an .npz. With lazy=True, members stored without compression are memory-mapped, not read."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        names = list(z.files)
        if not lazy:
            return {n: z[n] for n in names}
        with zipfile.ZipFile(path) as zf, open(path, "rb") as f:
            for n in names:
                info = zf.getinfo(n + ".npy")
                arr = None
                if info.compress_type == zipfile.ZIP_STORED:
                    f.seek(info.header_offset + 26)
                    name_len, extra_len = np.frombuffer(f.read(4), "<u2")
                    f.seek(info.header_offset + 30 + int(name_len) + int(extra_len))
                    version = np.lib.format.read_magic(f)
                    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, fortran, dtype = read_header(f)
                    if not fortran and not dtype.hasobject and len(shape) >= 2 and int(np.prod(shape)) > 0:
                        arr = np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape)
                out[n] = z[n] if arr is None else arr
    return out


@dataclass
class Sequence4D:
    times: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    interpolate: bool = True
    source: str = ""
    sh: Optional[np.ndarray] = None

    def __post_init__(self):
        self.times = np.asarray(self.times, np.float32).reshape(-1)
        K = len(self.times)
        if K == 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be non-empty and strictly increasing")
        N = self.means.shape[-2]
        for name, tail in (("means", (N, 3)), ("covs", (N, 6)), ("opacities", (N,)), ("colors", (N, 3))):
            a = getattr(self, name)
            if not isinstance(a, _Packed):
                a = np.asarray(a, np.float32)
            if a.shape not in (tail, (K,) + tail):
                raise ValueError(f"{name} has shape {a.shape}, expected {tail} or {(K,) + tail}")
            setattr(self, name, a)
        if self.sh is not None:
            a = np.asarray(self.sh, np.float32)
            S = a.shape[-2] if a.ndim >= 3 else 0
            if S not in (1, 4, 9, 16) or a.shape not in ((N, S, 3), (K, N, S, 3)):
                raise ValueError(f"sh has shape {a.shape}, expected {(N, 'S', 3)} or {(K, N, 'S', 3)} with S in 1, 4, 9, 16")
            self.sh = a

    @property
    def sh_degree(self) -> Optional[int]:
        return None if self.sh is None else int(round(self.sh.shape[-2] ** 0.5)) - 1

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

    def _bracket(self, t: float):
        """Keyframes around t and the weight of the second one."""
        ts = self.times
        t = float(np.clip(t, ts[0], ts[-1]))
        i1 = int(np.searchsorted(ts, t, side="left"))
        i1 = min(max(i1, 0), len(ts) - 1)
        i0 = i1 if ts[i1] == t else max(i1 - 1, 0)
        w = 0.0 if i0 == i1 else float((t - ts[i0]) / (ts[i1] - ts[i0]))
        if not self.interpolate:  # nearest keyframe
            i0 = i1 = i1 if w >= 0.5 else i0
            w = 0.0
        return i0, i1, w

    def at(self, t: float):
        """Gaussians at time t: (means [N,3], covs [N,6], opacities [N], colors [N,3])."""
        i0, i1, w = self._bracket(t)
        return (
            self._field(self.means, 2, i0, i1, w),
            self._field(self.covs, 2, i0, i1, w),
            self._field(self.opacities, 1, i0, i1, w),
            self._field(self.colors, 2, i0, i1, w),
        )

    def sh_at(self, t: float) -> Optional[np.ndarray]:
        """SH coefficients at time t, [N,S,3], or None when the sequence has none."""
        if self.sh is None:
            return None
        return self._field(self.sh, 3, *self._bracket(t))

    def compact(self) -> "Sequence4D":
        """Re-encodes the time-varying fields in place (see the module docstring); lossy. Returns self."""
        K = self.num_frames
        for name, static_ndim in (("means", 2), ("covs", 2), ("opacities", 1), ("colors", 2)):
            a = getattr(self, name)
            if isinstance(a, _Packed) or a.ndim == static_ndim:
                continue
            packed = {"means": _PackedMeans.encode, "covs": _PackedCovs.encode}.get(name, lambda x: _PackedU8.encode(x, name))(a)
            if packed is not None:
                setattr(self, name, packed)
        return self

    @property
    def is_compact(self) -> bool:
        return any(isinstance(getattr(self, n), _Packed) for n in ("means", "covs", "opacities", "colors"))

    def save(self, path: str, compact: bool = False) -> None:
        """Writes the .npz. compact=True stores the time-varying fields in the smaller lossy encoding
        (about 2.2x smaller); compact=False writes plain float32, also for a sequence that is compact in memory."""
        src = self
        if compact and not all(isinstance(getattr(self, n), _Packed) or getattr(self, n).ndim == d
                               for n, d in (("means", 2), ("covs", 2), ("opacities", 1), ("colors", 2))):
            src = Sequence4D(self.times, self.means, self.covs, self.opacities, self.colors).compact()
        files = {}
        for name in ("means", "covs", "opacities", "colors"):
            a = getattr(src, name)
            if isinstance(a, _Packed):
                files.update(a.files() if compact else {name: np.asarray(a)})
            else:
                files[name] = a
        if self.sh is not None:
            files["sh"] = self.sh
        np.savez(path, times=self.times, interpolate=np.bool_(self.interpolate), source=np.str_(self.source), **files)

    @staticmethod
    def load(path: str, lazy: bool = False) -> "Sequence4D":
        """Reads either layout. lazy=True memory-maps the big arrays instead of reading them: loading is
        immediate and only the keyframes that are used get paged in. The file must not be overwritten
        while a lazily loaded sequence is alive."""
        z = _npz_arrays(path, lazy)
        if "means_d16" in z:
            means = _PackedMeans(np.asarray(z["means_ref"], np.float32), z["means_d16"])
        else:
            means = z["means"]
        covs = _PackedCovs(np.asarray(z["covs_scale"], np.float32), z["covs_q"]) if "covs_q" in z else z["covs"]
        opac = _PackedU8(z["opacities_u8"], "opacities") if "opacities_u8" in z else z["opacities"]
        cols = _PackedU8(z["colors_u8"], "colors") if "colors_u8" in z else z["colors"]
        return Sequence4D(
            times=np.array(z["times"]),
            means=means,
            covs=covs,
            opacities=opac,
            colors=cols,
            interpolate=bool(z["interpolate"]) if "interpolate" in z else True,
            source=str(z["source"]) if "source" in z else "",
            sh=z["sh"] if "sh" in z else None,
        )

    def summary(self) -> str:
        dyn = [n for n, a, d in (("means", self.means, 2), ("covs", self.covs, 2),
                                 ("opacities", self.opacities, 1), ("colors", self.colors, 2)) if a.ndim > d]
        return (f"{self.num_gaussians:,} Gaussians x {self.num_frames} keyframes, "
                f"t in [{self.times[0]:.3f}, {self.times[-1]:.3f}], "
                f"time-varying: {', '.join(dyn) or 'none'}"
                + ("" if self.sh is None else f", SH degree {self.sh_degree}"
                   + (" (per keyframe)" if self.sh.ndim == 4 else ""))
                + (", compact storage" if self.is_compact else ""))


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Re-save a sequence in the compact layout (default) or as plain float32.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--float32", action="store_true", help="write plain float32 instead of the compact layout")
    a = ap.parse_args()
    if os.path.exists(a.output) and os.path.samefile(a.input, a.output):
        raise SystemExit("input and output must be different files")
    seq = Sequence4D.load(a.input, lazy=True)
    seq.save(a.output, compact=not a.float32)
    print(f"Wrote {a.output}: {os.path.getsize(a.input) / 1e6:.1f} MB -> {os.path.getsize(a.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
