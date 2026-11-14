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

Static group (written by `save` unless `split_static=False`): Gaussians that do not change over the keyframes in
any time-varying field are stored once instead of K times. The file then has
    static_index       [Ns] int64   positions of the static Gaussians among the N Gaussians (sorted)
    <field>            [K,Nd,...]   the time-varying fields above (plain or compact) hold only the Nd = N - Ns
                                    other Gaussians, in their original order
    <field>_static     [Ns,...] float32  value of the static Gaussians, for each of those fields
Fields without the K axis keep all N Gaussians. In memory nothing changes for the caller: every field still looks
like [K,N,...] and `at(t)` returns all N Gaussians in the original order.
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


class _Virtual:
    """Time-varying field that is not stored as one float32 array; behaves like a read-only float32 array [K, ...].

    `a[k]` with an integer k builds one keyframe. Anything else (slicing, numpy functions, methods such
    as .max) builds the whole field first, which costs the full float32 memory.
    """
    dtype = np.dtype(np.float32)
    tail: tuple = ()

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

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
        if name.startswith("__") or name in ("raw", "ref", "scale", "dyn", "static", "layout", "name"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)


class _Packed(_Virtual):
    """Time-varying field kept in its compact encoding (decoded one keyframe at a time)."""

    def __init__(self, raw: np.ndarray):
        self.raw = raw

    @property
    def shape(self):
        return (self.raw.shape[0], self.raw.shape[1]) + self.tail


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

    def take(self, sel):
        return _PackedMeans(self.ref[sel], self.raw[:, sel])


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

    def take(self, sel):
        return _PackedCovs(self.scale[sel], self.raw[:, sel])


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

    def take(self, sel):
        return _PackedU8(self.raw[:, sel], self.name)


class _Layout:
    """Where the static and the other (dynamic) Gaussians sit among the N Gaussians of a sequence."""

    def __init__(self, static_idx: np.ndarray, n: int):
        self.n = int(n)
        self.static_idx = np.asarray(static_idx, np.int64).reshape(-1)
        mask = np.zeros(self.n, bool)
        mask[self.static_idx] = True
        if len(self.static_idx) == 0 or mask.sum() != len(self.static_idx) or np.any(np.diff(self.static_idx) <= 0):
            raise ValueError("static_index must be non-empty, sorted, unique and inside the Gaussian range")
        self.dyn_idx = np.flatnonzero(~mask)
        a, ns = int(self.static_idx[0]), len(self.static_idx)
        self.run = (a, a + ns) if int(self.static_idx[-1]) - a + 1 == ns else None  # one contiguous block: plain copies

    def dyn_sel(self):
        """Index of the dynamic Gaussians, as a slice when they are one block."""
        if self.run and self.run[1] == self.n:
            return slice(0, self.run[0])
        if self.run and self.run[0] == 0:
            return slice(self.run[1], self.n)
        return self.dyn_idx

    def merge(self, d: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Dynamic rows [Nd,...] and static rows [Ns,...] -> all Gaussians [N,...] in the original order."""
        out = np.empty((self.n,) + s.shape[1:], np.float32)
        if self.run:
            a, b = self.run
            out[:a], out[a:b], out[b:] = d[:a], s, d[a:]
        else:
            out[self.dyn_idx], out[self.static_idx] = d, s
        return out


class _Split(_Virtual):
    """Time-varying field whose static Gaussians are stored once: dyn [K,Nd,...] (array or _Packed) + static [Ns,...]."""

    def __init__(self, dyn, static: np.ndarray, layout: _Layout):
        self.dyn, self.static, self.layout = dyn, np.asarray(static, np.float32), layout
        self.tail = tuple(self.static.shape[1:])
        if tuple(dyn.shape[1:]) != (len(layout.dyn_idx),) + self.tail or len(self.static) != len(layout.static_idx):
            raise ValueError(f"split field: dynamic part {tuple(dyn.shape)} and static part {self.static.shape} "
                             f"do not fit {len(layout.dyn_idx)} + {len(layout.static_idx)} Gaussians")

    @property
    def shape(self):
        return (self.dyn.shape[0], self.layout.n) + self.tail

    def frame(self, k):
        return self.layout.merge(self.dyn[k], self.static)

    def blend(self, i0: int, i1: int, w: float):
        d = self.dyn[i0] if (w == 0.0 or i0 == i1) else (1.0 - w) * self.dyn[i0] + w * self.dyn[i1]
        return self.layout.merge(d, self.static)


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
            if not isinstance(a, _Virtual):
                a = np.asarray(a, np.float32)
            if a.shape not in (tail, (K,) + tail):
                raise ValueError(f"{name} has shape {a.shape}, expected {tail} or {(K,) + tail}")
            setattr(self, name, a)
        if self.sh is not None:
            a = self.sh if isinstance(self.sh, _Virtual) else np.asarray(self.sh, np.float32)
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

    def _kfields(self):
        """Names of the fields that have the keyframe axis."""
        names = [n for n, d in (("means", 2), ("covs", 2), ("opacities", 1), ("colors", 2)) if getattr(self, n).ndim > d]
        return names + (["sh"] if self.sh is not None and self.sh.ndim == 4 else [])

    def _layout(self) -> Optional[_Layout]:
        return next((getattr(self, n).layout for n in self._kfields() if isinstance(getattr(self, n), _Split)), None)

    @property
    def num_static(self) -> int:
        """Gaussians kept in the static group (0 when the sequence is not split)."""
        lay = self._layout()
        return 0 if lay is None else len(lay.static_idx)

    def split_static(self, rtol: float = 1e-6) -> "Sequence4D":
        """Moves the Gaussians that do not change over the keyframes, in any time-varying field, to a static
        group stored once (see the module docstring). A value counts as unchanged when it differs from the first
        keyframe by at most rtol times its magnitude; the first keyframe's value is kept. In place; returns self."""
        names = self._kfields()
        if not names or self._layout() is not None:
            return self
        N = self.num_gaussians
        static = np.ones(N, bool)
        for n in names:
            a = getattr(self, n)
            ref = np.asarray(a[0])
            tol = rtol * np.abs(ref)
            for k in range(1, self.num_frames):
                static &= (np.abs(np.asarray(a[k]) - ref) <= tol).reshape(N, -1).all(1)
            if not static.any():
                return self
        if static.all():  # nothing moves: the fields simply lose the keyframe axis
            for n in names:
                setattr(self, n, np.array(getattr(self, n)[0], np.float32))
            return self
        lay = _Layout(np.flatnonzero(static), N)
        sel = lay.dyn_sel()
        for n in names:
            a = getattr(self, n)
            setattr(self, n, _Split(a.take(sel) if isinstance(a, _Packed) else a[:, sel], np.array(a[0][lay.static_idx], np.float32), lay))
        return self

    def merge_static(self) -> "Sequence4D":
        """Undoes split_static: every time-varying field becomes one float32 array again. In place; returns self."""
        for n in self._kfields():
            if isinstance(getattr(self, n), _Split):
                setattr(self, n, np.asarray(getattr(self, n)))
        return self

    def _field(self, a: np.ndarray, static_ndim: int, i0: int, i1: int, w: float) -> np.ndarray:
        if a.ndim == static_ndim:
            return a
        if isinstance(a, _Split):
            return a.blend(i0, i1, w)
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
        for name, static_ndim in (("means", 2), ("covs", 2), ("opacities", 1), ("colors", 2)):
            a = getattr(self, name)
            d = a.dyn if isinstance(a, _Split) else a
            if isinstance(d, _Packed) or a.ndim == static_ndim:
                continue
            packed = {"means": _PackedMeans.encode, "covs": _PackedCovs.encode}.get(name, lambda x: _PackedU8.encode(x, name))(d)
            if packed is not None:
                setattr(self, name, _Split(packed, a.static, a.layout) if isinstance(a, _Split) else packed)
        return self

    @property
    def is_compact(self) -> bool:
        return any(isinstance(getattr(a, "dyn", a), _Packed) for a in (self.means, self.covs, self.opacities, self.colors))

    def save(self, path: str, compact: bool = False, split_static: bool = True) -> None:
        """Writes the .npz. compact=True stores the time-varying fields in the smaller lossy encoding
        (about 2.2x smaller); compact=False writes plain float32, also for a sequence that is compact in memory.
        split_static=True stores Gaussians that never change once instead of at every keyframe (lossless up to
        the tolerance of `split_static`); False writes every Gaussian at every keyframe, as older versions did."""
        src = Sequence4D(self.times, self.means, self.covs, self.opacities, self.colors, sh=self.sh)  # shares the arrays
        src = src.split_static() if split_static else src.merge_static()
        if compact:
            src.compact()
        files = {}
        lay = src._layout()
        if lay is not None:
            files["static_index"] = lay.static_idx
        for name in ("means", "covs", "opacities", "colors", "sh"):
            a = getattr(src, name)
            if a is None:
                continue
            if isinstance(a, _Split):
                files[name + "_static"] = a.static
                a = a.dyn
            if isinstance(a, _Packed):
                files.update(a.files() if compact else {name: np.asarray(a)})
            else:
                files[name] = a
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
        fields = {"means": means, "covs": covs, "opacities": opac, "colors": cols, "sh": z["sh"] if "sh" in z else None}
        if "static_index" in z:
            idx = np.asarray(z["static_index"])
            split = [n for n in fields if n + "_static" in z]
            lay = _Layout(idx, len(idx) + fields[split[0]].shape[1])
            for n in split:
                fields[n] = _Split(fields[n], z[n + "_static"], lay)
        return Sequence4D(
            times=np.array(z["times"]),
            interpolate=bool(z["interpolate"]) if "interpolate" in z else True,
            source=str(z["source"]) if "source" in z else "",
            **fields,
        )

    def summary(self) -> str:
        dyn = [n for n, a, d in (("means", self.means, 2), ("covs", self.covs, 2),
                                 ("opacities", self.opacities, 1), ("colors", self.colors, 2)) if a.ndim > d]
        return (f"{self.num_gaussians:,} Gaussians x {self.num_frames} keyframes, "
                f"t in [{self.times[0]:.3f}, {self.times[-1]:.3f}], "
                f"time-varying: {', '.join(dyn) or 'none'}"
                + ("" if self.sh is None else f", SH degree {self.sh_degree}"
                   + (" (per keyframe)" if self.sh.ndim == 4 else ""))
                + (f", {self.num_static:,} static Gaussians stored once" if self.num_static else "")
                + (", compact storage" if self.is_compact else ""))


class GpuFrames:
    """Decodes and interpolates a sequence on the GPU with torch (for the gsplat renderer; `Sequence4D.at` is the CPU path).

    Only the keyframes around the current time are held on the GPU, decoded to float32 (a small least-recently-used
    cache): the raw arrays (float16 / int16 / uint8 in the compact layout) are copied to the GPU one keyframe at a
    time, the first time the slider reaches a new keyframe interval. Fields without a K axis and the static group
    are uploaded once, in `__init__`; their rows stay in the output buffers, so a frame only writes the dynamic rows.

    `at(t)` returns torch tensors (means [N,3], packed covs [N,6], opacities [N], colours [N,3] or SH [N,S,3]) in the
    original Gaussian order, as `Sequence4D.at` does. They are buffers reused by the next `at` call: use or copy them first.
    """

    def __init__(self, seq: "Sequence4D", device=None, cache: int = 3):
        import torch

        self.torch, self.seq, self.cache_size = torch, seq, max(int(cache), 2)
        self.device = torch.device("cuda") if device is None else torch.device(device)
        self.n = seq.num_gaussians
        self.fields: dict = {}   # name -> dict(kind, ...)
        self._cache: dict = {}   # (name, k) -> float32 tensor of the dynamic rows, least recently used first
        self._stage: dict = {}   # pinned staging buffers
        self.use_sh = seq.sh is not None
        for name, nd in (("means", 2), ("covs", 2), ("opacities", 1), ("colors", 2), ("sh", 3)):
            a = getattr(seq, name)
            if a is None or (name == "colors" and self.use_sh):  # the gsplat renderer ignores colours when SH exist
                continue
            if a.ndim == nd:
                self.fields[name] = {"kind": "const", "out": self._put(np.asarray(a, np.float32))}
                continue
            f = {"kind": "dyn", "src": a, "out": None, "run": None}
            if isinstance(a, _Split):
                lay = a.layout
                f.update(src=a.dyn, run=lay.run,
                         dyn_sel=torch.from_numpy(lay.dyn_idx).to(self.device),
                         static_sel=torch.from_numpy(lay.static_idx).to(self.device))
                f["out"] = torch.empty((self.n,) + a.static.shape[1:], dtype=torch.float32, device=self.device)
                static = self._put(a.static)
                if lay.run:
                    f["out"][lay.run[0]:lay.run[1]] = static
                else:
                    f["out"][f["static_sel"]] = static
            src = f["src"]
            if isinstance(src, _PackedMeans):
                f["ref"] = self._put(src.ref)
            elif isinstance(src, _PackedCovs):
                f["scale"] = self._put(src.scale)[:, None] / 32767.0
            self.fields[name] = f

    # ------------------------------------------------------------------ helpers
    def _put(self, a: np.ndarray):
        """Host array -> GPU tensor through a pinned staging buffer (one read of a memory-mapped array, one copy)."""
        torch = self.torch
        a = np.asarray(a)
        key = (a.shape, a.dtype.str)
        buf = self._stage.get(key)
        if buf is None:
            buf = self._stage[key] = torch.empty(a.shape, dtype=getattr(torch, a.dtype.name), pin_memory=True)
        buf.numpy()[...] = a
        return buf.to(self.device)  # synchronous, so the staging buffer can be reused at once

    def _decode(self, f: dict, k: int):
        """Dynamic rows of keyframe k as float32 on the GPU (same arithmetic as `_Packed*.frame`)."""
        torch, src = self.torch, f["src"]
        if isinstance(src, _PackedMeans):
            return f["ref"] + self._put(src.raw[k]).float()
        if isinstance(src, _PackedCovs):
            L = self._put(src.raw[k]).float() * f["scale"]
            l00, l10, l20, l11, l21, l22 = L.unbind(1)
            return torch.stack([l00 * l00, l00 * l10, l00 * l20, l10 * l10 + l11 * l11,
                                l10 * l20 + l11 * l21, l20 * l20 + l21 * l21 + l22 * l22], 1)
        if isinstance(src, _PackedU8):
            return self._put(src.raw[k]).float() * (1.0 / 255.0)
        return self._put(np.asarray(src[k], np.float32))

    def _frame(self, name: str, k: int):
        key = (name, k)
        v = self._cache.pop(key, None)
        if v is None:
            v = self._decode(self.fields[name], k)
        self._cache[key] = v  # most recently used goes last
        while len(self._cache) > self.cache_size * len(self.fields):
            self._cache.pop(next(iter(self._cache)))
        return v

    # ------------------------------------------------------------------ public
    def at(self, t: float):
        """(means [N,3], covs [N,6], opacities [N], colours [N,3] or SH [N,S,3]) at time t, torch float32 on the GPU."""
        i0, i1, w = self.seq._bracket(t)
        out = []
        for name in ("means", "covs", "opacities", "sh" if self.use_sh else "colors"):
            f = self.fields[name]
            if f["kind"] == "const":
                out.append(f["out"])
                continue
            d = self._frame(name, i0)
            if not (w == 0.0 or i0 == i1):
                d = self.torch.lerp(d, self._frame(name, i1), w)
            if f["out"] is None:
                out.append(d)
            elif f["run"]:
                a, b = f["run"]
                f["out"][:a], f["out"][b:] = d[:a], d[a:]
                out.append(f["out"])
            else:
                f["out"][f["dyn_sel"]] = d
                out.append(f["out"])
        return tuple(out)


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Re-save a sequence in the compact layout (default) or as plain float32.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--float32", action="store_true", help="write plain float32 instead of the compact layout")
    ap.add_argument("--no-split", action="store_true",
                    help="write every Gaussian at every keyframe instead of storing the static ones once")
    a = ap.parse_args()
    if os.path.exists(a.output) and os.path.samefile(a.input, a.output):
        raise SystemExit("input and output must be different files")
    seq = Sequence4D.load(a.input, lazy=True)
    seq.save(a.output, compact=not a.float32, split_static=not a.no_split)
    print(Sequence4D.load(a.output, lazy=True).summary())
    print(f"Wrote {a.output}: {os.path.getsize(a.input) / 1e6:.1f} MB -> {os.path.getsize(a.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
