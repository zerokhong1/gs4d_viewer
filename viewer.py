#!/usr/bin/env python3
"""4D Gaussian Splatting viewer with a time slider (viser front end).

    python viewer.py toy.npz
    python viewer.py scene.npz --renderer gsplat --up -y --port 8080
    python viewer.py ours.npz --compare baseline.npz      # two sequences, one time slider

Renderers:
    gsplat  images are rasterized on this machine's NVIDIA GPU with gsplat and streamed to the
            browser. Exact 3DGS alpha blending; handles millions of Gaussians.
    client  Gaussians are sent to the browser and drawn there with WebGL (viser's splat renderer).
            No CUDA needed; meant for small scenes or a laptop.
    auto    gsplat when CUDA and gsplat are available, otherwise client.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import viser

from gs4d import GpuFrames, Sequence4D, unpack_cov


def pick_renderer(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import gsplat  # noqa: F401
        import torch

        if torch.cuda.is_available():
            return "gsplat"
    except Exception:
        pass
    return "client"


def camera_matrices(wxyz, position, fov_y: float, width: int, height: int):
    """viser camera -> (world-to-camera 4x4, intrinsics 3x3), both float32.

    viser cameras follow the OpenCV convention (+z forward, +x right, +y down); fov_y is the
    vertical field of view in radians. Checked against the browser renderer in tests/check_camera.py.
    """
    w, x, y, z = (float(v) for v in wxyz)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    c2w[:3, 3] = position
    f = 0.5 * height / np.tan(0.5 * fov_y)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float32)
    return np.linalg.inv(c2w).astype(np.float32), K


def frame_cpu(seq: Sequence4D, t: float, min_opacity: float, scale: float, device, shade=None):
    """gsplat inputs at time t, decoded and interpolated with numpy, then uploaded: (means, full covs [M,3,3], opacities,
    colours or SH [M,S,3]) for the Gaussians whose opacity is at least min_opacity. The reference path.
    shade(means [N,3]) -> RGB [N,3], when given, replaces the colours (and the SH): "Colour by" speed / displacement."""
    import torch

    means, covs, opac, cols = seq.at(t)
    sh = seq.sh_at(t)
    opac = np.where(opac >= min_opacity, opac, 0.0).astype(np.float32)
    covs = (unpack_cov(covs) * np.float32(scale ** 2)).astype(np.float32)
    keep = opac > 0
    cols = shade(means) if shade is not None else cols if sh is None else sh
    return tuple(torch.from_numpy(np.ascontiguousarray(a[keep], np.float32)).to(device)
                 for a in (means, covs, opac, cols))


def frame_gpu(gpu: GpuFrames, t: float, min_opacity: float, scale: float, shade=None):
    """Same result as frame_cpu, with decoding, interpolation, unpacking and thresholding done on the GPU (shade gets torch tensors)."""
    torch = gpu.torch
    means, covs, opac, cols = gpu.at(t)
    if shade is not None:
        cols = shade(means)
    opac = torch.where(opac >= min_opacity, opac, torch.zeros_like(opac))
    keep = opac > 0
    c = covs[keep] * (scale ** 2)
    full = torch.empty((len(c), 3, 3), dtype=torch.float32, device=c.device)
    full[:, 0, 0], full[:, 0, 1], full[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
    full[:, 1, 0], full[:, 1, 1], full[:, 1, 2] = c[:, 1], c[:, 3], c[:, 4]
    full[:, 2, 0], full[:, 2, 1], full[:, 2, 2] = c[:, 2], c[:, 4], c[:, 5]
    return means[keep], full, opac[keep], cols[keep]


def load_cameras(path: str) -> list[dict]:
    """Cameras of a transforms_*.json (NeRF / D-NeRF / instant-ngp) or a cameras.json.

    transforms_*.json: `camera_angle_x` (or `fl_x` + `w`), frames with `transform_matrix` (OpenGL c2w), optional `time`
    and `file_path` (image, relative to the json, extension optional). cameras.json: `{"convention": "opencv"|"opengl",
    "frames": [...]}` (or a bare list), each frame with `c2w`/`transform_matrix` [4x4] and `fov_x`, `fov_y` or `fx`
    (+ `w`, `h`), optional `name`, `time`, `image`. Default convention: OpenGL when the file has `transform_matrix`,
    else OpenCV. Returns dicts with c2w (OpenCV, 4x4), fx (pixels at width w), w, h (0 when unknown), time (None if absent),
    image (path or None), name.
    """
    with open(path) as f:
        meta = json.load(f)
    top = meta if isinstance(meta, dict) else {}
    frames = meta if isinstance(meta, list) else (meta.get("frames") or meta.get("cameras") or [])
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for k, fr in enumerate(frames):
        raw = fr.get("transform_matrix", fr.get("c2w"))
        if raw is None:
            raise ValueError(f"{path}: frame {k} has no transform_matrix / c2w")
        c2w = np.array(raw, np.float64).reshape(4, 4)
        conv = str(fr.get("convention", top.get("convention", "opengl" if "transform_matrix" in fr else "opencv"))).lower()
        if conv == "opengl":
            c2w[:3, 1:3] *= -1  # OpenGL (+y up, -z forward) -> OpenCV (+y down, +z forward)
        img = fr.get("image", fr.get("file_path"))
        if img:
            img = os.path.join(base, img)
            if not os.path.splitext(img)[1]:
                img += ".png"
            if not os.path.exists(img):
                img = None
        w = int(fr.get("w", fr.get("width", top.get("w", top.get("width", 0)))))
        h = int(fr.get("h", fr.get("height", top.get("h", top.get("height", 0)))))
        if (not w or not h) and img:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.size
        w, h = (w, h) if w and h else (0, 0)
        get = lambda key: fr.get(key, top.get(key))  # noqa: E731
        if get("fx") or get("fl_x"):
            fx = float(get("fx") or get("fl_x"))
        elif get("fov_x") or get("camera_angle_x"):
            fx = 0.5 * (w or 1) / np.tan(0.5 * float(get("fov_x") or get("camera_angle_x")))
        elif get("fov_y") and w and h:
            fx = 0.5 * h / np.tan(0.5 * float(get("fov_y")))
        else:
            raise ValueError(f"{path}: frame {k} has no focal length (fx, fl_x, fov_x, fov_y or camera_angle_x)")
        t = fr.get("time")
        name = str(fr.get("name") or (os.path.splitext(os.path.basename(fr["file_path"]))[0] if fr.get("file_path")
                                       else os.path.splitext(os.path.basename(img))[0] if img else k))
        out.append(dict(c2w=c2w, fx=fx, w=w, h=h, time=None if t is None else float(t), image=img, name=name))
    if not out:
        raise ValueError(f"{path}: no cameras found")
    return out


def load_gt_image(path: str, W: int, H: int, white: bool) -> np.ndarray:
    """Ground-truth image as float32 [H,W,3] in 0..1; RGBA is composited on white or black, and the image is
    resized with Lanczos to W x H (same recipe as tests/check_real.py)."""
    from PIL import Image
    pil = Image.open(path).convert("RGBA")
    if pil.size != (W, H):
        pil = pil.resize((W, H), Image.LANCZOS)
    im = np.asarray(pil).astype(np.float32) / 255.0
    return im[..., :3] * im[..., 3:4] + ((1.0 - im[..., 3:4]) if white else 0.0)


# ---------------------------------------------------------------- camera path (WP34)
PATH_HOLD, PATH_RUN = "stay at the current time", "run through the sequence"
VIDEO_FIXED, VIDEO_PATH = "this camera (fixed)", "camera path"


def slerp(q0, q1, f: float) -> np.ndarray:
    """Spherical interpolation of two unit quaternions (wxyz), along the short arc."""
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:  # nearly the same rotation: the straight line is accurate and avoids dividing by sin(0)
        q = q0 + f * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def catmull_rom(points, i: int, f: float, closed: bool) -> np.ndarray:
    """Point at fraction f of segment i (from points[i] to points[i+1]) of the uniform Catmull-Rom spline through `points`.
    An open spline repeats its end points, so the tangent at the ends points at the neighbour."""
    n = len(points)
    at = (lambda k: points[k % n]) if closed else (lambda k: points[min(max(k, 0), n - 1)])
    p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
    return 0.5 * (2 * p1 + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def path_pose(keys: list[dict], u: float, closed: bool = False):
    """Camera on a path at u in [0, 1]: (position [3], wxyz [4], fov_y). Keys are dicts with position, wxyz, fov and are spaced
    evenly in u: position on a Catmull-Rom spline through the key positions, orientation slerped between the neighbouring keys,
    fov linear. A closed path returns to its first key at u = 1."""
    n = len(keys)
    if n == 1:
        k = keys[0]
        return np.array(k["position"], np.float64), np.array(k["wxyz"], np.float64), float(k["fov"])
    nseg = n if closed else n - 1
    s = min(max(float(u), 0.0), 1.0) * nseg
    i = min(int(np.floor(s)), nseg - 1)
    f = s - i
    pos = catmull_rom(np.array([k["position"] for k in keys], np.float64), i, f, closed)
    a, b = keys[i], keys[(i + 1) % n]
    return pos, slerp(a["wxyz"], b["wxyz"], f), float((1 - f) * a["fov"] + f * b["fov"])


def look_at_wxyz(position, target, up) -> np.ndarray:
    """Orientation (viser / OpenCV camera: +z forward, +x right, +y down) at `position` looking at `target`, world `up` is up."""
    from viser.transforms import SO3
    z = np.asarray(target, np.float64) - np.asarray(position, np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.asarray(SO3.from_matrix(np.stack([x, y, z], 1)).wxyz, np.float64)


def orbit_keys(centre, distance: float, view_dir, up_vec, fov: float, n: int = 12) -> list[dict]:
    """n keys on a circle around `centre` (closed path: one turn), at the elevation of `view_dir` (centre -> first camera),
    all looking at the centre."""
    up = np.asarray(up_vec, np.float64)
    d = np.asarray(view_dir, np.float64)
    d /= np.linalg.norm(d)
    vert = float(d @ up)
    horiz = d - vert * up
    horiz /= np.linalg.norm(horiz)
    side = np.cross(up, horiz)
    keys = []
    for k in range(n):
        a = 2 * np.pi * k / n
        dirk = np.cos(a) * horiz + np.sin(a) * side
        pos = np.asarray(centre, np.float64) + distance * (np.sqrt(max(1 - vert ** 2, 0.0)) * dirk + vert * up)
        keys.append(dict(position=pos, wxyz=look_at_wxyz(pos, centre, up), fov=float(fov)))
    return keys


def load_path(path: str) -> dict:
    """A camera path file: {"convention": "opencv"|"opengl", "duration": s, "closed": bool, "time": "hold"|"run",
    "frames": [{"c2w" or "transform_matrix": 4x4, "fov_y": radians}]} (the `--cameras` convention). Returns
    dict(keys, duration, closed, run) with keys in the viewer's convention (OpenCV)."""
    from viser.transforms import SO3
    with open(path) as f:
        meta = json.load(f)
    frames = meta if isinstance(meta, list) else meta.get("frames") or []
    top = meta if isinstance(meta, dict) else {}
    keys = []
    for k, fr in enumerate(frames):
        raw = fr.get("c2w", fr.get("transform_matrix"))
        if raw is None or fr.get("fov_y", top.get("fov_y")) is None:
            raise ValueError(f"{path}: key {k} needs c2w (or transform_matrix) and fov_y")
        c2w = np.array(raw, np.float64).reshape(4, 4)
        if str(fr.get("convention", top.get("convention", "opencv"))).lower() == "opengl":
            c2w[:3, 1:3] *= -1
        keys.append(dict(position=c2w[:3, 3].copy(), wxyz=np.asarray(SO3.from_matrix(c2w[:3, :3]).wxyz, np.float64),
                         fov=float(fr.get("fov_y", top.get("fov_y")))))
    if not keys:
        raise ValueError(f"{path}: no keys found")
    return dict(keys=keys, duration=float(top.get("duration", 10.0)), closed=bool(top.get("closed", False)),
                run=str(top.get("time", "hold")).lower() == "run")


def save_path(path: str, keys: list[dict], duration: float, closed: bool, run: bool):
    from viser.transforms import SO3
    frames = []
    for k in keys:
        c2w = np.eye(4)
        c2w[:3, :3] = SO3(np.asarray(k["wxyz"], np.float64)).as_matrix()
        c2w[:3, 3] = k["position"]
        frames.append({"c2w": c2w.tolist(), "fov_y": float(k["fov"])})
    with open(path, "w") as f:
        json.dump({"convention": "opencv", "duration": float(duration), "closed": bool(closed),
                   "time": "run" if run else "hold", "frames": frames}, f, indent=1)


# ---------------------------------------------------------------- motion: colour by speed / displacement, trails (WP35)
COLOUR, SPEED, DISPLACEMENT = "colour", "speed", "displacement"
_TURBO = [(0.0, (48, 18, 59)), (0.15, (70, 100, 230)), (0.3, (30, 190, 235)), (0.45, (40, 230, 150)), (0.6, (160, 250, 60)),
          (0.75, (250, 200, 40)), (0.9, (240, 90, 20)), (1.0, (130, 5, 5))]


def colour_lut(n: int = 256) -> np.ndarray:
    """Blue (0) -> green -> yellow -> red (1) colour scale, float32 [n,3] in 0..1."""
    x = np.linspace(0.0, 1.0, n)
    return (np.stack([np.interp(x, [a for a, _ in _TURBO], [c[j] for _, c in _TURBO]) for j in range(3)], 1) / 255.0).astype(np.float32)


class TrailData:
    """Paths of the Gaussians shown as trails: `idx` ranks up to max_m Gaussians, most worth drawing first, and `traj` [K,M,3] holds
    their centres at every keyframe.

    A Gaussian is a candidate when its opacity reaches min_peak at some keyframe and it moves; it is ranked by a weighted random
    draw (fixed seed, weight = path length over up to `sample` keyframes), so a longer path is likelier to come first, and asking
    for fewer trails keeps the first ones. Two passes over the keyframes: path length and peak opacity on a sample of them,
    then the centres of the chosen Gaussians at every keyframe."""

    def __init__(self, seq: Sequence4D, max_m: int = 5000, seed: int = 0, min_peak: float = 0.3, sample: int = 48):
        K, N = seq.num_frames, seq.num_gaussians
        ks = np.unique(np.linspace(0, K - 1, min(K, sample)).round().astype(int))
        length, peak, prev = np.zeros(N), np.zeros(N), None
        for k in ks:
            m = seq.keyframe_means(k).astype(np.float64)
            if prev is not None:
                length += np.linalg.norm(m - prev, axis=1)
            prev = m
            o = seq.opacities if seq.opacities.ndim == 1 else seq.opacities[int(k)]
            peak = np.maximum(peak, np.asarray(o, np.float64))
        cand = np.flatnonzero((peak >= min_peak) & (length > 0))
        if len(cand) == 0:  # nothing moves: any opaque Gaussian
            cand = np.flatnonzero(peak >= min_peak)
            w = np.ones(len(cand))
        else:
            w = length[cand] / length[cand].max()
        rng = np.random.default_rng(seed)
        order = np.argsort(-(rng.random(len(cand)) ** (1.0 / np.maximum(w, 1e-12))))  # weighted sampling without replacement
        self.idx = cand[order[:max_m]]
        self.times = np.asarray(seq.times, np.float64)
        self.traj = np.empty((K, len(self.idx), 3), np.float32)
        for k in range(K):
            self.traj[k] = seq.keyframe_means(k)[self.idx]

    def at(self, t: float, delta: float, m: int, samples: int = 24):
        """Positions [samples, m, 3] of the first m trails at `samples` times spread over [t - delta, t + delta] (clipped to the
        sequence), by linear interpolation between keyframes, and the fraction 0..1 of each sample in the window."""
        ts = np.clip(t + np.linspace(-delta, delta, samples), self.times[0], self.times[-1])
        i1 = np.clip(np.searchsorted(self.times, ts, side="left"), 0, len(self.times) - 1)
        i0 = np.where(self.times[i1] == ts, i1, np.maximum(i1 - 1, 0))
        w = np.where(i0 == i1, 0.0, (ts - self.times[i0]) / np.maximum(self.times[i1] - self.times[i0], 1e-12))
        m = min(int(m), len(self.idx))
        pos = (1 - w)[:, None, None] * self.traj[i0, :m] + w[:, None, None] * self.traj[i1, :m]
        return pos.astype(np.float32), np.linspace(0.0, 1.0, samples)


SIDE_BY_SIDE = "A | B"
SPLATS, POINTS = "splats", "points"  # "Draw as": the Gaussians themselves, or one dot at each Gaussian's centre
# initial camera direction (scene centre -> camera) for each world up axis
VIEW_DIR = {"+z": (1.0, -1.0, 0.6), "-z": (1.0, -1.0, -0.6), "+y": (1.0, 0.6, 1.0),
            "-y": (1.0, -0.6, 1.0), "+x": (0.6, 1.0, -1.0), "-x": (-0.6, 1.0, -1.0)}


def _peak_opacity(seq: Sequence4D) -> np.ndarray:
    """Largest opacity of every Gaussian over the keyframes [N], decoding one keyframe at a time (a compact or lazily
    loaded file never has its whole opacity field in memory)."""
    o = seq.opacities
    if o.ndim == 1:
        return np.asarray(o, np.float32)
    peak = np.array(o[0], np.float32)
    for k in range(1, len(o)):
        np.maximum(peak, o[k], out=peak)
    return peak


def _bounds(seq: Sequence4D):
    """Median centre and 90th-percentile radius of the first keyframe."""
    m0 = np.asarray(seq.means[0] if seq.means.ndim == 3 else seq.means)
    centre = np.median(m0, 0)
    return centre, float(np.percentile(np.linalg.norm(m0 - centre, axis=1), 90)) or 1.0


class Viewer:
    """seq_b is an optional second sequence to compare with seq: shown alone, or next to it, on the same
    time slider. Side by side means a split screen with the same camera in the gsplat renderer, and the
    second sequence moved sideways in the scene in the client renderer."""

    def __init__(self, seq: Sequence4D, args, seq_b: Sequence4D | None = None):
        self.seq, self.args = seq, args
        self.seqs = [seq] if seq_b is None else [seq, seq_b]
        files = [getattr(args, "file", None), getattr(args, "compare", None)]
        self.names = [f"{'AB'[i]}: {os.path.basename(f)}" if f else "AB"[i] for i, f in enumerate(files[:len(self.seqs)])]
        self.renderer = pick_renderer(args.renderer)
        self.t0 = float(min(s.times[0] for s in self.seqs))
        self.t1 = float(max(s.times[-1] for s in self.seqs))
        self.duration = max(self.t1 - self.t0, 1e-9)
        self.dirty = True
        self.cam_stamp: dict[int, float] = {}
        self.cameras = load_cameras(args.cameras) if getattr(args, "cameras", None) else []
        self.cam_idx = -1  # selected camera (-1: none)
        self.gt_cache: dict = {}
        self.gt_key = None
        self.gt_last = None

        # fixed subset for the browser renderer, so the splat count never changes between frames
        self.subsets = [None] * len(self.seqs)
        for i, s in enumerate(self.seqs):
            if self.renderer == "client" and s.num_gaussians > args.max_splats:
                self.subsets[i] = np.sort(np.argpartition(_peak_opacity(s), -args.max_splats)[-args.max_splats:])
                print(f"client renderer: showing the {args.max_splats:,} most opaque of {s.num_gaussians:,} Gaussians "
                      "(raise --max-splats or use --renderer gsplat)")

        self.centre, self.radius = _bounds(seq)
        self.motion_ok = all(q.interpolate for q in self.seqs)  # speed, displacement and trails need identities shared by keyframes
        self.lut = colour_lut()
        self.vmax_cache: dict = {}
        self.speed_np: dict = {}
        self.speed_gpu: dict = {}
        self.ref0_gpu: dict = {}
        self.trail_data: dict = {}
        self.trail_h: dict = {}
        self.trail_n: dict = {}
        self.path_keys: list[dict] = []  # camera path (WP34)
        self.path_u, self.path_playing = 0.0, False
        # client renderer, side by side: B is drawn this far to the right of A (as seen from the initial camera)
        up = np.zeros(3)
        up["xyz".index(args.up[1])] = 1.0 if args.up[0] == "+" else -1.0
        right = np.cross(-np.asarray(VIEW_DIR[args.up]), up)
        self.offset = right / np.linalg.norm(right) * 2.4 * max(_bounds(s)[1] for s in self.seqs)
        self.up_vec = up

        self.server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
        self.server.scene.set_up_direction(args.up)
        self._build_gui()
        self.server.on_client_connect(self._on_connect)
        if getattr(args, "camera_path", None):
            self._set_path(load_path(args.camera_path))
        self.splats = [None] * len(self.seqs)
        self.clouds = [None] * len(self.seqs)  # client renderer, "Draw as: points": viser point clouds
        self.cloud_n = [0] * len(self.seqs)
        self.labels = None
        self.gpu: list = [None] * len(self.seqs)  # GpuFrames per sequence, built on first use (gsplat renderer)
        self.cur: dict[int, tuple] = {}  # sequence index -> tensors of the current time step (gsplat renderer)
        if self.renderer == "gsplat":
            import torch

            self.torch = torch
            self.device = torch.device("cuda")

    # ---------------------------------------------------------------- GUI
    def _build_gui(self):
        g, s = self.server.gui, self.seq
        if len(self.seqs) == 1:
            g.add_markdown(f"**{s.num_gaussians:,}** Gaussians · **{s.num_frames}** keyframes · renderer **{self.renderer}**"
                           + (f"\n\n{s.source}" if s.source else ""))
            self.mode = None
        else:
            g.add_markdown(f"renderer **{self.renderer}**\n\n" + "\n\n".join(
                f"**{n}** · {q.num_gaussians:,} Gaussians · {q.num_frames} keyframes" for n, q in zip(self.names, self.seqs)))
            self.mode = g.add_dropdown("Show", options=("A", "B", SIDE_BY_SIDE), initial_value=SIDE_BY_SIDE)
            self.mode.on_update(lambda _e: setattr(self, "dirty", True))
        self.time = g.add_slider("Time", min=self.t0, max=self.t1, step=self.duration / 1000.0, initial_value=self.t0)
        self.frame = g.add_number("Keyframe", initial_value=0, min=0, max=s.num_frames - 1, step=1)
        self.play = g.add_button("Play / Pause")
        self.prev = g.add_button("◀ previous keyframe")
        self.next = g.add_button("next keyframe ▶")
        self.speed = g.add_slider("Speed (x)", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        self.loop = g.add_checkbox("Loop", initial_value=True)
        if self.cameras:
            self._build_camera_gui()
        self._build_path_gui()
        self._build_motion_gui()
        with g.add_folder("Display", expand_by_default=False):
            self.min_opacity = g.add_slider("Hide opacity below", min=0.0, max=0.5, step=0.005, initial_value=0.02)
            self.scale = g.add_slider("Gaussian size (x)", min=0.1, max=2.0, step=0.05, initial_value=1.0)
            self.draw_as = g.add_dropdown("Draw as", options=(SPLATS, POINTS), initial_value=getattr(self.args, "draw_as", SPLATS))
            # gsplat: diameter in screen pixels at any depth; client: viser sizes points in scene units, so the
            # slider is in thousandths of the scene radius (points shrink with distance)
            self.point_size = g.add_slider("Point size (px)" if self.renderer == "gsplat" else "Point size (radius / 1000)",
                                           min=1.0, max=20.0, step=0.5, initial_value=3.0 if self.renderer == "gsplat" else 6.0)
            self.target_fps = g.add_slider("Update rate (Hz)", min=2, max=60, step=1, initial_value=20)
            if self.renderer == "gsplat":
                self.max_res = g.add_slider("Max render height", min=240, max=2160, step=60,
                                            initial_value=self.args.max_res)
        with g.add_folder("Export video", expand_by_default=False):
            self.video_fps = g.add_number("Frames per second", initial_value=30, min=1, max=120, step=1)
            self.video_seconds = g.add_number("Video length (s)", initial_value=float(np.clip(round(self.duration, 1), 2.0, 20.0)),
                                              min=0.5, max=600.0, step=0.5)
            self.video_cam = g.add_dropdown("Camera", options=(VIDEO_FIXED, VIDEO_PATH), initial_value=VIDEO_FIXED)
            g.add_markdown("_Fixed camera: the whole time range in 'Video length'. Camera path: 'Path duration' decides the length._")
            self.video_btn = g.add_button("Export mp4")
        self.export_client = None  # set by the button, handled in the main loop
        self.playing = False
        self._setting = False

        @self.video_btn.on_click
        def _(e):
            if self.export_client is None:
                self.export_client = e.client

        @self.time.on_update
        def _(_e):
            self.dirty = True
            if not self._setting:
                self._setting = True
                self.frame.value = int(np.argmin(np.abs(s.times - self.time.value)))
                self._setting = False

        @self.frame.on_update
        def _(_e):
            if not self._setting:
                self.playing = False
                self._set_time(float(s.times[int(self.frame.value)]))

        @self.play.on_click
        def _(_e):
            if not self.playing and self.time.value >= self.t1 - 1e-6:
                self._set_time(self.t0)
            self.playing = not self.playing

        @self.prev.on_click
        def _(_e):
            self.playing = False
            i = int(np.searchsorted(s.times, self.time.value - 1e-6, side="left")) - 1
            self._set_time(float(s.times[max(i, 0)]))

        @self.next.on_click
        def _(_e):
            self.playing = False
            i = int(np.searchsorted(s.times, self.time.value + 1e-6, side="right"))
            self._set_time(float(s.times[min(i, s.num_frames - 1)]))

        for h in (self.min_opacity, self.scale, self.draw_as, self.point_size):
            h.on_update(lambda _e: setattr(self, "dirty", True))

    # ---------------------------------------------------------------- motion (WP35)
    def _build_motion_gui(self):
        g = self.server.gui
        with g.add_folder("Motion", expand_by_default=False):
            if not self.motion_ok:
                g.add_markdown("_Colour by speed / displacement and trails need keyframes that share Gaussian identities "
                               "(`interpolate=True`); this file has per-keyframe Gaussians._")
                self.colour_by = self.trails_on = None
                return
            self.colour_by = g.add_dropdown("Colour by", options=(COLOUR, SPEED, DISPLACEMENT), initial_value=COLOUR)
            self.scale_pct = g.add_slider("Scale maximum (%)", min=10, max=200, step=5, initial_value=100)
            g.add_image((np.tile(self.lut[None], (12, 1, 1)) * 255).astype(np.uint8), label="Scale")
            self.scale_info = g.add_markdown("")
            self.trails_on = g.add_checkbox("Trails", initial_value=False)
            self.trail_m = g.add_slider("Number of trails", min=10, max=5000, step=10, initial_value=500)
            self.trail_dt = g.add_slider("Window ± (time units)", min=self.duration / 200, max=self.duration / 2, step=self.duration / 200,
                                         initial_value=self.duration / 10)
            self.trail_info = g.add_markdown("_Trails follow the most moving, opaque Gaussians over the window, blue = past, red = future. "
                                             "They are lines drawn on top of the picture: no hiding by depth with the gsplat renderer, "
                                             "and with two files side by side there only for the client renderer._")
        for h in (self.colour_by, self.scale_pct, self.trails_on, self.trail_m, self.trail_dt):
            h.on_update(lambda _e: setattr(self, "dirty", True))
        self.colour_by.on_update(lambda _e: self._update_scale_info())
        self.scale_pct.on_update(lambda _e: self._update_scale_info())
        self._update_scale_info()

    def _colour_on(self) -> bool:
        return self.motion_ok and self.colour_by is not None and self.colour_by.value != COLOUR

    def _update_scale_info(self):
        if self._colour_on():
            kind = self.colour_by.value
            self.scale_info.content = (f"0 = blue, {self._vmax_eff():.3g} = red · {'scene units per time unit' if kind == SPEED else 'scene units'}"
                                       " (99th percentile over the sequence × the percentage above)")
        else:
            self.scale_info.content = "_Gaussians keep their own colour._"

    def _vmax(self, kind: str) -> float:
        """99th percentile of the positive values of speed / displacement over the whole sequence (pooled over all shown files)."""
        if kind not in self.vmax_cache:
            vals = []
            for q in self.seqs:
                for t in np.linspace(q.times[0], q.times[-1], min(40, 2 * q.num_frames)):
                    v = q.speed_at(t) if kind == SPEED else q.displacement_at(t)
                    v = v[v > 0]
                    vals.append(v[:: max(len(v) // 20000, 1)])
            v = np.concatenate(vals) if vals else np.zeros(0)
            self.vmax_cache[kind] = float(np.percentile(v, 99)) if len(v) else 1.0
            self.vmax_cache[kind] = self.vmax_cache[kind] or 1.0
        return self.vmax_cache[kind]

    def _vmax_eff(self) -> float:
        return self._vmax(self.colour_by.value) * self.scale_pct.value / 100.0

    def _shade_np(self, i: int, t: float, means: np.ndarray) -> np.ndarray:
        """RGB [N,3] of the Gaussians of sequence i at time t for the chosen "Colour by"."""
        q, kind = self.seqs[i], self.colour_by.value
        if kind == SPEED:
            seg = q.speed_segment(t)
            if self.speed_np.get(i, (None,))[0] != seg:
                self.speed_np[i] = (seg, q.segment_speed(seg))
            v = self.speed_np[i][1]
        else:
            v = q.displacement_at(t, means)
        return self.lut[(np.clip(v / self._vmax_eff(), 0.0, 1.0) * (len(self.lut) - 1) + 0.5).astype(np.int64)]

    def _shade_gpu(self, i: int, t: float, means):
        """Same on the GPU: `means` is the torch tensor of the Gaussian centres at time t."""
        torch, q, kind = self.torch, self.seqs[i], self.colour_by.value
        if kind == SPEED:
            seg = q.speed_segment(t)
            if self.speed_gpu.get(i, (None,))[0] != seg:  # piecewise constant: one upload per keyframe interval
                self.speed_gpu[i] = (seg, torch.from_numpy(q.segment_speed(seg)).to(self.device))
            v = self.speed_gpu[i][1]
        else:
            if i not in self.ref0_gpu:
                self.ref0_gpu[i] = torch.from_numpy(q.keyframe_means(0)).to(self.device)
            v = (means - self.ref0_gpu[i]).norm(dim=1)
        if not hasattr(self, "lut_t"):
            self.lut_t = torch.from_numpy(self.lut).to(self.device)
        n = len(self.lut) - 1
        return self.lut_t[((v / self._vmax_eff()).clamp(0.0, 1.0) * n + 0.5).long()]

    def _draw_trails(self):
        """Lines along the paths of the most moving Gaussians in the window around the current time (both renderers)."""
        if not self.motion_ok or self.trails_on is None:
            return
        shown = self._shown()
        for i in range(len(self.seqs)):
            want = self.trails_on.value and i in shown and (self.renderer == "client" or len(shown) == 1)
            h = self.trail_h.get(i)
            if not want:
                if h is not None:
                    h.visible = False
                continue
            if i not in self.trail_data:
                self.trail_info.content = "_Preparing trails (reading every keyframe once) ..._"
                self.trail_data[i] = TrailData(self.seqs[i])
                self.trail_info.content = self._trail_note()
            td = self.trail_data[i]
            m = min(int(self.trail_m.value), len(td.idx))
            pos, frac = td.at(self.time.value, float(self.trail_dt.value), m)
            segs = np.stack([pos[:-1], pos[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 3)  # [m*(S-1), 2, 3]
            col = np.tile((self.lut[(np.stack([frac[:-1], frac[1:]], 1) * (len(self.lut) - 1)).round().astype(int)] * 255).astype(np.uint8)[None],
                          (m, 1, 1, 1)).reshape(-1, 2, 3)
            if h is None or self.trail_n.get(i) != m:
                if h is not None:
                    h.remove()
                h = self.trail_h[i] = self.server.scene.add_line_segments(
                    f"/trails/{'ab'[i]}", points=segs, colors=col, thickness=1.5, thickness_units="screen")
                self.trail_n[i] = m
            else:
                h.points, h.visible = segs, True
            h.position = tuple(self.offset) if (self.renderer == "client" and len(shown) == 2 and i == 1) else (0.0, 0.0, 0.0)
            h.visible = True

    def _trail_note(self) -> str:
        n = len(next(iter(self.trail_data.values())).idx)
        return (f"_{n:,} trails prepared. Blue = past, red = future. Lines are drawn on top of the picture: no hiding by depth with the "
                "gsplat renderer, and with two files side by side only for the client renderer._")

    # ---------------------------------------------------------------- camera path (WP34)
    def _build_path_gui(self):
        g = self.server.gui
        self._path_setting = False
        self.path_handles: list = []
        with g.add_folder("Camera path", expand_by_default=False):
            self.path_info = g.add_markdown("No keys yet: move the camera, then add a key.")
            self.path_add = g.add_button("Add key from my camera")
            self.path_orbit = g.add_button("Replace by an orbit around the scene")
            self.path_sel = g.add_dropdown("Key", options=("-",), initial_value="-")
            self.path_goto = g.add_button("Go to key")
            self.path_del = g.add_button("Delete key")
            self.path_clear = g.add_button("Delete all keys")
            self.path_duration = g.add_slider("Path duration (s)", min=1.0, max=120.0, step=0.5, initial_value=10.0)
            self.path_time = g.add_dropdown("Scene time on the path", options=(PATH_HOLD, PATH_RUN), initial_value=PATH_HOLD)
            self.path_closed = g.add_checkbox("Closed loop", initial_value=False)
            self.path_pos = g.add_slider("Path position", min=0.0, max=1.0, step=0.001, initial_value=0.0)
            self.path_play = g.add_button("Play / stop path")
            self.path_save = g.add_button("Save path (json)")

        @self.path_add.on_click
        def _(e):
            cam = e.client.camera
            self.path_keys.append(dict(position=np.array(cam.position, np.float64), wxyz=np.array(cam.wxyz, np.float64), fov=float(cam.fov)))
            self._refresh_path(select=len(self.path_keys) - 1)

        @self.path_orbit.on_click
        def _(e):
            centre, dist = self._home()
            self.path_keys = orbit_keys(centre, dist, VIEW_DIR[self.args.up], self.up_vec, float(e.client.camera.fov))
            self.path_closed.value = True
            self._refresh_path(select=0)

        @self.path_goto.on_click
        def _(e):
            if self.path_keys:
                self.path_playing = False
                k = self.path_keys[self._path_selected()]
                self._move_clients([e.client], k["position"], k["wxyz"], k["fov"])

        @self.path_del.on_click
        def _(_e):
            if self.path_keys:
                i = self._path_selected()
                del self.path_keys[i]
                self._refresh_path(select=min(i, len(self.path_keys) - 1))

        @self.path_clear.on_click
        def _(_e):
            self.path_keys, self.path_playing = [], False
            self._refresh_path()

        @self.path_closed.on_update
        def _(_e):
            self._refresh_path(select=self._path_selected() if self.path_keys else None)

        @self.path_pos.on_update
        def _(_e):
            if not self._path_setting and len(self.path_keys) >= 1:
                self.path_playing = False
                self.path_u = float(self.path_pos.value)
                self._apply_path(self.path_u)

        @self.path_play.on_click
        def _(_e):
            if len(self.path_keys) < 2:
                return
            if not self.path_playing and self.path_u >= 1.0 - 1e-6:
                self.path_u = 0.0
            self.path_playing = not self.path_playing
            if self.path_playing and self.path_time.value == PATH_RUN:
                self.playing = False

        @self.path_save.on_click
        def _(e):
            if not self.path_keys:
                return
            out = getattr(self.args, "camera_path", None) or os.path.abspath(os.path.join(getattr(self.args, "video_dir", "."), "camera_path.json"))
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            save_path(out, self.path_keys, self.path_duration.value, self.path_closed.value, self.path_time.value == PATH_RUN)
            with open(out, "rb") as f:
                e.client.send_file_download(os.path.basename(out), f.read())
            e.client.add_notification("Camera path", f"Saved {len(self.path_keys)} keys to {out}", auto_close_seconds=6)

    def _home(self):
        """Centre and distance of the initial camera (as in _on_connect)."""
        centre, dist = self.centre, 2.5 * self.radius
        if self.renderer == "client" and len(self._shown()) == 2:
            centre, dist = centre + 0.5 * self.offset, dist + 0.9 * float(np.linalg.norm(self.offset))
        return centre, dist

    def _set_path(self, p: dict):
        """Installs a path read by load_path (the --camera-path file)."""
        self.path_keys = p["keys"]
        self.path_duration.value, self.path_closed.value = p["duration"], p["closed"]
        self.path_time.value = PATH_RUN if p["run"] else PATH_HOLD
        self._refresh_path(select=0)

    def _path_selected(self) -> int:
        v = self.path_sel.value
        return int(v.split(":")[0]) - 1 if v != "-" else 0

    def _refresh_path(self, select=None):
        """Updates the key list, the info text and the line and frustums drawn in the scene."""
        n = len(self.path_keys)
        opts = tuple(f"{i + 1}: ({', '.join(f'{x:.2f}' for x in k['position'])})" for i, k in enumerate(self.path_keys)) or ("-",)
        self.path_sel.options = opts
        self.path_sel.value = opts[min(select or 0, n - 1)] if n else "-"
        self.path_info.content = ("No keys yet: move the camera, then add a key." if n == 0 else
                                  f"{n} key{'s' if n != 1 else ''} · " + ("need at least 2 to play" if n < 2 else
                                                                         f"{'closed' if self.path_closed.value else 'open'} path"))
        for h in self.path_handles:
            h.remove()
        self.path_handles = []
        if n >= 2:
            closed = bool(self.path_closed.value)
            us = np.linspace(0.0, 1.0, 40 * (n if closed else n - 1) + 1)
            pts = np.array([path_pose(self.path_keys, u, closed)[0] for u in us], np.float32)
            self.path_handles.append(self.server.scene.add_line_segments(
                "/path/line", points=np.stack([pts[:-1], pts[1:]], 1), colors=(255, 120, 0), thickness=2.0, thickness_units="screen"))
        for i, k in enumerate(self.path_keys):
            self.path_handles.append(self.server.scene.add_camera_frustum(
                f"/path/key{i}", fov=k["fov"], aspect=16 / 9, scale=0.06 * self.radius, color=(255, 120, 0),
                wxyz=k["wxyz"], position=k["position"]))

    @staticmethod
    def _move_clients(clients, position, wxyz, fov):
        for c in clients:
            c.camera.wxyz, c.camera.position, c.camera.fov = tuple(wxyz), tuple(position), float(fov)

    def _apply_path(self, u: float):
        """Puts every client's camera at u on the path, and the slider at the matching time when the path runs through the sequence."""
        if not self.path_keys:
            return
        pos, wxyz, fov = path_pose(self.path_keys, u, bool(self.path_closed.value))
        self._move_clients(list(self.server.get_clients().values()), pos, wxyz, fov)
        if self.path_time.value == PATH_RUN:
            self._set_time(self.t0 + self.duration * u)
        self._path_setting = True
        self.path_pos.value = float(u)
        self._path_setting = False

    def _build_camera_gui(self):
        g, cams = self.server.gui, self.cameras
        n_img = sum(c["image"] is not None for c in cams)
        with g.add_folder(f"Cameras ({len(cams)})"):
            self.cam_names = [f"{i}: {c['name']}" + (f"  t={c['time']:.3g}" if c["time"] is not None else "") for i, c in enumerate(cams)]
            self.cam_none = "(free camera)"
            self.cam_dd = g.add_dropdown("Camera", options=(self.cam_none, *self.cam_names), initial_value=self.cam_none)
            self.cam_prev = g.add_button("◀ previous camera")
            self.cam_next = g.add_button("next camera ▶")
            self.gt_on = self.gt_bg = self.gt_info = None
            if self.renderer == "gsplat" and n_img:
                self.gt_on = g.add_checkbox("Ground truth beside render", initial_value=False)
                self.gt_bg = g.add_dropdown("Image background", options=("black", "white"),
                                            initial_value=self.args.gt_background)
                self.gt_info = g.add_markdown("Pick a camera with an image to compare.")
                for h in (self.gt_on, self.gt_bg):
                    h.on_update(lambda _e: setattr(self, "dirty", True))
            elif self.renderer == "gsplat":
                g.add_markdown("_The cameras file has no images that could be found: no ground-truth view._")

        @self.cam_dd.on_update
        def _(e):
            self._select_camera(self.cam_names.index(self.cam_dd.value) if self.cam_dd.value in self.cam_names else -1, e.client)

        @self.cam_prev.on_click
        def _(e):
            self._select_camera((self.cam_idx - 1) % len(cams) if self.cam_idx >= 0 else len(cams) - 1, e.client)

        @self.cam_next.on_click
        def _(e):
            self._select_camera((self.cam_idx + 1) % len(cams) if self.cam_idx >= 0 else 0, e.client)

    def _select_camera(self, idx: int, client):
        """Selects camera idx (-1: none): moves the clicking client's view there and the slider to the image's time."""
        self.cam_idx = idx
        self.dirty = True
        if idx < 0:
            return
        if self.cam_dd.value != self.cam_names[idx]:
            self.cam_dd.value = self.cam_names[idx]
        c = self.cameras[idx]
        if c["time"] is not None:
            self.playing = False
            self._set_time(c["time"])
        if client is not None:
            from viser.transforms import SO3
            R, pos = c["c2w"][:3, :3], c["c2w"][:3, 3]
            client.camera.wxyz = tuple(SO3.from_matrix(R).wxyz)
            client.camera.position = tuple(pos)
            if c["w"] and c["h"]:
                client.camera.fov = 2 * float(np.arctan(0.5 * c["h"] / c["fx"]))

    def _set_time(self, t: float):
        self.time.value = float(np.clip(t, self.t0, self.t1))
        self.dirty = True

    def _on_connect(self, client: viser.ClientHandle):
        d = np.asarray(VIEW_DIR[self.args.up]) / np.linalg.norm(VIEW_DIR[self.args.up])
        centre, dist = self._home()  # with two copies side by side in the client renderer both are framed
        client.camera.position = tuple(centre + dist * d)
        client.camera.look_at = tuple(centre)
        self.dirty = True

    def _shown(self) -> list[int]:
        """Indices of the sequences to draw."""
        if self.mode is None:
            return [0]
        return {"A": [0], "B": [1]}.get(self.mode.value, [0, 1])

    # ---------------------------------------------------------------- drawing
    def _current(self, i: int = 0, full_cov: bool = True):
        means, covs, opac, cols = self.seqs[i].at(self.time.value)
        if self._colour_on():
            cols = self._shade_np(i, self.time.value, means)
        sub = self.subsets[i]
        if sub is not None:
            means, covs, opac, cols = means[sub], covs[sub], opac[sub], cols[sub]
        opac = np.where(opac >= self.min_opacity.value, opac, 0.0).astype(np.float32)
        covs = (unpack_cov(covs) * np.float32(self.scale.value ** 2)).astype(np.float32) if full_cov else None
        return np.ascontiguousarray(means, np.float32), covs, opac, np.ascontiguousarray(cols, np.float32)

    def _points_on(self) -> bool:
        return self.draw_as.value == POINTS

    def _draw_client(self):
        shown = self._shown()
        side = len(shown) == 2
        pts = self._points_on()
        for i in range(len(self.seqs)):
            h, pc = self.splats[i], self.clouds[i]
            pos = tuple(self.offset) if (side and i == 1) else (0.0, 0.0, 0.0)
            if i in shown and pts:
                # one dot per Gaussian that passes the opacity threshold, in the Gaussian's base colour
                means, _, opac, cols = self._current(i, full_cov=False)
                keep = opac > 0
                xyz = means[keep]
                rgb = (np.clip(cols[keep], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                size = float(self.point_size.value) * self.radius / 1000.0
                if pc is None or self.cloud_n[i] != len(xyz):  # the count changed: send a new cloud under the same name
                    pc = self.clouds[i] = self.server.scene.add_point_cloud(
                        f"/points/{'ab'[i]}", points=xyz, colors=rgb, point_size=size, point_shape="circle",
                        point_shading="flat", precision="float32")
                    self.cloud_n[i] = len(xyz)
                else:
                    pc.points, pc.colors = xyz, rgb
                    if pc.point_size != size:
                        pc.point_size = size
                pc.position = pos
            elif i in shown:
                means, covs, opac, cols = self._current(i)
                cols = np.clip(cols, 0.0, 1.0)  # a file may hold colours outside [0,1] (Spacetime Gaussians); the browser needs RGB
                if h is None:
                    h = self.splats[i] = self.server.scene.add_gaussian_splats(
                        f"/gaussians/{'ab'[i]}", centers=means, covariances=covs, rgbs=cols, opacities=opac[:, None])
                else:
                    h.set_gaussians(centers=means, covariances=covs, rgbs=cols, opacities=opac[:, None])
                h.position = pos
            if h is not None:
                h.visible = i in shown and not pts
            if pc is not None:
                pc.visible = i in shown and pts
        if len(self.seqs) == 2:
            if self.labels is None:
                top = self.centre + 1.3 * self.radius * self.up_vec
                self.labels = [self.server.scene.add_label(f"/labels/{'ab'[i]}", n, position=tuple(top + i * self.offset))
                               for i, n in enumerate(self.names)]
            for lab in self.labels:
                lab.visible = side

    def _load_gsplat(self):
        """Tensors of the current time step for every shown sequence. Colours are SH coefficients [M,S,3]
        when the sequence has them. Decoding and interpolation run on the GPU (GpuFrames) unless --cpu-decode."""
        self.cur = {i: self._frame(i, self.min_opacity.value, self.scale.value) for i in self._shown()}

    def _frame(self, i: int, min_opacity: float, scale: float, plain: bool = False):
        """gsplat tensors of sequence i at the current time; with "Colour by" on, the colours are RGB from the speed / displacement
        (unless plain: the model as it is, for the ground-truth comparison)."""
        t, on = self.time.value, self._colour_on() and not plain
        if getattr(self.args, "cpu_decode", False):
            return frame_cpu(self.seqs[i], t, min_opacity, scale, self.device, (lambda m: self._shade_np(i, t, m)) if on else None)
        if self.gpu[i] is None:
            self.gpu[i] = GpuFrames(self.seqs[i], self.device)
        return frame_gpu(self.gpu[i], t, min_opacity, scale, (lambda m: self._shade_gpu(i, t, m)) if on else None)

    def render_gsplat(self, wxyz, position, fov_y: float, W: int, H: int) -> np.ndarray:
        """Current time step seen from a viser camera, as uint8 [H,W,3] on a white background.
        With two sequences shown, each gets half the width, drawn from the same camera."""
        shown = self._shown()
        if sorted(self.cur) != shown:
            self._load_gsplat()
        if len(shown) == 1:
            return self._render_one(shown[0], wxyz, position, fov_y, W, H)
        wl = W // 2
        halves = [self._render_one(i, wxyz, position, fov_y, w, H) for i, w in zip(shown, (wl, W - wl))]
        img = np.concatenate(halves, axis=1)
        img[:, wl - 1: wl + 1] = 90  # divider
        return self._caption(img, [(wl // 2, self.names[0]), (wl + (W - wl) // 2, self.names[1])])

    @staticmethod
    def _caption(img: np.ndarray, items) -> np.ndarray:
        """Writes each (centre x, text) at the bottom of the image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default(size=max(img.shape[0] // 36, 12))
        except (ImportError, TypeError):  # captions are optional (no Pillow, or Pillow < 10.1)
            return img
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for xc, text in items:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            x, y = xc - (r - l) // 2, img.shape[0] - (b - t) - 24
            draw.rectangle((x + l - 6, y + t - 5, x + r + 6, y + b + 5), fill=(255, 255, 255), outline=(90, 90, 90))
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
        return np.asarray(pil)

    def _raster(self, i: int, viewmat_np, K_np, W: int, H: int, white: bool = True, frame=None):
        """Current time step of sequence i for a world-to-camera matrix and intrinsics: float tensor [H,W,3] in 0..1 on the GPU,
        over a white or black background."""
        torch = self.torch
        from gsplat import rasterization

        means, covs, opac, cols = self.cur[i] if frame is None else frame
        sh_degree = None if (frame is None and self._colour_on()) else self.seqs[i].sh_degree  # shaded colours are plain RGB
        if len(means) == 0:  # everything hidden by the opacity threshold
            return torch.full((H, W, 3), 1.0 if white else 0.0, device=self.device)
        K = torch.from_numpy(K_np).to(self.device)
        viewmat = torch.from_numpy(viewmat_np).to(self.device)
        with torch.no_grad():
            if frame is None and self._points_on():
                # one opaque isotropic Gaussian per centre whose size on screen is the same at every depth:
                # sigma (world) = sigma (px) * depth / focal, with "Point size" as the full width at half maximum
                depth = (means @ viewmat[2, :3] + viewmat[2, 3]).clamp_min(1e-6)
                sigma = depth * (float(self.point_size.value) / 2.355 / float(K_np[0, 0]))
                quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(means), 4)
                img, alpha, _ = rasterization(means, quats, sigma[:, None].expand(-1, 3), torch.ones_like(opac), cols,
                                              viewmat[None], K[None], W, H, sh_degree=sh_degree)
            else:
                img, alpha, _ = rasterization(means, None, None, opac, cols, viewmat[None], K[None], W, H, covars=covs,
                                              sh_degree=sh_degree)
            if white:  # composite over white, the browser renderer's background
                img = img[0] + (1.0 - alpha[0])  # (gsplat 1.5.3 rejects `backgrounds` of shape [C,3] when `covars` is given)
            else:
                img = img[0]
        return img.clamp(0, 1)

    def _render_one(self, i: int, wxyz, position, fov_y: float, W: int, H: int) -> np.ndarray:
        viewmat_np, K_np = camera_matrices(wxyz, position, fov_y, W, H)
        img = self._raster(i, viewmat_np, K_np, W, H)
        return (img * 255).round().to(self.torch.uint8).cpu().numpy()

    # ---------------------------------------------------------------- ground truth next to the render
    def gt_compare(self, i: int, cam_idx: int, white: bool, max_h: int):
        """Render of sequence i at the current time from training camera cam_idx, next to its image: (render, ground truth) as
        float32 [H,W,3] in 0..1 and the PSNR in dB. Size: the image's own, capped at height max_h (check_real.py's recipe)."""
        c = self.cameras[cam_idx]
        H = min(max_h, c["h"])
        W = max(int(round(c["w"] * H / c["h"])), 1)
        key = (c["image"], W, H, white)
        if key not in self.gt_cache:
            if len(self.gt_cache) > 64:
                self.gt_cache.clear()
            self.gt_cache[key] = load_gt_image(c["image"], W, H, white)
        gt = self.gt_cache[key]
        f = c["fx"] * W / c["w"]
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
        # the model as it is: the Display sliders (opacity threshold, size) do not apply, so the PSNR is the model's
        pred = self._raster(i, np.linalg.inv(c["c2w"]).astype(np.float32), K, W, H, white, self._frame(i, 0.0, 1.0, plain=True)).cpu().numpy()
        mse = float(np.mean((pred - gt) ** 2))
        return pred, gt, 10 * np.log10(1.0 / max(mse, 1e-12))

    def _gt_view(self, W: int, H: int):
        """Picture for a client window of W x H showing render | ground truth of the selected camera, or None when the
        comparison is off. With two sequences shown: render A | ground truth | render B. Also writes the PSNR to the panel."""
        if self.gt_on is None or not self.gt_on.value:
            return None
        c = self.cameras[self.cam_idx] if self.cam_idx >= 0 else None
        if c is None or c["image"] is None or not c["w"]:
            self.gt_info.content = "Pick a camera that has an image."
            return None
        shown = self._shown()
        key = (self.cam_idx, self.time.value, self.max_res.value, self.gt_bg.value, tuple(shown))
        if self.gt_key != key or self.gt_last is None:
            res = [self.gt_compare(i, self.cam_idx, self.gt_bg.value == "white", int(self.max_res.value)) for i in shown]
            self.gt_last, self.gt_key = res, key
            off = c["time"] is not None and abs(c["time"] - self.time.value) > 1e-6 * max(self.duration, 1.0)
            pred = res[0][0]
            text = (f"**PSNR {res[0][2]:.2f} dB**" if len(self.seqs) == 1 else
                    " · ".join(f"**{'AB'[i]}: {r[2]:.2f} dB**" for i, r in zip(shown, res)))
            self.gt_info.content = (f"{text} · {pred.shape[1]}x{pred.shape[0]}"
                                    + (f"\n\n_Slider is at t={self.time.value:.3g}, the image is at t={c['time']:.3g}: "
                                       "the two differ in time._" if off else ""))
        res = self.gt_last
        label = (lambda i: "render") if len(self.seqs) == 1 else (lambda i: self.names[i])
        items = [(res[0][0], f"{label(shown[0])} · PSNR {res[0][2]:.2f} dB"), (res[0][1], f"ground truth · {c['name']}")]
        if len(res) == 2:  # the image in the middle, one model on each side
            items.append((res[1][0], f"{label(shown[1])} · PSNR {res[1][2]:.2f} dB"))
        from PIL import Image
        n = len(items)
        edges = [W * k // n for k in range(n + 1)]
        panels, captions = [], []
        for k, (im, text) in enumerate(items):
            w = edges[k + 1] - edges[k]
            pil = Image.fromarray((im * 255).round().astype(np.uint8))
            s = min(w / pil.size[0], H / pil.size[1])
            pil = pil.resize((max(int(pil.size[0] * s), 1), max(int(pil.size[1] * s), 1)), Image.BILINEAR)
            canvas = np.full((H, w, 3), 128, np.uint8)
            y0, x0 = (H - pil.size[1]) // 2, (w - pil.size[0]) // 2
            canvas[y0:y0 + pil.size[1], x0:x0 + pil.size[0]] = np.asarray(pil)
            panels.append(canvas)
            captions.append((edges[k] + w // 2, text))
        img = np.concatenate(panels, axis=1)
        for e in edges[1:-1]:
            img[:, e - 1: e + 1] = 255
        return self._caption(img, captions)

    def _draw_gsplat(self, time_changed: bool):
        if time_changed or sorted(self.cur) != self._shown():
            self._load_gsplat()
        clients = self.server.get_clients()
        for cid in [c for c in self.cam_stamp if c not in clients]:  # forget disconnected clients
            del self.cam_stamp[cid]
        for cid, client in clients.items():
            cam = client.camera
            stamp = cam.update_timestamp
            if not time_changed and self.cam_stamp.get(cid) == stamp:
                continue
            self.cam_stamp[cid] = stamp
            H = int(min(self.max_res.value, cam.image_height or self.max_res.value))
            W = max(int(round(H * cam.aspect)), 1)
            rgb = self._gt_view(W, H) if self.cameras else None
            if rgb is None:
                rgb = self.render_gsplat(cam.wxyz, cam.position, cam.fov, W, H)
            client.scene.set_background_image(rgb, format="jpeg", jpeg_quality=85)

    # ---------------------------------------------------------------- video export
    def _export_video(self, client: viser.ClientHandle):
        """Renders the whole time range from the client's current camera into an mp4 (fixed camera)."""
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            client.add_notification("Export video", "Needs: pip install imageio imageio-ffmpeg", color="red")
            return
        cam = client.camera
        wxyz, pos, fov = np.array(cam.wxyz), np.array(cam.position), float(cam.fov)
        along = self.video_cam.value == VIDEO_PATH
        if along and len(self.path_keys) < 2:
            client.add_notification("Export video", "Camera path: add at least 2 keys first.", color="red")
            return
        closed, run = bool(self.path_closed.value), self.path_time.value == PATH_RUN
        H = int(cam.image_height or 720)
        if self.renderer == "gsplat":
            H = int(min(self.max_res.value, H))
        H -= H % 2  # h264 needs even sizes
        W = max(int(round(H * cam.aspect)), 2)
        W -= W % 2
        fps = int(self.video_fps.value)
        n = max(int(round(fps * (self.path_duration.value if along else self.video_seconds.value))), 2)
        stem = os.path.splitext(os.path.basename(getattr(self.args, "file", "sequence")))[0]
        path = os.path.abspath(os.path.join(getattr(self.args, "video_dir", "."), f"{stem}{'_path' if along else ''}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t_before, self.playing, self.path_playing = self.time.value, False, False
        self.video_btn.disabled = True
        note = client.add_notification("Export video", f"Rendering {n} frames at {W}x{H} ...", loading=True,
                                       with_close_button=False, auto_close_seconds=None)
        try:
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
                for i in range(n):
                    if along:  # a closed path ends one step before its first key, so the video loops
                        u = i / n if closed else i / (n - 1)
                        pos, wxyz, fov = path_pose(self.path_keys, u, closed)
                        if run:
                            self._set_time(self.t0 + self.duration * u)
                    else:
                        self._set_time(self.t0 + self.duration * i / (n - 1))
                    if self.renderer == "gsplat":
                        self._load_gsplat()
                        frame = self.render_gsplat(wxyz, pos, fov, W, H)
                    else:  # ask the browser for the picture once it has received this time step
                        self._draw_client()
                        time.sleep(0.15)
                        frame = client.get_render(height=H, width=W, wxyz=wxyz, position=pos, fov=fov)[..., :3]
                    writer.append_data(np.ascontiguousarray(frame))
            with open(path, "rb") as f:
                client.send_file_download(os.path.basename(path), f.read())
            note.title, note.body, note.loading, note.with_close_button = "Video exported", path, False, True
            print(f"video: {path} ({n} frames, {W}x{H}, {fps} fps)")
        except Exception as err:  # keep the viewer alive
            note.title, note.body, note.loading, note.with_close_button, note.color = "Export failed", str(err), False, True, "red"
            print(f"video export failed: {err!r}")
        finally:
            self.video_btn.disabled = False
            self._set_time(t_before)

    # ---------------------------------------------------------------- loop
    def run(self):
        url = f"http://{'localhost' if self.args.host in ('127.0.0.1', '0.0.0.0') else self.args.host}:{self.args.port}"
        print("\n".join(f"{n}: {q.summary()}" for n, q in zip(self.names, self.seqs)) if len(self.seqs) == 2
              else self.seq.summary())
        print(f"renderer: {self.renderer}\nOpen {url} in a browser (Ctrl+C to quit)")
        last = time.time()
        while True:
            if self.export_client is not None:
                self._export_video(self.export_client)
                self.export_client, last = None, time.time()
            now = time.time()
            dt, last = now - last, now
            if self.playing:
                t = self.time.value + dt * self.speed.value
                if t >= self.t1:
                    if self.loop.value:
                        t = self.t0 + (t - self.t0) % self.duration
                    else:
                        t, self.playing = self.t1, False
                self._set_time(t)
            if self.path_playing:
                u = self.path_u + dt / max(float(self.path_duration.value), 1e-3)
                if u >= 1.0:
                    if self.loop.value:
                        u %= 1.0
                    else:
                        u, self.path_playing = 1.0, False
                self.path_u = u
                self._apply_path(u)
            time_changed, self.dirty = self.dirty, False
            if self.renderer == "client":
                if time_changed:
                    self._draw_client()
            else:
                self._draw_gsplat(time_changed)
            if time_changed:
                self._draw_trails()
            time.sleep(max(1.0 / self.target_fps.value - (time.time() - now), 0.001))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="sequence in the common .npz format (see the convert_*.py scripts)")
    ap.add_argument("--compare", metavar="FILE", default=None,
                    help="second sequence to compare with the first: shown alone or side by side, same time slider")
    ap.add_argument("--cameras", metavar="FILE", default=None,
                    help="training cameras: transforms_*.json (NeRF/D-NeRF) or cameras.json; adds a camera list to the panel "
                         "(jump to the camera and the image's time); gsplat renderer can show the image next to the render with PSNR")
    ap.add_argument("--camera-path", metavar="FILE", default=None,
                    help="camera path (json: keys with c2w + fov_y, duration, closed, time; same convention as --cameras): "
                         "play it or export an mp4 along it from the Camera path folder; the Save button writes back to this file")
    ap.add_argument("--gt-background", choices=["black", "white"], default="black",
                    help="background used for the ground-truth comparison (RGBA images and the render); default black as in tests/check_real.py")
    ap.add_argument("--renderer", choices=["auto", "gsplat", "client"], default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--up", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+z",
                    help="world up axis; COLMAP-style scenes are usually -y, Blender/D-NeRF scenes +z")
    ap.add_argument("--draw-as", choices=[SPLATS, POINTS], default=SPLATS,
                    help="initial value of 'Draw as' in the Display folder: the Gaussians, or one dot at each Gaussian's centre")
    ap.add_argument("--max-splats", type=int, default=300_000, help="client renderer: cap on Gaussians sent to the browser")
    ap.add_argument("--max-res", type=int, default=1080, help="gsplat renderer: initial cap on the render height")
    ap.add_argument("--cpu-decode", action="store_true",
                    help="gsplat renderer: decode and interpolate keyframes with numpy instead of on the GPU (reference path)")
    ap.add_argument("--video-dir", default=".", help="folder for the mp4 files written by the Export video button")
    args = ap.parse_args()
    try:
        seq = Sequence4D.load(args.file, lazy=True)
        seq_b = Sequence4D.load(args.compare, lazy=True) if args.compare else None
    except (OSError, ValueError) as e:   # missing file, or not in the common format (the message says what to do)
        raise SystemExit(f"viewer.py: {e}") from None
    Viewer(seq, args, seq_b).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
