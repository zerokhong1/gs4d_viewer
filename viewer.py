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
import threading
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


def frame_cpu(seq: Sequence4D, t: float, min_opacity: float, scale: float, device):
    """gsplat inputs at time t, decoded and interpolated with numpy, then uploaded: (means, full covs [M,3,3], opacities,
    colours or SH [M,S,3]) for the Gaussians whose opacity is at least min_opacity. The reference path."""
    import torch

    means, covs, opac, cols = seq.at(t)
    sh = seq.sh_at(t)
    opac = np.where(opac >= min_opacity, opac, 0.0).astype(np.float32)
    covs = (unpack_cov(covs) * np.float32(scale ** 2)).astype(np.float32)
    keep = opac > 0
    return tuple(torch.from_numpy(np.ascontiguousarray(a[keep], np.float32)).to(device)
                 for a in (means, covs, opac, cols if sh is None else sh))


def frame_gpu(gpu: GpuFrames, t: float, min_opacity: float, scale: float):
    """Same result as frame_cpu, with decoding, interpolation, unpacking and thresholding done on the GPU."""
    torch = gpu.torch
    means, covs, opac, cols = gpu.at(t)
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
    meta = json.load(open(path))
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


SIDE_BY_SIDE = "A | B"
# initial camera direction (scene centre -> camera) for each world up axis
VIEW_DIR = {"+z": (1.0, -1.0, 0.6), "-z": (1.0, -1.0, -0.6), "+y": (1.0, 0.6, 1.0),
            "-y": (1.0, -0.6, 1.0), "+x": (0.6, 1.0, -1.0), "-x": (-0.6, 1.0, -1.0)}


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
        self.lock = threading.Lock()
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
                peak = s.opacities.max(0) if s.opacities.ndim == 2 else s.opacities
                self.subsets[i] = np.sort(np.argpartition(peak, -args.max_splats)[-args.max_splats:])
                print(f"client renderer: showing the {args.max_splats:,} most opaque of {s.num_gaussians:,} Gaussians "
                      "(raise --max-splats or use --renderer gsplat)")

        self.centre, self.radius = _bounds(seq)
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
        self.splats = [None] * len(self.seqs)
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
        with g.add_folder("Display", expand_by_default=False):
            self.min_opacity = g.add_slider("Hide opacity below", min=0.0, max=0.5, step=0.005, initial_value=0.02)
            self.scale = g.add_slider("Gaussian size (x)", min=0.1, max=2.0, step=0.05, initial_value=1.0)
            self.target_fps = g.add_slider("Update rate (Hz)", min=2, max=60, step=1, initial_value=20)
            if self.renderer == "gsplat":
                self.max_res = g.add_slider("Max render height", min=240, max=2160, step=60,
                                            initial_value=self.args.max_res)
        with g.add_folder("Export video", expand_by_default=False):
            self.video_fps = g.add_number("Frames per second", initial_value=30, min=1, max=120, step=1)
            self.video_seconds = g.add_number("Video length (s)", initial_value=float(np.clip(round(self.duration, 1), 2.0, 20.0)),
                                              min=0.5, max=600.0, step=0.5)
            self.video_btn = g.add_button("Export mp4 from this camera")
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

        for h in (self.min_opacity, self.scale):
            h.on_update(lambda _e: setattr(self, "dirty", True))

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
        centre, dist = self.centre, 2.5 * self.radius
        if self.renderer == "client" and len(self._shown()) == 2:  # frame both copies
            centre, dist = centre + 0.5 * self.offset, dist + 0.9 * float(np.linalg.norm(self.offset))
        client.camera.position = tuple(centre + dist * d)
        client.camera.look_at = tuple(centre)
        self.dirty = True

    def _shown(self) -> list[int]:
        """Indices of the sequences to draw."""
        if self.mode is None:
            return [0]
        return {"A": [0], "B": [1]}.get(self.mode.value, [0, 1])

    # ---------------------------------------------------------------- drawing
    def _current(self, i: int = 0):
        means, covs, opac, cols = self.seqs[i].at(self.time.value)
        sub = self.subsets[i]
        if sub is not None:
            means, covs, opac, cols = means[sub], covs[sub], opac[sub], cols[sub]
        opac = np.where(opac >= self.min_opacity.value, opac, 0.0).astype(np.float32)
        covs = unpack_cov(covs) * np.float32(self.scale.value ** 2)
        return np.ascontiguousarray(means, np.float32), covs.astype(np.float32), opac, np.ascontiguousarray(cols, np.float32)

    def _draw_client(self):
        shown = self._shown()
        side = len(shown) == 2
        for i in range(len(self.seqs)):
            h = self.splats[i]
            if i in shown:
                means, covs, opac, cols = self._current(i)
                if h is None:
                    h = self.splats[i] = self.server.scene.add_gaussian_splats(
                        f"/gaussians/{'ab'[i]}", centers=means, covariances=covs, rgbs=cols, opacities=opac[:, None])
                else:
                    h.set_gaussians(centers=means, covariances=covs, rgbs=cols, opacities=opac[:, None])
                h.position = tuple(self.offset) if (side and i == 1) else (0.0, 0.0, 0.0)
            if h is not None:
                h.visible = i in shown
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

    def _frame(self, i: int, min_opacity: float, scale: float):
        if getattr(self.args, "cpu_decode", False):
            return frame_cpu(self.seqs[i], self.time.value, min_opacity, scale, self.device)
        if self.gpu[i] is None:
            self.gpu[i] = GpuFrames(self.seqs[i], self.device)
        return frame_gpu(self.gpu[i], self.time.value, min_opacity, scale)

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
        if len(means) == 0:  # everything hidden by the opacity threshold
            return torch.full((H, W, 3), 1.0 if white else 0.0, device=self.device)
        K = torch.from_numpy(K_np).to(self.device)
        viewmat = torch.from_numpy(viewmat_np).to(self.device)
        with torch.no_grad():
            img, alpha, _ = rasterization(means, None, None, opac, cols, viewmat[None], K[None], W, H, covars=covs,
                                          sh_degree=self.seqs[i].sh_degree)
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
        pred = self._raster(i, np.linalg.inv(c["c2w"]).astype(np.float32), K, W, H, white, self._frame(i, 0.0, 1.0)).cpu().numpy()
        mse = float(np.mean((pred - gt) ** 2))
        return pred, gt, 10 * np.log10(1.0 / max(mse, 1e-12))

    def _gt_view(self, W: int, H: int):
        """Picture for a client window of W x H showing render | ground truth of the selected camera, or None when the
        comparison is off. Also writes the PSNR to the panel."""
        if self.gt_on is None or not self.gt_on.value or len(self._shown()) != 1:
            return None
        c = self.cameras[self.cam_idx] if self.cam_idx >= 0 else None
        if c is None or c["image"] is None or not c["w"]:
            self.gt_info.content = "Pick a camera that has an image."
            return None
        key = (self.cam_idx, self.time.value, self.max_res.value, self.gt_bg.value)
        if self.gt_key != key or self.gt_last is None:
            pred, gt, psnr = self.gt_compare(self._shown()[0], self.cam_idx, self.gt_bg.value == "white", int(self.max_res.value))
            self.gt_last, self.gt_key = (pred, gt, psnr), key
            off = c["time"] is not None and abs(c["time"] - self.time.value) > 1e-6 * max(self.duration, 1.0)
            self.gt_info.content = (f"**PSNR {psnr:.2f} dB** · {pred.shape[1]}x{pred.shape[0]}"
                                    + (f"\n\n_Slider is at t={self.time.value:.3g}, the image is at t={c['time']:.3g}: "
                                       "the two differ in time._" if off else ""))
        pred, gt, psnr = self.gt_last
        from PIL import Image
        wl = W // 2
        panels = []
        for im, w in ((pred, wl), (gt, W - wl)):
            pil = Image.fromarray((im * 255).round().astype(np.uint8))
            s = min(w / pil.size[0], H / pil.size[1])
            pil = pil.resize((max(int(pil.size[0] * s), 1), max(int(pil.size[1] * s), 1)), Image.BILINEAR)
            canvas = np.full((H, w, 3), 128, np.uint8)
            y0, x0 = (H - pil.size[1]) // 2, (w - pil.size[0]) // 2
            canvas[y0:y0 + pil.size[1], x0:x0 + pil.size[0]] = np.asarray(pil)
            panels.append(canvas)
        img = np.concatenate(panels, axis=1)
        img[:, wl - 1: wl + 1] = 255
        return self._caption(img, [(wl // 2, f"render · PSNR {psnr:.2f} dB"), (wl + (W - wl) // 2, f"ground truth · {c['name']}")])

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
        H = int(cam.image_height or 720)
        if self.renderer == "gsplat":
            H = int(min(self.max_res.value, H))
        H -= H % 2  # h264 needs even sizes
        W = max(int(round(H * cam.aspect)), 2)
        W -= W % 2
        fps = int(self.video_fps.value)
        n = max(int(round(fps * self.video_seconds.value)), 2)
        stem = os.path.splitext(os.path.basename(getattr(self.args, "file", "sequence")))[0]
        path = os.path.abspath(os.path.join(getattr(self.args, "video_dir", "."), f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t_before, self.playing = self.time.value, False
        self.video_btn.disabled = True
        note = client.add_notification("Export video", f"Rendering {n} frames at {W}x{H} ...", loading=True,
                                       with_close_button=False, auto_close_seconds=None)
        try:
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
                for i in range(n):
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
            time_changed, self.dirty = self.dirty, False
            if self.renderer == "client":
                if time_changed:
                    self._draw_client()
            else:
                self._draw_gsplat(time_changed)
            time.sleep(max(1.0 / self.target_fps.value - (time.time() - now), 0.001))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="sequence in the common .npz format (see the convert_*.py scripts)")
    ap.add_argument("--compare", metavar="FILE", default=None,
                    help="second sequence to compare with the first: shown alone or side by side, same time slider")
    ap.add_argument("--cameras", metavar="FILE", default=None,
                    help="training cameras: transforms_*.json (NeRF/D-NeRF) or cameras.json; adds a camera list to the panel "
                         "(jump to the camera and the image's time); gsplat renderer can show the image next to the render with PSNR")
    ap.add_argument("--gt-background", choices=["black", "white"], default="black",
                    help="background used for the ground-truth comparison (RGBA images and the render); default black as in tests/check_real.py")
    ap.add_argument("--renderer", choices=["auto", "gsplat", "client"], default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--up", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+z",
                    help="world up axis; COLMAP-style scenes are usually -y, Blender/D-NeRF scenes +z")
    ap.add_argument("--max-splats", type=int, default=300_000, help="client renderer: cap on Gaussians sent to the browser")
    ap.add_argument("--max-res", type=int, default=1080, help="gsplat renderer: initial cap on the render height")
    ap.add_argument("--cpu-decode", action="store_true",
                    help="gsplat renderer: decode and interpolate keyframes with numpy instead of on the GPU (reference path)")
    ap.add_argument("--video-dir", default=".", help="folder for the mp4 files written by the Export video button")
    args = ap.parse_args()
    seq_b = Sequence4D.load(args.compare, lazy=True) if args.compare else None
    Viewer(Sequence4D.load(args.file, lazy=True), args, seq_b).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
