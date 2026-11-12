#!/usr/bin/env python3
"""4D Gaussian Splatting viewer with a time slider (viser front end).

    python viewer.py toy.npz
    python viewer.py scene.npz --renderer gsplat --up -y --port 8080

Renderers:
    gsplat  images are rasterized on this machine's NVIDIA GPU with gsplat and streamed to the
            browser. Exact 3DGS alpha blending; handles millions of Gaussians.
    client  Gaussians are sent to the browser and drawn there with WebGL (viser's splat renderer).
            No CUDA needed; meant for small scenes or a laptop.
    auto    gsplat when CUDA and gsplat are available, otherwise client.
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np
import viser

from gs4d import Sequence4D, unpack_cov


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


class Viewer:
    def __init__(self, seq: Sequence4D, args):
        self.seq, self.args = seq, args
        self.renderer = pick_renderer(args.renderer)
        self.t0, self.t1 = float(seq.times[0]), float(seq.times[-1])
        self.duration = max(self.t1 - self.t0, 1e-9)
        self.lock = threading.Lock()
        self.dirty = True
        self.cam_stamp: dict[int, float] = {}

        # fixed subset for the browser renderer, so the splat count never changes between frames
        self.subset = None
        if self.renderer == "client" and seq.num_gaussians > args.max_splats:
            peak = seq.opacities.max(0) if seq.opacities.ndim == 2 else seq.opacities
            self.subset = np.sort(np.argpartition(peak, -args.max_splats)[-args.max_splats:])
            print(f"client renderer: showing the {args.max_splats:,} most opaque of {seq.num_gaussians:,} Gaussians "
                  "(raise --max-splats or use --renderer gsplat)")

        m0 = seq.means[0] if seq.means.ndim == 3 else seq.means
        self.centre = np.median(m0, 0)
        self.radius = float(np.percentile(np.linalg.norm(m0 - self.centre, axis=1), 90)) or 1.0

        self.server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
        self.server.scene.set_up_direction(args.up)
        self._build_gui()
        self.server.on_client_connect(self._on_connect)
        self.splats = None
        self.cur = None  # tensors of the current time step (gsplat renderer)
        if self.renderer == "gsplat":
            import torch

            self.torch = torch
            self.device = torch.device("cuda")

    # ---------------------------------------------------------------- GUI
    def _build_gui(self):
        g, s = self.server.gui, self.seq
        g.add_markdown(f"**{s.num_gaussians:,}** Gaussians · **{s.num_frames}** keyframes · renderer **{self.renderer}**"
                       + (f"\n\n{s.source}" if s.source else ""))
        self.time = g.add_slider("Time", min=self.t0, max=self.t1, step=self.duration / 1000.0, initial_value=self.t0)
        self.frame = g.add_number("Keyframe", initial_value=0, min=0, max=s.num_frames - 1, step=1)
        self.play = g.add_button("Play / Pause")
        self.prev = g.add_button("◀ previous keyframe")
        self.next = g.add_button("next keyframe ▶")
        self.speed = g.add_slider("Speed (x)", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        self.loop = g.add_checkbox("Loop", initial_value=True)
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

    def _set_time(self, t: float):
        self.time.value = float(np.clip(t, self.t0, self.t1))
        self.dirty = True

    def _on_connect(self, client: viser.ClientHandle):
        direction = {"+z": (1.0, -1.0, 0.6), "-z": (1.0, -1.0, -0.6), "+y": (1.0, 0.6, 1.0),
                     "-y": (1.0, -0.6, 1.0), "+x": (0.6, 1.0, -1.0), "-x": (-0.6, 1.0, -1.0)}[self.args.up]
        d = np.asarray(direction) / np.linalg.norm(direction)
        client.camera.position = tuple(self.centre + 2.5 * self.radius * d)
        client.camera.look_at = tuple(self.centre)
        self.dirty = True

    # ---------------------------------------------------------------- drawing
    def _current(self):
        means, covs, opac, cols = self.seq.at(self.time.value)
        if self.subset is not None:
            means, covs, opac, cols = means[self.subset], covs[self.subset], opac[self.subset], cols[self.subset]
        opac = np.where(opac >= self.min_opacity.value, opac, 0.0).astype(np.float32)
        covs = unpack_cov(covs) * np.float32(self.scale.value ** 2)
        return np.ascontiguousarray(means, np.float32), covs.astype(np.float32), opac, np.ascontiguousarray(cols, np.float32)

    def _draw_client(self):
        means, covs, opac, cols = self._current()
        if self.splats is None:
            self.splats = self.server.scene.add_gaussian_splats("/gaussians", centers=means, covariances=covs,
                                                                rgbs=cols, opacities=opac[:, None])
        else:
            self.splats.set_gaussians(centers=means, covariances=covs, rgbs=cols, opacities=opac[:, None])

    def _load_gsplat(self):
        """Tensors of the current time step. Colours are SH coefficients [M,S,3] when the sequence has them."""
        torch = self.torch
        means, covs, opac, cols = self._current()
        sh = self.seq.sh_at(self.time.value)
        keep = opac > 0
        self.cur = tuple(torch.from_numpy(np.ascontiguousarray(a[keep], np.float32)).to(self.device)
                         for a in (means, covs, opac, cols if sh is None else sh))

    def render_gsplat(self, wxyz, position, fov_y: float, W: int, H: int) -> np.ndarray:
        """Current time step seen from a viser camera, as uint8 [H,W,3] on a white background."""
        torch = self.torch
        from gsplat import rasterization

        if self.cur is None:
            self._load_gsplat()
        means, covs, opac, cols = self.cur
        if len(means) == 0:  # everything hidden by the opacity threshold
            return np.full((H, W, 3), 255, np.uint8)
        viewmat_np, K_np = camera_matrices(wxyz, position, fov_y, W, H)
        K = torch.from_numpy(K_np).to(self.device)
        viewmat = torch.from_numpy(viewmat_np).to(self.device)
        with torch.no_grad():
            img, alpha, _ = rasterization(means, None, None, opac, cols, viewmat[None], K[None], W, H, covars=covs,
                                          sh_degree=self.seq.sh_degree)
            img = img[0] + (1.0 - alpha[0])  # composite over white, the browser renderer's background
            # (gsplat 1.5.3 rejects `backgrounds` of shape [C,3] when `covars` is given)
        return (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()

    def _draw_gsplat(self, time_changed: bool):
        if time_changed or self.cur is None:
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
        print(f"{self.seq.summary()}\nrenderer: {self.renderer}\nOpen {url} in a browser (Ctrl+C to quit)")
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
    ap.add_argument("--renderer", choices=["auto", "gsplat", "client"], default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--up", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+z",
                    help="world up axis; COLMAP-style scenes are usually -y, Blender/D-NeRF scenes +z")
    ap.add_argument("--max-splats", type=int, default=300_000, help="client renderer: cap on Gaussians sent to the browser")
    ap.add_argument("--max-res", type=int, default=1080, help="gsplat renderer: initial cap on the render height")
    ap.add_argument("--video-dir", default=".", help="folder for the mp4 files written by the Export video button")
    args = ap.parse_args()
    Viewer(Sequence4D.load(args.file, lazy=True), args).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
