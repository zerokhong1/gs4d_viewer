"""Open the output of a 4DGS method directly: recognise it, convert it, start the viewer.

    python -m gs4d.open <output_path> [--reconvert] [--dry-run] [viewer.py arguments]

Recognised (checked in this order, the first match wins; nothing is guessed):
    *.npz in the common format (times/means/covs/opacities/colors)   -> opened as it is
    params.npz (Dynamic 3D Gaussians) or a folder of per-frame .ply  -> convert_per_frame.py
    chkpnt*.pth (native 4DGS)                                        -> convert_native4d.py
    other *.ckpt / *.pth, by the key names in the checkpoint's pickle (read without torch):
        gauss_params quats1, quats2, scales_t (4D-Rotor GS, Nerfstudio) -> convert_4drotor.py
        fg.params.means, motion_bases.params (Shape of Motion)       -> convert_shape_of_motion.py
        scf._node_xyz, attach_ind (MoSca DynSCFGaussian)             -> convert_mosca.py
      when the keys cannot be read, by the name: step-*.ckpt -> 4D-Rotor GS, any other *.ckpt -> Shape of Motion,
      photometric_d_model_*.pth -> MoSca (the converter checks the keys and stops when they are not there)
    folder with checkpoints/last.ckpt (Shape of Motion), with nerfstudio_models/step-*.ckpt (4D-Rotor GS, the
      last step), or with photometric_d_model_*.pth (a MoSca log folder, the newest file) -> as that file
    folder with point_cloud/ and deform/                             -> convert_scgs.py when the ply has
                                                                        fea_* columns, else convert_deformable3dgs.py
    folder with point_cloud/iteration_*/deformation.pth (no deform/) -> convert_ed3dgs.py when the ply has embedding_*
                                                                        columns (E-D3DGS), else convert_4dgaussians.py
    point_cloud.ply with trbf_center (Spacetime Gaussians), or a
    folder holding point_cloud/iteration_*/point_cloud.ply of that kind -> convert_spacetime.py
Cameras: unless --cameras is given, the viewer gets --cameras for a cameras.json in the output's folder (or up to 3
folders above it) or, failing that, for the transforms_test.json of the dataset named by source_path in cfg_args.
When that dataset has the Nerfies / HyperNeRF layout (camera/, metadata.json, rgb/<N>x/) and the output is a 4DGaussians,
Deformable-3DGS or SC-GS folder, the held-out cameras are written to <name>.gs4d.cameras.json next to the output.
The converted file is <name>.gs4d.npz next to the output and is reused while it is newer than the output and the converter script.
D-NeRF scenes (cfg_args next to the output says dnerf / is_blender) get the time range [0, 1].
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMMON_KEYS = ("times", "means", "covs", "opacities", "colors")


class NotRecognised(Exception):
    pass


def _npz_keys(p: Path) -> list[str]:
    with np.load(p, allow_pickle=False) as z:
        return list(z.files)


def _ply_names(p: Path) -> tuple[str, ...]:
    # only the header is read
    names: list[str] = []
    with open(p, "rb") as f:
        if f.readline().strip() != b"ply":
            return ()
        in_vertex = False
        for raw in f:
            tok = raw.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "element":
                in_vertex = tok[1] == "vertex"
            elif tok[0] == "property" and in_vertex and tok[1] != "list":
                names.append(tok[2])
            elif tok[0] == "end_header":
                break
    return tuple(names)


CKPT_KEYS = {   # kind -> key names that must all be in the checkpoint's pickle (state dict keys are stored as plain text)
    "4drotor": (b"quats1", b"quats2", b"scales_t"),                  # Nerfstudio pipeline: _model.gauss_params.*
    "shape_of_motion": (b"fg.params.means", b"motion_bases.params"),
    "mosca": (b"scf._node_xyz", b"attach_ind"),                      # DynSCFGaussian state dict
}
CONVERTERS = {"4drotor": "convert_4drotor.py", "shape_of_motion": "convert_shape_of_motion.py", "mosca": "convert_mosca.py"}


def _pickle_bytes(p: Path, limit: int = 16 << 20) -> bytes:
    """The pickled part of a torch checkpoint, which holds the key names, read without torch: the data.pkl member of a
    zip checkpoint (torch.save since torch 1.6), else the start of the file (legacy format: the pickle comes before the
    tensor data). b"" when the file cannot be read."""
    try:
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                pkl = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
                if pkl is None:
                    return b""
                with z.open(pkl) as f:
                    return f.read(limit)
        with open(p, "rb") as f:
            return f.read(limit)
    except (OSError, zipfile.BadZipFile):
        return b""


def _ckpt_kind(p: Path) -> str | None:
    """'4drotor', 'shape_of_motion' or 'mosca' from the key names in the checkpoint, None when none of them match."""
    data = _pickle_bytes(p)
    return next((kind for kind, keys in CKPT_KEYS.items() if all(k in data for k in keys)), None)


def _nerfstudio_ckpts(folder: Path) -> list[Path]:
    """step-*.ckpt of a Nerfstudio run folder (in nerfstudio_models/) or of that folder itself, by step."""
    for d in (folder / "nerfstudio_models", folder):
        found = []
        for f in (d.glob("step-*.ckpt") if d.is_dir() else ()):
            m = re.fullmatch(r"step-(\d+)\.ckpt", f.name)
            if m and f.is_file():
                found.append((int(m.group(1)), f))
        if found:
            return [f for _, f in sorted(found)]
    return []


def _last_iteration(folder: Path, fname: str) -> Path | None:
    cands = []
    for d in folder.glob("iteration_*"):
        m = re.fullmatch(r"iteration_(\d+)", d.name)
        if m and (d / fname).is_file():
            cands.append((int(m.group(1)), d / fname))
    return max(cands)[1] if cands else None


def _is_dnerf(path: Path) -> bool:
    """cfg_args (written by the training scripts) in the folder of the output or one of its parents."""
    d = path if path.is_dir() else path.parent
    for cur in [d, *list(d.parents)[:4]]:
        cfg = cur / "cfg_args"
        if cfg.is_file():
            txt = cfg.read_text(errors="replace")
            return "dnerf" in txt.lower() or "is_blender=True" in txt
    return False


def _readable_cameras(p: Path) -> bool:
    """True when viewer.load_cameras can read the file: its frames carry transform_matrix or c2w. The cameras.json that
    3DGS training writes (position + rotation per camera) does not, so it is not passed on."""
    try:
        meta = json.loads(p.read_text())
        frames = meta if isinstance(meta, list) else (meta.get("frames") or meta.get("cameras") or [])
        return bool(frames) and all(("transform_matrix" in f or "c2w" in f) for f in frames[:1])
    except (OSError, ValueError, AttributeError):
        return False


def find_cameras(path: Path) -> Path | None:
    """A cameras file the viewer can read: cameras.json next to the output (or up to 3 folders above), else
    <source_path of cfg_args>/transforms_test.json. None when there is none (not an error)."""
    d = Path(path).resolve()
    d = d if d.is_dir() else d.parent
    for cur in [d, *list(d.parents)[:3]]:
        if (cur / "cameras.json").is_file() and _readable_cameras(cur / "cameras.json"):
            return cur / "cameras.json"
    for cur in [d, *list(d.parents)[:4]]:
        cfg = cur / "cfg_args"
        if cfg.is_file():
            m = re.search(r"source_path=(['\"])(.*?)\1", cfg.read_text(errors="replace"))
            if m:
                tj = Path(m.group(2))
                tj = tj if tj.is_absolute() else cur / tj
                if (tj / "transforms_test.json").is_file() and _readable_cameras(tj / "transforms_test.json"):
                    return tj / "transforms_test.json"
            return None
    return None


def nerfies_scene(path: Path) -> Path | None:
    """The dataset folder named by source_path in cfg_args when it has the Nerfies / HyperNeRF layout
    (camera/<id>.json, metadata.json, rgb/<N>x/), else None."""
    d = Path(path).resolve()
    d = d if d.is_dir() else d.parent
    for cur in [d, *list(d.parents)[:4]]:
        cfg = cur / "cfg_args"
        if cfg.is_file():
            m = re.search(r"source_path=(['\"])(.*?)\1", cfg.read_text(errors="replace"))
            if not m:
                return None
            src = Path(m.group(2))
            src = src if src.is_absolute() else cur / src
            ok = (src / "camera").is_dir() and (src / "metadata.json").is_file() and (src / "rgb").is_dir()
            return src if ok else None
    return None


def write_nerfies_cameras(scene: Path, out: Path, raw: bool) -> int:
    """Writes the cameras of a Nerfies / HyperNeRF dataset as a cameras.json viewer.load_cameras reads; returns how many.

    Held-out frames (dataset.json val_ids) when there are any, else every frame. Camera maths as in the methods' loaders
    (and tests/check_real.py::load_nerfies_val_set): w2c rotation = orientation, focal = focal_length / N for the rgb/<N>x
    images (the smallest ones present), principal point at the image centre, no lens distortion.
    raw=False (Deformable-3DGS, SC-GS): position = (position - center) * scale of scene.json, time = time_id / max.
    raw=True (hustvl 4DGaussians): positions as stored, time = warp_id / max."""
    meta = json.loads((scene / "metadata.json").read_text())
    ids = []
    if (scene / "dataset.json").is_file():
        ds = json.loads((scene / "dataset.json").read_text())
        ids = ds.get("val_ids") or ds.get("ids") or []
    ids = [i for i in (ids or sorted(meta)) if (scene / "camera" / f"{i}.json").is_file() and i in meta]
    factors = sorted(int(f.name[:-1]) for f in (scene / "rgb").iterdir() if f.is_dir() and re.fullmatch(r"\d+x", f.name))
    if not ids or not factors:
        return 0
    n = factors[-1]
    centre, scale = np.zeros(3), 1.0
    if not raw:
        sc = json.loads((scene / "scene.json").read_text())
        centre, scale = np.array(sc["center"], float), float(sc["scale"])
    tkey = "warp_id" if raw else "time_id"
    tmax = max(max(m[tkey] for m in meta.values()), 1)
    frames = []
    for i in ids:
        cam = json.loads((scene / "camera" / f"{i}.json").read_text())
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(cam["orientation"], float).T
        c2w[:3, 3] = (np.array(cam["position"], float) - centre) * scale
        img = scene / "rgb" / f"{n}x" / f"{i}.png"
        fr = dict(name=i, c2w=c2w.tolist(), fx=float(cam["focal_length"]) / n, time=meta[i][tkey] / tmax)
        if img.is_file():
            fr["image"] = str(img)
        else:
            fr["w"], fr["h"] = (int(round(v / n)) for v in cam["image_size"])
        frames.append(fr)
    out.write_text(json.dumps(dict(convention="opencv", source=str(scene), frames=frames)))
    return len(frames)


def _frame_plys(folder: Path) -> Path | None:
    """The folder that holds the per-frame ply files (the folder itself, or hustvl's gaussian_pertimestamp)."""
    for d in (folder, folder / "gaussian_pertimestamp"):
        plys = sorted(d.glob("*.ply")) if d.is_dir() else []
        if len(plys) >= 2:
            return d
    return None


def detect(path: Path) -> dict:
    """-> dict(kind, converter (script or None), src (what the converter reads), extra (converter args)).
    Raises NotRecognised with the list of checks made."""
    path = Path(path)
    checked: list[str] = []
    if not path.exists():
        raise NotRecognised(f"{path} does not exist")

    def no(msg):
        checked.append(msg)

    dn = ["--t0", "0", "--t1", "1"]
    if path.is_file():
        suf = path.suffix.lower()
        if suf == ".npz":
            keys = _npz_keys(path)
            if all(k in keys for k in COMMON_KEYS):
                return dict(kind="common", converter=None, src=path, extra=[])
            if "means3D" in keys and "unnorm_rotations" in keys:
                return dict(kind="dynamic3dgs", converter="convert_per_frame.py", src=path, extra=[])
            no(f"{path.name}: .npz with keys {sorted(keys)[:8]}; need {COMMON_KEYS} (common format) "
               "or means3D+unnorm_rotations (Dynamic 3D Gaussians params.npz)")
        elif suf == ".pth" and path.name.startswith("chkpnt"):
            return dict(kind="native4d", converter="convert_native4d.py", src=path,
                        extra=dn if _is_dnerf(path) else [])
        elif suf in (".ckpt", ".pth"):
            kind = _ckpt_kind(path)
            if kind is None:   # keys not readable or not known: the names the methods save under
                if suf == ".ckpt":
                    kind = "4drotor" if re.fullmatch(r"step-\d+\.ckpt", path.name) else "shape_of_motion"
                elif path.name.startswith("photometric_d_model"):
                    kind = "mosca"
            if kind is not None:
                return dict(kind=kind, converter=CONVERTERS[kind], src=path, extra=[])
            if path.name.startswith("photometric_s_model"):
                no(f"{path.name}: MoSca's static background (StaticGaussian), which convert_mosca.py does not read; "
                   "give the dynamic photometric_d_model_*.pth (or the log folder)")
            else:
                no(f"{path.name}: .pth that is not chkpnt*.pth (native 4DGS) and has no MoSca keys (scf._node_xyz, attach_ind)")
        elif suf == ".ply":
            names = _ply_names(path)
            if "trbf_center" in names and "motion_0" in names:
                return dict(kind="spacetime", converter="convert_spacetime.py", src=path, extra=[])
            hint = ("an E-D3DGS ply (embedding_* columns) needs its model folder (point_cloud/ + cfg_args)"
                    if any(n.startswith("embedding_") for n in names) else
                    "a plain or SC-GS/Deformable-3DGS ply needs its model folder (point_cloud/ + deform/)")
            no(f"{path.name}: ply without trbf_center/motion_0 (not Spacetime Gaussians); {hint}")
        else:
            no(f"{path.name}: suffix {suf!r} is not .npz, .pth, .ckpt or .ply")
    else:
        ck = path / "checkpoints" / "last.ckpt"
        pc, de = path / "point_cloud", path / "deform"
        if ck.is_file():
            return detect(ck)
        no(f"{ck} missing (Shape of Motion)")
        steps = _nerfstudio_ckpts(path)
        if steps:
            return detect(steps[-1])   # the last step; the .ckpt rule above decides (4D-Rotor GS by its keys or name)
        no(f"{path}: no nerfstudio_models/step-*.ckpt (4D-Rotor GS)")
        dyn = sorted(path.glob("photometric_d_model_*.pth"), key=lambda f: (f.stat().st_mtime, f.name))
        if dyn:
            return dict(kind="mosca", converter="convert_mosca.py", src=dyn[-1], extra=[])
        no(f"{path}: no photometric_d_model_*.pth (MoSca)")
        if pc.is_dir() and de.is_dir():
            ply = _last_iteration(pc, "point_cloud.ply")
            if ply is None:
                no(f"{pc}/iteration_*/point_cloud.ply missing (Deformable-3DGS / SC-GS)")
            else:
                names = _ply_names(ply)
                if any(n.startswith("fea_") for n in names):
                    return dict(kind="scgs", converter="convert_scgs.py", src=path,
                                extra=dn if _is_dnerf(path) else [])
                return dict(kind="deformable3dgs", converter="convert_deformable3dgs.py", src=path,
                            extra=dn if _is_dnerf(path) else [])
        else:
            no(f"{path}: not both point_cloud/ and deform/ (Deformable-3DGS / SC-GS)")
        if pc.is_dir():
            ply = _last_iteration(pc, "point_cloud.ply")
            if ply is not None and (ply.parent / "deformation.pth").is_file() and not de.is_dir():
                if any(n.startswith("embedding_") for n in _ply_names(ply)):   # per-Gaussian embeddings: E-D3DGS
                    return dict(kind="ed3dgs", converter="convert_ed3dgs.py", src=path, extra=[])
                return dict(kind="4dgaussians", converter="convert_4dgaussians.py", src=path,
                            extra=dn if _is_dnerf(path) else [])
            if ply is not None and "trbf_center" in _ply_names(ply):
                return dict(kind="spacetime", converter="convert_spacetime.py", src=ply, extra=[])
            no(f"{pc}/iteration_*/ has no deformation.pth (4DGaussians, E-D3DGS) and the ply has no trbf_center (Spacetime Gaussians)")
        for name in ("params.npz",):
            if (path / name).is_file():
                return detect(path / name)
        fr = _frame_plys(path)
        if fr is not None:
            if "trbf_center" in _ply_names(sorted(fr.glob("*.ply"))[0]):
                no(f"{fr}: ply files have trbf_center (Spacetime Gaussians, not per-frame)")
            else:
                return dict(kind="per_frame", converter="convert_per_frame.py", src=fr, extra=[])
        else:
            no(f"{path}: fewer than 2 .ply files (also looked in gaussian_pertimestamp/) for per-frame")
        ckpts = sorted(path.glob("chkpnt*.pth"))
        if ckpts:
            best = next((c for c in ckpts if c.name == "chkpnt_best.pth"), ckpts[-1])
            return dict(kind="native4d", converter="convert_native4d.py", src=best,
                        extra=dn if _is_dnerf(best) else [])
        no(f"{path}: no chkpnt*.pth (native 4DGS)")
    raise NotRecognised("\n".join("  - " + c for c in checked))


def _newest_mtime(p: Path) -> float:
    if p.is_file():
        return p.stat().st_mtime
    return max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()), default=p.stat().st_mtime)


def cache_path(arg: Path, info: dict) -> Path:
    arg = arg.resolve()
    if arg.is_file():
        return arg.with_name(arg.stem + ".gs4d.npz")
    return arg.parent / (arg.name + ".gs4d.npz")


def is_fresh(cache: Path, src: Path, converter: Path | None = None) -> bool:
    """The cache is newer than the output and, when given, than the converter script (a fixed converter converts again)."""
    if not cache.is_file():
        return False
    newest = max(_newest_mtime(src), converter.stat().st_mtime if converter is not None and converter.is_file() else 0.0)
    return cache.stat().st_mtime >= newest


COMPACT_OVER = 1 << 30   # estimated size above which --compact is passed to the converter
PLY_BYTES = 248          # one Gaussian in a 3DGS ply with SH degree 3 (62 float32)
OUT_BYTES = 40           # one Gaussian at one keyframe in the plain format (see README)
SH_BYTES = 192           # SH degree 3 of one Gaussian as float32 (16 x 3 x 4), what --sh adds


def _size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def estimate_bytes(info: dict, convert_args=()) -> int:
    """Rough size of the plain .npz the converter would write (an upper bound: static Gaussians and
    --min-opacity make it smaller). Per-frame input is already N*K frames; the others are one model times K."""
    src, kind = Path(info["src"]), info["kind"]
    if kind in ("dynamic3dgs", "per_frame"):
        return _size(src) * OUT_BYTES // (PLY_BYTES if kind == "per_frame" else OUT_BYTES)
    args = list(convert_args)
    k = int(args[args.index("--frames") + 1]) if "--frames" in args and args.index("--frames") + 1 < len(args) else 60
    sh = "--sh" in args
    if kind == "shape_of_motion":
        return _size(src) // 300 * 200 * OUT_BYTES   # ~300 B per Gaussian with optimiser state, assume ~200 frames
    if kind == "native4d":   # checkpoint holds ~400 bytes per Gaussian; --sh stores SH at every keyframe
        return _size(src) // 400 * k * (OUT_BYTES + (SH_BYTES if sh else 0))
    if kind == "4drotor":    # 65 float32 per Gaussian plus Adam's two moments (~780 B); only means and opacities change
        return _size(src) // 780 * (k * 16 + 228)   # per keyframe (16 B); covariance, colour, SH are written once (228 B)
    if kind == "mosca":      # ~100 B per Gaussian in the state dict (scaffold included); ~200 frames assumed, as for
        try:                 # Shape of Motion, one keyframe per --stride frames
            stride = max(int(args[args.index("--stride") + 1]), 1)
        except (ValueError, IndexError):
            stride = 1
        return _size(src) // 100 * max(200 // stride, 1) * OUT_BYTES
    if kind == "ed3dgs":     # the deltas change every field at every keyframe, the SH too with --sh
        ply = _last_iteration(src / "point_cloud", "point_cloud.ply")
        n = _size(ply) // max(4 * len(_ply_names(ply)), 1) if ply is not None else 0   # float32 columns, embeddings included
        return n * k * (OUT_BYTES + (SH_BYTES if sh else 0))
    ply = _last_iteration(src / "point_cloud", "point_cloud.ply") if src.is_dir() else src
    n = _size(ply) // PLY_BYTES
    return n * k * OUT_BYTES + (n * SH_BYTES if sh else 0)   # the other converters store SH once per Gaussian


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Any other argument is passed to viewer.py (--port, --renderer, ...).")
    ap.add_argument("path", help="output of a 4DGS method (file or folder)")
    ap.add_argument("--reconvert", action="store_true", help="convert again even if <name>.gs4d.npz is up to date")
    ap.add_argument("--dry-run", action="store_true", help="print what would be run, run nothing")
    ap.add_argument("--convert-args", default="", help='extra arguments for the converter, e.g. "--stride 4"')
    a, viewer_args = ap.parse_known_args(argv)
    path = Path(a.path)

    try:
        info = detect(path)
    except NotRecognised as e:
        print(f"gs4d.open: cannot tell what {path} is. Checked:\n{e}", file=sys.stderr)
        return 2

    if info["converter"] is None:
        target = info["src"]
        cmd_conv = None
    else:
        target = cache_path(path, info)
        fresh = is_fresh(target, info["src"], ROOT / info["converter"]) and not a.reconvert
        conv_args = a.convert_args.split()
        cmd_conv = None if fresh else [sys.executable, str(ROOT / info["converter"]), str(info["src"]),
                                       "-o", str(target), *info["extra"], *conv_args]
        if cmd_conv and "--compact" not in conv_args:
            est = estimate_bytes(info, [*info["extra"], *conv_args])
            if est > COMPACT_OVER:
                print(f"gs4d.open: estimated file {est / 2**30:.1f} GB > 1 GB, converting with --compact", flush=True)
                cmd_conv.append("--compact")
    # with --compare the cameras are still passed: the viewer shows render A | image | render B (WP29)
    given = any(x == "--cameras" or x.startswith("--cameras=") for x in viewer_args)
    cams = None if given else find_cameras(path)
    if cams is None and not given and info["kind"] in ("4dgaussians", "deformable3dgs", "scgs"):
        scene = nerfies_scene(path)   # HyperNeRF layout: no camera file to pass on, so one is written next to the cache
        if scene is not None:
            cams = target.with_name(target.name.replace(".gs4d.npz", "") + ".gs4d.cameras.json")
            if not a.dry_run:
                try:
                    if not write_nerfies_cameras(scene, cams, raw=info["kind"] == "4dgaussians"):
                        cams = None
                except (OSError, ValueError, KeyError) as e:
                    print(f"gs4d.open: could not read the cameras of {scene}: {e!r}", flush=True)
                    cams = None
    if cams is not None:
        viewer_args = [*viewer_args, "--cameras", str(cams)]
    cmd_view = [sys.executable, str(ROOT / "viewer.py"), str(target), *viewer_args]

    print(f"gs4d.open: {info['kind']} -> {info['converter'] or 'no conversion'}", flush=True)
    if a.dry_run:
        print("converter:", " ".join(cmd_conv) if cmd_conv else "(skipped: " + ("not needed" if info["converter"] is None else "cache is up to date") + ")")
        print("viewer:   ", " ".join(cmd_view))
        return 0
    if cmd_conv:
        tmp = target.with_name(target.stem + ".partial.npz")
        cmd_conv[cmd_conv.index("-o") + 1] = str(tmp)
        r = subprocess.run(cmd_conv)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            print(f"gs4d.open: {info['converter']} failed (exit {r.returncode})", file=sys.stderr)
            return r.returncode
        tmp.replace(target)
    else:
        print(f"gs4d.open: using {target}", flush=True)
    return subprocess.call(cmd_view)


if __name__ == "__main__":
    sys.exit(main())
