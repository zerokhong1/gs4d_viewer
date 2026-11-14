"""Open the output of a 4DGS method directly: recognise it, convert it, start the viewer.

    python -m gs4d.open <output_path> [--reconvert] [--dry-run] [viewer.py arguments]

Recognised (checked in this order, the first match wins; nothing is guessed):
    *.npz in the common format (times/means/covs/opacities/colors)   -> opened as it is
    params.npz (Dynamic 3D Gaussians) or a folder of per-frame .ply  -> convert_per_frame.py
    chkpnt*.pth (native 4DGS)                                        -> convert_native4d.py
    last.ckpt, or a folder with checkpoints/last.ckpt (Shape of Motion) -> convert_shape_of_motion.py
    folder with point_cloud/ and deform/                             -> convert_scgs.py when the ply has
                                                                        fea_* columns, else convert_deformable3dgs.py
    point_cloud.ply with trbf_center (Spacetime Gaussians), or a
    folder holding point_cloud/iteration_*/point_cloud.ply of that kind -> convert_spacetime.py
The converted file is <name>.gs4d.npz next to the output and is reused while the output is unchanged.
D-NeRF scenes (cfg_args next to the output says dnerf / is_blender) get the time range [0, 1].
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
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
        elif path.name == "last.ckpt" or suf == ".ckpt":
            return dict(kind="shape_of_motion", converter="convert_shape_of_motion.py", src=path, extra=[])
        elif suf == ".ply":
            names = _ply_names(path)
            if "trbf_center" in names and "motion_0" in names:
                return dict(kind="spacetime", converter="convert_spacetime.py", src=path, extra=[])
            no(f"{path.name}: ply without trbf_center/motion_0 (not Spacetime Gaussians); a plain or "
               "SC-GS/Deformable-3DGS ply needs its model folder (point_cloud/ + deform/)")
        else:
            no(f"{path.name}: suffix {suf!r} is not .npz, chkpnt*.pth, *.ckpt or .ply")
    else:
        ck = path / "checkpoints" / "last.ckpt"
        pc, de = path / "point_cloud", path / "deform"
        if ck.is_file():
            return dict(kind="shape_of_motion", converter="convert_shape_of_motion.py", src=ck, extra=[])
        no(f"{ck} missing (Shape of Motion)")
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
            if ply is not None and "trbf_center" in _ply_names(ply):
                return dict(kind="spacetime", converter="convert_spacetime.py", src=ply, extra=[])
            no(f"{pc}/iteration_*/point_cloud.ply has no trbf_center (Spacetime Gaussians)")
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


def is_fresh(cache: Path, src: Path) -> bool:
    return cache.is_file() and cache.stat().st_mtime >= _newest_mtime(src)


COMPACT_OVER = 1 << 30   # estimated size above which --compact is passed to the converter
PLY_BYTES = 248          # one Gaussian in a 3DGS ply with SH degree 3 (62 float32)
OUT_BYTES = 40           # one Gaussian at one keyframe in the plain format (see README)


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
    if kind == "shape_of_motion":
        return _size(src) // 300 * 200 * OUT_BYTES   # ~300 B per Gaussian with optimiser state, assume ~200 frames
    if kind == "native4d":
        return _size(src) * k * OUT_BYTES // 400                       # checkpoint holds ~400 bytes per Gaussian
    ply = _last_iteration(src / "point_cloud", "point_cloud.ply") if src.is_dir() else src
    return _size(ply) // PLY_BYTES * k * OUT_BYTES


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
        fresh = is_fresh(target, info["src"]) and not a.reconvert
        conv_args = a.convert_args.split()
        cmd_conv = None if fresh else [sys.executable, str(ROOT / info["converter"]), str(info["src"]),
                                       "-o", str(target), *info["extra"], *conv_args]
        if cmd_conv and "--compact" not in conv_args:
            est = estimate_bytes(info, [*info["extra"], *conv_args])
            if est > COMPACT_OVER:
                print(f"gs4d.open: estimated file {est / 2**30:.1f} GB > 1 GB, converting with --compact", flush=True)
                cmd_conv.append("--compact")
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
