#!/usr/bin/env python3
"""Runs every tests/test_*.py and tests/check_*.py, one at a time, and prints a table.

    python tests/run_all.py            # everything
    python tests/run_all.py sh static  # only files whose name contains one of the words
    python tests/run_all.py --list     # what would run, and what is missing for each

A file is "skipped" (not failed) when the machine lacks what it needs: CUDA, Chrome (Playwright), torch, or the
real data (../data_real, /home/cont/gs4d_train). Exit code 1 if any file failed. Use the same Python as for the
viewer (.venv); one GPU and 15 GB RAM are assumed, so files run sequentially.
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TIMEOUT = 1800
# What a file needs beyond numpy; anything not listed needs nothing.
NEEDS = {
    "check_gsplat.py": ("cuda", "chrome"),
    "check_camera.py": ("chrome",),
    "check_compare.py": ("chrome",),
    "check_points.py": ("chrome",),
    "check_path.py": ("chrome",),
    "check_trails.py": ("chrome",),
    "check_cameras.py": ("cuda", "data", "chrome"),
    "test_scgs_hash.py": ("torch",),
    "test_4drotor.py": ("torch",),
    "check_real.py": ("cuda", "data"),
    "test_deformable3dgs.py": ("torch",),
    "test_scgs.py": ("torch",),
    "test_shape_of_motion.py": ("torch",),
}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def probe():
    ok = {"torch": has_module("torch")}
    cuda = False
    if ok["torch"]:
        try:
            import torch
            cuda = bool(torch.cuda.is_available()) and has_module("gsplat")
        except Exception:
            pass
    ok["cuda"] = cuda
    chrome = False
    if has_module("playwright"):
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                chrome = Path(p.chromium.executable_path).exists()
        except Exception:
            pass
        if not chrome:
            chrome = any(Path(x).exists() for x in ("/usr/bin/google-chrome", "/usr/bin/chromium"))
    ok["chrome"] = chrome
    root = str(ROOT).split(f"{os.sep}.claude{os.sep}worktrees{os.sep}")[0]   # same rule as check_real.py
    data = os.environ.get("GS4D_DATA") or os.path.join(os.path.dirname(root), "data_real")
    train = os.environ.get("GS4D_TRAIN") or "/home/cont/gs4d_train"
    ok["data"] = os.path.isdir(data) and os.path.isdir(train)
    return ok


REASON = {"torch": "no torch", "cuda": "no CUDA (or gsplat)", "chrome": "no Chrome/Playwright", "data": "no data (../data_real, gs4d_train)"}


def main(argv):
    words = [a for a in argv if not a.startswith("--")]
    files = sorted(list(HERE.glob("test_*.py")) + list(HERE.glob("check_*.py")))
    files = [f for f in files if not words or any(w in f.name for w in words)]
    have = probe()
    rows, failed = [], 0
    for f in files:
        missing = [REASON[n] for n in NEEDS.get(f.name, ()) if not have[n]]
        if "--list" in argv:
            print(f"{f.name:28s} needs {NEEDS.get(f.name, ())} {'-> skip: ' + ', '.join(missing) if missing else ''}")
            continue
        if missing:
            rows.append((f.name, "skipped", "", ", ".join(missing)))
            continue
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, str(f)], cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
            out, code = (r.stdout + r.stderr), r.returncode
        except subprocess.TimeoutExpired:
            out, code = f"timeout after {TIMEOUT}s", 1
        dt = f"{time.time() - t0:.1f}s"
        if code == 0:
            rows.append((f.name, "pass", dt, ""))
        else:
            failed += 1
            rows.append((f.name, "FAILED", dt, f"exit {code}"))
            print(f"--- {f.name} (exit {code}) last output ---\n" + "\n".join(out.strip().splitlines()[-25:]) + "\n", flush=True)
    if "--list" in argv:
        return 0
    w = max((len(r[0]) for r in rows), default=10)
    print(f"{'file':{w}s}  {'result':8s}  {'time':>7s}  note")
    for n, res, dt, note in rows:
        print(f"{n:{w}s}  {res:8s}  {dt:>7s}  {note}")
    c = {k: sum(r[1] == k for r in rows) for k in ("pass", "skipped", "FAILED")}
    print(f"\n{c['pass']} passed, {c['skipped']} skipped, {c['FAILED']} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
