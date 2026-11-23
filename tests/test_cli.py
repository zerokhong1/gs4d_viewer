#!/usr/bin/env python3
"""The command-line entry points fail with one line that says what is wrong, not with a traceback, and Ctrl+C on
`python -m gs4d.open` stops the viewer quietly (CPU only; starts one viewer on a free local port).

    python tests/test_cli.py
"""
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args, **kw):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False, **kw)


def one_line_error(r, *words):
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "Traceback" not in out, out[-1500:]
    assert all(w in out for w in words), (words, out[-1500:])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_bad_inputs(tmp):
    other = os.path.join(tmp, "other.npz")
    np.savez(other, a=np.zeros(3))
    text = os.path.join(tmp, "notes.txt")
    with open(text, "w") as f:
        f.write("not an npz")
    one_line_error(run("viewer.py", os.path.join(tmp, "missing.npz")), "No such file")
    one_line_error(run("viewer.py", other), "not a sequence in the common format", "gs4d.open")
    one_line_error(run("viewer.py", text), "not an .npz file")
    one_line_error(run("-m", "gs4d", other, os.path.join(tmp, "out.npz")), "not a sequence in the common format")
    one_line_error(run("-m", "gs4d.open", text), "cannot tell what")
    print("ok  viewer.py, python -m gs4d and python -m gs4d.open refuse wrong input with a message, no traceback")


def test_ctrl_c(tmp):
    if os.name == "nt":   # process groups, SIGINT delivery and preexec_fn are POSIX-only
        print("skip  Ctrl+C check: POSIX only")
        return
    toy = os.path.join(tmp, "toy.npz")
    subprocess.run([sys.executable, "make_toy.py", "-o", toy, "--frames", "8", "--points", "300"], cwd=ROOT, check=True,
                   capture_output=True)
    port = free_port()
    # what a terminal does on Ctrl+C: SIGINT to the whole foreground process group (here: gs4d.open and its viewer)
    p = subprocess.Popen([sys.executable, "-m", "gs4d.open", toy, "--port", str(port), "--renderer", "client"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        deadline = time.time() + 60
        while time.time() < deadline:   # wait until the viewer serves the page (a real HTTP request: a bare TCP
            try:                        # connection makes the websocket server log a failed handshake)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            raise AssertionError("viewer did not start")
        time.sleep(1.0)
        os.killpg(p.pid, signal.SIGINT)
        out, _ = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert "Traceback" not in out, out[-2000:]
    assert p.returncode in (0, 130, -signal.SIGINT), (p.returncode, out[-2000:])
    print(f"ok  Ctrl+C on python -m gs4d.open: viewer and launcher stop without a traceback (exit {p.returncode})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_bad_inputs(d)
        test_ctrl_c(d)
    print("all checks passed")
