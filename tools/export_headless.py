#!/usr/bin/env python3
"""Batch-run attacks by driving the sandbox in headless Chrome.

The attack physics lives in JS. Reimplementing it in Python would give two
implementations that drift, so instead this drives dev/export.html headlessly and
reads the resulting bundle back out of the DOM. The JS core stays the single
source of truth for what an attack does.

    # one attack
    python3 tools/export_headless.py --scene kitti-open-road --mode inject \
        --phantom-range 30 --out out/

    # a sweep, from a JSON list of configs
    python3 tools/export_headless.py --configs sweep.json --out out/

Then convert to the pcdet split format the eval harness reads:

    python3 tools/import_attack.py out/*.atlasattack --out-dir splits/sandbox
"""
import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("no Chrome binary found (set CHROME=/path/to/chrome)")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep stdout to the export table
        pass


def serve(root, port):
    handler = partial(_QuietHandler, directory=root)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def element(dom, tag, ident):
    m = re.search(r'<%s[^>]*\bid="%s"[^>]*>(.*?)</%s>' % (tag, ident, tag), dom, re.S)
    return html.unescape(m.group(1)) if m else None


def run_one(chrome, port, cfg, timeout=180):
    url = "http://127.0.0.1:%d/dev/export.html?cfg=%s" % (
        port,
        urllib.parse.quote(json.dumps(cfg, separators=(",", ":"))),
    )
    profile = tempfile.mkdtemp()
    try:
        proc = subprocess.run(
            [
                chrome, "--headless", "--no-sandbox", "--disable-gpu",
                "--disable-dev-shm-usage", "--no-first-run",
                "--user-data-dir=" + profile,
                "--virtual-time-budget=120000",
                "--dump-dom", url,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    dom = proc.stdout
    if "EXPORT_DONE ok" not in dom:
        err = element(dom, "pre", "export-error") or dom[-800:]
        raise RuntimeError("export failed for cfg %r:\n%s" % (cfg, err))

    b64 = element(dom, "pre", "bundle-b64")
    meta = json.loads(element(dom, "pre", "bundle-meta"))
    return base64.b64decode(b64), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", help="JSON file: a list of config objects")
    ap.add_argument("--scene", default="kitti-open-road")
    ap.add_argument("--mode", default="inject", choices=["inject", "remove"])
    ap.add_argument("--phantom-range", type=float, default=30.0)
    ap.add_argument("--phantom-len", type=float, default=4.2)
    ap.add_argument("--shape", default="box", choices=["box", "wall"])
    ap.add_argument("--intensity-mode", default="range-conditioned")
    ap.add_argument("--attacker-range", type=float, default=0.2)
    ap.add_argument("--spoofer", default="cao2023")
    ap.add_argument("--spoofer-pos", default="25,6,0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="out")
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()

    if a.configs:
        configs = json.load(open(a.configs))
        if not isinstance(configs, list):
            raise SystemExit("--configs must contain a JSON list of config objects")
    else:
        configs = [
            {
                "scene": a.scene,
                "mode": a.mode,
                "shape": a.shape,
                "phantomRange": a.phantom_range,
                "phantomLen": a.phantom_len,
                "intensityMode": a.intensity_mode,
                "attackerRangeM": a.attacker_range,
                "spoofer": a.spoofer,
                "spooferPos": [float(v) for v in a.spoofer_pos.split(",")],
                "seed": a.seed,
            }
        ]

    chrome = os.environ.get("CHROME") or find_chrome()
    os.makedirs(a.out, exist_ok=True)
    httpd = serve(ROOT, a.port)
    written = []
    try:
        for i, cfg in enumerate(configs):
            data, meta = run_one(chrome, a.port, cfg)
            name = "%s_%s_%s_%03d.atlasattack" % (
                meta.get("scene", "frame"), meta.get("frameId", "x"), meta.get("mode", "x"), i
            )
            path = os.path.join(a.out, name)
            with open(path, "wb") as f:
                f.write(data)
            written.append(path)
            st = meta.get("stats", {})
            print(
                "%-52s %7d pts  feasible=%-5s cells=%-5s %s"
                % (
                    name,
                    meta.get("nPoints", 0),
                    meta.get("feasible"),
                    st.get("reachableCells"),
                    ("injected=%s suppressed=%s" % (st.get("injected"), st.get("suppressed")))
                    if meta.get("mode") == "inject"
                    else ("removed=%s replaced=%s" % (st.get("removed"), st.get("replaced"))),
                )
            )
    finally:
        httpd.shutdown()

    print("\nwrote %d bundle(s) to %s/" % (len(written), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
