#!/usr/bin/env python3
"""Cross-validate the sandbox against ATLAS's real functions on identical input.

Every fidelity claim so far rests on reading code and asserting it matches. That
method already produced eight errors, so this measures agreement instead: it runs
ATLAS's OWN functions - nuscenes_atlas_common.spoof_frame,
kitti_atlas_common.spoof_frame, removal.spoof_frame_ahfr_roadside - and the
sandbox, on the same exported frame and the same trace, and compares the integer
outcomes.

The comparison is made EXACT by removing randomness from both sides:

  * injection - the trace index, offset and yaw are fixed explicitly, so the
    placement is deterministic. With a single-return sensor P_TWO = 0, so the
    occlusion resolution is deterministic too.
  * removal   - p_remove = 1.0 and the frame-level scan gate is forced open, so
    every point in the sector is deleted and nothing is left to chance.

Intensity is left out of the comparison: it is drawn at random on both sides and
does not affect geometry.

    python3 tools/crossvalidate.py                 # default grid
    python3 tools/crossvalidate.py --frames kitti-open-road --verbose
"""
import argparse
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the widths ATLAS tabulates a removal probability for
ATLAS_AZ_WIDTHS = (10, 20, 30, 40, 50, 60)
PHM = os.environ.get("PHM_ROOT", os.path.expanduser("~/rsch/Polar-Hierarchical-Mamba"))
sys.path.insert(0, os.path.join(PHM, "atlas_generation"))


def _stub_plotting():
    """removal.py imports matplotlib and plotly at module level for its plotting
    helpers. We only call its physics, so stub them rather than requiring a
    plotting stack to be installed."""
    import types
    for name in ("matplotlib", "matplotlib.pyplot", "plotly",
                 "plotly.graph_objects", "plotly.io"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = lambda _n: (lambda *a, **k: None)
            sys.modules[name] = mod
    # package attributes so `import plotly.graph_objects as go` resolves
    sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]
    sys.modules["plotly"].io = sys.modules["plotly.io"]
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]


_stub_plotting()


class ZeroRng:
    """A generator whose every draw is the deterministic extreme.

    random() -> 0.0 opens the scan gate and passes every per-point removal draw;
    normal() -> 0.0 removes the aim jitter; lognormal -> the band median. That
    makes ATLAS's own code path deterministic without modifying it.
    """

    def random(self, size=None):
        return 0.0 if size is None else np.zeros(size, dtype=np.float64)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return 0.0 if size is None else np.zeros(size, dtype=np.float64)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        v = float(np.exp(mean))
        return v if size is None else np.full(size, v, dtype=np.float64)

    def choice(self, a, size=None, **kw):
        arr = np.asarray(a)
        return arr[0] if size is None else np.full(size, arr[0])

    def integers(self, low, high=None, **kw):
        return int(low)

    def uniform(self, low=0.0, high=1.0, **kw):
        return float(low)


# ---------------------------------------------------------------- asset readers
def read_atlasframe(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:8] == b"ATLFRAME", path
    hlen = struct.unpack("<I", raw[8:12])[0]
    h = json.loads(raw[12 : 12 + hlen])
    base = 12 + hlen
    q = h["posQuantum"]

    def block(name):
        spec = h["arrays"][name]
        off = base + spec["offset"]
        if spec["dtype"] == "int16":
            return np.frombuffer(raw, dtype="<i2", count=spec["count"], offset=off).astype(np.float64)
        return np.frombuffer(raw, dtype=np.uint8, count=spec["count"], offset=off).astype(np.float64)

    x = block("x") * q
    y = block("y") * q
    z = block("z") * q
    inten = block("intensity") * h["intensityScale"]
    return h, np.stack([x, y, z, inten], axis=1)


def read_atlastrace(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:8] == b"ATLTRACE", path
    hlen = struct.unpack("<I", raw[8:12])[0]
    h = json.loads(raw[12 : 12 + hlen])
    base = h["payloadOffset"]
    q = h["posQuantum"]

    def block(name):
        spec = h["arrays"][name]
        off = base + spec["offset"]
        if spec["dtype"] == "int16":
            return np.frombuffer(raw, dtype="<i2", count=spec["count"], offset=off).astype(np.float64)
        return np.frombuffer(raw, dtype=np.uint8, count=spec["count"], offset=off).astype(np.float64)

    x = block("x") * q
    y = block("y") * q
    z = block("z") * q
    inten = block("intensity") * h["intensityScale"]
    counts = h["counts"]
    prefix = np.concatenate([[0], np.cumsum(counts)])
    traces = []
    for i, n in enumerate(counts):
        s, e = int(prefix[i]), int(prefix[i] + n)
        traces.append({
            "points": np.stack([x[s:e], y[s:e], z[s:e], inten[s:e]], axis=1),
            "box": np.asarray(h["boxes"][i], dtype=np.float64),
        })
    return h, traces


# --------------------------------------------------------- ATLAS reference runs
def place_fixed(trace, offset_xy, yaw_new, common):
    """place_trace_relative's transform, with the random draws replaced by fixed
    values so both sides do exactly the same thing."""
    pts = trace["points"]
    box = trace["box"].copy()
    centered_xy = pts[:, :2] - box[:2]
    c, s = np.cos(-box[6]), np.sin(-box[6])
    R = np.array([[c, -s], [s, c]])
    rotated = centered_xy @ R.T

    c2, s2 = np.cos(yaw_new), np.sin(yaw_new)
    rotated = rotated @ np.array([[c2, -s2], [s2, c2]]).T
    planar = rotated + np.asarray(offset_xy, dtype=np.float64)

    box_new = box.copy()
    box_new[0] = offset_xy[0]
    box_new[1] = offset_xy[1]
    box_new[6] = yaw_new
    xyz = np.concatenate([planar, pts[:, 2:3]], axis=1)
    return xyz, pts[:, 3:4], box_new


def atlas_inject(dataset, scene_pts, trace, offset_xy, yaw_new):
    rng = ZeroRng()
    xyz, inten, box = place_fixed(trace, offset_xy, yaw_new, dataset)
    n_scene = scene_pts.shape[0]

    if dataset == "kitti":
        import kitti_atlas_common as K
        points = scene_pts.astype(np.float64)                 # (N,4)
        spoof = np.hstack([xyz, inten])                       # (M,4)
        fused, gt, n_r, n_k = K.spoof_frame(spoof, points, box, rng)
    else:
        import nuscenes_atlas_common as C
        lag = np.zeros((n_scene, 1))
        tim = np.zeros((n_scene, 1))
        points = np.hstack([scene_pts, tim, lag])             # (N,6)
        spoof = np.hstack([xyz, inten, np.zeros((xyz.shape[0], 1))])  # (M,5)
        fused, gt, n_r, n_k = C.spoof_frame(spoof, points, box, rng)

    scene_kept = fused.shape[0] - n_k
    return {
        "requested": int(n_r),
        "injected": int(n_k),
        "suppressed": int(n_scene - scene_kept),
        "placedChecksum": checksum_points(xyz),
    }


def atlas_remove(scene_pts, target_box, az_width, el_width=16.0):
    import removal
    rng = ZeroRng()
    pts = scene_pts.astype(np.float64)
    attacked, state, meta = removal.spoof_frame_ahfr_roadside(
        points=pts,
        target_box=np.asarray(target_box, dtype=np.float64),
        state=None,
        az_width_deg=az_width,
        el_width_deg=el_width,
        p_remove=1.0,
        center_jitter_std_deg=0.0,
        rng=rng,
    )
    return {
        "inSector": int(meta["n_points_in_sector"]),
        "removed": int(meta["n_points_removed"]),
        "azCenterDeg": float(np.rad2deg(meta["az_center_rad"])),
        "inactiveReason": meta.get("inactive_reason"),
    }


def checksum_points(xyz, quantum=0.005):
    """Order-independent-ish checksum of a placed phantom, so the two sides can be
    compared on placement before occlusion is blamed."""
    q = np.round(np.asarray(xyz)[:, :3] / quantum).astype(np.int64)
    h = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    return int(np.sum(h % 1000003) % (2**31))


# --------------------------------------------------------------- sandbox runner
class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(port):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(_Quiet, directory=ROOT))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def find_chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("no Chrome found")


def sandbox_run(chrome, port, cfgs):
    url = "http://127.0.0.1:%d/crossvalidate.html?cfg=%s" % (
        port, urllib.parse.quote(json.dumps(cfgs, separators=(",", ":")))
    )
    prof = tempfile.mkdtemp()
    try:
        out = subprocess.run(
            [chrome, "--headless", "--no-sandbox", "--disable-gpu",
             "--disable-dev-shm-usage", "--no-first-run", "--user-data-dir=" + prof,
             "--virtual-time-budget=120000", "--dump-dom", url],
            capture_output=True, text=True, timeout=300).stdout
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    m = re.search(r'<pre id="cv-out">(.*?)</pre>', out, re.S)
    if not m:
        err = re.search(r'<pre id="cv-error">(.*?)</pre>', out, re.S)
        raise SystemExit("sandbox run failed:\n" + (err.group(1) if err else out[-1500:]))
    import html as _html
    return json.loads(_html.unescape(m.group(1)))


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="kitti-open-road,kitti-dense-traffic,nuscenes-boston")
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(ROOT, "assets/frames/manifest.json")))
    by_slug = {m["slug"]: m for m in manifest}
    frames = [f for f in a.frames.split(",") if f in by_slug]

    # A grid wide enough that agreement is evidence rather than luck: several
    # traces (different point counts and shapes), offsets spanning near/far and
    # both sides of the ego axis, non-zero yaws, and every tabulated wedge width.
    cfgs = []
    for slug in frames:
        ds = by_slug[slug]["dataset"]
        for ti in (0, 5, 17, 42, 101):
            for off, yaw in (([15.0, 0.0], 0.0), ([18.0, 2.5], 0.1),
                             ([25.0, -4.0], -0.25), ([32.0, 5.5], 0.3),
                             ([12.0, -1.5], 1.2)):
                cfgs.append({"kind": "inject", "frame": slug, "dataset": ds,
                             "traceIdx": ti, "offsetXY": off, "yaw": yaw,
                             "difficulty": "medium"})
        for w in ATLAS_AZ_WIDTHS:
            for tgt in (0, 1, 2):
                cfgs.append({"kind": "remove", "frame": slug, "dataset": ds,
                             "azWidth": w, "targetIndex": tgt})

    print("cross-validating %d configurations against ATLAS's own functions\n" % len(cfgs))

    chrome = find_chrome()
    httpd = serve(a.port)
    try:
        js = sandbox_run(chrome, a.port, cfgs)
    finally:
        httpd.shutdown()

    caches = {}
    agree = disagree = skipped = 0
    failures = []

    for cfg, got in zip(cfgs, js):
        slug = cfg["frame"]
        if slug not in caches:
            fh, pts = read_atlasframe(os.path.join(ROOT, "assets/frames", by_slug[slug]["file"]))
            caches[slug] = (fh, pts)
        fh, pts = caches[slug]

        if got.get("error"):
            skipped += 1
            continue

        if cfg["kind"] == "inject":
            key = (cfg["dataset"], cfg["difficulty"])
            if key not in caches:
                caches[key] = read_atlastrace(os.path.join(
                    ROOT, "assets/traces", "%s-%s.atlastrace" % key))
            th, traces = caches[key]
            ref = atlas_inject(cfg["dataset"], pts, traces[cfg["traceIdx"]],
                               cfg["offsetXY"], cfg["yaw"])
            fields = ("requested", "injected", "suppressed", "placedChecksum")
            label = "inject %s trace=%d off=%s" % (slug, cfg["traceIdx"], cfg["offsetXY"])
        else:
            boxes = fh.get("boxes", [])
            if cfg["targetIndex"] >= len(boxes):
                skipped += 1
                continue
            b = boxes[cfg["targetIndex"]]
            tb = list(b["center"]) + list(b["size"]) + [b["yaw"]]
            ref = atlas_remove(pts, tb, cfg["azWidth"])
            fields = ("inSector", "removed")
            label = "remove %s az=%d target=%d" % (slug, cfg["azWidth"], cfg["targetIndex"])

        bad = [f for f in fields if ref.get(f) != got.get(f)]
        if bad:
            disagree += 1
            failures.append((label, {f: (ref.get(f), got.get(f)) for f in bad}))
        else:
            agree += 1
        if a.verbose:
            mark = "ok " if not bad else "DIFF"
            print("  %s %-52s %s" % (mark, label, {f: ref.get(f) for f in fields}))

    print("\n%d agree, %d disagree, %d skipped" % (agree, disagree, skipped))
    if failures:
        print("\ndisagreements (ATLAS, sandbox):")
        for label, d in failures[:20]:
            print("  %-52s %s" % (label, d))
        return 1
    print("\nExact agreement on every deterministic outcome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
