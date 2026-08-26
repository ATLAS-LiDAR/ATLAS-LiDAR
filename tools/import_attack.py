#!/usr/bin/env python3
"""Convert .atlasattack bundles into the pcdet split format ATLAS already evals.

The two harnesses use DIFFERENT layouts, and getting it wrong fails with an
opaque "'list' object has no attribute 'get'":

  kitti     eval_matrix_kitti.eval_asr   globs *.pkl and loads ONE FRAME per file
  nuscenes  eval_matrix_nuscenes.eval_asr loads `{scene}_d.pkl` as a LIST of frames

Either way the harness touches exactly three fields:

    fr["points"][:, :4]      the (possibly perturbed) cloud
    fr["frame_id"]           KITTI lidar_idx, e.g. "000001"
    fr.get("spoof_gt")       the phantom box [x,y,z,dx,dy,dz,heading]

so a sandbox-generated attack can be scored by the existing harness with no
changes to it at all:

    python3 tools/import_attack.py out/*.atlasattack --out-dir splits/sandbox
    # then, in the pcdet env:
    python3 eval_matrix_kitti.py --split_dir splits/sandbox --attack_type inject

`--verify` re-reads what was written and asserts those fields are present and
correctly typed, which is the check that matters before spending GPU time.
"""
import argparse
import glob
import json
import os
import pickle as pkl
import struct
import sys
from collections import defaultdict

import numpy as np

MAGIC = b"ATLATTK1"


def read_bundle(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:8] != MAGIC:
        raise SystemExit("%s: bad magic %r (not an .atlasattack bundle)" % (path, raw[:8]))
    hlen = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + hlen].decode("utf-8"))
    off = header["payloadOffset"]
    n = header["nPoints"]
    pts = np.frombuffer(raw, dtype=np.float32, count=n * 4, offset=off).reshape(n, 4)
    return header, pts


def frame_dict(header, pts):
    """One entry of a `{segment}_d.pkl` list."""
    fr = {
        "points": np.ascontiguousarray(pts, dtype=np.float32),
        "frame_id": header["frameId"],
        "scheduled_attack": bool(header.get("scheduledAttack", True)),
        # provenance, ignored by the harness but invaluable when reading results
        "atlas_sandbox": {
            "attack": header.get("attack"),
            "spoofer": header.get("spoofer"),
            "stats": header.get("stats"),
            "sensor": header.get("sensor"),
            "sensorSpec": header.get("sensorSpec"),
            "sceneSlug": header.get("sceneSlug"),
        },
    }
    if header.get("spoofGt") is not None:
        fr["spoof_gt"] = np.asarray(header["spoofGt"], dtype=np.float32)
    # nuScenes-style harnesses also read metadata.token
    fr["metadata"] = {"token": header["frameId"]}
    return fr


def verify(dpkl_path, attack_type):
    seg = pkl.load(open(dpkl_path, "rb"))
    if not isinstance(seg, list) or not seg:
        raise SystemExit("%s: expected a non-empty list of frames" % dpkl_path)
    problems = []
    n_with_gt = 0
    for i, fr in enumerate(seg):
        for k in ("points", "frame_id"):
            if k not in fr:
                problems.append("frame %d missing %r" % (i, k))
        p = fr.get("points")
        if p is None:
            continue
        if not isinstance(p, np.ndarray) or p.dtype != np.float32:
            problems.append("frame %d points must be a float32 ndarray, got %r" % (i, getattr(p, "dtype", type(p))))
        if p.ndim != 2 or p.shape[1] < 4:
            problems.append("frame %d points must be (N,>=4), got %r" % (i, getattr(p, "shape", None)))
        if not np.isfinite(p).all():
            problems.append("frame %d points contain non-finite values" % i)
        g = fr.get("spoof_gt")
        if g is not None:
            n_with_gt += 1
            if np.asarray(g).shape[0] < 7:
                problems.append("frame %d spoof_gt must have >= 7 elements" % i)

    if n_with_gt == 0:
        # eval_asr does `if spoof_gt is None: continue`, so a split with no
        # target box is silently skipped and reports ASR = nan. Say which cause
        # applies, using the mode each bundle recorded rather than the CLI flag.
        modes = {(fr.get("atlas_sandbox") or {}).get("attack", {}).get("mode") for fr in seg}
        infeasible = sum(
            1 for fr in seg
            if ((fr.get("atlas_sandbox") or {}).get("stats") or {}).get("feasible") is False
        )
        if modes == {"remove"}:
            problems.append(
                "removal split carries no spoof_gt: a removal attack has no phantom, so "
                "the target must be the victim's own GT box - re-export with "
                "targetBoxIndex set to the object being erased"
            )
        elif infeasible:
            problems.append(
                "no frame carries spoof_gt, and %d/%d frame(s) recorded feasible=false: "
                "the attack was physically infeasible (no reachable cells), so no phantom "
                "landed. Move the spoofer or raise its power and re-export."
                % (infeasible, len(seg))
            )
        else:
            problems.append(
                "no frame carries spoof_gt, so eval_asr would skip every frame and "
                "report ASR = nan"
            )
    if problems:
        print("VERIFY FAILED for %s:" % dpkl_path)
        for p in problems:
            print("  -", p)
        return False
    print(
        "verified %s: %d frames, %d with spoof_gt, points float32 (N,4)"
        % (os.path.basename(dpkl_path), len(seg), n_with_gt)
    )
    return True


def verify_kitti(paths, attack_type):
    """Verify the one-frame-per-pkl layout eval_matrix_kitti expects."""
    problems = []
    n_with_gt = 0
    for p in paths:
        fr = pkl.load(open(p, "rb"))
        if not isinstance(fr, dict):
            problems.append("%s: expected a single frame dict, got %s - "
                            "eval_matrix_kitti does fr.get('spoof_gt') and would "
                            "raise \"'list' object has no attribute 'get'\""
                            % (os.path.basename(p), type(fr).__name__))
            continue
        pts = fr.get("points")
        if pts is None or pts.dtype != np.float32 or pts.ndim != 2 or pts.shape[1] < 4:
            problems.append("%s: points must be float32 (N,>=4)" % os.path.basename(p))
        if "frame_id" not in fr:
            problems.append("%s: missing frame_id" % os.path.basename(p))
        if fr.get("spoof_gt") is not None:
            n_with_gt += 1
    if n_with_gt == 0:
        problems.append("no file carries spoof_gt, so eval_asr counts zero "
                        "targets and reports ASR = nan")
    if problems:
        print("VERIFY FAILED:")
        for pr in problems:
            print("  -", pr)
        return False
    print("verified %d per-frame pkl(s): %d with spoof_gt, points float32 (N,4)"
          % (len(paths), n_with_gt))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+", help=".atlasattack files (globs ok)")
    ap.add_argument("--out-dir", default="splits/sandbox")
    ap.add_argument("--annos-dir", default=None, help="defaults to <out-dir>/../annos")
    ap.add_argument("--segment", default=None, help="segment name; default groups by scene slug")
    ap.add_argument("--attack-type", default="inject", choices=["inject", "removal"])
    ap.add_argument(
        "--layout", default="auto", choices=["auto", "kitti", "nuscenes"],
        help="kitti = one frame per .pkl (what eval_matrix_kitti globs); "
             "nuscenes = a list of frames in {scene}_d.pkl; auto picks from the "
             "bundle's dataset field",
    )
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    paths = []
    for b in a.bundles:
        paths.extend(sorted(glob.glob(b)) if any(c in b for c in "*?[") else [b])
    if not paths:
        raise SystemExit("no bundles matched")

    dataset_dir = a.out_dir
    annos_dir = a.annos_dir or os.path.join(os.path.dirname(os.path.abspath(a.out_dir)), "annos")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(annos_dir, exist_ok=True)

    segments = defaultdict(list)
    metas = defaultdict(list)
    datasets_seen = set()
    for p in paths:
        header, pts = read_bundle(p)
        datasets_seen.add(header.get("dataset"))
        seg = a.segment or header.get("sceneSlug") or "sandbox"
        segments[seg].append(frame_dict(header, pts))
        metas[seg].append(
            {
                "source_bundle": os.path.basename(p),
                "attack": header.get("attack"),
                "spoofer": header.get("spoofer"),
                "stats": header.get("stats"),
                "spoof_gt": header.get("spoofGt"),
            }
        )

    layout = a.layout
    if layout == "auto":
        datasets = {h for h in datasets_seen}
        layout = "kitti" if datasets == {"kitti"} else "nuscenes"
        print("layout: %s (auto-detected from dataset=%s)" % (layout, sorted(datasets)))

    ok = True
    for seg, frames in segments.items():
        apath = os.path.join(annos_dir, "%s_a.pkl" % seg)
        with open(apath, "wb") as f:
            pkl.dump({"segment": seg, "spoof_annos": metas[seg],
                      "source": "atlas-lidar sandbox export", "layout": layout}, f)
        total = sum(fr["points"].shape[0] for fr in frames)

        if layout == "kitti":
            # one frame per file: eval_matrix_kitti globs *.pkl and pkl.load()s
            # each into a single frame dict
            paths = []
            for i, fr in enumerate(frames):
                dpath = os.path.join(dataset_dir, "%s_%03d.pkl" % (seg, i))
                with open(dpath, "wb") as f:
                    pkl.dump(fr, f)
                paths.append(dpath)
            print("wrote %d per-frame pkl(s) for %s (%d points) and %s"
                  % (len(paths), seg, total, apath))
            if a.verify:
                ok = verify_kitti(paths, a.attack_type) and ok
        else:
            dpath = os.path.join(dataset_dir, "%s_d.pkl" % seg)
            with open(dpath, "wb") as f:
                pkl.dump(frames, f)
            print("wrote %s (%d frames, %d points) and %s"
                  % (dpath, len(frames), total, apath))
            if a.verify:
                ok = verify(dpath, a.attack_type) and ok

    if not ok:
        return 1
    print("\nScore it with the existing harness (in the pcdet env):")
    print("  python3 eval_matrix_kitti.py --split_dir %s --attack_type %s"
          % (dataset_dir, "inject" if a.attack_type == "inject" else "removal"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
