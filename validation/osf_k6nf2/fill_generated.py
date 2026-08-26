#!/usr/bin/env python3
"""Measure per-cell injection fill on the ACTUAL generated ATLAS splits.

This is stricter than measuring the trace library: `spoof_frame` ground-anchors
the trace, resamples intensity, and occlusion-filters points against the scene
before writing, so the phantom that reaches the detector is not the raw trace.
Here we recover exactly the points that were injected into each frame and
compute the same realizability statistics used for the real capture.

Recovery: `spoof_frame` concatenates [scene_kept, spoof_kept] in that order and
records n_spoof_k, so the injected points are the LAST n_spoof_k rows.
"""
import argparse
import glob
import os
import pickle as pkl
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SENSORS = {"hdl32": (32, 41.33, 0.3316), "hdl64": (64, 26.9, 0.1728)}


def phantom_stats(xyz, nbeam, vfov, azres):
    r = np.linalg.norm(xyz, axis=1)
    az = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    if az.max() - az.min() > 180:
        az = np.where(az < 0, az + 360, az)
    el = np.rad2deg(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    span = float(az.max() - az.min())
    pitch = vfov / nbeam
    nb = len(np.unique(np.round(el / pitch)))
    n_az = max(span / azres, 1.0)
    cells = n_az * max(nb, 1)
    return dict(span=span, nbeam=nb, npts=len(r), rng=float(np.median(r)),
                fill=len(r) / cells,
                depth_std=float(r.std()),
                inten_med=float("nan"))


def scan_nuscenes(split_dir, sensor="hdl32", limit=None):
    nbeam, vfov, azres = SENSORS[sensor]
    out = []
    files = sorted(glob.glob(os.path.join(split_dir, "*_d.pkl")))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            frames = pkl.load(open(f, "rb"))
        except Exception:
            continue
        if isinstance(frames, dict):
            frames = [frames]
        for fr in frames:
            nk = fr.get("n_spoof_k", 0)
            if not nk or nk < 3:
                continue
            P = np.asarray(fr["points"])
            spf = P[-int(nk):]
            s = phantom_stats(spf[:, :3].astype(np.float64), nbeam, vfov, azres)
            s["inten_med"] = float(np.median(spf[:, 3]))
            s["n_spoof_r"] = fr.get("n_spoof_r", np.nan)
            out.append(s)
    return out


def scan_kitti(split_dir, sensor="hdl64", limit=None):
    """KITTI writes one pkl per frame; recover the phantom by matching points
    inside spoof_gt rather than a count field."""
    nbeam, vfov, azres = SENSORS[sensor]
    out = []
    files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            fr = pkl.load(open(f, "rb"))
        except Exception:
            continue
        g = fr.get("spoof_gt")
        if g is None:
            continue
        P = np.asarray(fr["points"], dtype=np.float64)
        nk = fr.get("n_spoof_k")
        if nk and nk >= 3:
            spf = P[-int(nk):]
        else:
            # KITTI's save_frame stores no n_spoof_k, but spoof_frame appends
            # [scene_kept, spoof_kept], so the phantom is the maximal SUFFIX of
            # points lying inside spoof_gt. Walking back from the end recovers
            # it exactly, without pulling in real scene points that happen to
            # fall inside the box.
            g = np.asarray(g, dtype=np.float64)
            cx, cy, cz, dx, dy, dz, yaw = g[:7]
            d = P[:, :3] - np.array([cx, cy, cz])
            c, s_ = np.cos(-yaw), np.sin(-yaw)
            xr = d[:, 0] * c - d[:, 1] * s_
            yr = d[:, 0] * s_ + d[:, 1] * c
            inside = ((np.abs(xr) <= dx / 2 * 1.05) & (np.abs(yr) <= dy / 2 * 1.05)
                      & (np.abs(d[:, 2]) <= dz / 2 * 1.15))
            k = 0
            for j in range(len(inside) - 1, -1, -1):
                if not inside[j]:
                    break
                k += 1
            if k < 3:
                continue
            spf = P[-k:]
        if len(spf) < 3:
            continue
        st = phantom_stats(spf[:, :3], nbeam, vfov, azres)
        st["inten_med"] = float(np.median(spf[:, 3]))
        out.append(st)
    return out


def report(rows, label, real_fill_med, env_fill, env_span):
    if not rows:
        print(f"  {label:34s}  no phantoms found")
        return
    fill = np.array([r["fill"] for r in rows])
    span = np.array([r["span"] for r in rows])
    npts = np.array([r["npts"] for r in rows])
    rng = np.array([r["rng"] for r in rows])
    inten = np.array([r["inten_med"] for r in rows])
    viol = (fill > 1.0)
    print(f"  {label:34s} n={len(rows):5d}  fill med {np.median(fill):.3f}"
          f"  ratio {np.median(fill)/real_fill_med:5.2f}"
          f"  >1.0 {viol.mean()*100:5.1f}%  max {fill.max():.2f}"
          f"  span {np.median(span):6.2f}  pts {np.median(npts):5.0f}"
          f"  r {np.median(rng):5.1f}  I {np.median(inten):6.1f}")
    return dict(fill=fill, span=span, npts=npts, rng=rng, inten=inten)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nuscenes_dir")
    p.add_argument("--kitti_dir")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--real_fill", type=float, default=0.513)
    p.add_argument("--env_fill", type=float, default=0.590)
    p.add_argument("--env_span", type=float, default=7.18)
    a = p.parse_args()

    print(f"real demonstrated fill median = {a.real_fill:.3f}, "
          f"envelope p95 = {a.env_fill:.3f}, span envelope = {a.env_span:.2f} deg")

    res = {}
    if a.nuscenes_dir:
        print("\n=== nuScenes generated injection splits (HDL-32E) ===")
        for mode in ("relative", "global"):
            for d in ("easy", "medium", "hard"):
                sd = os.path.join(a.nuscenes_dir, f"inject_{mode}_{d}")
                if not os.path.isdir(sd):
                    continue
                rows = scan_nuscenes(sd, "hdl32", a.limit)
                res[f"ns_{mode}_{d}"] = report(
                    rows, f"inject_{mode}_{d}", a.real_fill, a.env_fill, a.env_span)
    if a.kitti_dir:
        print("\n=== KITTI generated injection splits (HDL-64E) ===")
        for d in ("easy", "medium", "hard"):
            sd = os.path.join(a.kitti_dir, f"inject_{d}")
            if not os.path.isdir(sd):
                continue
            rows = scan_kitti(sd, "hdl64", a.limit)
            res[f"kt_{d}"] = report(
                rows, f"inject_{d}", a.real_fill, a.env_fill, a.env_span)


if __name__ == "__main__":
    main()
