#!/usr/bin/env python3
"""Direct per-cell occupancy of injected phantoms -- the exact physical limit.

Rather than estimating the number of victim-sampled cells from
(azimuth span x beams spanned), this bins every phantom point into the cell it
would actually land in:

    az_cell = floor(azimuth / azimuth_resolution)
    beam    = argmin |elevation - beam_inclination|      (true asset table)

A single-return spinning LiDAR emits ONE pulse per (az_cell, beam) per
revolution, so it can report at most one range there. Therefore

    ppc = n_points / n_distinct_occupied_cells

is >= 1 by construction, and ppc > 1 means the phantom asks the sensor to
report several ranges in one cell -- physically impossible to inject.

This uses the real beam-inclination tables shipped with the generator, so the
binning matches what spoof_frame itself used.
"""
import argparse
import glob
import os
import pickle as pkl

import numpy as np

ASSETS = {
    "hdl32": ("nuscenes_beam_inclinations.npy", "nuscenes_azimuth_res.npy"),
    "hdl64": ("kitti_beam_inclinations.npy", "kitti_azimuth_res.npy"),
}


def load_assets(asset_dir, sensor):
    inc_f, az_f = ASSETS[sensor]
    inc = np.load(os.path.join(asset_dir, inc_f))
    azres = np.load(os.path.join(asset_dir, az_f))
    azres = float(np.atleast_1d(azres).ravel()[0])
    return np.sort(np.asarray(inc, dtype=np.float64).ravel()), azres


def occupancy(xyz, inclinations, dtheta):
    """Return (npts, n_cells, points_per_cell) using the generator's binning."""
    r = np.linalg.norm(xyz, axis=1)
    ok = r > 1e-6
    xyz, r = xyz[ok], r[ok]
    if len(r) < 3:
        return None
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.arcsin(np.clip(xyz[:, 2] / r, -1, 1))
    az_idx = np.floor(np.mod(theta, 2 * np.pi) / dtheta).astype(np.int64)
    beam = np.abs(inclinations[:, None] - phi).argmin(axis=0)
    cells = len(set(zip(az_idx.tolist(), beam.tolist())))
    return len(r), cells, len(r) / max(cells, 1)


def scan(files, sensor, asset_dir, kind, limit=None):
    inc, azres_deg = load_assets(asset_dir, sensor)
    # asset azimuth res is stored in degrees; generator uses radians internally
    dtheta = np.deg2rad(azres_deg) if azres_deg > 0.01 else azres_deg
    inc_rad = np.deg2rad(inc) if np.abs(inc).max() > 1.6 else inc
    out = []
    if limit:
        files = files[:limit]
    for f in files:
        try:
            obj = pkl.load(open(f, "rb"))
        except Exception:
            continue
        frames = obj if isinstance(obj, list) else [obj]
        for fr in frames:
            if fr.get("spoof_gt") is None:
                continue
            P = np.asarray(fr["points"], dtype=np.float64)
            nk = fr.get("n_spoof_k")
            if nk and int(nk) >= 3:
                spf = P[-int(nk):, :3]
            elif kind == "kitti":
                g = np.asarray(fr["spoof_gt"], dtype=np.float64)
                cx, cy, cz, dx, dy, dz, yaw = g[:7]
                d = P[:, :3] - np.array([cx, cy, cz])
                c, s_ = np.cos(-yaw), np.sin(-yaw)
                xr = d[:, 0] * c - d[:, 1] * s_
                yr = d[:, 0] * s_ + d[:, 1] * c
                ins = ((np.abs(xr) <= dx / 2 * 1.05) & (np.abs(yr) <= dy / 2 * 1.05)
                       & (np.abs(d[:, 2]) <= dz / 2 * 1.15))
                k = 0
                for j in range(len(ins) - 1, -1, -1):
                    if not ins[j]:
                        break
                    k += 1
                if k < 3:
                    continue
                spf = P[-k:, :3]
            else:
                continue
            res = occupancy(spf, inc_rad, dtheta)
            if res:
                out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes_dir")
    ap.add_argument("--kitti_dir")
    ap.add_argument("--ns_assets", default="../atlas_generation/assets_trainval")
    ap.add_argument("--kt_assets", default="../atlas_generation/assets_kitti")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    print("points-per-cell (ppc): 1.0 = exactly one return per victim-sampled")
    print("cell (physically ideal). ppc > 1 = impossible to inject.\n")
    print(f"{'split':32s} {'n':>5s} {'ppc med':>8s} {'ppc p95':>8s} "
          f"{'%>1.05':>7s} {'%>1.5':>7s} {'cells':>7s} {'pts':>6s}")
    print("-" * 88)

    jobs = []
    if a.nuscenes_dir:
        for m in ("relative", "global"):
            for d in ("easy", "medium", "hard"):
                jobs.append((f"ns inject_{m}_{d}",
                             sorted(glob.glob(os.path.join(
                                 a.nuscenes_dir, f"inject_{m}_{d}", "*_d.pkl"))),
                             "hdl32", a.ns_assets, "nuscenes"))
    if a.kitti_dir:
        for d in ("easy", "medium", "hard"):
            jobs.append((f"kitti inject_{d}",
                         sorted(glob.glob(os.path.join(
                             a.kitti_dir, f"inject_{d}", "*.pkl"))),
                         "hdl64", a.kt_assets, "kitti"))

    for label, files, sensor, adir, kind in jobs:
        if not files:
            continue
        rows = scan(files, sensor, adir, kind, a.limit)
        if not rows:
            print(f"{label:32s}  none")
            continue
        npts = np.array([r[0] for r in rows], float)
        cells = np.array([r[1] for r in rows], float)
        ppc = np.array([r[2] for r in rows], float)
        print(f"{label:32s} {len(rows):5d} {np.median(ppc):8.3f} "
              f"{np.percentile(ppc,95):8.3f} {(ppc>1.05).mean()*100:6.1f}% "
              f"{(ppc>1.5).mean()*100:6.1f}% {np.median(cells):7.0f} "
              f"{np.median(npts):6.0f}")


if __name__ == "__main__":
    main()
