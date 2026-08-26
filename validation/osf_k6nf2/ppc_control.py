#!/usr/bin/env python3
"""Control experiment for the points-per-cell (ppc) metric.

ppc = n_points / n_distinct_(az_cell, beam) cells is only interpretable relative
to what REAL vehicle points score under the SAME binning. Two effects inflate it
independently of any attack:

  * nuScenes clouds are 10-sweep accumulated, so one real car legitimately
    contributes up to ~10 returns per cell.
  * the beam-inclination table is exact for nuScenes (from the ring index) but
    QUANTILE-ESTIMATED for KITTI, so KITTI binning mis-assigns points and
    collapses them into fewer cells, inflating ppc for real and fake alike.

So we measure ppc for real GT-car points in clean frames, and compare ATLAS
phantoms against that baseline rather than against the ideal 1.0.
"""
import argparse
import os
import pickle as pkl
import sys

import numpy as np


def load_assets(asset_dir, inc_name, az_name):
    inc = np.sort(np.load(os.path.join(asset_dir, inc_name)).ravel().astype(np.float64))
    dtheta = float(np.load(os.path.join(asset_dir, az_name)).ravel()[0])
    return inc, dtheta


def ppc(xyz, inc, dtheta):
    r = np.linalg.norm(xyz, axis=1)
    m = r > 1e-6
    xyz, r = xyz[m], r[m]
    if len(r) < 3:
        return None
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.arcsin(np.clip(xyz[:, 2] / r, -1, 1))
    az = np.floor(np.mod(theta, 2 * np.pi) / dtheta).astype(np.int64)
    beam = np.abs(inc[:, None] - phi).argmin(axis=0)
    cells = len(set(zip(az.tolist(), beam.tolist())))
    return len(r), cells, len(r) / max(cells, 1)


def in_box(P, box, pad=1.0):
    cx, cy, cz, dx, dy, dz, yaw = box[:7]
    d = P[:, :3] - np.array([cx, cy, cz])
    c, s = np.cos(-yaw), np.sin(-yaw)
    xr = d[:, 0] * c - d[:, 1] * s
    yr = d[:, 0] * s + d[:, 1] * c
    return ((np.abs(xr) <= dx / 2 * pad) & (np.abs(yr) <= dy / 2 * pad)
            & (np.abs(d[:, 2]) <= dz / 2 * pad))


def kitti_control(kitti_root, asset_dir, n_frames=400):
    inc, dtheta = load_assets(asset_dir, "kitti_beam_inclinations.npy",
                              "kitti_azimuth_res.npy")
    infos = pkl.load(open(os.path.join(kitti_root, "kitti_infos_val.pkl"), "rb"))
    rows = []
    for info in infos[:n_frames]:
        fid = info["point_cloud"]["lidar_idx"]
        vp = os.path.join(kitti_root, "training", "velodyne", f"{fid}.bin")
        if not os.path.exists(vp):
            continue
        P = np.fromfile(vp, dtype=np.float32).reshape(-1, 4).astype(np.float64)
        a = info["annos"]
        names = a["name"][a["name"] != "DontCare"]
        boxes = a.get("gt_boxes_lidar")
        if boxes is None or len(names) != len(boxes):
            continue
        for nm, b in zip(names, boxes):
            if nm != "Car":
                continue
            sel = P[in_box(P, b)]
            r = ppc(sel[:, :3], inc, dtheta)
            if r:
                rows.append(r)
    return rows


def nuscenes_control(atlas_dir, asset_dir, n_scenes=40):
    """Use the generated splits' own frames: real GT car points from gt_boxes,
    in the SAME accumulated clouds the phantom was injected into."""
    import glob
    inc, dtheta = load_assets(asset_dir, "nuscenes_beam_inclinations.npy",
                              "nuscenes_azimuth_res.npy")
    files = sorted(glob.glob(os.path.join(atlas_dir, "inject_relative_easy",
                                          "*_d.pkl")))[:n_scenes]
    rows = []
    for f in files:
        try:
            seg = pkl.load(open(f, "rb"))
        except Exception:
            continue
        for fr in seg:
            P = np.asarray(fr["points"], dtype=np.float64)
            nk = int(fr.get("n_spoof_k", 0) or 0)
            scene = P[:-nk] if nk else P            # exclude the phantom
            gtb = np.asarray(fr.get("gt_boxes"))
            if gtb is None or gtb.ndim != 2 or len(gtb) == 0:
                continue
            for b in gtb:
                # last col of nuScenes gt_boxes after prepare_data = class label
                if int(round(b[-1])) != 1:          # car
                    continue
                sel = scene[in_box(scene, b)]
                r = ppc(sel[:, :3], inc, dtheta)
                if r:
                    rows.append(r)
    return rows


def summarise(rows, label):
    if not rows:
        print(f"  {label:38s} none")
        return
    p = np.array([r[2] for r in rows])
    n = np.array([r[0] for r in rows], float)
    c = np.array([r[1] for r in rows], float)
    print(f"  {label:38s} n={len(rows):5d}  ppc med {np.median(p):6.3f}  "
          f"p95 {np.percentile(p,95):6.3f}  pts med {np.median(n):5.0f}  "
          f"cells med {np.median(c):5.0f}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root")
    ap.add_argument("--nuscenes_atlas")
    ap.add_argument("--kt_assets", default="../atlas_generation/assets_kitti")
    ap.add_argument("--ns_assets", default="../atlas_generation/assets_trainval")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()

    print("CONTROL: ppc of REAL GT-car points under the generator's own binning")
    print("(this is the baseline an ATLAS phantom should match, not 1.0)\n")
    if a.nuscenes_atlas:
        r = nuscenes_control(a.nuscenes_atlas, a.ns_assets, n_scenes=40)
        summarise(r, "nuScenes REAL cars (10-sweep accum)")
    if a.kitti_root:
        r = kitti_control(a.kitti_root, a.kt_assets, a.n)
        summarise(r, "KITTI REAL cars (single sweep)")


if __name__ == "__main__":
    main()
