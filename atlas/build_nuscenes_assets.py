#!/usr/bin/env python
"""
Build the nuScenes analogues of the Waymo-only assets that the ATLAS
generation scripts load at import time:

  waymo_top_lidar_inclinations.npy  ->  nuscenes_beam_inclinations.npy
  range_conditioned_stats.npy       ->  nuscenes_range_conditioned_stats.npy
  trace_{easy,medium,hard}.pkl      ->  (extracted from nuScenes mini GT boxes)

nuScenes LiDAR (Velodyne HDL-32E, 32 beams) stores raw points as
[x, y, z, intensity, ring]. We exploit the ring index to recover exact
per-beam inclinations and the empirical azimuth step, and sample intensity
statistics range-conditioned. Traces are real GT object point clouds binned
by point density (easy = dense/near, hard = sparse/far).

Run from the atlas_generation/ directory (inside the `atlas` conda env).
"""
import os
import sys
import argparse
import pickle as pkl
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True,
                   help="nuScenes dataroot containing samples/ sweeps/ and the info pkls")
    p.add_argument("--info_train", default="nuscenes_infos_10sweeps_train.pkl")
    p.add_argument("--info_val", default="nuscenes_infos_10sweeps_val.pkl")
    p.add_argument("--trace_split", default="train+val", choices=["train", "val", "train+val"],
                   help="which split(s) to extract TRACES from. Use 'train' to "
                        "build the attacker's library from train only and avoid "
                        "train/val leakage when injecting into val.")
    p.add_argument("--out_dir", default=".", help="where to write the asset files")
    p.add_argument("--trace_class", default="car",
                   help="GT class to extract traces from")
    p.add_argument("--n_scan_files", type=int, default=200,
                   help="how many keyframe scans to scan for beam/intensity stats")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_raw(dataroot, lidar_path):
    """Raw nuScenes keyframe scan: [x, y, z, intensity, ring] (N, 5)."""
    return np.fromfile(os.path.join(dataroot, lidar_path),
                       dtype=np.float32).reshape(-1, 5)


def cart_2_spherical(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x) % (2 * np.pi)
    r_safe = np.where(r == 0, 1e-9, r)
    phi = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))
    return r, theta, phi


# --------------------------------------------------------------------------
# 1. Beam inclinations + azimuth resolution (from the ring index)
# --------------------------------------------------------------------------
def build_beam_geometry(dataroot, infos, n_files, out_dir):
    n_rings = 32
    phi_sum = np.zeros(n_rings)
    phi_cnt = np.zeros(n_rings)
    az_steps = []

    for info in infos[:n_files]:
        raw = load_raw(dataroot, info["lidar_path"])
        r, theta, phi = cart_2_spherical(raw)
        ring = raw[:, 4].astype(np.int32)
        valid = (r > 1.0) & (ring >= 0) & (ring < n_rings)
        np.add.at(phi_sum, ring[valid], phi[valid])
        np.add.at(phi_cnt, ring[valid], 1)

        # azimuth step: within one ring, consecutive fired points along the scan
        for rr in range(n_rings):
            m = valid & (ring == rr)
            if m.sum() < 50:
                continue
            az = np.sort(theta[m])
            d = np.diff(az)
            d = d[(d > 1e-5) & (d < np.deg2rad(2.0))]  # drop wrap/gaps
            if d.size:
                az_steps.append(np.median(d))

    inclinations = phi_sum / np.maximum(phi_cnt, 1)
    # sort ascending so nearest-beam search is well-behaved (like the waymo asset)
    inclinations = np.sort(inclinations)
    dtheta = float(np.median(az_steps)) if az_steps else np.deg2rad(0.33)

    np.save(os.path.join(out_dir, "nuscenes_beam_inclinations.npy"), inclinations)
    np.save(os.path.join(out_dir, "nuscenes_azimuth_res.npy"), np.array([dtheta]))
    print(f"[geometry] {n_rings} beams | "
          f"incl range {np.rad2deg(inclinations.min()):.2f}..{np.rad2deg(inclinations.max()):.2f} deg | "
          f"dtheta = {np.rad2deg(dtheta):.4f} deg")
    return inclinations, dtheta


# --------------------------------------------------------------------------
# 2. Range-conditioned intensity statistics
# --------------------------------------------------------------------------
def build_range_stats(dataroot, infos, n_files, out_dir):
    # range bands (m) matching the near->far layout the spoofer queries
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 80)]
    per_band = defaultdict(list)

    for info in infos[:n_files]:
        raw = load_raw(dataroot, info["lidar_path"])
        r, _, _ = cart_2_spherical(raw)
        intensity = raw[:, 3]
        for (lo, hi) in bands:
            m = (r >= lo) & (r < hi) & (intensity > 0)
            if m.any():
                per_band[(lo, hi)].append(intensity[m])

    stats = {}
    for band, chunks in per_band.items():
        I = np.concatenate(chunks)
        logI = np.log(np.clip(I, 1e-3, None))
        stats[band] = {
            "log_mu": float(logI.mean()),
            "log_sigma": float(logI.std() + 1e-6),
            "intensity_p1": float(np.percentile(I, 1)),
            "intensity_p99": float(np.percentile(I, 99)),
            # nuScenes has no elongation; col 4 is timestamp. Keep a pool of
            # zeros so downstream code that writes col 4 stays a no-op/time-safe.
            "elong_pool": np.zeros(1, dtype=np.float32),
            "n": int(I.size),
        }
        print(f"[intensity] band {band}: n={I.size:>7d} "
              f"log_mu={stats[band]['log_mu']:.3f} "
              f"p1={stats[band]['intensity_p1']:.1f} p99={stats[band]['intensity_p99']:.1f}")

    np.save(os.path.join(out_dir, "nuscenes_range_conditioned_stats.npy"),
            stats, allow_pickle=True)
    return stats


# --------------------------------------------------------------------------
# 3. Traces (GT object point clusters), binned easy/medium/hard by density
# --------------------------------------------------------------------------
def build_traces(dataroot, infos, cls, out_dir, seed, max_per_bucket=4000):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    import torch

    traces = []
    for info in infos:
        gt_boxes = info["gt_boxes"]          # (M, 9): x,y,z,dx,dy,dz,yaw,vx,vy
        gt_names = info["gt_names"]
        mask = gt_names == cls
        if not mask.any():
            continue
        raw = load_raw(dataroot, info["lidar_path"])
        pts = raw[:, :4].astype(np.float32)  # x,y,z,intensity (drop ring)

        boxes7 = gt_boxes[mask][:, :7].astype(np.float32)
        # points_in_boxes_cpu expects (N,3) pts and (M,7) boxes -> (M, N) mask
        idx = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(pts[:, :3]),
            torch.from_numpy(boxes7),
        ).numpy()  # (M, N) 0/1

        for bi in range(boxes7.shape[0]):
            in_box = idx[bi] > 0
            n = int(in_box.sum())
            if n < 5:
                continue
            obj_pts = pts[in_box]
            # store [x,y,z,intensity, col4]; col4 = 0.0 (timestamp placeholder)
            obj_pts5 = np.hstack([obj_pts, np.zeros((n, 1), np.float32)])
            box = boxes7[bi].copy()
            rng_center = float(np.sqrt(box[0] ** 2 + box[1] ** 2))
            traces.append({"points": obj_pts5, "box": box,
                           "n_points": n, "range": rng_center})

    n_pts = np.array([t["n_points"] for t in traces])
    print(f"[traces] extracted {len(traces)} '{cls}' objects | "
          f"points/obj min/med/max = {n_pts.min()}/{int(np.median(n_pts))}/{n_pts.max()}")

    # difficulty by density: easy = dense (near), hard = sparse (far)
    q33, q66 = np.percentile(n_pts, [33, 66])
    buckets = {
        "easy":   [t for t in traces if t["n_points"] >= q66],
        "medium": [t for t in traces if q33 <= t["n_points"] < q66],
        "hard":   [t for t in traces if t["n_points"] < q33],
    }
    rng = np.random.default_rng(seed)
    for name, lst in buckets.items():
        # cap each bucket for a bounded, memory-safe, reasonably-sized library
        if len(lst) > max_per_bucket:
            sel = rng.choice(len(lst), size=max_per_bucket, replace=False)
            lst = [lst[i] for i in sel]
        path = os.path.join(out_dir, f"trace_{name}.pkl")
        with open(path, "wb") as f:
            pkl.dump(lst, f)
        rr = np.array([t["range"] for t in lst]) if lst else np.array([0.0])
        print(f"[traces] {name:6s}: {len(lst):4d} objs | "
              f"mean range {rr.mean():.1f}m -> {path}")
    return buckets


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    os.makedirs(args.out_dir, exist_ok=True)

    def _load(name):
        with open(os.path.join(args.dataroot, name), "rb") as f:
            return pkl.load(f)

    train_infos = _load(args.info_train)
    val_infos = _load(args.info_val)

    # Beam geometry + intensity stats are sensor properties -> use all scans.
    stats_infos = train_infos + val_infos
    # Traces are the attacker's library -> restrict to the requested split so we
    # can avoid injecting val objects back into val (train/val leakage).
    trace_infos = {"train": train_infos, "val": val_infos,
                   "train+val": train_infos + val_infos}[args.trace_split]
    print(f"stats from {len(stats_infos)} infos | traces from "
          f"{len(trace_infos)} infos (split={args.trace_split})")

    build_beam_geometry(args.dataroot, stats_infos, args.n_scan_files, args.out_dir)
    build_range_stats(args.dataroot, stats_infos, args.n_scan_files, args.out_dir)
    build_traces(args.dataroot, trace_infos, args.trace_class, args.out_dir, args.seed)
    print("done.")


if __name__ == "__main__":
    main()
