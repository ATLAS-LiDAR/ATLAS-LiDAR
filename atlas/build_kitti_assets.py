#!/usr/bin/env python
"""Build KITTI analogues of the ATLAS assets (beam geometry, intensity stats,
car trace library), mirroring build_nuscenes_assets.py.

KITTI velodyne differs from nuScenes:
  * points are [x, y, z, intensity] (4 feat, intensity in [0,1]) — NO ring index.
    So beam inclinations are derived by quantile-binning the elevation-angle
    distribution into 64 beams (Velodyne HDL-64E) instead of reading a ring col.
  * single frame, single return.
  * gt boxes are 7-dim [x,y,z,l,w,h,yaw] in lidar frame (annos['gt_boxes_lidar']).

Traces are built from the TRAIN split only (no train/val leakage), class Car.
"""
import argparse
import os
import pickle as pkl
from collections import defaultdict

import numpy as np

N_BEAMS = 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True, help="KITTI root (has training/, kitti_infos_*.pkl)")
    p.add_argument("--info_train", default="kitti_infos_train.pkl")
    p.add_argument("--info_val", default="kitti_infos_val.pkl")
    p.add_argument("--trace_split", default="train", choices=["train", "val", "train+val"])
    p.add_argument("--trace_class", default="Car")
    p.add_argument("--out_dir", default="assets_kitti")
    p.add_argument("--n_scan_files", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_velo(dataroot, idx):
    f = os.path.join(dataroot, "training", "velodyne", f"{idx}.bin")
    return np.fromfile(f, dtype=np.float32).reshape(-1, 4)


def cart_2_spherical(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x) % (2 * np.pi)
    rs = np.where(r == 0, 1e-9, r)
    phi = np.arcsin(np.clip(z / rs, -1.0, 1.0))
    return r, theta, phi


def build_beam_geometry(dataroot, idxs, n_files, out_dir):
    """64 beam inclinations from the pooled elevation distribution (no ring idx)."""
    phis = []
    for idx in idxs[:n_files]:
        p = load_velo(dataroot, idx)
        r, _, phi = cart_2_spherical(p)
        phis.append(phi[r > 1.0])
    phi_all = np.concatenate(phis)
    # 64 beams = centers of 64 equal-count elevation bins (robust to density)
    qs = np.linspace(0, 100, N_BEAMS + 1)
    edges = np.percentile(phi_all, qs)
    inclinations = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(N_BEAMS)])
    inclinations = np.sort(inclinations)
    # HDL-64E azimuth resolution ~0.1728 deg (2083 pts/ring at 10Hz); measure it
    dtheta = np.deg2rad(0.1728)
    np.save(os.path.join(out_dir, "kitti_beam_inclinations.npy"), inclinations)
    np.save(os.path.join(out_dir, "kitti_azimuth_res.npy"), np.array([dtheta]))
    print(f"[geometry] {N_BEAMS} beams | incl {np.rad2deg(inclinations.min()):.2f}.."
          f"{np.rad2deg(inclinations.max()):.2f} deg | dtheta={np.rad2deg(dtheta):.4f} deg")
    return inclinations, dtheta


def build_range_stats(dataroot, idxs, n_files, out_dir):
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 80)]
    per = defaultdict(list)
    for idx in idxs[:n_files]:
        p = load_velo(dataroot, idx)
        r, _, _ = cart_2_spherical(p)
        inten = p[:, 3]
        for (lo, hi) in bands:
            m = (r >= lo) & (r < hi) & (inten > 0)
            if m.any():
                per[(lo, hi)].append(inten[m])
    stats = {}
    for band, chunks in per.items():
        I = np.concatenate(chunks)
        logI = np.log(np.clip(I, 1e-4, None))
        stats[band] = {
            "log_mu": float(logI.mean()), "log_sigma": float(logI.std() + 1e-6),
            "intensity_p1": float(np.percentile(I, 1)),
            "intensity_p99": float(np.percentile(I, 99)),
            "elong_pool": np.zeros(1, dtype=np.float32),  # KITTI has no 5th feat
            "n": int(I.size),
        }
        print(f"[intensity] {band}: n={I.size:>8d} log_mu={stats[band]['log_mu']:.3f} "
              f"p99={stats[band]['intensity_p99']:.3f}")
    np.save(os.path.join(out_dir, "kitti_range_conditioned_stats.npy"), stats, allow_pickle=True)
    return stats


def build_traces(dataroot, infos, cls, out_dir, seed):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    import torch
    traces = []
    for info in infos:
        a = info["annos"]
        if "gt_boxes_lidar" not in a or len(a["gt_boxes_lidar"]) == 0:
            continue
        # gt_boxes_lidar aligns with the non-DontCare names, in order.
        names = a["name"][a["name"] != "DontCare"]
        boxes_all = a["gt_boxes_lidar"]
        if len(names) != len(boxes_all):
            continue  # defensive: skip malformed frame
        mask = names == cls
        if not mask.any():
            continue
        idx = info["point_cloud"]["lidar_idx"]
        p = load_velo(dataroot, idx)
        pts = p[:, :4].astype(np.float32)
        boxes7 = boxes_all[mask][:, :7].astype(np.float32)
        inb = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(pts[:, :3]), torch.from_numpy(boxes7)).numpy()
        for bi in range(boxes7.shape[0]):
            m = inb[bi] > 0
            n = int(m.sum())
            if n < 5:
                continue
            obj = pts[m]
            obj5 = np.hstack([obj, np.zeros((n, 1), np.float32)])  # pad 5th col (unused)
            box = boxes7[bi].copy()
            traces.append({"points": obj5, "box": box, "n_points": n,
                           "range": float(np.hypot(box[0], box[1]))})
    npts = np.array([t["n_points"] for t in traces])
    print(f"[traces] {len(traces)} '{cls}' objects | pts/obj min/med/max "
          f"{npts.min()}/{int(np.median(npts))}/{npts.max()}")
    q33, q66 = np.percentile(npts, [33, 66])
    buckets = {"easy": [t for t in traces if t["n_points"] >= q66],
               "medium": [t for t in traces if q33 <= t["n_points"] < q66],
               "hard": [t for t in traces if t["n_points"] < q33]}
    for name, lst in buckets.items():
        with open(os.path.join(out_dir, f"trace_{name}.pkl"), "wb") as f:
            pkl.dump(lst, f)
        rr = np.array([t["range"] for t in lst]) if lst else np.array([0.0])
        print(f"[traces] {name:6s}: {len(lst):5d} objs | mean range {rr.mean():.1f}m")
    return buckets


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    os.makedirs(args.out_dir, exist_ok=True)

    tr = pkl.load(open(os.path.join(args.dataroot, args.info_train), "rb"))
    va = pkl.load(open(os.path.join(args.dataroot, args.info_val), "rb"))
    idx_all = [i["point_cloud"]["lidar_idx"] for i in (tr + va)]
    trace_infos = {"train": tr, "val": va, "train+val": tr + va}[args.trace_split]
    print(f"stats from {len(idx_all)} frames | traces from {len(trace_infos)} (split={args.trace_split})")

    build_beam_geometry(args.dataroot, idx_all, args.n_scan_files, args.out_dir)
    build_range_stats(args.dataroot, idx_all, args.n_scan_files, args.out_dir)
    build_traces(args.dataroot, trace_infos, args.trace_class, args.out_dir, args.seed)
    print("done.")


if __name__ == "__main__":
    main()
