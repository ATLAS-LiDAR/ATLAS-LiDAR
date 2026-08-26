#!/usr/bin/env python
"""
KITTI ATLAS machinery — the single-frame analogue of nuscenes_atlas_common.py.

Structural differences from nuScenes that shape this port:
  * KITTI frames are INDEPENDENT (no scenes/sequences, no ego-pose, no sweeps).
    -> There is no "global vs relative" distinction: with a single frame and no
       ego-motion, a world-fixed and ego-fixed phantom are identical. So KITTI
       injection is ONE mode: place a car trace in front of the ego, per frame.
    -> Every val frame is its own target (no 32-frame toggle windows).
  * points are (N,4) [x,y,z,intensity]; kept 4-wide throughout (no time/lag col).
  * 64-beam HDL-64E geometry + intensity stats from build_kitti_assets.py.
  * removal reuses the dataset-agnostic A-HFR sector logic in removal.py.

Output per variant: one `<frame_id>.pkl` per target frame, holding
{points (N,4), spoof_gt (7,), frame_id}. Eval iterates the same frames.
"""
import os
import pickle as pkl

import numpy as np

_ASSET_DIR = os.environ.get("KITTI_ATLAS_ASSET_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_kitti"))
ELEVATIONS = np.load(os.path.join(_ASSET_DIR, "kitti_beam_inclinations.npy"))
DTHETA = float(np.load(os.path.join(_ASSET_DIR, "kitti_azimuth_res.npy"))[0])
STATS = np.load(os.path.join(_ASSET_DIR, "kitti_range_conditioned_stats.npy"),
                allow_pickle=True).item()
P_TWO = 0.0  # single return


def cart_2_spherical(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x) % (2 * np.pi)
    rs = np.where(r == 0, 1e-9, r)
    phi = np.arcsin(np.clip(z / rs, -1.0, 1.0))
    return r, theta, phi


def get_stats_for_range(r):
    for (lo, hi), v in STATS.items():
        if lo <= r < hi:
            return v
    return None


def center_trace(trace):
    pts = trace["points"][:, :2]
    box = trace["box"]
    c, s = np.cos(-box[6]), np.sin(-box[6])
    rot = (pts - box[:2]) @ np.array([[c, -s], [s, c]]).T
    b = box.copy(); b[0] = b[1] = 0.0; b[6] = 0.0
    return rot, b


def place_trace(traces, seed):
    """Place a car trace in front of the ego (KITTI FOV: +x forward)."""
    rng = np.random.default_rng(seed)
    t = traces[int(rng.integers(len(traces)))]
    # forward offset within camera FOV; KITTI x in [0,70], keep 12-30m, small |y|
    offset = np.array([rng.uniform(12, 30), rng.uniform(-6, 6)], dtype=np.float32)
    ctr, cbox = center_trace(t)
    yaw = float(rng.uniform(-0.3, 0.3))
    c, s = np.cos(yaw), np.sin(yaw)
    ctr = ctr @ np.array([[c, -s], [s, c]]).T
    cbox[6] = yaw
    planar = ctr + offset
    xyz = np.concatenate([planar, t["points"][:, 2:3]], axis=1)
    spoof = np.hstack([xyz, t["points"][:, 3:4]])  # (M,4) x,y,z,intensity
    box = cbox.copy(); box[0] += offset[0]; box[1] += offset[1]
    anno = {"perturbation": "injection", "seed": int(seed),
            "offset": offset.tolist(), "yaw": yaw}
    return spoof, anno, box


def ground_anchor(points, spoof, gt, radius=2.0):
    gtb = gt.copy(); spf = spoof.copy()
    ctr = spf[:, :2].mean(0)
    d = np.linalg.norm(points[:, :2] - ctr, axis=1)
    loc = d < radius
    if loc.sum() > 20:
        gz = np.percentile(points[loc, 2], 5)
        dz = gz - spf[:, 2].min()
        spf[:, 2] += dz; gtb[2] += dz
    return spf, gtb


def spoof_frame(spoof_points, points, gt_box, rng):
    """Inject occlusion-consistent phantom into a KITTI (N,4) cloud."""
    if spoof_points is None:
        return points, gt_box, 0, 0
    n_r = spoof_points.shape[0]
    spf, gt_box = ground_anchor(points, spoof_points, gt_box)

    r_avg = float(np.mean(np.sqrt(np.sum(spf[:, :3] ** 2, axis=1))))
    band = get_stats_for_range(r_avg)
    if band is not None:
        I = rng.lognormal(band["log_mu"], band["log_sigma"], size=spf.shape[0])
        spf[:, 3] = np.clip(I, band["intensity_p1"], band["intensity_p99"])

    r_t, th_t, ph_t = cart_2_spherical(spf)
    r_s, th_s, ph_s = cart_2_spherical(points[:, :3])
    az_s = np.floor(np.mod(th_s, 2 * np.pi) / DTHETA).astype(np.int32)
    bm_s = np.abs(ELEVATIONS[:, None] - ph_s).argmin(axis=0)
    order = np.lexsort((r_s, bm_s, az_s))
    az_s, bm_s, r_s = az_s[order], bm_s[order], r_s[order]
    keys = az_s * 1000 + bm_s
    uk, gstart = np.unique(keys, return_index=True)
    gend = np.r_[gstart[1:], len(keys)]
    k2s = {k: (s, e) for k, s, e in zip(uk, gstart, gend)}

    rm_spoof, rm_scene = [], []
    for i, (r, th, ph) in enumerate(zip(r_t, th_t, ph_t)):
        az = int(np.floor((th % (2 * np.pi)) / DTHETA))
        bm = int(np.argmin(np.abs(ELEVATIONS - ph)))
        key = az * 1000 + bm
        if key not in k2s:
            continue
        st, en = k2s[key]
        rg = r_s[st:en]; ig = order[st:en]
        if len(rg) == 0:
            continue
        if rg[0] < r:
            rm_spoof.append(i)
        elif r < rg[0]:
            rm_scene.extend(ig.tolist())
        else:
            k = np.searchsorted(rg, r, side="right")
            if k < len(rg):
                rm_scene.extend(ig[k:].tolist())

    ms = np.ones(spf.shape[0], bool); ms[rm_spoof] = False
    spf_keep = spf[ms]
    mc = np.ones(points.shape[0], bool); mc[rm_scene] = False
    fused = np.concatenate([points[mc], spf_keep], axis=0)
    return fused, gt_box, n_r, spf_keep.shape[0]


def load_points_kitti(dataset, index):
    """Load one KITTI frame's (FOV-filtered) points as (N,4) + its data_dict."""
    dd = dataset[index]
    pts = dd["points"]  # (N,4) after prepare_data (voxel keys added)
    for k in ("voxels", "voxel_coords", "voxel_num_points"):
        dd.pop(k, None)
    return pts[:, :4].astype(np.float32), dd


def frame_car_boxes(info):
    """Non-DontCare gt boxes with a Car mask, from a KITTI info."""
    a = info["annos"]
    if "gt_boxes_lidar" not in a or len(a["gt_boxes_lidar"]) == 0:
        return np.zeros((0, 7)), np.array([])
    names = a["name"][a["name"] != "DontCare"]
    boxes = a["gt_boxes_lidar"]
    if len(names) != len(boxes):
        return np.zeros((0, 7)), np.array([])
    return boxes[:, :7].astype(np.float64), names


def save_frame(out_dir, frame_id, points, spoof_gt, anno):
    with open(os.path.join(out_dir, f"{frame_id}.pkl"), "wb") as f:
        pkl.dump({"frame_id": frame_id, "points": points.astype(np.float32),
                  "spoof_gt": spoof_gt, "anno": anno}, f)
