#!/usr/bin/env python3
"""Compare ATLAS generated removal splits against the REAL PRA capture.

Real-side quantities (measured by extract_removal_real / the sector analysis):
  * wedge angular width (deg)             -- the attack angle Delta-theta
  * fraction of genuine returns removed inside the wedge
  * number of victim channels affected
  * whole-cloud impact (returns lost / total)

This script measures exactly the same four quantities on the generated
removal_az{10..60} splits, so the comparison is like-for-like.

For each attacked frame we recover the removal wedge from the stored metadata
(az_center_rad, az_width_deg, n_points_removed, n_points_in_sector) and also
recompute the affected-channel count geometrically from the surviving cloud.
"""
import argparse
import glob
import os
import pickle as pkl

import numpy as np


def load_assets(asset_dir, sensor="nuscenes"):
    inc = np.sort(np.load(os.path.join(
        asset_dir, f"{sensor}_beam_inclinations.npy")).ravel().astype(np.float64))
    dtheta = float(np.load(os.path.join(
        asset_dir, f"{sensor}_azimuth_res.npy")).ravel()[0])
    return inc, dtheta


def scan_split(split_dir, inc, dtheta, limit=None):
    """Attack bookkeeping lives in the *_annos sidecar (spoof_annos: {frame_idx:
    meta}); the point cloud lives in the *_d.pkl segment file."""
    annos_dir = split_dir + "_annos"
    files = sorted(glob.glob(os.path.join(annos_dir, "*_a.pkl")))
    if limit:
        files = files[:limit]
    rows = []
    for af in files:
        try:
            an = pkl.load(open(af, "rb"))
        except Exception:
            continue
        seg_name = an.get("segment") or os.path.basename(af)[:-len("_a.pkl")]
        df = os.path.join(split_dir, f"{seg_name}_d.pkl")
        try:
            seg = pkl.load(open(df, "rb"))
        except Exception:
            continue
        for idx, m in (an.get("spoof_annos") or {}).items():
            if not isinstance(m, dict) or not m.get("active_this_frame", False):
                continue
            nin = m.get("n_points_in_sector", 0)
            nrem = m.get("n_points_removed", 0)
            if not nin or idx >= len(seg):
                continue
            P = np.asarray(seg[idx]["points"], dtype=np.float64)
            # channels spanned by the wedge in the SURVIVING cloud
            azc = m.get("az_center_rad")
            w = np.deg2rad(m.get("az_width_deg", 0.0))
            nch = np.nan
            if azc is not None and w > 0 and len(P):
                r = np.linalg.norm(P[:, :3], axis=1)
                ok = r > 1e-6
                th = np.arctan2(P[ok, 1], P[ok, 0])
                d = np.abs(np.arctan2(np.sin(th - azc), np.cos(th - azc)))
                sel = d <= w / 2
                if sel.sum():
                    phi = np.arcsin(np.clip(P[ok][sel, 2] / r[ok][sel], -1, 1))
                    beam = np.abs(inc[:, None] - phi).argmin(axis=0)
                    nch = len(np.unique(beam))
            rows.append(dict(
                width_deg=float(m.get("az_width_deg", np.nan)),
                p_remove_cfg=float(m.get("p_remove", np.nan)),
                removed=int(nrem),
                in_sector=int(nin),
                rate=nrem / max(nin, 1),
                n_before=int(m.get("n_points_before", 0)),
                n_after=int(m.get("n_points_after", 0)),
                nch=nch,
                rng=float(m.get("target_range_m") or np.nan),
            ))
    return rows


# Real-side quantities from real_removal_v3.py (az_res=0.4, the VLP-16's true
# per-cell resolution once the two interleaved firings are merged).
#
# `rate_env` averages over the whole saturated-echo envelope, INCLUDING the
# fringe where the spoofer is off-target. It is NOT comparable to ATLAS's
# p_remove, which is the deletion rate inside the wedge the attacker is aiming
# at. Use `rate_aim` (peak cell, best-aimed decile of frames) for that; it is
# also the quantity the paper's Table 3 (0.836-0.927) reports.
REAL = {
    "last_mode": dict(width=6.80, rate_env=0.407, rate_peak=0.749,
                      rate_aim=0.826, nch=16, nray=113, target_m=2.01),
    "dual_mode": dict(width=6.80, rate_env=0.466, rate_peak=0.724,
                      rate_aim=0.867, nch=16, nray=131, target_m=2.03),
}
PAPER_TABLE3 = dict(car=(0.836, 0.891, 0.927), robot=(0.854, 0.924, 0.959))
PAPER_MAX_ATTACK_ANGLE_DEG = 45.0   # az50/az60 are extrapolations beyond this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes_dir", required=True)
    ap.add_argument("--assets", default="../atlas_generation/assets_trainval")
    ap.add_argument("--sensor", default="nuscenes")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    inc, dtheta = load_assets(a.assets, a.sensor)

    print("REAL PRA capture (VLP-16, Cao et al. USENIX Sec '23):")
    for k, v in REAL.items():
        chord = 2 * v["target_m"] * np.sin(np.deg2rad(v["width"]) / 2)
        print(f"  {k:10s} envelope {v['width']:.2f} deg @ {v['target_m']:.2f} m "
              f"(chord {chord:.2f} m)  channels {v['nch']}/16")
        print(f"  {'':10s} removal: envelope-avg {v['rate_env']:.3f}  "
              f"peak cell {v['rate_peak']:.3f}  "
              f"AIM POINT/best frames {v['rate_aim']:.3f}  <- compare to p_remove")
    print(f"  paper Table 3 (moving car, RP=100/>=95/>=90): "
          f"{PAPER_TABLE3['car']}")
    print(f"  paper max demonstrated attack angle: "
          f"{PAPER_MAX_ATTACK_ANGLE_DEG:.0f} deg "
          f"(LGSVL end-to-end runs used 5 and 10 deg)")

    print(f"\n{'split':16s} {'n':>5s} {'width':>7s} {'p_cfg':>6s} "
          f"{'rate':>7s} {'IQR':>15s} {'pts_rm':>7s} {'in_sec':>7s} "
          f"{'chan':>6s} {'cloud%':>7s} {'chord@25m':>10s} {'':>5s}")
    print("-" * 118)
    for w in (10, 20, 30, 40, 50, 60):
        sd = os.path.join(a.nuscenes_dir, f"removal_az{w}")
        if not os.path.isdir(sd):
            continue
        rows = scan_split(sd, inc, dtheta, a.limit)
        if not rows:
            print(f"removal_az{w:<6d} no active frames found")
            continue
        rate = np.array([r["rate"] for r in rows])
        rm = np.array([r["removed"] for r in rows], float)
        ins = np.array([r["in_sector"] for r in rows], float)
        nb = np.array([r["n_before"] for r in rows], float)
        nch = np.array([r["nch"] for r in rows], float)
        cloud = 100 * rm / np.maximum(nb, 1)
        wmed = np.median([r["width_deg"] for r in rows])
        chord = 2 * 25.0 * np.sin(np.deg2rad(wmed) / 2)
        flag = "EXTRAP" if wmed > PAPER_MAX_ATTACK_ANGLE_DEG else ""
        print(f"removal_az{w:<6d} {len(rows):5d} {wmed:7.1f} "
              f"{np.median([r['p_remove_cfg'] for r in rows]):6.2f} "
              f"{np.median(rate):7.3f} "
              f"[{np.percentile(rate,25):.3f},{np.percentile(rate,75):.3f}]".ljust(16)
              + f"{np.median(rm):7.0f} {np.median(ins):7.0f} "
              f"{np.nanmedian(nch):6.1f} {np.median(cloud):7.2f} "
              f"{chord:10.2f} {flag:>6s}")
    print("\nEXTRAP = azimuth width exceeds the 45 deg max attack angle "
          "demonstrated in Cao et al.; treat as capability extrapolation.")


if __name__ == "__main__":
    main()
