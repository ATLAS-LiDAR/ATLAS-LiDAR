#!/usr/bin/env python3
"""Measure the REAL removal signal in the PRA capture (Cao et al., USENIX Sec '23).

The attack injects saturated echoes below the LiDAR's Minimum Operational
Threshold (MOT). Because the sensor prioritises the strongest echo and the
firmware discards sub-MOT returns, the GENUINE returns on those rays vanish.
So the observable is not the spoof points -- it is a WEDGE OF MISSING RETURNS.

Method
------
Build a per-(azimuth_cell, channel) occupancy grid for every revolution:

    occ[az_cell, ch] = 1 if a valid (dist > 0) return was reported

A clean reference is the per-cell probability of a return, estimated from the
frames the attack is NOT active in (identified independently, by absence of
saturated near-field returns). For each attacked frame we then measure, per
azimuth cell, the fraction of the reference channels that went missing:

    depletion(az) = 1 - n_returns_now(az) / n_returns_reference(az)

A contiguous run of high-depletion azimuth cells is the attack wedge. We report
its angular width and the removal rate inside it -- exactly the two quantities
ATLAS's removal model is parameterised by (az_width_deg, p_remove).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_velo import iter_packets, decode_blocks  # noqa: E402

NCH = 16          # VLP-16
MOT_M = 0.40      # VLP-16 internal Minimum Operational Threshold (paper Tab.1)
SAT_REFL = 200    # saturation cue for spoofed echoes


def frames_from_pcap(path, az_res_deg=0.4):
    """Yield per-revolution (occupancy, near_sat_count) arrays.

    occupancy : (n_az, NCH) uint8 -- valid return present
    ranges    : (n_az, NCH) float32 -- range in m (0 = no return)
    refl      : (n_az, NCH) uint8
    """
    n_az = int(round(360.0 / az_res_deg))
    occ = np.zeros((n_az, NCH), np.uint8)
    rng = np.zeros((n_az, NCH), np.float32)
    rfl = np.zeros((n_az, NCH), np.uint8)
    prev_az = None
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, _, rmode, _ = decode_blocks(pl)
        for az, chans in blocks:
            if prev_az is not None and az - prev_az < -180:
                yield occ, rng, rfl
                occ = np.zeros((n_az, NCH), np.uint8)
                rng = np.zeros((n_az, NCH), np.float32)
                rfl = np.zeros((n_az, NCH), np.uint8)
            prev_az = az
            cell = int(az / az_res_deg) % n_az
            # VLP-16 packs two firings (16 ch each) per 32-channel block
            for half in range(2):
                for c in range(NCH):
                    d, r = chans[half * NCH + c]
                    if d > 0.0:
                        occ[cell, c] = 1
                        rng[cell, c] = d
                        rfl[cell, c] = r


def analyse(path, az_res_deg=0.4, depl_thresh=0.5, verbose=True):
    n_az = int(round(360.0 / az_res_deg))
    occ_all, rng_all, rfl_all = [], [], []
    for occ, rng, rfl in frames_from_pcap(path, az_res_deg):
        occ_all.append(occ)
        rng_all.append(rng)
        rfl_all.append(rfl)
    if not occ_all:
        return None
    OCC = np.stack(occ_all)                     # (F, n_az, NCH)
    RNG = np.stack(rng_all)
    RFL = np.stack(rfl_all)
    F = OCC.shape[0]

    # --- identify attacked frames: saturated returns inside the spoof region.
    # The spoof echoes land below the MOT; the sensor reports some of them
    # before filtering, and they are always saturated.
    sat_near = ((RFL >= SAT_REFL) & (RNG > 0)).sum(axis=(1, 2))
    # attacked = frames in the upper mode of the saturated-return count
    thr = max(3, int(np.percentile(sat_near, 60)))
    attacked = sat_near >= thr
    clean = ~attacked
    if clean.sum() < 5 or attacked.sum() < 5:
        # fall back to a split on total returns
        tot = OCC.sum(axis=(1, 2))
        attacked = tot < np.percentile(tot, 40)
        clean = ~attacked

    # --- reference coverage per (az, ch) from clean frames
    ref = OCC[clean].mean(axis=0)                # (n_az, NCH) in [0,1]
    ref_az = ref.sum(axis=1)                    # expected returns per az cell

    rows = []
    for i in np.flatnonzero(attacked):
        now_az = OCC[i].sum(axis=1).astype(float)
        valid = ref_az > 2.0                    # only cells that normally return
        depl = np.zeros(n_az)
        depl[valid] = 1.0 - now_az[valid] / ref_az[valid]
        depl = np.clip(depl, 0, 1)
        # widest contiguous run above threshold (circular)
        hot = depl >= depl_thresh
        if not hot.any():
            continue
        d2 = np.concatenate([hot, hot])
        best_len = best_start = 0
        run = start = 0
        for j, v in enumerate(d2):
            if v:
                if run == 0:
                    start = j
                run += 1
                if run > best_len:
                    best_len, best_start = run, start
            else:
                run = 0
        best_len = min(best_len, n_az)
        seg = [(best_start + k) % n_az for k in range(best_len)]
        width_deg = best_len * az_res_deg
        inside = depl[seg]
        n_ref_in = ref_az[seg].sum()
        n_now_in = now_az[seg].sum()
        rows.append(dict(
            frame=int(i),
            width_deg=width_deg,
            center_deg=((best_start + best_len / 2) % n_az) * az_res_deg,
            p_remove=float(1.0 - n_now_in / max(n_ref_in, 1e-9)),
            depl_mean=float(inside.mean()),
            n_removed=float(n_ref_in - n_now_in),
            n_ref=float(n_ref_in),
        ))

    if verbose:
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"  revolutions          : {F}")
        print(f"  attacked / clean     : {int(attacked.sum())} / {int(clean.sum())}")
        print(f"  wedges measured      : {len(rows)}")
        if rows:
            w = np.array([r["width_deg"] for r in rows])
            p = np.array([r["p_remove"] for r in rows])
            c = np.array([r["center_deg"] for r in rows])
            nr = np.array([r["n_removed"] for r in rows])
            def s(a, n, f="%.2f"):
                print(f"  {n:21s}: med {f % np.median(a)}  "
                      f"IQR [{f % np.percentile(a,25)}, {f % np.percentile(a,75)}]  "
                      f"min {f % a.min()}  max {f % a.max()}")
            s(w, "wedge width (deg)")
            s(p, "removal rate p_remove", "%.3f")
            s(c, "wedge centre (deg)")
            s(nr, "genuine returns lost", "%.0f")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", nargs="+")
    ap.add_argument("--az_res", type=float, default=0.4)
    ap.add_argument("--depl_thresh", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    allrows = {}
    for p in a.pcaps:
        r = analyse(p, a.az_res, a.depl_thresh)
        if r:
            allrows[os.path.basename(p)] = r
    if a.out:
        np.savez_compressed(a.out, **{
            k: np.array([[d["width_deg"], d["p_remove"], d["center_deg"],
                          d["n_removed"], d["n_ref"]] for d in v])
            for k, v in allrows.items()})
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
