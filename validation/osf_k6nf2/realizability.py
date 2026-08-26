#!/usr/bin/env python3
"""Realizability analysis: are ATLAS synthetic phantoms within the physical
capability envelope demonstrated by a real LiDAR injection attack?

The right question is NOT "do the two point clouds look the same" -- the real
bench attack and a nuScenes phantom are different objects seen by different
sensors. The question an attacker cares about is capability:

  1. ANGULAR FOOTPRINT. A spoofer can only inject where its beam illuminates
     the victim aperture, giving a bounded azimuth sector. If a synthetic
     phantom spans a wider sector than the real attack achieved, it is not
     realizable from a single spoofer position.

  2. INJECTION DENSITY. The hard physical limit is one spoofed return per
     (azimuth cell x beam) -- the attacker cannot create returns in cells the
     victim never samples. So the meaningful quantity is

         fill = n_spoof_points / (n_az_cells * n_beams_spanned)

     fill <= 1.0 is realizable; the real attack shows what fill is achieved in
     practice. A synthetic phantom with a LOWER fill than the real attack is
     strictly easier to produce than what has already been demonstrated.

  3. FIRING BUDGET. Points the attacker must inject per revolution, versus what
     the demonstrated attack injected.

This script computes those three quantities for the real capture and for every
ATLAS phantom, and reports what fraction of ATLAS phantoms are inside the
demonstrated envelope.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SENSORS = {
    # name: (n_beams, vfov_deg, azimuth_resolution_deg)
    "vlp16":  (16, 30.0,  0.40),      # real bench capture, 10 Hz
    "hdl32":  (32, 41.33, 0.3316),    # nuScenes
    "hdl64":  (64, 26.9,  0.1728),    # KITTI
}


def cells_and_fill(az_span_deg, n_beams_spanned, npts, azres):
    n_az = max(az_span_deg / azres, 1.0)
    cells = n_az * max(n_beams_spanned, 1)
    return n_az, cells, npts / cells


def real_envelope(npz_files, sensor="vlp16", dual=None):
    nbeam, vfov, azres = SENSORS[sensor]
    rows = []
    for f in npz_files:
        d = np.load(f)
        off = d["offsets"]
        is_dual = ("dual" in os.path.basename(f)) if dual is None else dual
        for i in range(len(d["frame"])):
            s, e = off[i], off[i + 1]
            if e - s < 20:
                continue
            az, ch, rg = d["az"][s:e], d["ch"][s:e], d["rng"][s:e]
            span = float(az.max() - az.min())
            nb = len(np.unique(ch))
            npts = e - s
            # in dual-return mode each cell can carry two returns; normalise to
            # distinct (az,beam) cells actually occupied so fill is comparable
            uniq = len(set(zip(np.round(az / azres).astype(int), ch.tolist())))
            n_az, cells, fill = cells_and_fill(span, nb, npts, azres)
            rows.append(dict(span=span, nbeam=nb, npts=npts, cells=cells,
                             fill=fill, uniq=uniq,
                             fill_uniq=uniq / cells, rng=float(np.median(rg)),
                             dual=is_dual))
    return rows


def atlas_envelope(traces, sensor):
    nbeam, vfov, azres = SENSORS[sensor]
    pitch = vfov / nbeam
    rows = []
    for t in traces:
        p = np.asarray(t["points"], dtype=np.float64)
        if p.shape[0] < 3:
            continue
        xyz = p[:, :3]
        r = np.linalg.norm(xyz, axis=1)
        az = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
        if az.max() - az.min() > 180:
            az = np.where(az < 0, az + 360, az)
        el = np.rad2deg(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        span = float(az.max() - az.min())
        nb = len(np.unique(np.round(el / pitch)))
        n_az, cells, fill = cells_and_fill(span, nb, p.shape[0], azres)
        rows.append(dict(span=span, nbeam=nb, npts=p.shape[0], cells=cells,
                         fill=fill, rng=float(np.median(r))))
    return rows


def summarise(rows, label, keys=("span", "nbeam", "npts", "fill")):
    print(f"\n  {label}  (n={len(rows)})")
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"    {k:10s} med {np.median(v):9.3f}  p5 {np.percentile(v,5):9.3f}"
              f"  p95 {np.percentile(v,95):9.3f}  max {v.max():9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--sensor", default="hdl32", choices=["hdl32", "hdl64"])
    ap.add_argument("--label", default="nuScenes")
    a = ap.parse_args()

    R = real_envelope(a.real)
    print("=" * 78)
    print(" DEMONSTRATED PHYSICAL CAPABILITY  (real VLP-16 injection capture)")
    print("=" * 78)
    summarise(R, "real spoofed clusters")
    fu = np.array([r["fill_uniq"] for r in R])
    fl = np.array([r["fill"] for r in R])
    print(f"\n    occupancy of available (az x beam) cells : "
          f"med {np.median(fu):.3f}  p95 {np.percentile(fu,95):.3f}")
    print(f"    returns per cell incl. dual returns      : "
          f"med {np.median(fl):.3f}  max {fl.max():.3f}")
    # the envelope: the attack demonstrated it can fill this much
    env_span = np.percentile([r["span"] for r in R], 95)
    env_fill = np.percentile(fu, 95)
    env_npts = np.percentile([r["npts"] for r in R], 95)
    print(f"\n    >> ENVELOPE (p95): az span <= {env_span:.2f} deg, "
          f"cell fill <= {env_fill:.3f}, {env_npts:.0f} pts/revolution")

    print("\n" + "=" * 78)
    print(f" ATLAS SYNTHETIC PHANTOMS ({a.label}, {a.sensor})")
    print("=" * 78)
    print(f"\n{'trace':8s} {'n':>6s} {'az span med':>12s} {'fill med':>9s} "
          f"{'fill p95':>9s} {'% span ok':>10s} {'% fill ok':>10s} {'% both':>8s}")
    print("-" * 78)
    allrows = {}
    for name in ("easy", "medium", "hard"):
        path = os.path.join(a.traces, f"trace_{name}.pkl")
        if not os.path.exists(path):
            continue
        tr = pickle.load(open(path, "rb"))
        S = atlas_envelope(tr, a.sensor)
        allrows[name] = S
        span = np.array([r["span"] for r in S])
        fill = np.array([r["fill"] for r in S])
        ok_s = (span <= env_span).mean() * 100
        ok_f = (fill <= env_fill).mean() * 100
        ok_b = ((span <= env_span) & (fill <= env_fill)).mean() * 100
        print(f"{name:8s} {len(S):6d} {np.median(span):12.2f} "
              f"{np.median(fill):9.3f} {np.percentile(fill,95):9.3f} "
              f"{ok_s:9.1f}% {ok_f:9.1f}% {ok_b:7.1f}%")

    print("\n  Interpretation: 'fill' is the fraction of victim-sampled")
    print("  (azimuth x beam) cells the attacker must occupy. Values below the")
    print("  demonstrated envelope mean the synthetic phantom is EASIER to")
    print("  physically produce than the attack already realised on hardware.")

    # physical hard limit check
    print("\n" + "=" * 78)
    print(" HARD PHYSICAL LIMIT CHECK  (fill > 1.0 is impossible)")
    print("=" * 78)
    for name, S in allrows.items():
        fill = np.array([r["fill"] for r in S])
        viol = (fill > 1.0)
        print(f"  trace_{name:7s}: {viol.sum():5d}/{len(S)} "
              f"({viol.mean()*100:5.2f}%) exceed 1 point per cell"
              f"   max fill {fill.max():.2f}")


if __name__ == "__main__":
    main()
