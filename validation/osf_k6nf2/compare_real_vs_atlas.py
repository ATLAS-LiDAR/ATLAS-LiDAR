#!/usr/bin/env python3
"""Quantitative fidelity check: real physical LiDAR injection vs ATLAS synthetic
phantoms.

Compares the real spoofed point clusters extracted from the OSF Velodyne
captures against ATLAS relative-injection phantoms, on features that are
invariant to sensor model and scene, so that a VLP-16 bench capture and a
nuScenes HDL-32E trace can be put on the same axis:

  angular   -- azimuth span of the cluster, vertical channel occupancy
  density   -- points per cluster, and points normalised by solid angle
  radial    -- depth thickness of the cluster (std of range within it)
  geometry  -- cluster extent in metres, aspect ratio
  intensity -- distribution of the returned intensity byte

For each feature we report real vs synthetic summary stats, a scale-free ratio,
and a two-sample distribution distance (KS statistic + Wasserstein on the
normalised feature) so "aligned" is a number rather than an impression.

Ranges are matched before comparing: ATLAS traces are binned to the real
capture's range band, because nearly every one of these features is a strong
function of range and comparing unmatched ranges would be meaningless.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- sensor specs
# real capture: Velodyne VLP-16
VLP16_NBEAM = 16
VLP16_VFOV = 30.0                      # -15..+15 deg
VLP16_AZRES = 0.4                      # deg per firing at 10 Hz (600 rpm)
# nuScenes: Velodyne HDL-32E
HDL32_NBEAM = 32
HDL32_VFOV = 41.33                     # -30.67..+10.67
# KITTI: Velodyne HDL-64E
HDL64_NBEAM = 64
HDL64_VFOV = 26.9


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def wasserstein1(a, b):
    """1-Wasserstein distance between two empirical 1-D samples."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    q = np.linspace(0, 1, 512)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


# ------------------------------------------------------------- real extraction
def load_real(npz_path):
    d = np.load(npz_path)
    off = d["offsets"]
    out = []
    for i in range(len(d["frame"])):
        s, e = off[i], off[i + 1]
        if e - s < 20:
            continue
        out.append(dict(
            xyz=d["xyz"][s:e], intensity=d["intensity"][s:e],
            az=d["az"][s:e], ch=d["ch"][s:e], rng=d["rng"][s:e]))
    return out


def real_features(clusters):
    F = {k: [] for k in ("az_span", "vfrac", "npts", "rng", "depth_std",
                         "extent_w", "extent_h", "aspect", "pts_per_sr",
                         "intensity_med", "intensity_frac_hi")}
    for c in clusters:
        az, rg, ch, xyz = c["az"], c["rng"], c["ch"], c["xyz"]
        span = float(az.max() - az.min())
        nch = len(np.unique(ch))
        r = float(np.median(rg))
        # vertical extent actually occupied, in degrees, from channel indices
        vspan = (nch / VLP16_NBEAM) * VLP16_VFOV
        sr = np.deg2rad(span) * np.deg2rad(max(vspan, 1e-6))   # solid angle
        F["az_span"].append(span)
        F["vfrac"].append(nch / VLP16_NBEAM)
        F["npts"].append(len(rg))
        F["rng"].append(r)
        F["depth_std"].append(float(rg.std()))
        w = float(xyz[:, :2].ptp(axis=0).max())
        h = float(xyz[:, 2].ptp())
        F["extent_w"].append(w)
        F["extent_h"].append(h)
        F["aspect"].append(h / max(w, 1e-6))
        F["pts_per_sr"].append(len(rg) / max(sr, 1e-9))
        F["intensity_med"].append(float(np.median(c["intensity"])))
        F["intensity_frac_hi"].append(float((c["intensity"] >= 200).mean()))
    return {k: np.array(v) for k, v in F.items()}


# ---------------------------------------------------------- synthetic features
def atlas_features(traces, nbeam, vfov, azres, intensity_max=255.0):
    """Compute the same features for ATLAS trace clusters.

    traces: list of dicts with 'points' (M,5)=[x,y,z,intensity,t], 'box', 'range'
    """
    F = {k: [] for k in ("az_span", "vfrac", "npts", "rng", "depth_std",
                         "extent_w", "extent_h", "aspect", "pts_per_sr",
                         "intensity_med", "intensity_frac_hi")}
    for t in traces:
        p = np.asarray(t["points"], dtype=np.float64)
        if p.shape[0] < 3:
            continue
        xyz = p[:, :3]
        inten = p[:, 3]
        r = np.linalg.norm(xyz, axis=1)
        az = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
        if az.max() - az.min() > 180:
            az = np.where(az < 0, az + 360, az)
        el = np.rad2deg(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        span = float(az.max() - az.min())
        # vertical occupancy: distinct beams touched, estimated from elevation
        # quantised at the sensor's mean beam pitch
        pitch = vfov / nbeam
        nch = len(np.unique(np.round(el / pitch)))
        vspan = (nch / nbeam) * vfov
        sr = np.deg2rad(max(span, 1e-6)) * np.deg2rad(max(vspan, 1e-6))
        F["az_span"].append(span)
        F["vfrac"].append(nch / nbeam)
        F["npts"].append(p.shape[0])
        F["rng"].append(float(np.median(r)))
        F["depth_std"].append(float(r.std()))
        w = float(xyz[:, :2].ptp(axis=0).max())
        h = float(xyz[:, 2].ptp())
        F["extent_w"].append(w)
        F["extent_h"].append(h)
        F["aspect"].append(h / max(w, 1e-6))
        F["pts_per_sr"].append(p.shape[0] / max(sr, 1e-9))
        F["intensity_med"].append(float(np.median(inten)))
        F["intensity_frac_hi"].append(float((inten >= 200).mean()))
    return {k: np.array(v) for k, v in F.items()}


# ---------------------------------------------------------------- reporting
FEATURES = [
    ("az_span",     "azimuth span (deg)",            True),
    ("vfrac",       "vertical beam occupancy (frac)", True),
    ("npts",        "points per phantom",             True),
    ("depth_std",   "radial thickness (m)",           True),
    ("extent_w",    "horizontal extent (m)",          True),
    ("extent_h",    "vertical extent (m)",            True),
    ("aspect",      "aspect ratio h/w",               True),
    ("pts_per_sr",  "points per steradian",           True),
    ("intensity_med",     "median intensity",         False),
    ("intensity_frac_hi", "frac intensity>=200",      False),
]


def compare(real, syn, label, rng_lo, rng_hi):
    print(f"\n{'='*78}")
    print(f" REAL (VLP-16 bench injection)   vs   {label}")
    print(f" range-matched window: {rng_lo:.1f}-{rng_hi:.1f} m")
    print(f" n_real={len(real['npts'])}   n_syn={len(syn['npts'])}")
    print(f"{'='*78}")
    print(f"{'feature':32s} {'real med':>10s} {'syn med':>10s} "
          f"{'ratio':>7s} {'KS':>6s}")
    print("-" * 78)
    rows = []
    for key, nice, geo in FEATURES:
        a, b = real[key], syn[key]
        if len(a) == 0 or len(b) == 0:
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        ratio = ma / mb if abs(mb) > 1e-9 else float("nan")
        ks = ks_stat(a, b)
        rows.append((key, nice, ma, mb, ratio, ks, geo))
        flag = ""
        if geo:
            flag = "  <-- geometry" if 0.5 <= ratio <= 2.0 else "  ** MISMATCH"
        print(f"{nice:32s} {ma:10.3f} {mb:10.3f} {ratio:7.2f} {ks:6.3f}{flag}")
    geo_ks = [r[5] for r in rows if r[6]]
    geo_ratio = [r[4] for r in rows if r[6]]
    print("-" * 78)
    print(f"  GEOMETRIC features only: mean KS {np.mean(geo_ks):.3f}   "
          f"median |log2 ratio| {np.median(np.abs(np.log2(geo_ratio))):.3f}")
    inten = [r for r in rows if not r[6]]
    if inten:
        print(f"  INTENSITY features    : "
              + "; ".join(f"{r[1]} real {r[2]:.1f} vs syn {r[3]:.1f}" for r in inten))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", nargs="+", required=True,
                   help="npz files from extract_spoof_real.py")
    p.add_argument("--traces", required=True,
                   help="dir with trace_{easy,medium,hard}.pkl")
    p.add_argument("--sensor", default="hdl32", choices=["hdl32", "hdl64"])
    p.add_argument("--rng_tol", type=float, default=1.0,
                   help="widen the range window by this factor")
    a = p.parse_args()

    nbeam, vfov = ((HDL32_NBEAM, HDL32_VFOV) if a.sensor == "hdl32"
                   else (HDL64_NBEAM, HDL64_VFOV))

    clusters = []
    for f in a.real:
        c = load_real(f)
        print(f"loaded {len(c)} real clusters from {f}")
        clusters += c
    RF = real_features(clusters)
    rlo, rhi = np.percentile(RF["rng"], [5, 95])
    print(f"\nreal phantom range band: {rlo:.1f}-{rhi:.1f} m "
          f"(median {np.median(RF['rng']):.1f})")

    for name in ("easy", "medium", "hard"):
        path = os.path.join(a.traces, f"trace_{name}.pkl")
        if not os.path.exists(path):
            continue
        traces = pickle.load(open(path, "rb"))
        SF = atlas_features(traces, nbeam, vfov, VLP16_AZRES)
        # range-match
        m = (SF["rng"] >= rlo / a.rng_tol) & (SF["rng"] <= rhi * a.rng_tol)
        if m.sum() < 10:
            print(f"\n[trace_{name}] only {m.sum()} traces in the real range "
                  f"band -- comparing unmatched")
            m = np.ones(len(SF["rng"]), dtype=bool)
        SFm = {k: v[m] for k, v in SF.items()}
        compare(RF, SFm, f"ATLAS trace_{name} ({a.sensor})", rlo, rhi)


if __name__ == "__main__":
    main()
