#!/usr/bin/env python3
"""Generate src/core/sensor-assets.js from the ATLAS generation assets.

The sandbox must agree numerically with the ATLAS pipeline, so beam
inclinations, azimuth resolution and the range-conditioned intensity statistics
are emitted from the same .npy files the generator loads rather than being
transcribed by hand.

Usage:
  python3 tools/gen_sensor_assets.py [--phm /path/to/Polar-Hierarchical-Mamba]
"""
import argparse
import hashlib
import os

import numpy as np

DEFAULT_PHM = os.path.expanduser("~/rsch/Polar-Hierarchical-Mamba")


def sha8(path):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return h[:8]


def fmt_floats(arr, per_line=6, indent=4, prec=9):
    out, line = [], []
    for i, v in enumerate(arr):
        line.append(f"{v:.{prec}f}")
        if len(line) == per_line and i != len(arr) - 1:
            out.append(" " * indent + ", ".join(line) + ",")
            line = []
    if line:
        out.append(" " * indent + ", ".join(line))
    return "\n".join(out)


def load_bands(path):
    stats = np.load(path, allow_pickle=True).item()
    bands = []
    for (lo, hi), v in sorted(stats.items()):
        bands.append(
            dict(
                rMin=float(lo),
                rMax=float(hi),
                logMu=float(v["log_mu"]),
                logSigma=float(v["log_sigma"]),
                p1=float(v["intensity_p1"]),
                p99=float(v["intensity_p99"]),
                n=int(v["n"]),
            )
        )
    return bands


def js_bands(bands, indent=6):
    pad = " " * indent
    rows = []
    for b in bands:
        rows.append(
            pad
            + "{ rMin: %g, rMax: %g, logMu: %.9f, logSigma: %.9f, p1: %g, p99: %g, n: %d }"
            % (b["rMin"], b["rMax"], b["logMu"], b["logSigma"], b["p1"], b["p99"], b["n"])
        )
    return ",\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phm", default=DEFAULT_PHM)
    ap.add_argument("--out", default="src/core/sensor-assets.js")
    a = ap.parse_args()

    gen = os.path.join(a.phm, "atlas_generation")
    src = {
        "nuscenes": dict(
            inc=os.path.join(gen, "nuscenes_beam_inclinations.npy"),
            az=os.path.join(gen, "nuscenes_azimuth_res.npy"),
            st=os.path.join(gen, "nuscenes_range_conditioned_stats.npy"),
        ),
        "kitti": dict(
            inc=os.path.join(gen, "assets_kitti", "kitti_beam_inclinations.npy"),
            az=os.path.join(gen, "assets_kitti", "kitti_azimuth_res.npy"),
            st=os.path.join(gen, "assets_kitti", "kitti_range_conditioned_stats.npy"),
        ),
    }

    for tag, paths in src.items():
        for k, p in paths.items():
            if not os.path.exists(p):
                raise SystemExit(f"missing asset: {p}")

    ns_inc = np.load(src["nuscenes"]["inc"]).astype(float)
    ns_az = float(np.load(src["nuscenes"]["az"])[0])
    ki_inc = np.load(src["kitti"]["inc"]).astype(float)
    ki_az = float(np.load(src["kitti"]["az"])[0])
    ns_bands = load_bands(src["nuscenes"]["st"])
    ki_bands = load_bands(src["kitti"]["st"])

    prov = []
    for tag, paths in src.items():
        for k in ("inc", "az", "st"):
            p = paths[k]
            prov.append(f" *   {os.path.relpath(p, a.phm)}  sha256:{sha8(p)}")
    prov = "\n".join(prov)

    body = f"""/**
 * GENERATED FILE - do not edit by hand.
 * Regenerate with:  python3 tools/gen_sensor_assets.py
 *
 * Beam inclinations, azimuth resolution and range-conditioned intensity
 * statistics measured from the real datasets, emitted from the same .npy assets
 * that atlas_generation/*_atlas_common.py loads. Sources:
{prov}
 *
 * `provenance: "measured"` means the inclinations were fitted from real dataset
 * returns. `"nominal"` means they are datasheet values expanded uniformly across
 * the vertical FOV - honest to the spec sheet, but not measured hardware.
 */

/** nuScenes LiDAR: Velodyne HDL-32E, 32 beams, measured inclinations (rad). */
export const HDL32E_INCLINATIONS = new Float64Array([
{fmt_floats(ns_inc)}
]);

/** KITTI LiDAR: Velodyne HDL-64E, 64 beams, measured inclinations (rad). */
export const HDL64E_INCLINATIONS = new Float64Array([
{fmt_floats(ki_inc)}
]);

/** Measured azimuth resolution (radians per cell). */
export const HDL32E_AZ_RES = {ns_az:.10f};
export const HDL64E_AZ_RES = {ki_az:.10f};

/**
 * Range-conditioned intensity statistics, per dataset. A phantom's intensity is
 * drawn lognormal(logMu, logSigma) clipped to [p1, p99] for the band containing
 * its mean range - this is what keeps an injected object in-distribution.
 *
 * Note the differing scales: nuScenes intensity is 0-255, KITTI is normalised
 * 0-1. ATLAS_REAL_WORLD_VALIDATION.md measures detector ASR collapsing past
 * I ~= 100 on nuScenes, which is exactly the p99 of real car returns.
 */
export const INTENSITY_BANDS = {{
  nuscenes: {{
    scaleMax: 255,
    bands: [
{js_bands(ns_bands)}
    ],
  }},
  kitti: {{
    scaleMax: 1.0,
    bands: [
{js_bands(ki_bands)}
    ],
  }},
}};

/** Uniformly spaced inclinations across a vertical FOV, ascending (radians). */
export function uniformInclinations(nBeams, vfovMinDeg, vfovMaxDeg) {{
  const out = new Float64Array(nBeams);
  const lo = (vfovMinDeg * Math.PI) / 180;
  const hi = (vfovMaxDeg * Math.PI) / 180;
  const step = nBeams > 1 ? (hi - lo) / (nBeams - 1) : 0;
  for (let i = 0; i < nBeams; i++) out[i] = lo + i * step;
  return out;
}}
"""
    with open(a.out, "w") as f:
        f.write(body)
    print(f"wrote {a.out}")
    print(f"  HDL-32E: {len(ns_inc)} beams, az_res {np.rad2deg(ns_az):.5f} deg")
    print(f"  HDL-64E: {len(ki_inc)} beams, az_res {np.rad2deg(ki_az):.5f} deg")
    print(f"  intensity bands: nuscenes {len(ns_bands)}, kitti {len(ki_bands)}")


if __name__ == "__main__":
    main()
