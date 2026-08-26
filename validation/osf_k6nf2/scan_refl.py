#!/usr/bin/env python3
"""Find injected returns by reflectivity saturation.

An injection attack drives the receiver with a laser far brighter than any
diffuse surface, so spoofed returns pin the 8-bit reflectivity byte near 255
while real diffuse targets sit well below. This scans for high-reflectivity
returns and reports their azimuth sector, range and per-frame timeline.
"""
import sys
from collections import defaultdict

import numpy as np

from parse_velo import iter_packets, decode_blocks

VLP16_INCL = np.array([-15, 1, -13, 3, -11, 5, -9, 7,
                       -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float64)


def main(path, rmin=200, nbins=72, show_frames=40):
    prev_az = None
    fidx = 0
    cur = []
    cur_tot = 0
    timeline = []
    az_hist = defaultdict(int)
    refl_hist = np.zeros(256, dtype=np.int64)
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        for bi, (az, chans) in enumerate(blocks):
            if prev_az is not None and az - prev_az < -180:
                timeline.append((fidx, cur_tot, np.array(cur) if cur else None))
                fidx += 1
                cur, cur_tot = [], 0
            prev_az = az
            for ci, (dist, refl) in enumerate(chans):
                if dist <= 0:
                    continue
                cur_tot += 1
                refl_hist[refl] += 1
                if refl >= rmin:
                    cur.append((az, ci % 16, dist, refl))
                    az_hist[int(az // 5) * 5] += 1
    timeline.append((fidx, cur_tot, np.array(cur) if cur else None))

    tot_hi = sum(len(a) for _, _, a in timeline if a is not None)
    print(f"\n=== {path}  (reflectivity >= {rmin}) ===")
    print(f"  frames {len(timeline)}   high-refl returns {tot_hi}")
    print(f"  reflectivity distribution: "
          f"0-49:{refl_hist[:50].sum()}  50-99:{refl_hist[50:100].sum()}  "
          f"100-199:{refl_hist[100:200].sum()}  200-254:{refl_hist[200:255].sum()}  "
          f"255:{refl_hist[255]}")
    if not tot_hi:
        print("  none")
        return
    print("  azimuth sectors (deg: count), top 10:")
    for az, c in sorted(az_hist.items(), key=lambda kv: -kv[1])[:10]:
        print(f"      {az:3d}-{az+5:3d}: {c}")
    nz = [(f, len(a)) for f, tot, a in timeline if a is not None and len(a)]
    print(f"  frames with high-refl returns: {len(nz)}/{len(timeline)}")
    print("      " + ", ".join(f"{f}:{c}" for f, c in nz[:show_frames])
          + (" ..." if len(nz) > show_frames else ""))
    best = max(timeline, key=lambda t: 0 if t[2] is None else len(t[2]))
    A = best[2]
    print(f"  --- peak frame {best[0]}: {len(A)} high-refl of {best[1]} total ---")
    az = A[:, 0]
    # unwrap in case the cluster straddles 0/360
    if az.max() - az.min() > 180:
        az = np.where(az > 180, az - 360, az)
    print(f"      azimuth {az.min():.1f}..{az.max():.1f} deg (span {az.max()-az.min():.1f})")
    print(f"      range   {A[:,2].min():.1f}..{A[:,2].max():.1f} m "
          f"(mean {A[:,2].mean():.2f} std {A[:,2].std():.2f})")
    print(f"      channels {np.unique(A[:,1].astype(int)).tolist()}")
    return timeline


if __name__ == "__main__":
    args = sys.argv[1:]
    rmin = 200
    if args and args[0].isdigit():
        rmin = int(args[0]); args = args[1:]
    for p in args:
        main(p, rmin)
