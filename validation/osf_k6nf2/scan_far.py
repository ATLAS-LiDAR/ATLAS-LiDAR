#!/usr/bin/env python3
"""Scan a full Velodyne pcap for the azimuth/range/time footprint of returns
beyond a background threshold, printing a per-frame timeline so an injected
object shows up as a burst confined to a narrow azimuth sector."""
import sys
from collections import defaultdict

import numpy as np

from parse_velo import iter_packets, decode_blocks

VLP16_INCL = np.array([-15, 1, -13, 3, -11, 5, -9, 7,
                       -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float64)


def main(path, thresh=8.0, nbins=72):
    prev_az = None
    fidx = 0
    cur_far = []
    cur_tot = 0
    timeline = []
    az_hist = defaultdict(int)
    ch_hist = defaultdict(int)
    rng_all = []
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        for bi, (az, chans) in enumerate(blocks):
            if prev_az is not None and az - prev_az < -180:
                timeline.append((fidx, cur_tot, len(cur_far),
                                 np.array(cur_far) if cur_far else None))
                fidx += 1
                cur_far, cur_tot = [], 0
            prev_az = az
            for ci, (dist, refl) in enumerate(chans):
                if dist <= 0:
                    continue
                cur_tot += 1
                if dist > thresh:
                    cur_far.append((az, ci % 16, dist, refl))
                    az_hist[int(az // 5) * 5] += 1
                    ch_hist[ci % 16] += 1
                    rng_all.append(dist)
    timeline.append((fidx, cur_tot, len(cur_far),
                     np.array(cur_far) if cur_far else None))

    print(f"\n=== {path}  (returns > {thresh} m) ===")
    print(f"  frames: {len(timeline)}   far returns: {sum(t[2] for t in timeline)}")
    if not rng_all:
        print("  none")
        return
    R = np.array(rng_all)
    print(f"  far range: min {R.min():.1f}  med {np.median(R):.1f}  "
          f"max {R.max():.1f} m")
    print("  azimuth sectors (deg: count), top 10:")
    for az, c in sorted(az_hist.items(), key=lambda kv: -kv[1])[:10]:
        print(f"      {az:3d}-{az+5:3d}: {c}")
    print("  channels (elev: count), top 8:")
    for ch, c in sorted(ch_hist.items(), key=lambda kv: -kv[1])[:8]:
        print(f"      ch{ch:2d} ({VLP16_INCL[ch]:+5.1f} deg): {c}")

    print("  per-frame far-count timeline (frame:count, nonzero only):")
    nz = [(f, c) for f, tot, c, arr in timeline if c]
    line = ", ".join(f"{f}:{c}" for f, c in nz[:60])
    print("      " + line + (" ..." if len(nz) > 60 else ""))
    # the frame with the most far returns: dump its cluster
    best = max(timeline, key=lambda t: t[2])
    if best[3] is not None and best[2] > 5:
        A = best[3]
        print(f"  --- frame {best[0]} has {best[2]} far returns "
              f"(of {best[1]} total) ---")
        print(f"      azimuth {A[:,0].min():.1f}-{A[:,0].max():.1f} deg "
              f"(span {A[:,0].max()-A[:,0].min():.1f})")
        print(f"      range   {A[:,2].min():.1f}-{A[:,2].max():.1f} m "
              f"(mean {A[:,2].mean():.2f}, std {A[:,2].std():.2f})")
        print(f"      refl    {A[:,3].min():.0f}-{A[:,3].max():.0f} "
              f"(mean {A[:,3].mean():.1f})")
        chs = np.unique(A[:, 1].astype(int))
        print(f"      channels {chs.tolist()}")


if __name__ == "__main__":
    thresh = 8.0
    args = sys.argv[1:]
    if args and args[0].replace(".", "").isdigit():
        thresh = float(args[0]); args = args[1:]
    for p in args:
        main(p, thresh)
