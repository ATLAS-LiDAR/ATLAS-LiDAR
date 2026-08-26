#!/usr/bin/env python3
"""Dump the range/reflectivity structure of one azimuth sector, per return type.

Used to characterise the injected object in the Multimode Analysis captures,
where the attack sits in a fixed azimuth wedge and the question is what the
strongest-vs-last return pair looks like inside it.
"""
import sys
from collections import defaultdict

import numpy as np

from parse_velo import iter_packets, decode_blocks


def collect(path, az_lo, az_hi, max_frames=None):
    prev_az = None
    fidx = 0
    rows = []          # (frame, az, ch, dist, refl, ret_idx)
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        dual = rmode == 0x39
        for bi, (az, chans) in enumerate(blocks):
            if prev_az is not None and az - prev_az < -180:
                fidx += 1
                if max_frames and fidx >= max_frames:
                    return np.array(rows), rmode
            prev_az = az
            if not (az_lo <= az <= az_hi):
                continue
            ridx = (bi % 2) if dual else 0
            for ci, (dist, refl) in enumerate(chans):
                if dist > 0:
                    rows.append((fidx, az, ci % 16, dist, refl, ridx))
    return np.array(rows), rmode


def report(path, az_lo, az_hi, max_frames=120):
    R, rmode = collect(path, az_lo, az_hi, max_frames)
    mode = {0x37: "strongest", 0x38: "last", 0x39: "dual"}.get(rmode, hex(rmode))
    print(f"\n=== {path}  sector {az_lo}-{az_hi} deg  mode={mode} ===")
    if not len(R):
        print("  no returns"); return
    print(f"  returns {len(R)} over {int(R[:,0].max())+1} frames")
    for ri in sorted(set(R[:, 5].astype(int))):
        S = R[R[:, 5] == ri]
        tag = {0: "last" if rmode == 0x39 else mode, 1: "strongest"}[ri] \
            if rmode == 0x39 else mode
        print(f"  -- return set '{tag}' : {len(S)}")
        d, rf = S[:, 3], S[:, 4]
        print(f"     range  min {d.min():.2f} p25 {np.percentile(d,25):.2f} "
              f"med {np.median(d):.2f} p75 {np.percentile(d,75):.2f} max {d.max():.2f}")
        print(f"     refl   med {np.median(rf):.0f}  mean {rf.mean():.1f}  "
              f">=200: {(rf>=200).sum()} ({(rf>=200).mean()*100:.1f}%)  "
              f"==255: {(rf==255).sum()}")
        # bimodality: histogram of range
        h, e = np.histogram(d, bins=12, range=(0, max(d.max(), 1)))
        print("     range hist: " + ", ".join(
            f"{e[i]:.0f}-{e[i+1]:.0f}m:{h[i]}" for i in range(12) if h[i]))
        # are the bright ones farther or nearer?
        if (rf >= 200).any() and (rf < 200).any():
            print(f"     bright(>=200) mean range {d[rf>=200].mean():.2f} m  vs "
                  f"dim mean {d[rf<200].mean():.2f} m")


if __name__ == "__main__":
    lo, hi = float(sys.argv[1]), float(sys.argv[2])
    for p in sys.argv[3:]:
        report(p, lo, hi)
