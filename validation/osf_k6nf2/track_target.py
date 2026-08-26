#!/usr/bin/env python3
"""Find the moving object in a Velodyne capture by background subtraction.

Builds a per-(azimuth-bin, channel) median range over the whole capture as the
static background, then per frame reports cells whose range departs from that
background -- the moving target. Prints its azimuth/range track so the motion
profile of the experiment is visible.
"""
import sys

import numpy as np

from parse_velo import iter_packets, decode_blocks

NAZ = 360          # 1-degree azimuth bins
NCH = 16


def load(path, max_frames=400):
    """Return (F, NAZ, NCH) range image stack, nan where no return."""
    stack = []
    img = np.full((NAZ, NCH), np.nan, dtype=np.float32)
    ref = np.zeros((NAZ, NCH), dtype=np.float32)
    prev_az = None
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        for az, chans in blocks:
            if prev_az is not None and az - prev_az < -180:
                stack.append((img, ref))
                if len(stack) >= max_frames:
                    return stack
                img = np.full((NAZ, NCH), np.nan, dtype=np.float32)
                ref = np.zeros((NAZ, NCH), dtype=np.float32)
            prev_az = az
            b = int(az) % NAZ
            for ci, (dist, refl) in enumerate(chans):
                if dist > 0:
                    c = ci % NCH
                    img[b, c] = dist
                    ref[b, c] = refl
    stack.append((img, ref))
    return stack


def main(path, max_frames=250, tol=1.0):
    stack = load(path, max_frames)
    R = np.stack([s[0] for s in stack])          # (F, NAZ, NCH)
    L = np.stack([s[1] for s in stack])
    print(f"\n=== {path} ===")
    print(f"  frames loaded {len(stack)}")
    bg = np.nanmedian(R, axis=0)                 # static background
    with np.errstate(invalid="ignore"):
        dev = np.abs(R - bg[None]) > tol
    dev &= ~np.isnan(R)
    print(f"  cells deviating from background: {dev.sum()} "
          f"({dev.mean()*100:.2f}% of filled cells)")
    print("  per-frame moving-cluster centroid (frame: az_deg, range_m, ncell, refl):")
    lines = 0
    for f in range(len(stack)):
        idx = np.argwhere(dev[f])
        if len(idx) < 8:
            continue
        az = idx[:, 0].astype(float)
        # unwrap if straddling 0
        if az.max() - az.min() > 180:
            az = np.where(az > 180, az - 360, az)
        rr = R[f][dev[f]]
        ll = L[f][dev[f]]
        # keep the dominant azimuth cluster
        med = np.median(az)
        keep = np.abs(az - med) < 20
        if keep.sum() < 8:
            continue
        if lines < 40 and f % max(1, len(stack) // 40) == 0:
            print(f"      {f:4d}: az {np.median(az[keep])%360:6.1f}  "
                  f"r {np.median(rr[keep]):6.2f}  n {keep.sum():4d}  "
                  f"refl {ll[keep].mean():5.1f}")
            lines += 1


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
