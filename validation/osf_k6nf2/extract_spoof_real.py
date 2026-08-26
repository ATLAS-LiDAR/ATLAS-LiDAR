#!/usr/bin/env python3
"""Isolate the injected (spoofed) point set from a real Velodyne capture and
dump it as a per-frame point cloud for comparison against ATLAS phantoms.

Detection rule (deliberately conservative, two independent cues that a real
diffuse surface cannot satisfy simultaneously):
  1. reflectivity >= REFL_MIN     -- an injection laser saturates the receiver
  2. range > RANGE_MIN            -- beyond the physical rig / room background

We then keep only the dominant azimuth cluster per frame, which is the
attacker's angular sector.

Outputs an .npz with, per frame: xyz, intensity, azimuth, channel.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_velo import iter_packets, decode_blocks

VLP16_INCL = np.array([-15, 1, -13, 3, -11, 5, -9, 7,
                       -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float64)


def spherical_to_xyz(r, az_deg, elev_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(elev_deg)
    xy = r * np.cos(el)
    # Velodyne: azimuth measured clockwise from +Y
    x = xy * np.sin(az)
    y = xy * np.cos(az)
    z = r * np.sin(el)
    return x, y, z


def extract(path, refl_min=200, range_min=6.0, min_pts=8, max_frames=None):
    frames = []          # list of dict
    prev_az = None
    cur = []
    fidx = 0

    def flush():
        nonlocal cur
        if len(cur) >= min_pts:
            A = np.array(cur)          # az, ch, dist, refl
            az = A[:, 0].copy()
            if az.max() - az.min() > 180:
                az = np.where(az > 180, az - 360, az)
            med = np.median(az)
            keep = np.abs(az - med) < 25.0        # dominant angular cluster
            if keep.sum() >= min_pts:
                A = A[keep]
                azk = az[keep]
                el = VLP16_INCL[A[:, 1].astype(int)]
                x, y, z = spherical_to_xyz(A[:, 2], A[:, 0], el)
                frames.append(dict(
                    frame=fidx,
                    xyz=np.stack([x, y, z], 1).astype(np.float32),
                    intensity=A[:, 3].astype(np.float32),
                    az=azk.astype(np.float32),
                    ch=A[:, 1].astype(np.int16),
                    rng=A[:, 2].astype(np.float32)))
        cur = []

    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        for az, chans in blocks:
            if prev_az is not None and az - prev_az < -180:
                flush()
                fidx += 1
                if max_frames and fidx >= max_frames:
                    return frames
            prev_az = az
            for ci, (dist, refl) in enumerate(chans):
                if dist > range_min and refl >= refl_min:
                    cur.append((az, ci % 16, dist, refl))
    flush()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pcap")
    p.add_argument("--out", required=True)
    p.add_argument("--refl_min", type=int, default=200)
    p.add_argument("--range_min", type=float, default=6.0)
    p.add_argument("--min_pts", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=None)
    a = p.parse_args()

    fr = extract(a.pcap, a.refl_min, a.range_min, a.min_pts, a.max_frames)
    print(f"{a.pcap}: {len(fr)} frames with an injected cluster")
    if not fr:
        return
    npts = np.array([len(f["xyz"]) for f in fr])
    rng = np.concatenate([f["rng"] for f in fr])
    az = np.concatenate([f["az"] for f in fr])
    inten = np.concatenate([f["intensity"] for f in fr])
    nch = np.array([len(np.unique(f["ch"])) for f in fr])
    azspan = np.array([f["az"].max() - f["az"].min() for f in fr])
    print(f"  pts/frame   : med {np.median(npts):.0f}  "
          f"[{npts.min()}..{npts.max()}]")
    print(f"  range m     : med {np.median(rng):.2f}  "
          f"p5 {np.percentile(rng,5):.2f}  p95 {np.percentile(rng,95):.2f}")
    print(f"  az span deg : med {np.median(azspan):.1f}  max {azspan.max():.1f}")
    print(f"  channels/fr : med {np.median(nch):.0f}  max {nch.max()}")
    print(f"  intensity   : med {np.median(inten):.0f}  "
          f"frac==255 {(inten==255).mean():.3f}")

    np.savez_compressed(
        a.out,
        frame=np.array([f["frame"] for f in fr]),
        offsets=np.cumsum([0] + [len(f["xyz"]) for f in fr]),
        xyz=np.concatenate([f["xyz"] for f in fr]),
        intensity=inten, az=az, ch=np.concatenate([f["ch"] for f in fr]),
        rng=rng)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
