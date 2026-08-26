#!/usr/bin/env python3
"""Locate injected (spoofed) returns in Velodyne VLP-16 pcaps.

Approach: build a per-frame (azimuth-bin x laser-channel) range image, then look
for concentrated clusters of returns that are (a) far from the local
background, and (b) confined to a narrow azimuth sector -- the signature of a
laser-injection attack, where the attacker's pulses only enter the receiver
while the mirror sweeps past their position.
"""
import sys
from collections import defaultdict

import numpy as np

from parse_velo import iter_packets, decode_blocks

VLP16_INCL = np.array([-15, 1, -13, 3, -11, 5, -9, 7,
                       -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float64)


def frames(path, max_frames=None):
    """Yield per-frame arrays of (azimuth_deg, channel, dist_m, refl, ret_idx)."""
    cur = []
    prev_az = None
    nf = 0
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, tstamp, rmode, pid = decode_blocks(pl)
        dual = (rmode == 0x39)
        for bi, (az, chans) in enumerate(blocks):
            if prev_az is not None and az - prev_az < -180:
                if cur:
                    yield np.array(cur), rmode
                    nf += 1
                    if max_frames and nf >= max_frames:
                        return
                cur = []
            prev_az = az
            # in dual mode blocks pair up: even=last, odd=strongest
            ridx = (bi % 2) if dual else 0
            for ci, (dist, refl) in enumerate(chans):
                if dist > 0:
                    # two firing groups per block, 16 channels each
                    cur.append((az, ci % 16, dist, refl, ridx))
    if cur:
        yield np.array(cur), rmode


def azimuth_profile(fr, nbins=72):
    """Mean/max range and count per 5-degree azimuth bin."""
    az = fr[:, 0] % 360
    b = np.minimum((az / (360 / nbins)).astype(int), nbins - 1)
    prof = np.zeros((nbins, 4))
    for i in range(nbins):
        m = b == i
        if m.any():
            d = fr[m, 2]
            prof[i] = [m.sum(), d.mean(), d.max(), np.percentile(d, 90)]
    return prof


def analyse(path, max_frames=60, far_thresh=None):
    print(f"\n=== {path} ===")
    prof_acc = None
    nf = 0
    far_by_bin = defaultdict(int)
    all_far = []
    rmode = None
    for fr, rmode in frames(path, max_frames):
        nf += 1
        p = azimuth_profile(fr)
        prof_acc = p if prof_acc is None else prof_acc + p
        # "far" returns = beyond the bulk of the scene
        thr = far_thresh if far_thresh else np.percentile(fr[:, 2], 99.0)
        far = fr[fr[:, 2] > max(thr, 15.0)]
        for az in far[:, 0]:
            far_by_bin[int(az // 5) * 5] += 1
        if len(far):
            all_far.append(far)
    prof_acc /= max(nf, 1)
    print(f"  frames analysed: {nf}  return_mode={hex(rmode) if rmode else '?'}")

    # azimuth sectors with the most far returns
    top = sorted(far_by_bin.items(), key=lambda kv: -kv[1])[:8]
    print("  top azimuth sectors by far-return count (deg -> count):")
    for az, c in top:
        print(f"      {az:3d}-{az+5:3d}: {c}")

    if all_far:
        F = np.concatenate(all_far)
        print(f"  far returns total {len(F)}  "
              f"range {F[:,2].min():.1f}-{F[:,2].max():.1f} m  "
              f"mean {F[:,2].mean():.1f}  refl mean {F[:,3].mean():.1f}")
        # channel distribution -> a real object spans several channels,
        # an injection often hits a specific elevation band
        ch, cnt = np.unique(F[:, 1].astype(int), return_counts=True)
        print("  far-return channel histogram (elev deg: count):")
        order = np.argsort(-cnt)[:8]
        for i in order:
            print(f"      ch{ch[i]:2d} ({VLP16_INCL[ch[i]]:+5.1f}): {cnt[i]}")
    return prof_acc, far_by_bin


def dual_compare(path, max_frames=40):
    """In dual-return mode, compare the strongest vs last return sets."""
    print(f"\n--- dual-return comparison: {path} ---")
    n = 0
    s_only = l_only = both = 0
    s_r, l_r = [], []
    for fr, rmode in frames(path, max_frames):
        if rmode != 0x39:
            print("  not dual mode, skipping")
            return
        n += 1
        key = defaultdict(dict)
        for az, ch, d, refl, ri in fr:
            key[(round(az, 1), int(ch))][int(ri)] = d
        for k, v in key.items():
            if 0 in v and 1 in v:
                both += 1
                l_r.append(v[0]); s_r.append(v[1])
            elif 1 in v:
                s_only += 1
            else:
                l_only += 1
    print(f"  frames {n}: both-returns {both}, strongest-only {s_only}, last-only {l_only}")
    if both:
        s = np.array(s_r); l = np.array(l_r)
        gap = np.abs(l - s)
        print(f"  |last - strongest| : mean {gap.mean():.2f} m, "
              f"p95 {np.percentile(gap,95):.2f}, max {gap.max():.2f}")
        big = gap > 2.0
        print(f"  pairs separated >2 m: {big.sum()} ({big.mean()*100:.2f}%)"
              f"  -> strongest {s[big].mean() if big.any() else 0:.1f} m vs "
              f"last {l[big].mean() if big.any() else 0:.1f} m")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(p)
