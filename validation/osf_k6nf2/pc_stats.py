#!/usr/bin/env python3
"""Point-cloud statistics comparing ATLAS-generated clouds to real LiDAR hardware.

IMPORTANT framing
-----------------
The real capture is a VLP-16 (16 beams, 0.4 deg az, indoor bench, ~2 m target);
the ATLAS splits are 32-beam nuScenes street scenes. Raw per-ring/per-azimuth
densities CANNOT agree between them and it would be dishonest to present them as
if they did -- the sensors and scenes differ.

What CAN be compared, and is the actual claim, is whether ATLAS clouds obey the
same *sensor-physics invariants* that real spinning-LiDAR data obeys. These are
sensor-independent and are exactly what a naive synthetic generator gets wrong:

  I1  Range quantization      -- real returns land on a fixed 2 mm grid.
  I2  Beam-inclination lattice-- elevations collapse onto N discrete rings, i.e.
                                 the elevation histogram is a comb, not smooth.
  I3  Azimuth lattice         -- azimuths land on a fixed angular grid.
  I4  One-return-per-ray      -- a (ring, az_cell) pair holds <=1 return in a
                                 single sweep (>1 only via sweep accumulation).
  I5  Inverse-square density  -- points per object fall as ~1/r^2.
  I6  Ring-arc density law    -- points along a ring scale with r*dtheta.

For each we report the real hardware value, the ATLAS value, and (where the
sensors differ) the *predicted* ATLAS value from that sensor's own spec, so the
comparison is against the right target rather than against the VLP-16 number.

Per-ring and per-azimuth densities are still reported, both raw and normalized,
because they are the conventional descriptors -- but they are labelled as
sensor-dependent, not as evidence of agreement.
"""
import argparse
import glob
import os
import pickle as pkl
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_velo import iter_packets, decode_blocks, VLP16_INCL  # noqa: E402

VLP16_AZ_RES = 0.2      # deg, per firing (spec)
VLP16_DIST_UNIT = 0.002  # m, raw distance LSB


# ----------------------------------------------------------------- real side
def real_frames(path, max_frames=60):
    """Yield per-revolution structured arrays of genuine returns.

    Returns arrays of (range, azimuth_deg, ring, refl). Uses the VLP-16's two
    firings per 32-channel block, keeping each firing's own azimuth.
    """
    out = []
    cur = []
    prev_az = None
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        blocks, _, rmode, _ = decode_blocks(pl)
        for az, chans in blocks:
            if prev_az is not None and az - prev_az < -180:
                if cur:
                    out.append(np.array(cur, dtype=[
                        ("r", "f8"), ("az", "f8"), ("ring", "i4"), ("refl", "i4")]))
                    if len(out) >= max_frames:
                        return out
                cur = []
            prev_az = az
            # The VLP-16 packs TWO firings per 32-channel block. The second
            # fires ~55.3 us later, i.e. half a block's azimuth advance later
            # (0.2 deg at 600 rpm). Assigning both firings the block azimuth
            # collapses the azimuth lattice onto a 0.4 deg grid and forces the
            # I6 neighbour-gap ratio to exactly 2.0 -- an artifact of the
            # decoder, not of the sensor.
            for half in range(2):
                az_f = (az + half * VLP16_AZ_RES) % 360.0
                for c in range(16):
                    d, rf = chans[half * 16 + c]
                    if d > 0.0:
                        cur.append((d, az_f, c, rf))
    if cur:
        out.append(np.array(cur, dtype=[
            ("r", "f8"), ("az", "f8"), ("ring", "i4"), ("refl", "i4")]))
    return out


def raw_distance_units(path, max_packets=4000):
    """Collect RAW u16 distance codes, to test the 2 mm quantization directly."""
    import struct
    codes = []
    n = 0
    for ts, dport, pl in iter_packets(path):
        if len(pl) != 1206:
            continue
        n += 1
        for b in range(12):
            off = b * 100 + 4
            for c in range(32):
                d = struct.unpack_from("<H", pl, off + c * 3)[0]
                if d:
                    codes.append(d)
        if n >= max_packets:
            break
    return np.array(codes, dtype=np.int64)


# ---------------------------------------------------------------- ATLAS side
def load_assets(asset_dir, sensor="nuscenes"):
    inc = np.sort(np.load(os.path.join(
        asset_dir, f"{sensor}_beam_inclinations.npy")).ravel().astype(np.float64))
    dth = float(np.load(os.path.join(
        asset_dir, f"{sensor}_azimuth_res.npy")).ravel()[0])
    return inc, dth


def atlas_frames(split_dir, n_scenes=6, n_per=6):
    """Load point clouds from an ATLAS split.

    Two on-disk layouts exist: nuScenes writes per-scene SEGMENT files
    (`*_d.pkl`, a list of frame dicts); KITTI writes one dict per FRAME
    (`000001.pkl`).
    """
    out = []
    files = sorted(glob.glob(os.path.join(split_dir, "*_d.pkl")))[:n_scenes]
    if files:                                   # nuScenes: segment lists
        for f in files:
            try:
                seg = pkl.load(open(f, "rb"))
            except Exception:
                continue
            step = max(1, len(seg) // n_per)
            for fr in seg[::step][:n_per]:
                out.append(np.asarray(fr["points"], dtype=np.float64))
        return out
    files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))[:n_scenes * n_per]
    for f in files:                             # KITTI: one dict per frame
        try:
            fr = pkl.load(open(f, "rb"))
        except Exception:
            continue
        if isinstance(fr, dict) and "points" in fr:
            out.append(np.asarray(fr["points"], dtype=np.float64))
    return out


def fov_crop(P, az_abs_max=40.0, x_min=5.0):
    """Match the ATLAS KITTI splits' camera-FOV crop.

    The generated KITTI splits are cropped to roughly |azimuth| <= 40 deg and
    x > 5 m, while raw KITTI bins are full 360 deg. Comparing per-ring density
    or azimuth uniformity between the two without this crop measures the crop,
    not the generator.
    """
    az = np.rad2deg(np.arctan2(P[:, 1], P[:, 0]))
    return P[(np.abs(az) <= az_abs_max) & (P[:, 0] > x_min)]


def kitti_clean_frames(kitti_root, n=36, crop=True):
    """Raw single-sweep KITTI velodyne bins -- the CONTROL.

    Any invariant ATLAS fails here too is inherited from the dataset, not
    introduced by the generator. This is what makes the ATLAS numbers
    interpretable.
    """
    infos = pkl.load(open(os.path.join(kitti_root, "kitti_infos_val.pkl"), "rb"))
    out = []
    for info in infos:
        fid = info["point_cloud"]["lidar_idx"]
        p = os.path.join(kitti_root, "training", "velodyne", f"{fid}.bin")
        if os.path.exists(p):
            P = np.fromfile(p, dtype=np.float32).reshape(-1, 4).astype(np.float64)
            out.append(fov_crop(P) if crop else P)
        if len(out) >= n:
            break
    return out


def spherical(P):
    r = np.linalg.norm(P[:, :3], axis=1)
    m = r > 1e-6
    r = r[m]
    P = P[m]
    az = np.rad2deg(np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi))
    el = np.rad2deg(np.arcsin(np.clip(P[:, 2] / r, -1, 1)))
    return r, az, el, P


# ------------------------------------------------------------------ metrics
def quant_step(vals, lo, hi, n=4000):
    """Estimate the quantization step of `vals` by maximizing grid alignment.

    For a candidate step q, residuals mod q concentrate near 0 for the true q.
    Score = mean cos(2*pi*v/q) (a Fourier/Rayleigh statistic); 1.0 = perfect grid.
    """
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) > 200000:
        v = np.random.default_rng(0).choice(v, 200000, replace=False)
    qs = np.linspace(lo, hi, n)
    best_q, best_s = None, -2.0
    for q in qs:
        s = np.cos(2 * np.pi * v / q).mean()
        if s > best_s:
            best_s, best_q = s, q
    return best_q, best_s


def grid_alignment(vals, q):
    """Fraction of values within 5% of a step of the q-grid, and Rayleigh score."""
    v = np.asarray(vals, dtype=np.float64)
    res = np.mod(v, q) / q
    res = np.minimum(res, 1 - res)
    return float((res < 0.05).mean()), float(np.cos(2 * np.pi * v / q).mean())


def ring_comb_stats(el, inc_deg):
    """How tightly elevations snap to the beam lattice."""
    d = np.abs(el[:, None] - inc_deg[None, :])
    j = d.argmin(axis=1)
    resid = d[np.arange(len(el)), j]
    return dict(n_rings_used=len(np.unique(j)),
                resid_med=float(np.median(resid)),
                resid_p95=float(np.percentile(resid, 95)),
                frac_within_0p1=float((resid < 0.1).mean()))


def occupancy_stats(az, el, inc_deg, dth_deg):
    """One-return-per-ray test: returns per (ring, az_cell) cell."""
    ring = np.abs(el[:, None] - inc_deg[None, :]).argmin(axis=1)
    cell = np.floor(az / dth_deg).astype(np.int64)
    key = ring.astype(np.int64) * 100000 + cell
    _, cnt = np.unique(key, return_counts=True)
    return dict(mean_per_cell=float(cnt.mean()),
                frac_cells_single=float((cnt == 1).mean()),
                p95=float(np.percentile(cnt, 95)), max=int(cnt.max()))


def radial_density(r, nbins=8, rmin=3.0, rmax=None):
    """Points per unit solid angle vs range -> should fall as ~1/r^2."""
    rmax = rmax or np.percentile(r, 98)
    edges = np.linspace(rmin, rmax, nbins + 1)
    cs, ct = [], []
    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() < 20:
            continue
        mid = 0.5 * (edges[i] + edges[i + 1])
        shell = edges[i + 1] - edges[i]
        cs.append(mid)
        ct.append(m.sum() / shell)
    if len(cs) < 3:
        return None
    cs, ct = np.array(cs), np.array(ct)
    sl, ic = np.polyfit(np.log(cs), np.log(ct), 1)
    pred = np.polyval([sl, ic], np.log(cs))
    ss = 1 - ((np.log(ct) - pred) ** 2).sum() / (
        (np.log(ct) - np.log(ct).mean()) ** 2).sum()
    return dict(slope=float(sl), r2=float(ss), centers=cs, counts=ct)


def fmt(x, f="%.4f"):
    return "n/a" if x is None else (f % x)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nuscenes", "kitti"], default="kitti")
    ap.add_argument("--atlas_dir", required=True,
                    help="<root>/atlas_val containing the generated splits")
    ap.add_argument("--assets", default=None)
    ap.add_argument("--kitti_root", default=None,
                    help="for the clean single-sweep control")
    ap.add_argument("--splits", nargs="+",
                    default=["removal_az20", "inject_easy"])
    ap.add_argument("--real", nargs="+",
                    default=["Multimode Analysis/last_mode.pcap",
                             "Multimode Analysis/dual_mode.pcap"])
    ap.add_argument("--max_real_frames", type=int, default=40)
    ap.add_argument("--n_frames", type=int, default=36)
    a = ap.parse_args()

    assets = a.assets or f"../atlas_generation/assets_{'kitti' if a.dataset=='kitti' else 'trainval'}"
    inc, dth = load_assets(assets, a.dataset)
    inc_deg = np.rad2deg(inc)
    dth_deg = np.rad2deg(dth)
    vlp_inc = np.sort(np.array(VLP16_INCL, dtype=np.float64))

    print("=" * 82)
    print(f"ATLAS ({a.dataset}) vs REAL k6nf2 capture -- sensor-physics invariants")
    print("=" * 82)
    print(f"  REAL  VLP-16  : 16 beams {vlp_inc.min():+.1f}..{vlp_inc.max():+.1f} deg, "
          f"az {VLP16_AZ_RES} deg, range LSB {VLP16_DIST_UNIT*1000:.0f} mm, single sweep")
    print(f"  ATLAS {a.dataset:8s}: {len(inc_deg)} beams "
          f"{inc_deg.min():+.2f}..{inc_deg.max():+.2f} deg, az {dth_deg:.4f} deg"
          + (", single sweep" if a.dataset == "kitti" else ", 10-sweep accumulated"))
    print("\n  Absolute densities cannot agree (different sensors + scenes). The")
    print("  invariants below are sensor-INDEPENDENT: they test whether ATLAS output")
    print("  is structured like real spinning-LiDAR data at all.")

    # ---- assemble sources: (label, kind, frames, inclinations, az_res)
    sources = []
    for rp in a.real:
        fr = real_frames(rp, a.max_real_frames)
        if not fr:
            continue
        sources.append(dict(
            label=f"REAL {os.path.basename(rp)}", kind="real", path=rp,
            r=np.concatenate([f["r"] for f in fr]),
            az=np.concatenate([f["az"] for f in fr]),
            el=np.concatenate([np.array([VLP16_INCL[c] for c in f["ring"]], float)
                               for f in fr]),
            frames=[(f["r"], f["az"],
                     np.array([VLP16_INCL[c] for c in f["ring"]], float))
                    for f in fr],
            inc=vlp_inc, dth=VLP16_AZ_RES))
    for sp in a.splits:
        sd = os.path.join(a.atlas_dir, sp)
        if not os.path.isdir(sd):
            print(f"  [skip] {sd} not found")
            continue
        Fs = atlas_frames(sd, n_scenes=a.n_frames, n_per=6)
        if not Fs:
            print(f"  [skip] {sp}: no frames")
            continue
        sph = [spherical(P)[:3] for P in Fs]
        sources.append(dict(
            label=f"ATLAS {sp}", kind="atlas",
            r=np.concatenate([s[0] for s in sph]),
            az=np.concatenate([s[1] for s in sph]),
            el=np.concatenate([s[2] for s in sph]),
            frames=sph, inc=inc_deg, dth=dth_deg))
    if a.kitti_root:
        Fs = kitti_clean_frames(a.kitti_root, a.n_frames)
        if Fs:
            sph = [spherical(P)[:3] for P in Fs]
            sources.append(dict(
                label="CONTROL clean KITTI", kind="control",
                r=np.concatenate([s[0] for s in sph]),
                az=np.concatenate([s[1] for s in sph]),
                el=np.concatenate([s[2] for s in sph]),
                frames=sph, inc=inc_deg, dth=dth_deg))

    W = max(len(s["label"]) for s in sources) + 1

    print("\n" + "=" * 82)
    print("I1  RANGE QUANTIZATION -- real hardware reports range on a 2 mm integer grid")
    print("=" * 82)
    for s in sources:
        fa, ry = grid_alignment(s["r"], VLP16_DIST_UNIT)
        q, sc = quant_step(s["r"], 0.0015, 0.0035, 1500)
        print(f"  {s['label']:{W}s} on-2mm-grid {fa:.4f}  Rayleigh {ry:+.4f}  "
              f"best step {q*1000:.3f} mm")
    print("  (chance level for an unquantized cloud is 0.10)")

    print("\n" + "=" * 82)
    print("I2  BEAM LATTICE -- elevations must snap to N discrete rings")
    print("=" * 82)
    for s in sources:
        st = ring_comb_stats(s["el"], s["inc"])
        print(f"  {s['label']:{W}s} rings {st['n_rings_used']:3d}/{len(s['inc'])}  "
              f"resid med {st['resid_med']:.4f} deg  p95 {st['resid_p95']:.4f}  "
              f"frac<0.1deg {st['frac_within_0p1']:.4f}")

    print("\n" + "=" * 82)
    print("I3  AZIMUTH LATTICE -- azimuths must land on the sensor's angular grid")
    print("=" * 82)
    for s in sources:
        fa, ry = grid_alignment(s["az"], s["dth"])
        print(f"  {s['label']:{W}s} on-grid({s['dth']:.4f} deg) {fa:.4f}  "
              f"Rayleigh {ry:+.4f}")

    print("\n" + "=" * 82)
    print("I4  ONE RETURN PER RAY -- a (ring, az_cell) holds <=1 return per sweep")
    print("=" * 82)
    for s in sources:
        rr, az, el = s["frames"][0]
        st = occupancy_stats(az, el, s["inc"], s["dth"])
        print(f"  {s['label']:{W}s} mean/cell {st['mean_per_cell']:.3f}  "
              f"frac==1 {st['frac_cells_single']:.4f}  p95 {st['p95']:.1f}  "
              f"max {st['max']}")

    print("\n" + "=" * 82)
    print("I5  RADIAL DENSITY LAW -- log-log slope of returns vs range")
    print("=" * 82)
    for s in sources:
        rmin = 1.0 if s["kind"] == "real" else 3.0
        d = radial_density(s["r"], nbins=8, rmin=rmin)
        if d:
            print(f"  {s['label']:{W}s} slope {d['slope']:+.3f}  R2 {d['r2']:.4f}")

    print("\n" + "=" * 82)
    print("I6  RING-ARC SPACING -- neighbour gap along a ring must grow as r*dtheta")
    print("=" * 82)
    for s in sources:
        rr, az, el = s["frames"][0]
        ring = np.abs(el[:, None] - s["inc"][None, :]).argmin(axis=1)
        ratios = []
        for k in np.unique(ring):
            m = ring == k
            if m.sum() < 40:
                continue
            o = np.argsort(az[m])
            aa, rrk = az[m][o], rr[m][o]
            gap = np.diff(aa)
            g = gap[(gap > 0) & (gap < 5 * s["dth"])]
            if len(g) < 20:
                continue
            ratios.append(np.median(g) / s["dth"])
        if ratios:
            ratios = np.array(ratios)
            print(f"  {s['label']:{W}s} median neighbour gap / az_res = "
                  f"{np.median(ratios):.3f}  (ideal 1.0)  over {len(ratios)} rings")

    print("\n" + "=" * 82)
    print("PER-RING DENSITY -- sensor-dependent; shape/CV is the comparable part")
    print("=" * 82)
    for s in sources:
        per = []
        for rr, az, el in s["frames"]:
            ring = np.abs(el[:, None] - s["inc"][None, :]).argmin(axis=1)
            per.append(np.bincount(ring, minlength=len(s["inc"])))
        cnt = np.mean(per, axis=0)
        nz = cnt[cnt > 0]
        print(f"  {s['label']:{W}s} pts/ring/frame med {np.median(nz):7.1f}  "
              f"CV {nz.std()/nz.mean():.4f}  occupied {len(nz)}/{len(s['inc'])}")

    print("\n" + "=" * 82)
    print("PER-AZIMUTH UNIFORMITY -- CV over 5-deg bins WITHIN each source's own")
    print("occupied field of view (ATLAS KITTI splits are FOV-cropped)")
    print("=" * 82)
    for s in sources:
        h = np.histogram(s["az"], bins=72, range=(0, 360))[0].astype(float)
        occ = h > 0.02 * h.max()          # inside the FOV, not a stray bin
        hh = h[occ]
        print(f"  {s['label']:{W}s} CV {hh.std()/hh.mean():.4f}  "
              f"occupied {len(hh)}/72 bins")


if __name__ == "__main__":
    main()
