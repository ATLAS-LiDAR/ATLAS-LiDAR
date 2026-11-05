"""Scene-level (non-adversarial) perturbation operators for the ATLAS CVPR'27 paper (Table resub_perturbation_spec).

All operators act on single-frame points in the sensor/vehicle frame, columns
  x, y, z, intensity (tanh scale, [0,1)), elongation (tanh scale), time (0 for the current sweep).
Every random draw comes from a hash-seeded stream keyed by (seed, operator, segment, local frame[, block]),
so datasets are reproducible and frame-paired with every other dataset on the same 50 segments.

Per-frame operators  f(points, frame, rng, **params) -> (points, meta)
  spray      added evidence, natural   clusters behind vehicles faster than --min_speed (water film level)
  crosstalk  added evidence, natural   a fraction of points displaced (Robo3D-style cross-talk)
  fog        removed evidence, weather attenuation + back-scatter, extinction coefficient alpha
Sequence operators  f(points_list, attack_blocks, seg_rng, **params) -> (points_list, metas)
  blackout, stuck, delayed, dropped    temporal faults on the attack blocks (stream faults)
pose_noise  only perturbs the ego pose used to align history sweeps (points unchanged); see README.

These are simplified, documented models, not bit-exact reimplementations of the cited simulators:
  spray     after the qualitative model of road-spray clusters trailing vehicles (e.g. Linnhoff et al. 2022);
            point density grows with speed^2 and water-film level, falls with range^2; very low intensity.
  crosstalk Robo3D (Kong et al. ICCV'23) cross-talk: displaced points with large Gaussian offsets.
  fog       after Hahner et al. ICCV'21: hard-target power ~ exp(-2 alpha R); returns whose attenuated
            intensity drops below the detection floor are lost, and some of those beams return a soft
            back-scatter point closer to the sensor. MOR (visibility) = 3 / alpha.
"""
import hashlib

import numpy as np

DUMMY_POINT = np.array([[70.0, 0.0, -1.5, 0.0, 0.0, 0.0]], dtype=np.float32)   # blackout frame (see README)


def stream(seed, *parts):
    h = hashlib.sha256(":".join(str(p) for p in (seed,) + parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _ranges(xyz):
    return np.linalg.norm(xyz, axis=1)


# ----------------------------------------------------------------------------------------------- spray
SPRAY_BASE = {1: 60.0, 2: 250.0}   # expected points per vehicle at 20 m/s and 15 m range; level 1 ~1 mm, level 2 ~2-3 mm


def spray(points, frame, rng, level=1, min_speed=15.0):
    boxes = np.asarray(frame.get("gt_boxes", np.zeros((0, 10))), dtype=np.float64)
    added, clusters = [], []
    for b in boxes:
        if int(b[-1]) != 1 or b.shape[0] < 10:
            continue
        speed = float(np.hypot(b[7], b[8]))
        if speed < min_speed:
            continue
        r = max(float(np.hypot(b[0], b[1])), 5.0)
        lam = SPRAY_BASE[level] * (speed / 20.0) ** 2 * min((15.0 / r) ** 2, 4.0)
        n = int(rng.poisson(lam))
        if n == 0:
            continue
        L = 2.0 + 0.15 * speed                                   # cloud length behind the rear bumper
        u = rng.uniform(0, 1, n)
        lx = -b[3] / 2 - u * L                                   # local x (behind)
        ly = rng.uniform(-1, 1, n) * (b[4] / 2 + 0.3 + 0.2 * u)  # widens with distance
        lz = -b[5] / 2 + rng.uniform(0, 1, n) * (0.4 + 1.1 * u)  # rises with distance
        c, s = np.cos(b[6]), np.sin(b[6])
        x = b[0] + c * lx - s * ly
        y = b[1] + s * lx + c * ly
        z = b[2] + lz
        pts = np.stack([x, y, z, rng.uniform(0.0, 0.05, n), np.zeros(n), np.zeros(n)], axis=1)
        added.append(pts.astype(np.float32))
        clusters.append({"vehicle_box": b[:7].astype(np.float32), "speed": speed, "n": n,
                         "cloud_box": np.array([b[0] - c * (b[3] / 2 + L / 2), b[1] - s * (b[3] / 2 + L / 2),
                                                b[2] - b[5] / 2 + 0.75, L, b[4] + 0.8, 1.5, b[6]], np.float32)})
    out = np.concatenate([points] + added, axis=0) if added else points
    return out, {"n_added": int(sum(len(a) for a in added)), "clusters": clusters}


# ------------------------------------------------------------------------------------------- crosstalk
def crosstalk(points, frame, rng, frac=0.006, sigma=3.0):
    n = int(round(frac * len(points)))
    out = points.copy()
    if n:
        idx = rng.choice(len(points), size=n, replace=False)
        out[idx, :3] += rng.normal(0.0, sigma, (n, 3)).astype(points.dtype)
    return out, {"n_displaced": n}


# ------------------------------------------------------------------------------------------------- fog
def fog(points, frame, rng, alpha=0.02, floor=0.005, p_scatter=0.5):
    xyz = points[:, :3].astype(np.float64)
    R = _ranges(xyz)
    att = np.exp(-2.0 * alpha * R)
    inten = points[:, 3] * att
    lost = inten < floor
    out = points.copy()
    out[:, 3] = inten.astype(points.dtype)
    keep = ~lost
    sc = lost & (rng.uniform(0, 1, len(points)) < p_scatter)
    scat = points[sc].copy()
    if len(scat):
        Rs = np.clip(R[sc] * rng.beta(2.0, 5.0, len(scat)), 1.5, None)
        scat[:, :3] = (xyz[sc] * (Rs / np.maximum(R[sc], 1e-6))[:, None]).astype(points.dtype)
        scat[:, 3] = rng.uniform(0.0, 0.02, len(scat)).astype(points.dtype)
        scat[:, 4] = 0.0
    res = np.concatenate([out[keep], scat], axis=0)
    return res, {"n_lost": int(lost.sum()), "n_scatter": int(len(scat)), "mor_m": 3.0 / alpha}


# --------------------------------------------------------------------------------- temporal faults (sequence)
def blackout(pts, blocks, rng, length=3):
    metas = {}
    for b0, b1 in blocks:
        s = int(rng.integers(b0, max(b0, b1 - length) + 1))
        for t in range(s, min(s + length, b1)):
            pts[t] = DUMMY_POINT.copy(); metas[t] = {"fault": "blackout", "start": s}
    return pts, metas


def stuck(pts, blocks, rng, repeats=4):
    metas = {}
    for b0, b1 in blocks:
        s = int(rng.integers(b0, max(b0, b1 - repeats) + 1))
        for t in range(s + 1, min(s + repeats, b1)):
            pts[t] = pts[s].copy(); metas[t] = {"fault": "stuck", "source": s}
    return pts, metas


def delayed(pts, blocks, rng, delay=1):
    clean = list(pts)
    metas = {}
    for b0, b1 in blocks:
        for t in range(b0, b1):
            pts[t] = clean[t - delay].copy(); metas[t] = {"fault": "delayed", "source": t - delay}
    return pts, metas


def dropped(pts, blocks, rng, p=0.25):
    metas = {}
    for b0, b1 in blocks:
        last = b0 - 1
        for t in range(b0, b1):
            if rng.uniform() < p:
                pts[t] = pts[last].copy(); metas[t] = {"fault": "dropped", "source": last}
            else:
                last = t
    return pts, metas


def pose_noise(pose, rng, rot_sigma=0.04, trans_sigma=0.004):
    """Noisy ego pose (4x4) = pose @ delta, delta = small rotation (rad, about x/y/z) + translation (m)."""
    a = rng.normal(0.0, rot_sigma, 3)
    cx, sx, cy, sy, cz, sz = np.cos(a[0]), np.sin(a[0]), np.cos(a[1]), np.sin(a[1]), np.cos(a[2]), np.sin(a[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    d = np.eye(4); d[:3, :3] = Rz @ Ry @ Rx; d[:3, 3] = rng.normal(0.0, trans_sigma, 3)
    return np.asarray(pose, np.float64) @ d


PER_FRAME = {"spray": spray, "crosstalk": crosstalk, "fog": fog}
SEQUENCE = {"blackout": blackout, "stuck": stuck, "delayed": delayed, "dropped": dropped}
