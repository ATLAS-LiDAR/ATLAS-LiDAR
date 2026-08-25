#!/usr/bin/env python3
"""Export a nuScenes driving sequence (keyframes + ego poses) as a web asset.

    python3 tools/export_sequence.py --list
    python3 tools/export_sequence.py --scene 0 --max-frames 20

WHY nuSCENES ONLY
KITTI's object-detection split on disk is a set of independent frames: no
temporal ordering and no ego poses, so it cannot make a sequence. nuScenes
keyframes carry `ref_from_car` and `car_from_global`, which compose into the
ego->global pose this needs. The cost is a sparser 32-beam cloud (~34k points
vs KITTI's ~120k), which is fine for visualization.

TIMING
pcdet stores nuScenes timestamps already divided by 1e6, i.e. in SECONDS, and
keyframes are 2 Hz. `dtSec` is written into the header because
removal.match_with_velocity extrapolates a constant-velocity prediction with it -
its docstring warns that passing Waymo's 0.1 for nuScenes makes the prediction 5x
too short and silently drops the track.

Format (.atlasseq): 8-byte magic, uint32 LE header length (space-padded so the
payload is 8-byte aligned), UTF-8 JSON header, then int16 x/y/z at 5 mm and uint8
intensity, concatenated across frames. Frame k occupies
[prefix[k], prefix[k] + counts[k]).
"""
import argparse
import json
import os
import pickle
import struct

import numpy as np

PHM = os.environ.get("PHM_ROOT", os.path.expanduser("~/rsch/Polar-Hierarchical-Mamba"))
MAGIC = b"ATLASSEQ"
POS_QUANTUM = 0.005
SCENE_GAP_S = 2.0  # consecutive keyframes are 0.5 s apart; scenes jump by days


def load_infos(version):
    root = os.path.join(PHM, "data", "nuscenes", version)
    for split in ("val", "train"):
        p = os.path.join(root, f"nuscenes_infos_10sweeps_{split}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), root
    raise SystemExit(f"no nuScenes infos found under {root}")


def group_scenes(infos):
    """Split the flat keyframe list into scenes on large timestamp jumps."""
    scenes = []
    cur = [0]
    for i in range(1, len(infos)):
        if infos[i]["timestamp"] - infos[i - 1]["timestamp"] > SCENE_GAP_S:
            scenes.append(cur)
            cur = []
        cur.append(i)
    if cur:
        scenes.append(cur)
    return scenes


def ego_to_global(info):
    """ATLAS: global_to_ref = ref_from_car @ car_from_global; pose = inv(that)."""
    ref_from_car = np.asarray(info["ref_from_car"], dtype=np.float64)
    car_from_global = np.asarray(info["car_from_global"], dtype=np.float64)
    return np.linalg.inv(ref_from_car @ car_from_global)


def read_sweep(root, info):
    path = os.path.join(root, info["lidar_path"])
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(root), info["lidar_path"])
    # sample sweep only - never the accumulated cloud, which would put ~9 returns
    # in every lattice cell and destroy the per-cell model
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]


def boxes_of(info):
    out = []
    gt = np.asarray(info.get("gt_boxes", np.zeros((0, 9))), dtype=np.float64)
    names = info.get("gt_names", [])
    for b, nm in zip(gt, names):
        out.append({
            "center": [float(b[0]), float(b[1]), float(b[2])],
            "size": [float(b[3]), float(b[4]), float(b[5])],
            "yaw": float(b[6]),
            "velocity": [float(b[7]) if len(b) > 7 else 0.0,
                         float(b[8]) if len(b) > 8 else 0.0],
            "label": str(nm),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", type=int, default=0, help="scene index within the split")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="assets/sequences")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    infos, root = load_infos(a.version)
    scenes = group_scenes(infos)

    if a.list:
        print(f"{a.version}: {len(infos)} keyframes in {len(scenes)} scene(s)")
        for k, idxs in enumerate(scenes):
            n_boxes = np.median([len(infos[i].get("gt_names", [])) for i in idxs])
            span = infos[idxs[-1]]["timestamp"] - infos[idxs[0]]["timestamp"]
            print(f"  scene {k}: {len(idxs)} keyframes, {span:.1f} s, median {n_boxes:.0f} boxes")
        return

    if a.scene >= len(scenes):
        raise SystemExit(f"scene {a.scene} out of range (0..{len(scenes)-1})")
    idxs = scenes[a.scene][:: a.stride][: a.max_frames]
    if len(idxs) < 2:
        raise SystemExit("need at least 2 frames for a sequence")

    times = [infos[i]["timestamp"] for i in idxs]
    dts = np.diff(times)
    dt_sec = float(np.median(dts))

    xs, ys, zs, its, counts, frames = [], [], [], [], [], []
    for i in idxs:
        info = infos[i]
        pts = read_sweep(root, info)
        keep = np.isfinite(pts[:, :3]).all(axis=1) & (np.abs(pts[:, :3]) < 163.0).all(axis=1)
        pts = pts[keep]
        xs.append(pts[:, 0].astype(np.float64))
        ys.append(pts[:, 1].astype(np.float64))
        zs.append(pts[:, 2].astype(np.float64))
        its.append(pts[:, 3].astype(np.float64))
        counts.append(int(pts.shape[0]))
        frames.append({
            "frameId": info["token"],
            "timestamp": float(info["timestamp"]),
            "nPoints": int(pts.shape[0]),
            "pose": [float(v) for v in ego_to_global(info).reshape(-1)],
            "boxes": boxes_of(info),
        })

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    inten = np.concatenate(its)

    qx = np.round(x / POS_QUANTUM).astype(np.int16)
    qy = np.round(y / POS_QUANTUM).astype(np.int16)
    qz = np.round(z / POS_QUANTUM).astype(np.int16)
    q_i = np.round(np.clip(inten, 0, 255)).astype(np.uint8)

    blocks, offset, arrays = [], 0, {}
    for key, arr in (("x", qx), ("y", qy), ("z", qz), ("intensity", q_i)):
        if arr.dtype == np.int16 and offset % 2:
            blocks.append(b"\x00")
            offset += 1
        arrays[key] = {"offset": offset, "count": int(arr.size),
                       "dtype": "int16" if arr.dtype == np.int16 else "uint8"}
        buf = np.ascontiguousarray(arr).tobytes()
        blocks.append(buf)
        offset += len(buf)

    slug = a.slug or f"nuscenes-scene{a.scene}"
    header = {
        "format": "ATLASSEQ",
        "version": 1,
        "dataset": "nuscenes",
        "sensor": "hdl32e",
        "slug": slug,
        "label": f"nuScenes driving sequence {a.scene} ({len(idxs)} keyframes)",
        "nFrames": len(idxs),
        "dtSec": dt_sec,
        "hzKeyframe": round(1.0 / dt_sec, 3) if dt_sec else None,
        "stride": a.stride,
        "posQuantum": POS_QUANTUM,
        "intensityScale": 1.0,
        "counts": counts,
        "frames": frames,
        "arrays": arrays,
        "source": f"{a.version} scene index {a.scene}",
        "payloadOffset": 0,
    }

    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(12 + len(hjson))) % 8
    header["payloadOffset"] = 12 + len(hjson) + pad
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(12 + len(hjson))) % 8
    if header["payloadOffset"] != 12 + len(hjson) + pad:
        header["payloadOffset"] = 12 + len(hjson) + pad
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        extra = header["payloadOffset"] - 12 - len(hjson)
        if extra < 0:
            header["payloadOffset"] = 12 + len(hjson) + ((-(12 + len(hjson))) % 8)
            hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = header["payloadOffset"] - 12 - len(hjson)
    pad = max(0, header["payloadOffset"] - 12 - len(hjson))

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{slug}.atlasseq")
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        for b in blocks:
            f.write(b)

    print(f"  {os.path.basename(out):32s} {len(idxs):3d} frames  "
          f"{sum(counts):8d} pts  dt {dt_sec:.2f}s  {os.path.getsize(out)/1024/1024:.2f} MB")

    mpath = os.path.join(a.out, "manifest.json")
    entry = {"file": os.path.basename(out), "slug": slug, "label": header["label"],
             "dataset": "nuscenes", "sensor": "hdl32e", "nFrames": len(idxs),
             "dtSec": dt_sec, "nPoints": int(sum(counts))}
    existing = []
    if os.path.exists(mpath):
        existing = [e for e in json.load(open(mpath)) if e["slug"] != slug]
    json.dump(existing + [entry], open(mpath, "w"), indent=2)
    print(f"wrote {mpath} ({len(existing)+1} sequence(s))")


if __name__ == "__main__":
    main()
