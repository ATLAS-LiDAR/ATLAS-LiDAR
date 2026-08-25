#!/usr/bin/env python3
"""Export real driving frames from KITTI / nuScenes into a compact web asset.

    python3 tools/export_frames.py --list kitti
    python3 tools/export_frames.py --dataset kitti --ids 000001,000010 --name urban
    python3 tools/export_frames.py --library          # export the built-in set

SINGLE SWEEPS ONLY. nuScenes infos accumulate 10 sweeps, which puts ~9 returns
in every (azimuth x beam) cell and destroys the one-return-per-ray property the
whole cell-occupancy model rests on (ATLAS_REAL_WORLD_VALIDATION.md measures
exactly this: KITTI single-sweep scores 0.83 on the I4 invariant, accumulated
nuScenes fails it by construction). We therefore read the sample sweep directly.

Format (.atlasframe): 8-byte magic, uint32 little-endian header length, UTF-8
JSON header, then the payload blocks the header points at. Positions are int16
quantized at 5 mm (+-163 m); intensity is uint8 with a scale factor. That is
7 bytes/point, so a 120k-point HDL-64E frame is ~840 kB.
"""
import argparse
import json
import os
import pickle
import struct
import sys

import numpy as np

PHM = os.environ.get("PHM_ROOT", os.path.expanduser("~/rsch/Polar-Hierarchical-Mamba"))
MAGIC = b"ATLFRAME"
POS_QUANTUM = 0.005  # metres per int16 step

# Built-in library: (dataset, frame id, slug, human label)
LIBRARY = [
    ("kitti", "000001", "kitti-open-road", "Open road - truck, car, cyclist (HDL-64E)"),
    ("kitti", "003219", "kitti-dense-traffic", "Dense traffic - 19 cars (HDL-64E)"),
    ("kitti", "005663", "kitti-crossing-vru", "Crossing - 13 pedestrians, 4 cyclists (HDL-64E)"),
    ("kitti", "001464", "kitti-parked-row", "Parked row - 17 cars, 2 vans (HDL-64E)"),
    ("nuscenes", "0", "nuscenes-boston", "Boston street, single sweep (HDL-32E)"),
]


def kitti_root():
    return os.path.join(PHM, "data", "kitti")


def nuscenes_root(version="v1.0-mini"):
    return os.path.join(PHM, "data", "nuscenes", version)


def load_kitti_infos():
    out = {}
    for split in ("train", "val"):
        p = os.path.join(kitti_root(), f"kitti_infos_{split}.pkl")
        if not os.path.exists(p):
            continue
        for info in pickle.load(open(p, "rb")):
            out[info["point_cloud"]["lidar_idx"]] = info
    return out


def read_kitti_frame(idx, infos):
    info = infos.get(idx)
    if info is None:
        raise SystemExit(f"KITTI frame {idx} not present in kitti_infos_*.pkl")
    for sub in ("training", "testing"):
        bin_path = os.path.join(kitti_root(), sub, "velodyne", f"{idx}.bin")
        if os.path.exists(bin_path):
            break
    else:
        raise SystemExit(f"no velodyne bin for KITTI {idx}")

    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    annos = info.get("annos", {})
    boxes, labels = [], []
    if "gt_boxes_lidar" in annos:
        names = annos["name"]
        for b, nm in zip(annos["gt_boxes_lidar"], names):
            if nm in ("DontCare",):
                continue
            boxes.append([float(v) for v in b[:7]])
            labels.append(str(nm))
    return pts, boxes, labels, {"sensor": "hdl64e", "dataset": "kitti", "frameId": idx}


def load_nuscenes_infos(version):
    for split in ("val", "train"):
        p = os.path.join(nuscenes_root(version), f"nuscenes_infos_10sweeps_{split}.pkl")
        if os.path.exists(p):
            yield from pickle.load(open(p, "rb"))


def read_nuscenes_frame(token_or_index, version="v1.0-mini"):
    infos = list(load_nuscenes_infos(version))
    if not infos:
        raise SystemExit("no nuScenes infos pkl found")
    info = None
    if token_or_index.isdigit():
        info = infos[int(token_or_index)]
    else:
        for i in infos:
            if i["token"].startswith(token_or_index):
                info = i
                break
    if info is None:
        raise SystemExit(f"nuScenes frame {token_or_index} not found")

    # The sample sweep only - never the accumulated cloud (see module docstring).
    path = os.path.join(nuscenes_root(version), info["lidar_path"])
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(nuscenes_root(version)), info["lidar_path"])
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]

    boxes, labels = [], []
    for b, nm in zip(info.get("gt_boxes", []), info.get("gt_names", [])):
        boxes.append([float(v) for v in b[:7]])
        labels.append(str(nm))
    return pts, boxes, labels, {
        "sensor": "hdl32e",
        "dataset": "nuscenes",
        "frameId": info["token"],
    }


def write_atlasframe(out_path, pts, boxes, labels, meta, name):
    xyz = pts[:, :3].astype(np.float64)
    inten = pts[:, 3].astype(np.float64)

    keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 163.0).all(axis=1)
    xyz, inten = xyz[keep], inten[keep]
    n = xyz.shape[0]

    q = np.round(xyz / POS_QUANTUM).astype(np.int16)

    # KITTI intensity is 0-1, nuScenes 0-255; store uint8 plus a scale factor.
    imax = float(inten.max()) if n else 1.0
    if imax <= 1.0 + 1e-6:
        i8 = np.round(np.clip(inten, 0, 1) * 255).astype(np.uint8)
        intensity_scale = 1.0 / 255.0
    else:
        i8 = np.round(np.clip(inten, 0, 255)).astype(np.uint8)
        intensity_scale = 1.0

    blocks, offset, arrays = [], 0, {}
    for key, arr in (("x", q[:, 0]), ("y", q[:, 1]), ("z", q[:, 2]), ("intensity", i8)):
        buf = np.ascontiguousarray(arr).tobytes()
        if offset % 2 and arr.dtype != np.uint8:
            blocks.append(b"\x00")
            offset += 1
        arrays[key] = {
            "offset": offset,
            "count": int(arr.size),
            "dtype": "int16" if arr.dtype == np.int16 else "uint8",
        }
        blocks.append(buf)
        offset += len(buf)

    header = {
        "version": 1,
        "name": name,
        "nPoints": n,
        "posQuantum": POS_QUANTUM,
        "intensityScale": intensity_scale,
        "arrays": arrays,
        "boxes": [
            {"center": b[:3], "size": b[3:6], "yaw": b[6], "label": l}
            for b, l in zip(boxes, labels)
        ],
        **meta,
    }
    hjson = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    # The payload starts at 12 + len(header), and int16 views need an even byte
    # offset (JS throws otherwise). Pad the header with spaces - which JSON
    # ignores - so the payload always begins 8-byte aligned regardless of how
    # long the header happens to be for this frame.
    pad = (-(12 + len(hjson))) % 8
    hjson += b" " * pad

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hjson)))
        f.write(hjson)
        for b in blocks:
            f.write(b)

    size_kb = os.path.getsize(out_path) / 1024
    print(
        f"  {os.path.basename(out_path):32s} {n:7d} pts  {len(header['boxes']):3d} boxes  {size_kb:7.1f} kB"
    )
    return header


def list_candidates(dataset, limit=25):
    if dataset == "kitti":
        infos = load_kitti_infos()
        rows = []
        for idx, info in sorted(infos.items()):
            a = info.get("annos", {})
            names = [n for n in a.get("name", []) if n != "DontCare"]
            if not names:
                continue
            from collections import Counter

            rows.append((idx, len(names), dict(Counter(names))))
        rows.sort(key=lambda r: -r[1])
        for idx, n, c in rows[:limit]:
            print(f"  {idx}  {n:3d} objects  {c}")
    else:
        for i, info in enumerate(load_nuscenes_infos("v1.0-mini")):
            if i >= limit:
                break
            from collections import Counter

            print(f"  {i:3d} {info['token'][:12]}  {dict(Counter(info.get('gt_names', [])))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kitti", "nuscenes"])
    ap.add_argument("--ids", help="comma-separated frame ids / tokens / indices")
    ap.add_argument("--name", default=None)
    ap.add_argument("--list", dest="list_ds", choices=["kitti", "nuscenes"])
    ap.add_argument("--library", action="store_true", help="export the built-in scene set")
    ap.add_argument("--out", default="assets/frames")
    ap.add_argument("--version", default="v1.0-mini")
    a = ap.parse_args()

    if a.list_ds:
        list_candidates(a.list_ds)
        return

    jobs = []
    if a.library:
        jobs = LIBRARY
    elif a.dataset and a.ids:
        for i, fid in enumerate(a.ids.split(",")):
            slug = f"{a.dataset}-{fid}" if not a.name else (a.name if i == 0 else f"{a.name}-{i}")
            jobs.append((a.dataset, fid, slug, slug))
    else:
        ap.error("use --library, --list DATASET, or --dataset X --ids A,B")

    kitti_infos = load_kitti_infos() if any(j[0] == "kitti" for j in jobs) else {}

    print(f"exporting {len(jobs)} frame(s) to {a.out}/")
    manifest = []
    for dataset, fid, slug, label in jobs:
        try:
            if dataset == "kitti":
                pts, boxes, labels, meta = read_kitti_frame(fid, kitti_infos)
            else:
                pts, boxes, labels, meta = read_nuscenes_frame(fid, a.version)
        except SystemExit as e:
            print(f"  SKIP {slug}: {e}")
            continue
        meta["label"] = label
        h = write_atlasframe(os.path.join(a.out, f"{slug}.atlasframe"), pts, boxes, labels, meta, slug)
        manifest.append(
            {
                "file": f"{slug}.atlasframe",
                "slug": slug,
                "label": label,
                "dataset": h["dataset"],
                "sensor": h["sensor"],
                "nPoints": h["nPoints"],
                "nBoxes": len(h["boxes"]),
            }
        )

    if manifest:
        mpath = os.path.join(a.out, "manifest.json")
        existing = []
        if os.path.exists(mpath):
            existing = [m for m in json.load(open(mpath)) if m["slug"] not in {x["slug"] for x in manifest}]
        json.dump(existing + manifest, open(mpath, "w"), indent=2)
        print(f"wrote {mpath} ({len(existing) + len(manifest)} scenes)")


if __name__ == "__main__":
    main()
