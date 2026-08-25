#!/usr/bin/env python3
"""Export a subset of the ATLAS phantom trace libraries as web assets.

ATLAS does not inject a synthetic shape. It samples a REAL recorded object point
cloud from trace_{easy,medium,hard}.pkl, centres it, yaws it slightly and offsets
it in front of the ego. The difficulty names are point density, measured on the
nuScenes libraries:

    easy    1282 traces, median 186 points   (dense phantom, med range 12 m)
    medium  1303 traces, median  26 points   (           med range 23 m)
    hard    1174 traces, median   7 points   (sparse phantom, med range 32 m)

which is what `inject_relative_easy` / `_medium` / `_hard` refer to.

Traces are exported RAW (absolute lidar-frame xyz, as stored in the pkl) so the
browser can run ATLAS's own `center_trace` faithfully rather than us pre-baking a
transform ATLAS applies differently.

    python3 tools/export_traces.py                    # nuScenes + KITTI, 200 each
    python3 tools/export_traces.py --per-library 400

Format (.atlastrace): 8-byte magic, uint32 LE header length (space-padded so the
payload is 8-byte aligned), UTF-8 JSON header, then int16 x/y/z at 5 mm and
uint8 intensity, concatenated across traces. Trace k occupies
[prefix[k], prefix[k] + counts[k]).
"""
import argparse
import json
import os
import pickle
import struct

import numpy as np

PHM = os.environ.get("PHM_ROOT", os.path.expanduser("~/rsch/Polar-Hierarchical-Mamba"))
MAGIC = b"ATLTRACE"
POS_QUANTUM = 0.005

SOURCES = {
    # dataset -> (directory holding trace_*.pkl, intensity scale max)
    "nuscenes": (os.path.join(PHM, "atlas_generation"), 255.0),
    "kitti": (os.path.join(PHM, "atlas_generation", "assets_kitti"), 1.0),
}
DIFFICULTIES = ("easy", "medium", "hard")


def pick_subset(n_total, k):
    """Evenly spaced indices, so the subset is deterministic and spans the library."""
    if k >= n_total:
        return list(range(n_total))
    return [int(round(i)) for i in np.linspace(0, n_total - 1, k)]


def export(dataset, difficulty, per_library, out_dir):
    src_dir, scale_max = SOURCES[dataset]
    path = os.path.join(src_dir, f"trace_{difficulty}.pkl")
    if not os.path.exists(path):
        print(f"  SKIP {dataset}/{difficulty}: {path} not found")
        return None

    with open(path, "rb") as f:
        lib = pickle.load(f)
    idxs = pick_subset(len(lib), per_library)

    xs, ys, zs, its, counts, boxes, ranges = [], [], [], [], [], [], []
    for i in idxs:
        t = lib[i]
        p = np.asarray(t["points"], dtype=np.float64)
        if p.shape[0] == 0:
            continue
        keep = np.isfinite(p[:, :3]).all(axis=1) & (np.abs(p[:, :3]) < 163.0).all(axis=1)
        p = p[keep]
        if p.shape[0] == 0:
            continue
        xs.append(p[:, 0])
        ys.append(p[:, 1])
        zs.append(p[:, 2])
        its.append(p[:, 3])
        counts.append(int(p.shape[0]))
        boxes.append([float(v) for v in np.asarray(t["box"], dtype=np.float64)[:7]])
        ranges.append(float(t.get("range", np.linalg.norm(np.asarray(t["box"])[:3]))))

    if not counts:
        print(f"  SKIP {dataset}/{difficulty}: no usable traces")
        return None

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    inten = np.concatenate(its)

    qx = np.round(x / POS_QUANTUM).astype(np.int16)
    qy = np.round(y / POS_QUANTUM).astype(np.int16)
    qz = np.round(z / POS_QUANTUM).astype(np.int16)
    if scale_max <= 1.0 + 1e-6:
        q_i = np.round(np.clip(inten, 0, 1) * 255).astype(np.uint8)
        intensity_scale = 1.0 / 255.0
    else:
        q_i = np.round(np.clip(inten, 0, 255)).astype(np.uint8)
        intensity_scale = 1.0

    blocks, offset, arrays = [], 0, {}
    for key, arr in (("x", qx), ("y", qy), ("z", qz), ("intensity", q_i)):
        if arr.dtype == np.int16 and offset % 2:
            blocks.append(b"\x00")
            offset += 1
        arrays[key] = {
            "offset": offset,
            "count": int(arr.size),
            "dtype": "int16" if arr.dtype == np.int16 else "uint8",
        }
        buf = np.ascontiguousarray(arr).tobytes()
        blocks.append(buf)
        offset += len(buf)

    header = {
        "format": "ATLTRACE",
        "version": 1,
        "dataset": dataset,
        "difficulty": difficulty,
        "nTraces": len(counts),
        "nTracesInLibrary": len(lib),
        "totalPoints": int(x.shape[0]),
        "posQuantum": POS_QUANTUM,
        "intensityScale": intensity_scale,
        "intensityScaleMax": scale_max,
        "counts": counts,
        "boxes": boxes,
        "ranges": ranges,
        "arrays": arrays,
        "source": os.path.relpath(path, PHM),
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
        hjson += b" " * ((-(12 + len(hjson))) % 8)
        header["payloadOffset"] = 12 + len(hjson)
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8").ljust(len(hjson), b" ")
    pad = header["payloadOffset"] - 12 - len(hjson)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{dataset}-{difficulty}.atlastrace")
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        for b in blocks:
            f.write(b)

    med = int(np.median(counts))
    print(
        f"  {os.path.basename(out):34s} {len(counts):4d}/{len(lib)} traces  "
        f"median {med:4d} pts  {os.path.getsize(out)/1024:8.1f} kB"
    )
    return {
        "file": os.path.basename(out),
        "dataset": dataset,
        "difficulty": difficulty,
        "nTraces": len(counts),
        "medianPoints": med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-library", type=int, default=200)
    ap.add_argument("--out", default="assets/traces")
    ap.add_argument("--datasets", default="nuscenes,kitti")
    a = ap.parse_args()

    manifest = []
    print(f"exporting up to {a.per_library} traces per library -> {a.out}/")
    for dataset in a.datasets.split(","):
        if dataset not in SOURCES:
            raise SystemExit(f"unknown dataset {dataset!r}")
        for diff in DIFFICULTIES:
            entry = export(dataset, diff, a.per_library, a.out)
            if entry:
                manifest.append(entry)

    if manifest:
        mpath = os.path.join(a.out, "manifest.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {mpath} ({len(manifest)} libraries)")


if __name__ == "__main__":
    main()
